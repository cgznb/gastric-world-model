"""Patient-normalized losses and latent-state diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .errors import DataContractError
from .model.types import BeliefState, PredictionDistribution
from .survival import masked_patient_mean

Reduction = Literal["none", "mean", "sum"]


def diagonal_gaussian_kl_per_dimension(
    posterior_mean: Tensor,
    posterior_log_std: Tensor,
    prior_mean: Tensor,
    prior_log_std: Tensor,
) -> Tensor:
    """Return diagonal-Gaussian ``KL(q||p)`` without reducing latent dimensions."""

    if not (
        posterior_mean.shape == posterior_log_std.shape == prior_mean.shape == prior_log_std.shape
    ):
        raise ValueError("All Gaussian tensors must have the same shape")
    if posterior_mean.ndim != 3:
        raise ValueError("Gaussian state tensors must have shape [B,M,D]")
    q_log = posterior_log_std.clamp(-20.0, 10.0)
    p_log = prior_log_std.clamp(-20.0, 10.0)
    variance_ratio = torch.exp(2.0 * (q_log - p_log))
    mean_term = (posterior_mean - prior_mean).square() * torch.exp(-2.0 * p_log)
    return p_log - q_log + 0.5 * (variance_ratio + mean_term - 1.0)


def diagonal_gaussian_kl(
    posterior_mean: Tensor,
    posterior_log_std: Tensor,
    prior_mean: Tensor,
    prior_log_std: Tensor,
    *,
    free_nats_per_token: float = 0.0,
    reduction: Reduction = "mean",
) -> Tensor:
    """KL(q||p), summed over stochastic dimensions and normalized over tokens."""

    if free_nats_per_token < 0:
        raise ValueError("free_nats_per_token must be nonnegative")
    per_dimension = diagonal_gaussian_kl_per_dimension(
        posterior_mean,
        posterior_log_std,
        prior_mean,
        prior_log_std,
    )
    per_token = per_dimension.sum(dim=-1)
    if free_nats_per_token:
        per_token = per_token.clamp_min(free_nats_per_token)
    per_patient = per_token.mean(dim=1)
    if reduction == "none":
        return per_patient
    if reduction == "sum":
        return per_patient.sum()
    if reduction == "mean":
        return per_patient.mean()
    raise ValueError(f"Unknown reduction: {reduction}")


def future_feature_loss(
    prediction: PredictionDistribution,
    target: Tensor,
    valid: Tensor,
    *,
    objective: Literal["huber_cosine", "gaussian_nll"] = "huber_cosine",
) -> Tensor:
    """Compare a prior prediction with a frozen target, normalized by patient."""

    if target.requires_grad:
        raise DataContractError(
            code="TARGET_REQUIRES_GRAD",
            message="Frozen future-observation targets must not require gradients.",
        )
    if target.shape != prediction.mean.shape:
        raise ValueError("Future target and prediction mean shapes differ")
    if valid.shape != target.shape[:2] or valid.dtype is not torch.bool:
        raise ValueError("valid must be boolean with shape [B,K]")
    if objective == "huber_cosine":
        huber = F.smooth_l1_loss(prediction.mean, target, reduction="none").mean(dim=-1)
        cosine = 1.0 - F.cosine_similarity(prediction.mean, target, dim=-1, eps=1e-8)
        per_token = huber + cosine
    elif objective == "gaussian_nll":
        log_std = prediction.log_std.clamp(min=-6.0, max=2.0)
        inverse_variance = torch.exp(-2.0 * log_std)
        per_token = (0.5 * (prediction.mean - target).square() * inverse_variance + log_std).mean(
            dim=-1
        )
    else:
        raise ValueError(f"Unknown future feature objective: {objective}")
    return masked_patient_mean(per_token, valid)


def state_kl(
    posterior: BeliefState,
    prior: BeliefState,
    *,
    free_nats_per_token: float = 0.0,
) -> Tensor:
    if posterior.stochastic_mean is None or prior.stochastic_mean is None:
        return posterior.memory.new_zeros(())
    assert posterior.stochastic_log_std is not None
    assert prior.stochastic_log_std is not None
    return diagonal_gaussian_kl(
        posterior.stochastic_mean,
        posterior.stochastic_log_std,
        prior.stochastic_mean,
        prior.stochastic_log_std,
        free_nats_per_token=free_nats_per_token,
    )


@dataclass(frozen=True)
class LatentDiagnostics:
    target_variance: float
    prediction_variance: float
    target_norm: float
    prediction_norm: float
    effective_kl_dimensions: int | None
    mean_kl_per_dimension: float | None
    valid_patients: int
    valid_tokens: int
    valid_fraction: float
    prediction_collapsed: bool


def latent_diagnostics(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
    *,
    kl_per_dimension: Tensor | None = None,
    kl_valid: Tensor | None = None,
    active_kl_threshold: float = 0.01,
    collapse_variance_floor: float = 1e-8,
) -> LatentDiagnostics:
    if prediction.ndim != 3 or prediction.shape != target.shape or valid.shape != target.shape[:2]:
        raise ValueError("Diagnostic prediction, target, and mask shapes differ")
    if valid.dtype != torch.bool:
        raise ValueError("Diagnostic mask must be boolean")
    if valid.device != target.device or prediction.device != target.device:
        raise ValueError("Diagnostic prediction, target, and mask must share a device")
    if not math.isfinite(active_kl_threshold) or active_kl_threshold < 0:
        raise ValueError("active_kl_threshold must be finite and nonnegative")
    if not math.isfinite(collapse_variance_floor) or collapse_variance_floor < 0:
        raise ValueError("collapse_variance_floor must be finite and nonnegative")
    count = int(valid.sum().item())
    if count == 0:
        raise ValueError("Diagnostics require at least one valid target token")
    target_values = target[valid].detach().float()
    prediction_values = prediction[valid].detach().float()
    if not torch.isfinite(target_values).all() or not torch.isfinite(prediction_values).all():
        raise ValueError("Valid diagnostic prediction and target values must be finite")

    target_variance = target_values.var(dim=0, unbiased=False).mean()
    prediction_variance = prediction_values.var(dim=0, unbiased=False).mean()
    effective: int | None = None
    mean_kl: float | None = None
    if kl_per_dimension is not None:
        if kl_per_dimension.ndim != 3:
            raise ValueError("kl_per_dimension must have shape [B,M,D]")
        if kl_per_dimension.device != target.device:
            raise ValueError("KL diagnostics and feature diagnostics must share a device")
        if kl_valid is None:
            selected_kl = kl_per_dimension.reshape(-1, kl_per_dimension.shape[-1])
        else:
            if kl_valid.dtype != torch.bool or kl_valid.device != kl_per_dimension.device:
                raise ValueError("kl_valid must be boolean and share the KL tensor device")
            if kl_valid.shape == (kl_per_dimension.shape[0],):
                kl_valid = kl_valid[:, None].expand(kl_per_dimension.shape[:2])
            elif kl_valid.shape != kl_per_dimension.shape[:2]:
                raise ValueError("kl_valid must have shape [B] or [B,M]")
            if not kl_valid.any():
                raise ValueError("KL diagnostics require at least one valid state token")
            selected_kl = kl_per_dimension[kl_valid]
        selected_kl = selected_kl.detach().float()
        if not torch.isfinite(selected_kl).all():
            raise ValueError("Valid KL diagnostics must be finite")
        mean_kl_by_dimension = selected_kl.mean(dim=0)
        effective = int((mean_kl_by_dimension > active_kl_threshold).sum().item())
        mean_kl = float(mean_kl_by_dimension.mean().cpu())
    elif kl_valid is not None:
        raise ValueError("kl_valid requires kl_per_dimension")

    return LatentDiagnostics(
        target_variance=float(target_variance.cpu()),
        prediction_variance=float(prediction_variance.cpu()),
        target_norm=float(target_values.norm(dim=-1).mean().cpu()),
        prediction_norm=float(prediction_values.norm(dim=-1).mean().cpu()),
        effective_kl_dimensions=effective,
        mean_kl_per_dimension=mean_kl,
        valid_patients=int(valid.any(dim=1).sum().item()),
        valid_tokens=count,
        valid_fraction=count / valid.numel(),
        prediction_collapsed=bool(
            target_variance > collapse_variance_floor
            and prediction_variance <= collapse_variance_floor
        ),
    )


class FrozenModuleGuard:
    """In-memory assertion that a fixed teacher did not change across optimization."""

    def __init__(self, module: nn.Module):
        self._snapshot = {
            name: value.detach().cpu().clone() for name, value in module.state_dict().items()
        }

    def assert_unchanged(self, module: nn.Module) -> None:
        current = module.state_dict()
        if current.keys() != self._snapshot.keys():
            raise RuntimeError("Frozen target state keys changed")
        changed = [
            name
            for name, original in self._snapshot.items()
            if not torch.equal(current[name].detach().cpu(), original)
        ]
        if changed:
            raise RuntimeError(f"Frozen target parameters changed: {', '.join(changed)}")


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module
