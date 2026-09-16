from __future__ import annotations

import pytest
import torch
from torch import nn

from stageworld.errors import DataContractError
from stageworld.losses import (
    FrozenModuleGuard,
    diagonal_gaussian_kl,
    diagonal_gaussian_kl_per_dimension,
    freeze_module,
    future_feature_loss,
    latent_diagnostics,
)
from stageworld.model.types import PredictionDistribution


def test_gaussian_kl_identity_and_gradient() -> None:
    mean = torch.zeros(2, 3, 4, requires_grad=True)
    log_std = torch.zeros_like(mean, requires_grad=True)
    kl = diagonal_gaussian_kl(mean, log_std, mean.detach(), log_std.detach())
    assert kl.item() == pytest.approx(0.0)
    (kl + mean.square().mean()).backward()
    assert mean.grad is not None


def test_future_target_must_be_frozen() -> None:
    prediction = PredictionDistribution(
        modality="ct",
        mean=torch.zeros(2, 1, 4, requires_grad=True),
        log_std=torch.zeros(2, 1, 4, requires_grad=True),
        target_time=torch.ones(2),
        provenance="predicted_not_observed",
        scenario="test",
    )
    with pytest.raises(DataContractError) as exc:
        future_feature_loss(
            prediction,
            torch.ones(2, 1, 4, requires_grad=True),
            torch.ones(2, 1, dtype=torch.bool),
        )
    assert exc.value.code == "TARGET_REQUIRES_GRAD"


def test_feature_loss_and_diagnostics_detect_nonconstant_signal() -> None:
    target = torch.arange(8, dtype=torch.float32).reshape(2, 1, 4)
    mean = target.clone().add(0.2).requires_grad_()
    prediction = PredictionDistribution(
        modality="ct",
        mean=mean,
        log_std=torch.zeros_like(mean),
        target_time=torch.ones(2),
        provenance="predicted_not_observed",
        scenario="test",
    )
    valid = torch.ones(2, 1, dtype=torch.bool)
    loss = future_feature_loss(prediction, target, valid)
    loss.backward()
    assert mean.grad is not None and mean.grad.abs().sum() > 0
    diagnostics = latent_diagnostics(mean.detach(), target, valid)
    assert diagnostics.target_variance > 0
    assert diagnostics.prediction_variance > 0
    assert diagnostics.target_norm > 0
    assert diagnostics.prediction_norm > 0
    assert diagnostics.valid_patients == 2
    assert diagnostics.valid_tokens == 2
    assert diagnostics.valid_fraction == 1.0
    assert diagnostics.effective_kl_dimensions is None
    assert diagnostics.mean_kl_per_dimension is None
    assert not diagnostics.prediction_collapsed


def test_latent_diagnostics_detects_constant_prediction_and_ignores_masked_values() -> None:
    target = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
            [[3.0, 4.0, 5.0], [float("nan"), float("nan"), float("nan")]],
        ]
    )
    prediction = torch.tensor(
        [
            [[4.0, 5.0, 6.0], [4.0, 5.0, 6.0]],
            [[4.0, 5.0, 6.0], [float("nan"), float("nan"), float("nan")]],
        ]
    )
    valid = torch.tensor([[True, True], [True, False]])

    diagnostics = latent_diagnostics(prediction, target, valid)

    assert diagnostics.target_variance > 0
    assert diagnostics.prediction_variance == pytest.approx(0.0)
    assert diagnostics.prediction_norm > 0
    assert diagnostics.valid_patients == 2
    assert diagnostics.valid_tokens == 3
    assert diagnostics.valid_fraction == pytest.approx(0.75)
    assert diagnostics.prediction_collapsed

    zero_prediction = PredictionDistribution(
        modality="ct",
        mean=torch.zeros_like(target),
        log_std=torch.zeros_like(target),
        target_time=torch.ones(2),
        provenance="predicted_not_observed",
        scenario="test",
    )
    exact_prediction = PredictionDistribution(
        modality="ct",
        mean=target.nan_to_num(),
        log_std=torch.zeros_like(target),
        target_time=torch.ones(2),
        provenance="predicted_not_observed",
        scenario="test",
    )
    assert future_feature_loss(zero_prediction, target.nan_to_num(), valid) > future_feature_loss(
        exact_prediction, target.nan_to_num(), valid
    )


def test_latent_diagnostics_counts_effective_kl_dimensions_on_valid_rows_only() -> None:
    posterior_mean = torch.tensor(
        [
            [[0.2, 0.01, 0.0], [0.2, 0.01, 0.0]],
            [[0.0, 0.0, 100.0], [0.0, 0.0, 100.0]],
        ]
    )
    zeros = torch.zeros_like(posterior_mean)
    per_dimension = diagonal_gaussian_kl_per_dimension(
        posterior_mean,
        zeros,
        zeros,
        zeros,
    )
    prediction = torch.tensor([[[0.0, 1.0]], [[1.0, 2.0]]])
    target = prediction + 0.5
    valid = torch.ones(2, 1, dtype=torch.bool)

    diagnostics = latent_diagnostics(
        prediction,
        target,
        valid,
        kl_per_dimension=per_dimension,
        kl_valid=torch.tensor([True, False]),
        active_kl_threshold=0.01,
    )

    assert diagnostics.effective_kl_dimensions == 1
    assert diagnostics.mean_kl_per_dimension == pytest.approx((0.02 + 0.00005) / 3)


def test_latent_diagnostics_rejects_ambiguous_kl_mask() -> None:
    values = torch.ones(2, 1, 2)
    with pytest.raises(ValueError, match=r"shape \[B\] or \[B,M\]"):
        latent_diagnostics(
            values,
            values,
            torch.ones(2, 1, dtype=torch.bool),
            kl_per_dimension=torch.zeros(2, 3, 4),
            kl_valid=torch.ones(2, 1, dtype=torch.bool),
        )


def test_frozen_module_guard_detects_change() -> None:
    teacher = freeze_module(nn.Linear(3, 2))
    guard = FrozenModuleGuard(teacher)
    guard.assert_unchanged(teacher)
    with torch.no_grad():
        teacher.weight.add_(1.0)
    with pytest.raises(RuntimeError, match="changed"):
        guard.assert_unchanged(teacher)
