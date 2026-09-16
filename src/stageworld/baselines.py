"""Matched-input neural survival baselines.

These models deliberately share :class:`PiecewiseHazardHead` so architecture
comparisons do not silently change survival parameterization.  Classes named
as adaptations are project-owned comparators, not reproductions of unlicensed
or incompatible upstream implementations.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .survival import PiecewiseHazardHead, PiecewiseHazardOutput


@dataclass(frozen=True)
class BaselineMetadata:
    name: str
    implementation_scope: str
    exact_upstream_reproduction: bool = False


def _masked_observation_mean(
    values: Tensor,
    valid: Tensor | None,
    *,
    expected_dim: int,
    name: str,
) -> tuple[Tensor, Tensor]:
    if values.ndim not in (2, 3) or values.shape[-1] != expected_dim:
        raise ValueError(f"{name} must have shape [B, {expected_dim}] or [B, N, {expected_dim}]")
    if values.ndim == 2:
        if valid is None:
            present = torch.ones(values.shape[0], dtype=torch.bool, device=values.device)
        else:
            present = torch.as_tensor(valid, dtype=torch.bool, device=values.device)
            if present.shape == (values.shape[0], 1):
                present = present.squeeze(1)
            if present.shape != (values.shape[0],):
                raise ValueError(f"{name} valid mask must have shape [B]")
        pooled = torch.where(present[:, None], values, torch.zeros_like(values))
        return pooled, present

    if valid is None:
        mask = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
    else:
        mask = torch.as_tensor(valid, dtype=torch.bool, device=values.device)
        if mask.shape != values.shape[:2]:
            raise ValueError(f"{name} valid mask must have shape [B, N]")
    safe_values = torch.where(mask[..., None], values, torch.zeros_like(values))
    counts = mask.sum(dim=1, keepdim=True)
    pooled = safe_values.sum(dim=1) / counts.clamp_min(1)
    return pooled, counts.squeeze(1) > 0


class DirectFusionSurvival(nn.Module):
    """A transparent static multimodal fusion control."""

    metadata = BaselineMetadata(
        name="direct_fusion",
        implementation_scope="StageWorld matched-input concatenation control",
    )

    def __init__(
        self,
        input_dims: Mapping[str, int],
        cuts: Tensor | Sequence[float],
        *,
        hidden_dim: int = 64,
        num_causes: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not input_dims or hidden_dim < 1:
            raise ValueError("at least one modality and a positive hidden_dim are required")
        if any(dim < 1 for dim in input_dims.values()):
            raise ValueError("all modality dimensions must be positive")
        self.modality_names = tuple(input_dims)
        self.input_dims = dict(input_dims)
        self.projections = nn.ModuleDict(
            {name: nn.Linear(dim, hidden_dim) for name, dim in input_dims.items()}
        )
        self.missing_embeddings = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(hidden_dim)) for name in input_dims}
        )
        fused_dim = len(input_dims) * (hidden_dim + 1)
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.survival_head = PiecewiseHazardHead(hidden_dim, cuts, num_causes=num_causes)

    def forward(
        self,
        features: Mapping[str, Tensor],
        valid_masks: Mapping[str, Tensor] | None = None,
    ) -> PiecewiseHazardOutput:
        if set(features) != set(self.modality_names):
            raise ValueError(f"features must contain exactly {self.modality_names}")
        valid_masks = {} if valid_masks is None else valid_masks
        pieces: list[Tensor] = []
        batch_size: int | None = None
        for name in self.modality_names:
            values = features[name]
            if batch_size is None:
                batch_size = values.shape[0]
            elif values.shape[0] != batch_size:
                raise ValueError("all modalities must have the same batch size")
            pooled, present = _masked_observation_mean(
                values,
                valid_masks.get(name),
                expected_dim=self.input_dims[name],
                name=name,
            )
            encoded = self.projections[name](pooled)
            missing = self.missing_embeddings[name].expand_as(encoded)
            encoded = torch.where(present[:, None], encoded, missing)
            pieces.extend((encoded, present.to(encoded).unsqueeze(1)))
        fused = self.fusion(torch.cat(pieces, dim=-1))
        return self.survival_head(fused)


def _continuous_time_encoding(times: Tensor, dim: int) -> Tensor:
    if dim < 1:
        raise ValueError("time encoding dimension must be positive")
    half = dim // 2
    if half == 0:
        return torch.log1p(times).unsqueeze(-1)
    frequency = torch.exp(
        torch.arange(half, dtype=times.dtype, device=times.device)
        * (-math.log(10_000.0) / max(half - 1, 1))
    )
    phase = torch.log1p(times).unsqueeze(-1) * frequency
    encoded = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
    if encoded.shape[-1] < dim:
        encoded = torch.cat((encoded, torch.log1p(times).unsqueeze(-1)), dim=-1)
    return encoded


def _validated_sequence(
    sequence: Tensor,
    valid: Tensor,
    times: Tensor | None,
    *,
    input_dim: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if sequence.ndim != 3 or sequence.shape[-1] != input_dim:
        raise ValueError(f"sequence must have shape [B, T, {input_dim}]")
    mask = torch.as_tensor(valid, dtype=torch.bool, device=sequence.device)
    if mask.shape != sequence.shape[:2]:
        raise ValueError("valid must have shape [B, T]")
    if torch.any(mask.sum(dim=1) == 0):
        raise ValueError("every patient must have at least one valid observation")
    if times is None:
        time_values = torch.arange(
            sequence.shape[1], dtype=sequence.dtype, device=sequence.device
        ).expand(sequence.shape[0], -1)
    else:
        time_values = torch.as_tensor(times, dtype=sequence.dtype, device=sequence.device)
        if time_values.shape != sequence.shape[:2]:
            raise ValueError("times must have shape [B, T]")
    if not torch.isfinite(time_values[mask]).all() or torch.any(time_values[mask] < 0):
        raise ValueError("visible times must be finite and nonnegative")
    safe_sequence = torch.where(mask[..., None], sequence, torch.zeros_like(sequence))
    safe_times = torch.where(mask, time_values, torch.zeros_like(time_values))
    return safe_sequence, mask, safe_times


class LongitudinalTransformerSurvival(nn.Module):
    """Ordinary masked longitudinal Transformer baseline."""

    metadata = BaselineMetadata(
        name="longitudinal_transformer",
        implementation_scope="StageWorld matched-input sequence control",
    )

    def __init__(
        self,
        input_dim: int,
        cuts: Tensor | Sequence[float],
        *,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        num_causes: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or hidden_dim % num_heads != 0:
            raise ValueError("dimensions must be positive and hidden_dim divisible by num_heads")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
            enable_nested_tensor=False,
        )
        self.survival_head = PiecewiseHazardHead(hidden_dim, cuts, num_causes=num_causes)

    def forward(
        self,
        sequence: Tensor,
        valid: Tensor,
        times: Tensor | None = None,
    ) -> PiecewiseHazardOutput:
        sequence, mask, times = _validated_sequence(
            sequence, valid, times, input_dim=self.input_dim
        )
        tokens = self.input_projection(sequence)
        tokens = tokens + _continuous_time_encoding(times, self.hidden_dim)
        encoded = self.encoder(tokens, src_key_padding_mask=~mask)
        encoded = torch.where(mask[..., None], encoded, torch.zeros_like(encoded))
        pooled = encoded.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        return self.survival_head(pooled)


class GRUDynamicSurvival(nn.Module):
    """Compact recurrent dynamic-risk control using the shared rate head."""

    metadata = BaselineMetadata(
        name="gru_dynamic",
        implementation_scope="StageWorld GRU prefix baseline",
    )

    def __init__(
        self,
        input_dim: int,
        cuts: Tensor | Sequence[float],
        *,
        hidden_dim: int = 64,
        num_layers: int = 1,
        num_causes: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or num_layers < 1:
            raise ValueError("input_dim, hidden_dim and num_layers must be positive")
        self.input_dim = input_dim
        self.gru = nn.GRU(
            input_size=input_dim + 1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.survival_head = PiecewiseHazardHead(hidden_dim, cuts, num_causes=num_causes)

    def forward(
        self,
        sequence: Tensor,
        valid: Tensor,
        times: Tensor | None = None,
    ) -> PiecewiseHazardOutput:
        sequence, mask, times = _validated_sequence(
            sequence, valid, times, input_dim=self.input_dim
        )
        if torch.any(mask[:, 1:] & ~mask[:, :-1]):
            raise ValueError("GRU valid masks must be contiguous prefixes")
        recurrent_input = torch.cat((sequence, torch.log1p(times).unsqueeze(-1)), dim=-1)
        lengths = mask.sum(dim=1).to(device="cpu", dtype=torch.long)
        packed = nn.utils.rnn.pack_padded_sequence(
            recurrent_input, lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return self.survival_head(hidden[-1])


class DynamicDeepHitAdaptation(GRUDynamicSurvival):
    """Clean-room dynamic GRU comparator, explicitly not Dynamic-DeepHit."""

    metadata = BaselineMetadata(
        name="dynamic_deephit_adaptation",
        implementation_scope=(
            "Independent PyTorch GRU adaptation with StageWorld piecewise-rate head; "
            "not the upstream TensorFlow joint-PMF/ranking implementation"
        ),
        exact_upstream_reproduction=False,
    )


def _as_tokens(values: Tensor, *, expected_dim: int, name: str) -> Tensor:
    if values.ndim == 2 and values.shape[-1] == expected_dim:
        return values.unsqueeze(1)
    if values.ndim == 3 and values.shape[-1] == expected_dim:
        return values
    raise ValueError(f"{name} must have shape [B, {expected_dim}] or [B, N, {expected_dim}]")


class CTSMambaFeatureAdaptation(nn.Module):
    """Independent paired-CT co-attention feature baseline, not CTSMamba."""

    metadata = BaselineMetadata(
        name="ctsmamba_feature_adaptation",
        implementation_scope=(
            "Independent paired frozen-feature co-attention model with StageWorld hazard head; "
            "not the upstream image/segmentation/Mamba implementation"
        ),
        exact_upstream_reproduction=False,
    )

    def __init__(
        self,
        input_dim: int,
        cuts: Tensor | Sequence[float],
        *,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_causes: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or hidden_dim % num_heads != 0:
            raise ValueError("dimensions must be positive and hidden_dim divisible by num_heads")
        self.input_dim = input_dim
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.pre_to_post = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.post_to_pre = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.survival_head = PiecewiseHazardHead(hidden_dim, cuts, num_causes=num_causes)

    def forward(
        self,
        baseline_ct: Tensor,
        post_ct: Tensor,
        baseline_valid: Tensor | None = None,
        post_valid: Tensor | None = None,
    ) -> PiecewiseHazardOutput:
        baseline = _as_tokens(baseline_ct, expected_dim=self.input_dim, name="baseline_ct")
        post = _as_tokens(post_ct, expected_dim=self.input_dim, name="post_ct")
        if baseline.shape[0] != post.shape[0]:
            raise ValueError("paired CT batches must have the same size")
        baseline_mask = self._token_mask(baseline, baseline_valid, "baseline_valid")
        post_mask = self._token_mask(post, post_valid, "post_valid")
        if torch.any(baseline_mask.sum(dim=1) == 0) or torch.any(post_mask.sum(dim=1) == 0):
            raise ValueError("paired-CT adaptation requires both CT observations")
        baseline = self.projection(
            torch.where(baseline_mask[..., None], baseline, torch.zeros_like(baseline))
        )
        post = self.projection(torch.where(post_mask[..., None], post, torch.zeros_like(post)))
        baseline_context, _ = self.pre_to_post(
            baseline, post, post, key_padding_mask=~post_mask, need_weights=False
        )
        post_context, _ = self.post_to_pre(
            post, baseline, baseline, key_padding_mask=~baseline_mask, need_weights=False
        )
        baseline_pool = self._masked_mean(baseline_context, baseline_mask)
        post_pool = self._masked_mean(post_context, post_mask)
        return self.survival_head(self.fusion(torch.cat((baseline_pool, post_pool), dim=-1)))

    @staticmethod
    def _token_mask(tokens: Tensor, valid: Tensor | None, name: str) -> Tensor:
        if valid is None:
            return torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        result = torch.as_tensor(valid, dtype=torch.bool, device=tokens.device)
        if result.shape == (tokens.shape[0],) and tokens.shape[1] == 1:
            result = result.unsqueeze(1)
        if result.shape != tokens.shape[:2]:
            raise ValueError(f"{name} must have shape [B, N]")
        return result

    @staticmethod
    def _masked_mean(tokens: Tensor, mask: Tensor) -> Tensor:
        safe = torch.where(mask[..., None], tokens, torch.zeros_like(tokens))
        return safe.sum(dim=1) / mask.sum(dim=1, keepdim=True)


__all__ = [
    "BaselineMetadata",
    "CTSMambaFeatureAdaptation",
    "DirectFusionSurvival",
    "DynamicDeepHitAdaptation",
    "GRUDynamicSurvival",
    "LongitudinalTransformerSurvival",
]
