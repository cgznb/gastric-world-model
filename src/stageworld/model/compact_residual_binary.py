"""Deterministic spatial residual forecasts with baseline-only endpoint inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from stageworld.binary_endpoints import BinaryEndpointConfig, BinaryEndpointModel
from stageworld.data.baseline_clinical import CT6_CLINICAL_SCHEMA, BaselineClinical
from stageworld.encoders.base import ObservationTokens
from stageworld.errors import DataContractError
from stageworld.model.components import ContinuousTimeEncoder
from stageworld.model.generated_s1 import BaselineClinicalEncoder, clean_action_padding
from stageworld.model.types import ActionTokens


@dataclass(frozen=True)
class CompactConfig:
    architecture: str = "residual"
    hidden_dim: int = 64
    input_dim: int = 768
    action_dim: int = 82
    dropout: float = 0.1
    schema_version: str = "ct6-compact-binary-v1"

    def __post_init__(self) -> None:
        if self.architecture not in ("direct", "residual") or self.hidden_dim % 8:
            raise ValueError("Use a direct/residual model with channels divisible by eight")


@dataclass
class ForecastOutput:
    logits: Tensor
    ct_mean: Tensor | None
    initial: Tensor
    generated: Tensor

    @property
    def probabilities(self) -> Tensor:
        return self.logits.sigmoid()


def spatial_order(observation: ObservationTokens) -> Tensor:
    """Canonicalize Cartesian grid levels, preserving BF16-rounded coordinates."""
    coords = observation.coords
    if coords is None or coords.shape[1:] != (27, 3) or not observation.valid.all():
        raise DataContractError(code="COMPACT_GRID", message="Require a complete 3x3x3 CT grid")
    coords = coords.float()
    sorted_coords = coords.sort(dim=1).values
    levels = sorted_coords[:, [0, 9, 18], :]
    if not torch.isfinite(coords).all() or (levels[:, 1:] <= levels[:, :-1]).any():
        raise DataContractError(code="COMPACT_GRID", message="Invalid CT coordinates")
    expected = levels[:, :, None, :].expand(-1, -1, 9, -1)
    if not torch.allclose(sorted_coords.reshape(-1, 3, 9, 3), expected, atol=1e-4, rtol=0):
        raise DataContractError(code="COMPACT_GRID", message="Require three Cartesian grid levels")
    # Cache extraction computes physical centers in BF16 before converting to FP32.
    # Level ranks preserve voxel adjacency without assuming exact metric spacing.
    index = (coords[:, :, None, :] - levels[:, None, :, :]).abs().argmin(2)
    key = (index * index.new_tensor([9, 3, 1])).sum(-1)
    ordered, permutation = key.sort(dim=1)
    if not torch.equal(ordered, torch.arange(27, device=coords.device).expand_as(ordered)):
        raise DataContractError(code="COMPACT_GRID", message="Repeated or missing grid position")
    return permutation


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.film = nn.Linear(dim, 2 * dim)
        self.net = nn.Sequential(
            nn.Conv3d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(dim, dim, 3, padding=1),
        )

    def forward(self, state: Tensor, condition: Tensor) -> Tensor:
        gamma, beta = self.film(condition).chunk(2, -1)
        hidden = self.norm(state) * (1 + gamma.tanh()[:, :, None, None, None])
        hidden = hidden + beta[:, :, None, None, None]
        return state + self.net(hidden)


class CompactBinaryModel(nn.Module):
    def __init__(self, config: CompactConfig):
        super().__init__()
        self.config = config
        dim = config.hidden_dim
        self.ct_projection = nn.Sequential(nn.Linear(config.input_dim, dim), nn.LayerNorm(dim))
        self.clinical = BaselineClinicalEncoder(dim, CT6_CLINICAL_SCHEMA)
        self.action = nn.Sequential(nn.Linear(config.action_dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.time = ContinuousTimeEncoder(dim)
        self.condition = (
            nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, dim))
            if config.architecture == "residual"
            else None
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(dim, config.dropout) for _ in range(2)]
            if config.architecture == "residual"
            else []
        )
        self.feature_decoder = (
            nn.Linear(dim, config.input_dim) if config.architecture == "residual" else None
        )
        # Direct fusion includes declared time, so comparisons have the same input access.
        width = dim * (5 if config.architecture == "residual" else 4)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(width),
                    nn.Linear(width, dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(dim, 1),
                )
                for _ in range(2)
            ]
        )

    def forward(
        self,
        *,
        ct0: ObservationTokens,
        baseline: BaselineClinical,
        s0_time: Tensor,
        scenario_actions: ActionTokens,
        target_time: Tensor,
        horizons: Tensor,
        scenario: str,
        deterministic: bool = True,
        generator: torch.Generator | None = None,
        generated_edit: str | None = None,
    ) -> ForecastOutput:
        del horizons, deterministic, generator
        if not scenario.strip() or not torch.isfinite(target_time).all():
            raise DataContractError(code="COMPACT_QUERY", message="Declare finite scenario times")
        if (
            not torch.isfinite(s0_time).all()
            or (target_time <= s0_time).any()
            or (
                ct0.available_time[ct0.valid] > s0_time[:, None].expand_as(ct0.valid)[ct0.valid]
            ).any()
        ):
            raise DataContractError(code="COMPACT_PREFIX", message="Invalid baseline prefix")
        permutation = spatial_order(ct0)
        values = ct0.values.gather(1, permutation[..., None].expand_as(ct0.values))
        initial = self.ct_projection(values)
        clinical = self.clinical(baseline).mean(1)
        actions = clean_action_padding(scenario_actions)
        count = actions.valid.sum(1, keepdim=True).clamp_min(1)
        action = (self.action(actions.values) * actions.valid[..., None]).sum(1) / count
        time = self.time(target_time - s0_time)
        if self.config.architecture == "direct":
            generated = initial
            vector = torch.cat((initial.mean(1), clinical, action, time), -1)
            ct_mean = None
        else:
            assert self.condition is not None
            condition = self.condition(torch.cat((clinical, action, time), -1))
            state = initial.transpose(1, 2).reshape(-1, self.config.hidden_dim, 3, 3, 3)
            for block in self.blocks:
                state = block(state, condition)
            generated = state.flatten(2).transpose(1, 2)
            if generated_edit == "shuffle":
                generated = generated.roll(1, 0)
            elif generated_edit == "initial":
                generated = initial
            elif generated_edit is not None:
                raise ValueError("Unknown diagnostic edit")
            vector = torch.cat(
                (
                    initial.mean(1),
                    generated.mean(1),
                    (generated - initial).mean(1),
                    clinical,
                    action,
                ),
                -1,
            )
            assert self.feature_decoder is not None
            ct_mean = self.feature_decoder(generated).mean(1, keepdim=True)
        return ForecastOutput(
            torch.cat([head(vector) for head in self.heads], -1).float(),
            None if ct_mean is None else ct_mean.float(),
            initial,
            generated,
        )


class DeterministicLegacyModel(nn.Module):
    """Keep the old architecture; use state means and omit posterior/KL supervision."""

    def __init__(self, config: BinaryEndpointConfig):
        super().__init__()
        self.core = BinaryEndpointModel(config)

    def forward(self, **inputs: Any) -> ForecastOutput:
        inputs["deterministic"] = True
        output = self.core(**inputs)
        return ForecastOutput(
            output.logits,
            output.future_ct.mean,
            self.core._state_tokens(output.baseline.state),
            self.core._state_tokens(output.state_s1_pred),
        )

    @property
    def heads(self) -> nn.Module:
        return self.core.endpoint_decoder
