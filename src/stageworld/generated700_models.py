"""Conditional spatial generation and endpoint-specific two-way fusion.

TwoWayFusion adapts CLARITY's MIT-licensed TwoWayCrossAttentionLayer;
see docs/CLARITY_LICENSE.txt and docs/GENERATED_V2_SOURCES.md. The spatial
residual dynamics and gated aggregation are task-specific implementations.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F

from stageworld.binary700_models import endpoint_loss, world_loss  # noqa: F401


class ConditionTokens(nn.Module):
    frequencies: torch.Tensor

    def __init__(self, hidden: int):
        super().__init__()
        self.clinical = nn.Linear(32, hidden)
        self.treatment = nn.Linear(82, hidden)
        self.time = nn.Sequential(nn.Linear(65, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.types = nn.Parameter(torch.randn(1, 6, hidden) * 0.02)
        self.register_buffer("frequencies", torch.exp(torch.linspace(0, -math.log(10000), 32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 361:
            raise ValueError("Require CT6(32), four no-cycle treatment tokens(328), and time(1)")
        phase = x[:, -1:] * self.frequencies
        time = self.time(torch.cat((x[:, -1:], phase.sin(), phase.cos()), 1))
        return (
            torch.cat(
                (
                    self.clinical(x[:, :32])[:, None],
                    self.treatment(x[:, 32:360].reshape(-1, 4, 82)),
                    time[:, None],
                ),
                1,
            )
            + self.types
        )


class SpatialTransition(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.transformer = nn.TransformerEncoderLayer(
            hidden, 4, hidden * 4, 0.1, "gelu", batch_first=True, norm_first=True
        )
        self.local = nn.Conv3d(hidden, hidden, 3, padding=1, groups=hidden)
        self.context = nn.Conv3d(hidden, hidden, 3, padding=2, dilation=2, groups=hidden)
        self.mix = nn.Conv3d(hidden, hidden, 1)
        self.film = nn.Linear(hidden, 2 * hidden)
        self.norm = nn.GroupNorm(8, hidden)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.transformer(tokens)
        conditions, image = tokens[:, :6], tokens[:, 6:]
        volume = image.transpose(1, 2).reshape(-1, image.shape[-1], 3, 3, 3)
        scale, shift = self.film(conditions.mean(1)).chunk(2, -1)
        normalized = self.norm(volume) * (1 + 0.1 * scale[:, :, None, None, None])
        normalized = normalized + 0.1 * shift[:, :, None, None, None]
        volume = volume + self.mix(F.gelu(self.local(normalized) + self.context(normalized)))
        return torch.cat((conditions, volume.flatten(2).transpose(1, 2)), 1)


class FutureWorld(nn.Module):
    input_mean: torch.Tensor
    input_scale: torch.Tensor
    coordinates: torch.Tensor

    def __init__(self, tabular_dim: int, image_dim: int = 768, hidden: int = 128):
        super().__init__()
        if tabular_dim != 361:
            raise ValueError("Unsupported clinical/treatment schema")
        self.hidden = hidden
        self.image = nn.Sequential(nn.Linear(image_dim, hidden), nn.LayerNorm(hidden))
        self.condition = ConditionTokens(hidden)
        coordinates = torch.stack(torch.meshgrid(*([torch.linspace(-1, 1, 3)] * 3), indexing="ij"))
        self.register_buffer("coordinates", coordinates.flatten(1).T[None])
        self.position = nn.Linear(3, hidden, bias=False)
        self.blocks = nn.ModuleList([SpatialTransition(hidden) for _ in range(4)])
        self.state_delta = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 2 * hidden),
            nn.GELU(),
            nn.Linear(2 * hidden, image_dim),
        )
        last = self.decoder[-1]
        assert isinstance(last, nn.Linear)
        nn.init.normal_(last.weight, std=0.001)
        nn.init.zeros_(last.bias)
        self.register_buffer("input_mean", torch.zeros(image_dim))
        self.register_buffer("input_scale", torch.ones(image_dim))

    @torch.no_grad()
    def fit_statistics(self, ct0: torch.Tensor, ct1: torch.Tensor) -> None:
        self.input_mean.copy_(ct0.mean((0, 1)))
        self.input_scale.copy_(ct0.std((0, 1), unbiased=False).clamp_min(0.05))
        last = self.decoder[-1]
        assert isinstance(last, nn.Linear)
        last.bias.copy_((ct1.mean((0, 1)) - ct0.mean((0, 1))) / self.input_scale)

    def forward(
        self, x: torch.Tensor, ct0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        initial = self.image((ct0 - self.input_mean) / self.input_scale) + self.position(
            self.coordinates
        )
        tokens = torch.cat((self.condition(x), initial), 1)
        for block in self.blocks:
            tokens = block(tokens)
        generated = initial + self.state_delta(tokens[:, 6:])
        prediction = ct0 + self.decoder(generated) * self.input_scale
        return initial, generated, prediction.float()


class TwoWayFusion(nn.Module):
    """Pre-norm two-way attention adapted from CLARITY (MIT, Tianxingjian Ding)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.attention = nn.ModuleList(
            [nn.MultiheadAttention(hidden, 4, dropout=0.1, batch_first=True) for _ in range(2)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(4)])
        self.feedforward = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden, hidden * 4),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden * 4, hidden),
                )
                for _ in range(2)
            ]
        )
        self.dropout = nn.Dropout(0.1)
        self.final = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(2)])

    def forward(
        self, baseline: torch.Tensor, future: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        baseline = baseline + self.dropout(
            self.attention[0](self.norms[0](baseline), future, future, need_weights=False)[0]
        )
        future = future + self.dropout(
            self.attention[1](self.norms[1](future), baseline, baseline, need_weights=False)[0]
        )
        baseline = baseline + self.dropout(self.feedforward[0](self.norms[2](baseline)))
        future = future + self.dropout(self.feedforward[1](self.norms[3](future)))
        return self.final[0](baseline), self.final[1](future)


class GatedPool(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.content = nn.Linear(hidden, hidden)
        self.gate = nn.Linear(hidden, hidden)
        self.score = nn.Linear(hidden, 1, bias=False)

    def forward(self, values: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        context = values + query[:, None]
        scores = self.score(self.content(context).tanh() * self.gate(context).sigmoid())
        return (scores.softmax(1) * values).sum(1)


class EndpointResidual(nn.Module):
    def __init__(self, tabular_dim: int, hidden: int):
        super().__init__()
        self.condition = nn.Sequential(
            nn.Linear(tabular_dim, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.fusion = TwoWayFusion(hidden)
        self.pools = nn.ModuleList([GatedPool(hidden) for _ in range(3)])
        self.body = nn.Sequential(
            nn.Linear(4 * hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
        last = self.body[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(
        self, x: torch.Tensor, initial: torch.Tensor, generated: torch.Tensor
    ) -> torch.Tensor:
        query = self.condition(x)
        baseline, future = self.fusion(initial, generated)
        parts = [
            pool(values, query)
            for pool, values in zip(self.pools, (baseline, future, future - baseline), strict=True)
        ]
        value = self.body(torch.cat((*parts, query), 1)).squeeze(-1)
        return 2 * (value / 2).tanh()


class AnchoredClassifier(nn.Module):
    anchor_weight: torch.Tensor
    anchor_bias: torch.Tensor
    residual_scale: torch.Tensor

    def __init__(
        self,
        tabular_dim: int,
        family: str,
        anchor_weight: torch.Tensor,
        anchor_bias: torch.Tensor,
        image_dim: int = 768,
    ):
        super().__init__()
        if family not in ("generated", "generated_frozen"):
            raise ValueError("Unsupported generated-v2 family")
        self.family = family
        self.register_buffer("anchor_weight", anchor_weight.float())
        self.register_buffer("anchor_bias", anchor_bias.float())
        self.register_buffer("residual_scale", torch.ones(2))
        self.world = FutureWorld(tabular_dim, image_dim)
        self.world.requires_grad_(family != "generated_frozen")
        self.endpoints = nn.ModuleList(
            [EndpointResidual(tabular_dim, self.world.hidden) for _ in range(2)]
        )

    def train(self, mode: bool = True) -> AnchoredClassifier:
        super().train(mode)
        if self.family == "generated_frozen":
            self.world.eval()
        return self

    def forward_with_features(
        self, x: torch.Tensor, ct0: torch.Tensor, ct0_valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ct0 = torch.where(ct0_valid[:, None, None], ct0, torch.zeros_like(ct0))
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.family != "generated_frozen"):
            initial, generated, features = self.world(x, ct0)
        delta = torch.stack([head(x, initial, generated) for head in self.endpoints], 1)
        anchor = F.linear(x.float(), self.anchor_weight, self.anchor_bias)
        return anchor + delta.float() * ct0_valid[:, None] * self.residual_scale, features

    def forward(self, x: torch.Tensor, ct0: torch.Tensor, ct0_valid: torch.Tensor) -> torch.Tensor:
        return self.forward_with_features(x, ct0, ct0_valid)[0]


@lru_cache(maxsize=8)
def _directions(dimension: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1729)
    directions = torch.randn(dimension, 64, generator=generator)
    return F.normalize(directions, dim=0).to(device)


def feature_set_loss(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Permutation-invariant supervision: crop tokens are not anatomically paired."""
    if not valid.any():
        return prediction.sum() * 0
    p, y = prediction[valid].float(), target[valid].detach().float()
    dimension = p.shape[-1]
    directions = _directions(dimension, p.device)
    with torch.autocast(p.device.type, enabled=False):
        projected_p, projected_y = (p @ directions).sort(1).values, (y @ directions).sort(1).values
    global_loss = world_loss(
        p.mean(1), y.mean(1), torch.ones(len(p), dtype=torch.bool, device=p.device)
    )
    distribution = (projected_p - projected_y).square().mean()
    spread = F.smooth_l1_loss(p.std(1, unbiased=False), y.std(1, unbiased=False))
    return global_loss + 0.25 * distribution + 0.1 * spread
