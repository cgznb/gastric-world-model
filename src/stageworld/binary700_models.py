"""Independent endpoint residuals anchored to a fitted clinical logistic model."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from stageworld.model.compact_residual_binary import ResidualBlock


class FutureWorld(nn.Module):
    def __init__(self, tabular_dim: int, image_dim: int = 768, hidden: int = 32):
        super().__init__()
        self.image = nn.Sequential(nn.Linear(image_dim, hidden), nn.LayerNorm(hidden))
        self.condition = nn.Sequential(
            nn.Linear(tabular_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.blocks = nn.ModuleList([ResidualBlock(hidden, 0.1) for _ in range(2)])
        self.decoder = nn.Linear(hidden, image_dim)
        self.hidden = hidden
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(
        self, x: torch.Tensor, ct0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        initial = self.image(ct0)
        state = initial.transpose(1, 2).reshape(-1, self.hidden, 3, 3, 3)
        condition = self.condition(x)
        for block in self.blocks:
            state = block(state, condition)
        generated = state.flatten(2).transpose(1, 2)
        return initial, generated, self.decoder(generated.mean(1)).float()


class EndpointResidual(nn.Module):
    def __init__(self, tabular_dim: int, family: str, image_dim: int = 768):
        super().__init__()
        self.family = family
        self.image = (
            nn.Sequential(nn.Linear(image_dim, 32), nn.LayerNorm(32)) if family == "ct" else None
        )
        self.query = nn.Linear(tabular_dim, 32) if family != "tabular" else None
        width = tabular_dim + (0 if family == "tabular" else 32 if family == "ct" else 96)
        self.body = nn.Sequential(
            nn.Linear(width, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )
        last = self.body[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def pool(self, x: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        assert self.query is not None
        score = (tokens * self.query(x)[:, None]).sum(-1) / math.sqrt(32)
        return (score.softmax(1)[..., None] * tokens).sum(1)

    def forward(
        self,
        x: torch.Tensor,
        ct0: torch.Tensor,
        available: torch.Tensor,
        world: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        if self.family == "tabular":
            inputs = x
        elif self.family == "ct":
            assert self.image is not None
            inputs = torch.cat((x, self.pool(x, self.image(ct0))), 1)
        else:
            assert world is not None
            initial, generated = world
            inputs = torch.cat(
                (
                    x,
                    self.pool(x, initial),
                    self.pool(x, generated),
                    self.pool(x, generated - initial),
                ),
                1,
            )
        delta = 2 * (self.body(inputs).squeeze(-1) / 2).tanh()
        return delta if self.family == "tabular" else delta * available


class AnchoredClassifier(nn.Module):
    anchor_weight: torch.Tensor
    anchor_bias: torch.Tensor

    def __init__(
        self,
        tabular_dim: int,
        family: str,
        anchor_weight: torch.Tensor,
        anchor_bias: torch.Tensor,
        image_dim: int = 768,
    ):
        super().__init__()
        if family not in ("tabular", "ct", "generated"):
            raise ValueError("Unknown residual architecture")
        self.family = family
        self.register_buffer("anchor_weight", anchor_weight.float())
        self.register_buffer("anchor_bias", anchor_bias.float())
        self.world = FutureWorld(tabular_dim, image_dim) if family == "generated" else None
        if self.world is not None:
            self.world.requires_grad_(False)
        self.endpoints = nn.ModuleList(
            [EndpointResidual(tabular_dim, family, image_dim) for _ in range(2)]
        )

    def train(self, mode: bool = True) -> AnchoredClassifier:
        super().train(mode)
        if self.world is not None:
            self.world.eval()
        return self

    def forward(self, x: torch.Tensor, ct0: torch.Tensor, ct0_valid: torch.Tensor) -> torch.Tensor:
        ct0 = torch.where(ct0_valid[:, None, None], ct0, torch.zeros_like(ct0))
        anchor = F.linear(x.float(), self.anchor_weight.float(), self.anchor_bias.float())
        world = None
        if self.world is not None:
            with torch.no_grad():
                initial, generated, _ = self.world(x, ct0)
                world = initial, generated
        delta = torch.stack([head(x, ct0, ct0_valid, world) for head in self.endpoints], 1)
        return anchor.float() + delta.float()


def endpoint_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
    positive_weight: torch.Tensor,
    loss: str = "bce",
) -> torch.Tensor:
    terms = []
    for i in range(2):
        selected = valid[:, i]
        if not selected.any():
            continue
        z, y = logits[selected, i].float(), labels[selected, i].float()
        weight = torch.where(y.bool(), positive_weight[i], 1.0)
        per_label = F.binary_cross_entropy_with_logits(z, y, reduction="none")
        if loss == "focal":
            p_t = torch.where(y.bool(), z.sigmoid(), (-z).sigmoid())
            per_label = per_label * (1 - p_t).square()
        elif loss != "bce":
            raise ValueError("Unknown endpoint loss")
        terms.append((per_label * weight).sum() / weight.sum())
    return sum(terms, logits.sum() * 0)


def world_loss(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if not valid.any():
        return prediction.sum() * 0
    p, y = prediction[valid].float(), target[valid].detach().float()
    return F.smooth_l1_loss(p, y) + (1 - F.cosine_similarity(p, y, dim=-1)).mean()
