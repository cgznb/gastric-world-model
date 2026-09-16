"""Causal event recurrence with intermediate targets outside the forward API."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from stageworld.event_spec import ABSENT, CONFLICT, PRESENT, UNKNOWN
from stageworld.generated700_models import GatedPool, SpatialTransition, TwoWayFusion


@dataclass
class EventInputs:
    x: torch.Tensor
    ct0: torch.Tensor
    events: torch.Tensor

    def to(self, device: torch.device) -> EventInputs:
        return EventInputs(self.x.to(device), self.ct0.to(device), self.events.to(device))

    def validate(self, image_dim: int) -> None:
        n = len(self.x)
        if self.x.shape != (n, 360) or self.ct0.shape != (n, 27, image_dim):
            raise ValueError("Require CT6/no-cycle 360-vector and complete CT0 tokens")
        if self.events.shape != (n, 3) or self.events.dtype != torch.long:
            raise ValueError("Require three integer event-status codes")
        if not torch.isfinite(self.x).all() or not torch.isfinite(self.ct0).all():
            raise ValueError("Nonfinite permitted inputs")
        if ((self.events < ABSENT) | (self.events > CONFLICT)).any():
            raise ValueError("Unknown event-status code")
        if ((self.events[:, 2] == PRESENT) & (self.events[:, 1] != PRESENT)).any():
            raise ValueError("Postoperative event requires confirmed surgery")


@dataclass
class EventOutput:
    states: torch.Tensor
    ct1: torch.Tensor
    pcr_logit: torch.Tensor
    recurrence_logit: torch.Tensor
    last_stage: torch.Tensor
    incomplete_history: torch.Tensor


class StateHead(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.fusion = TwoWayFusion(hidden)
        self.pool = GatedPool(hidden)
        self.readout = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        query, tokens = self.fusion(self.query.expand(len(state), -1, -1), state)
        pooled = self.pool(tokens, query[:, 0])
        return self.readout(torch.cat((pooled, query[:, 0]), -1)).squeeze(-1).float()


class EventWorld(nn.Module):
    input_mean: torch.Tensor
    input_scale: torch.Tensor
    coordinates: torch.Tensor

    def __init__(self, image_dim: int = 768, hidden: int = 128, layers: int = 4):
        super().__init__()
        self.image_dim, self.hidden, self.layers = image_dim, hidden, layers
        self.image = nn.Sequential(nn.Linear(image_dim, hidden), nn.LayerNorm(hidden))
        self.clinical = nn.Sequential(nn.Linear(32, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.treatment = nn.Linear(82, hidden)
        self.event_codes = nn.ModuleList([nn.Embedding(4, hidden) for _ in range(3)])
        self.blocks = nn.ModuleList([SpatialTransition(hidden) for _ in range(layers)])
        self.delta = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
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
        axes = [torch.linspace(-1, 1, 3)] * 3
        self.register_buffer(
            "coordinates", torch.stack(torch.meshgrid(*axes, indexing="ij")).flatten(1).T[None]
        )
        self.position = nn.Linear(3, hidden, bias=False)
        self.register_buffer("input_mean", torch.zeros(image_dim))
        self.register_buffer("input_scale", torch.ones(image_dim))

    @torch.no_grad()
    def fit_statistics(self, ct0: torch.Tensor, ct1: torch.Tensor, ct1_valid: torch.Tensor) -> None:
        self.input_mean.copy_(ct0.mean((0, 1)))
        self.input_scale.copy_(ct0.std((0, 1), unbiased=False).clamp_min(0.05))
        if ct1_valid.any():
            last = self.decoder[-1]
            assert isinstance(last, nn.Linear)
            last.bias.copy_(
                (ct1[ct1_valid].mean((0, 1)) - ct0[ct1_valid].mean((0, 1))) / self.input_scale
            )

    def forward(self, inputs: EventInputs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs.validate(self.image_dim)
        clinical = self.clinical(inputs.x[:, :32])
        state = self.image((inputs.ct0 - self.input_mean) / self.input_scale)
        state = state + self.position(self.coordinates) + clinical[:, None]
        states = [state]
        last_stage = torch.zeros(len(state), dtype=torch.long, device=state.device)
        for stage in range(3):
            active = inputs.events[:, stage] == PRESENT
            if active.any():
                actions = (
                    self.treatment(inputs.x[:, 32:].reshape(-1, 4, 82))
                    if stage == 0
                    else state.new_zeros(len(state), 4, self.hidden)
                )
                conditions = torch.cat(
                    (
                        clinical[:, None],
                        self.event_codes[stage](inputs.events[:, stage])[:, None],
                        actions,
                    ),
                    1,
                )
                tokens = torch.cat((conditions, state), 1)
                for block in self.blocks:
                    tokens = block(tokens)
                candidate = state + self.delta(tokens[:, 6:])
                state = torch.where(active[:, None, None], candidate, state)
                last_stage = torch.where(active, stage + 1, last_stage)
            states.append(state)
        trajectory = torch.stack(states, 1)
        ct1 = inputs.ct0 + self.decoder(trajectory[:, 1]) * self.input_scale
        return trajectory, ct1.float(), last_stage


class EventModel(nn.Module):
    def __init__(self, image_dim: int = 768, hidden: int = 128, layers: int = 4):
        super().__init__()
        self.world = EventWorld(image_dim, hidden, layers)
        self.pcr_head = StateHead(hidden)
        self.recurrence_head = StateHead(hidden)

    def forward(self, inputs: EventInputs) -> EventOutput:
        states, ct1, last_stage = self.world(inputs)
        return EventOutput(
            states,
            ct1,
            self.pcr_head(states[:, 2]),
            self.recurrence_head(states[:, -1]),
            last_stage,
            ((inputs.events == UNKNOWN) | (inputs.events == CONFLICT)).any(1),
        )

    def dimensions(self) -> dict:
        return {
            "image_dim": self.world.image_dim,
            "hidden": self.world.hidden,
            "layers": self.world.layers,
        }


def masked_bce(
    logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor, positive_weight: float = 1.0
) -> torch.Tensor:
    if not valid.any():
        return logits.sum() * 0
    y = labels[valid].detach().float()
    if not ((y == 0) | (y == 1)).all():
        raise ValueError("Valid binary targets must be zero or one")
    weight = torch.where(y.bool(), positive_weight, 1.0)
    loss = F.binary_cross_entropy_with_logits(logits[valid].float(), y, reduction="none")
    return (loss * weight).sum() / weight.sum()
