"""Event-conditioned feature dynamics with stage adapters and ensemble readouts.

The implementation combines architectural ideas independently; it is not an
upstream model reproduction. Observed follow-up CT and endpoint labels are never
forward inputs. Ensemble variation is not a calibrated uncertainty estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from stageworld.event_models import EventInputs, EventOutput
from stageworld.event_spec import CONFLICT, PRESENT, UNKNOWN

CLINICAL_WIDTHS = (3, 2, 2, 12, 9, 4)
VARIANTS = ("full", "no_stage_adapter", "no_transition")


@dataclass
class EventV2Output(EventOutput):
    member_logits: torch.Tensor
    stage_increments: torch.Tensor


class ClinicalTokenizer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.fields = nn.ModuleList(
            nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.LayerNorm(hidden))
            for width in CLINICAL_WIDTHS
        )
        self.field_identity = nn.Parameter(torch.randn(6, hidden) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fields = torch.split(x[:, :32], list(CLINICAL_WIDTHS), dim=-1)
        return (
            torch.stack([layer(field) for layer, field in zip(self.fields, fields, strict=True)], 1)
            + self.field_identity[None]
        )


class StageTransition(nn.Module):
    def __init__(self, hidden: int, rank: int, adapters: bool):
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(4))
        self.self_attention = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.condition_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Linear(4 * hidden, hidden)
        )
        self.gates = nn.Linear(hidden, 4 * hidden)
        nn.init.zeros_(self.gates.weight)
        nn.init.constant_(self.gates.bias, -2.0)
        self.adapters = nn.ModuleList()
        if adapters:
            for _ in range(3):
                down = nn.Linear(hidden, rank, bias=False)
                up = nn.Linear(rank, hidden, bias=False)
                adapter = nn.Sequential(down, nn.GELU(), up)
                nn.init.normal_(down.weight, std=0.02)
                nn.init.normal_(up.weight, std=0.02)
                self.adapters.append(adapter)

    def forward(self, state: torch.Tensor, condition: torch.Tensor, stage: int) -> torch.Tensor:
        condition = self.condition_norm(condition)
        gates = (0.25 * torch.sigmoid(self.gates(condition.mean(1)))).chunk(4, dim=-1)
        query = self.norms[0](state)
        state = (
            state
            + gates[0][:, None] * self.self_attention(query, query, query, need_weights=False)[0]
        )
        state = (
            state
            + gates[1][:, None]
            * self.cross_attention(self.norms[1](state), condition, condition, need_weights=False)[
                0
            ]
        )
        state = state + gates[2][:, None] * self.ffn(self.norms[2](state))
        if self.adapters:
            state = state + gates[3][:, None] * self.adapters[stage](self.norms[3](state))
        return state


class EventV2World(nn.Module):
    input_mean: torch.Tensor
    input_scale: torch.Tensor
    coordinates: torch.Tensor

    def __init__(self, image_dim: int, hidden: int, layers: int, rank: int, variant: str):
        super().__init__()
        self.image_dim, self.hidden, self.layers = image_dim, hidden, layers
        self.rank, self.variant = rank, variant
        self.clinical = ClinicalTokenizer(hidden)
        self.treatment = nn.Sequential(nn.Linear(82, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.action_identity = nn.Parameter(torch.randn(4, hidden) * 0.02)
        self.event_codes = nn.ModuleList(nn.Embedding(4, hidden) for _ in range(3))
        self.image = nn.Sequential(nn.Linear(image_dim, hidden), nn.LayerNorm(hidden))
        self.position = nn.Linear(3, hidden, bias=False)
        self.baseline_self_norm = nn.LayerNorm(hidden)
        self.baseline_self_attention = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.baseline_norm = nn.LayerNorm(hidden)
        self.baseline_attention = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.baseline_ffn = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 4 * hidden),
            nn.GELU(),
            nn.Linear(4 * hidden, hidden),
        )
        self.blocks = nn.ModuleList(
            StageTransition(hidden, rank, variant != "no_stage_adapter") for _ in range(layers)
        )
        self.decoder = self._decoder(hidden, image_dim)
        self.source_decoder = self._decoder(hidden, image_dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        last = self.decoder[-1]
        assert isinstance(last, nn.Linear)
        nn.init.normal_(last.weight, std=0.001)
        nn.init.zeros_(last.bias)
        axes = [torch.linspace(-1, 1, 3)] * 3
        self.register_buffer(
            "coordinates", torch.stack(torch.meshgrid(*axes, indexing="ij")).flatten(1).T[None]
        )
        self.register_buffer("input_mean", torch.zeros(image_dim))
        self.register_buffer("input_scale", torch.ones(image_dim))

    @staticmethod
    def _decoder(hidden: int, image_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 2 * hidden),
            nn.GELU(),
            nn.Linear(2 * hidden, image_dim),
        )

    def validate(self, inputs: EventInputs) -> None:
        if inputs.x.ndim != 2 or not len(inputs.x):
            raise ValueError("Require a nonempty batch of clinical and treatment vectors")
        inputs.validate(self.image_dim)
        if not inputs.x.is_floating_point() or not inputs.ct0.is_floating_point():
            raise ValueError("Clinical vectors and CT0 tokens must be floating point")
        if not (inputs.x.device == inputs.ct0.device == inputs.events.device):
            raise ValueError("All event inputs must share a device")

    @torch.no_grad()
    def fit_statistics(self, ct0: torch.Tensor, ct1: torch.Tensor, ct1_valid: torch.Tensor) -> None:
        if ct0.ndim != 3 or ct0.shape[1:] != (27, self.image_dim) or not len(ct0):
            raise ValueError("Training CT0 must have shape [N,27,image_dim]")
        if ct1.shape != ct0.shape or ct1_valid.shape != (len(ct0),):
            raise ValueError("Training CT1 and validity must match CT0")
        if ct1_valid.dtype != torch.bool:
            raise ValueError("CT1 validity must be boolean")
        if not torch.isfinite(ct0).all() or not torch.isfinite(ct1[ct1_valid]).all():
            raise ValueError("Observed training CT features must be finite")
        self.input_mean.copy_(ct0.mean((0, 1)))
        self.input_scale.copy_(ct0.std((0, 1), unbiased=False).clamp_min(0.05))
        last = self.decoder[-1]
        assert isinstance(last, nn.Linear)
        last.bias.zero_()
        if ct1_valid.any():
            last.bias.copy_(
                (ct1[ct1_valid].mean((0, 1)) - ct0[ct1_valid].mean((0, 1))) / self.input_scale
            )

    def encode_baseline(
        self, inputs: EventInputs, clinical: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        normalized = (inputs.ct0 - self.input_mean) / self.input_scale
        # Keep small gated and low-rank updates in float32 under BF16 autocast.
        state = self.image(normalized).float()
        if mask is not None:
            state = torch.where(mask[..., None], self.mask_token, state)
        state = state + self.position(self.coordinates)
        query = self.baseline_self_norm(state)
        state = state + self.baseline_self_attention(query, query, query, need_weights=False)[0]
        state = (
            state
            + self.baseline_attention(
                self.baseline_norm(state), clinical, clinical, need_weights=False
            )[0]
        )
        return state + self.baseline_ffn(state)

    def forward(
        self, inputs: EventInputs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.validate(inputs)
        clinical = self.clinical(inputs.x)
        state = self.encode_baseline(inputs, clinical)
        states = [state]
        last_stage = torch.zeros(len(state), dtype=torch.long, device=state.device)
        for stage in range(3):
            active = inputs.events[:, stage] == PRESENT
            if active.any() and self.variant != "no_transition":
                conditions = [clinical, self.event_codes[stage](inputs.events[:, stage])[:, None]]
                if stage == 0:
                    actions = self.treatment(inputs.x[:, 32:].reshape(-1, 4, 82))
                    conditions.append(actions + self.action_identity[None])
                condition = torch.cat(conditions, 1)
                candidate = state
                for block in self.blocks:
                    candidate = block(candidate, condition, stage)
                state = torch.where(active[:, None, None], candidate, state)
            last_stage = torch.where(active, stage + 1, last_stage)
            states.append(state)
        trajectory = torch.stack(states, 1)
        ct1 = inputs.ct0 + self.decoder(trajectory[:, 1]) * self.input_scale
        return trajectory, ct1.float(), last_stage, clinical


class BatchEnsembleLinear(nn.Module):
    """Shared dense weights with a rank-one modulation per ensemble member."""

    def __init__(self, in_features: int, out_features: int, members: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.input_scale = nn.Parameter(torch.empty(members, in_features))
        self.output_scale = nn.Parameter(torch.empty(members, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.normal_(self.input_scale, mean=1.0, std=0.1)
        nn.init.normal_(self.output_scale, mean=1.0, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x * self.input_scale[None]) * self.output_scale[None] + self.bias


def mean_probability_logit(member_logits: torch.Tensor) -> torch.Tensor:
    """Aggregate probabilities stably without converting saturated sigmoid values."""
    logits = member_logits.float()
    log_positive = torch.logsumexp(F.logsigmoid(logits), dim=-1) - math.log(logits.shape[-1])
    log_negative = torch.logsumexp(F.logsigmoid(-logits), dim=-1) - math.log(logits.shape[-1])
    return log_positive - log_negative


class TrajectoryHead(nn.Module):
    def __init__(self, hidden: int, members: int):
        super().__init__()
        self.members = members
        self.query = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.condition = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
        self.type_identity = nn.Parameter(torch.randn(3, hidden) * 0.02)
        self.token_norm = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.readout = nn.Sequential(
            BatchEnsembleLinear(3 * hidden, hidden, members),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            BatchEnsembleLinear(hidden, 64, members),
            nn.GELU(),
            BatchEnsembleLinear(64, 1, members),
        )

    def forward(
        self, baseline: torch.Tensor, current: torch.Tensor, clinical: torch.Tensor
    ) -> torch.Tensor:
        change = current - baseline
        tokens = torch.cat(
            [value + self.type_identity[k] for k, value in enumerate((baseline, current, change))],
            dim=1,
        )
        tokens = self.token_norm(tokens)
        query = self.query + self.condition(clinical.mean(1))[:, None]
        attended = query + self.attention(query, tokens, tokens, need_weights=False)[0]
        features = torch.cat((attended[:, 0], current.mean(1), change.mean(1)), dim=-1)
        return self.readout(features[:, None].expand(-1, self.members, -1)).squeeze(-1).float()


class EventV2Model(nn.Module):
    def __init__(
        self,
        image_dim: int = 768,
        hidden: int = 128,
        layers: int = 3,
        rank: int = 8,
        members: int = 4,
        variant: str = "full",
    ):
        super().__init__()
        if image_dim < 1 or hidden < 4 or hidden % 4 or layers < 1 or rank < 1 or members < 1:
            raise ValueError("Positive dimensions required; hidden must be divisible by four")
        if variant not in VARIANTS:
            raise ValueError(f"Unknown architecture variant: {variant}")
        self.members = members
        self.world = EventV2World(image_dim, hidden, layers, rank, variant)
        self.pcr_head = TrajectoryHead(hidden, members)
        self.recurrence_head = TrajectoryHead(hidden, members)

    def forward(self, inputs: EventInputs) -> EventV2Output:
        states, ct1, last_stage, clinical = self.world(inputs)
        pcr = self.pcr_head(states[:, 0], states[:, 2], clinical)
        recurrence = self.recurrence_head(states[:, 0], states[:, 3], clinical)
        return EventV2Output(
            states=states,
            ct1=ct1,
            pcr_logit=mean_probability_logit(pcr),
            recurrence_logit=mean_probability_logit(recurrence),
            last_stage=last_stage,
            incomplete_history=((inputs.events == UNKNOWN) | (inputs.events == CONFLICT)).any(1),
            member_logits=torch.stack((pcr, recurrence), dim=1),
            stage_increments=states[:, 1:] - states[:, :-1],
        )

    def masked_source_loss(self, inputs: EventInputs, mask: torch.Tensor) -> torch.Tensor:
        self.world.validate(inputs)
        if mask.shape != (len(inputs.x), 27) or mask.dtype != torch.bool:
            raise ValueError("Source reconstruction mask must be boolean with shape [B,27]")
        if mask.device != inputs.ct0.device:
            raise ValueError("Source reconstruction mask must share the CT0 device")
        clinical = self.world.clinical(inputs.x)
        state = self.world.encode_baseline(inputs, clinical, mask)
        predicted = self.world.source_decoder(state).float()
        if not mask.any():
            return predicted.sum() * 0.0
        target = ((inputs.ct0 - self.world.input_mean) / self.world.input_scale).detach().float()
        return F.smooth_l1_loss(predicted[mask], target[mask])

    def dimensions(self) -> dict:
        return {
            "image_dim": self.world.image_dim,
            "hidden": self.world.hidden,
            "layers": self.world.layers,
            "rank": self.world.rank,
            "members": self.members,
            "variant": self.world.variant,
        }
