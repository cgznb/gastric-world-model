"""Generated-S1 binary pCR and recorded recurrence/metastasis endpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from typing import Any

import torch
from torch import Tensor, nn

from stageworld.config import StageWorldConfig
from stageworld.data.baseline_clinical import CT6_CLINICAL_SCHEMA, BaselineClinical
from stageworld.encoders.base import ObservationTokens
from stageworld.errors import DataContractError
from stageworld.generated_training import (
    GeneratedTrainingBatch,
    observed_supervision,
)
from stageworld.losses import future_feature_loss
from stageworld.model.components import ContinuousTimeEncoder
from stageworld.model.generated_s1 import BaselineContext, GeneratedS1Config, GeneratedS1Model
from stageworld.model.types import ActionTokens, BeliefState, PredictionDistribution
from stageworld.synthetic_workflow import build_model
from stageworld.training import (
    LossReport,
    LossWeights,
    StageWorldTrainer,
    TrainingPhase,
    WorldModelBatch,
    _masked_state_kl,
    _zero,
)

ENDPOINTS = ("pcr", "recurrence")
BINARY_PROTOCOL = "ct6-pcr-recurrence-cv5-v1"
LABEL_CONTRACT: dict[str, Any] = {
    "schema_version": "gastric-recorded-binary-endpoints-v1",
    "endpoint_order": list(ENDPOINTS),
    "pcr": {"column": "BJ", "positive": 1, "negative": 2, "definition": "recorded_pcr"},
    "recurrence": {
        "column": "CG",
        "positive": 1,
        "negative": 0,
        "definition": "recorded_recurrence_metastasis_status",
    },
    "missing_policy": "mask_this_endpoint_only",
    "horizon": None,
    "censoring_likelihood": False,
    "pathology_conflict_policy": "retain_source_label_and_report_conflict",
}


@dataclass(frozen=True)
class BinaryEndpointConfig(GeneratedS1Config):
    model_version: str = "stageworld-generated-s1-pcr-recurrence-v1"
    clinical_schema_version: str = CT6_CLINICAL_SCHEMA
    survival_task: str = "s1_pred_only"
    prediction_task: str = "pcr_recurrence_binary"

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.readout_mode != "history_generated"
            or self.prediction_task != "pcr_recurrence_binary"
        ):
            raise ValueError("Use the main generated binary endpoint model")


@dataclass
class BinaryEndpointPrediction:
    baseline: BaselineContext
    state_s1_pred: BeliefState
    future_ct: PredictionDistribution
    logits: Tensor
    scenario: str

    @property
    def probabilities(self) -> Tensor:
        return self.logits.sigmoid()


class BinaryEndpointDecoder(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(2, dim) * dim**-0.5)
        self.time_encoder = ContinuousTimeEncoder(dim)
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.outputs = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1)) for _ in ENDPOINTS]
        )

    def forward(self, tokens: Tensor, valid: Tensor, target_days: Tensor) -> Tensor:
        query = self.queries[None] + self.time_encoder(target_days)[:, None]
        normalized = self.norm(tokens)
        hidden, _ = self.attention(
            query, normalized, normalized, key_padding_mask=~valid, need_weights=False
        )
        return torch.cat([head(hidden[:, i]) for i, head in enumerate(self.outputs)], -1).float()


class BinaryEndpointModel(GeneratedS1Model):
    config: BinaryEndpointConfig

    def __init__(self, config: BinaryEndpointConfig):
        super().__init__(config)
        self.config = config
        del self.survival_decoder
        self.readout_source = self.survival_source
        del self.survival_source
        self.endpoint_decoder = BinaryEndpointDecoder(
            config.hidden_dim, config.attention_heads, config.dropout
        )

    def read_survival(self, *args: Any, **kwargs: Any) -> Any:
        raise DataContractError(
            code="SURVIVAL_DISABLED", message="This model predicts binary endpoints."
        )

    def predict_survival(self, *args: Any, **kwargs: Any) -> Any:
        raise DataContractError(
            code="SURVIVAL_DISABLED", message="This model predicts binary endpoints."
        )

    def predict_generated_s1(  # type: ignore[override]
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
    ) -> BinaryEndpointPrediction:
        if not scenario.strip():
            raise DataContractError(
                code="SCENARIO_REQUIRED", message="Declare the treatment scenario."
            )
        context = self.initialize_baseline(
            ct0, baseline, s0_time, deterministic=deterministic, generator=generator
        )
        state = self.rollout_prior(
            context.state,
            scenario_actions,
            target_time,
            deterministic=deterministic,
            generator=generator,
        )
        generated = self._state_tokens(state)
        tokens = torch.cat(
            (
                context.tokens + self.readout_source.weight[0],
                self._scenario_tokens(scenario_actions, s0_time) + self.readout_source.weight[1],
                generated + self.readout_source.weight[2],
            ),
            1,
        )
        valid = torch.cat(
            (
                context.valid,
                scenario_actions.valid,
                torch.ones(generated.shape[:2], dtype=torch.bool, device=generated.device),
            ),
            1,
        )
        logits = self.endpoint_decoder(tokens, valid, self._timeline_to_days(target_time))
        future = self.predict_future_observation(
            state, "ct", scenario=scenario, target_time=target_time
        )
        return BinaryEndpointPrediction(context, state, future, logits, scenario)

    def forward(self, **kwargs: Any) -> BinaryEndpointPrediction:  # type: ignore[override]
        return self.predict_generated_s1(**kwargs)


def build_binary_model(config: StageWorldConfig) -> BinaryEndpointModel:
    with torch.random.fork_rng(devices=[]):
        values = asdict(build_model(config).config)
    values["model_version"] = "stageworld-generated-s1-pcr-recurrence-v1"
    return BinaryEndpointModel(BinaryEndpointConfig(**values))


@dataclass(frozen=True, kw_only=True)
class BinaryEndpointBatch(GeneratedTrainingBatch):
    endpoint_labels: Tensor
    endpoint_valid: Tensor

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.survival_valid.any():
            raise DataContractError(
                code="BINARY_SURVIVAL_LABELS", message="Disable survival labels."
            )
        if self.endpoint_labels.shape != (self.batch_size, 2) or self.endpoint_valid.shape != (
            self.batch_size,
            2,
        ):
            raise DataContractError(code="BINARY_LABEL_SHAPE", message="Bind two endpoint columns.")
        if self.endpoint_valid.dtype != torch.bool:
            raise DataContractError(code="BINARY_LABEL_MASK", message="Use boolean validity masks.")
        active = self.endpoint_labels[self.endpoint_valid]
        if not torch.isfinite(active).all() or ((active != 0) & (active != 1)).any():
            raise DataContractError(
                code="BINARY_LABEL_VALUE", message="Active labels must be 0 or 1."
            )

    @classmethod
    def from_generated(
        cls, batch: GeneratedTrainingBatch, labels: Tensor, valid: Tensor
    ) -> BinaryEndpointBatch:
        return cls(
            **{field.name: getattr(batch, field.name) for field in fields(GeneratedTrainingBatch)},
            endpoint_labels=labels,
            endpoint_valid=valid,
        )

    def to(self, device: torch.device | str) -> BinaryEndpointBatch:
        return self.from_generated(
            GeneratedTrainingBatch.from_world(
                WorldModelBatch.to(self, device), self.baseline.to(device)
            ),
            self.endpoint_labels.to(device),
            self.endpoint_valid.to(device),
        )


@dataclass(frozen=True)
class BinaryLossWeights(LossWeights):
    survival: float = 0.0
    pcr: float = 1.0
    recurrence: float = 1.0


class BinaryEndpointTrainer(StageWorldTrainer):
    prediction_contract_key = "endpoint_contract"

    def prediction_contract(self, endpoint: str) -> dict[str, Any]:
        if endpoint != "pcr_recurrence":
            raise DataContractError(
                code="BINARY_CONTRACT", message="Use the binary checkpoint contract."
            )
        return LABEL_CONTRACT

    @staticmethod
    def _effective_totals(
        batches: Sequence[WorldModelBatch], phase: TrainingPhase
    ) -> dict[str, int]:
        totals = {name: 0 for name in ("future_ct", "kl_ct", *ENDPOINTS)}
        for batch in batches:
            if not isinstance(batch, BinaryEndpointBatch):
                raise DataContractError(code="BINARY_BATCH", message="Bind endpoint batches.")
            totals["future_ct"] += int(batch.future_ct_valid.any(1).sum())
            totals["kl_ct"] += int(batch.future_ct_valid.any(1).sum())
            if phase is TrainingPhase.JOINT_ENDPOINTS:
                for i, name in enumerate(ENDPOINTS):
                    totals[name] += int(batch.endpoint_valid[:, i].sum())
            elif phase is not TrainingPhase.WORLD_PRETRAIN or batch.endpoint_valid.any():
                raise DataContractError(
                    code="BINARY_PHASE", message="World batches must be unlabelled."
                )
        return totals

    @staticmethod
    def component_factors(
        phase: TrainingPhase, weights: LossWeights, kl_beta: float, totals: Mapping[str, int]
    ) -> dict[str, float]:
        if not isinstance(weights, BinaryLossWeights):
            raise DataContractError(code="BINARY_WEIGHTS", message="Bind binary task weights.")
        if not 0 <= kl_beta <= 1:
            raise ValueError("KL beta must be inside [0, 1]")
        joint = phase is TrainingPhase.JOINT_ENDPOINTS
        return {
            "future_ct": weights.future_ct,
            "kl_ct": weights.kl * kl_beta,
            "pcr": weights.pcr if joint else 0,
            "recurrence": weights.recurrence if joint else 0,
        }

    def compute_loss(
        self,
        batch: WorldModelBatch,
        *,
        phase: TrainingPhase,
        weights: LossWeights,
        kl_beta: float,
    ) -> LossReport:
        if not isinstance(batch, BinaryEndpointBatch) or not isinstance(
            self.core_model, BinaryEndpointModel
        ):
            raise DataContractError(
                code="BINARY_TRAINING", message="Bind binary model and batches."
            )
        totals = self._effective_totals((batch,), phase)
        output: BinaryEndpointPrediction = self.model(
            **batch.prediction_inputs(deterministic=False)
        )
        supervised = observed_supervision(self.core_model, output, batch, deterministic=False)
        zero = _zero(output.state_s1_pred.memory)
        components = {
            "future_ct": future_feature_loss(
                supervised.future_ct, batch.future_ct_target, batch.future_ct_valid
            )
            if totals["future_ct"]
            else zero,
            "kl_ct": _masked_state_kl(
                supervised.post_update, supervised.pre_update, batch.future_ct_valid.any(1)
            ),
        }
        for i, name in enumerate(ENDPOINTS):
            valid = batch.endpoint_valid[:, i]
            components[name] = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    output.logits[valid, i], batch.endpoint_labels[valid, i].float()
                )
                if totals[name]
                else zero
            )
        factors = self.component_factors(phase, weights, kl_beta, totals)
        total = sum((components[k] * factors[k] for k in totals if totals[k] and factors[k]), zero)
        return LossReport(
            total, components, totals, {"ct1_input_to_prediction": 0, "survival_enabled": 0}
        )
