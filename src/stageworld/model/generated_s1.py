"""Generated S1 prognosis with direct baseline and declared-treatment memory."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

import torch
from torch import Tensor, nn

from stageworld.data.baseline_clinical import (
    BASELINE_CLINICAL_SCHEMA,
    CT6_CLINICAL_SCHEMA,
    BaselineClinical,
    clinical_fields,
)
from stageworld.encoders.base import ObservationTokens
from stageworld.errors import DataContractError
from stageworld.survival import risk_probability, survival_probability

from .stageworld import StageWorldModel, StageWorldModelConfig
from .types import ActionTokens, BeliefState, PredictionDistribution, StagePrediction

READOUT_MODES = ("history_generated", "history_only", "generated_only")


def clean_action_padding(actions: ActionTokens) -> ActionTokens:
    actions.validate()
    changes: dict[str, Any] = {
        field.name: value.masked_fill(
            ~actions.valid[..., None] if value.ndim == 3 else ~actions.valid, 0
        )
        for field in fields(actions)
        if isinstance(value := getattr(actions, field.name), Tensor) and field.name != "valid"
    }
    return replace(actions, **changes)


@dataclass(frozen=True)
class GeneratedS1Config(StageWorldModelConfig):
    survival_task: str = "s0_s1_pred"
    model_version: str = "stageworld-generated-s1-v1"
    clinical_schema_version: str = BASELINE_CLINICAL_SCHEMA
    readout_mode: str = "history_generated"

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.survival_task not in ("s0_s1_pred", "s1_pred_only")
            or self.readout_mode not in READOUT_MODES
            or self.survival_causes != 1
            or self.clinical_schema_version not in (BASELINE_CLINICAL_SCHEMA, CT6_CLINICAL_SCHEMA)
        ):
            raise ValueError("Unsupported generated-S1 readout, endpoint or clinical schema")


class BaselineClinicalEncoder(nn.Module):
    def __init__(self, hidden_dim: int, schema_version: str = BASELINE_CLINICAL_SCHEMA):
        super().__init__()
        self.schema_version = schema_version
        self.field_specs = clinical_fields(schema_version)
        self.fields = nn.Embedding(len(self.field_specs), hidden_dim)
        self.categories = nn.ModuleList(
            [nn.Embedding(len(field.categories) + 1, hidden_dim) for field in self.field_specs]
        )
        self.numeric = nn.Linear(1, hidden_dim)
        self.missing = nn.Embedding(len(self.field_specs), hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, baseline: BaselineClinical) -> Tensor:
        baseline.validate()
        if baseline.schema_version != self.schema_version:
            raise DataContractError(code="CLINICAL_SCHEMA_MISMATCH", message="Schema differs.")
        tokens = []
        for index, field in enumerate(self.field_specs):
            if field.categories:
                value = self.categories[index](baseline.categories[:, index])
            else:
                value = self.numeric(baseline.values[:, index : index + 1])
            value = torch.where(
                baseline.observed[:, index : index + 1],
                value,
                self.missing.weight[index].expand_as(value),
            )
            tokens.append(value + self.fields.weight[index])
        return self.norm(torch.stack(tokens, dim=1))


@dataclass
class BaselineContext:
    state: BeliefState
    tokens: Tensor
    valid: Tensor


@dataclass
class GeneratedS1Prediction:
    baseline: BaselineContext
    state_s1_pred: BeliefState
    survival_s0: StagePrediction | None
    survival_s1_pred: StagePrediction
    future_ct: PredictionDistribution
    scenario: str
    readout_mode: str
    future_ct_objective: str = "huber_cosine"
    future_ct_variance_supervised: bool = False


class GeneratedS1Model(StageWorldModel):
    config: GeneratedS1Config

    def __init__(self, config: GeneratedS1Config):
        super().__init__(config)
        self.config = config
        self.baseline_clinical_encoder = BaselineClinicalEncoder(
            config.hidden_dim, config.clinical_schema_version
        )
        self.survival_source = nn.Embedding(3, config.hidden_dim)

    def initialize_baseline(
        self,
        ct0: ObservationTokens,
        baseline: BaselineClinical,
        s0_time: Tensor,
        *,
        deterministic: bool,
        generator: torch.Generator | None = None,
    ) -> BaselineContext:
        baseline.validate()
        self._validate_observation(ct0)
        ct0 = replace(
            ct0,
            values=ct0.values.masked_fill(~ct0.valid[..., None], 0),
            acquired_time=ct0.acquired_time.masked_fill(~ct0.valid, 0),
            available_time=ct0.available_time.masked_fill(~ct0.valid, 0),
            coords=None if ct0.coords is None else ct0.coords.masked_fill(~ct0.valid[..., None], 0),
        )
        self._validate_time("s0_time", s0_time, batch=ct0.batch_size, device=ct0.values.device)
        if (
            ct0.modality_name != "ct"
            or baseline.values.shape[0] != ct0.batch_size
            or baseline.values.device != ct0.values.device
            or not ct0.valid.any(1).all()
        ):
            raise DataContractError(
                code="BASELINE_INPUT_MISMATCH", message="Bind baseline CT data."
            )
        if (ct0.valid & self._time_precedes(s0_time[:, None], ct0.available_time)).any():
            raise DataContractError(code="BASELINE_CT_UNAVAILABLE", message="CT0 is not available.")
        ct_tokens, ct_valid, provenance = self._combine_observations([ct0], s0_time)
        clinical_tokens = self.baseline_clinical_encoder(baseline)
        tokens = torch.cat((ct_tokens, clinical_tokens), dim=1)
        valid = torch.cat(
            (
                ct_valid,
                torch.ones(clinical_tokens.shape[:2], dtype=torch.bool, device=tokens.device),
            ),
            dim=1,
        )
        state = self._sample_state(
            self.initializer(tokens, valid),
            s0_time,
            posterior=True,
            deterministic=deterministic,
            generator=generator,
            provenance=(*provenance, self.config.clinical_schema_version),
            flags=("static_pre_treatment_clinical_fields",),
        )
        return BaselineContext(state, tokens, valid)

    def _scenario_tokens(self, actions: ActionTokens, reference_time: Tensor) -> Tensor:
        actions = clean_action_padding(actions)
        values = self.action_projection(actions.values.masked_fill(~actions.valid[..., None], 0))
        if actions.event_type is not None:
            values = values + self.action_type_embedding(actions.event_type.long().clamp(0, 15))
        if actions.planned_or_delivered is not None:
            values = values + self.action_status_embedding(
                actions.planned_or_delivered.long().clamp(0, 3)
            )
        if actions.known_exposure is not None:
            values = values + self.exposure_projection(actions.known_exposure.to(values)[..., None])
        times = actions.event_time.masked_fill(~actions.valid, 0)
        return values + self.action_time(
            self._relative_time_in_days(times, reference_time[:, None])
        )

    def read_survival(
        self,
        context: BaselineContext,
        state: BeliefState,
        actions: ActionTokens,
        horizons: Tensor,
        *,
        observed: bool = False,
    ) -> StagePrediction:
        self._validate_state_compatibility(state)
        if (
            horizons.ndim != 1
            or horizons.numel() == 0
            or not torch.isfinite(horizons).all()
            or (horizons < 0).any()
            or (horizons[1:] < horizons[:-1]).any()
        ):
            raise DataContractError(
                code="GENERATED_HORIZONS_INVALID", message="Use ordered horizons."
            )
        blocks: list[Tensor] = []
        masks: list[Tensor] = []
        if self.config.readout_mode != "generated_only":
            blocks.extend(
                (
                    context.tokens + self.survival_source.weight[0],
                    self._scenario_tokens(actions, context.state.query_time)
                    + self.survival_source.weight[1],
                )
            )
            masks.extend((context.valid, actions.valid))
        if self.config.readout_mode != "history_only":
            state_tokens = self._state_tokens(state)
            blocks.append(state_tokens + self.survival_source.weight[2])
            masks.append(
                torch.ones(state_tokens.shape[:2], dtype=torch.bool, device=state_tokens.device)
            )
        rates = self.survival_decoder(
            torch.cat(blocks, dim=1),
            self._timeline_to_days(state.query_time),
            1,
            torch.cat(masks, dim=1),
        ).float()
        single = rates.squeeze(-1)
        flags = (
            *state.quality_flags,
            "s1_observed" if observed else "s1_pred_no_ct1",
            f"readout_{self.config.readout_mode}",
        )
        return StagePrediction(
            stage="S1",
            query_time=state.query_time,
            endpoint="os",
            horizon_grid=horizons,
            rates=rates,
            survival=survival_probability(single, horizons, self.survival_cutpoints),
            risk=risk_probability(single, horizons, self.survival_cutpoints),
            input_manifest=state.provenance,
            model_version=self.config.model_version,
            quality_flags=flags,
            simulated=not observed,
        )

    def predict_generated_s1(
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
    ) -> GeneratedS1Prediction:
        if not scenario.strip():
            raise DataContractError(
                code="SCENARIO_REQUIRED", message="Declare a treatment scenario."
            )
        context = self.initialize_baseline(
            ct0, baseline, s0_time, deterministic=deterministic, generator=generator
        )
        survival_s0 = (
            self.predict_survival(context.state, "os", horizons, stage="S0")
            if self.config.survival_task == "s0_s1_pred"
            else None
        )
        state = self.rollout_prior(
            context.state,
            scenario_actions,
            target_time,
            deterministic=deterministic,
            generator=generator,
        )
        survival_s1 = self.read_survival(context, state, scenario_actions, horizons)
        future = self.predict_future_observation(
            state, "ct", scenario=scenario, target_time=target_time
        )
        return GeneratedS1Prediction(
            context, state, survival_s0, survival_s1, future, scenario, self.config.readout_mode
        )

    def rollout_prior(
        self,
        state: BeliefState,
        actions: ActionTokens,
        target_time: Tensor,
        *,
        deterministic: bool,
        generator: torch.Generator | None = None,
    ) -> BeliefState:
        actions = clean_action_padding(actions)
        self._validate_time(
            "target_time", target_time, batch=state.memory.shape[0], device=state.memory.device
        )
        replay_times = self._action_replay_times(actions)
        # Endpoint summaries need one transition per patient, with no intermediate event.
        if (~actions.valid | self._times_match(replay_times, target_time[:, None])).all():
            return self.predict_prior(
                state, actions, target_time, deterministic=deterministic, generator=generator
            )
        return self._replay_to(
            state, (actions,), target_time, deterministic=deterministic, generator=generator
        )

    def forward(  # type: ignore[override]
        self,
        *,
        ct0: ObservationTokens,
        baseline: BaselineClinical,
        s0_time: Tensor,
        scenario_actions: ActionTokens,
        target_time: Tensor,
        horizons: Tensor,
        scenario: str,
        deterministic: bool = False,
    ) -> GeneratedS1Prediction:
        return self.predict_generated_s1(
            ct0=ct0,
            baseline=baseline,
            s0_time=s0_time,
            scenario_actions=scenario_actions,
            target_time=target_time,
            horizons=horizons,
            scenario=scenario,
            deterministic=deterministic,
        )
