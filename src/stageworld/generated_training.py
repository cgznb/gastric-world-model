"""Generated-state losses on the existing patient-normalized trainer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any, Protocol

import torch
from torch import Tensor

from stageworld.data.baseline_clinical import BaselineClinical
from stageworld.errors import DataContractError
from stageworld.losses import future_feature_loss
from stageworld.model.generated_s1 import BaselineContext, GeneratedS1Model, GeneratedS1Prediction
from stageworld.model.types import BeliefState, PredictionDistribution
from stageworld.survival import piecewise_exponential_nll
from stageworld.training import (
    LossReport,
    LossWeights,
    StageWorldTrainer,
    TrainingPhase,
    WorldModelBatch,
    _component_weights,
    _masked_state_kl,
    _zero,
)

RETROSPECTIVE_SCENARIO = "retrospective_observed_interval_treatment_condition_not_baseline_fact"


@dataclass(frozen=True, kw_only=True)
class GeneratedTrainingBatch(WorldModelBatch):
    baseline: BaselineClinical

    def __post_init__(self) -> None:
        super().__post_init__()
        self.baseline.validate()
        if self.baseline.values.shape[0] != self.batch_size:
            raise DataContractError(code="BASELINE_BATCH_SIZE", message="Baseline batch differs.")
        if self.future_pathology_valid.any() or self.survival_valid[:, 2].any():
            raise DataContractError(
                code="GENERATED_S2_DISABLED", message="S2 is outside this protocol."
            )

    @classmethod
    def from_world(
        cls, batch: WorldModelBatch, baseline: BaselineClinical
    ) -> GeneratedTrainingBatch:
        return cls(
            **{field.name: getattr(batch, field.name) for field in fields(WorldModelBatch)},
            baseline=baseline,
        )

    def to(self, device: torch.device | str) -> GeneratedTrainingBatch:
        return self.from_world(super().to(device), self.baseline.to(device))

    def prediction_inputs(self, *, deterministic: bool) -> dict[str, Any]:
        return {
            "ct0": self.ct0,
            "baseline": self.baseline,
            "s0_time": self.s0_time,
            "scenario_actions": self.treatment_actions,
            "target_time": self.s1_time,
            "horizons": self.horizons,
            "scenario": RETROSPECTIVE_SCENARIO,
            "deterministic": deterministic,
        }


@dataclass
class GeneratedSupervision:
    future_ct: PredictionDistribution
    pre_update: BeliefState
    post_update: BeliefState
    observed_s1: BeliefState


class GeneratedStateOutput(Protocol):
    baseline: BaselineContext
    state_s1_pred: BeliefState
    future_ct: PredictionDistribution


def observed_supervision(
    model: GeneratedS1Model,
    output: GeneratedStateOutput,
    batch: GeneratedTrainingBatch,
    *,
    deterministic: bool,
) -> GeneratedSupervision:
    actions = batch.treatment_actions
    times = model._action_replay_times(actions)
    before = actions.valid & ~model._time_precedes(batch.ct1_acquisition_time[:, None], times)
    if model._times_match(batch.s1_time, batch.ct1_acquisition_time).all():
        prior = output.state_s1_pred
        future = output.future_ct
    else:
        prior = model.rollout_prior(
            output.baseline.state,
            model._mask_actions(actions, before),
            batch.ct1_acquisition_time,
            deterministic=deterministic,
        )
        future = model.predict_future_observation(
            prior, "ct", scenario=RETROSPECTIVE_SCENARIO, target_time=batch.ct1_acquisition_time
        )
    unavailable = model._optional_row_mask(
        "ct1_unavailable_event_mask",
        batch.ct1_unavailable_event_mask,
        batch=batch.batch_size,
        device=batch.ct0.values.device,
    )
    available = model._resolve_observation_availability(
        "ct1",
        [batch.ct1],
        batch.ct1_availability_time,
        batch.s1_time,
        unavailable,
    )
    model._validate_stage_times(
        "ct1",
        acquisition_time=batch.ct1_acquisition_time,
        availability_time=available,
        query_time=batch.s1_time,
    )
    expected = batch.ct1_acquisition_time[:, None].expand_as(batch.ct1.acquired_time)
    if (batch.ct1.valid & ~model._times_match(expected, batch.ct1.acquired_time)).any():
        raise DataContractError(
            code="CT_TARGET_TIME_MISMATCH", message="CT target acquisition differs."
        )
    if (
        model._times_match(prior.query_time, available).all()
        and model._times_match(available, batch.s1_time).all()
        and not (actions.valid & ~before).any()
    ):
        posterior = model.update_posterior(
            prior, [batch.ct1], available, deterministic=deterministic
        )
        return GeneratedSupervision(future, prior, posterior, posterior)
    pre, post, final = model._stage_update(
        prior,
        [batch.ct1],
        model._mask_actions(actions, actions.valid & ~before),
        batch.ct1_acquisition_time,
        available,
        batch.s1_time,
        unavailable,
        deterministic=deterministic,
        generator=None,
    )
    return GeneratedSupervision(future, pre, post, final)


class GeneratedS1Trainer(StageWorldTrainer):
    def compute_loss(
        self,
        batch: WorldModelBatch,
        *,
        phase: TrainingPhase,
        weights: LossWeights,
        kl_beta: float,
    ) -> LossReport:
        if not isinstance(batch, GeneratedTrainingBatch) or not isinstance(
            self.core_model, GeneratedS1Model
        ):
            raise DataContractError(
                code="GENERATED_TRAINING_CONTRACT", message="Use generated batches."
            )
        output: GeneratedS1Prediction = self.model(**batch.prediction_inputs(deterministic=False))
        s1_only = self.core_model.config.survival_task == "s1_pred_only"
        if s1_only and batch.survival_valid[:, (0, 2)].any():
            raise DataContractError(code="S1_ONLY_LABELS", message="Only S1 labels may be active.")
        supervision = observed_supervision(self.core_model, output, batch, deterministic=False)
        if supervision.future_ct.mean.shape != batch.future_ct_target.shape:
            raise DataContractError(
                code="GENERATED_TARGET_SHAPE", message="CT target shape differs."
            )
        zero = _zero(output.state_s1_pred.memory)
        ct_rows = batch.future_ct_valid.any(dim=1)
        components: dict[str, Tensor] = {
            "future_ct": future_feature_loss(
                supervision.future_ct, batch.future_ct_target, batch.future_ct_valid
            )
            if ct_rows.any()
            else zero,
            "kl_ct": _masked_state_kl(supervision.post_update, supervision.pre_update, ct_rows),
            "future_pathology": zero,
            "kl_pathology": zero,
            "survival_s2": zero,
        }
        for stage, prediction in enumerate((output.survival_s0, output.survival_s1_pred)):
            if prediction is None:
                continue
            valid = batch.survival_valid[:, stage]
            components[f"survival_s{stage}"] = (
                piecewise_exponential_nll(
                    prediction.rates.float().squeeze(-1),
                    batch.survival_durations[:, stage].float(),
                    batch.survival_events[:, stage],
                    self.core_model.survival_cutpoints.float(),
                    valid_mask=valid,
                    zero_time_policy="allow",
                )
                if valid.any()
                else zero
            )
        effective = self._effective_totals((batch,), phase)
        if s1_only:
            components = {k: components[k] for k in ("future_ct", "kl_ct", "survival_s1")}
        factors = _component_weights(phase, weights, kl_beta, effective)
        total = sum(
            (
                components[name] * factors[name]
                for name in components
                if effective[name] > 0 and factors[name] > 0
            ),
            zero,
        )
        return LossReport(
            total,
            components,
            effective,
            {
                "s1_uses_generated_state": int(
                    self.core_model.config.readout_mode != "history_only"
                ),
                "s1_reads_ct1": 0,
                "readout_has_history": int(self.core_model.config.readout_mode != "generated_only"),
                "readout_has_generated": int(self.core_model.config.readout_mode != "history_only"),
            },
        )


class S1OnlyTrainer(GeneratedS1Trainer):
    @staticmethod
    def _effective_totals(
        batches: Sequence[WorldModelBatch], phase: TrainingPhase
    ) -> dict[str, int]:
        if any(b.survival_valid[:, (0, 2)].any() for b in batches):
            raise DataContractError(code="S1_ONLY_LABELS", message="Only S1 labels may be active.")
        totals = StageWorldTrainer._effective_totals(batches, phase)
        return {name: totals[name] for name in ("future_ct", "kl_ct", "survival_s1")}
