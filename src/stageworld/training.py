"""Bounded, restartable training for the StageWorld token world model.

The training loop deliberately owns no raw-data loading.  It accepts tensor
batches that have already passed the temporal firewall and keeps future
targets separate from model inputs.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from stageworld.artifacts import new_artifact_id
from stageworld.encoders.base import ObservationTokens
from stageworld.errors import ArtifactError, ConfigurationError, DataContractError, ResourceError
from stageworld.losses import (
    FrozenModuleGuard,
    diagonal_gaussian_kl,
    diagonal_gaussian_kl_per_dimension,
    future_feature_loss,
    latent_diagnostics,
)
from stageworld.model import ActionTokens, StageWorldModel
from stageworld.survival import cause_specific_nll, piecewise_exponential_nll

CHECKPOINT_SCHEMA_VERSION = "stageworld-checkpoint-v4"


class TrainingPhase(StrEnum):
    WORLD_PRETRAIN = "world_pretrain"
    JOINT_SURVIVAL = "joint_survival"
    JOINT_ENDPOINTS = "joint_endpoints"


@dataclass(frozen=True)
class LossWeights:
    survival: float = 1.0
    future_ct: float = 1.0
    future_pathology: float = 1.0
    kl: float = 0.01

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise ConfigurationError(
                code="INVALID_LOSS_WEIGHT",
                message="All loss weights must be finite and nonnegative.",
                details={"weights": values},
            )


DEFAULT_LOSS_WEIGHTS = LossWeights()


def _move_observation(value: ObservationTokens, device: torch.device) -> ObservationTokens:
    return ObservationTokens(
        values=value.values.to(device),
        valid=value.valid.to(device),
        modality=value.modality.to(device),
        acquired_time=value.acquired_time.to(device),
        available_time=value.available_time.to(device),
        provenance=value.provenance,
        source_id=value.source_id,
        modality_name=value.modality_name,
        coords=None if value.coords is None else value.coords.to(device),
        coordinate_system=value.coordinate_system,
        quality_flags=value.quality_flags,
    )


def _move_actions(value: ActionTokens, device: torch.device) -> ActionTokens:
    return ActionTokens(
        values=value.values.to(device),
        valid=value.valid.to(device),
        event_time=value.event_time.to(device),
        available_time=value.available_time.to(device),
        event_type=None if value.event_type is None else value.event_type.to(device),
        planned_or_delivered=(
            None if value.planned_or_delivered is None else value.planned_or_delivered.to(device)
        ),
        known_exposure=(None if value.known_exposure is None else value.known_exposure.to(device)),
        provenance=value.provenance,
    )


@dataclass(frozen=True)
class WorldModelBatch:
    """One patient batch with firewall-approved inputs and detached targets."""

    patient_ids: tuple[str, ...]
    ct0: ObservationTokens
    clinical0: ObservationTokens
    s0_time: Tensor
    treatment_actions: ActionTokens
    ct1_acquisition_time: Tensor
    ct1: ObservationTokens
    s1_time: Tensor
    surgery_actions: ActionTokens
    pathology_acquisition_time: Tensor
    pathology: ObservationTokens
    s2_time: Tensor
    horizons: Tensor
    future_ct_target: Tensor
    future_ct_valid: Tensor
    future_pathology_target: Tensor
    future_pathology_valid: Tensor
    survival_durations: Tensor
    survival_events: Tensor
    survival_valid: Tensor
    ct1_availability_time: Tensor | None = None
    ct1_unavailable_event_mask: Tensor | None = None
    clinical1: ObservationTokens | None = None
    s1_update_actions: ActionTokens | None = None
    pathology_availability_time: Tensor | None = None
    pathology_unavailable_event_mask: Tensor | None = None
    clinical2: ObservationTokens | None = None
    s2_update_actions: ActionTokens | None = None

    def __post_init__(self) -> None:
        batch = len(self.patient_ids)
        if batch == 0 or len(set(self.patient_ids)) != batch:
            raise DataContractError(
                code="INVALID_TRAINING_PATIENTS",
                message="A training batch requires unique anonymous patient identifiers.",
            )
        if self.ct0.batch_size != batch or self.clinical0.batch_size != batch:
            raise DataContractError(
                code="TRAINING_BATCH_SIZE_MISMATCH",
                message="Baseline tensors do not match patient_ids.",
            )
        if self.ct1.batch_size != batch or self.pathology.batch_size != batch:
            raise DataContractError(
                code="TRAINING_BATCH_SIZE_MISMATCH",
                message="Future observation tensors do not match patient_ids.",
            )
        for name, observation in (("clinical1", self.clinical1), ("clinical2", self.clinical2)):
            if observation is not None and observation.batch_size != batch:
                raise DataContractError(
                    code="TRAINING_BATCH_SIZE_MISMATCH",
                    message=f"{name} tensors do not match patient_ids.",
                )
        for name, actions in (
            ("s1_update_actions", self.s1_update_actions),
            ("s2_update_actions", self.s2_update_actions),
        ):
            if actions is not None and actions.values.shape[0] != batch:
                raise DataContractError(
                    code="TRAINING_BATCH_SIZE_MISMATCH",
                    message=f"{name} tensors do not match patient_ids.",
                )
        if self.future_ct_target.requires_grad or self.future_pathology_target.requires_grad:
            raise DataContractError(
                code="TARGET_REQUIRES_GRAD",
                message="Future encoder targets must be detached from online optimization.",
            )
        for name, target, valid in (
            ("future_ct", self.future_ct_target, self.future_ct_valid),
            ("future_pathology", self.future_pathology_target, self.future_pathology_valid),
        ):
            if target.ndim != 3 or target.shape[0] != batch:
                raise DataContractError(
                    code="INVALID_FUTURE_TARGET",
                    message=f"{name} target must have shape [B,K,D].",
                )
            if valid.shape != target.shape[:2] or valid.dtype is not torch.bool:
                raise DataContractError(
                    code="INVALID_FUTURE_TARGET_MASK",
                    message=f"{name} valid mask must be boolean with shape [B,K].",
                )
        if self.survival_durations.shape != (batch, 3):
            raise DataContractError(
                code="INVALID_SURVIVAL_LABEL_SHAPE",
                message="survival_durations must have shape [B,3] for S0/S1/S2.",
            )
        if self.survival_events.shape != (batch, 3):
            raise DataContractError(
                code="INVALID_SURVIVAL_LABEL_SHAPE",
                message="survival_events must have shape [B,3].",
            )
        if self.survival_valid.shape != (batch, 3) or self.survival_valid.dtype is not torch.bool:
            raise DataContractError(
                code="INVALID_SURVIVAL_LABEL_MASK",
                message="survival_valid must be boolean with shape [B,3].",
            )
        for name in (
            "s0_time",
            "ct1_acquisition_time",
            "s1_time",
            "pathology_acquisition_time",
            "s2_time",
        ):
            if getattr(self, name).shape != (batch,):
                raise DataContractError(
                    code="INVALID_TRAINING_TIME_SHAPE",
                    message=f"{name} must have shape [B].",
                )
        for name in ("ct1_availability_time", "pathology_availability_time"):
            value = getattr(self, name)
            if value is not None and value.shape != (batch,):
                raise DataContractError(
                    code="INVALID_TRAINING_TIME_SHAPE",
                    message=f"{name} must have shape [B].",
                )
        for name in (
            "ct1_unavailable_event_mask",
            "pathology_unavailable_event_mask",
        ):
            value = getattr(self, name)
            if value is not None and (value.shape != (batch,) or value.dtype is not torch.bool):
                raise DataContractError(
                    code="INVALID_UNAVAILABLE_EVENT_MASK",
                    message=f"{name} must be a boolean tensor with shape [B].",
                )
        if self.horizons.ndim != 1:
            raise DataContractError(
                code="INVALID_HORIZONS",
                message="horizons must be a rank-one tensor.",
            )

    @property
    def batch_size(self) -> int:
        return len(self.patient_ids)

    def to(self, device: torch.device | str) -> WorldModelBatch:
        destination = torch.device(device)
        return WorldModelBatch(
            patient_ids=self.patient_ids,
            ct0=_move_observation(self.ct0, destination),
            clinical0=_move_observation(self.clinical0, destination),
            s0_time=self.s0_time.to(destination),
            treatment_actions=_move_actions(self.treatment_actions, destination),
            ct1_acquisition_time=self.ct1_acquisition_time.to(destination),
            ct1=_move_observation(self.ct1, destination),
            s1_time=self.s1_time.to(destination),
            surgery_actions=_move_actions(self.surgery_actions, destination),
            pathology_acquisition_time=self.pathology_acquisition_time.to(destination),
            pathology=_move_observation(self.pathology, destination),
            s2_time=self.s2_time.to(destination),
            horizons=self.horizons.to(destination),
            future_ct_target=self.future_ct_target.to(destination),
            future_ct_valid=self.future_ct_valid.to(destination),
            future_pathology_target=self.future_pathology_target.to(destination),
            future_pathology_valid=self.future_pathology_valid.to(destination),
            survival_durations=self.survival_durations.to(destination),
            survival_events=self.survival_events.to(destination),
            survival_valid=self.survival_valid.to(destination),
            ct1_availability_time=(
                None
                if self.ct1_availability_time is None
                else self.ct1_availability_time.to(destination)
            ),
            ct1_unavailable_event_mask=(
                None
                if self.ct1_unavailable_event_mask is None
                else self.ct1_unavailable_event_mask.to(destination)
            ),
            clinical1=(
                None if self.clinical1 is None else _move_observation(self.clinical1, destination)
            ),
            s1_update_actions=(
                None
                if self.s1_update_actions is None
                else _move_actions(self.s1_update_actions, destination)
            ),
            pathology_availability_time=(
                None
                if self.pathology_availability_time is None
                else self.pathology_availability_time.to(destination)
            ),
            pathology_unavailable_event_mask=(
                None
                if self.pathology_unavailable_event_mask is None
                else self.pathology_unavailable_event_mask.to(destination)
            ),
            clinical2=(
                None if self.clinical2 is None else _move_observation(self.clinical2, destination)
            ),
            s2_update_actions=(
                None
                if self.s2_update_actions is None
                else _move_actions(self.s2_update_actions, destination)
            ),
        )


@dataclass
class LossReport:
    total: Tensor
    components: dict[str, Tensor]
    effective_n: dict[str, int]
    diagnostics: dict[str, float | int]

    def detached(self) -> dict[str, Any]:
        return {
            "total": float(self.total.detach().float().cpu()),
            "components": {
                name: float(value.detach().float().cpu()) for name, value in self.components.items()
            },
            "effective_n": dict(self.effective_n),
            "diagnostics": dict(self.diagnostics),
        }


def _zero(reference: Tensor) -> Tensor:
    return reference.sum() * 0.0


def _masked_state_kl(posterior: Any, prior: Any, valid_rows: Tensor) -> Tensor:
    if posterior.stochastic_mean is None or prior.stochastic_mean is None:
        return _zero(posterior.memory)
    if not valid_rows.any():
        return _zero(posterior.memory)
    per_patient = diagonal_gaussian_kl(
        posterior.stochastic_mean,
        posterior.stochastic_log_std,
        prior.stochastic_mean,
        prior.stochastic_log_std,
        reduction="none",
    )
    return per_patient[valid_rows].mean()


def _component_weights(
    phase: TrainingPhase,
    weights: LossWeights,
    kl_beta: float,
    effective_n: Mapping[str, int],
) -> dict[str, float]:
    if not math.isfinite(kl_beta) or kl_beta < 0:
        raise ConfigurationError(
            code="INVALID_KL_BETA", message="KL beta must be finite and nonnegative."
        )
    active_survival = sum(
        effective_n.get(name, 0) > 0 for name in ("survival_s0", "survival_s1", "survival_s2")
    )
    stage_weight = weights.survival / max(active_survival, 1)
    result = {
        "future_ct": weights.future_ct,
        "future_pathology": weights.future_pathology,
        "kl_ct": weights.kl * kl_beta,
        "kl_pathology": weights.kl * kl_beta,
        "survival_s0": stage_weight if phase is TrainingPhase.JOINT_SURVIVAL else 0.0,
        "survival_s1": stage_weight if phase is TrainingPhase.JOINT_SURVIVAL else 0.0,
        "survival_s2": stage_weight if phase is TrainingPhase.JOINT_SURVIVAL else 0.0,
    }
    return result


def _core_stageworld_model(
    model: StageWorldModel | DistributedDataParallel,
) -> StageWorldModel:
    if isinstance(model, DistributedDataParallel):
        if not isinstance(model.module, StageWorldModel):
            raise ConfigurationError(
                code="INVALID_DDP_MODEL",
                message="DistributedDataParallel must directly wrap StageWorldModel.",
            )
        return model.module
    return model


def compute_world_model_loss(
    model: StageWorldModel | DistributedDataParallel,
    batch: WorldModelBatch,
    *,
    phase: TrainingPhase,
    weights: LossWeights = DEFAULT_LOSS_WEIGHTS,
    kl_beta: float = 1.0,
) -> LossReport:
    """Compute losses without allowing targets into the forward trajectory."""

    core_model = _core_stageworld_model(model)
    output = model(
        ct0=batch.ct0,
        clinical0=batch.clinical0,
        s0_time=batch.s0_time,
        treatment_actions=batch.treatment_actions,
        ct1_acquisition_time=batch.ct1_acquisition_time,
        ct1=batch.ct1,
        s1_time=batch.s1_time,
        surgery_actions=batch.surgery_actions,
        pathology_acquisition_time=batch.pathology_acquisition_time,
        pathology=batch.pathology,
        s2_time=batch.s2_time,
        horizons=batch.horizons,
        ct1_availability_time=batch.ct1_availability_time,
        ct1_unavailable_event_mask=batch.ct1_unavailable_event_mask,
        clinical1=batch.clinical1,
        s1_update_actions=batch.s1_update_actions,
        pathology_availability_time=batch.pathology_availability_time,
        pathology_unavailable_event_mask=batch.pathology_unavailable_event_mask,
        clinical2=batch.clinical2,
        s2_update_actions=batch.s2_update_actions,
        deterministic=False,
    )
    if output.future_ct.mean.shape != batch.future_ct_target.shape:
        raise DataContractError(
            code="FUTURE_CT_TARGET_SHAPE_MISMATCH",
            message="Configured CT decoder and frozen target shapes differ.",
        )
    if output.future_pathology.mean.shape != batch.future_pathology_target.shape:
        raise DataContractError(
            code="FUTURE_PATHOLOGY_TARGET_SHAPE_MISMATCH",
            message="Configured pathology decoder and frozen target shapes differ.",
        )

    ct_rows = batch.future_ct_valid.any(dim=1)
    pathology_rows = batch.future_pathology_valid.any(dim=1)
    components: dict[str, Tensor] = {
        "future_ct": (
            future_feature_loss(output.future_ct, batch.future_ct_target, batch.future_ct_valid)
            if batch.future_ct_valid.any()
            else _zero(output.future_ct.mean)
        ),
        "future_pathology": (
            future_feature_loss(
                output.future_pathology,
                batch.future_pathology_target,
                batch.future_pathology_valid,
            )
            if batch.future_pathology_valid.any()
            else _zero(output.future_pathology.mean)
        ),
        "kl_ct": _masked_state_kl(output.post_ct1_update, output.pre_ct1_update, ct_rows),
        "kl_pathology": _masked_state_kl(
            output.post_pathology_update, output.pre_pathology_update, pathology_rows
        ),
    }
    predictions = (output.survival_s0, output.survival_s1, output.survival_s2)
    for index, prediction in enumerate(predictions):
        name = f"survival_s{index}"
        valid = batch.survival_valid[:, index]
        rates = prediction.rates.float()
        if not valid.any():
            components[name] = _zero(rates)
        elif rates.shape[-1] == 1:
            components[name] = piecewise_exponential_nll(
                rates.squeeze(-1),
                batch.survival_durations[:, index].float(),
                batch.survival_events[:, index],
                core_model.survival_cutpoints.float(),
                valid_mask=valid,
                zero_time_policy="allow",
            )
        else:
            components[name] = cause_specific_nll(
                rates,
                batch.survival_durations[:, index].float(),
                batch.survival_events[:, index],
                core_model.survival_cutpoints.float(),
                valid_mask=valid,
                zero_time_policy="allow",
            )

    effective_n = {
        "future_ct": int(ct_rows.sum().item()),
        "future_pathology": int(pathology_rows.sum().item()),
        "kl_ct": int(ct_rows.sum().item()),
        "kl_pathology": int(pathology_rows.sum().item()),
        **{
            f"survival_s{index}": int(batch.survival_valid[:, index].sum().item())
            for index in range(3)
        },
    }
    factors = _component_weights(phase, weights, kl_beta, effective_n)
    total = sum(
        components[name] * factors[name]
        for name in components
        if effective_n[name] > 0 and factors[name] > 0
    )
    if not isinstance(total, Tensor):
        total = _zero(output.state_s0.memory)

    diagnostics: dict[str, float | int] = {
        "ct_valid_patients": effective_n["future_ct"],
        "pathology_valid_patients": effective_n["future_pathology"],
        "state_norm": float(output.state_s2.memory.detach().float().norm().cpu()),
    }
    if batch.future_ct_valid.any():
        ct_kl_per_dimension = None
        if output.post_ct1_update.stochastic_mean is not None:
            assert output.post_ct1_update.stochastic_log_std is not None
            assert output.pre_ct1_update.stochastic_mean is not None
            assert output.pre_ct1_update.stochastic_log_std is not None
            ct_kl_per_dimension = diagonal_gaussian_kl_per_dimension(
                output.post_ct1_update.stochastic_mean,
                output.post_ct1_update.stochastic_log_std,
                output.pre_ct1_update.stochastic_mean,
                output.pre_ct1_update.stochastic_log_std,
            )
        values = latent_diagnostics(
            output.future_ct.mean.detach(),
            batch.future_ct_target,
            batch.future_ct_valid,
            kl_per_dimension=ct_kl_per_dimension,
            kl_valid=ct_rows if ct_kl_per_dimension is not None else None,
        )
        diagnostics.update(
            ct_target_variance=values.target_variance,
            ct_prediction_variance=values.prediction_variance,
            ct_target_norm=values.target_norm,
            ct_prediction_norm=values.prediction_norm,
            ct_valid_tokens=values.valid_tokens,
            ct_valid_fraction=values.valid_fraction,
            ct_prediction_collapsed=values.prediction_collapsed,
        )
        if values.effective_kl_dimensions is not None:
            diagnostics["ct_effective_kl_dimensions"] = values.effective_kl_dimensions
        if values.mean_kl_per_dimension is not None:
            diagnostics["ct_mean_kl_per_dimension"] = values.mean_kl_per_dimension
    if batch.future_pathology_valid.any():
        pathology_kl_per_dimension = None
        if output.post_pathology_update.stochastic_mean is not None:
            assert output.post_pathology_update.stochastic_log_std is not None
            assert output.pre_pathology_update.stochastic_mean is not None
            assert output.pre_pathology_update.stochastic_log_std is not None
            pathology_kl_per_dimension = diagonal_gaussian_kl_per_dimension(
                output.post_pathology_update.stochastic_mean,
                output.post_pathology_update.stochastic_log_std,
                output.pre_pathology_update.stochastic_mean,
                output.pre_pathology_update.stochastic_log_std,
            )
        values = latent_diagnostics(
            output.future_pathology.mean.detach(),
            batch.future_pathology_target,
            batch.future_pathology_valid,
            kl_per_dimension=pathology_kl_per_dimension,
            kl_valid=pathology_rows if pathology_kl_per_dimension is not None else None,
        )
        diagnostics.update(
            pathology_target_variance=values.target_variance,
            pathology_prediction_variance=values.prediction_variance,
            pathology_target_norm=values.target_norm,
            pathology_prediction_norm=values.prediction_norm,
            pathology_valid_tokens=values.valid_tokens,
            pathology_valid_fraction=values.valid_fraction,
            pathology_prediction_collapsed=values.prediction_collapsed,
        )
        if values.effective_kl_dimensions is not None:
            diagnostics["pathology_effective_kl_dimensions"] = (
                values.effective_kl_dimensions
            )
        if values.mean_kl_per_dimension is not None:
            diagnostics["pathology_mean_kl_per_dimension"] = values.mean_kl_per_dimension
    return LossReport(
        total=total, components=components, effective_n=effective_n, diagnostics=diagnostics
    )


@dataclass(frozen=True)
class CheckpointMetadata:
    checkpoint_id: str
    model_version: str
    endpoint: str
    mode: str
    config_lineage_id: str
    data_lineage_id: str
    cohort_artifact_id: str
    split_version: str
    ct_feature_artifact_id: str
    pathology_feature_artifact_id: str
    timeline_contract_version: str
    outcome_contract_version: str
    training_seed: int
    source_schema_version: str
    cohort_schema_version: str
    feature_schema_version: str
    phase: str
    selection_rule: str
    parent_checkpoint_id: str | None = None
    parent_weight_version: str | None = None
    parent_phase: str | None = None
    parent_config_lineage_id: str | None = None
    parent_data_lineage_id: str | None = None
    parent_cohort_artifact_id: str | None = None
    parent_split_version: str | None = None
    parent_ct_feature_artifact_id: str | None = None
    parent_pathology_feature_artifact_id: str | None = None
    parent_timeline_contract_version: str | None = None
    parent_outcome_contract_version: str | None = None

    def __post_init__(self) -> None:
        required_names = (
            "checkpoint_id",
            "model_version",
            "endpoint",
            "mode",
            "config_lineage_id",
            "data_lineage_id",
            "cohort_artifact_id",
            "split_version",
            "ct_feature_artifact_id",
            "pathology_feature_artifact_id",
            "timeline_contract_version",
            "outcome_contract_version",
            "source_schema_version",
            "cohort_schema_version",
            "feature_schema_version",
            "phase",
            "selection_rule",
        )
        if any(
            not isinstance(getattr(self, name), str) or not getattr(self, name).strip()
            for name in required_names
        ):
            raise ConfigurationError(
                code="INCOMPLETE_CHECKPOINT_METADATA",
                message="Every checkpoint lineage field must be non-empty.",
            )
        if self.endpoint.lower() not in {"os", "pcr_recurrence"}:
            raise ConfigurationError(
                code="ENDPOINT_NOT_CONFIGURED",
                message="Checkpoint endpoint must be OS or pcr_recurrence.",
            )
        if self.mode not in {"synthetic", "real_features", "real_images"}:
            raise ConfigurationError(
                code="INVALID_RUN_MODE", message="Checkpoint mode is unsupported."
            )
        if self.phase not in {item.value for item in TrainingPhase}:
            raise ConfigurationError(
                code="INVALID_TRAINING_PHASE",
                message="Checkpoint phase must match a configured training phase.",
            )
        parent_names = (
            "parent_checkpoint_id",
            "parent_weight_version",
            "parent_phase",
            "parent_config_lineage_id",
            "parent_data_lineage_id",
            "parent_cohort_artifact_id",
            "parent_split_version",
            "parent_ct_feature_artifact_id",
            "parent_pathology_feature_artifact_id",
            "parent_timeline_contract_version",
            "parent_outcome_contract_version",
        )
        parent_values = tuple(getattr(self, name) for name in parent_names)
        has_parent = any(value is not None for value in parent_values)
        if has_parent and any(
            not isinstance(value, str) or not value.strip() for value in parent_values
        ):
            raise ConfigurationError(
                code="INCOMPLETE_PARENT_CHECKPOINT_LINEAGE",
                message="Parent checkpoint lineage must be either absent or fully specified.",
            )
        if (
            self.phase in {TrainingPhase.JOINT_SURVIVAL.value, TrainingPhase.JOINT_ENDPOINTS.value}
            and not has_parent
        ):
            raise ConfigurationError(
                code="PARENT_CHECKPOINT_LINEAGE_REQUIRED",
                message=(
                    "Joint survival checkpoints must identify the exact "
                    "world-pretraining parent."
                ),
            )
        if self.phase == TrainingPhase.WORLD_PRETRAIN.value and has_parent:
            raise ConfigurationError(
                code="UNEXPECTED_PARENT_CHECKPOINT_LINEAGE",
                message="World-pretraining checkpoints cannot declare a transfer parent.",
            )
        if has_parent and self.parent_phase != TrainingPhase.WORLD_PRETRAIN.value:
            raise ConfigurationError(
                code="INVALID_PARENT_CHECKPOINT_PHASE",
                message="The joint checkpoint parent must be a world-pretraining checkpoint.",
            )
        if (
            isinstance(self.training_seed, bool)
            or not isinstance(self.training_seed, int)
            or self.training_seed < 0
        ):
            raise ConfigurationError(
                code="INVALID_TRAINING_SEED",
                message="Checkpoint training_seed must be a nonnegative integer.",
            )


@dataclass
class TrainerState:
    optimizer_step: int = 0
    microbatch_count: int = 0
    epoch: int = 0


def checkpoint_snapshot_path(path: str | Path, weight_version: str) -> Path:
    """Return the append-only snapshot path for an exact serialized weight state."""

    if (
        not isinstance(weight_version, str)
        or not weight_version.strip()
        or Path(weight_version).name != weight_version
        or weight_version in {".", ".."}
    ):
        raise ArtifactError(
            code="INVALID_CHECKPOINT_WEIGHT_VERSION",
            message="Checkpoint weight_version cannot be used as a snapshot filename.",
        )
    target = Path(path)
    return target.parent / "checkpoint_versions" / f"{weight_version}.pt"


def _write_new_checkpoint_snapshot(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ArtifactError(
                code="CHECKPOINT_SNAPSHOT_ALREADY_EXISTS",
                message="An append-only checkpoint snapshot already uses this weight version.",
                details={"weight_version": path.stem},
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _replace_checkpoint_pointer(snapshot: Path, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with snapshot.open("rb") as source, os.fdopen(descriptor, "wb") as destination:
            shutil.copyfileobj(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_payload_mismatches(
    left: object,
    right: object,
    *,
    path: str = "checkpoint",
) -> tuple[str, ...]:
    """Return structural or value differences without persisting a content digest."""

    differences: list[str] = []

    def compare(first: object, second: object, location: str) -> None:
        if isinstance(first, Tensor) or isinstance(second, Tensor):
            if not isinstance(first, Tensor) or not isinstance(second, Tensor):
                differences.append(location)
                return
            if (
                first.shape != second.shape
                or first.dtype != second.dtype
                or first.layout != second.layout
                or not torch.equal(first.detach().cpu(), second.detach().cpu())
            ):
                differences.append(location)
            return
        if isinstance(first, Mapping) or isinstance(second, Mapping):
            if not isinstance(first, Mapping) or not isinstance(second, Mapping):
                differences.append(location)
                return
            first_keys = set(first)
            second_keys = set(second)
            for key in sorted(first_keys | second_keys, key=repr):
                child = f"{location}.{key}"
                if key not in first_keys or key not in second_keys:
                    differences.append(child)
                else:
                    compare(first[key], second[key], child)
            return
        if isinstance(first, (tuple, list)) and isinstance(second, (tuple, list)):
            if type(first) is not type(second) or len(first) != len(second):  # noqa: E721
                differences.append(location)
                return
            for index, (first_item, second_item) in enumerate(
                zip(first, second, strict=True)
            ):
                compare(first_item, second_item, f"{location}[{index}]")
            return
        if isinstance(first, (tuple, list)) or isinstance(second, (tuple, list)):
            differences.append(location)
            return
        if type(first) is not type(second) or first != second:  # noqa: E721
            differences.append(location)

    compare(left, right, path)
    return tuple(differences)


def model_state_mismatches(
    left: Mapping[str, Tensor], right: Mapping[str, Tensor]
) -> tuple[str, ...]:
    return checkpoint_payload_mismatches(left, right, path="model_state")


def _copy_model_state(value: Mapping[str, Tensor]) -> dict[str, Tensor]:
    copied: dict[str, Tensor] = {}
    for name, tensor in value.items():
        if not isinstance(name, str) or not isinstance(tensor, Tensor):
            raise ArtifactError(
                code="PARENT_CHECKPOINT_STATE_INCOMPATIBLE",
                message="Transfer-parent model state must contain named tensors only.",
            )
        copied[name] = tensor.detach().cpu().clone()
    if not copied:
        raise ArtifactError(
            code="PARENT_CHECKPOINT_STATE_INCOMPATIBLE",
            message="Transfer-parent model state cannot be empty.",
        )
    return copied


def _model_state_structure_mismatches(
    reference: Mapping[str, Tensor], candidate: Mapping[str, Tensor]
) -> tuple[str, ...]:
    mismatches: list[str] = []
    for name in sorted(set(reference) | set(candidate)):
        if name not in reference or name not in candidate:
            mismatches.append(name)
            continue
        expected = reference[name]
        actual = candidate[name]
        if (
            not isinstance(actual, Tensor)
            or actual.shape != expected.shape
            or actual.dtype != expected.dtype
            or actual.layout != expected.layout
        ):
            mismatches.append(name)
    return tuple(mismatches)


def _numpy_rng_state() -> dict[str, Any]:
    state = cast(
        tuple[str, np.ndarray[Any, Any], int, int, float],
        np.random.get_state(),
    )
    return {
        "algorithm": state[0],
        "keys": torch.as_tensor(state[1].astype(np.int64)),
        "position": int(state[2]),
        "has_gaussian": int(state[3]),
        "cached_gaussian": float(state[4]),
    }


def _restore_numpy_rng_state(value: Mapping[str, Any]) -> None:
    keys = torch.as_tensor(value["keys"], dtype=torch.int64).cpu().numpy().astype(np.uint32)
    np.random.set_state(
        (
            str(value["algorithm"]),
            keys,
            int(value["position"]),
            int(value["has_gaussian"]),
            float(value["cached_gaussian"]),
        )
    )


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": _numpy_rng_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(value: Mapping[str, Any]) -> None:
    random.setstate(tuple(value["python"]))
    _restore_numpy_rng_state(value["numpy"])
    torch.set_rng_state(torch.as_tensor(value["torch_cpu"], dtype=torch.uint8).cpu())
    if torch.cuda.is_available() and value.get("torch_cuda"):
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(item, dtype=torch.uint8).cpu() for item in value["torch_cuda"]]
        )


def optimizer_parameter_report(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, int]:
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    frozen = {id(parameter) for parameter in model.parameters() if not parameter.requires_grad}
    if optimized - trainable:
        raise ConfigurationError(
            code="OPTIMIZER_CONTAINS_FROZEN_PARAMETER",
            message="Optimizer groups contain frozen or foreign parameters.",
        )
    if trainable - optimized:
        raise ConfigurationError(
            code="TRAINABLE_PARAMETER_OMITTED",
            message="A trainable model parameter is absent from optimizer groups.",
        )
    return {
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "optimized_tensors": len(optimized),
        "frozen_tensors": len(frozen),
    }


class LocalEventLogger:
    """Append aggregate JSON events; raw identifiers and common secret fields are rejected."""

    _forbidden_keys = {
        "patient_id",
        "patient_ids",
        "name",
        "medical_record_number",
        "accession",
        "token",
        "api_key",
        "secret",
    }

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def write(self, event: Mapping[str, Any]) -> None:
        lowered = {str(key).lower() for key in event}
        forbidden = lowered & self._forbidden_keys
        if forbidden:
            raise DataContractError(
                code="SENSITIVE_LOG_FIELD",
                message="Training logs accept aggregate fields only.",
                details={"fields": sorted(forbidden)},
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(event), ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class ExperimentRegistry:
    """A local status registry that retains failed and cancelled runs."""

    fields = (
        "run_id",
        "status",
        "mode",
        "phase",
        "seed",
        "config_lineage_id",
        "data_lineage_id",
        "started_at_unix",
        "finished_at_unix",
        "failure_code",
    )

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def update(self, record: Mapping[str, Any]) -> None:
        if record.get("status") not in {"running", "completed", "failed", "cancelled"}:
            raise ValueError("invalid experiment status")
        rows: list[dict[str, str]] = []
        if self.path.exists():
            with self.path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        normalized = {field: str(record.get(field, "")) for field in self.fields}
        rows = [row for row in rows if row.get("run_id") != normalized["run_id"]]
        rows.append(normalized)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fields)
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
        finally:
            Path(temporary_name).unlink(missing_ok=True)


class StageWorldTrainer:
    """Native PyTorch trainer with exact optimizer-step checkpoint recovery."""

    def __init__(
        self,
        model: StageWorldModel | DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        *,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        device: torch.device | str = "cpu",
        mixed_precision: str = "off",
        grad_clip_norm: float = 1.0,
        survival_time_unit: str = "year",
        survival_parameterization: str = "piecewise_constant_hazard_rate",
        survival_open_tail_interval: bool = True,
        teacher: nn.Module | None = None,
        event_logger: LocalEventLogger | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model: StageWorldModel | DistributedDataParallel
        if isinstance(model, DistributedDataParallel):
            self.model = model
            if not self.model.find_unused_parameters:
                raise ConfigurationError(
                    code="DDP_UNUSED_PARAMETER_DETECTION_REQUIRED",
                    message=(
                        "StageWorld DDP requires find_unused_parameters=True because modality, "
                        "training-phase, and missing-target masks can leave branches inactive."
                    ),
                )
            for parameter in self.model.parameters():
                parameter_device = parameter.device
                if parameter_device.type != self.device.type or (
                    self.device.index is not None and parameter_device.index != self.device.index
                ):
                    raise ConfigurationError(
                        code="DDP_DEVICE_MISMATCH",
                        message=(
                            "Construct DistributedDataParallel on the same device requested by "
                            "StageWorldTrainer; the trainer will not move an initialized wrapper."
                        ),
                    )
        else:
            self.model = model.to(self.device)
        self.core_model = _core_stageworld_model(self.model)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip_norm = float(grad_clip_norm)
        self.event_logger = event_logger
        self.state = TrainerState()
        self.survival_time_unit = survival_time_unit
        self.survival_parameterization = survival_parameterization
        self.survival_open_tail_interval = survival_open_tail_interval
        if not math.isfinite(self.grad_clip_norm) or self.grad_clip_norm <= 0:
            raise ConfigurationError(
                code="INVALID_GRADIENT_CLIP",
                message="grad_clip_norm must be finite and positive.",
            )
        if self.survival_time_unit not in {"day", "year"}:
            raise ConfigurationError(
                code="INVALID_SURVIVAL_TIME_UNIT",
                message="survival_time_unit must be day or year.",
            )
        if self.survival_parameterization != "piecewise_constant_hazard_rate":
            raise ConfigurationError(
                code="INVALID_SURVIVAL_PARAMETERIZATION",
                message="The trainer currently supports piecewise constant hazard rates only.",
            )
        if not isinstance(self.survival_open_tail_interval, bool):
            raise ConfigurationError(
                code="INVALID_SURVIVAL_TAIL_POLICY",
                message="survival_open_tail_interval must be boolean.",
            )
        optimizer_parameter_report(self.model, optimizer)
        if teacher is not None:
            if any(parameter.requires_grad for parameter in teacher.parameters()):
                raise ConfigurationError(
                    code="TEACHER_NOT_FROZEN",
                    message="The future-target teacher must be frozen before training.",
                )
            optimized = {
                id(parameter) for group in optimizer.param_groups for parameter in group["params"]
            }
            if any(id(parameter) in optimized for parameter in teacher.parameters()):
                raise ConfigurationError(
                    code="TEACHER_IN_ONLINE_OPTIMIZER",
                    message="Frozen teacher parameters cannot enter online optimizer groups.",
                )
            self.teacher_guard: FrozenModuleGuard | None = FrozenModuleGuard(teacher)
        else:
            self.teacher_guard = None
        self.teacher = teacher
        self.mixed_precision = self._resolve_precision(mixed_precision)
        scaler_enabled = self.mixed_precision == "fp16"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=scaler_enabled)

    def _resolve_precision(self, value: str) -> str:
        if value not in {"off", "auto", "fp16", "bf16"}:
            raise ConfigurationError(
                code="INVALID_MIXED_PRECISION",
                message="mixed_precision must be off, auto, fp16, or bf16.",
            )
        if value == "auto":
            return "fp16" if self.device.type == "cuda" else "off"
        if value == "fp16" and self.device.type != "cuda":
            raise ConfigurationError(
                code="FP16_REQUIRES_CUDA",
                message="fp16 training is only enabled on CUDA in this trainer.",
            )
        return value

    def _autocast(self) -> Any:
        if self.mixed_precision == "off":
            return nullcontext()
        dtype = torch.float16 if self.mixed_precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    @staticmethod
    def _effective_totals(
        batches: Sequence[WorldModelBatch], phase: TrainingPhase
    ) -> dict[str, int]:
        names = (
            "future_ct",
            "future_pathology",
            "kl_ct",
            "kl_pathology",
            "survival_s0",
            "survival_s1",
            "survival_s2",
        )
        totals = {name: 0 for name in names}
        for batch in batches:
            ct = int(batch.future_ct_valid.any(dim=1).sum().item())
            pathology = int(batch.future_pathology_valid.any(dim=1).sum().item())
            totals["future_ct"] += ct
            totals["kl_ct"] += ct
            totals["future_pathology"] += pathology
            totals["kl_pathology"] += pathology
            if phase is TrainingPhase.JOINT_SURVIVAL:
                for index in range(3):
                    totals[f"survival_s{index}"] += int(batch.survival_valid[:, index].sum().item())
        return totals

    def _distributed_microbatch_world_size(self, local_count: int) -> int:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            if local_count == 0:
                raise DataContractError(
                    code="EMPTY_GRADIENT_ACCUMULATION",
                    message="At least one microbatch is required for an optimizer step.",
                )
            return 1
        world_size = torch.distributed.get_world_size()
        value = torch.tensor(local_count, dtype=torch.int64, device=self.device)
        gathered = [torch.zeros_like(value) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, value)
        counts = tuple(int(item.item()) for item in gathered)
        if not counts or min(counts) == 0:
            raise DataContractError(
                code="EMPTY_GRADIENT_ACCUMULATION",
                message="Every distributed rank requires at least one microbatch.",
                details={"microbatches_by_rank": counts},
            )
        if len(set(counts)) != 1:
            raise DataContractError(
                code="DISTRIBUTED_MICROBATCH_COUNT_MISMATCH",
                message="Distributed ranks must execute the same number of backward passes.",
                details={"microbatches_by_rank": counts},
            )
        return world_size

    def _global_effective_totals(
        self, local_totals: Mapping[str, int]
    ) -> dict[str, int]:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return dict(local_totals)
        names = tuple(local_totals)
        counts = torch.tensor(
            [local_totals[name] for name in names],
            dtype=torch.int64,
            device=self.device,
        )
        torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
        return {
            name: int(count)
            for name, count in zip(names, counts.detach().cpu().tolist(), strict=True)
        }

    def _global_component_means(
        self,
        local_sums: Mapping[str, float],
        global_totals: Mapping[str, int],
    ) -> dict[str, float]:
        names = tuple(local_sums)
        sums = torch.tensor(
            [local_sums[name] for name in names],
            dtype=torch.float64,
            device=self.device,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(sums, op=torch.distributed.ReduceOp.SUM)
        reduced = sums.detach().cpu().tolist()
        return {
            name: (float(total) / global_totals[name] if global_totals[name] > 0 else 0.0)
            for name, total in zip(names, reduced, strict=True)
        }

    def optimizer_step(
        self,
        microbatches: Sequence[WorldModelBatch],
        *,
        phase: TrainingPhase,
        weights: LossWeights = DEFAULT_LOSS_WEIGHTS,
        kl_beta: float = 1.0,
    ) -> dict[str, Any]:
        try:
            return self._optimizer_step(
                microbatches,
                phase=phase,
                weights=weights,
                kl_beta=kl_beta,
            )
        except torch.OutOfMemoryError as error:
            self.optimizer.zero_grad(set_to_none=True)
            raise ResourceError(
                code="CUDA_OUT_OF_MEMORY",
                message="The training step exceeded available accelerator memory.",
                remediation=(
                    "Reduce the patient microbatch or token budget, enable an approved mixed-"
                    "precision mode, close unrelated GPU workloads, then resume from the last "
                    "complete checkpoint. StageWorld does not retry with changed settings."
                ),
                details={"device": self.device.type, "phase": phase.value},
            ) from error

    def compute_loss(
        self,
        batch: WorldModelBatch,
        *,
        phase: TrainingPhase,
        weights: LossWeights,
        kl_beta: float,
    ) -> LossReport:
        """Allow declared protocols to reuse optimization and checkpoint machinery."""
        return compute_world_model_loss(
            self.model, batch, phase=phase, weights=weights, kl_beta=kl_beta
        )

    @staticmethod
    def component_factors(
        phase: TrainingPhase, weights: LossWeights, kl_beta: float, totals: Mapping[str, int]
    ) -> dict[str, float]:
        return _component_weights(phase, weights, kl_beta, totals)

    prediction_contract_key = "survival_contract"

    def prediction_contract(self, endpoint: str) -> dict[str, Any]:
        return {
            "schema_version": "stageworld-survival-contract-v1",
            "endpoint": endpoint.lower(),
            "parameterization": self.survival_parameterization,
            "time_unit": self.survival_time_unit,
            "cutpoints": tuple(self.core_model.config.survival_cutpoints),
            "open_tail_interval": self.survival_open_tail_interval,
            "num_causes": self.core_model.config.survival_causes,
        }

    def _optimizer_step(
        self,
        microbatches: Sequence[WorldModelBatch],
        *,
        phase: TrainingPhase,
        weights: LossWeights = DEFAULT_LOSS_WEIGHTS,
        kl_beta: float = 1.0,
    ) -> dict[str, Any]:
        world_size = self._distributed_microbatch_world_size(len(microbatches))
        local_totals = self._effective_totals(microbatches, phase)
        global_totals = self._global_effective_totals(local_totals)
        factors = self.component_factors(phase, weights, kl_beta, global_totals)
        if not any(
            global_totals[name] > 0 and factors[name] > 0 for name in global_totals
        ):
            raise DataContractError(
                code="NO_EFFECTIVE_TRAINING_TARGETS",
                message="The optimizer step contains no active, valid supervision.",
            )
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        component_sums: dict[str, float] = {name: 0.0 for name in global_totals}
        diagnostics: dict[str, float | int] = {}
        for batch in microbatches:
            on_device = batch.to(self.device)
            with self._autocast():
                report = self.compute_loss(
                    on_device,
                    phase=phase,
                    weights=weights,
                    kl_beta=kl_beta,
                )
                step_loss = _zero(report.total)
                for name, component in report.components.items():
                    global_count = global_totals[name]
                    if global_count == 0 or factors[name] == 0:
                        continue
                    local_count = report.effective_n[name]
                    coefficient = factors[name] * world_size * local_count / global_count
                    step_loss = step_loss + component * coefficient
                    component_sums[name] += (
                        float(component.detach().float().cpu()) * local_count
                    )
            self.scaler.scale(step_loss).backward()
            diagnostics.update(report.diagnostics)
            self.state.microbatch_count += 1
        self.scaler.unscale_(self.optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip_norm, error_if_nonfinite=True
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
        self.state.optimizer_step += 1
        if self.teacher_guard is not None and self.teacher is not None:
            self.teacher_guard.assert_unchanged(self.teacher)
        aggregate = self._global_component_means(component_sums, global_totals)
        weighted_total = sum(aggregate[name] * factors[name] for name in aggregate)
        result: dict[str, Any] = {
            "optimizer_step": self.state.optimizer_step,
            "phase": phase.value,
            "loss": weighted_total,
            "components": aggregate,
            "effective_n": global_totals,
            "gradient_norm": float(torch.as_tensor(gradient_norm).detach().float().cpu()),
            "kl_beta": float(kl_beta),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "diagnostics": diagnostics,
        }
        if self.event_logger is not None:
            self.event_logger.write(result)
        return result

    def evaluate(
        self,
        batches: Iterable[WorldModelBatch],
        *,
        phase: TrainingPhase,
        weights: LossWeights = DEFAULT_LOSS_WEIGHTS,
        kl_beta: float = 1.0,
    ) -> list[dict[str, Any]]:
        guard = FrozenModuleGuard(self.model)
        self.model.eval()
        reports: list[dict[str, Any]] = []
        with torch.inference_mode():
            for batch in batches:
                with self._autocast():
                    report = self.compute_loss(
                        batch.to(self.device),
                        phase=phase,
                        weights=weights,
                        kl_beta=kl_beta,
                    )
                reports.append(report.detached())
        guard.assert_unchanged(self.model)
        return reports

    def save_checkpoint(
        self,
        path: str | Path,
        metadata: CheckpointMetadata,
        *,
        sampler_state: Mapping[str, Any] | None = None,
        transfer_parent_model_state: Mapping[str, Tensor] | None = None,
    ) -> Path:
        if metadata.model_version != self.core_model.config.model_version:
            raise ArtifactError(
                code="CHECKPOINT_MODEL_VERSION_MISMATCH",
                message="Checkpoint metadata and model version differ.",
            )
        if metadata.phase in {
            TrainingPhase.JOINT_SURVIVAL.value, TrainingPhase.JOINT_ENDPOINTS.value
        }:
            if transfer_parent_model_state is None:
                raise ArtifactError(
                    code="PARENT_CHECKPOINT_STATE_REQUIRED",
                    message=(
                        "Joint checkpoints must retain the transferred parent model state "
                        "for exact resume validation."
                    ),
                )
            parent_state = _copy_model_state(transfer_parent_model_state)
            changed = _model_state_structure_mismatches(
                self.core_model.state_dict(), parent_state
            )
            if changed:
                raise ArtifactError(
                    code="PARENT_CHECKPOINT_STATE_INCOMPATIBLE",
                    message="Transfer-parent model state does not match the joint architecture.",
                    details={"tensors": list(changed)},
                )
        else:
            if transfer_parent_model_state is not None:
                raise ArtifactError(
                    code="UNEXPECTED_PARENT_CHECKPOINT_STATE",
                    message="World-pretraining checkpoints cannot embed a transfer-parent state.",
                )
            parent_state = None
        weight_version = new_artifact_id("weights")
        payload: dict[str, Any] = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "weight_version": weight_version,
            "metadata": asdict(metadata),
            "model_config": asdict(self.core_model.config),
            self.prediction_contract_key: self.prediction_contract(metadata.endpoint),
            "model_state": self.core_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": None if self.scheduler is None else self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "rng_state": capture_rng_state(),
            "trainer_state": asdict(self.state),
            "sampler_state": dict(sampler_state or {}),
            "transfer_parent_model_state": parent_state,
        }
        target = Path(path)
        snapshot = checkpoint_snapshot_path(target, weight_version)
        _write_new_checkpoint_snapshot(snapshot, payload)
        _replace_checkpoint_pointer(snapshot, target)
        return target

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        expected: CheckpointMetadata,
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        try:
            payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, EOFError) as error:
            raise ArtifactError(
                code="CHECKPOINT_UNREADABLE",
                message="Checkpoint is absent, truncated, or uses unsupported objects.",
                remediation="Resume from a complete StageWorld checkpoint.",
            ) from error
        if not isinstance(payload, Mapping):
            raise ArtifactError(
                code="CHECKPOINT_INVALID", message="Checkpoint root must be a mapping."
            )
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ArtifactError(
                code="CHECKPOINT_SCHEMA_MISMATCH",
                message="Checkpoint schema is absent or unsupported.",
            )
        weight_version = payload.get("weight_version")
        if not isinstance(weight_version, str) or not weight_version.strip():
            raise ArtifactError(
                code="CHECKPOINT_WEIGHT_VERSION_MISSING",
                message="Checkpoint has no immutable trained-weight version identifier.",
            )
        actual_metadata = payload.get("metadata")
        if not isinstance(actual_metadata, Mapping):
            raise ArtifactError(
                code="CHECKPOINT_METADATA_MISSING",
                message="Checkpoint lineage metadata is missing.",
            )
        expected_fields = asdict(expected)
        mismatched = {
            key: {"expected": value, "actual": actual_metadata.get(key)}
            for key, value in expected_fields.items()
            if actual_metadata.get(key) != value
        }
        if mismatched:
            raise ArtifactError(
                code="CHECKPOINT_CONTRACT_MISMATCH",
                message="Checkpoint endpoint, mode, version, phase, or lineage differs.",
                details={"fields": mismatched},
            )
        parent_state = payload.get("transfer_parent_model_state")
        if expected.phase in {
            TrainingPhase.JOINT_SURVIVAL.value, TrainingPhase.JOINT_ENDPOINTS.value
        }:
            if not isinstance(parent_state, Mapping) or not parent_state:
                raise ArtifactError(
                    code="PARENT_CHECKPOINT_STATE_REQUIRED",
                    message="Joint checkpoint lacks its transferred parent model state.",
                )
            if not all(
                isinstance(name, str) and isinstance(tensor, Tensor)
                for name, tensor in parent_state.items()
            ):
                raise ArtifactError(
                    code="PARENT_CHECKPOINT_STATE_INCOMPATIBLE",
                    message="Joint checkpoint parent state must contain named tensors only.",
                )
            changed = _model_state_structure_mismatches(
                self.core_model.state_dict(), cast(Mapping[str, Tensor], parent_state)
            )
            if changed:
                raise ArtifactError(
                    code="PARENT_CHECKPOINT_STATE_INCOMPATIBLE",
                    message="Joint checkpoint parent state does not match the model architecture.",
                    details={"tensors": list(changed)},
                )
        elif parent_state is not None:
            raise ArtifactError(
                code="UNEXPECTED_PARENT_CHECKPOINT_STATE",
                message="World-pretraining checkpoint unexpectedly embeds a parent model state.",
            )
        expected_model_config = asdict(self.core_model.config)
        actual_model_config = payload.get("model_config")
        actual_model_mapping = (
            dict(actual_model_config) if isinstance(actual_model_config, Mapping) else {}
        )
        if actual_model_mapping != expected_model_config:
            model_config_changes = sorted(
                key
                for key in set(actual_model_mapping) | set(expected_model_config)
                if actual_model_mapping.get(key) != expected_model_config.get(key)
            )
            raise ArtifactError(
                code="CHECKPOINT_MODEL_CONFIG_MISMATCH",
                message="Checkpoint model configuration differs from the resume target.",
                details={"fields": model_config_changes},
            )
        expected_survival_contract = self.prediction_contract(expected.endpoint)
        actual_survival_contract = payload.get(self.prediction_contract_key)
        actual_survival_mapping = (
            dict(actual_survival_contract)
            if isinstance(actual_survival_contract, Mapping)
            else {}
        )
        if actual_survival_mapping != expected_survival_contract:
            survival_contract_changes = sorted(
                key
                for key in set(actual_survival_mapping) | set(expected_survival_contract)
                if actual_survival_mapping.get(key) != expected_survival_contract.get(key)
            )
            raise ArtifactError(
                code="CHECKPOINT_SURVIVAL_CONTRACT_MISMATCH",
                message="Checkpoint survival contract differs from the resume target.",
                details={"fields": survival_contract_changes},
            )
        try:
            self.core_model.load_state_dict(payload["model_state"], strict=True)
            self.optimizer.load_state_dict(payload["optimizer_state"])
            scheduler_state = payload.get("scheduler_state")
            if self.scheduler is None and scheduler_state is not None:
                raise ArtifactError(
                    code="SCHEDULER_CONTRACT_MISMATCH",
                    message="Checkpoint has a scheduler but the trainer does not.",
                )
            if self.scheduler is not None:
                if scheduler_state is None:
                    raise ArtifactError(
                        code="SCHEDULER_CONTRACT_MISMATCH",
                        message="Trainer expects scheduler state absent from checkpoint.",
                    )
                self.scheduler.load_state_dict(scheduler_state)
            self.scaler.load_state_dict(payload.get("scaler_state", {}))
            self.state = TrainerState(**dict(payload["trainer_state"]))
            if restore_rng:
                restore_rng_state(payload["rng_state"])
        except ArtifactError:
            raise
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise ArtifactError(
                code="CHECKPOINT_STATE_INCOMPATIBLE",
                message="Checkpoint state is incomplete or incompatible with this trainer.",
            ) from error
        return dict(payload.get("sampler_state", {}))


def new_checkpoint_metadata(
    *,
    model: StageWorldModel,
    mode: str,
    config_lineage_id: str,
    data_lineage_id: str,
    cohort_artifact_id: str,
    split_version: str,
    ct_feature_artifact_id: str,
    pathology_feature_artifact_id: str,
    timeline_contract_version: str,
    outcome_contract_version: str,
    training_seed: int,
    source_schema_version: str,
    cohort_schema_version: str,
    feature_schema_version: str,
    phase: TrainingPhase,
    selection_rule: str,
    parent_checkpoint_id: str | None = None,
    parent_weight_version: str | None = None,
    parent_phase: str | None = None,
    parent_config_lineage_id: str | None = None,
    parent_data_lineage_id: str | None = None,
    parent_cohort_artifact_id: str | None = None,
    parent_split_version: str | None = None,
    parent_ct_feature_artifact_id: str | None = None,
    parent_pathology_feature_artifact_id: str | None = None,
    parent_timeline_contract_version: str | None = None,
    parent_outcome_contract_version: str | None = None,
) -> CheckpointMetadata:
    return CheckpointMetadata(
        checkpoint_id=new_artifact_id("checkpoint"),
        model_version=model.config.model_version,
        endpoint="os",
        mode=mode,
        config_lineage_id=config_lineage_id,
        data_lineage_id=data_lineage_id,
        cohort_artifact_id=cohort_artifact_id,
        split_version=split_version,
        ct_feature_artifact_id=ct_feature_artifact_id,
        pathology_feature_artifact_id=pathology_feature_artifact_id,
        timeline_contract_version=timeline_contract_version,
        outcome_contract_version=outcome_contract_version,
        training_seed=training_seed,
        source_schema_version=source_schema_version,
        cohort_schema_version=cohort_schema_version,
        feature_schema_version=feature_schema_version,
        phase=phase.value,
        selection_rule=selection_rule,
        parent_checkpoint_id=parent_checkpoint_id,
        parent_weight_version=parent_weight_version,
        parent_phase=parent_phase,
        parent_config_lineage_id=parent_config_lineage_id,
        parent_data_lineage_id=parent_data_lineage_id,
        parent_cohort_artifact_id=parent_cohort_artifact_id,
        parent_split_version=parent_split_version,
        parent_ct_feature_artifact_id=parent_ct_feature_artifact_id,
        parent_pathology_feature_artifact_id=parent_pathology_feature_artifact_id,
        parent_timeline_contract_version=parent_timeline_contract_version,
        parent_outcome_contract_version=parent_outcome_contract_version,
    )


def bounded_fit(
    trainer: StageWorldTrainer,
    batches: Sequence[WorldModelBatch],
    *,
    phase: TrainingPhase,
    max_steps: int,
    max_minutes: float,
    accumulation_steps: int = 1,
    weights: LossWeights = DEFAULT_LOSS_WEIGHTS,
    kl_warmup_steps: int = 0,
) -> list[dict[str, Any]]:
    """Cycle deterministic batches within an explicit smoke/approved budget."""

    if max_steps <= 0 or max_minutes <= 0 or accumulation_steps <= 0:
        raise ConfigurationError(
            code="INVALID_TRAINING_BUDGET",
            message="max_steps, max_minutes, and accumulation_steps must be positive.",
        )
    if not batches:
        raise DataContractError(
            code="EMPTY_TRAINING_DATA", message="Training requires at least one batch."
        )
    started = time.monotonic()
    history: list[dict[str, Any]] = []
    cursor = 0
    while trainer.state.optimizer_step < max_steps:
        if (time.monotonic() - started) / 60.0 >= max_minutes:
            break
        selected = [batches[(cursor + index) % len(batches)] for index in range(accumulation_steps)]
        cursor = (cursor + accumulation_steps) % len(batches)
        next_step = trainer.state.optimizer_step + 1
        beta = 1.0 if kl_warmup_steps <= 0 else min(1.0, next_step / kl_warmup_steps)
        history.append(trainer.optimizer_step(selected, phase=phase, weights=weights, kl_beta=beta))
    return history


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointMetadata",
    "ExperimentRegistry",
    "LocalEventLogger",
    "LossReport",
    "LossWeights",
    "StageWorldTrainer",
    "TrainerState",
    "TrainingPhase",
    "WorldModelBatch",
    "bounded_fit",
    "capture_rng_state",
    "checkpoint_payload_mismatches",
    "checkpoint_snapshot_path",
    "compute_world_model_loss",
    "model_state_mismatches",
    "new_checkpoint_metadata",
    "optimizer_parameter_report",
    "restore_rng_state",
]
