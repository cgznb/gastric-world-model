"""Strict contracts for temporally safe, feature-only inference."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, cast

import torch
from torch import Tensor

from stageworld.config import RunMode
from stageworld.encoders import EncoderProvenance, ObservationTokens
from stageworld.errors import ArtifactError, DataContractError

CHECKPOINT_SCHEMA_VERSION = "stageworld-checkpoint-v4"
SURVIVAL_CONTRACT_SCHEMA_VERSION = "stageworld-survival-contract-v1"
FEATURE_INPUT_SCHEMA_VERSION = "stageworld-feature-input-v1"
PREDICTION_SCHEMA_VERSION = "stageworld-prediction-v1"


def _required_text(value: object, field_name: str, *, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactError(
            code=code,
            message=f"Checkpoint field '{field_name}' must be a non-empty versioned value.",
            details={"field": field_name},
        )
    return value.strip()


def _required_mapping(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ArtifactError(
            code="MISSING_CHECKPOINT_VERSION",
            message=f"Checkpoint field '{field_name}' must be a non-empty mapping.",
            details={"field": field_name},
        )
    return dict(value)


def _finite_cutpoints(value: object) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        values: tuple[object, ...] = ()
    else:
        try:
            values = tuple(cast(Iterable[object], value))
        except TypeError:
            values = ()
    try:
        cutpoints = tuple(float(cast(Any, item)) for item in values)
    except (TypeError, ValueError):
        cutpoints = ()
    if (
        len(cutpoints) < 2
        or cutpoints[0] != 0.0
        or any(not math.isfinite(item) for item in cutpoints)
        or any(right <= left for left, right in zip(cutpoints, cutpoints[1:], strict=False))
    ):
        raise ArtifactError(
            code="CHECKPOINT_SURVIVAL_CONTRACT_MISMATCH",
            message="Checkpoint survival cutpoints must start at zero and increase strictly.",
        )
    return cutpoints


@dataclass(frozen=True, slots=True)
class CheckpointContract:
    """Inference-critical metadata stored beside a model state dictionary.

    ``checkpoint_id`` identifies the logical training/checkpoint lineage, while
    ``weight_version`` identifies the exact serialized weight state.  The latter
    must therefore be used anywhere cached states or predictions bind to model
    parameters. Neither identifier is a persisted file digest.
    """

    schema_version: str
    checkpoint_id: str
    weight_version: str
    model_version: str
    model_config: Mapping[str, Any]
    endpoint: str
    survival_contract: Mapping[str, Any]
    mode: RunMode
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
    step: int
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
        for name in (
            "schema_version",
            "checkpoint_id",
            "weight_version",
            "model_version",
            "endpoint",
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
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ArtifactError(
                    code="MISSING_CHECKPOINT_VERSION",
                    message="Checkpoint metadata lacks an inference-critical version field.",
                    details={"field": name},
                )
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ArtifactError(
                code="CHECKPOINT_SCHEMA_MISMATCH",
                message="Checkpoint schema is not supported by this inference runtime.",
                details={
                    "expected": CHECKPOINT_SCHEMA_VERSION,
                    "received": self.schema_version,
                },
            )
        if self.phase not in {"world_pretrain", "joint_survival"}:
            raise ArtifactError(
                code="INVALID_CHECKPOINT_PHASE",
                message="Checkpoint phase must be world_pretrain or joint_survival.",
            )
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise ArtifactError(
                code="INVALID_CHECKPOINT_STEP",
                message="Checkpoint step must be a nonnegative integer.",
            )
        if (
            isinstance(self.training_seed, bool)
            or not isinstance(self.training_seed, int)
            or self.training_seed < 0
        ):
            raise ArtifactError(
                code="INVALID_CHECKPOINT_SEED",
                message="Checkpoint training_seed must be a nonnegative integer.",
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
            raise ArtifactError(
                code="INCOMPLETE_PARENT_CHECKPOINT_LINEAGE",
                message="Parent checkpoint lineage must be either absent or fully specified.",
            )
        if self.phase == "joint_survival" and not has_parent:
            raise ArtifactError(
                code="PARENT_CHECKPOINT_LINEAGE_REQUIRED",
                message=(
                    "Joint survival checkpoints must identify the exact "
                    "world-pretraining parent."
                ),
            )
        if self.phase == "world_pretrain" and has_parent:
            raise ArtifactError(
                code="UNEXPECTED_PARENT_CHECKPOINT_LINEAGE",
                message="World-pretraining checkpoints cannot declare a transfer parent.",
            )
        if has_parent and self.parent_phase != "world_pretrain":
            raise ArtifactError(
                code="INVALID_PARENT_CHECKPOINT_PHASE",
                message="The joint checkpoint parent must be a world-pretraining checkpoint.",
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CheckpointContract:
        """Parse metadata without importing or trusting a training implementation."""

        required = (
            "schema_version",
            "checkpoint_id",
            "weight_version",
            "model_version",
            "model_config",
            "endpoint",
            "survival_contract",
            "mode",
            "config_lineage_id",
            "data_lineage_id",
            "cohort_artifact_id",
            "split_version",
            "ct_feature_artifact_id",
            "pathology_feature_artifact_id",
            "timeline_contract_version",
            "outcome_contract_version",
            "training_seed",
            "source_schema_version",
            "cohort_schema_version",
            "feature_schema_version",
            "phase",
            "step",
        )
        missing = [name for name in required if name not in value or value[name] in (None, "")]
        if missing:
            raise ArtifactError(
                code="MISSING_CHECKPOINT_VERSION",
                message="Checkpoint metadata is incomplete; inference is refused.",
                remediation="Use a checkpoint with the complete stageworld-checkpoint-v4 metadata.",
                details={"missing": missing},
            )
        try:
            mode = RunMode(str(value["mode"]))
        except ValueError as exc:
            raise ArtifactError(
                code="INVALID_CHECKPOINT_MODE",
                message="Checkpoint mode is not a recognized StageWorld run mode.",
            ) from exc
        step = value["step"]
        if isinstance(step, bool) or not isinstance(step, int):
            raise ArtifactError(
                code="INVALID_CHECKPOINT_STEP",
                message="Checkpoint step must be a nonnegative integer.",
            )
        training_seed = value["training_seed"]
        if isinstance(training_seed, bool) or not isinstance(training_seed, int):
            raise ArtifactError(
                code="INVALID_CHECKPOINT_SEED",
                message="Checkpoint training_seed must be a nonnegative integer.",
            )
        model_config = _required_mapping(value["model_config"], "model_config")
        survival_contract = _required_mapping(
            value["survival_contract"], "survival_contract"
        )
        survival_required = (
            "schema_version",
            "endpoint",
            "parameterization",
            "time_unit",
            "cutpoints",
            "open_tail_interval",
            "num_causes",
        )
        missing_survival = [
            name
            for name in survival_required
            if name not in survival_contract or survival_contract[name] in (None, "")
        ]
        if missing_survival:
            raise ArtifactError(
                code="CHECKPOINT_SURVIVAL_CONTRACT_MISMATCH",
                message="Checkpoint survival contract is incomplete.",
                details={"missing": missing_survival},
            )
        if survival_contract["schema_version"] != SURVIVAL_CONTRACT_SCHEMA_VERSION:
            raise ArtifactError(
                code="CHECKPOINT_SURVIVAL_CONTRACT_MISMATCH",
                message="Checkpoint survival contract schema is unsupported.",
            )
        endpoint = _required_text(
            value["endpoint"], "endpoint", code="MISSING_CHECKPOINT_VERSION"
        ).lower()
        survival_endpoint = str(survival_contract["endpoint"]).strip().lower()
        if survival_endpoint != endpoint:
            raise ArtifactError(
                code="CHECKPOINT_SURVIVAL_CONTRACT_MISMATCH",
                message="Checkpoint endpoint and survival contract endpoint differ.",
            )
        if survival_contract["parameterization"] != "piecewise_constant_hazard_rate":
            raise ArtifactError(
                code="CHECKPOINT_SURVIVAL_CONTRACT_MISMATCH",
                message="Checkpoint survival parameterization is unsupported.",
            )
        if survival_contract["time_unit"] not in {"day", "year"}:
            raise ArtifactError(
                code="CHECKPOINT_SURVIVAL_CONTRACT_MISMATCH",
                message="Checkpoint survival time unit must be day or year.",
            )
        cutpoints = _finite_cutpoints(survival_contract["cutpoints"])
        open_tail = survival_contract["open_tail_interval"]
        causes = survival_contract["num_causes"]
        if not isinstance(open_tail, bool) or (
            isinstance(causes, bool) or not isinstance(causes, int) or causes <= 0
        ):
            raise ArtifactError(
                code="CHECKPOINT_SURVIVAL_CONTRACT_MISMATCH",
                message="Checkpoint open-tail and cause-count settings are invalid.",
            )
        model_version = _required_text(
            value["model_version"], "model_version", code="MISSING_CHECKPOINT_VERSION"
        )
        if model_config.get("model_version") != model_version:
            raise ArtifactError(
                code="CHECKPOINT_MODEL_CONFIG_MISMATCH",
                message="Checkpoint model version differs from its serialized model config.",
            )
        try:
            model_cutpoints = tuple(float(item) for item in model_config["survival_cutpoints"])
            model_causes = int(model_config["survival_causes"])
        except (KeyError, TypeError, ValueError) as error:
            raise ArtifactError(
                code="CHECKPOINT_MODEL_CONFIG_MISMATCH",
                message="Checkpoint model config lacks its survival output dimensions.",
            ) from error
        if model_cutpoints != cutpoints or model_causes != causes:
            raise ArtifactError(
                code="CHECKPOINT_SURVIVAL_CONTRACT_MISMATCH",
                message="Checkpoint model and survival contracts disagree.",
            )
        survival_contract = {
            **survival_contract,
            "endpoint": survival_endpoint,
            "cutpoints": cutpoints,
            "num_causes": causes,
            "open_tail_interval": open_tail,
        }
        return cls(
            schema_version=_required_text(
                value["schema_version"], "schema_version", code="MISSING_CHECKPOINT_VERSION"
            ),
            checkpoint_id=_required_text(
                value["checkpoint_id"], "checkpoint_id", code="MISSING_CHECKPOINT_VERSION"
            ),
            weight_version=_required_text(
                value["weight_version"], "weight_version", code="MISSING_CHECKPOINT_VERSION"
            ),
            model_version=model_version,
            model_config=MappingProxyType(model_config),
            endpoint=endpoint,
            survival_contract=MappingProxyType(survival_contract),
            mode=mode,
            config_lineage_id=_required_text(
                value["config_lineage_id"],
                "config_lineage_id",
                code="MISSING_CHECKPOINT_VERSION",
            ),
            data_lineage_id=_required_text(
                value["data_lineage_id"], "data_lineage_id", code="MISSING_CHECKPOINT_VERSION"
            ),
            cohort_artifact_id=_required_text(
                value["cohort_artifact_id"],
                "cohort_artifact_id",
                code="MISSING_CHECKPOINT_VERSION",
            ),
            split_version=_required_text(
                value["split_version"], "split_version", code="MISSING_CHECKPOINT_VERSION"
            ),
            ct_feature_artifact_id=_required_text(
                value["ct_feature_artifact_id"],
                "ct_feature_artifact_id",
                code="MISSING_CHECKPOINT_VERSION",
            ),
            pathology_feature_artifact_id=_required_text(
                value["pathology_feature_artifact_id"],
                "pathology_feature_artifact_id",
                code="MISSING_CHECKPOINT_VERSION",
            ),
            timeline_contract_version=_required_text(
                value["timeline_contract_version"],
                "timeline_contract_version",
                code="MISSING_CHECKPOINT_VERSION",
            ),
            outcome_contract_version=_required_text(
                value["outcome_contract_version"],
                "outcome_contract_version",
                code="MISSING_CHECKPOINT_VERSION",
            ),
            training_seed=training_seed,
            source_schema_version=_required_text(
                value["source_schema_version"],
                "source_schema_version",
                code="MISSING_CHECKPOINT_VERSION",
            ),
            cohort_schema_version=_required_text(
                value["cohort_schema_version"],
                "cohort_schema_version",
                code="MISSING_CHECKPOINT_VERSION",
            ),
            feature_schema_version=_required_text(
                value["feature_schema_version"],
                "feature_schema_version",
                code="MISSING_CHECKPOINT_VERSION",
            ),
            phase=_required_text(value["phase"], "phase", code="MISSING_CHECKPOINT_VERSION"),
            step=step,
            parent_checkpoint_id=value.get("parent_checkpoint_id"),
            parent_weight_version=value.get("parent_weight_version"),
            parent_phase=value.get("parent_phase"),
            parent_config_lineage_id=value.get("parent_config_lineage_id"),
            parent_data_lineage_id=value.get("parent_data_lineage_id"),
            parent_cohort_artifact_id=value.get("parent_cohort_artifact_id"),
            parent_split_version=value.get("parent_split_version"),
            parent_ct_feature_artifact_id=value.get("parent_ct_feature_artifact_id"),
            parent_pathology_feature_artifact_id=value.get(
                "parent_pathology_feature_artifact_id"
            ),
            parent_timeline_contract_version=value.get("parent_timeline_contract_version"),
            parent_outcome_contract_version=value.get("parent_outcome_contract_version"),
        )

    @classmethod
    def from_checkpoint_payload(cls, value: Mapping[str, Any]) -> CheckpointContract:
        """Parse the nested trainer artifact without trusting a mutable sidecar file."""

        metadata = value.get("metadata")
        trainer_state = value.get("trainer_state")
        if not isinstance(metadata, Mapping) or not isinstance(trainer_state, Mapping):
            raise ArtifactError(
                code="CHECKPOINT_METADATA_MISSING",
                message="Checkpoint lacks inference-critical metadata or trainer state.",
            )
        return cls.from_mapping(
            {
                "schema_version": value.get("schema_version"),
                "checkpoint_id": metadata.get("checkpoint_id"),
                "weight_version": value.get("weight_version"),
                "model_version": metadata.get("model_version"),
                "model_config": value.get("model_config"),
                "endpoint": metadata.get("endpoint"),
                "survival_contract": value.get("survival_contract"),
                "mode": metadata.get("mode"),
                "config_lineage_id": metadata.get("config_lineage_id"),
                "data_lineage_id": metadata.get("data_lineage_id"),
                "cohort_artifact_id": metadata.get("cohort_artifact_id"),
                "split_version": metadata.get("split_version"),
                "ct_feature_artifact_id": metadata.get("ct_feature_artifact_id"),
                "pathology_feature_artifact_id": metadata.get(
                    "pathology_feature_artifact_id"
                ),
                "timeline_contract_version": metadata.get("timeline_contract_version"),
                "outcome_contract_version": metadata.get("outcome_contract_version"),
                "training_seed": metadata.get("training_seed"),
                "source_schema_version": metadata.get("source_schema_version"),
                "cohort_schema_version": metadata.get("cohort_schema_version"),
                "feature_schema_version": metadata.get("feature_schema_version"),
                "phase": metadata.get("phase"),
                "step": trainer_state.get("optimizer_step"),
                "parent_checkpoint_id": metadata.get("parent_checkpoint_id"),
                "parent_weight_version": metadata.get("parent_weight_version"),
                "parent_phase": metadata.get("parent_phase"),
                "parent_config_lineage_id": metadata.get("parent_config_lineage_id"),
                "parent_data_lineage_id": metadata.get("parent_data_lineage_id"),
                "parent_cohort_artifact_id": metadata.get("parent_cohort_artifact_id"),
                "parent_split_version": metadata.get("parent_split_version"),
                "parent_ct_feature_artifact_id": metadata.get(
                    "parent_ct_feature_artifact_id"
                ),
                "parent_pathology_feature_artifact_id": metadata.get(
                    "parent_pathology_feature_artifact_id"
                ),
                "parent_timeline_contract_version": metadata.get(
                    "parent_timeline_contract_version"
                ),
                "parent_outcome_contract_version": metadata.get(
                    "parent_outcome_contract_version"
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class ModalityFeatureContract:
    modality: str
    provenance: EncoderProvenance
    record_feature_version: str | None = None

    def __post_init__(self) -> None:
        if self.modality not in {"ct", "pathology", "clinical"}:
            raise DataContractError(
                code="INVALID_FEATURE_MODALITY",
                message="Feature contract modality must be ct, pathology, or clinical.",
            )
        if self.record_feature_version is not None and not self.record_feature_version.strip():
            raise DataContractError(
                code="MISSING_FEATURE_VERSION",
                message="Configured record feature versions must be non-empty.",
            )


@dataclass(frozen=True, slots=True)
class FeatureInputContract:
    """The exact feature spaces expected by a checkpoint."""

    schema_version: str
    modalities: tuple[ModalityFeatureContract, ...]
    action_feature_version: str

    def __post_init__(self) -> None:
        if not self.schema_version:
            raise DataContractError(
                code="MISSING_FEATURE_SCHEMA",
                message="Feature input schema_version is required.",
            )
        if self.schema_version != FEATURE_INPUT_SCHEMA_VERSION:
            raise DataContractError(
                code="FEATURE_SCHEMA_MISMATCH",
                message="Feature input contract uses an unsupported schema.",
                details={
                    "expected": FEATURE_INPUT_SCHEMA_VERSION,
                    "received": self.schema_version,
                },
            )
        names = [item.modality for item in self.modalities]
        if not names or len(names) != len(set(names)):
            raise DataContractError(
                code="INVALID_FEATURE_MODALITIES",
                message="Feature modalities must be non-empty and unique.",
            )
        if not self.action_feature_version:
            raise DataContractError(
                code="MISSING_ACTION_FEATURE_VERSION",
                message="Action feature version is required.",
            )

    def for_modality(self, modality: str) -> ModalityFeatureContract:
        for item in self.modalities:
            if item.modality == modality:
                return item
        raise DataContractError(
            code="FEATURE_MODALITY_NOT_CONFIGURED",
            message=f"No feature contract is configured for modality '{modality}'.",
        )


@dataclass(frozen=True, slots=True)
class ActionFeature:
    """One structured treatment-event feature vector supplied from a local manifest."""

    values: Tensor
    feature_version: str
    known_exposure: float = 1.0

    def __post_init__(self) -> None:
        if self.values.ndim != 1 or self.values.numel() == 0:
            raise DataContractError(
                code="INVALID_ACTION_FEATURE",
                message="Action feature values must be a non-empty rank-one tensor.",
            )
        if self.values.requires_grad or not torch.isfinite(self.values).all():
            raise DataContractError(
                code="INVALID_ACTION_FEATURE",
                message="Inference action features must be detached and finite.",
            )
        if not self.feature_version:
            raise DataContractError(
                code="MISSING_ACTION_FEATURE_VERSION",
                message="Action feature version is required.",
            )
        if not math.isfinite(self.known_exposure) or self.known_exposure < 0:
            raise DataContractError(
                code="INVALID_KNOWN_EXPOSURE",
                message="known_exposure must be finite and nonnegative.",
            )


class IdentityStatus(StrEnum):
    ANONYMOUS = "anonymous"


@dataclass(frozen=True, slots=True)
class FeatureManifest:
    """In-memory anonymous feature lookup; future entries are resolved only after filtering."""

    schema_version: str
    manifest_lineage_id: str
    patient_id: str
    mode: RunMode
    observation_features: Mapping[str, ObservationTokens] = field(default_factory=dict)
    clinical_features: Mapping[str, ObservationTokens] = field(default_factory=dict)
    action_features: Mapping[str, ActionFeature] = field(default_factory=dict)
    identity_status: IdentityStatus = IdentityStatus.ANONYMOUS

    def __post_init__(self) -> None:
        if not self.schema_version:
            raise DataContractError(
                code="MISSING_FEATURE_SCHEMA",
                message="Feature manifest schema_version is required.",
            )
        if not self.manifest_lineage_id:
            raise DataContractError(
                code="MISSING_MANIFEST_LINEAGE",
                message="Feature manifest requires an explicit lineage identifier.",
            )
        if not self.patient_id:
            raise DataContractError(
                code="MISSING_ANONYMOUS_PATIENT_ID",
                message="Feature manifest requires an anonymous patient identifier.",
            )
        if self.identity_status is not IdentityStatus.ANONYMOUS:
            raise DataContractError(
                code="ANONYMOUS_ID_REQUIRED",
                message="Inference accepts only explicitly anonymous patient histories.",
            )
        object.__setattr__(
            self, "observation_features", MappingProxyType(dict(self.observation_features))
        )
        object.__setattr__(
            self, "clinical_features", MappingProxyType(dict(self.clinical_features))
        )
        object.__setattr__(self, "action_features", MappingProxyType(dict(self.action_features)))


class ScenarioKind(StrEnum):
    OBSERVED_HISTORY = "observed_history"
    PREDICTED_OBSERVATION = "predicted_observation"
    HYPOTHETICAL = "hypothetical"


@dataclass(frozen=True, slots=True)
class Scenario:
    kind: ScenarioKind
    label: str
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise DataContractError(
                code="MISSING_SCENARIO_LABEL",
                message="Every simulated or observed prediction requires a scenario label.",
            )

    @classmethod
    def observed_history(cls) -> Scenario:
        return cls(kind=ScenarioKind.OBSERVED_HISTORY, label="observed_history")


@dataclass(frozen=True, slots=True)
class InputReference:
    record_kind: str
    record_id: str
    source_type: str
    feature_version: str

    def as_dict(self) -> dict[str, str]:
        return {
            "record_kind": self.record_kind,
            "record_id": self.record_id,
            "source_type": self.source_type,
            "feature_version": self.feature_version,
        }


@dataclass(frozen=True, slots=True)
class PredictionOutput:
    """Serializable batch-one prediction contract for a legal observed prefix."""

    patient_id: str
    population: str
    query_id: str
    stage: str
    query_time_days: float
    endpoint: str
    horizon_days: tuple[float, ...]
    survival: tuple[float, ...]
    risk: tuple[float, ...]
    hazard_rates: tuple[tuple[float, ...], ...]
    cif: tuple[tuple[float, ...], ...] | None
    checkpoint_id: str
    weight_version: str
    model_version: str
    feature_schema_version: str
    feature_manifest_lineage_id: str
    input_manifest: tuple[InputReference, ...]
    scenario: Scenario
    quality_flags: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    uncertainty_summary: Mapping[str, Any] = field(default_factory=dict)
    simulated: bool = False
    schema_version: str = PREDICTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.checkpoint_id.strip() or not self.weight_version.strip():
            raise DataContractError(
                code="MISSING_PREDICTION_WEIGHT_LINEAGE",
                message="Predictions require both checkpoint and exact weight identifiers.",
            )
        count = len(self.horizon_days)
        if count == 0 or len(self.survival) != count or len(self.risk) != count:
            raise DataContractError(
                code="INVALID_PREDICTION_SHAPE",
                message="Horizon, survival, and risk vectors must be non-empty and aligned.",
            )
        if any(not math.isfinite(value) for value in (*self.survival, *self.risk)):
            raise DataContractError(
                code="NONFINITE_PREDICTION",
                message="Prediction probabilities must be finite.",
            )
        if any(value < 0 or value > 1 for value in (*self.survival, *self.risk)):
            raise DataContractError(
                code="INVALID_PREDICTION_PROBABILITY",
                message="Prediction probabilities must lie in [0, 1].",
            )
        if self.simulated != (self.scenario.kind is not ScenarioKind.OBSERVED_HISTORY):
            raise DataContractError(
                code="SCENARIO_SIMULATION_MISMATCH",
                message="Scenario kind and simulated flag disagree.",
            )
        object.__setattr__(
            self,
            "uncertainty_summary",
            MappingProxyType(dict(self.uncertainty_summary)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "patient_id": self.patient_id,
            "population": self.population,
            "query_id": self.query_id,
            "stage": self.stage,
            "query_time_days": self.query_time_days,
            "endpoint": self.endpoint,
            "horizon_days": list(self.horizon_days),
            "survival": list(self.survival),
            "risk": list(self.risk),
            "hazard_rates": [list(row) for row in self.hazard_rates],
            "cif": None if self.cif is None else [list(row) for row in self.cif],
            "checkpoint_id": self.checkpoint_id,
            "weight_version": self.weight_version,
            "model_version": self.model_version,
            "feature_schema_version": self.feature_schema_version,
            "feature_manifest_lineage_id": self.feature_manifest_lineage_id,
            "input_manifest": [item.as_dict() for item in self.input_manifest],
            "scenario": {
                "kind": self.scenario.kind.value,
                "label": self.scenario.label,
                "assumptions": list(self.scenario.assumptions),
            },
            "quality_flags": list(self.quality_flags),
            "assumptions": list(self.assumptions),
            "uncertainty_summary": dict(self.uncertainty_summary),
            "simulated": self.simulated,
        }


@dataclass(frozen=True, slots=True)
class FutureObservationOutput:
    modality: str
    mean: Tensor
    log_std: Tensor
    target_time_days: float
    provenance: str
    scenario: Scenario
    simulated: bool = True

    def __post_init__(self) -> None:
        if self.scenario.kind is ScenarioKind.OBSERVED_HISTORY or not self.simulated:
            raise DataContractError(
                code="FUTURE_SCENARIO_NOT_SIMULATED",
                message=(
                    "A future observation must be explicitly labelled predicted or hypothetical."
                ),
            )
        if self.provenance != "predicted_not_observed":
            raise DataContractError(
                code="FUTURE_PROVENANCE_MISMATCH",
                message="Generated feature predictions cannot be labelled as observed evidence.",
            )
