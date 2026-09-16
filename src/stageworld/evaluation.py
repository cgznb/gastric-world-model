"""Censoring-aware evaluation with explicit fit/evaluation provenance.

The primitives in this module operate on one landmark population at a time.
They deliberately keep estimation of censoring and calibration objects separate
from evaluation so a test or external outcome cannot mutate fitted state.

Event type ``0`` denotes censoring and positive integers denote observed event
causes.  Single-risk metrics reject competing events unless an explicit cause
and cumulative-incidence interpretation are supplied.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .artifacts import atomic_write_json, new_artifact_id, read_json

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
MetricStatus = Literal["ok", "not_estimable"]
PREDICTION_SCHEMA_VERSION = "stageworld.predictions.v3"


_ARTIFACT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _validate_artifact_id(value: str, *, name: str) -> None:
    if not _ARTIFACT_ID_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a nonempty path-safe identifier")


class SplitRole(StrEnum):
    """Role of a cohort in the locked evaluation protocol."""

    TRAIN = "train"
    DEVELOPMENT = "development"
    TEST = "test"
    EXTERNAL = "external"
    TEMPORAL = "temporal"


class RunMode(StrEnum):
    """Artifact namespace; synthetic output must never look like real output."""

    SYNTHETIC = "synthetic"
    REAL = "real"


class ScoreDirection(StrEnum):
    """Ordering semantics for a discrimination score."""

    HIGHER_RISK = "higher_is_worse"
    HIGHER_SURVIVAL = "higher_is_better"


class CensoringRule(StrEnum):
    """Predeclared source used to estimate the censoring distribution."""

    DEVELOPMENT_REFERENCE = "development_reference"
    EXTERNAL_COHORT = "external_cohort"


class CompetingControlDefinition(StrEnum):
    """Control definition for cumulative/dynamic cause-specific AUC."""

    EVENT_FREE = "event_free_at_horizon"
    NOT_CASE = "not_target_cause_by_horizon"


def _float_vector(values: Sequence[float], *, name: str) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _event_vector(values: Sequence[int]) -> IntArray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError("event_types must be one-dimensional")
    if not np.isfinite(raw.astype(np.float64)).all():
        raise ValueError("event_types must be finite")
    result = raw.astype(np.int64)
    if not np.equal(raw, result).all() or np.any(result < 0):
        raise ValueError("event_types must be nonnegative integers")
    return result


def _probability_vector(values: Sequence[float], *, name: str) -> FloatArray:
    result = _float_vector(values, name=name)
    if np.any((result < 0.0) | (result > 1.0)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


@dataclass(frozen=True)
class EvaluationCohort:
    """Endpoint labels for one stage/query population.

    Times are landmark-relative observed event or censoring times.  Patient IDs
    must be unique here; repeated stage rows belong in a patient bootstrap plan,
    not in a pooled clinical metric.
    """

    cohort_id: str
    split_role: SplitRole
    endpoint: str
    patient_ids: tuple[str, ...]
    times: tuple[float, ...]
    event_types: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.cohort_id or not self.endpoint:
            raise ValueError("cohort_id and endpoint must be nonempty")
        if not isinstance(self.split_role, SplitRole):
            object.__setattr__(self, "split_role", SplitRole(self.split_role))
        count = len(self.patient_ids)
        if count == 0:
            raise ValueError("evaluation cohort cannot be empty")
        if len(self.times) != count or len(self.event_types) != count:
            raise ValueError("patient_ids, times and event_types must have equal length")
        if any(not patient_id for patient_id in self.patient_ids):
            raise ValueError("patient IDs must be nonempty")
        if len(set(self.patient_ids)) != count:
            raise ValueError("a landmark evaluation cohort must contain one row per patient")
        times = _float_vector(self.times, name="times")
        if np.any(times <= 0.0):
            raise ValueError("landmark-relative evaluation times must be positive")
        _event_vector(self.event_types)

    @property
    def size(self) -> int:
        return len(self.patient_ids)

    def arrays(self) -> tuple[FloatArray, IntArray]:
        return (
            np.asarray(self.times, dtype=np.float64),
            np.asarray(self.event_types, dtype=np.int64),
        )


@dataclass(frozen=True)
class EvaluationProtocol:
    """Immutable, predeclared choices required for evaluation."""

    protocol_id: str
    endpoint: str
    horizons: tuple[float, ...]
    evaluation_role: SplitRole
    censoring_rule: CensoringRule = CensoringRule.DEVELOPMENT_REFERENCE
    external_censoring_predeclared: bool = False
    primary_cause: int | None = None
    competing_control_definition: CompetingControlDefinition = CompetingControlDefinition.EVENT_FREE

    def __post_init__(self) -> None:
        if not self.protocol_id or not self.endpoint:
            raise ValueError("protocol_id and endpoint must be nonempty")
        if not isinstance(self.evaluation_role, SplitRole):
            object.__setattr__(self, "evaluation_role", SplitRole(self.evaluation_role))
        if not isinstance(self.censoring_rule, CensoringRule):
            object.__setattr__(self, "censoring_rule", CensoringRule(self.censoring_rule))
        if not isinstance(self.competing_control_definition, CompetingControlDefinition):
            object.__setattr__(
                self,
                "competing_control_definition",
                CompetingControlDefinition(self.competing_control_definition),
            )
        horizons = _float_vector(self.horizons, name="protocol horizons")
        if np.any(horizons <= 0.0) or np.any(horizons[1:] <= horizons[:-1]):
            raise ValueError("protocol horizons must be positive and strictly increasing")
        if self.primary_cause is not None and self.primary_cause < 1:
            raise ValueError("primary_cause must be a positive event code")
        if self.censoring_rule is CensoringRule.EXTERNAL_COHORT:
            if self.evaluation_role is not SplitRole.EXTERNAL:
                raise ValueError("external censoring is only valid for an external evaluation")
            if not self.external_censoring_predeclared:
                raise ValueError("external censoring requires explicit protocol predeclaration")

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "endpoint": self.endpoint,
            "horizons": list(self.horizons),
            "evaluation_role": self.evaluation_role.value,
            "censoring_rule": self.censoring_rule.value,
            "external_censoring_predeclared": self.external_censoring_predeclared,
            "primary_cause": self.primary_cause,
            "competing_control_definition": self.competing_control_definition.value,
        }


@dataclass(frozen=True)
class CensoringDistribution:
    """Kaplan-Meier estimate of G(t)=P(C>t) from a locked reference cohort."""

    artifact_id: str
    source_cohort_id: str
    source_role: SplitRole
    event_times: tuple[float, ...]
    survival_after: tuple[float, ...]
    max_followup: float
    n_patients: int

    @classmethod
    def fit(
        cls,
        cohort: EvaluationCohort,
        *,
        artifact_id: str | None = None,
    ) -> CensoringDistribution:
        """Fit only the censoring mechanism; positive event causes are removals."""

        times, event_types = cohort.arrays()
        unique_times = np.unique(times)
        probability = 1.0
        survival_after: list[float] = []
        for time in unique_times:
            at_risk = int(np.count_nonzero(times >= time))
            censored = int(np.count_nonzero((times == time) & (event_types == 0)))
            if at_risk <= 0:
                raise AssertionError("Kaplan-Meier risk set cannot be empty")
            probability *= 1.0 - censored / at_risk
            survival_after.append(probability)
        return cls(
            artifact_id=artifact_id or new_artifact_id("censoring"),
            source_cohort_id=cohort.cohort_id,
            source_role=cohort.split_role,
            event_times=tuple(float(value) for value in unique_times),
            survival_after=tuple(float(value) for value in survival_after),
            max_followup=float(times.max()),
            n_patients=cohort.size,
        )

    def probability(self, times: float | Sequence[float], *, side: str = "right") -> FloatArray:
        """Evaluate G immediately before (``left``) or after (``right``) a time."""

        if side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'")
        query = np.asarray(times, dtype=np.float64)
        if not np.isfinite(query).all() or np.any(query < 0.0):
            raise ValueError("censoring query times must be finite and nonnegative")
        knots = np.asarray(self.event_times, dtype=np.float64)
        values = np.asarray(self.survival_after, dtype=np.float64)
        search_side: Literal["left", "right"] = "left" if side == "left" else "right"
        indices = np.searchsorted(knots, query, side=search_side) - 1
        output = np.ones_like(query, dtype=np.float64)
        selected = indices >= 0
        output[selected] = values[indices[selected]]
        return output

    def support_reason(self, horizon: float) -> str | None:
        if not math.isfinite(horizon) or horizon <= 0.0:
            return "invalid_horizon"
        if horizon > self.max_followup:
            return "outside_censoring_support"
        if float(self.probability(horizon, side="right")) <= 0.0:
            return "zero_censoring_survival"
        return None


def validate_censoring_policy(
    censoring: CensoringDistribution,
    cohort: EvaluationCohort,
    protocol: EvaluationProtocol,
) -> None:
    """Reject accidental outcome-dependent censoring fits on test data."""

    if cohort.endpoint != protocol.endpoint:
        raise ValueError("evaluation cohort endpoint does not match protocol")
    if cohort.split_role is not protocol.evaluation_role:
        raise ValueError("evaluation cohort role does not match protocol")
    if protocol.censoring_rule is CensoringRule.DEVELOPMENT_REFERENCE:
        if censoring.source_role not in {SplitRole.TRAIN, SplitRole.DEVELOPMENT}:
            raise ValueError("reference censoring must be fit on train/development outcomes")
        return
    if censoring.source_role is not SplitRole.EXTERNAL:
        raise ValueError("predeclared external censoring must be fit on the external cohort")
    if censoring.source_cohort_id != cohort.cohort_id:
        raise ValueError("external censoring estimate belongs to a different cohort")


@dataclass(frozen=True)
class PredictionLineage:
    """Version lineage without persisted checksum values."""

    artifact_id: str
    checkpoint_schema_version: str
    checkpoint_id: str
    weight_version: str
    model_artifact_id: str
    model_version: str
    checkpoint_endpoint: str
    config_lineage_id: str
    config_version: str
    data_lineage_id: str
    input_artifact_id: str
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
    checkpoint_phase: str
    checkpoint_step: int
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
    schema_version: str = PREDICTION_SCHEMA_VERSION
    run_mode: RunMode = RunMode.SYNTHETIC

    def __post_init__(self) -> None:
        fields = (
            self.artifact_id,
            self.checkpoint_schema_version,
            self.checkpoint_id,
            self.weight_version,
            self.model_artifact_id,
            self.model_version,
            self.checkpoint_endpoint,
            self.config_lineage_id,
            self.config_version,
            self.data_lineage_id,
            self.input_artifact_id,
            self.cohort_artifact_id,
            self.split_version,
            self.ct_feature_artifact_id,
            self.pathology_feature_artifact_id,
            self.timeline_contract_version,
            self.outcome_contract_version,
            self.source_schema_version,
            self.cohort_schema_version,
            self.feature_schema_version,
            self.checkpoint_phase,
            self.schema_version,
        )
        if any(not isinstance(value, str) or not value.strip() for value in fields):
            raise ValueError("all prediction lineage fields must be nonempty")
        for name, value in (
            ("artifact_id", self.artifact_id),
            ("checkpoint_id", self.checkpoint_id),
            ("weight_version", self.weight_version),
            ("model_artifact_id", self.model_artifact_id),
            ("data_lineage_id", self.data_lineage_id),
            ("input_artifact_id", self.input_artifact_id),
            ("cohort_artifact_id", self.cohort_artifact_id),
            ("split_version", self.split_version),
            ("ct_feature_artifact_id", self.ct_feature_artifact_id),
            ("pathology_feature_artifact_id", self.pathology_feature_artifact_id),
        ):
            _validate_artifact_id(value, name=name)
        if self.schema_version != PREDICTION_SCHEMA_VERSION:
            raise ValueError("prediction lineage schema is unsupported")
        if (
            isinstance(self.training_seed, bool)
            or not isinstance(self.training_seed, int)
            or self.training_seed < 0
        ):
            raise ValueError("prediction lineage training_seed must be a nonnegative integer")
        if (
            isinstance(self.checkpoint_step, bool)
            or not isinstance(self.checkpoint_step, int)
            or self.checkpoint_step < 0
        ):
            raise ValueError("prediction lineage checkpoint_step must be a nonnegative integer")
        aliases = (
            ("model_artifact_id", self.model_artifact_id, self.weight_version),
            ("config_version", self.config_version, self.config_lineage_id),
            ("input_artifact_id", self.input_artifact_id, self.data_lineage_id),
        )
        inconsistent = [name for name, alias, canonical in aliases if alias != canonical]
        if inconsistent:
            raise ValueError(
                "prediction lineage aliases disagree with checkpoint lineage: "
                + ", ".join(inconsistent)
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
        if any(value is not None for value in parent_values) and any(
            not isinstance(value, str) or not value.strip() for value in parent_values
        ):
            raise ValueError("parent prediction lineage must be either absent or fully specified")
        if all(value is not None for value in parent_values):
            for name in (
                "parent_checkpoint_id",
                "parent_weight_version",
                "parent_data_lineage_id",
                "parent_cohort_artifact_id",
                "parent_split_version",
                "parent_ct_feature_artifact_id",
                "parent_pathology_feature_artifact_id",
            ):
                _validate_artifact_id(str(getattr(self, name)), name=name)
        if not isinstance(self.run_mode, RunMode):
            object.__setattr__(self, "run_mode", RunMode(self.run_mode))

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "checkpoint_schema_version": self.checkpoint_schema_version,
            "checkpoint_id": self.checkpoint_id,
            "weight_version": self.weight_version,
            "model_artifact_id": self.model_artifact_id,
            "model_version": self.model_version,
            "checkpoint_endpoint": self.checkpoint_endpoint,
            "config_lineage_id": self.config_lineage_id,
            "config_version": self.config_version,
            "data_lineage_id": self.data_lineage_id,
            "input_artifact_id": self.input_artifact_id,
            "cohort_artifact_id": self.cohort_artifact_id,
            "split_version": self.split_version,
            "ct_feature_artifact_id": self.ct_feature_artifact_id,
            "pathology_feature_artifact_id": self.pathology_feature_artifact_id,
            "timeline_contract_version": self.timeline_contract_version,
            "outcome_contract_version": self.outcome_contract_version,
            "training_seed": self.training_seed,
            "source_schema_version": self.source_schema_version,
            "cohort_schema_version": self.cohort_schema_version,
            "feature_schema_version": self.feature_schema_version,
            "checkpoint_phase": self.checkpoint_phase,
            "checkpoint_step": self.checkpoint_step,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "parent_weight_version": self.parent_weight_version,
            "parent_phase": self.parent_phase,
            "parent_config_lineage_id": self.parent_config_lineage_id,
            "parent_data_lineage_id": self.parent_data_lineage_id,
            "parent_cohort_artifact_id": self.parent_cohort_artifact_id,
            "parent_split_version": self.parent_split_version,
            "parent_ct_feature_artifact_id": self.parent_ct_feature_artifact_id,
            "parent_pathology_feature_artifact_id": self.parent_pathology_feature_artifact_id,
            "parent_timeline_contract_version": self.parent_timeline_contract_version,
            "parent_outcome_contract_version": self.parent_outcome_contract_version,
            "schema_version": self.schema_version,
            "run_mode": self.run_mode.value,
        }


@dataclass(frozen=True)
class PredictionRecord:
    patient_id: str
    stage: str
    query_time: float
    endpoint: str
    horizon: float
    risk: float
    fold: int | None
    seed: int
    model_name: str
    input_artifact_id: str
    state_quality: str = "observed_prefix"

    def __post_init__(self) -> None:
        if not self.patient_id or not self.stage or not self.endpoint or not self.model_name:
            raise ValueError("prediction identity fields must be nonempty")
        if not self.input_artifact_id or not self.state_quality:
            raise ValueError("prediction provenance fields must be nonempty")
        if not math.isfinite(self.query_time) or self.query_time < 0.0:
            raise ValueError("query_time must be finite and nonnegative")
        if not math.isfinite(self.horizon) or self.horizon <= 0.0:
            raise ValueError("horizon must be finite and positive")
        if not math.isfinite(self.risk) or not 0.0 <= self.risk <= 1.0:
            raise ValueError("risk must be a finite probability")

    @property
    def match_key(self) -> tuple[str, str, float, str, float, int | None, int]:
        return (
            self.patient_id,
            self.stage,
            self.query_time,
            self.endpoint,
            self.horizon,
            self.fold,
            self.seed,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "stage": self.stage,
            "query_time": self.query_time,
            "endpoint": self.endpoint,
            "horizon": self.horizon,
            "survival": 1.0 - self.risk,
            "risk": self.risk,
            "fold": self.fold,
            "seed": self.seed,
            "model_name": self.model_name,
            "input_artifact_id": self.input_artifact_id,
            "state_quality": self.state_quality,
        }


@dataclass(frozen=True)
class PredictionArtifact:
    lineage: PredictionLineage
    records: tuple[PredictionRecord, ...]

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("prediction artifact cannot be empty")
        keys = [record.match_key for record in self.records]
        if len(set(keys)) != len(keys):
            raise ValueError("prediction artifact contains duplicate prediction keys")
        if any(
            record.input_artifact_id != self.lineage.input_artifact_id for record in self.records
        ):
            raise ValueError("record input lineage differs from prediction lineage")
        if any(
            record.seed != self.lineage.training_seed
            or record.endpoint.casefold() != self.lineage.checkpoint_endpoint.casefold()
            for record in self.records
        ):
            raise ValueError("record endpoint or seed differs from prediction lineage")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.lineage.schema_version,
            "lineage": self.lineage.as_dict(),
            "records": [record.as_dict() for record in self.records],
        }


def read_prediction_artifact(path: str | Path) -> PredictionArtifact:
    payload = read_json(path)
    if payload.get("schema_version") != PREDICTION_SCHEMA_VERSION:
        raise ValueError("prediction artifact schema is unsupported")
    lineage_payload = payload.get("lineage")
    records_payload = payload.get("records")
    if not isinstance(lineage_payload, dict) or not isinstance(records_payload, list):
        raise ValueError("prediction artifact must contain lineage and records")

    def required_text(name: str) -> str:
        value = lineage_payload.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"prediction lineage field '{name}' must be nonempty text")
        return value

    def required_integer(name: str) -> int:
        value = lineage_payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"prediction lineage field '{name}' must be an integer")
        return value

    lineage = PredictionLineage(
        artifact_id=required_text("artifact_id"),
        checkpoint_schema_version=required_text("checkpoint_schema_version"),
        checkpoint_id=required_text("checkpoint_id"),
        weight_version=required_text("weight_version"),
        model_artifact_id=required_text("model_artifact_id"),
        model_version=required_text("model_version"),
        checkpoint_endpoint=required_text("checkpoint_endpoint"),
        config_lineage_id=required_text("config_lineage_id"),
        config_version=required_text("config_version"),
        data_lineage_id=required_text("data_lineage_id"),
        input_artifact_id=required_text("input_artifact_id"),
        cohort_artifact_id=required_text("cohort_artifact_id"),
        split_version=required_text("split_version"),
        ct_feature_artifact_id=required_text("ct_feature_artifact_id"),
        pathology_feature_artifact_id=required_text("pathology_feature_artifact_id"),
        timeline_contract_version=required_text("timeline_contract_version"),
        outcome_contract_version=required_text("outcome_contract_version"),
        training_seed=required_integer("training_seed"),
        source_schema_version=required_text("source_schema_version"),
        cohort_schema_version=required_text("cohort_schema_version"),
        feature_schema_version=required_text("feature_schema_version"),
        checkpoint_phase=required_text("checkpoint_phase"),
        checkpoint_step=required_integer("checkpoint_step"),
        parent_checkpoint_id=lineage_payload.get("parent_checkpoint_id"),
        parent_weight_version=lineage_payload.get("parent_weight_version"),
        parent_phase=lineage_payload.get("parent_phase"),
        parent_config_lineage_id=lineage_payload.get("parent_config_lineage_id"),
        parent_data_lineage_id=lineage_payload.get("parent_data_lineage_id"),
        parent_cohort_artifact_id=lineage_payload.get("parent_cohort_artifact_id"),
        parent_split_version=lineage_payload.get("parent_split_version"),
        parent_ct_feature_artifact_id=lineage_payload.get("parent_ct_feature_artifact_id"),
        parent_pathology_feature_artifact_id=lineage_payload.get(
            "parent_pathology_feature_artifact_id"
        ),
        parent_timeline_contract_version=lineage_payload.get(
            "parent_timeline_contract_version"
        ),
        parent_outcome_contract_version=lineage_payload.get("parent_outcome_contract_version"),
        schema_version=required_text("schema_version"),
        run_mode=RunMode(required_text("run_mode")),
    )
    records: list[PredictionRecord] = []
    for raw in records_payload:
        if not isinstance(raw, dict):
            raise ValueError("prediction records must be objects")
        records.append(
            PredictionRecord(
                patient_id=str(raw["patient_id"]),
                stage=str(raw["stage"]),
                query_time=float(raw["query_time"]),
                endpoint=str(raw["endpoint"]),
                horizon=float(raw["horizon"]),
                risk=float(raw["risk"]),
                fold=None if raw.get("fold") is None else int(raw["fold"]),
                seed=int(raw["seed"]),
                model_name=str(raw["model_name"]),
                input_artifact_id=str(raw["input_artifact_id"]),
                state_quality=str(raw["state_quality"]),
            )
        )
    return PredictionArtifact(lineage=lineage, records=tuple(records))


@dataclass(frozen=True)
class MetricLineage:
    artifact_id: str
    prediction_artifact_ids: tuple[str, ...]
    model_artifact_ids: tuple[str, ...]
    censoring_artifact_id: str
    protocol_id: str
    evaluator_version: str = "stageworld.evaluation.v1"
    run_mode: RunMode = RunMode.SYNTHETIC

    def __post_init__(self) -> None:
        if not self.artifact_id or not self.censoring_artifact_id or not self.protocol_id:
            raise ValueError("metric lineage IDs must be nonempty")
        if not self.prediction_artifact_ids or not self.model_artifact_ids:
            raise ValueError("metrics must reference predictions and models")
        _validate_artifact_id(self.artifact_id, name="artifact_id")
        _validate_artifact_id(self.censoring_artifact_id, name="censoring_artifact_id")
        for value in (*self.prediction_artifact_ids, *self.model_artifact_ids):
            _validate_artifact_id(value, name="referenced artifact ID")
        if not isinstance(self.run_mode, RunMode):
            object.__setattr__(self, "run_mode", RunMode(self.run_mode))

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "prediction_artifact_ids": list(self.prediction_artifact_ids),
            "model_artifact_ids": list(self.model_artifact_ids),
            "censoring_artifact_id": self.censoring_artifact_id,
            "protocol_id": self.protocol_id,
            "evaluator_version": self.evaluator_version,
            "run_mode": self.run_mode.value,
        }


def build_metric_lineage(
    predictions: Sequence[PredictionArtifact],
    *,
    censoring_artifact_id: str,
    protocol_id: str,
    artifact_id: str | None = None,
) -> MetricLineage:
    if not predictions:
        raise ValueError("at least one prediction artifact is required")
    modes = {artifact.lineage.run_mode for artifact in predictions}
    if len(modes) != 1:
        raise ValueError("synthetic and real prediction artifacts cannot be combined")
    return MetricLineage(
        artifact_id=artifact_id or new_artifact_id("metric"),
        prediction_artifact_ids=tuple(artifact.lineage.artifact_id for artifact in predictions),
        model_artifact_ids=tuple(
            dict.fromkeys(artifact.lineage.model_artifact_id for artifact in predictions)
        ),
        censoring_artifact_id=censoring_artifact_id,
        protocol_id=protocol_id,
        run_mode=next(iter(modes)),
    )


@dataclass(frozen=True)
class MetricResult:
    metric: str
    status: MetricStatus
    estimate: float | None
    reason: str | None
    horizon: float | None
    n_patients: int
    n_events: int
    effective_n: int
    lineage: MetricLineage | None = None
    details: tuple[tuple[str, str | int | float | bool], ...] = ()

    def __post_init__(self) -> None:
        if self.status == "ok":
            if self.estimate is None or not math.isfinite(self.estimate):
                raise ValueError("an estimable metric requires a finite estimate")
            if self.reason is not None:
                raise ValueError("an estimable metric cannot carry a failure reason")
        elif self.status == "not_estimable":
            if self.estimate is not None or not self.reason:
                raise ValueError("not_estimable requires no estimate and an explicit reason")
        else:
            raise ValueError(f"unknown metric status: {self.status}")

    @property
    def estimable(self) -> bool:
        return self.status == "ok"

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "metric": self.metric,
            "status": self.status,
            "estimate": self.estimate,
            "reason": self.reason,
            "horizon": self.horizon,
            "n_patients": self.n_patients,
            "n_events": self.n_events,
            "effective_n": self.effective_n,
            "details": dict(self.details),
        }
        if self.lineage is not None:
            payload["lineage"] = self.lineage.as_dict()
        return payload


def _not_estimable(
    metric: str,
    reason: str,
    cohort: EvaluationCohort,
    horizon: float | None,
    *,
    cause: int | None = None,
    effective_n: int = 0,
    lineage: MetricLineage | None = None,
) -> MetricResult:
    times, events = cohort.arrays()
    event_mask = events > 0 if cause is None else events == cause
    if horizon is not None:
        event_mask &= times <= horizon
    n_events = int(np.count_nonzero(event_mask))
    return MetricResult(
        metric=metric,
        status="not_estimable",
        estimate=None,
        reason=reason,
        horizon=horizon,
        n_patients=cohort.size,
        n_events=n_events,
        effective_n=effective_n,
        lineage=lineage,
    )


def _validate_metric_context(
    cohort: EvaluationCohort,
    scores: Sequence[float],
    horizon: float,
    censoring: CensoringDistribution,
    protocol: EvaluationProtocol | None,
) -> tuple[FloatArray, FloatArray, IntArray, str | None]:
    values = _float_vector(scores, name="scores")
    if values.size != cohort.size:
        raise ValueError("one score is required per patient")
    if protocol is not None:
        validate_censoring_policy(censoring, cohort, protocol)
        if horizon not in protocol.horizons:
            raise ValueError("horizon was not predeclared in the evaluation protocol")
    elif censoring.source_role not in {SplitRole.TRAIN, SplitRole.DEVELOPMENT}:
        raise ValueError(
            "test, temporal or external censoring requires an explicit predeclared protocol"
        )
    times, events = cohort.arrays()
    return values, times, events, censoring.support_reason(horizon)


def _ranking_scores(scores: FloatArray, direction: ScoreDirection) -> FloatArray:
    if not isinstance(direction, ScoreDirection):
        direction = ScoreDirection(direction)
    return scores if direction is ScoreDirection.HIGHER_RISK else -scores


def _target_cause(events: IntArray, cause: int | None) -> int:
    if cause is None:
        if np.any(events > 1):
            raise ValueError(
                "competing-risk outcomes require an explicit cause and CIF predictions"
            )
        return 1
    if cause < 1:
        raise ValueError("cause must be a positive event code")
    return cause


def _protocol_cause(
    events: IntArray,
    cause: int | None,
    protocol: EvaluationProtocol | None,
) -> int:
    selected = cause
    if protocol is not None and protocol.primary_cause is not None:
        if selected is not None and selected != protocol.primary_cause:
            raise ValueError("requested cause differs from the predeclared primary cause")
        selected = protocol.primary_cause
    return _target_cause(events, selected)


def cumulative_dynamic_auc(
    cohort: EvaluationCohort,
    scores: Sequence[float],
    horizon: float,
    censoring: CensoringDistribution,
    *,
    direction: ScoreDirection = ScoreDirection.HIGHER_RISK,
    cause: int | None = None,
    control_definition: CompetingControlDefinition = CompetingControlDefinition.EVENT_FREE,
    protocol: EvaluationProtocol | None = None,
    lineage: MetricLineage | None = None,
) -> MetricResult:
    """IPCW cumulative/dynamic AUC with explicit score direction.

    For competing outcomes ``scores`` must be the target-cause CIF.  EVENT_FREE
    excludes prior competing events; NOT_CASE counts an observed competing event
    as a control and weights it at its observed event time.
    """

    values, times, events, support_reason = _validate_metric_context(
        cohort, scores, horizon, censoring, protocol
    )
    metric = "cumulative_dynamic_auc"
    target = _protocol_cause(events, cause, protocol)
    if support_reason is not None:
        return _not_estimable(
            metric, support_reason, cohort, horizon, cause=target, lineage=lineage
        )
    if not isinstance(control_definition, CompetingControlDefinition):
        control_definition = CompetingControlDefinition(control_definition)
    risk_scores = _ranking_scores(values, direction)
    cases = (events == target) & (times <= horizon)
    if control_definition is CompetingControlDefinition.EVENT_FREE:
        controls = times > horizon
    else:
        controls = (times > horizon) | ((events > 0) & (events != target) & (times <= horizon))
    if not np.any(cases):
        return _not_estimable(
            metric,
            "no_cases_by_horizon",
            cohort,
            horizon,
            cause=target,
            lineage=lineage,
        )
    if not np.any(controls):
        return _not_estimable(
            metric,
            "no_controls_at_horizon",
            cohort,
            horizon,
            cause=target,
            lineage=lineage,
        )

    case_g = censoring.probability(times[cases], side="left")
    if np.any(case_g <= 0.0):
        return _not_estimable(
            metric,
            "zero_case_censoring_survival",
            cohort,
            horizon,
            cause=target,
            lineage=lineage,
        )
    case_weights = 1.0 / case_g
    control_weights = np.full(
        int(np.count_nonzero(controls)), 1.0 / float(censoring.probability(horizon, side="right"))
    )
    if control_definition is CompetingControlDefinition.NOT_CASE:
        control_times = times[controls]
        control_events = events[controls]
        observed_competing = (control_events > 0) & (control_times <= horizon)
        if np.any(observed_competing):
            control_g = censoring.probability(
                control_times[observed_competing].tolist(), side="left"
            )
            if np.any(control_g <= 0.0):
                return _not_estimable(
                    metric,
                    "zero_control_censoring_survival",
                    cohort,
                    horizon,
                    cause=target,
                    lineage=lineage,
                )
            control_weights[observed_competing] = 1.0 / control_g

    comparisons = risk_scores[cases, None] - risk_scores[None, controls]
    credit = (comparisons > 0.0).astype(np.float64) + 0.5 * (comparisons == 0.0)
    pair_weights = case_weights[:, None] * control_weights[None, :]
    denominator = float(pair_weights.sum())
    if denominator <= 0.0:
        return _not_estimable(
            metric,
            "no_weighted_pairs",
            cohort,
            horizon,
            cause=target,
            lineage=lineage,
        )
    estimate = float((credit * pair_weights).sum() / denominator)
    return MetricResult(
        metric=metric,
        status="ok",
        estimate=estimate,
        reason=None,
        horizon=horizon,
        n_patients=cohort.size,
        n_events=int(np.count_nonzero(cases)),
        effective_n=int(np.count_nonzero(cases) + np.count_nonzero(controls)),
        lineage=lineage,
        details=(("score_direction", direction.value), ("target_cause", target)),
    )


def ipcw_concordance_index(
    cohort: EvaluationCohort,
    scores: Sequence[float],
    tau: float,
    censoring: CensoringDistribution,
    *,
    direction: ScoreDirection = ScoreDirection.HIGHER_RISK,
    protocol: EvaluationProtocol | None = None,
    lineage: MetricLineage | None = None,
) -> MetricResult:
    """Uno IPCW concordance truncated at ``tau`` for a single endpoint."""

    values, times, events, support_reason = _validate_metric_context(
        cohort, scores, tau, censoring, protocol
    )
    metric = "ipcw_concordance_index"
    if np.any(events > 1):
        return _not_estimable(
            metric,
            "competing_risk_requires_dedicated_concordance_definition",
            cohort,
            tau,
            lineage=lineage,
        )
    if support_reason is not None:
        return _not_estimable(metric, support_reason, cohort, tau, lineage=lineage)
    risk_scores = _ranking_scores(values, direction)
    numerator = 0.0
    denominator = 0.0
    comparable_events = 0
    for index in np.flatnonzero((events == 1) & (times <= tau)):
        controls = times > times[index]
        n_controls = int(np.count_nonzero(controls))
        if n_controls == 0:
            continue
        g_value = float(censoring.probability(times[index], side="left"))
        if g_value <= 0.0:
            return _not_estimable(
                metric, "zero_event_censoring_survival", cohort, tau, lineage=lineage
            )
        weight = 1.0 / (g_value * g_value)
        differences = risk_scores[index] - risk_scores[controls]
        credit = float(np.count_nonzero(differences > 0.0))
        credit += 0.5 * float(np.count_nonzero(differences == 0.0))
        numerator += weight * credit
        denominator += weight * n_controls
        comparable_events += 1
    if denominator <= 0.0:
        reason = (
            "no_events_by_tau"
            if not np.any((events == 1) & (times <= tau))
            else "no_comparable_pairs"
        )
        return _not_estimable(metric, reason, cohort, tau, lineage=lineage)
    return MetricResult(
        metric=metric,
        status="ok",
        estimate=numerator / denominator,
        reason=None,
        horizon=tau,
        n_patients=cohort.size,
        n_events=int(np.count_nonzero((events == 1) & (times <= tau))),
        effective_n=comparable_events,
        lineage=lineage,
        details=(("score_direction", direction.value), ("truncation", tau)),
    )


def _ipcw_binary_components(
    times: FloatArray,
    events: IntArray,
    risks: FloatArray,
    horizon: float,
    censoring: CensoringDistribution,
    *,
    cause: int,
    control_definition: CompetingControlDefinition,
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]] | None:
    cases = (events == cause) & (times <= horizon)
    at_horizon = times > horizon
    competing = (events > 0) & (events != cause) & (times <= horizon)
    known = cases | at_horizon
    if control_definition is CompetingControlDefinition.NOT_CASE:
        known |= competing
    targets = np.zeros_like(risks)
    targets[cases] = 1.0
    weights = np.zeros_like(risks)
    if np.any(cases):
        case_g = censoring.probability(times[cases], side="left")
        if np.any(case_g <= 0.0):
            return None
        weights[cases] = 1.0 / case_g
    if np.any(at_horizon):
        horizon_g = float(censoring.probability(horizon, side="right"))
        if horizon_g <= 0.0:
            return None
        weights[at_horizon] = 1.0 / horizon_g
    if control_definition is CompetingControlDefinition.NOT_CASE and np.any(competing):
        competing_g = censoring.probability(times[competing], side="left")
        if np.any(competing_g <= 0.0):
            return None
        weights[competing] = 1.0 / competing_g
    return targets, weights, known


def ipcw_brier_score(
    cohort: EvaluationCohort,
    predicted_risk: Sequence[float],
    horizon: float,
    censoring: CensoringDistribution,
    *,
    cause: int | None = None,
    control_definition: CompetingControlDefinition = CompetingControlDefinition.NOT_CASE,
    protocol: EvaluationProtocol | None = None,
    lineage: MetricLineage | None = None,
) -> MetricResult:
    """IPCW Brier score for all-cause risk or an explicit cause-specific CIF."""

    risks = _probability_vector(predicted_risk, name="predicted_risk")
    _, times, events, support_reason = _validate_metric_context(
        cohort, risks.tolist(), horizon, censoring, protocol
    )
    metric = "ipcw_brier_score"
    target = _protocol_cause(events, cause, protocol)
    if support_reason is not None:
        return _not_estimable(
            metric, support_reason, cohort, horizon, cause=target, lineage=lineage
        )
    if not isinstance(control_definition, CompetingControlDefinition):
        control_definition = CompetingControlDefinition(control_definition)
    components = _ipcw_binary_components(
        times,
        events,
        risks,
        horizon,
        censoring,
        cause=target,
        control_definition=control_definition,
    )
    if components is None:
        return _not_estimable(
            metric,
            "zero_censoring_survival_for_weighted_row",
            cohort,
            horizon,
            cause=target,
            lineage=lineage,
        )
    targets, weights, known = components
    effective_n = int(np.count_nonzero(known))
    if effective_n == 0:
        return _not_estimable(
            metric,
            "no_observable_status_at_horizon",
            cohort,
            horizon,
            cause=target,
            lineage=lineage,
        )
    estimate = float(np.sum(weights * np.square(targets - risks)) / cohort.size)
    return MetricResult(
        metric=metric,
        status="ok",
        estimate=estimate,
        reason=None,
        horizon=horizon,
        n_patients=cohort.size,
        n_events=int(np.count_nonzero((events == target) & (times <= horizon))),
        effective_n=effective_n,
        lineage=lineage,
        details=(("target_cause", target), ("control_definition", control_definition.value)),
    )


def integrated_brier_score(
    cohort: EvaluationCohort,
    predicted_risk: Sequence[Sequence[float]],
    horizons: Sequence[float],
    censoring: CensoringDistribution,
    *,
    cause: int | None = None,
    protocol: EvaluationProtocol | None = None,
    lineage: MetricLineage | None = None,
) -> MetricResult:
    """Trapezoidal mean Brier score over a predeclared supported window."""

    grid = _float_vector(horizons, name="horizons")
    if grid.size < 2 or np.any(grid <= 0.0) or np.any(grid[1:] <= grid[:-1]):
        raise ValueError("IBS requires at least two positive, increasing horizons")
    predictions = np.asarray(predicted_risk, dtype=np.float64)
    if predictions.shape != (cohort.size, grid.size):
        raise ValueError("predicted_risk must have shape [patients, horizons]")
    if not np.isfinite(predictions).all() or np.any((predictions < 0.0) | (predictions > 1.0)):
        raise ValueError("predicted risks must be finite probabilities")
    results = [
        ipcw_brier_score(
            cohort,
            predictions[:, index].tolist(),
            float(horizon),
            censoring,
            cause=cause,
            protocol=protocol,
            lineage=lineage,
        )
        for index, horizon in enumerate(grid)
    ]
    failed = next((result for result in results if not result.estimable), None)
    if failed is not None:
        return _not_estimable(
            "integrated_brier_score",
            f"component_not_estimable:{failed.reason}",
            cohort,
            float(grid[-1]),
            cause=cause,
            lineage=lineage,
        )
    values = np.asarray([result.estimate for result in results], dtype=np.float64)
    estimate = float(np.trapezoid(values, grid) / (grid[-1] - grid[0]))
    return MetricResult(
        metric="integrated_brier_score",
        status="ok",
        estimate=estimate,
        reason=None,
        horizon=float(grid[-1]),
        n_patients=cohort.size,
        n_events=results[-1].n_events,
        effective_n=min(result.effective_n for result in results),
        lineage=lineage,
        details=(("window_start", float(grid[0])), ("window_end", float(grid[-1]))),
    )


@dataclass(frozen=True)
class CalibrationBinSpec:
    """Prediction-only bin cut points fitted and frozen on development data."""

    cut_points: tuple[float, ...]
    requested_bins: int
    fitted_on_cohort_id: str
    source_prediction_artifact_id: str

    @classmethod
    def fit(
        cls,
        cohort: EvaluationCohort,
        predicted_risk: Sequence[float],
        *,
        bins: int,
        source_prediction_artifact_id: str,
    ) -> CalibrationBinSpec:
        if cohort.split_role is not SplitRole.DEVELOPMENT:
            raise ValueError("calibration bins may only be fit on the development split")
        if bins < 2:
            raise ValueError("at least two calibration bins are required")
        risks = _probability_vector(predicted_risk, name="predicted_risk")
        if risks.size != cohort.size:
            raise ValueError("one predicted risk is required per patient")
        quantiles = np.quantile(risks, np.arange(1, bins, dtype=np.float64) / bins)
        cut_points = tuple(float(value) for value in np.unique(quantiles))
        return cls(cut_points, bins, cohort.cohort_id, source_prediction_artifact_id)

    def assignments(self, predicted_risk: Sequence[float]) -> IntArray:
        risks = _probability_vector(predicted_risk, name="predicted_risk")
        return np.searchsorted(np.asarray(self.cut_points), risks, side="right").astype(np.int64)


def _observed_cumulative_incidence(
    times: FloatArray,
    event_types: IntArray,
    horizon: float,
    cause: int,
) -> float | None:
    informative = np.any((event_types > 0) & (times <= horizon)) or np.any(times > horizon)
    if not informative:
        return None
    survival = 1.0
    incidence = 0.0
    for time in np.unique(times[times <= horizon]):
        at_risk = int(np.count_nonzero(times >= time))
        if at_risk == 0:
            continue
        events_at_time = (times == time) & (event_types > 0)
        target_events = int(np.count_nonzero(events_at_time & (event_types == cause)))
        all_events = int(np.count_nonzero(events_at_time))
        incidence += survival * target_events / at_risk
        survival *= 1.0 - all_events / at_risk
    return float(incidence)


@dataclass(frozen=True)
class CalibrationPoint:
    bin_index: int
    status: MetricStatus
    predicted_risk: float
    observed_risk: float | None
    n_patients: int
    n_events: int
    reason: str | None = None


@dataclass(frozen=True)
class CalibrationCurveResult:
    status: MetricStatus
    points: tuple[CalibrationPoint, ...]
    weighted_absolute_error: float | None
    reason: str | None
    horizon: float
    estimator: str
    n_patients: int
    n_events: int

    def __post_init__(self) -> None:
        if self.status == "ok" and self.weighted_absolute_error is None:
            raise ValueError("estimable calibration requires a finite summary")
        if self.status == "not_estimable" and not self.reason:
            raise ValueError("non-estimable calibration requires a reason")

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "weighted_absolute_error": self.weighted_absolute_error,
            "reason": self.reason,
            "horizon": self.horizon,
            "estimator": self.estimator,
            "n_patients": self.n_patients,
            "n_events": self.n_events,
            "points": [
                {
                    "bin_index": point.bin_index,
                    "status": point.status,
                    "predicted_risk": point.predicted_risk,
                    "observed_risk": point.observed_risk,
                    "n_patients": point.n_patients,
                    "n_events": point.n_events,
                    "reason": point.reason,
                }
                for point in self.points
            ],
        }


def calibration_curve(
    cohort: EvaluationCohort,
    predicted_risk: Sequence[float],
    horizon: float,
    censoring: CensoringDistribution,
    bin_spec: CalibrationBinSpec,
    *,
    cause: int | None = None,
    protocol: EvaluationProtocol | None = None,
) -> CalibrationCurveResult:
    """Frozen-bin KM/Aalen-Johansen calibration points.

    The observed risk is Kaplan-Meier for a single endpoint and Aalen-Johansen
    when an explicit competing-risk cause is requested.
    """

    risks = _probability_vector(predicted_risk, name="predicted_risk")
    _, times, events, support_reason = _validate_metric_context(
        cohort, risks.tolist(), horizon, censoring, protocol
    )
    target = _protocol_cause(events, cause, protocol)
    estimator = "kaplan_meier" if not np.any(events > 1) else "aalen_johansen"
    event_count = int(np.count_nonzero((events == target) & (times <= horizon)))
    if support_reason is not None:
        return CalibrationCurveResult(
            "not_estimable", (), None, support_reason, horizon, estimator, cohort.size, event_count
        )
    if event_count == 0:
        return CalibrationCurveResult(
            "not_estimable",
            (),
            None,
            "no_events_by_horizon",
            horizon,
            estimator,
            cohort.size,
            0,
        )
    assignments = bin_spec.assignments(risks.tolist())
    points: list[CalibrationPoint] = []
    weighted_error = 0.0
    for bin_index in range(len(bin_spec.cut_points) + 1):
        selected = assignments == bin_index
        count = int(np.count_nonzero(selected))
        if count == 0:
            continue
        predicted = float(risks[selected].mean())
        observed = _observed_cumulative_incidence(
            times[selected], events[selected], horizon, target
        )
        n_events = int(
            np.count_nonzero((events[selected] == target) & (times[selected] <= horizon))
        )
        if observed is None:
            points.append(
                CalibrationPoint(
                    bin_index,
                    "not_estimable",
                    predicted,
                    None,
                    count,
                    n_events,
                    "no_observable_status_at_horizon",
                )
            )
            continue
        points.append(CalibrationPoint(bin_index, "ok", predicted, observed, count, n_events))
        weighted_error += count * abs(predicted - observed)
    if any(point.status != "ok" for point in points):
        return CalibrationCurveResult(
            "not_estimable",
            tuple(points),
            None,
            "one_or_more_bins_not_estimable",
            horizon,
            estimator,
            cohort.size,
            event_count,
        )
    return CalibrationCurveResult(
        "ok",
        tuple(points),
        weighted_error / cohort.size,
        None,
        horizon,
        estimator,
        cohort.size,
        event_count,
    )


def _weighted_isotonic(
    scores: FloatArray, targets: FloatArray, weights: FloatArray
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    order = np.argsort(scores, kind="stable")
    x = scores[order]
    y = targets[order]
    w = weights[order]
    unique_x, inverse = np.unique(x, return_inverse=True)
    sums_w = np.bincount(inverse, weights=w)
    sums_y = np.bincount(inverse, weights=w * y)
    means = sums_y / sums_w
    blocks: list[list[float | int]] = []
    for index, (mean, weight) in enumerate(zip(means, sums_w, strict=True)):
        blocks.append([index, index, float(weight), float(mean)])
        while len(blocks) >= 2 and float(blocks[-2][3]) > float(blocks[-1][3]):
            right = blocks.pop()
            left = blocks.pop()
            total_weight = float(left[2]) + float(right[2])
            pooled = (
                float(left[2]) * float(left[3]) + float(right[2]) * float(right[3])
            ) / total_weight
            blocks.append([int(left[0]), int(right[1]), total_weight, pooled])
    bounds = tuple(float(unique_x[int(block[1])]) for block in blocks)
    values = tuple(float(np.clip(float(block[3]), 0.0, 1.0)) for block in blocks)
    return bounds, values


@dataclass(frozen=True)
class IPCWIsotonicCalibrator:
    """Development-only weighted isotonic mapping frozen before evaluation."""

    upper_bounds: tuple[float, ...]
    calibrated_values: tuple[float, ...]
    horizon: float
    cause: int
    fitted_on_cohort_id: str
    source_prediction_artifact_id: str
    censoring_artifact_id: str
    effective_n: int

    @classmethod
    def fit(
        cls,
        cohort: EvaluationCohort,
        predicted_risk: Sequence[float],
        horizon: float,
        censoring: CensoringDistribution,
        *,
        source_prediction_artifact_id: str,
        cause: int | None = None,
        protocol: EvaluationProtocol | None = None,
    ) -> IPCWIsotonicCalibrator:
        if cohort.split_role is not SplitRole.DEVELOPMENT:
            raise ValueError("calibration mapping may only be fit on development outcomes")
        risks = _probability_vector(predicted_risk, name="predicted_risk")
        _, times, events, support_reason = _validate_metric_context(
            cohort, risks.tolist(), horizon, censoring, protocol
        )
        if support_reason is not None:
            raise ValueError(f"calibration horizon is not estimable: {support_reason}")
        target = _protocol_cause(events, cause, protocol)
        components = _ipcw_binary_components(
            times,
            events,
            risks,
            horizon,
            censoring,
            cause=target,
            control_definition=CompetingControlDefinition.NOT_CASE,
        )
        if components is None:
            raise ValueError("calibration weights are undefined under the censoring model")
        targets, weights, known = components
        if not np.any(targets[known] == 1.0) or not np.any(targets[known] == 0.0):
            raise ValueError("calibration requires an observed case and control")
        bounds, values = _weighted_isotonic(risks[known], targets[known], weights[known])
        return cls(
            upper_bounds=bounds,
            calibrated_values=values,
            horizon=horizon,
            cause=target,
            fitted_on_cohort_id=cohort.cohort_id,
            source_prediction_artifact_id=source_prediction_artifact_id,
            censoring_artifact_id=censoring.artifact_id,
            effective_n=int(np.count_nonzero(known)),
        )

    def transform(self, predicted_risk: Sequence[float]) -> FloatArray:
        risks = _probability_vector(predicted_risk, name="predicted_risk")
        bounds = np.asarray(self.upper_bounds, dtype=np.float64)
        values = np.asarray(self.calibrated_values, dtype=np.float64)
        indices = np.searchsorted(bounds, risks, side="left")
        indices = np.clip(indices, 0, len(values) - 1)
        return values[indices].copy()


@dataclass(frozen=True)
class MatchedPredictionSet:
    keys: tuple[tuple[str, str, float, str, float, int | None, int], ...]
    patient_ids: tuple[str, ...]
    left_risk: tuple[float, ...]
    right_risk: tuple[float, ...]
    input_artifact_id: str


def match_prediction_artifacts(
    left: PredictionArtifact,
    right: PredictionArtifact,
) -> MatchedPredictionSet:
    """Align two methods only when population, query and input lineage match."""

    if left.lineage.run_mode is not right.lineage.run_mode:
        raise ValueError("cannot compare synthetic and real predictions")
    if left.lineage.input_artifact_id != right.lineage.input_artifact_id:
        raise ValueError("matched-input comparison requires identical input lineage")
    if (
        left.lineage.cohort_artifact_id != right.lineage.cohort_artifact_id
        or left.lineage.split_version != right.lineage.split_version
    ):
        raise ValueError("matched comparison requires identical cohort and split lineage")
    left_by_key = {record.match_key: record for record in left.records}
    right_by_key = {record.match_key: record for record in right.records}
    if left_by_key.keys() != right_by_key.keys():
        raise ValueError("matched comparison requires identical patient/query keys")
    keys = tuple(sorted(left_by_key))
    for key in keys:
        if left_by_key[key].input_artifact_id != right_by_key[key].input_artifact_id:
            raise ValueError("matched rows have different input lineage")
    return MatchedPredictionSet(
        keys=keys,
        patient_ids=tuple(key[0] for key in keys),
        left_risk=tuple(left_by_key[key].risk for key in keys),
        right_risk=tuple(right_by_key[key].risk for key in keys),
        input_artifact_id=left.lineage.input_artifact_id,
    )


@dataclass(frozen=True)
class PatientBootstrapPlan:
    """Patient draws whose expansion keeps every row/stage of a patient together."""

    unique_patient_ids: tuple[str, ...]
    patient_draws: tuple[tuple[int, ...], ...]
    seed: int

    def expand_rows(self, patient_ids: Sequence[str], replicate: int) -> IntArray:
        if replicate < 0 or replicate >= len(self.patient_draws):
            raise IndexError("bootstrap replicate is out of range")
        row_groups: dict[str, list[int]] = {
            patient_id: [] for patient_id in self.unique_patient_ids
        }
        for row_index, patient_id in enumerate(patient_ids):
            if patient_id not in row_groups:
                raise ValueError("row patient IDs differ from the bootstrap plan")
            row_groups[patient_id].append(row_index)
        if any(not rows for rows in row_groups.values()):
            raise ValueError("bootstrap plan contains a patient absent from rows")
        expanded: list[int] = []
        for patient_position in self.patient_draws[replicate]:
            patient_id = self.unique_patient_ids[patient_position]
            expanded.extend(row_groups[patient_id])
        return np.asarray(expanded, dtype=np.int64)


def make_patient_bootstrap_plan(
    patient_ids: Sequence[str], *, n_resamples: int, seed: int
) -> PatientBootstrapPlan:
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not patient_ids or any(not patient_id for patient_id in patient_ids):
        raise ValueError("patient_ids must be nonempty")
    unique_ids = tuple(dict.fromkeys(patient_ids))
    generator = np.random.default_rng(seed)
    draws = generator.integers(
        0, len(unique_ids), size=(n_resamples, len(unique_ids)), endpoint=False
    )
    return PatientBootstrapPlan(
        unique_patient_ids=unique_ids,
        patient_draws=tuple(tuple(int(value) for value in row) for row in draws),
        seed=seed,
    )


BootstrapStatistic = Callable[[FloatArray, IntArray], float | MetricResult]


@dataclass(frozen=True)
class PairedBootstrapResult:
    status: MetricStatus
    estimate_difference: float | None
    confidence_lower: float | None
    confidence_upper: float | None
    confidence_level: float
    n_resamples: int
    n_estimable_resamples: int
    n_patients: int
    reason: str | None
    resampling_unit: str = "patient"
    seed_variability_included: bool = False


def _statistic_value(
    statistic: BootstrapStatistic, values: FloatArray, indices: IntArray
) -> float | None:
    result = statistic(values, indices)
    if isinstance(result, MetricResult):
        return result.estimate if result.estimable else None
    value = float(result)
    return value if math.isfinite(value) else None


def paired_patient_bootstrap(
    patient_ids: Sequence[str],
    left_values: Sequence[float],
    right_values: Sequence[float],
    statistic: BootstrapStatistic,
    *,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 0,
    plan: PatientBootstrapPlan | None = None,
) -> PairedBootstrapResult:
    """Paired method difference using one shared patient draw per replicate.

    The callback receives sampled values and the corresponding original row
    indices.  This permits censoring-aware callbacks to index their outcomes
    with exactly the same draw used for each method.
    """

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    left = _float_vector(left_values, name="left_values")
    right = _float_vector(right_values, name="right_values")
    if left.shape != right.shape or left.size != len(patient_ids):
        raise ValueError("patient IDs and paired values must have equal length")
    active_plan = plan or make_patient_bootstrap_plan(
        patient_ids, n_resamples=n_resamples, seed=seed
    )
    if len(active_plan.patient_draws) != n_resamples:
        raise ValueError("bootstrap plan replicate count differs from n_resamples")
    expected_ids = tuple(dict.fromkeys(patient_ids))
    if active_plan.unique_patient_ids != expected_ids:
        raise ValueError("bootstrap plan patient order differs from supplied rows")

    original_indices = np.arange(left.size, dtype=np.int64)
    point_left = _statistic_value(statistic, left, original_indices)
    point_right = _statistic_value(statistic, right, original_indices)
    if point_left is None or point_right is None:
        return PairedBootstrapResult(
            "not_estimable",
            None,
            None,
            None,
            confidence_level,
            n_resamples,
            0,
            len(expected_ids),
            "point_metric_not_estimable",
        )

    differences: list[float] = []
    for replicate in range(n_resamples):
        indices = active_plan.expand_rows(patient_ids, replicate)
        left_estimate = _statistic_value(statistic, left[indices], indices)
        right_estimate = _statistic_value(statistic, right[indices], indices)
        if left_estimate is None or right_estimate is None:
            continue
        differences.append(left_estimate - right_estimate)
    if not differences:
        return PairedBootstrapResult(
            "not_estimable",
            point_left - point_right,
            None,
            None,
            confidence_level,
            n_resamples,
            0,
            len(expected_ids),
            "no_estimable_bootstrap_replicates",
        )
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(np.asarray(differences), [alpha, 1.0 - alpha])
    return PairedBootstrapResult(
        "ok",
        point_left - point_right,
        float(lower),
        float(upper),
        confidence_level,
        n_resamples,
        len(differences),
        len(expected_ids),
        None,
    )


@dataclass(frozen=True)
class SeedSummary:
    mean: float
    sample_standard_deviation: float | None
    n_seeds: int
    seeds: tuple[int, ...]
    uncertainty_unit: str = "training_seed"


def summarize_seed_variability(estimates: Mapping[int, float]) -> SeedSummary:
    if not estimates:
        raise ValueError("at least one seed estimate is required")
    seeds = tuple(sorted(estimates))
    values = _float_vector([estimates[seed] for seed in seeds], name="seed estimates")
    standard_deviation = float(values.std(ddof=1)) if values.size > 1 else None
    return SeedSummary(float(values.mean()), standard_deviation, len(seeds), seeds)


@dataclass(frozen=True)
class AblationSpec:
    ablation_id: str
    change: str
    held_constant: tuple[str, ...]
    interpretation: str
    status: str = "planned"


def default_ablation_registry() -> tuple[AblationSpec, ...]:
    """Locked A1-A10 definitions; registering an ablation is not running it."""

    return (
        AblationSpec(
            "A1",
            "disable future CT representation loss",
            ("real CT1 posterior update", "state architecture", "survival inputs"),
            "isolates future-CT supervision",
        ),
        AblationSpec(
            "A2",
            "disable future pathology representation loss",
            ("real S2 pathology posterior update", "state architecture", "survival inputs"),
            "isolates future-pathology supervision",
        ),
        AblationSpec(
            "A3",
            "disable response auxiliary loss",
            ("future representation losses", "observations", "state architecture"),
            "isolates response supervision",
        ),
        AblationSpec(
            "A4",
            "replace prior/posterior separation with direct history fusion",
            ("legal observations", "foundation features", "survival head"),
            "tests the predictive-update structure",
        ),
        AblationSpec(
            "A5",
            "remove treatment content",
            ("observed timing", "legal observations", "state capacity"),
            "tests treatment-associated conditioning, not a causal effect",
        ),
        AblationSpec(
            "A6",
            "remove continuous-time encoding",
            ("treatment content", "legal observations", "state capacity"),
            "isolates continuous timing",
        ),
        AblationSpec(
            "A7",
            "use deterministic latent state",
            ("state width", "observations", "losses except KL"),
            "tests stochastic-state necessity",
        ),
        AblationSpec(
            "A8",
            "replace the foundation encoder",
            ("cohort", "endpoint", "downstream architecture", "training budget"),
            "tests dependence on the frozen feature source",
        ),
        AblationSpec(
            "A9",
            "disable encoder fine-tuning",
            ("encoder initialization", "cohort", "downstream architecture"),
            "isolates parameter-efficient adaptation",
        ),
        AblationSpec(
            "A10",
            "apply policy-aligned missingness and time-proxy sensitivity",
            ("endpoint", "query definition", "trained checkpoint"),
            "tests reliance on missingness or timing proxies",
        ),
    )


@dataclass(frozen=True)
class FigureLineage:
    artifact_id: str
    figure_kind: str
    prediction_artifact_ids: tuple[str, ...]
    metric_artifact_ids: tuple[str, ...]
    protocol_id: str
    run_mode: RunMode

    def __post_init__(self) -> None:
        _validate_artifact_id(self.artifact_id, name="artifact_id")
        for value in (*self.prediction_artifact_ids, *self.metric_artifact_ids):
            _validate_artifact_id(value, name="referenced artifact ID")
        if not self.figure_kind or not self.protocol_id:
            raise ValueError("figure_kind and protocol_id must be nonempty")
        if not isinstance(self.run_mode, RunMode):
            object.__setattr__(self, "run_mode", RunMode(self.run_mode))

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "figure_kind": self.figure_kind,
            "prediction_artifact_ids": list(self.prediction_artifact_ids),
            "metric_artifact_ids": list(self.metric_artifact_ids),
            "protocol_id": self.protocol_id,
            "run_mode": self.run_mode.value,
        }


@dataclass(frozen=True)
class EvaluationArtifactWriter:
    """Write traceable artifacts under an explicit synthetic/real namespace."""

    root: Path
    run_mode: RunMode

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        if not isinstance(self.run_mode, RunMode):
            object.__setattr__(self, "run_mode", RunMode(self.run_mode))

    def _check_mode(self, mode: RunMode) -> None:
        if mode is not self.run_mode:
            raise ValueError("artifact run mode does not match writer namespace")

    def write_predictions(self, artifact: PredictionArtifact) -> Path:
        self._check_mode(artifact.lineage.run_mode)
        path = (
            self.root / self.run_mode.value / "predictions" / f"{artifact.lineage.artifact_id}.json"
        )
        return atomic_write_json(path, artifact.as_dict())

    def write_metric(self, result: MetricResult) -> Path:
        if result.lineage is None:
            raise ValueError("persisted metrics require complete lineage")
        self._check_mode(result.lineage.run_mode)
        path = self.root / self.run_mode.value / "metrics" / f"{result.lineage.artifact_id}.json"
        return atomic_write_json(path, result.as_dict())

    def write_figure_lineage(self, lineage: FigureLineage) -> Path:
        self._check_mode(lineage.run_mode)
        if not lineage.prediction_artifact_ids or not lineage.metric_artifact_ids:
            raise ValueError("figures must reference prediction and metric artifacts")
        path = self.root / self.run_mode.value / "figures" / f"{lineage.artifact_id}.json"
        return atomic_write_json(path, lineage.as_dict())
