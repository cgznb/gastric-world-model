"""Typed data contracts for temporally safe StageWorld cohorts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias

from stageworld.errors import DataContractError

Scalar: TypeAlias = str | int | float | bool | None


class Stage(StrEnum):
    S0 = "s0"
    S1 = "s1"
    S2 = "s2"


class ClockKind(StrEnum):
    """The three clocks that must not be collapsed into one value."""

    EVENT_OR_ACQUISITION = "event_or_acquisition"
    INFORMATION_AVAILABILITY = "information_availability"
    PREDICTION_HORIZON = "prediction_horizon"


class TimePrecision(StrEnum):
    DATETIME = "datetime"
    DAY = "day"
    MONTH = "month"
    UNKNOWN = "unknown"


class MissingCategory(StrEnum):
    MISSING = "missing"
    NOT_YET_AVAILABLE = "not_yet_available"
    NOT_APPLICABLE = "not_applicable"
    NOT_PERFORMED = "not_performed"
    FAILED_QC = "failed_qc"
    UNKNOWN = "unknown"


class Modality(StrEnum):
    CT = "ct"
    PATHOLOGY = "pathology"
    OTHER = "other"


class ObservationRole(StrEnum):
    BASELINE_CT = "baseline_ct"
    POST_TREATMENT_CT = "post_treatment_ct"
    SURGICAL_PATHOLOGY = "surgical_pathology"
    OTHER = "other"


class SourceType(StrEnum):
    OBSERVED = "observed"
    SYNTHETIC = "synthetic"
    PREDICTED = "predicted"


class AvailabilityBasis(StrEnum):
    RECORDED = "recorded"
    RECORDED_WITH_ORDER = "recorded_with_order"
    PROTOCOL_DEFINED_ORDER = "protocol_defined_order"
    INFERRED_CONSERVATIVE = "inferred_conservative"
    RESEARCH_ASSUMPTION = "research_assumption"
    UNKNOWN = "unknown"


class QualityStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    UNKNOWN = "unknown"


class TreatmentStatus(StrEnum):
    PLANNED = "planned"
    DELIVERED = "delivered"
    UNKNOWN = "unknown"


class TreatmentKind(StrEnum):
    SYSTEMIC = "systemic"
    CHEMOTHERAPY = "chemotherapy"
    IMMUNOTHERAPY = "immunotherapy"
    TARGETED = "targeted"
    SURGERY = "surgery"
    OTHER = "other"


class EventType(StrEnum):
    DEATH = "death"
    RECURRENCE = "recurrence"
    DFS_EVENT = "dfs_event"
    OTHER = "other"


class AdjudicationStatus(StrEnum):
    CONFIRMED = "confirmed"
    PENDING = "pending"
    AMBIGUOUS = "ambiguous"


class QueryEligibility(StrEnum):
    ELIGIBLE = "eligible"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class DataMode(StrEnum):
    SYNTHETIC = "synthetic"
    REAL_FEATURES = "real_features"
    REAL_IMAGES = "real_images"


def _require_finite(name: str, value: float | None, *, optional: bool = False) -> None:
    if value is None:
        if optional:
            return
        raise DataContractError("missing_time", f"{name} is required")
    if not math.isfinite(float(value)):
        raise DataContractError("invalid_time", f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class Patient:
    patient_id: str
    site_id: str
    cohort_id: str
    eligibility_version: str
    baseline_origin_local: float
    index_event_type: str
    baseline_eligible: bool = True

    def __post_init__(self) -> None:
        if not self.patient_id:
            raise DataContractError("missing_patient_id", "patient_id must be non-empty")
        _require_finite("baseline_origin_local", self.baseline_origin_local)


@dataclass(frozen=True, slots=True)
class Observation:
    observation_id: str
    patient_id: str
    modality: Modality
    role: ObservationRole
    source_type: SourceType
    acquired_at_days: float
    available_at_days: float | None
    time_precision: TimePrecision
    availability_basis: AvailabilityBasis
    local_asset_id: str | None = None
    quality_status: QualityStatus = QualityStatus.UNKNOWN
    missing_reason: MissingCategory | None = None
    phase: str | None = None
    tissue_source: str | None = None
    feature_version: str | None = None

    def __post_init__(self) -> None:
        if not self.observation_id or not self.patient_id:
            raise DataContractError(
                "missing_observation_identity", "observation_id and patient_id are required"
            )
        _require_finite("acquired_at_days", self.acquired_at_days)
        _require_finite("available_at_days", self.available_at_days, optional=True)
        if (
            self.available_at_days is not None
            and self.available_at_days < self.acquired_at_days
        ):
            raise DataContractError(
                "availability_before_acquisition",
                "An observation cannot be available before it is acquired",
                details={"observation_id": self.observation_id},
            )
        if self.quality_status is QualityStatus.FAILED and self.missing_reason not in {
            MissingCategory.FAILED_QC,
            MissingCategory.UNKNOWN,
        }:
            raise DataContractError(
                "qc_missing_reason_mismatch",
                "A failed-QC observation must carry failed_qc or unknown missing_reason",
            )
        if self.missing_reason is None and self.local_asset_id is None:
            raise DataContractError(
                "observation_without_asset",
                "An available observation needs an asset or an explicit missing_reason",
            )


@dataclass(frozen=True, slots=True)
class ClinicalMeasurement:
    measurement_id: str
    patient_id: str
    field_name: str
    typed_value: Scalar
    unit: str | None
    source_type: SourceType
    acquired_at_days: float
    available_at_days: float | None
    time_precision: TimePrecision
    availability_basis: AvailabilityBasis
    missing_reason: MissingCategory | None = None
    provenance: str = ""

    def __post_init__(self) -> None:
        if not self.measurement_id or not self.patient_id or not self.field_name:
            raise DataContractError(
                "missing_measurement_identity",
                "measurement_id, patient_id, and field_name are required",
            )
        _require_finite("acquired_at_days", self.acquired_at_days)
        _require_finite("available_at_days", self.available_at_days, optional=True)
        if self.available_at_days is not None and self.available_at_days < self.acquired_at_days:
            raise DataContractError(
                "availability_before_measurement",
                "A measurement cannot be available before it is acquired",
            )


@dataclass(frozen=True, slots=True)
class Treatment:
    patient_id: str
    event_id: str
    treatment_kind: TreatmentKind
    standardized_components: tuple[str, ...]
    regimen_code: str | None
    planned_or_delivered: TreatmentStatus
    start_days: float | None
    end_days: float | None
    available_at_days: float | None
    time_precision: TimePrecision
    availability_basis: AvailabilityBasis
    cycles_delivered_by_event: int | None = None
    dose_if_available: float | None = None
    surgery_type: str | None = None
    intent_if_confirmed: str | None = None
    data_granularity: str = "event"

    def __post_init__(self) -> None:
        if not self.patient_id or not self.event_id:
            raise DataContractError(
                "missing_treatment_identity", "patient_id and event_id are required"
            )
        for name, value in (
            ("start_days", self.start_days),
            ("end_days", self.end_days),
            ("available_at_days", self.available_at_days),
        ):
            _require_finite(name, value, optional=True)
        if self.start_days is not None and self.end_days is not None:
            if self.end_days < self.start_days:
                raise DataContractError(
                    "treatment_end_before_start", "Treatment end cannot precede start"
                )
        if self.cycles_delivered_by_event is not None and self.cycles_delivered_by_event < 0:
            raise DataContractError(
                "negative_treatment_cycles", "Treatment cycle count cannot be negative"
            )


@dataclass(frozen=True, slots=True)
class Outcome:
    patient_id: str
    endpoint_name: str
    event_type: EventType
    source_status: Scalar
    event_date_days: float | None
    censor_date_days: float | None
    adjudication_status: AdjudicationStatus
    origin_definition: str
    reason_of_censoring: str | None
    label_version: str

    def __post_init__(self) -> None:
        if not self.patient_id or not self.endpoint_name:
            raise DataContractError(
                "missing_outcome_identity", "patient_id and endpoint_name are required"
            )
        _require_finite("event_date_days", self.event_date_days, optional=True)
        _require_finite("censor_date_days", self.censor_date_days, optional=True)


@dataclass(frozen=True, slots=True)
class Query:
    query_id: str
    patient_id: str
    stage: Stage
    query_time_days: float
    prediction_horizon_days: float
    time_precision: TimePrecision = TimePrecision.DAY
    eligibility: QueryEligibility = QueryEligibility.ELIGIBLE
    eligibility_basis: str = "predeclared"
    target_time_days: float | None = None

    def __post_init__(self) -> None:
        if not self.query_id or not self.patient_id:
            raise DataContractError(
                "missing_query_identity", "query_id and patient_id are required"
            )
        _require_finite("query_time_days", self.query_time_days)
        _require_finite("prediction_horizon_days", self.prediction_horizon_days)
        _require_finite("target_time_days", self.target_time_days, optional=True)
        if self.prediction_horizon_days <= 0:
            raise DataContractError(
                "invalid_prediction_horizon", "prediction_horizon_days must be positive"
            )
        if self.target_time_days is not None and self.target_time_days < self.query_time_days:
            raise DataContractError(
                "target_before_query", "target_time_days cannot precede query_time_days"
            )


@dataclass(frozen=True, slots=True)
class PrefixExclusion:
    record_kind: str
    record_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class Prefix:
    patient: Patient
    query: Query
    observations: tuple[Observation, ...] = ()
    clinical_measurements: tuple[ClinicalMeasurement, ...] = ()
    treatments: tuple[Treatment, ...] = ()
    exclusions: tuple[PrefixExclusion, ...] = ()

    @property
    def patient_id(self) -> str:
        return self.patient.patient_id


@dataclass(frozen=True, slots=True)
class SurvivalLabel:
    patient_id: str
    endpoint_name: str
    query_time_days: float
    remaining_time_days: float
    event: bool
    label_version: str


@dataclass(frozen=True, slots=True)
class Landmark:
    landmark_id: str
    patient_id: str
    stage: Stage
    query_time_days: float
    prefix: Prefix
    label: SurvivalLabel


@dataclass(frozen=True, slots=True)
class LandmarkExclusion:
    query_id: str
    patient_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class LandmarkSet:
    landmarks: tuple[Landmark, ...]
    exclusions: tuple[LandmarkExclusion, ...]


@dataclass(frozen=True, slots=True)
class DataIssue:
    patient_id: str | None
    code: str
    source_field: str | None = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class Cohort:
    patients: tuple[Patient, ...]
    observations: tuple[Observation, ...]
    clinical_measurements: tuple[ClinicalMeasurement, ...]
    treatments: tuple[Treatment, ...]
    outcomes: tuple[Outcome, ...]
    queries: tuple[Query, ...]
    issues: tuple[DataIssue, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        patient_ids = [patient.patient_id for patient in self.patients]
        if len(patient_ids) != len(set(patient_ids)):
            raise DataContractError("duplicate_patient", "patient_id values must be unique")
        known = set(patient_ids)
        for collection in (
            self.observations,
            self.clinical_measurements,
            self.treatments,
            self.outcomes,
            self.queries,
        ):
            unknown = {record.patient_id for record in collection} - known
            if unknown:
                raise DataContractError(
                    "unknown_patient_reference",
                    "A cohort record references a patient absent from patients",
                    details={"unknown_count": len(unknown)},
                )
