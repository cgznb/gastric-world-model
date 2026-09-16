"""Model-external construction of temporally legal patient prefixes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum

from stageworld.errors import DataContractError

from .contracts import (
    AvailabilityBasis,
    ClinicalMeasurement,
    DataMode,
    Observation,
    ObservationRole,
    Patient,
    Prefix,
    PrefixExclusion,
    Query,
    SourceType,
    Stage,
    TimePrecision,
    Treatment,
    TreatmentKind,
    TreatmentStatus,
)


class SameDayPolicy(StrEnum):
    REJECT = "reject"
    CONSERVATIVE_EXCLUDE = "conservative_exclude"
    REQUIRE_RECORDED_ORDER = "require_recorded_order"


_STAGE_RANK = {Stage.S0: 0, Stage.S1: 1, Stage.S2: 2}


def _default_roles() -> dict[Stage, frozenset[ObservationRole]]:
    return {
        Stage.S0: frozenset({ObservationRole.BASELINE_CT}),
        Stage.S1: frozenset(
            {ObservationRole.BASELINE_CT, ObservationRole.POST_TREATMENT_CT}
        ),
        Stage.S2: frozenset(
            {
                ObservationRole.BASELINE_CT,
                ObservationRole.POST_TREATMENT_CT,
                ObservationRole.SURGICAL_PATHOLOGY,
            }
        ),
    }


@dataclass(frozen=True, slots=True)
class FeaturePolicy:
    """Explicit allow rules; unlisted clinical fields fail closed by exclusion."""

    mode: DataMode
    clinical_min_stage: Mapping[str, Stage] = field(default_factory=dict)
    allowed_observation_roles: Mapping[Stage, frozenset[ObservationRole]] = field(
        default_factory=_default_roles
    )
    forbidden_clinical_fields: frozenset[str] = frozenset(
        {
            "event",
            "event_date",
            "censor_date",
            "survival_status",
            "survival_time",
            "followup_duration",
            "recurrence_status",
            "recurrence_time",
            "patient_id",
            "imaging_id",
            "pathology_id",
            "name",
        }
    )
    same_day_policy: SameDayPolicy = SameDayPolicy.CONSERVATIVE_EXCLUDE
    strict_unknown_availability_basis: bool = True
    allow_predicted_observations: bool = False

    def source_allowed(self, source_type: SourceType) -> bool:
        if source_type is SourceType.PREDICTED:
            return self.allow_predicted_observations
        if self.mode is DataMode.SYNTHETIC:
            return source_type in {SourceType.OBSERVED, SourceType.SYNTHETIC}
        return source_type is SourceType.OBSERVED

    def clinical_stage(self, field_name: str) -> Stage | None:
        if field_name in self.clinical_min_stage:
            return self.clinical_min_stage[field_name]
        normalized = field_name.strip().casefold()
        return next(
            (
                stage
                for configured, stage in self.clinical_min_stage.items()
                if configured.strip().casefold() == normalized
            ),
            None,
        )

    def clinical_field_forbidden(self, field_name: str) -> bool:
        normalized = field_name.strip().casefold()
        return any(
            configured.strip().casefold() == normalized
            for configured in self.forbidden_clinical_fields
        )


@dataclass(frozen=True, slots=True)
class _Visibility:
    include: bool
    reason: str = ""


class FeatureFirewall:
    """Build legal prefixes without ever receiving outcome records.

    Filtering happens before records are assembled into a model-facing object. Future
    modalities are ignored without adding exclusion metadata, so their existence cannot
    alter an S0 prefix or its eligibility.
    """

    def __init__(
        self,
        patients: Iterable[Patient],
        observations: Iterable[Observation],
        clinical_measurements: Iterable[ClinicalMeasurement],
        treatments: Iterable[Treatment],
        policy: FeaturePolicy,
    ) -> None:
        patient_list = tuple(patients)
        self._patients = {patient.patient_id: patient for patient in patient_list}
        if len(self._patients) != len(patient_list):
            raise DataContractError("duplicate_patient", "patient_id values must be unique")
        self._observations = self._group(observations)
        self._measurements = self._group(clinical_measurements)
        self._treatments = self._group(treatments)
        self.policy = policy
        known = set(self._patients)
        referenced = set(self._observations) | set(self._measurements) | set(self._treatments)
        if unknown := referenced - known:
            raise DataContractError(
                "unknown_patient_reference",
                "Feature records reference patients absent from the patient table",
                details={"unknown_count": len(unknown)},
            )

    @staticmethod
    def _group(records: Iterable[object]) -> dict[str, tuple[object, ...]]:
        grouped: dict[str, list[object]] = defaultdict(list)
        for record in records:
            grouped[getattr(record, "patient_id")].append(record)  # noqa: B009
        return {patient_id: tuple(items) for patient_id, items in grouped.items()}

    def _visibility(
        self,
        *,
        record_id: str,
        available_at_days: float | None,
        availability_basis: AvailabilityBasis,
        time_precision: TimePrecision,
        query: Query,
    ) -> _Visibility:
        if available_at_days is None:
            raise DataContractError(
                "available_at_missing",
                "A stage-permitted record has unknown available_at and cannot enter a prefix",
                remediation=(
                    "Provide a confirmed or explicitly inferred conservative availability time"
                ),
                details={"record_id": record_id, "query_id": query.query_id},
            )
        if (
            self.policy.strict_unknown_availability_basis
            and availability_basis is AvailabilityBasis.UNKNOWN
        ):
            raise DataContractError(
                "availability_basis_unknown",
                "A stage-permitted record has no usable availability basis",
                details={"record_id": record_id, "query_id": query.query_id},
            )
        if available_at_days > query.query_time_days:
            return _Visibility(False, "not_yet_available")
        same_day_ambiguous = (
            available_at_days == query.query_time_days
            and time_precision in {TimePrecision.DAY, TimePrecision.MONTH, TimePrecision.UNKNOWN}
            and availability_basis
            not in {
                AvailabilityBasis.RECORDED_WITH_ORDER,
                AvailabilityBasis.PROTOCOL_DEFINED_ORDER,
            }
        )
        if not same_day_ambiguous:
            return _Visibility(True)
        if self.policy.same_day_policy is SameDayPolicy.REJECT:
            raise DataContractError(
                "same_day_order_ambiguous",
                "Record and query share an imprecise date without a known order",
                details={"record_id": record_id, "query_id": query.query_id},
            )
        return _Visibility(False, "same_day_order_ambiguous")

    def _build(
        self,
        patient_id: str,
        query: Query,
        *,
        observation_roles: frozenset[ObservationRole] | None = None,
        include_planned_treatments: bool = True,
    ) -> Prefix:
        if query.patient_id != patient_id:
            raise DataContractError(
                "query_patient_mismatch", "query.patient_id does not match requested patient"
            )
        try:
            patient = self._patients[patient_id]
        except KeyError as exc:
            raise DataContractError(
                "patient_not_found", "Requested patient is not present"
            ) from exc
        if query.stage is Stage.S0 and not patient.baseline_eligible:
            raise DataContractError(
                "baseline_ineligible", "Patient is not eligible by baseline-known criteria"
            )

        roles = observation_roles or self.policy.allowed_observation_roles[query.stage]
        observations: list[Observation] = []
        measurements: list[ClinicalMeasurement] = []
        treatments: list[Treatment] = []
        exclusions: list[PrefixExclusion] = []

        for raw in self._observations.get(patient_id, ()):
            observation = raw
            assert isinstance(observation, Observation)
            # Do not expose even the existence of a role that is illegal at this stage.
            if observation.role not in roles:
                continue
            if not self.policy.source_allowed(observation.source_type):
                raise DataContractError(
                    "source_type_not_allowed",
                    "Observation provenance is not allowed in the configured data mode",
                    details={"observation_id": observation.observation_id},
                )
            visible = self._visibility(
                record_id=observation.observation_id,
                available_at_days=observation.available_at_days,
                availability_basis=observation.availability_basis,
                time_precision=observation.time_precision,
                query=query,
            )
            if visible.include:
                observations.append(observation)

        for raw in self._measurements.get(patient_id, ()):
            measurement = raw
            assert isinstance(measurement, ClinicalMeasurement)
            if self.policy.clinical_field_forbidden(measurement.field_name):
                continue
            minimum_stage = self.policy.clinical_stage(measurement.field_name)
            if minimum_stage is None or _STAGE_RANK[minimum_stage] > _STAGE_RANK[query.stage]:
                continue
            if not self.policy.source_allowed(measurement.source_type):
                raise DataContractError(
                    "source_type_not_allowed",
                    "Clinical provenance is not allowed in the configured data mode",
                    details={"measurement_id": measurement.measurement_id},
                )
            visible = self._visibility(
                record_id=measurement.measurement_id,
                available_at_days=measurement.available_at_days,
                availability_basis=measurement.availability_basis,
                time_precision=measurement.time_precision,
                query=query,
            )
            if visible.include:
                measurements.append(measurement)

        for raw in self._treatments.get(patient_id, ()):
            treatment = raw
            assert isinstance(treatment, Treatment)
            minimum_stage = self._treatment_min_stage(treatment)
            if _STAGE_RANK[minimum_stage] > _STAGE_RANK[query.stage]:
                continue
            if treatment.planned_or_delivered is TreatmentStatus.PLANNED:
                if not include_planned_treatments:
                    continue
            elif treatment.planned_or_delivered is TreatmentStatus.UNKNOWN:
                continue
            elif treatment.start_days is None or treatment.end_days is None:
                raise DataContractError(
                    "delivered_treatment_time_missing",
                    "Delivered aggregate treatment needs real start and end times",
                    details={"event_id": treatment.event_id},
                )
            elif (
                treatment.start_days > query.query_time_days
                or treatment.end_days > query.query_time_days
            ):
                continue
            visible = self._visibility(
                record_id=treatment.event_id,
                available_at_days=treatment.available_at_days,
                availability_basis=treatment.availability_basis,
                time_precision=treatment.time_precision,
                query=query,
            )
            if visible.include:
                treatments.append(treatment)

        observations.sort(key=lambda x: (x.available_at_days or 0.0, x.observation_id))
        measurements.sort(key=lambda x: (x.available_at_days or 0.0, x.measurement_id))
        treatments.sort(key=lambda x: (x.available_at_days or 0.0, x.event_id))
        exclusions.sort(key=lambda x: (x.record_kind, x.record_id, x.reason))
        return Prefix(
            patient=patient,
            query=query,
            observations=tuple(observations),
            clinical_measurements=tuple(measurements),
            treatments=tuple(treatments),
            exclusions=tuple(exclusions),
        )

    @staticmethod
    def _treatment_min_stage(treatment: Treatment) -> Stage:
        if treatment.treatment_kind is TreatmentKind.SURGERY:
            return Stage.S2
        if treatment.planned_or_delivered is TreatmentStatus.PLANNED:
            return Stage.S0
        return Stage.S1

    def build_prefix(self, patient_id: str, query: Query) -> Prefix:
        """Build the main clinical prefix; query.target_time_days is intentionally ignored."""

        return self._build(patient_id, replace(query, target_time_days=None))

    def build_ct_target_context(self, target: Observation, source_query: Query) -> Prefix:
        """Build prior context for a future CT target using only pre-acquisition history.

        A CT acquired before a late report is supervised at its acquisition time. Treatments
        completed after acquisition are therefore unavailable here even if a later clinical
        query can legally observe them.
        """

        if target.role is not ObservationRole.POST_TREATMENT_CT:
            raise DataContractError(
                "invalid_ct_target", "CT target context requires a post-treatment CT"
            )
        if target.patient_id != source_query.patient_id:
            raise DataContractError(
                "query_patient_mismatch", "Target and source query patients differ"
            )
        target_query = replace(
            source_query,
            query_id=f"{source_query.query_id}:ct-target",
            stage=Stage.S1,
            query_time_days=target.acquired_at_days,
            target_time_days=None,
        )
        return self._build(
            target.patient_id,
            target_query,
            observation_roles=frozenset({ObservationRole.BASELINE_CT}),
            include_planned_treatments=False,
        )
