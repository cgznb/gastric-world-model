"""Deterministic, explicitly synthetic cohort fixtures for engineering tests."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .contracts import (
    AdjudicationStatus,
    AvailabilityBasis,
    ClinicalMeasurement,
    Cohort,
    DataIssue,
    DataMode,
    EventType,
    MissingCategory,
    Modality,
    Observation,
    ObservationRole,
    Outcome,
    Patient,
    QualityStatus,
    Query,
    QueryEligibility,
    SourceType,
    Stage,
    TimePrecision,
    Treatment,
    TreatmentKind,
    TreatmentStatus,
)
from .firewall import FeaturePolicy


@dataclass(frozen=True)
class SyntheticTimeline:
    """One shared dual-clock contract for all synthetic workflow representations."""

    baseline_acquired: float = -1.0
    s0_query: float = 0.0
    systemic_event: float = 25.0
    systemic_available: float = 26.0
    ct1_acquired: float = 30.0
    post_ct_event: float = 31.0
    post_ct_available: float = 31.0
    ct1_available: float = 32.0
    s1_query: float = 32.0
    surgery_event: float = 40.0
    surgery_available: float = 40.0
    pathology_acquired: float = 40.0
    pathology_available: float = 42.0
    s2_query: float = 45.0


SYNTHETIC_TIMELINE = SyntheticTimeline()


def default_synthetic_policy() -> FeaturePolicy:
    return FeaturePolicy(
        mode=DataMode.SYNTHETIC,
        clinical_min_stage={
            "age_years": Stage.S0,
            "baseline_stage": Stage.S0,
            "radiologic_response": Stage.S1,
            "yp_stage": Stage.S2,
        },
    )


def generate_synthetic_cohort(*, patient_count: int = 8, seed: int = 29) -> Cohort:
    """Create fictional data with edge cases, never a clinical-data substitute."""

    if patient_count < 6:
        raise ValueError("patient_count must be at least 6 to retain all edge cases")
    rng = random.Random(seed)
    patients: list[Patient] = []
    observations: list[Observation] = []
    measurements: list[ClinicalMeasurement] = []
    treatments: list[Treatment] = []
    outcomes: list[Outcome] = []
    queries: list[Query] = []

    for index in range(patient_count):
        patient_id = f"SYN-{index:04d}"
        no_surgery = index == 3
        missing_post_treatment_ct = index == 5
        patients.append(
            Patient(
                patient_id=patient_id,
                site_id=f"SYN-SITE-{index % 2}",
                cohort_id="synthetic-v1",
                eligibility_version="synthetic-baseline-v1",
                baseline_origin_local=0.0,
                index_event_type="synthetic_first_assessment",
            )
        )
        observations.append(
            Observation(
                observation_id=f"{patient_id}-ct0",
                patient_id=patient_id,
                modality=Modality.CT,
                role=ObservationRole.BASELINE_CT,
                source_type=SourceType.SYNTHETIC,
                acquired_at_days=SYNTHETIC_TIMELINE.baseline_acquired,
                available_at_days=SYNTHETIC_TIMELINE.s0_query,
                time_precision=TimePrecision.DAY,
                availability_basis=AvailabilityBasis.RECORDED_WITH_ORDER,
                local_asset_id=f"asset-{patient_id}-ct0",
                quality_status=QualityStatus.PASSED,
                phase="synthetic_portal_venous",
                feature_version="synthetic-v1",
            )
        )
        observations.append(
            Observation(
                observation_id=f"{patient_id}-ct1",
                patient_id=patient_id,
                modality=Modality.CT,
                role=ObservationRole.POST_TREATMENT_CT,
                source_type=SourceType.SYNTHETIC,
                acquired_at_days=SYNTHETIC_TIMELINE.ct1_acquired,
                available_at_days=SYNTHETIC_TIMELINE.ct1_available,
                time_precision=TimePrecision.DATETIME,
                availability_basis=AvailabilityBasis.RECORDED,
                local_asset_id=(
                    None if missing_post_treatment_ct else f"asset-{patient_id}-ct1"
                ),
                quality_status=(
                    QualityStatus.UNKNOWN
                    if missing_post_treatment_ct
                    else QualityStatus.PASSED
                ),
                missing_reason=(
                    MissingCategory.MISSING if missing_post_treatment_ct else None
                ),
                phase="synthetic_portal_venous",
                feature_version="synthetic-v1",
            )
        )
        measurements.extend(
            [
                ClinicalMeasurement(
                    measurement_id=f"{patient_id}-age",
                    patient_id=patient_id,
                    field_name="age_years",
                    typed_value=40 + index,
                    unit="years",
                    source_type=SourceType.SYNTHETIC,
                    acquired_at_days=SYNTHETIC_TIMELINE.baseline_acquired,
                    available_at_days=SYNTHETIC_TIMELINE.s0_query,
                    time_precision=TimePrecision.DAY,
                    availability_basis=AvailabilityBasis.RECORDED_WITH_ORDER,
                    provenance="synthetic_generator",
                ),
                ClinicalMeasurement(
                    measurement_id=f"{patient_id}-baseline-stage",
                    patient_id=patient_id,
                    field_name="baseline_stage",
                    typed_value=f"synthetic-cT{2 + index % 3}",
                    unit=None,
                    source_type=SourceType.SYNTHETIC,
                    acquired_at_days=SYNTHETIC_TIMELINE.baseline_acquired,
                    available_at_days=SYNTHETIC_TIMELINE.s0_query,
                    time_precision=TimePrecision.DAY,
                    availability_basis=AvailabilityBasis.RECORDED_WITH_ORDER,
                    provenance="synthetic_generator",
                ),
                ClinicalMeasurement(
                    measurement_id=f"{patient_id}-followup",
                    patient_id=patient_id,
                    field_name="followup_duration",
                    typed_value=100 + index,
                    unit="days",
                    source_type=SourceType.SYNTHETIC,
                    acquired_at_days=0.0,
                    available_at_days=0.0,
                    time_precision=TimePrecision.DAY,
                    availability_basis=AvailabilityBasis.RECORDED_WITH_ORDER,
                    provenance="synthetic_forbidden_field_test",
                ),
                ClinicalMeasurement(
                    measurement_id=f"{patient_id}-response",
                    patient_id=patient_id,
                    field_name="radiologic_response",
                    typed_value=float(rng.uniform(-1.0, 1.0)),
                    unit=None,
                    source_type=SourceType.SYNTHETIC,
                    acquired_at_days=SYNTHETIC_TIMELINE.ct1_acquired,
                    available_at_days=SYNTHETIC_TIMELINE.ct1_available,
                    time_precision=TimePrecision.DATETIME,
                    availability_basis=AvailabilityBasis.RECORDED,
                    provenance="synthetic_generator",
                ),
            ]
        )
        treatments.extend(
            [
                Treatment(
                    patient_id=patient_id,
                    event_id=f"{patient_id}-planned-systemic",
                    treatment_kind=TreatmentKind.SYSTEMIC,
                    standardized_components=("synthetic-agent-a",),
                    regimen_code="SYN-PLAN",
                    planned_or_delivered=TreatmentStatus.PLANNED,
                    start_days=5.0,
                    end_days=25.0,
                    available_at_days=0.0,
                    time_precision=TimePrecision.DAY,
                    availability_basis=AvailabilityBasis.RECORDED_WITH_ORDER,
                    data_granularity="summary",
                ),
                Treatment(
                    patient_id=patient_id,
                    event_id=f"{patient_id}-delivered-systemic",
                    treatment_kind=TreatmentKind.SYSTEMIC,
                    standardized_components=("synthetic-agent-a",),
                    regimen_code="SYN-DELIVERED",
                    planned_or_delivered=TreatmentStatus.DELIVERED,
                    start_days=5.0,
                    end_days=SYNTHETIC_TIMELINE.systemic_event,
                    available_at_days=SYNTHETIC_TIMELINE.systemic_available,
                    time_precision=TimePrecision.DATETIME,
                    availability_basis=AvailabilityBasis.RECORDED,
                    cycles_delivered_by_event=2,
                    data_granularity="summary",
                ),
            ]
        )
        if index == 0:
            treatments.append(
                Treatment(
                    patient_id=patient_id,
                    event_id=f"{patient_id}-post-ct-treatment",
                    treatment_kind=TreatmentKind.OTHER,
                    standardized_components=("synthetic-after-acquisition",),
                    regimen_code="SYN-LATE",
                    planned_or_delivered=TreatmentStatus.DELIVERED,
                    start_days=SYNTHETIC_TIMELINE.post_ct_event,
                    end_days=SYNTHETIC_TIMELINE.post_ct_event,
                    available_at_days=SYNTHETIC_TIMELINE.post_ct_available,
                    time_precision=TimePrecision.DATETIME,
                    availability_basis=AvailabilityBasis.RECORDED,
                    data_granularity="event",
                )
            )
        if not no_surgery:
            treatments.append(
                Treatment(
                    patient_id=patient_id,
                    event_id=f"{patient_id}-surgery",
                    treatment_kind=TreatmentKind.SURGERY,
                    standardized_components=(),
                    regimen_code=None,
                    planned_or_delivered=TreatmentStatus.DELIVERED,
                    start_days=SYNTHETIC_TIMELINE.surgery_event,
                    end_days=SYNTHETIC_TIMELINE.surgery_event,
                    available_at_days=SYNTHETIC_TIMELINE.surgery_available,
                    time_precision=TimePrecision.DATETIME,
                    availability_basis=AvailabilityBasis.RECORDED,
                    surgery_type="synthetic-gastrectomy",
                    intent_if_confirmed="synthetic-curative",
                )
            )
            measurements.append(
                ClinicalMeasurement(
                    measurement_id=f"{patient_id}-yp-stage",
                    patient_id=patient_id,
                    field_name="yp_stage",
                    typed_value=f"synthetic-ypT{index % 4}",
                    unit=None,
                    source_type=SourceType.SYNTHETIC,
                    acquired_at_days=SYNTHETIC_TIMELINE.pathology_acquired,
                    available_at_days=SYNTHETIC_TIMELINE.pathology_available,
                    time_precision=TimePrecision.DATETIME,
                    availability_basis=AvailabilityBasis.RESEARCH_ASSUMPTION,
                    provenance="synthetic_generator",
                )
            )
            if index == 2:
                observations.append(
                    Observation(
                        observation_id=f"{patient_id}-missing-slide",
                        patient_id=patient_id,
                        modality=Modality.PATHOLOGY,
                        role=ObservationRole.SURGICAL_PATHOLOGY,
                        source_type=SourceType.SYNTHETIC,
                        acquired_at_days=SYNTHETIC_TIMELINE.pathology_acquired,
                        available_at_days=SYNTHETIC_TIMELINE.pathology_available,
                        time_precision=TimePrecision.DATETIME,
                        availability_basis=AvailabilityBasis.RESEARCH_ASSUMPTION,
                        missing_reason=MissingCategory.MISSING,
                        tissue_source="synthetic-resection",
                    )
                )
            else:
                slide_count = 2 if index == 0 else 1
                for slide_index in range(slide_count):
                    failed = index == 4
                    observations.append(
                        Observation(
                            observation_id=f"{patient_id}-slide-{slide_index}",
                            patient_id=patient_id,
                            modality=Modality.PATHOLOGY,
                            role=ObservationRole.SURGICAL_PATHOLOGY,
                            source_type=SourceType.SYNTHETIC,
                            acquired_at_days=SYNTHETIC_TIMELINE.pathology_acquired,
                            available_at_days=SYNTHETIC_TIMELINE.pathology_available,
                            time_precision=TimePrecision.DATETIME,
                            availability_basis=AvailabilityBasis.RESEARCH_ASSUMPTION,
                            local_asset_id=(
                                None if failed else f"asset-{patient_id}-slide-{slide_index}"
                            ),
                            quality_status=(
                                QualityStatus.FAILED if failed else QualityStatus.PASSED
                            ),
                            missing_reason=(MissingCategory.FAILED_QC if failed else None),
                            tissue_source="synthetic-resection",
                            feature_version="synthetic-v1",
                        )
                    )

        event = index % 3 == 0
        event_time = 50.0 + index if event else None
        censor_time = 95.0 + index if not event else 80.0 + index
        outcomes.append(
            Outcome(
                patient_id=patient_id,
                endpoint_name="os",
                event_type=EventType.DEATH,
                source_status=1 if event else 0,
                event_date_days=event_time,
                censor_date_days=censor_time,
                adjudication_status=AdjudicationStatus.CONFIRMED,
                origin_definition="synthetic_baseline_day_zero",
                reason_of_censoring=None if event else "synthetic_end_of_followup",
                label_version="synthetic-os-v1",
            )
        )
        queries.extend(
            [
                Query(
                    f"{patient_id}-s0",
                    patient_id,
                    Stage.S0,
                    SYNTHETIC_TIMELINE.s0_query,
                    60.0,
                ),
                Query(
                    f"{patient_id}-s1",
                    patient_id,
                    Stage.S1,
                    SYNTHETIC_TIMELINE.s1_query,
                    60.0,
                ),
                Query(
                    f"{patient_id}-s2",
                    patient_id,
                    Stage.S2,
                    SYNTHETIC_TIMELINE.s2_query,
                    60.0,
                    eligibility=(
                        QueryEligibility.NOT_APPLICABLE
                        if no_surgery
                        else QueryEligibility.ELIGIBLE
                    ),
                    eligibility_basis="synthetic_surgery_path",
                ),
            ]
        )
        if index == 0:
            queries.append(Query(f"{patient_id}-after-event", patient_id, Stage.S2, 80.0, 30.0))

    issues = (
        DataIssue(
            patient_id="SYN-0005",
            code="invalid_date_rejected",
            source_field="synthetic_invalid_date",
            detail="A deliberately invalid raw date was rejected before typed cohort creation",
        ),
    )
    return Cohort(
        patients=tuple(patients),
        observations=tuple(observations),
        clinical_measurements=tuple(measurements),
        treatments=tuple(treatments),
        outcomes=tuple(outcomes),
        queries=tuple(queries),
        issues=issues,
    )
