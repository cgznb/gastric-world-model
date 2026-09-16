from __future__ import annotations

import pytest

from stageworld.data import (
    AdjudicationStatus,
    DataMode,
    EventType,
    FeatureFirewall,
    FeaturePolicy,
    LandmarkBuilder,
    MissingCategory,
    Outcome,
    OutcomeBuilder,
    OutcomeDefinition,
    Patient,
    Query,
    QueryEligibility,
    Stage,
    ZeroTimePolicy,
    default_synthetic_policy,
    generate_synthetic_cohort,
)
from stageworld.errors import ConfigurationError, DataContractError


def _definition(**overrides: object) -> OutcomeDefinition:
    values: dict[str, object] = {
        "endpoint_name": "os",
        "event_type": EventType.DEATH,
        "event_code": 1,
        "origin_definition": "synthetic_origin",
        "label_version": "v1",
        "mode": DataMode.SYNTHETIC,
        "status_mapping_confirmed": True,
        "origin_confirmed": True,
        "timeline_confirmed": True,
    }
    values.update(overrides)
    return OutcomeDefinition(**values)  # type: ignore[arg-type]


def _outcome(*, event: bool = True, event_time: float | None = 10.0) -> Outcome:
    return Outcome(
        "SYN-P",
        "os",
        EventType.DEATH,
        1 if event else 0,
        event_time if event else None,
        20.0,
        AdjudicationStatus.CONFIRMED,
        "synthetic_origin",
        None if event else "synthetic_censor",
        "v1",
    )


def _firewall() -> FeatureFirewall:
    patient = Patient("SYN-P", "SYN-SITE", "SYN", "v1", 0.0, "synthetic_origin")
    return FeatureFirewall((patient,), (), (), (), FeaturePolicy(mode=DataMode.SYNTHETIC))


def test_real_outcome_config_hard_fails_until_all_clinical_fields_confirmed() -> None:
    definition = OutcomeDefinition(
        endpoint_name="os",
        event_type=EventType.DEATH,
        event_code=None,
        origin_definition=None,
        label_version="unconfirmed",
        mode=DataMode.REAL_FEATURES,
    )
    with pytest.raises(ConfigurationError) as error:
        OutcomeBuilder((), definition)
    assert error.value.code == "endpoint_contract_unconfirmed"
    assert set(error.value.details["missing"]) >= {
        "event_code",
        "origin_definition",
        "status_mapping_confirmed",
        "origin_confirmed",
        "timeline_confirmed",
    }


def test_outcome_builder_is_independent_and_recomputes_remaining_time() -> None:
    builder = OutcomeBuilder((_outcome(),), _definition())
    label = builder.build_label("SYN-P", 4.25)
    assert label.remaining_time_days == pytest.approx(5.75)
    assert label.event is True
    assert not hasattr(builder, "observations")
    assert not hasattr(builder, "clinical_measurements")


def test_t18_landmark_builder_excludes_event_at_or_before_query_without_clipping() -> None:
    builder = LandmarkBuilder(_firewall(), OutcomeBuilder((_outcome(),), _definition()))
    queries = (
        Query("before", "SYN-P", Stage.S0, 5.0, 30.0),
        Query("same", "SYN-P", Stage.S1, 10.0, 30.0),
        Query("after", "SYN-P", Stage.S2, 11.0, 30.0),
    )
    result = builder.build(queries)
    assert len(result.landmarks) == 1
    assert result.landmarks[0].label.remaining_time_days == 5.0
    assert {item.reason for item in result.exclusions} == {
        "zero_followup_at_query",
        "event_already_occurred",
    }
    with pytest.raises(DataContractError) as error:
        builder.outcome_builder.build_label("SYN-P", 11.0)
    assert error.value.code == "negative_remaining_time"


def test_same_day_event_requires_explicit_policy() -> None:
    default_builder = OutcomeBuilder((_outcome(),), _definition())
    with pytest.raises(DataContractError) as error:
        default_builder.build_label("SYN-P", 10.0)
    assert error.value.code == "zero_remaining_time"
    allowed = OutcomeBuilder(
        (_outcome(),), _definition(zero_time_policy=ZeroTimePolicy.ALLOW_EVENT)
    )
    assert allowed.build_label("SYN-P", 10.0).remaining_time_days == 0.0


def test_status_date_conflict_is_not_silently_resolved() -> None:
    conflict = Outcome(
        "SYN-P",
        "os",
        EventType.DEATH,
        0,
        9.0,
        20.0,
        AdjudicationStatus.CONFIRMED,
        "synthetic_origin",
        "synthetic_censor",
        "v1",
    )
    with pytest.raises(DataContractError) as error:
        OutcomeBuilder((conflict,), _definition()).build_label("SYN-P", 1.0)
    assert error.value.code == "event_date_status_conflict"


def test_unknown_landmark_eligibility_fails_instead_of_using_future_completion() -> None:
    builder = LandmarkBuilder(_firewall(), OutcomeBuilder((_outcome(),), _definition()))
    query = Query(
        "unknown",
        "SYN-P",
        Stage.S2,
        5.0,
        30.0,
        eligibility=QueryEligibility.UNKNOWN,
    )
    with pytest.raises(DataContractError) as error:
        builder.build((query,))
    assert error.value.code == "query_eligibility_unknown"


def test_non_surgery_and_missing_slide_are_distinct_synthetic_paths() -> None:
    cohort = generate_synthetic_cohort()
    firewall = FeatureFirewall(
        cohort.patients,
        cohort.observations,
        cohort.clinical_measurements,
        cohort.treatments,
        default_synthetic_policy(),
    )
    outcomes = OutcomeBuilder(
        cohort.outcomes,
        OutcomeDefinition(
            "os",
            EventType.DEATH,
            1,
            "synthetic_baseline_day_zero",
            "synthetic-os-v1",
            DataMode.SYNTHETIC,
            True,
            True,
            True,
        ),
    )
    s2_queries = [query for query in cohort.queries if query.stage is Stage.S2]
    result = LandmarkBuilder(firewall, outcomes).build(s2_queries)

    non_surgery = next(item for item in s2_queries if item.patient_id == "SYN-0003")
    missing_slide = next(item for item in result.landmarks if item.patient_id == "SYN-0002")
    assert non_surgery.eligibility is QueryEligibility.NOT_APPLICABLE
    assert any(
        item.patient_id == non_surgery.patient_id and item.reason == "node_not_applicable"
        for item in result.exclusions
    )
    pathology = [
        item for item in missing_slide.prefix.observations if item.modality.value == "pathology"
    ]
    assert len(pathology) == 1
    assert pathology[0].missing_reason is MissingCategory.MISSING
