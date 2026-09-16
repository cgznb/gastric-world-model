from __future__ import annotations

from dataclasses import replace

import pytest

from stageworld.data import (
    AvailabilityBasis,
    ClinicalMeasurement,
    DataMode,
    FeatureFirewall,
    FeaturePolicy,
    MissingCategory,
    Modality,
    Observation,
    ObservationRole,
    Patient,
    Query,
    SameDayPolicy,
    SourceType,
    Stage,
    TimePrecision,
    Treatment,
    TreatmentKind,
    TreatmentStatus,
)
from stageworld.errors import DataContractError


def _patient() -> Patient:
    return Patient("SYN-P", "SYN-SITE", "SYN", "v1", 0.0, "synthetic_origin")


def _observation(
    observation_id: str,
    role: ObservationRole,
    acquired: float,
    available: float | None,
    *,
    precision: TimePrecision = TimePrecision.DATETIME,
    basis: AvailabilityBasis = AvailabilityBasis.RECORDED,
    asset: str | None = "synthetic-asset",
    missing: MissingCategory | None = None,
) -> Observation:
    modality = Modality.PATHOLOGY if role is ObservationRole.SURGICAL_PATHOLOGY else Modality.CT
    return Observation(
        observation_id,
        "SYN-P",
        modality,
        role,
        SourceType.SYNTHETIC,
        acquired,
        available,
        precision,
        basis,
        local_asset_id=asset,
        missing_reason=missing,
    )


def _treatment(
    event_id: str,
    kind: TreatmentKind,
    status: TreatmentStatus,
    start: float,
    end: float,
    available: float,
    components: tuple[str, ...],
) -> Treatment:
    return Treatment(
        "SYN-P",
        event_id,
        kind,
        components,
        None,
        status,
        start,
        end,
        available,
        TimePrecision.DATETIME,
        AvailabilityBasis.RECORDED,
    )


def _policy(*, same_day: SameDayPolicy = SameDayPolicy.CONSERVATIVE_EXCLUDE) -> FeaturePolicy:
    return FeaturePolicy(mode=DataMode.SYNTHETIC, same_day_policy=same_day)


def _firewall(
    observations: tuple[Observation, ...], treatments: tuple[Treatment, ...] = ()
) -> FeatureFirewall:
    return FeatureFirewall((_patient(),), observations, (), treatments, _policy())


def test_t02_s0_prefix_is_invariant_to_future_records() -> None:
    baseline = _observation("ct0", ObservationRole.BASELINE_CT, -1.0, -0.5)
    future_ct = _observation("ct1", ObservationRole.POST_TREATMENT_CT, 30.0, 32.0)
    future_path = _observation(
        "path",
        ObservationRole.SURGICAL_PATHOLOGY,
        40.0,
        None,
        asset=None,
        missing=MissingCategory.UNKNOWN,
    )
    delivered = _treatment(
        "delivered", TreatmentKind.SYSTEMIC, TreatmentStatus.DELIVERED, 5.0, 25.0, 26.0, ("a",)
    )
    surgery = _treatment(
        "surgery", TreatmentKind.SURGERY, TreatmentStatus.DELIVERED, 40.0, 40.0, 41.0, ()
    )
    query = Query("q0", "SYN-P", Stage.S0, 0.0, 30.0)

    prefix_a = _firewall((baseline, future_ct, future_path), (delivered, surgery)).build_prefix(
        "SYN-P", query
    )
    prefix_b = _firewall(
        (baseline, replace(future_ct, local_asset_id="changed")),
        (replace(delivered, standardized_components=("changed",)),),
    ).build_prefix("SYN-P", query)

    assert prefix_a == prefix_b
    assert [item.observation_id for item in prefix_a.observations] == ["ct0"]
    assert not prefix_a.treatments


def test_t03_s1_prefix_is_invariant_to_surgery_and_pathology() -> None:
    baseline = _observation("ct0", ObservationRole.BASELINE_CT, -1.0, -0.5)
    ct1 = _observation("ct1", ObservationRole.POST_TREATMENT_CT, 30.0, 31.0)
    path = _observation("path", ObservationRole.SURGICAL_PATHOLOGY, 40.0, 42.0)
    treatment = _treatment(
        "delivered", TreatmentKind.SYSTEMIC, TreatmentStatus.DELIVERED, 5.0, 25.0, 26.0, ("a",)
    )
    surgery = _treatment(
        "surgery", TreatmentKind.SURGERY, TreatmentStatus.DELIVERED, 40.0, 40.0, 41.0, ()
    )
    query = Query("q1", "SYN-P", Stage.S1, 35.0, 30.0)
    prefix_a = _firewall((baseline, ct1, path), (treatment, surgery)).build_prefix("SYN-P", query)
    prefix_b = _firewall(
        (baseline, ct1, replace(path, local_asset_id="changed")), (treatment,)
    ).build_prefix("SYN-P", query)

    assert prefix_a == prefix_b
    assert {item.observation_id for item in prefix_a.observations} == {"ct0", "ct1"}
    assert [item.event_id for item in prefix_a.treatments] == ["delivered"]


def test_t04_future_roles_are_structurally_absent_from_s0() -> None:
    future = _observation("path", ObservationRole.SURGICAL_PATHOLOGY, 40.0, 42.0)
    prefix = _firewall((future,)).build_prefix("SYN-P", Query("q", "SYN-P", Stage.S0, 0.0, 30.0))
    assert prefix.observations == ()
    assert prefix.exclusions == ()
    assert not hasattr(prefix, "outcomes")


def test_t05_current_legal_observation_changes_prefix() -> None:
    first = _observation("ct0", ObservationRole.BASELINE_CT, -1.0, -0.5)
    second = replace(first, local_asset_id="different-current-input")
    query = Query("q", "SYN-P", Stage.S0, 0.0, 30.0)
    assert _firewall((first,)).build_prefix("SYN-P", query) != _firewall((second,)).build_prefix(
        "SYN-P", query
    )


def test_t06_delayed_observation_appears_only_at_48_hour_boundary() -> None:
    delayed = _observation("ct0", ObservationRole.BASELINE_CT, 0.0, 2.0)
    firewall = _firewall((delayed,))
    at_24_hours = firewall.build_prefix("SYN-P", Query("q24", "SYN-P", Stage.S0, 1.0, 30.0))
    at_48_hours = firewall.build_prefix("SYN-P", Query("q48", "SYN-P", Stage.S0, 2.0, 30.0))
    assert at_24_hours.observations == ()
    assert at_24_hours.exclusions == ()
    assert [item.observation_id for item in at_48_hours.observations] == ["ct0"]


def test_t07_unknown_availability_never_defaults_to_zero() -> None:
    unknown = _observation(
        "ct0", ObservationRole.BASELINE_CT, -1.0, None, asset="asset", missing=None
    )
    with pytest.raises(DataContractError, match="unknown available_at") as error:
        _firewall((unknown,)).build_prefix("SYN-P", Query("q", "SYN-P", Stage.S0, 0.0, 30.0))
    assert error.value.code == "available_at_missing"


def test_t07_same_day_ambiguity_rejects_or_excludes_conservatively() -> None:
    same_day = _observation(
        "ct0",
        ObservationRole.BASELINE_CT,
        -1.0,
        0.0,
        precision=TimePrecision.DAY,
        basis=AvailabilityBasis.RECORDED,
    )
    query = Query("q", "SYN-P", Stage.S0, 0.0, 30.0)
    conservative = _firewall((same_day,)).build_prefix("SYN-P", query)
    assert conservative.observations == ()
    assert conservative.exclusions == ()
    rejecting = FeatureFirewall(
        (_patient(),), (same_day,), (), (), _policy(same_day=SameDayPolicy.REJECT)
    )
    with pytest.raises(DataContractError) as error:
        rejecting.build_prefix("SYN-P", query)
    assert error.value.code == "same_day_order_ambiguous"


def test_t08_s0_eligibility_and_payload_ignore_future_modality_existence() -> None:
    baseline = _observation("ct0", ObservationRole.BASELINE_CT, -1.0, -0.5)
    future_missing = _observation(
        "path",
        ObservationRole.SURGICAL_PATHOLOGY,
        40.0,
        None,
        asset=None,
        missing=MissingCategory.NOT_PERFORMED,
    )
    query = Query("q", "SYN-P", Stage.S0, 0.0, 30.0)
    with_future = _firewall((baseline, future_missing)).build_prefix("SYN-P", query)
    without_future = _firewall((baseline,)).build_prefix("SYN-P", query)
    assert with_future == without_future
    assert with_future.patient.baseline_eligible


def test_t09_simulation_target_time_is_removed_from_main_prefix() -> None:
    baseline = _observation("ct0", ObservationRole.BASELINE_CT, -1.0, -0.5)
    firewall = _firewall((baseline,))
    first = firewall.build_prefix(
        "SYN-P", Query("q", "SYN-P", Stage.S0, 0.0, 30.0, target_time_days=20.0)
    )
    second = firewall.build_prefix(
        "SYN-P", Query("q", "SYN-P", Stage.S0, 0.0, 30.0, target_time_days=80.0)
    )
    assert first == second
    assert first.query.target_time_days is None


def test_t10_ct_target_context_excludes_post_acquisition_treatment() -> None:
    baseline = _observation("ct0", ObservationRole.BASELINE_CT, -1.0, -0.5)
    target = _observation("ct1", ObservationRole.POST_TREATMENT_CT, 30.0, 32.0)
    before = _treatment(
        "before", TreatmentKind.SYSTEMIC, TreatmentStatus.DELIVERED, 5.0, 25.0, 26.0, ("a",)
    )
    after = _treatment(
        "after", TreatmentKind.OTHER, TreatmentStatus.DELIVERED, 31.0, 31.0, 31.0, ("b",)
    )
    firewall = _firewall((baseline, target), (before, after))
    source_query = Query("q", "SYN-P", Stage.S0, 0.0, 30.0)
    target_context = firewall.build_ct_target_context(target, source_query)
    clinical_context = firewall.build_prefix("SYN-P", Query("q1", "SYN-P", Stage.S1, 32.0, 30.0))

    assert [item.event_id for item in target_context.treatments] == ["before"]
    assert {item.event_id for item in clinical_context.treatments} == {"before", "after"}
    assert [item.observation_id for item in target_context.observations] == ["ct0"]
    assert {item.observation_id for item in clinical_context.observations} == {"ct0", "ct1"}


def test_outcome_proxy_field_is_blocked_case_insensitively() -> None:
    measurement = ClinicalMeasurement(
        "m",
        "SYN-P",
        "Survival_Time",
        99,
        "days",
        SourceType.SYNTHETIC,
        -1.0,
        -0.5,
        TimePrecision.DATETIME,
        AvailabilityBasis.RECORDED,
    )
    policy = FeaturePolicy(
        mode=DataMode.SYNTHETIC,
        clinical_min_stage={"Survival_Time": Stage.S0},
    )
    firewall = FeatureFirewall((_patient(),), (), (measurement,), (), policy)
    prefix = firewall.build_prefix("SYN-P", Query("q", "SYN-P", Stage.S0, 0.0, 30.0))
    assert prefix.clinical_measurements == ()


def test_real_mode_rejects_synthetic_observations() -> None:
    baseline = _observation("ct0", ObservationRole.BASELINE_CT, -1.0, -0.5)
    firewall = FeatureFirewall(
        (_patient(),),
        (baseline,),
        (),
        (),
        FeaturePolicy(mode=DataMode.REAL_FEATURES),
    )
    with pytest.raises(DataContractError) as error:
        firewall.build_prefix("SYN-P", Query("q", "SYN-P", Stage.S0, 0.0, 30.0))
    assert error.value.code == "source_type_not_allowed"
