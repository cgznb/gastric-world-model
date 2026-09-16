from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace

import pytest
import torch

from stageworld.config import RunMode
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
    QualityStatus,
    Query,
    SourceType,
    Stage,
    TimePrecision,
    Treatment,
    TreatmentKind,
    TreatmentStatus,
)
from stageworld.encoders import EncoderProvenance, ObservationTokens
from stageworld.errors import ArtifactError, DataContractError
from stageworld.inference import (
    CHECKPOINT_SCHEMA_VERSION,
    FEATURE_INPUT_SCHEMA_VERSION,
    ActionFeature,
    CheckpointContract,
    FeatureInputContract,
    FeatureManifest,
    InferenceEngine,
    InferenceStateCache,
    ModalityFeatureContract,
    Scenario,
    ScenarioKind,
)
from stageworld.model import ActionTokens, StageWorldModel, StageWorldModelConfig


def _provenance(modality: str, dim: int, *, version: str = "weights-v1") -> EncoderProvenance:
    return EncoderProvenance(
        encoder_name=f"synthetic_{modality}_encoder",
        source_version=version,
        component_versions=(("weights", version),),
        preprocess_version="synthetic-preprocess-v1",
        feature_dim=dim,
    )


PROVENANCE = {
    "ct": _provenance("ct", 4),
    "pathology": _provenance("pathology", 6),
    "clinical": _provenance("clinical", 3),
}


def _model(
    *,
    model_version: str = "unit-model-v1",
    timeline_time_unit: str = "day",
) -> StageWorldModel:
    torch.manual_seed(11)
    return StageWorldModel(
        StageWorldModelConfig(
            hidden_dim=8,
            state_tokens=2,
            stochastic_dim=2,
            use_stochastic_state=False,
            attention_heads=2,
            transition_blocks=1,
            observation_blocks=1,
            resampler_blocks=1,
            dropout=0.0,
            action_input_dim=2,
            modality_input_dims=(("ct", 4), ("pathology", 6), ("clinical", 3)),
            resampled_tokens=(("ct", 2), ("pathology", 2), ("clinical", 1)),
            future_output_dims=(("ct", 4), ("pathology", 6)),
            future_output_tokens=(("ct", 1), ("pathology", 1)),
            survival_cutpoints=(0.0, 1.0, 3.0),
            max_rollout_days=1000.0,
            timeline_time_unit=timeline_time_unit,
            model_version=model_version,
        )
    ).eval()


def _checkpoint(**changes: object) -> CheckpointContract:
    model_config = asdict(_model().config)
    survival_contract: dict[str, object] = {
        "schema_version": "stageworld-survival-contract-v1",
        "endpoint": "os",
        "parameterization": "piecewise_constant_hazard_rate",
        "time_unit": "year",
        "cutpoints": (0.0, 1.0, 3.0),
        "open_tail_interval": True,
        "num_causes": 1,
    }
    values: dict[str, object] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_id": "synthetic-joint-step-12",
        "weight_version": "synthetic-weights-step-12",
        "model_version": "unit-model-v1",
        "model_config": model_config,
        "endpoint": "os",
        "survival_contract": survival_contract,
        "mode": "synthetic",
        "config_lineage_id": "synthetic-config-v1",
        "data_lineage_id": "synthetic-data-v1",
        "cohort_artifact_id": "synthetic-cohort-artifact-v1",
        "split_version": "synthetic-split-v1",
        "ct_feature_artifact_id": "synthetic-ct-features-v1",
        "pathology_feature_artifact_id": "synthetic-pathology-features-v1",
        "timeline_contract_version": "synthetic-timeline-contract-v1",
        "outcome_contract_version": "synthetic-outcome-contract-v1",
        "training_seed": 17,
        "source_schema_version": "stageworld-synthetic-source-v2",
        "cohort_schema_version": "stageworld-cohort-build-v1",
        "feature_schema_version": "stageworld-synthetic-features-v1",
        "phase": "joint_survival",
        "step": 12,
        "parent_checkpoint_id": "synthetic-world-pretrain-step-12",
        "parent_weight_version": "synthetic-world-pretrain-weights-step-12",
        "parent_phase": "world_pretrain",
        "parent_config_lineage_id": "synthetic-config-v1",
        "parent_data_lineage_id": "synthetic-data-v1",
        "parent_cohort_artifact_id": "synthetic-cohort-artifact-v1",
        "parent_split_version": "synthetic-split-v1",
        "parent_ct_feature_artifact_id": "synthetic-ct-features-v1",
        "parent_pathology_feature_artifact_id": "synthetic-pathology-features-v1",
        "parent_timeline_contract_version": "synthetic-timeline-contract-v1",
        "parent_outcome_contract_version": "synthetic-outcome-contract-v1",
    }
    values.update(changes)
    if "model_version" in changes and "model_config" not in changes:
        values["model_config"] = {**model_config, "model_version": changes["model_version"]}
    if "endpoint" in changes and "survival_contract" not in changes:
        values["survival_contract"] = {
            **survival_contract,
            "endpoint": changes["endpoint"],
        }
    return CheckpointContract.from_mapping(values)


def _feature_contract() -> FeatureInputContract:
    return FeatureInputContract(
        schema_version=FEATURE_INPUT_SCHEMA_VERSION,
        modalities=tuple(
            ModalityFeatureContract(
                modality=name,
                provenance=provenance,
                record_feature_version="record-feature-v1" if name != "clinical" else None,
            )
            for name, provenance in PROVENANCE.items()
        ),
        action_feature_version="action-feature-v1",
    )


def _patient(patient_id: str) -> Patient:
    return Patient(
        patient_id=patient_id,
        site_id="SYN-SITE",
        cohort_id="synthetic-inference-v1",
        eligibility_version="synthetic-eligibility-v1",
        baseline_origin_local=0.0,
        index_event_type="synthetic-index",
    )


def _observation(
    patient_id: str,
    suffix: str,
    modality: Modality,
    role: ObservationRole,
    acquired: float,
    available: float,
) -> Observation:
    return Observation(
        observation_id=f"{patient_id}-{suffix}",
        patient_id=patient_id,
        modality=modality,
        role=role,
        source_type=SourceType.SYNTHETIC,
        acquired_at_days=acquired,
        available_at_days=available,
        time_precision=TimePrecision.DATETIME,
        availability_basis=AvailabilityBasis.RECORDED,
        local_asset_id=f"synthetic-asset-{patient_id}-{suffix}",
        quality_status=QualityStatus.PASSED,
        feature_version="record-feature-v1",
    )


def _clinical(patient_id: str) -> ClinicalMeasurement:
    return ClinicalMeasurement(
        measurement_id=f"{patient_id}-age",
        patient_id=patient_id,
        field_name="age_years",
        typed_value=60,
        unit="years",
        source_type=SourceType.SYNTHETIC,
        acquired_at_days=-1.0,
        available_at_days=0.0,
        time_precision=TimePrecision.DATETIME,
        availability_basis=AvailabilityBasis.RECORDED,
        provenance="synthetic-generator-v1",
    )


def _treatments(patient_id: str) -> tuple[Treatment, Treatment]:
    return (
        Treatment(
            patient_id=patient_id,
            event_id=f"{patient_id}-nac",
            treatment_kind=TreatmentKind.CHEMOTHERAPY,
            standardized_components=("synthetic-a",),
            regimen_code="synthetic-regimen",
            planned_or_delivered=TreatmentStatus.DELIVERED,
            start_days=1.0,
            end_days=9.0,
            available_at_days=9.0,
            time_precision=TimePrecision.DATETIME,
            availability_basis=AvailabilityBasis.RECORDED,
        ),
        Treatment(
            patient_id=patient_id,
            event_id=f"{patient_id}-surgery",
            treatment_kind=TreatmentKind.SURGERY,
            standardized_components=(),
            regimen_code=None,
            planned_or_delivered=TreatmentStatus.DELIVERED,
            start_days=20.0,
            end_days=20.0,
            available_at_days=20.0,
            time_precision=TimePrecision.DATETIME,
            availability_basis=AvailabilityBasis.RECORDED,
        ),
    )


def _records(
    patient_id: str,
) -> tuple[tuple[Observation, ...], tuple[ClinicalMeasurement, ...], tuple[Treatment, ...]]:
    return (
        (
            _observation(
                patient_id,
                "ct0",
                Modality.CT,
                ObservationRole.BASELINE_CT,
                -1.0,
                0.0,
            ),
            _observation(
                patient_id,
                "ct1",
                Modality.CT,
                ObservationRole.POST_TREATMENT_CT,
                10.0,
                12.0,
            ),
            _observation(
                patient_id,
                "path",
                Modality.PATHOLOGY,
                ObservationRole.SURGICAL_PATHOLOGY,
                20.0,
                22.0,
            ),
        ),
        (_clinical(patient_id),),
        _treatments(patient_id),
    )


def _tokens(
    record_id: str,
    modality: str,
    acquired: float,
    available: float,
    *,
    offset: float = 0.0,
    provenance: EncoderProvenance | None = None,
) -> ObservationTokens:
    dimensions = {"ct": 4, "pathology": 6, "clinical": 3}
    modality_ids = {"ct": 0, "pathology": 1, "clinical": 2}
    dimension = dimensions[modality]
    values = torch.arange(1, dimension + 1, dtype=torch.float32).reshape(1, 1, dimension)
    return ObservationTokens(
        values=values + offset,
        valid=torch.ones(1, 1, dtype=torch.bool),
        modality=torch.full((1, 1), modality_ids[modality], dtype=torch.long),
        acquired_time=torch.full((1, 1), acquired),
        available_time=torch.full((1, 1), available),
        provenance=provenance or PROVENANCE[modality],
        source_id=((f"source-{record_id}",),),
        modality_name=modality,
    )


def _manifest(patient_id: str, *, offset: float = 0.0, **changes: object) -> FeatureManifest:
    observations, _, treatments = _records(patient_id)
    values: dict[str, object] = {
        "schema_version": FEATURE_INPUT_SCHEMA_VERSION,
        "manifest_lineage_id": f"manifest-{patient_id}-v1",
        "patient_id": patient_id,
        "mode": RunMode.SYNTHETIC,
        "observation_features": {
            observations[0].observation_id: _tokens(
                observations[0].observation_id, "ct", -1.0, 0.0, offset=offset
            ),
            observations[1].observation_id: _tokens(
                observations[1].observation_id, "ct", 10.0, 12.0, offset=offset + 2.0
            ),
            observations[2].observation_id: _tokens(
                observations[2].observation_id,
                "pathology",
                20.0,
                22.0,
                offset=offset + 4.0,
            ),
        },
        "clinical_features": {
            f"{patient_id}-age": _tokens(f"{patient_id}-age", "clinical", -1.0, 0.0, offset=offset)
        },
        "action_features": {
            treatments[0].event_id: ActionFeature(
                torch.tensor([1.0 + offset, 0.5]), "action-feature-v1", 1.0
            ),
            treatments[1].event_id: ActionFeature(
                torch.tensor([0.2, 1.0 + offset]), "action-feature-v1", 1.0
            ),
        },
    }
    values.update(changes)
    return FeatureManifest(**values)  # type: ignore[arg-type]


def _firewall(patient_ids: tuple[str, ...]) -> FeatureFirewall:
    all_observations: list[Observation] = []
    all_measurements: list[ClinicalMeasurement] = []
    all_treatments: list[Treatment] = []
    for patient_id in patient_ids:
        observations, measurements, treatments = _records(patient_id)
        all_observations.extend(observations)
        all_measurements.extend(measurements)
        all_treatments.extend(treatments)
    return FeatureFirewall(
        tuple(_patient(patient_id) for patient_id in patient_ids),
        all_observations,
        all_measurements,
        all_treatments,
        FeaturePolicy(
            mode=DataMode.SYNTHETIC,
            clinical_min_stage={"age_years": Stage.S0},
        ),
    )


def _queries(patient_id: str) -> tuple[Query, ...]:
    return (
        Query("q-s0", patient_id, Stage.S0, 0.0, 365.25),
        Query("q-s1-before-report", patient_id, Stage.S1, 11.0, 365.25),
        Query("q-s1-after-report", patient_id, Stage.S1, 12.0, 365.25),
        Query("q-s2", patient_id, Stage.S2, 22.0, 365.25),
    )


def _engine(*, cache: InferenceStateCache | None = None) -> InferenceEngine:
    return InferenceEngine(
        _model(),
        _checkpoint(),
        _feature_contract(),
        cache=cache,
        survival_time_unit="year",
    )


def test_checkpoint_requires_schema_weight_lineage_and_compatible_model() -> None:
    assert CHECKPOINT_SCHEMA_VERSION == "stageworld-checkpoint-v4"
    model_config = asdict(_model().config)
    base = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_id": "weights-v1",
        "weight_version": "weight-state-v1",
        "model_version": "unit-model-v1",
        "model_config": model_config,
        "endpoint": "os",
        "survival_contract": {
            "schema_version": "stageworld-survival-contract-v1",
            "endpoint": "os",
            "parameterization": "piecewise_constant_hazard_rate",
            "time_unit": "year",
            "cutpoints": (0.0, 1.0, 3.0),
            "open_tail_interval": True,
            "num_causes": 1,
        },
        "mode": "synthetic",
        "config_lineage_id": "config-v1",
        "data_lineage_id": "data-v1",
        "cohort_artifact_id": "cohort-artifact-v1",
        "split_version": "split-v1",
        "ct_feature_artifact_id": "ct-features-v1",
        "pathology_feature_artifact_id": "pathology-features-v1",
        "timeline_contract_version": "timeline-contract-v1",
        "outcome_contract_version": "outcome-contract-v1",
        "training_seed": 17,
        "source_schema_version": "source-v1",
        "cohort_schema_version": "cohort-v1",
        "feature_schema_version": "features-v1",
        "phase": "joint_survival",
        "step": 1,
        "parent_checkpoint_id": "parent-checkpoint-v1",
        "parent_weight_version": "parent-weight-state-v1",
        "parent_phase": "world_pretrain",
        "parent_config_lineage_id": "config-v1",
        "parent_data_lineage_id": "data-v1",
        "parent_cohort_artifact_id": "cohort-artifact-v1",
        "parent_split_version": "split-v1",
        "parent_ct_feature_artifact_id": "ct-features-v1",
        "parent_pathology_feature_artifact_id": "pathology-features-v1",
        "parent_timeline_contract_version": "timeline-contract-v1",
        "parent_outcome_contract_version": "outcome-contract-v1",
    }
    for missing in (
        "schema_version",
        "checkpoint_id",
        "weight_version",
        "model_version",
        "model_config",
        "survival_contract",
        "cohort_artifact_id",
        "ct_feature_artifact_id",
        "pathology_feature_artifact_id",
        "timeline_contract_version",
        "outcome_contract_version",
        "training_seed",
        "source_schema_version",
        "cohort_schema_version",
        "feature_schema_version",
    ):
        invalid = dict(base)
        invalid.pop(missing)
        with pytest.raises(ArtifactError) as error:
            CheckpointContract.from_mapping(invalid)
        assert error.value.code == "MISSING_CHECKPOINT_VERSION"

    with pytest.raises(ArtifactError) as error:
        CheckpointContract.from_mapping({**base, "schema_version": "legacy-v0"})
    assert error.value.code == "CHECKPOINT_SCHEMA_MISMATCH"

    for field, value in (
        ("ct_feature_artifact_id", None),
        ("pathology_feature_artifact_id", 17),
        ("config_lineage_id", "  "),
    ):
        with pytest.raises(ArtifactError) as error:
            CheckpointContract.from_mapping({**base, field: value})
        assert error.value.code == "MISSING_CHECKPOINT_VERSION"

    with pytest.raises(ArtifactError) as error:
        CheckpointContract.from_mapping({**base, "phase": "fine_tune"})
    assert error.value.code == "INVALID_CHECKPOINT_PHASE"

    for field in ("parent_ct_feature_artifact_id", "parent_pathology_feature_artifact_id"):
        with pytest.raises(ArtifactError) as error:
            CheckpointContract.from_mapping({**base, field: None})
        assert error.value.code == "INCOMPLETE_PARENT_CHECKPOINT_LINEAGE"

    with pytest.raises(ArtifactError) as error:
        InferenceEngine(_model(), _checkpoint(model_version="different-model"), _feature_contract())
    assert error.value.code == "MODEL_VERSION_MISMATCH"

    with pytest.raises(ArtifactError) as error:
        InferenceEngine(_model(), _checkpoint(endpoint="rfs"), _feature_contract())
    assert error.value.code == "CHECKPOINT_ENDPOINT_MISMATCH"


def test_checkpoint_contract_rejects_internal_model_and_survival_mismatch() -> None:
    model_config = asdict(_model().config)
    with pytest.raises(ArtifactError) as error:
        _checkpoint(
            model_config={**model_config, "survival_cutpoints": (0.0, 2.0, 3.0)}
        )
    assert error.value.code == "CHECKPOINT_SURVIVAL_CONTRACT_MISMATCH"

    with pytest.raises(ArtifactError) as error:
        _checkpoint(model_config={**model_config, "model_version": "other-model-v1"})
    assert error.value.code == "CHECKPOINT_MODEL_CONFIG_MISMATCH"


def test_inference_rejects_current_config_masquerade_and_time_unit_override() -> None:
    changed_model = StageWorldModel(replace(_model().config, max_rollout_days=999.0))
    with pytest.raises(ArtifactError) as error:
        InferenceEngine(changed_model, _checkpoint(), _feature_contract())
    assert error.value.code == "MODEL_CONFIG_MISMATCH"

    with pytest.raises(ArtifactError) as error:
        InferenceEngine(
            _model(),
            _checkpoint(),
            _feature_contract(),
            survival_time_unit="day",
        )
    assert error.value.code == "SURVIVAL_CONTRACT_MISMATCH"


def test_mode_schema_endpoint_and_visible_feature_versions_are_hard_gates() -> None:
    patient_id = "ANON-GATE"
    query = _queries(patient_id)[0]
    firewall = _firewall((patient_id,))
    engine = _engine()

    with pytest.raises(ArtifactError) as error:
        engine.predict_query(
            firewall,
            _manifest(patient_id, mode=RunMode.REAL_FEATURES),
            query,
        )
    assert error.value.code == "INFERENCE_MODE_MISMATCH"

    with pytest.raises(ArtifactError) as error:
        engine.predict_query(
            firewall,
            _manifest(patient_id, schema_version="legacy-feature-schema"),
            query,
        )
    assert error.value.code == "FEATURE_SCHEMA_MISMATCH"

    with pytest.raises(ArtifactError) as error:
        engine.predict_query(firewall, _manifest(patient_id), query, endpoint="dfs")
    assert error.value.code == "ENDPOINT_VERSION_MISMATCH"

    manifest = _manifest(patient_id)
    bad_ct0 = _tokens(
        f"{patient_id}-ct0",
        "ct",
        -1.0,
        0.0,
        provenance=_provenance("ct", 4, version="other-weights-v2"),
    )
    bad_features = dict(manifest.observation_features)
    bad_features[f"{patient_id}-ct0"] = bad_ct0
    with pytest.raises(ArtifactError) as error:
        engine.predict_query(
            firewall,
            replace(
                manifest,
                manifest_lineage_id="manifest-bad-visible-version",
                observation_features=bad_features,
            ),
            query,
        )
    assert error.value.code == "FEATURE_VERSION_MISMATCH"


def test_future_feature_entries_are_not_resolved_before_their_available_at() -> None:
    patient_id = "ANON-LATE"
    firewall = _firewall((patient_id,))
    manifest = _manifest(patient_id)
    # The late CT entry is intentionally incompatible. It must be invisible at day 11.
    feature_values = dict(manifest.observation_features)
    feature_values[f"{patient_id}-ct1"] = _tokens(
        f"{patient_id}-ct1",
        "ct",
        10.0,
        12.0,
        provenance=_provenance("ct", 4, version="wrong-future-version"),
    )
    incompatible_future = replace(
        manifest,
        manifest_lineage_id="manifest-with-unavailable-incompatible-future",
        observation_features=feature_values,
    )

    before = _engine().predict_query(firewall, incompatible_future, _queries(patient_id)[1])
    assert all(item.record_id != f"{patient_id}-ct1" for item in before.input_manifest)

    with pytest.raises(ArtifactError) as error:
        _engine().predict_query(firewall, incompatible_future, _queries(patient_id)[2])
    assert error.value.code == "FEATURE_VERSION_MISMATCH"


def test_history_replay_applies_delayed_ct_and_pathology_only_after_arrival() -> None:
    patient_id = "ANON-HISTORY"
    firewall = _firewall((patient_id,))
    manifest = _manifest(patient_id)
    engine = _engine()
    queries = _queries(patient_id)

    results = engine.predict_patient_history(firewall, manifest, queries)
    ids = [{item.record_id for item in result.input_manifest} for result in results]
    assert f"{patient_id}-ct1" not in ids[1]
    assert f"{patient_id}-ct1" in ids[2]
    assert f"{patient_id}-path" not in ids[2]
    assert f"{patient_id}-path" in ids[3]
    assert "delayed_post_treatment_ct_update" not in results[1].quality_flags
    assert "delayed_post_treatment_ct_update" in results[2].quality_flags

    before = engine.replay_query(firewall, manifest, queries[1]).state.memory
    after = engine.replay_query(firewall, manifest, queries[2]).state.memory
    assert not torch.allclose(before, after)


def test_history_replay_is_equivalent_for_day_and_year_model_clocks() -> None:
    patient_id = "SYN-UNIT-CLOCK"
    firewall = _firewall((patient_id,))
    manifest = _manifest(patient_id)
    query = _queries(patient_id)[3]
    day_model = _model(timeline_time_unit="day")
    year_model = _model(timeline_time_unit="year")
    year_model.load_state_dict(day_model.state_dict())
    day_engine = InferenceEngine(
        day_model,
        _checkpoint(model_config=asdict(day_model.config)),
        _feature_contract(),
    )
    year_engine = InferenceEngine(
        year_model,
        _checkpoint(model_config=asdict(year_model.config)),
        _feature_contract(),
    )

    day_replay = day_engine.replay_query(firewall, manifest, query)
    year_replay = year_engine.replay_query(firewall, manifest, query)

    assert torch.allclose(day_replay.state.memory, year_replay.state.memory, atol=2e-6)
    assert torch.allclose(
        day_replay.state.query_time,
        year_replay.state.query_time * year_model.config.days_per_year,
        atol=1e-5,
    )
    day_prediction = day_engine.predict_query(firewall, manifest, query)
    year_prediction = year_engine.predict_query(firewall, manifest, query)
    assert torch.allclose(
        torch.tensor(day_prediction.risk),
        torch.tensor(year_prediction.risk),
        atol=2e-6,
    )


def test_history_replay_consumes_each_action_once_at_its_replay_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patient_id = "SYN-UNIT-ACTION-ONCE"
    model = _model()
    original_predict_prior = model.predict_prior
    action_calls: list[tuple[float, tuple[str, ...]]] = []

    def record_predict_prior(
        state: object,
        actions: ActionTokens,
        target_time: torch.Tensor,
        **kwargs: object,
    ) -> object:
        if actions.valid.any():
            action_calls.append((float(target_time.item()), actions.provenance))
        return original_predict_prior(  # type: ignore[arg-type]
            state, actions, target_time, **kwargs
        )

    monkeypatch.setattr(model, "predict_prior", record_predict_prior)
    engine = InferenceEngine(model, _checkpoint(), _feature_contract())
    engine.replay_query(
        _firewall((patient_id,)),
        _manifest(patient_id),
        _queries(patient_id)[3],
    )

    flattened = [item for _, provenance in action_calls for item in provenance]
    assert len(flattened) == 2
    assert len(set(flattened)) == 2
    assert [target for target, _ in action_calls] == [9.0, 20.0]


@pytest.mark.parametrize(
    ("missing_reason", "quality_status"),
    (
        (MissingCategory.MISSING, QualityStatus.UNKNOWN),
        (MissingCategory.FAILED_QC, QualityStatus.FAILED),
    ),
)
def test_missing_observation_preserves_known_clocks_without_posterior_update(
    monkeypatch: pytest.MonkeyPatch,
    missing_reason: MissingCategory,
    quality_status: QualityStatus,
) -> None:
    patient_id = "SYN-UNIT-MISSING-BOUNDARY"
    observations, measurements, treatments = _records(patient_id)
    missing_ct = replace(
        observations[1],
        local_asset_id=None,
        quality_status=quality_status,
        missing_reason=missing_reason,
    )
    firewall = FeatureFirewall(
        (_patient(patient_id),),
        (observations[0], missing_ct, observations[2]),
        measurements,
        treatments,
        FeaturePolicy(mode=DataMode.SYNTHETIC, clinical_min_stage={"age_years": Stage.S0}),
    )
    model = _model()
    original_predict_prior = model.predict_prior
    original_update_posterior = model.update_posterior
    transitions: list[tuple[float, int]] = []
    posterior_updates: list[float] = []

    def record_predict_prior(
        state: object,
        actions: ActionTokens,
        target_time: torch.Tensor,
        **kwargs: object,
    ) -> object:
        transitions.append((float(target_time.item()), int(actions.valid.sum().item())))
        return original_predict_prior(  # type: ignore[arg-type]
            state, actions, target_time, **kwargs
        )

    def record_update_posterior(
        state: object,
        observations: object,
        availability_time: torch.Tensor,
        **kwargs: object,
    ) -> object:
        posterior_updates.append(float(availability_time.item()))
        return original_update_posterior(  # type: ignore[arg-type]
            state, observations, availability_time, **kwargs
        )

    monkeypatch.setattr(model, "predict_prior", record_predict_prior)
    monkeypatch.setattr(model, "update_posterior", record_update_posterior)
    engine = InferenceEngine(model, _checkpoint(), _feature_contract())
    replay = engine.replay_query(
        firewall,
        _manifest(patient_id),
        _queries(patient_id)[2],
    )

    assert transitions == [(9.0, 1), (10.0, 0), (12.0, 0)]
    assert posterior_updates == []
    assert "observation_unavailable" in replay.quality_flags
    assert ("observation_failed_qc" in replay.quality_flags) == (
        quality_status is QualityStatus.FAILED
    )


def test_untyped_clinical_null_does_not_create_an_observation_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patient_id = "SYN-UNIT-UNTYPED-NULL"
    observations, measurements, _ = _records(patient_id)
    untyped_null = ClinicalMeasurement(
        measurement_id=f"{patient_id}-untyped-null",
        patient_id=patient_id,
        field_name="stage1_null",
        typed_value=None,
        unit=None,
        source_type=SourceType.SYNTHETIC,
        acquired_at_days=10.0,
        available_at_days=12.0,
        time_precision=TimePrecision.DATETIME,
        availability_basis=AvailabilityBasis.RECORDED,
        provenance="synthetic-untyped-null",
    )
    firewall = FeatureFirewall(
        (_patient(patient_id),),
        (observations[0],),
        (*measurements, untyped_null),
        (),
        FeaturePolicy(
            mode=DataMode.SYNTHETIC,
            clinical_min_stage={"age_years": Stage.S0, "stage1_null": Stage.S1},
        ),
    )
    model = _model()
    original_predict_prior = model.predict_prior
    transitions: list[float] = []

    def record_predict_prior(
        state: object,
        actions: ActionTokens,
        target_time: torch.Tensor,
        **kwargs: object,
    ) -> object:
        transitions.append(float(target_time.item()))
        return original_predict_prior(  # type: ignore[arg-type]
            state, actions, target_time, **kwargs
        )

    monkeypatch.setattr(model, "predict_prior", record_predict_prior)
    replay = InferenceEngine(model, _checkpoint(), _feature_contract()).replay_query(
        firewall,
        _manifest(patient_id),
        Query("q-untyped-null", patient_id, Stage.S1, 12.0, 365.25),
    )

    assert transitions == [12.0]
    assert "clinical_measurement_unavailable" in replay.quality_flags


def test_t34_s2_cache_cannot_be_reused_for_s0_query() -> None:
    patient_id = "ANON-CACHE"
    firewall = _firewall((patient_id,))
    manifest = _manifest(patient_id)
    cache = InferenceStateCache(max_entries=8)
    engine = _engine(cache=cache)
    queries = _queries(patient_id)

    s2 = engine.replay_query(firewall, manifest, queries[3])
    s0 = engine.replay_query(firewall, manifest, queries[0])
    assert not s2.cache_hit
    assert not s0.cache_hit
    assert s2.cache_key != s0.cache_key
    assert s2.cache_key.prefix != s0.cache_key.prefix
    assert s2.cache_key.query_time_days == 22.0
    assert s0.cache_key.query_time_days == 0.0
    assert len(cache) == 2
    assert f"source-{patient_id}-path" in s2.state.provenance
    assert f"source-{patient_id}-path" not in s0.state.provenance

    repeated = engine.replay_query(firewall, manifest, queries[0])
    assert repeated.cache_hit
    repeated.state.memory.add_(1000.0)
    cached_again = engine.replay_query(firewall, manifest, queries[0])
    assert not torch.equal(repeated.state.memory, cached_again.state.memory)


def test_cache_isolated_by_exact_weight_version_after_resume() -> None:
    patient_id = "ANON-WEIGHT-CACHE"
    firewall = _firewall((patient_id,))
    manifest = _manifest(patient_id)
    cache = InferenceStateCache(max_entries=8)
    first = _engine(cache=cache)
    query = _queries(patient_id)[0]

    initial = first.replay_query(firewall, manifest, query)
    assert not initial.cache_hit
    repeated = first.replay_query(firewall, manifest, query)
    assert repeated.cache_hit

    resumed_weights = InferenceEngine(
        _model(),
        _checkpoint(weight_version="synthetic-weights-after-resume"),
        _feature_contract(),
        cache=cache,
        survival_time_unit="year",
    )
    refreshed = resumed_weights.replay_query(firewall, manifest, query)
    assert not refreshed.cache_hit
    assert refreshed.cache_key.checkpoint_id == initial.cache_key.checkpoint_id
    assert refreshed.cache_key.weight_version != initial.cache_key.weight_version


def test_concurrent_anonymous_patients_remain_isolated() -> None:
    patient_ids = ("ANON-CONCURRENT-A", "ANON-CONCURRENT-B")
    firewall = _firewall(patient_ids)
    manifests = (
        _manifest(patient_ids[0], offset=0.0),
        _manifest(patient_ids[1], offset=7.0),
    )
    cache = InferenceStateCache(max_entries=8)
    engine = _engine(cache=cache)

    def run(index: int):
        patient_id = patient_ids[index]
        return engine.predict_query(firewall, manifests[index], _queries(patient_id)[3])

    with ThreadPoolExecutor(max_workers=2) as executor:
        outputs = tuple(executor.map(run, (0, 1)))

    for patient_id, output in zip(patient_ids, outputs, strict=True):
        assert output.patient_id == patient_id
        assert all(item.record_id.startswith(patient_id) for item in output.input_manifest)
    assert {key.patient_id for key in cache.keys()} == set(patient_ids)
    assert outputs[0].risk != outputs[1].risk


def test_prediction_contract_is_serializable_and_scenario_is_explicit() -> None:
    patient_id = "ANON-OUTPUT"
    firewall = _firewall((patient_id,))
    manifest = _manifest(patient_id)
    engine = _engine()
    query = _queries(patient_id)[0]
    original = manifest.observation_features[f"{patient_id}-ct0"].values.clone()

    output = engine.predict_query(
        firewall,
        manifest,
        query,
        horizons_days=(30.0, 365.25),
    )
    payload = output.as_dict()
    json.dumps(payload)
    assert payload["schema_version"] == "stageworld-prediction-v1"
    assert payload["scenario"] == {
        "kind": "observed_history",
        "label": "observed_history",
        "assumptions": [],
    }
    assert payload["simulated"] is False
    assert payload["endpoint"] == "os"
    assert payload["horizon_days"] == [0.0, 30.0, 365.25]
    assert payload["survival"][0] == pytest.approx(1.0)
    assert payload["risk"][0] == pytest.approx(0.0)
    assert output.model_version == "unit-model-v1"
    assert output.checkpoint_id == "synthetic-joint-step-12"
    assert output.weight_version == "synthetic-weights-step-12"
    assert payload["weight_version"] == "synthetic-weights-step-12"
    assert torch.equal(original, manifest.observation_features[f"{patient_id}-ct0"].values)

    replay = engine.replay_query(firewall, manifest, query)
    scenario = Scenario(
        kind=ScenarioKind.PREDICTED_OBSERVATION,
        label="synthetic_ct_feature_projection",
        assumptions=("declared_synthetic_scenario",),
    )
    future = engine.predict_future_observation(replay.state, "ct", scenario)
    assert future.provenance == "predicted_not_observed"
    assert future.scenario.label == "synthetic_ct_feature_projection"
    assert future.simulated

    with pytest.raises(DataContractError) as error:
        engine.predict_future_observation(replay.state, "ct", Scenario.observed_history())
    assert error.value.code == "FUTURE_SCENARIO_LABEL_REQUIRED"


def test_t09_future_scenario_rollout_does_not_change_observed_s0_prediction() -> None:
    patient_id = "ANON-SIMULATION"
    firewall = _firewall((patient_id,))
    manifest = _manifest(patient_id)
    engine = _engine()
    query = _queries(patient_id)[0]
    observed_before = engine.predict_query(firewall, manifest, query)
    replay = engine.replay_query(firewall, manifest, query)
    observed_memory = replay.state.memory.clone()
    observed_time = replay.state.query_time.clone()
    scenario = Scenario(
        kind=ScenarioKind.HYPOTHETICAL,
        label="declared_future_ct_scenario",
        assumptions=("not_a_causal_treatment_effect",),
    )

    current = engine.predict_future_observation(replay.state, "ct", scenario)
    future_without_actions = engine.predict_future_observation(
        replay.state,
        "ct",
        scenario,
        target_time_days=20.0,
    )
    hypothetical_actions = ActionTokens(
        values=torch.tensor([[[1.0, -0.5]]]),
        valid=torch.ones(1, 1, dtype=torch.bool),
        event_time=torch.tensor([[10.0]]),
        available_time=torch.tensor([[10.0]]),
        event_type=torch.tensor([[1]]),
        planned_or_delivered=torch.tensor([[1]]),
        known_exposure=torch.tensor([[1.0]]),
        provenance=("declared-hypothetical-action",),
    )
    future_with_actions = engine.predict_future_observation(
        replay.state,
        "ct",
        scenario,
        target_time_days=20.0,
        actions=hypothetical_actions,
    )

    assert current.target_time_days == pytest.approx(0.0)
    assert future_without_actions.target_time_days == pytest.approx(20.0)
    assert future_with_actions.target_time_days == pytest.approx(20.0)
    assert not torch.equal(current.mean, future_without_actions.mean)
    assert not torch.equal(future_without_actions.mean, future_with_actions.mean)
    assert torch.equal(replay.state.memory, observed_memory)
    assert torch.equal(replay.state.query_time, observed_time)
    assert engine.predict_query(firewall, manifest, query).as_dict() == observed_before.as_dict()


def test_future_scenario_rollout_uses_existing_time_and_action_gates() -> None:
    patient_id = "ANON-SIMULATION-GATES"
    firewall = _firewall((patient_id,))
    manifest = _manifest(patient_id)
    engine = _engine()
    replay = engine.replay_query(firewall, manifest, _queries(patient_id)[0])
    scenario = Scenario(kind=ScenarioKind.PREDICTED_OBSERVATION, label="future_ct")

    with pytest.raises(DataContractError) as backward:
        engine.predict_future_observation(
            replay.state,
            "ct",
            scenario,
            target_time_days=-1.0,
        )
    assert backward.value.code == "BACKWARD_TRANSITION"

    unavailable_action = ActionTokens(
        values=torch.ones(1, 1, 2),
        valid=torch.ones(1, 1, dtype=torch.bool),
        event_time=torch.tensor([[30.0]]),
        available_time=torch.tensor([[30.0]]),
    )
    with pytest.raises(DataContractError) as future_action:
        engine.predict_future_observation(
            replay.state,
            "ct",
            scenario,
            target_time_days=20.0,
            actions=unavailable_action,
        )
    assert future_action.value.code == "FUTURE_ACTION_IN_PRIOR"
