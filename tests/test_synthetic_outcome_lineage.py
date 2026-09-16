from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from stageworld.artifacts import atomic_write_json, read_json
from stageworld.config import StageWorldConfig, load_config
from stageworld.data import (
    DataMode,
    EventType,
    FeatureFirewall,
    LandmarkBuilder,
    OutcomeBuilder,
    OutcomeDefinition,
    Stage,
    default_synthetic_policy,
    load_cohort_json,
)
from stageworld.errors import ArtifactError
from stageworld.synthetic_workflow import (
    BUILD_SCHEMA,
    FEATURE_SCHEMA,
    SOURCE_SCHEMA,
    _atomic_torch_save,
    _load_tensor_artifact,
    build_synthetic_cohort,
    extract_synthetic_features,
    load_synthetic_batches,
    make_synthetic_artifacts,
    synthetic_data_root,
)

ROOT = Path(__file__).resolve().parents[1]
STAGES = (Stage.S0, Stage.S1, Stage.S2)


def _config(tmp_path: Path) -> StageWorldConfig:
    base = load_config(ROOT / "configs/project.synthetic.yaml")
    return replace(
        base,
        paths=replace(
            base.paths,
            output_root=str(tmp_path / "synthetic-run"),
            feature_root=str(tmp_path / "synthetic-run" / "features"),
        ),
    )


def _definition() -> OutcomeDefinition:
    return OutcomeDefinition(
        endpoint_name="os",
        event_type=EventType.DEATH,
        event_code=1,
        origin_definition="synthetic_baseline_day_zero",
        label_version="synthetic-os-v1",
        mode=DataMode.SYNTHETIC,
        status_mapping_confirmed=True,
        origin_confirmed=True,
        timeline_confirmed=True,
    )


def _cohort_payload(config: StageWorldConfig) -> tuple[Path, dict[str, Any]]:
    path = synthetic_data_root(config) / "cohort.json"
    return path, read_json(path)


def test_source_os_labels_equal_outcome_and_legal_landmark_builders(tmp_path: Path) -> None:
    config = _config(tmp_path)
    generated = make_synthetic_artifacts(config, patient_count=12)
    built = build_synthetic_cohort(config)
    root = synthetic_data_root(config)
    source = _load_tensor_artifact(root / "source_tensors.pt", SOURCE_SCHEMA)
    cohort = load_cohort_json(root / "cohort.json")
    firewall = FeatureFirewall(
        cohort.patients,
        cohort.observations,
        cohort.clinical_measurements,
        cohort.treatments,
        default_synthetic_policy(),
    )
    landmarks = LandmarkBuilder(firewall, OutcomeBuilder(cohort.outcomes, _definition())).build(
        cohort.queries
    )
    labels_by_query = {
        landmark.prefix.query.query_id: landmark.label for landmark in landmarks.landmarks
    }
    exclusions_by_query = {item.query_id: item.reason for item in landmarks.exclusions}
    durations = torch.as_tensor(source["survival_durations"])
    events = torch.as_tensor(source["survival_events"])
    valid = torch.as_tensor(source["survival_valid"])

    patient_ids = tuple(patient.patient_id for patient in cohort.patients)
    assert tuple(source["patient_ids"]) == patient_ids
    assert source["survival_stage_order"] == tuple(stage.value for stage in STAGES)
    assert source["survival_time_unit"] == "year"
    for row, patient_id in enumerate(patient_ids):
        for column, stage in enumerate(STAGES):
            query_id = f"{patient_id}-{stage.value}"
            label = labels_by_query.get(query_id)
            if label is None:
                assert query_id in exclusions_by_query
                assert not bool(valid[row, column])
                assert float(durations[row, column]) == 0.0
                assert int(events[row, column]) == 0
                continue
            assert bool(valid[row, column])
            assert float(durations[row, column]) * 365.25 == pytest.approx(
                label.remaining_time_days, abs=1e-5
            )
            assert bool(events[row, column]) is label.event

    assert exclusions_by_query["SYN-0003-s2"] == "node_not_applicable"
    assert exclusions_by_query["SYN-0000-after-event"] == "event_already_occurred"
    assert built["stage_landmark_counts"] == {
        stage.value: int(valid[:, column].sum()) for column, stage in enumerate(STAGES)
    }
    cohort_payload = read_json(root / "cohort.json")
    manifest = read_json(root / "manifest.json")
    cohort_artifact_id = generated["cohort_artifact_id"]
    assert source["cohort_artifact_id"] == cohort_artifact_id
    assert manifest["cohort_artifact_id"] == cohort_artifact_id
    assert cohort_payload["synthetic_lineage"]["cohort_artifact_id"] == cohort_artifact_id
    assert built["cohort_artifact_id"] == cohort_artifact_id
    assert source["outcome_contract"] == manifest["outcome_contract"]
    assert source["outcome_contract"] == cohort_payload["synthetic_lineage"]["outcome_contract"]
    assert source["outcome_contract"]["label_version"] == "synthetic-os-v1"


def test_changed_cohort_outcome_or_timeline_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    make_synthetic_artifacts(config, patient_count=12)
    cohort_path, payload = _cohort_payload(config)
    payload["outcomes"][0]["event_date_days"] += 1.0
    atomic_write_json(cohort_path, payload)
    with pytest.raises(ArtifactError) as outcome_error:
        build_synthetic_cohort(config)
    assert outcome_error.value.code == "SYNTHETIC_SURVIVAL_LABEL_MISMATCH"

    make_synthetic_artifacts(config, patient_count=12)
    cohort_path, payload = _cohort_payload(config)
    canonical_s1 = next(item for item in payload["queries"] if item["query_id"] == "SYN-0000-s1")
    canonical_s1["query_time_days"] += 1.0
    atomic_write_json(cohort_path, payload)
    with pytest.raises(ArtifactError) as timeline_error:
        build_synthetic_cohort(config)
    assert timeline_error.value.code == "SYNTHETIC_COHORT_TIMELINE_MISMATCH"


def test_cohort_and_source_linkage_tampering_hard_fails(tmp_path: Path) -> None:
    config = _config(tmp_path)
    make_synthetic_artifacts(config, patient_count=12)
    cohort_path, payload = _cohort_payload(config)
    payload["synthetic_lineage"]["cohort_artifact_id"] = "replaced-cohort-artifact-v1"
    atomic_write_json(cohort_path, payload)
    with pytest.raises(ArtifactError) as cohort_error:
        build_synthetic_cohort(config)
    assert cohort_error.value.code == "SYNTHETIC_COHORT_LINEAGE_MISMATCH"

    make_synthetic_artifacts(config, patient_count=12)
    source_path = synthetic_data_root(config) / "source_tensors.pt"
    source = _load_tensor_artifact(source_path, SOURCE_SCHEMA)
    source["cohort_artifact_id"] = "replaced-source-link-v1"
    _atomic_torch_save(source_path, source)
    with pytest.raises(ArtifactError) as source_error:
        build_synthetic_cohort(config)
    assert source_error.value.code == "SYNTHETIC_SOURCE_MANIFEST_MISMATCH"

    make_synthetic_artifacts(config, patient_count=12)
    cohort_path, payload = _cohort_payload(config)
    payload["synthetic_lineage"]["outcome_contract"]["label_version"] = "replaced-os-v2"
    atomic_write_json(cohort_path, payload)
    with pytest.raises(ArtifactError) as outcome_contract_error:
        build_synthetic_cohort(config)
    assert outcome_contract_error.value.code == "SYNTHETIC_COHORT_LINEAGE_MISMATCH"


def test_build_binding_and_legacy_source_schema_are_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    make_synthetic_artifacts(config, patient_count=12)
    build_synthetic_cohort(config)
    extract_synthetic_features(config, "ct")
    extract_synthetic_features(config, "pathology")
    build_path = synthetic_data_root(config) / "cohort_build.json"
    build_payload = read_json(build_path)
    assert build_payload["schema_version"] == BUILD_SCHEMA
    build_payload["cohort_artifact_id"] = "replaced-build-cohort-v1"
    atomic_write_json(build_path, build_payload)
    with pytest.raises(ArtifactError) as build_error:
        load_synthetic_batches(config)
    assert build_error.value.code == "COHORT_BUILD_COHORT_LINEAGE_MISMATCH"

    build_synthetic_cohort(config)
    build_payload = read_json(build_path)
    build_payload["outcome_contract"]["label_version"] = "replaced-os-v2"
    atomic_write_json(build_path, build_payload)
    with pytest.raises(ArtifactError) as contract_error:
        load_synthetic_batches(config)
    assert contract_error.value.code == "COHORT_BUILD_CONTRACT_LINEAGE_MISMATCH"

    build_synthetic_cohort(config)
    extract_synthetic_features(config, "ct")
    feature_path = Path(config.paths.feature_root or "") / "ct.pt"
    feature = _load_tensor_artifact(feature_path, FEATURE_SCHEMA)
    feature["outcome_contract_version"] = "replaced-os-contract-v2"
    _atomic_torch_save(feature_path, feature)
    with pytest.raises(ArtifactError) as feature_error:
        load_synthetic_batches(config)
    assert feature_error.value.code == "FEATURE_DATA_LINEAGE_MISMATCH"

    source_path = synthetic_data_root(config) / "source_tensors.pt"
    source = _load_tensor_artifact(source_path, SOURCE_SCHEMA)
    source["schema_version"] = "stageworld-synthetic-source-v2"
    _atomic_torch_save(source_path, source)
    with pytest.raises(ArtifactError) as schema_error:
        _load_tensor_artifact(source_path, SOURCE_SCHEMA)
    assert schema_error.value.code == "SYNTHETIC_ARTIFACT_SCHEMA_MISMATCH"


def test_extra_legal_query_cannot_change_canonical_build_counts(tmp_path: Path) -> None:
    config = _config(tmp_path)
    make_synthetic_artifacts(config, patient_count=12)
    cohort_path, payload = _cohort_payload(config)
    extra = dict(next(item for item in payload["queries"] if item["query_id"] == "SYN-0001-s1"))
    extra["query_id"] = "SYN-0001-extra-s1"
    payload["queries"].append(extra)
    atomic_write_json(cohort_path, payload)

    with pytest.raises(ArtifactError) as error:
        build_synthetic_cohort(config)
    assert error.value.code == "SYNTHETIC_COHORT_QUERY_UNIVERSE_MISMATCH"


def test_split_assignments_are_recomputed_not_trusted_by_version(tmp_path: Path) -> None:
    config = _config(tmp_path)
    make_synthetic_artifacts(config, patient_count=12)
    build_synthetic_cohort(config)
    extract_synthetic_features(config, "ct")
    extract_synthetic_features(config, "pathology")
    build_path = synthetic_data_root(config) / "cohort_build.json"
    build = read_json(build_path)
    original_version = build["split_version"]
    first_split = build["assignments"][0]["split"]
    build["assignments"][0]["split"] = "test" if first_split != "test" else "train"
    assert build["split_version"] == original_version
    atomic_write_json(build_path, build)

    with pytest.raises(ArtifactError) as error:
        load_synthetic_batches(config)
    assert error.value.code == "COHORT_SPLIT_POLICY_MISMATCH"
    assert "assignments" in error.value.details["fields"]


def test_feature_rows_remain_bound_to_ordered_patient_ids(tmp_path: Path) -> None:
    config = _config(tmp_path)
    make_synthetic_artifacts(config, patient_count=12)
    build_synthetic_cohort(config)
    extract_synthetic_features(config, "ct")
    extract_synthetic_features(config, "pathology")
    feature_path = Path(config.paths.feature_root or "") / "ct.pt"
    feature = _load_tensor_artifact(feature_path, FEATURE_SCHEMA)
    feature["patient_ids"] = tuple(reversed(feature["patient_ids"]))
    for name in ("ct0", "ct1"):
        token_payload = feature[name]
        for field in (
            "values",
            "valid",
            "modality",
            "acquired_time",
            "available_time",
            "coords",
        ):
            value = token_payload[field]
            if isinstance(value, torch.Tensor):
                token_payload[field] = value.flip(0)
        token_payload["source_id"] = list(reversed(token_payload["source_id"]))
        token_payload["quality_flags"] = list(reversed(token_payload["quality_flags"]))
    _atomic_torch_save(feature_path, feature)

    with pytest.raises(ArtifactError) as error:
        load_synthetic_batches(config)
    assert error.value.code == "SYNTHETIC_FEATURE_ARTIFACT_MISMATCH"
    assert any("patient_ids" in field for field in error.value.details["fields"])
