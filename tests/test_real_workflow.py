from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from stageworld.artifacts import atomic_write_private_json, read_json
from stageworld.cache import CacheProvenance, FeatureCache
from stageworld.config import (
    ClinicalSettings,
    EncoderSettings,
    ModelSettings,
    PathSettings,
    PermissionSettings,
    ProjectSettings,
    RunMode,
    StageWorldConfig,
)
from stageworld.data.paired_ct import (
    FIRST_ACQUISITION_SELECTION_POLICY_VERSION,
    PAIRED_CT_COHORT_SCHEMA,
    PRIVATE_ASSET_SCHEMA,
    PRIVATE_SPLIT_SCHEMA,
)
from stageworld.encoders import (
    SWINUNETR_COMPONENT_VERSION,
    SWINUNETR_FEATURE_DIM,
    SWINUNETR_PREPROCESS_VERSION,
    SWINUNETR_SOURCE_VERSION,
    EncoderProvenance,
    ObservationTokens,
)
from stageworld.errors import ArtifactError
from stageworld.real_workflow import (
    REAL_CT_FEATURE_SCHEMA,
    _select_feature_bindings,
    load_real_world_pretrain_batches,
)


def _config(tmp_path: Path) -> StageWorldConfig:
    weight = tmp_path / "weights.pt"
    weight.write_bytes(b"contract-only-test")
    return StageWorldConfig(
        project=ProjectSettings(name="real-contract-test", mode=RunMode.REAL_IMAGES),
        paths=PathSettings(
            approved_data_root=str(tmp_path),
            feature_root=str(tmp_path / "features"),
            output_root=str(tmp_path / "output"),
        ),
        permissions=PermissionSettings(approved_weight_licenses=("Apache-2.0",)),
        encoders=EncoderSettings(
            ct_weight_path=str(weight),
            ct_source_version=SWINUNETR_SOURCE_VERSION,
            ct_component_version=SWINUNETR_COMPONENT_VERSION,
            ct_preprocess_version=SWINUNETR_PREPROCESS_VERSION,
            ct_inference_precision="off",
        ),
        clinical=ClinicalSettings(
            development_stages=("s0", "s1"),
            ct_series_selection_policy_version=FIRST_ACQUISITION_SELECTION_POLICY_VERSION,
        ),
        model=ModelSettings(
            ct_encoder="swinunetr",
            pathology_encoder="disabled_for_s0_s1",
            pathology_alternatives=(),
            pathology_tokens=0,
            ct_input_dim=SWINUNETR_FEATURE_DIM,
        ),
    )


def test_real_feature_cache_can_be_outside_read_only_input_root(tmp_path: Path) -> None:
    config = _config(tmp_path)
    approved_input = tmp_path / "approved-input"
    approved_input.mkdir()
    external_cache = tmp_path / "private-cache"
    config = replace(
        config,
        paths=replace(
            config.paths,
            approved_data_root=str(approved_input),
            ct_root=str(approved_input / "ct"),
            feature_root=str(external_cache),
        ),
    )

    config.validate(command="extract-features")


def _encoder_provenance() -> EncoderProvenance:
    return EncoderProvenance(
        encoder_name="swinunetr",
        source_version=SWINUNETR_SOURCE_VERSION,
        component_versions=(("swinunetr", SWINUNETR_COMPONENT_VERSION),),
        preprocess_version=SWINUNETR_PREPROCESS_VERSION,
        feature_dim=SWINUNETR_FEATURE_DIM,
    )


def _cache_provenance() -> CacheProvenance:
    return CacheProvenance.from_encoder(
        _encoder_provenance(),
        schema_version=REAL_CT_FEATURE_SCHEMA,
        patch_sampling_version="deterministic-center-crop-96-v1",
        split_version="fixed-split-v1",
        target_transform_version="spatial-mean-768-v1",
        teacher_version=SWINUNETR_COMPONENT_VERSION,
    )


def _tokens(asset_id: str, *, post_treatment: bool) -> ObservationTokens:
    observation_time = 10.0 if post_treatment else 0.0
    return ObservationTokens(
        values=torch.full((1, 1, SWINUNETR_FEATURE_DIM), observation_time + 1.0),
        valid=torch.ones(1, 1, dtype=torch.bool),
        modality=torch.zeros(1, 1, dtype=torch.long),
        acquired_time=torch.full((1, 1), observation_time),
        available_time=torch.full((1, 1), observation_time),
        provenance=_encoder_provenance(),
        source_id=((asset_id,),),
        modality_name="ct",
    )


def _write_real_feature_fixture(config: StageWorldConfig) -> None:
    restricted = config.output_root / "data" / "restricted"
    lineage = {"data_lineage_id": "data-v1", "cohort_artifact_id": "cohort-v1"}
    patients = (("train-patient", "train"), ("validation-patient", "validation"))
    queries = [
        {"patient_id": patient_id, "stage": stage, "query_time_days": time}
        for patient_id, _ in patients
        for stage, time in (("s0", 0.0), ("s1", 10.0))
    ]
    atomic_write_private_json(
        restricted / "input_cohort.json",
        {
            **lineage,
            "private_schema_version": PAIRED_CT_COHORT_SCHEMA,
            "queries": queries,
        },
    )
    atomic_write_private_json(
        restricted / "asset_bindings.json",
        {
            **lineage,
            "schema_version": PRIVATE_ASSET_SCHEMA,
            "phase_selection_policy_version": FIRST_ACQUISITION_SELECTION_POLICY_VERSION,
            "bindings": [],
        },
    )
    atomic_write_private_json(
        restricted / "split_assignments.json",
        {
            **lineage,
            "schema_version": PRIVATE_SPLIT_SCHEMA,
            "split_version": "fixed-split-v1",
            "assignments": [
                {"patient_id": patient_id, "split": split} for patient_id, split in patients
            ]
            + [{"patient_id": "held-out-patient", "split": "test"}],
        },
    )

    provenance = _cache_provenance()
    cache = FeatureCache(Path(config.paths.feature_root or "") / "ct_entries")
    entries = []
    for patient_id, split in patients:
        for role in ("baseline_ct", "post_treatment_ct"):
            asset_id = f"{patient_id}-{role}"
            cache.begin(asset_id, provenance)
            cache.store(
                asset_id,
                provenance,
                _tokens(asset_id, post_treatment=role == "post_treatment_ct"),
            )
            entries.append(
                {
                    "patient_id": patient_id,
                    "split": split,
                    "role": role,
                    "cache_entry_id": asset_id,
                }
            )
    atomic_write_private_json(
        Path(config.paths.feature_root or "") / "ct_manifest.json",
        {
            **lineage,
            "schema_version": REAL_CT_FEATURE_SCHEMA,
            "feature_artifact_id": "features-v1",
            "encoder_provenance": _encoder_provenance().as_dict(),
            "cache_provenance": provenance.as_dict(),
            "entries": entries,
            "outcome_data_read": False,
            "test_features_included": False,
        },
    )


def test_feature_selection_excludes_test_before_pixel_access() -> None:
    splits = {
        "assignments": [
            {"patient_id": patient, "split": split}
            for patient, split in (
                ("train-p", "train"),
                ("val-p", "validation"),
                ("test-p", "test"),
            )
        ]
    }
    assets = {
        "bindings": [
            {"patient_id": patient, "role": role}
            for patient in ("train-p", "val-p", "test-p")
            for role in ("baseline_ct", "post_treatment_ct")
        ]
    }

    selected = _select_feature_bindings(
        assets,
        splits,
        limit_per_split=None,
        include_test=False,
    )

    assert {split for _, split in selected} == {"train", "validation"}
    assert all(binding["patient_id"] != "test-p" for binding, _ in selected)


def test_real_world_batches_do_not_read_outcomes_or_include_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _write_real_feature_fixture(config)
    opened: list[str] = []

    def audited_read(path: str | Path) -> dict[str, object]:
        opened.append(Path(path).name)
        return read_json(path)

    monkeypatch.setattr("stageworld.real_workflow.read_json", audited_read)

    bundle = load_real_world_pretrain_batches(config)

    assert "outcomes.json" not in opened
    assert bundle.training_patient_count == 1
    assert bundle.validation_patient_count == 1
    assert bundle.held_out_test_patient_count == 1
    batches = tuple(bundle.batches_by_split["train"] + bundle.batches_by_split["validation"])
    assert all("held-out-patient" not in batch.patient_ids for batch in batches)
    assert all(not batch.survival_valid.any() for batch in batches)
    assert all(not batch.survival_events.any() for batch in batches)


def test_world_pretraining_rejects_manifest_built_with_test_features(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_real_feature_fixture(config)
    manifest_path = Path(config.paths.feature_root or "") / "ct_manifest.json"
    manifest = read_json(manifest_path)
    manifest["test_features_included"] = True
    atomic_write_private_json(manifest_path, manifest)

    with pytest.raises(ArtifactError) as error:
        load_real_world_pretrain_batches(config)
    assert error.value.code == "WORLD_PRETRAIN_TEST_ISOLATION_FAILED"
