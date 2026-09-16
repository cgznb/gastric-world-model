import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from stageworld.data.tumor_roi import (
    TUMOR_LABELS,
    TUMOR_MODEL_FILES,
    bind_tumor_model,
    gastric_tumor_candidates,
    tumor_field_of_view,
    validate_tumor_model,
)
from stageworld.errors import ArtifactError, DataContractError


def test_candidates_are_tumor_voxels_and_keep_invasion_outside_stomach():
    labels = np.zeros((60, 60, 60), np.uint8)
    labels[10:20, 10:20, 10:20] = 11
    labels[20:40, 12:15, 12:15] = 14
    labels[50:52, 50:52, 50:52] = 14
    candidate, qc = gastric_tumor_candidates(labels, np.diag([2.0, 2.0, 2.0, 1]))
    assert candidate.sum() == 20 * 3 * 3
    assert not candidate[labels != 14].any()
    assert candidate[39, 13, 13] == 1
    assert qc["pan_cancer_component_count"] == 2
    assert qc["gastric_candidate_component_count"] == 1
    assert qc["primary_tumor_identity_confirmed"] is False


def test_physical_distance_and_oblique_rotation_not_voxel_distance():
    labels = np.zeros((50, 50, 50), np.uint8)
    labels[10:15, 10:15, 10:15] = 11
    labels[20, 12, 12] = 14
    affine = np.diag([2.0, 1.0, 1.0, 1.0])
    assert gastric_tumor_candidates(labels, affine)[0].sum() == 0
    affine[0, 0] = 1.0
    assert gastric_tumor_candidates(labels, affine)[0].sum() == 1
    rotation = np.eye(4)
    rotation[:2, :2] = [[0, -1], [1, 0]]
    assert gastric_tumor_candidates(labels, rotation @ affine)[0].sum() == 1


def test_stomach_boundary_and_small_multifocal_candidates_are_not_organ_qc_exclusions():
    labels = np.zeros((50, 50, 50), np.uint8)
    labels[:20, 10:20, 10:20] = 11
    labels[21, 11, 11] = 14
    labels[21, 18, 18] = 14
    mask, qc = gastric_tumor_candidates(labels, np.eye(4))
    assert qc["gastric_candidate_component_count"] == 2
    _, crop_qc = tumor_field_of_view(mask, np.eye(4))
    assert crop_qc["tumor_volume_ml"] == pytest.approx(0.002)


def test_empty_prediction_is_unresolved_not_complete_response():
    labels = np.zeros((20, 20, 20), np.uint8)
    labels[5:10, 5:10, 5:10] = 11
    mask, _ = gastric_tumor_candidates(labels, np.eye(4))
    with pytest.raises(DataContractError) as error:
        tumor_field_of_view(mask, np.eye(4))
    assert error.value.code == "TUMOR_NOT_DETECTED_REVIEW_REQUIRED"


def test_tumor_margin_preserves_physical_extent():
    mask = np.zeros((100, 100, 120), np.uint8)
    mask[20:40, 30:50, 15:95] = 1
    affine = np.diag([3.0, -3.0, 3.0, 1.0])
    target, qc = tumor_field_of_view(mask, affine)
    assert qc["field_of_view_mm"] == 280
    corners = np.array([[19.5, 29.5, 14.5, 1], [39.5, 49.5, 94.5, 1]]) @ affine.T
    low = target[:3, 3] - 0.5 * np.diag(target)[:3]
    assert np.all(corners[:, :3].min(0) - low >= 20 - 1e-5)
    assert np.all(low + 280 - corners[:, :3].max(0) >= 20 - 1e-5)


def test_missing_or_whole_stomach_model_rejected_and_cache_cannot_change_model(tmp_path):
    model = tmp_path / "model"
    with pytest.raises(ArtifactError):
        validate_tumor_model(model)
    for name in TUMOR_MODEL_FILES:
        path = model / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    (model / "dataset.json").write_text(
        json.dumps(
            {
                "channel_names": {"0": "CT"},
                "labels": {"background": 0, "stomach": 1},
                "file_ending": ".nii.gz",
            }
        )
    )
    (model / "plans.json").write_text(json.dumps({"configurations": {"3d_fullres": {}}}))
    with pytest.raises(ArtifactError) as error:
        validate_tumor_model(model)
    assert error.value.code == "TUMOR_SEGMENTER_TASK_MISMATCH"
    dataset = json.loads((model / "dataset.json").read_text())
    dataset["labels"] = TUMOR_LABELS
    (model / "dataset.json").write_text(json.dumps(dataset))
    first = bind_tumor_model(model, tmp_path / "cache", create=True)
    assert bind_tumor_model(model, tmp_path / "cache", create=False) == first
    (model / TUMOR_MODEL_FILES[-1]).write_text("changed")
    with pytest.raises(ArtifactError) as error:
        bind_tumor_model(model, tmp_path / "cache", create=False)
    assert error.value.code == "TUMOR_MODEL_BINDING_MISMATCH"


def test_candidate_protocol_preserves_world_model_but_rejects_stomach_cache(tmp_path):
    from stageworld.config import load_config
    from stageworld.data.gastric_roi import GASTRIC_ROI_VERSION
    from stageworld.errors import ConfigurationError
    from stageworld.real_survival import _authorize, _validate_training_schedule

    configs = Path(__file__).resolve().parents[1] / "configs"
    old = load_config(configs / "project.weiai-os-v1-regimen-roi-100ep.yaml")
    candidate = load_config(configs / "project.weiai-os-v1-tumor-candidate-roi-100ep.yaml")
    assert candidate.model == old.model
    assert candidate.clinical == old.clinical
    assert candidate.survival == old.survival
    assert candidate.output_root != old.output_root
    candidate = replace(
        candidate,
        paths=replace(
            candidate.paths,
            approved_data_root=str(tmp_path),
            treatment_manifest=str(tmp_path / "treatments.json"),
        ),
    )
    _authorize(candidate)
    _validate_training_schedule(candidate, 16)
    with pytest.raises(ConfigurationError) as error:
        _authorize(
            replace(
                candidate,
                encoders=replace(candidate.encoders, ct_preprocess_version=GASTRIC_ROI_VERSION),
            )
        )
    assert error.value.code == "REAL_OS_PROTOCOL_MISMATCH"
    with pytest.raises(ConfigurationError) as error:
        _validate_training_schedule(
            replace(candidate, training=replace(candidate.training, world_pretrain_steps=1599)), 16
        )
    assert error.value.code == "OS_100_EPOCH_BUDGET_REQUIRED"
