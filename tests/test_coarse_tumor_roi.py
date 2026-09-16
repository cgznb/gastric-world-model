from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from stageworld.artifacts import read_json
from stageworld.data.tumor_roi import (
    COARSE_TUMOR_ROI_VERSION,
    TumorSegmenter,
    coarse_tumor_field_of_view,
    gastric_tumor_candidates,
)
from stageworld.errors import ConfigurationError, DataContractError


def test_relaxed_association_keeps_whole_candidate_and_physical_crop_margin():
    labels = np.zeros((60, 60, 60), np.uint8)
    labels[5:10, 10:20, 10:20] = 11
    labels[20:40, 12:15, 12:15] = 14
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    assert not gastric_tumor_candidates(labels, affine)[0].any()
    target, candidate, qc = coarse_tumor_field_of_view(labels, affine)
    assert candidate.sum() == 20 * 3 * 3
    assert qc["roi_source"] == "tumor_candidate" and qc["margin_mm"] == 30
    assert qc["outside_mask_ct_preserved"] is True
    crop_low = target[:3, 3] - np.diag(target)[:3] / 2
    crop_high = crop_low + qc["field_of_view_mm"]
    lesion_low, lesion_high = np.array([39, 23, 23]), np.array([79, 29, 29])
    assert np.all(lesion_low - crop_low >= 30)
    assert np.all(crop_high - lesion_high >= 30)


def test_distant_tumor_uses_stomach_fallback_without_positive_detection_claim():
    labels = np.zeros((80, 80, 80), np.uint8)
    labels[5:15, 5:15, 5:15] = 11
    labels[65:70, 65:70, 65:70] = 14
    _, candidate, qc = coarse_tumor_field_of_view(labels, np.eye(4))
    assert not candidate.any()
    assert qc["roi_source"] == "stomach_fallback" and qc["margin_mm"] == 40
    assert qc["fallback_used"] is True and qc["complete_response_inferred"] is False


def test_empty_tumor_and_boundary_touching_stomach_still_yield_a_crop():
    labels = np.zeros((30, 40, 50), np.uint8)
    labels[:12, 10:25, 12:35] = 11
    target, candidate, qc = coarse_tumor_field_of_view(labels, np.eye(4))
    assert not candidate.any() and np.isfinite(target).all()
    assert qc["localization_touches_scan_boundary"] is True
    assert qc["roi_source"] == "stomach_fallback"


def test_absent_anatomic_localization_is_not_replaced_with_an_invented_tumor():
    with pytest.raises(DataContractError) as error:
        coarse_tumor_field_of_view(np.zeros((20, 20, 20), np.uint8), np.eye(4))
    assert error.value.code == "COARSE_ANATOMIC_ROI_UNAVAILABLE"


def test_coarse_preprocessing_preserves_ct_outside_the_pseudomask(tmp_path, monkeypatch):
    import stageworld.data.tumor_roi as module

    monkeypatch.setattr(module, "bind_tumor_model", lambda *a, **kw: "test-model")
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "2.8.1")
    monkeypatch.setattr(module, "resolve_selected_series_files", lambda *a, **kw: [Path("fake")])
    values = np.full((48, 48, 48), 200.0, np.float32)
    affine = np.diag([4.0, 4.0, 4.0, 1.0])
    source = SimpleNamespace(GetSpacing=lambda: (4.0, 4.0, 4.0))
    monkeypatch.setattr(module, "_read_dicom_volume", lambda _: (values, affine, source))
    labels = np.zeros(values.shape, np.uint8)
    labels[12:20, 12:24, 12:24] = 11
    labels[21:23, 18:20, 18:20] = 14
    segmenter = TumorSegmenter(tmp_path, tmp_path, preprocess_version=COARSE_TUMOR_ROI_VERSION)
    monkeypatch.setattr(segmenter, "_predict", lambda image, path: nib.save(
        nib.Nifti1Image(labels, affine), path
    ))
    processed = segmenter.preprocess(
        {"asset_id": "test-asset", "selected_series_id": "test-series"},
        approved_root=tmp_path, pseudonymizer=None,
    )
    assert processed.image.shape == (1, 96, 96, 96)
    assert int((processed.image > 0.55).sum()) > int((labels == 14).sum()) * 100
    assert read_json(tmp_path / "test-asset.qc.json")["roi_source"] == "tumor_candidate"
    crop = nib.load(tmp_path / "test-asset.crop.nii.gz")
    assert crop.shape == (96, 96, 96)
    assert float(np.asarray(crop.dataobj).max()) == pytest.approx(200.0)


def test_coarse_protocol_preserves_model_and_rejects_old_roi_preprocessing(tmp_path):
    from stageworld.config import load_config
    from stageworld.data.tumor_roi import TUMOR_ROI_VERSION
    from stageworld.real_survival import _authorize, _validate_training_schedule

    configs = Path(__file__).resolve().parents[1] / "configs"
    config = load_config(configs / "project.weiai-os-v1-tumor-coarse-roi-100ep.yaml")
    old = load_config(configs / "project.weiai-os-v1-regimen-roi-100ep.yaml")
    assert config.model == old.model and config.clinical == old.clinical
    assert config.survival == old.survival and config.output_root != old.output_root
    config = replace(config, paths=replace(
        config.paths, approved_data_root=str(tmp_path),
        treatment_manifest=str(tmp_path / "treatments.json"),
    ))
    _authorize(config)
    _validate_training_schedule(config, 16)
    with pytest.raises(ConfigurationError):
        _authorize(replace(config, encoders=replace(
            config.encoders, ct_preprocess_version=TUMOR_ROI_VERSION
        )))


def test_mask_reuse_preserves_predictions_but_isolates_roi_provenance(tmp_path):
    from stageworld.artifacts import atomic_write_private_json
    from stageworld.data.tumor_roi import (
        TUMOR_LABELS,
        TUMOR_MODEL_FILES,
        TUMOR_ROI_VERSION,
        bind_tumor_model,
        reuse_tumor_mask_predictions,
    )
    from stageworld.errors import ArtifactError

    model, old, new = (tmp_path / name for name in ("model", "old", "new"))
    for name in TUMOR_MODEL_FILES:
        path = model / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic-test-model")
    atomic_write_private_json(model / "dataset.json", {
        "labels": TUMOR_LABELS, "channel_names": {"0": "CT"}, "file_ending": ".nii.gz",
    })
    atomic_write_private_json(model / "plans.json", {"configurations": {"3d_fullres": {}}})
    old_id = bind_tumor_model(model, old, create=True)
    labels = np.zeros((8, 8, 8), np.uint8)
    labels[2:5, 2:5, 2:5] = 11
    nib.save(nib.Nifti1Image(labels, np.eye(4)), old / "test-scan.nii.gz")
    record = {
        "preprocess_version": TUMOR_ROI_VERSION, "model_artifact_id": old_id,
        "selected_series_id": "test-series", "source_shape": list(labels.shape),
        "source_affine": np.eye(4).tolist(), "outcome_data_read": False,
    }
    atomic_write_private_json(old / "test-scan.json", record)
    assert reuse_tumor_mask_predictions(model, old, new)["imported_full_label_masks"] == 1
    assert (old / "test-scan.nii.gz").read_bytes() == (new / "test-scan.nii.gz").read_bytes()
    imported = read_json(new / "test-scan.json")
    assert imported["preprocess_version"] == COARSE_TUMOR_ROI_VERSION
    assert imported["model_artifact_id"] != old_id
    assert read_json(old / "test-scan.json") == record
    assert reuse_tumor_mask_predictions(model, old, new)["already_present"] == 1
    (model / "fold_all/checkpoint_final.pth").write_text("different-model")
    with pytest.raises(ArtifactError):
        reuse_tumor_mask_predictions(model, old, new)
