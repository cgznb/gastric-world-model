from __future__ import annotations

import json
import stat
import warnings
from pathlib import Path

import numpy as np
import pytest

from stageworld.data import (
    FIRST_ACQUISITION_SELECTION_POLICY_VERSION,
    HMACPseudonymizer,
    run_selected_ct_qc,
)

pydicom = pytest.importorskip("pydicom")


def _write_pixel_series(directory: Path, *, series_uid: str) -> None:
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    pixels = (np.arange(512 * 512, dtype=np.uint16) % 4096).reshape(512, 512)
    study_uid = generate_uid()
    for index in range(16):
        path = directory / f"slice-{index:02d}.dcm"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            file_meta = FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = CTImageStorage
            file_meta.MediaStorageSOPInstanceUID = generate_uid()
            file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
            dataset.SOPClassUID = CTImageStorage
            dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
            dataset.Modality = "CT"
            dataset.StudyInstanceUID = study_uid
            dataset.SeriesInstanceUID = series_uid
            dataset.Rows = 512
            dataset.Columns = 512
            dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            dataset.ImagePositionPatient = [0.0, 0.0, float(index) * 10.0]
            dataset.PixelSpacing = [0.8, 0.8]
            dataset.BurnedInAnnotation = "NO"
            dataset.PhotometricInterpretation = "MONOCHROME2"
            dataset.SamplesPerPixel = 1
            dataset.BitsAllocated = 16
            dataset.BitsStored = 16
            dataset.HighBit = 15
            dataset.PixelRepresentation = 0
            dataset.RescaleSlope = 1.0
            dataset.RescaleIntercept = -1024.0
            dataset.PixelData = pixels.tobytes()
            dataset.save_as(path, enforce_file_format=True)


def test_selected_ct_qc_is_outcome_blind_redacted_and_pixel_backed(tmp_path: Path) -> None:
    from pydicom.uid import generate_uid

    approved = tmp_path / "approved"
    approved.mkdir()
    pseudonymizer = HMACPseudonymizer(b"q" * 32)
    bindings = []
    raw_uids = []
    raw_paths = []
    for role in ("baseline_ct", "post_treatment_ct"):
        directory = approved / role
        directory.mkdir()
        raw_uid = generate_uid()
        _write_pixel_series(directory, series_uid=raw_uid)
        raw_uids.append(raw_uid)
        raw_paths.append(str(directory.resolve()))
        bindings.append(
            {
                "asset_id": f"CT-{role}",
                "role": role,
                "local_path": str(directory),
                "selected_series_id": pseudonymizer.token(
                    "dicom-series", raw_uid, prefix="SERIES"
                ),
            }
        )
    manifest_path = tmp_path / "asset_bindings.json"
    manifest_path.write_text(
        json.dumps(
            {
                "data_lineage_id": "data-lineage",
                "cohort_artifact_id": "cohort-artifact",
                "phase_selection_policy_version": (
                    FIRST_ACQUISITION_SELECTION_POLICY_VERSION
                ),
                "bindings": bindings,
            }
        ),
        encoding="utf-8",
    )

    result = run_selected_ct_qc(
        asset_manifest_path=manifest_path,
        approved_root=approved,
        output_root=tmp_path / "output",
        pseudonymizer=pseudonymizer,
        sample_per_role=1,
        pixel_slices=3,
        sample_seed=17,
    )

    assert result["sampled_study_count"] == 2
    assert result["automatic_qc_status_counts"] == {"pass": 2}
    assert result["axial_orientation_count"] == 2
    assert result["pixel_decode_complete_study_count"] == 2
    assert result["z_span_mm"] == {
        "minimum": 150.0,
        "median": 150.0,
        "maximum": 150.0,
        "below_150_mm_count": 0,
    }
    assert result["outcome_data_read"] is False
    assert result["anatomic_gastric_coverage_verified"] is False
    assert result["clinical_training_ready"] is False

    aggregate_path = Path(str(result["artifact"]))
    restricted_path = Path(str(result["restricted_artifact"]))
    aggregate_text = aggregate_path.read_text(encoding="utf-8")
    assert all(raw_uid not in aggregate_text for raw_uid in raw_uids)
    assert all(raw_path not in aggregate_text for raw_path in raw_paths)
    assert stat.S_IMODE(restricted_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(restricted_path.parent.stat().st_mode) == 0o700
