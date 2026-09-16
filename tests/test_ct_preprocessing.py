from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from stageworld.data import ct_preprocessing
from stageworld.data.ct_preprocessing import (
    SWINUNETR_INPUT_SHAPE,
    SWINUNETR_SPACING_MM,
    preprocess_selected_ct,
    resolve_selected_series_files,
)
from stageworld.data.paired_ct import HMACPseudonymizer


class _ImageGeometry:
    def GetSpacing(self) -> tuple[float, float, float]:
        return (2.0, 2.0, 4.0)


def test_selected_series_resolution_captures_dicom_warnings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pseudonymizer = HMACPseudonymizer(b"x" * 32)
    raw_series_id = "private-source-uid"
    selected_series_id = pseudonymizer.token(
        "dicom-series", raw_series_id, prefix="SERIES"
    )
    for index in range(16):
        (tmp_path / f"slice-{index:02d}.dcm").touch()

    def warned_read(path: Path, **_: object) -> SimpleNamespace:
        warnings.warn("private DICOM value must not reach logs", UserWarning, stacklevel=2)
        index = int(path.stem.removeprefix("slice-"))
        return SimpleNamespace(
            Modality="CT",
            SeriesInstanceUID=raw_series_id,
            Rows=8,
            Columns=8,
            ImageOrientationPatient=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            ImagePositionPatient=(0.0, 0.0, float(index)),
            PhotometricInterpretation="MONOCHROME2",
        )

    monkeypatch.setattr("pydicom.dcmread", warned_read)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        selected = resolve_selected_series_files(
            {"local_path": str(tmp_path), "selected_series_id": selected_series_id},
            approved_root=tmp_path,
            pseudonymizer=pseudonymizer,
        )

    assert len(selected) == 16
    assert caught == []


def test_deterministic_ct_preprocessing_preserves_geometry_and_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("monai")
    source = np.full((48, 56, 24), -1000.0, dtype=np.float32)
    source[8:40, 8:48, 4:20] = 500.0
    affine = np.diag((2.0, 2.0, 4.0, 1.0))
    files = tuple(tmp_path / f"slice-{index}.dcm" for index in range(24))
    monkeypatch.setattr(
        ct_preprocessing,
        "resolve_selected_series_files",
        lambda *args, **kwargs: files,
    )
    monkeypatch.setattr(
        ct_preprocessing,
        "_read_dicom_volume",
        lambda selected: (source, affine, _ImageGeometry()),
    )

    processed = preprocess_selected_ct(
        {},
        approved_root=tmp_path,
        pseudonymizer=object(),  # type: ignore[arg-type]
    )

    assert processed.image.shape == (1, *SWINUNETR_INPUT_SHAPE)
    assert processed.image.dtype == torch.float32
    assert torch.isfinite(processed.image).all()
    assert 0.0 <= float(processed.image.min()) <= float(processed.image.max()) <= 1.0
    assert processed.selected_file_count == 24
    assert processed.source_shape_xyz == source.shape
    assert processed.source_spacing_xyz_mm == (2.0, 2.0, 4.0)
    assert torch.allclose(
        processed.geometry.spacing_mm,
        torch.tensor([SWINUNETR_SPACING_MM]),
    )
    assert torch.equal(
        processed.geometry.spatial_shape,
        torch.tensor([SWINUNETR_INPUT_SHAPE]),
    )
    assert processed.geometry.origin_mm.shape == (1, 3)
    assert processed.geometry.direction.shape == (1, 3, 3)
