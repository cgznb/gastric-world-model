from __future__ import annotations

import json
import stat
import warnings
from pathlib import Path

import pytest

from stageworld.data import HMACPseudonymizer, build_dicom_metadata_index

pydicom = pytest.importorskip("pydicom")


def _write_ct(
    path: Path,
    *,
    study_uid: str,
    series_uid: str,
    index: int,
    description: str = "portal venous private test phrase",
    acquisition_time: str | None = None,
    bolus_start_time: str | None = None,
    series_number: int = 1,
) -> None:
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

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
        dataset.StudyDate = "20200304"
        if acquisition_time is not None:
            dataset.AcquisitionDateTime = f"20200304{acquisition_time}"
            dataset.AcquisitionDate = "20200304"
            dataset.AcquisitionTime = acquisition_time
            dataset.SeriesDate = "20200304"
            dataset.SeriesTime = acquisition_time
            dataset.ContentDate = "20200304"
            dataset.ContentTime = acquisition_time
        if bolus_start_time is not None:
            dataset.ContrastBolusStartTime = bolus_start_time
        dataset.SeriesNumber = series_number
        dataset.AcquisitionNumber = series_number
        dataset.Rows = 512
        dataset.Columns = 512
        dataset.ImagePositionPatient = [0.0, 0.0, float(index) * 5.0]
        dataset.SliceThickness = 5.0
        dataset.SeriesDescription = description
        dataset.ContrastBolusAgent = "test contrast"
        dataset.save_as(path, enforce_file_format=True)


def test_dicom_index_is_header_only_redacted_ranked_and_restartable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydicom.uid import generate_uid

    directory = tmp_path / "study"
    directory.mkdir()
    study_uid = "1.2.826.0.1.3680043.8.498.123.01"
    series_uid = generate_uid()
    for index in range(16):
        _write_ct(
            directory / f"slice-{index:02d}.dcm",
            study_uid=study_uid,
            series_uid=series_uid,
            index=index,
        )
    cache = tmp_path / "cache"
    pseudonymizer = HMACPseudonymizer(b"i" * 32)
    first = build_dicom_metadata_index(
        (directory,), cache_root=cache, tokenize=pseudonymizer.token, workers=2
    )
    summary = first[directory.resolve()]
    assert summary.ct_file_count == 16
    assert summary.header_error_count == 0
    assert summary.header_warning_count >= 16
    assert summary.warning_capture_complete is True
    assert summary.single_study_date is not None
    series = summary.studies[0].series[0]
    assert series.plausible_volume is True
    assert series.phase_candidates == ("venous_or_portal",)
    cache_path = next(cache.glob("*.json"))
    cache_text = cache_path.read_text(encoding="utf-8")
    assert "private test phrase" not in cache_text
    assert study_uid not in cache_text
    assert series_uid not in cache_text
    assert stat.S_IMODE(cache.stat().st_mode) == 0o700
    assert stat.S_IMODE(cache_path.stat().st_mode) == 0o600

    legacy_payload = json.loads(cache_text)
    legacy_payload.pop("warning_capture_complete")
    cache_path.write_text(json.dumps(legacy_payload), encoding="utf-8")
    reindexed = build_dicom_metadata_index(
        (directory,), cache_root=cache, tokenize=pseudonymizer.token, workers=1
    )
    assert reindexed[directory.resolve()].warning_capture_complete is True
    assert json.loads(cache_path.read_text(encoding="utf-8"))[
        "warning_capture_complete"
    ] is True

    def unexpected_scan(*args: object, **kwargs: object) -> None:
        raise AssertionError("a matching restart cache must avoid rescanning DICOM headers")

    monkeypatch.setattr("stageworld.data.dicom_index.inspect_dicom_directory", unexpected_scan)
    second = build_dicom_metadata_index(
        (directory,), cache_root=cache, tokenize=pseudonymizer.token, workers=1
    )
    assert second == reindexed
    assert json.loads(cache_text)["schema_version"] == "paired-ct-dicom-index-v2"


def test_dicom_index_records_conservative_temporal_phase_evidence(tmp_path: Path) -> None:
    from pydicom.uid import generate_uid

    directory = tmp_path / "timed-study"
    directory.mkdir()
    study_uid = generate_uid()
    specifications = (
        ("pre", "abdomen volume", "095930", "100000"),
        ("arterial", "arterial phase", "100030", "100000"),
        ("venous", "portal venous phase", "100105", "100000"),
        ("venous-recon", "thin reconstruction", "100105", None),
        ("late-recon", "late reconstruction", "100300", None),
    )
    for series_number, (name, description, acquired, bolus) in enumerate(
        specifications, start=1
    ):
        series_uid = generate_uid()
        for index in range(16):
            _write_ct(
                directory / f"{name}-{index:02d}.dcm",
                study_uid=study_uid,
                series_uid=series_uid,
                index=index,
                description=description,
                acquisition_time=acquired,
                bolus_start_time=bolus,
                series_number=series_number,
            )

    result = build_dicom_metadata_index(
        (directory,),
        cache_root=tmp_path / "cache",
        tokenize=HMACPseudonymizer(b"t" * 32).token,
        workers=1,
    )
    series = result[directory.resolve()].studies[0].series
    by_number = {item.series_number: item for item in series}

    assert by_number[1].timing_phase_candidate == "noncontrast"
    assert by_number[1].timing_phase_basis == "bolus_delay"
    assert by_number[1].timing_phase_confidence == "high"
    assert by_number[2].phase_candidates == ("arterial",)
    assert by_number[2].timing_phase_candidate == "arterial"
    assert by_number[3].phase_candidates == ("venous_or_portal",)
    assert by_number[3].timing_phase_candidate == "venous_or_portal"
    assert by_number[4].phase_candidates == ()
    assert by_number[4].timing_phase_candidate == "venous_or_portal"
    assert by_number[4].timing_phase_basis == "same_time_phase_anchor"
    assert by_number[4].timing_phase_confidence == "moderate"
    assert by_number[5].timing_phase_candidate == "delayed"
    assert by_number[5].timing_phase_basis == "relative_order_to_phase_anchors"
    assert by_number[5].timing_phase_confidence == "low"
    assert {item.temporal_cluster_count for item in series} == {4}
