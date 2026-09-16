from __future__ import annotations

import json
import stat
from datetime import date, datetime
from pathlib import Path

import openpyxl
import pytest

from stageworld.config import load_config
from stageworld.data import (
    FIRST_ACQUISITION_SELECTION_POLICY_VERSION,
    PHASE_SELECTION_POLICY_VERSION,
    CTSeriesSummary,
    CTStudySummary,
    DataMode,
    DICOMDirectorySummary,
    DirectorySourceState,
    HMACPseudonymizer,
    PairedCTMapping,
    build_paired_ct_cohort,
    load_paired_ct_mapping,
    write_paired_ct_artifacts,
)
from stageworld.errors import ConfigurationError

ROOT = Path(__file__).resolve().parents[1]


def _mapping() -> PairedCTMapping:
    return PairedCTMapping(
        sheet_index=0,
        field_header_row=1,
        data_start_row=2,
        expected_columns=90,
        columns={
            "row_key": "A",
            "baseline_origin": "E",
            "os_status": "BS",
            "death_date": "BU",
            "censor_date": "BV",
            "baseline_ct_link": "BW",
            "post_treatment_ct_link": "BZ",
        },
        event_code=1,
        censor_code=0,
        outcome_label_version="OS-v1",
        cohort_version="paired-ct-v1",
        permit_real_supervised_build=True,
        require_complete_pair=True,
        elapsed_time_only=True,
    )


def _workbook(path: Path, rows: list[dict[str, object]]) -> None:
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    for column in range(1, 91):
        worksheet.cell(1, column, f"field-{column}")
    for row_number, values in enumerate(rows, start=2):
        for column, value in values.items():
            worksheet[f"{column}{row_number}"] = value
    workbook.save(path)


def _summary(
    path: Path,
    acquired: date,
    *,
    study_id: str | None = None,
    count: int = 20,
) -> DICOMDirectorySummary:
    series = CTSeriesSummary(
        series_id=f"SERIES-{path.name}",
        file_count=count,
        rows=512,
        columns=512,
        positioned_slice_count=count,
        unique_position_count=count,
        median_slice_thickness_mm=5.0,
        phase_candidates=(),
        contrast_tag_fraction=0.0,
        plausible_volume=True,
        candidate_score=10000.0 + count,
    )
    study = CTStudySummary(
        study_id=study_id or f"STUDY-{path.name}",
        file_count=count,
        acquisition_dates=(acquired.isoformat(),),
        series=(series,),
    )
    return DICOMDirectorySummary(
        schema_version="paired-ct-dicom-index-v2",
        directory_id=f"DIR-{path.parent.name}-{path.name}",
        source_state=DirectorySourceState(0, 0, 0),
        readable_file_count=count,
        ct_file_count=count,
        non_ct_file_count=0,
        header_error_count=0,
        missing_study_uid_count=0,
        missing_series_uid_count=0,
        studies=(study,),
        warning_capture_complete=True,
    )


def _timed_summary(
    path: Path,
    acquired: date,
    *,
    header_phase_evidence: bool = True,
) -> DICOMDirectorySummary:
    specifications = (
        ("plain", (), None, None, "none", 0, 0.0),
        ("arterial", ("arterial",), 30.0, "arterial", "high", 1, 60.0),
        (
            "venous",
            ("venous_or_portal",),
            75.0,
            "venous_or_portal",
            "high",
            2,
            120.0,
        ),
    )
    series = tuple(
        CTSeriesSummary(
            series_id=f"SERIES-{path.name}-{name}",
            file_count=20,
            rows=512,
            columns=512,
            positioned_slice_count=20,
            unique_position_count=20,
            median_slice_thickness_mm=5.0,
            phase_candidates=(phases if header_phase_evidence else ()),
            contrast_tag_fraction=0.0,
            plausible_volume=True,
            candidate_score=10020.0,
            acquisition_time_source="acquisition_datetime",
            acquisition_offset_seconds=offset,
            temporal_cluster_index=cluster,
            temporal_cluster_count=3,
            contrast_bolus_delay_seconds=(bolus_delay if header_phase_evidence else None),
            timing_phase_candidate=(timing_phase if header_phase_evidence else None),
            timing_phase_basis=(
                "bolus_delay"
                if header_phase_evidence and timing_phase is not None
                else None
            ),
            timing_phase_confidence=(confidence if header_phase_evidence else "none"),
        )
        for name, phases, bolus_delay, timing_phase, confidence, cluster, offset in specifications
    )
    study = CTStudySummary(
        study_id=f"STUDY-{path.name}",
        file_count=60,
        acquisition_dates=(acquired.isoformat(),),
        series=series,
    )
    return DICOMDirectorySummary(
        schema_version="paired-ct-dicom-index-v2",
        directory_id=f"DIR-{path.parent.name}-{path.name}",
        source_state=DirectorySourceState(0, 0, 0),
        readable_file_count=60,
        ct_file_count=60,
        non_ct_file_count=0,
        header_error_count=0,
        missing_study_uid_count=0,
        missing_series_uid_count=0,
        studies=(study,),
        warning_capture_complete=True,
    )


def _asset_tree(root: Path, names: tuple[str, ...], *, second_batch: tuple[str, ...] = ()) -> None:
    for batch, assets in (("batch-a", names), ("batch-b", second_batch)):
        for name in assets:
            (root / batch / name).mkdir(parents=True, exist_ok=True)


def _patch_index(
    monkeypatch: pytest.MonkeyPatch,
    acquired: dict[str, date],
    *,
    duplicate_study_ids: dict[tuple[str, str], str] | None = None,
) -> None:
    duplicate_study_ids = duplicate_study_ids or {}

    def fake_index(
        directories: object,
        *,
        cache_root: Path,
        tokenize: object,
        workers: int,
    ) -> dict[Path, DICOMDirectorySummary]:
        del cache_root, tokenize, workers
        result = {}
        for raw_path in directories:  # type: ignore[union-attr]
            path = Path(raw_path)
            study_id = duplicate_study_ids.get((path.parent.name, path.name))
            count = 24 if path.parent.name == "batch-b" else 20
            result[path] = _summary(
                path,
                acquired[path.name],
                study_id=study_id,
                count=count,
            )
        return result

    monkeypatch.setattr("stageworld.data.paired_ct.build_dicom_metadata_index", fake_index)


def test_os_v1_paired_builder_separates_inputs_and_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workbook = tmp_path / "clinical.xlsx"
    _workbook(
        workbook,
        [
            {
                "A": "raw-patient-a",
                "E": datetime(2020, 1, 1),
                "BS": 1,
                "BU": datetime(2022, 1, 1),
                "BV": datetime(2022, 1, 2),
                "BW": "baseline-a",
                "BZ": "post-a",
            },
            {
                "A": "raw-patient-b",
                "E": "2020/02/01",
                "BS": "0",
                "BV": "2023.02.01",
                "BW": "baseline-b",
                "BZ": "post-b",
            },
        ],
    )
    ct_root = tmp_path / "ct"
    names = ("baseline-a", "post-a", "baseline-b", "post-b")
    _asset_tree(ct_root, names)
    _patch_index(
        monkeypatch,
        {
            "baseline-a": date(2019, 12, 20),
            "post-a": date(2020, 6, 1),
            "baseline-b": date(2020, 2, 2),
            "post-b": date(2020, 8, 1),
        },
    )

    result = build_paired_ct_cohort(
        workbook_path=workbook,
        ct_root=ct_root,
        mapping=_mapping(),
        pseudonymizer=HMACPseudonymizer(b"k" * 32),
        mode=DataMode.REAL_IMAGES,
        metadata_cache_root=tmp_path / "cache",
        split_seed=17,
        prediction_horizon_days=1095.75,
        workers=2,
    )

    assert len(result.input_cohort.patients) == 2
    assert result.input_cohort.outcomes == ()
    assert result.input_cohort.treatments == ()
    assert len(result.outcomes) == 2
    assert {outcome.source_status for outcome in result.outcomes} == {0, 1}
    event = next(outcome for outcome in result.outcomes if outcome.source_status == 1)
    censor = next(outcome for outcome in result.outcomes if outcome.source_status == 0)
    assert event.event_date_days == 731.0
    assert event.censor_date_days is None
    assert censor.event_date_days is None
    assert censor.censor_date_days == 1096.0
    assert all("raw-patient" not in patient.patient_id for patient in result.input_cohort.patients)
    s0_queries = [query for query in result.input_cohort.queries if query.stage.value == "s0"]
    assert sorted(query.query_time_days for query in s0_queries) == [0.0, 1.0]
    assert result.stage_landmark_counts == {"s0": 2, "s1": 2}
    assert result.stage_event_counts == {"s0": 1, "s1": 1}


def test_invalid_os_rows_and_shared_assets_are_excluded_without_relabeling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workbook = tmp_path / "clinical.xlsx"
    rows = [
        {
            "A": "good",
            "E": datetime(2020, 1, 1),
            "BS": 0,
            "BU": 0,
            "BV": datetime(2023, 1, 1),
            "BW": "base-good",
            "BZ": "post-good",
        },
        {
            "A": "death-without-date",
            "E": datetime(2020, 1, 1),
            "BS": 1,
            "BV": datetime(2023, 1, 1),
            "BW": "base-missing-death",
            "BZ": "post-missing-death",
        },
        {
            "A": "censor-with-death-date",
            "E": datetime(2020, 1, 1),
            "BS": 0,
            "BU": datetime(2022, 1, 1),
            "BV": datetime(2023, 1, 1),
            "BW": "base-conflict",
            "BZ": "post-conflict",
        },
        {
            "A": "shared-a",
            "E": datetime(2020, 1, 1),
            "BS": 0,
            "BV": datetime(2023, 1, 1),
            "BW": "shared-base",
            "BZ": "post-shared-a",
        },
        {
            "A": "shared-b",
            "E": datetime(2020, 1, 1),
            "BS": 0,
            "BV": datetime(2023, 1, 1),
            "BW": "shared-base",
            "BZ": "post-shared-b",
        },
    ]
    _workbook(workbook, rows)
    names = tuple(
        {
            str(row[column])
            for row in rows
            for column in ("BW", "BZ")
            if column in row
        }
    )
    ct_root = tmp_path / "ct"
    _asset_tree(ct_root, names)
    acquired = {
        name: date(2019, 12, 20) if "base" in name else date(2020, 6, 1)
        for name in names
    }
    _patch_index(monkeypatch, acquired)

    result = build_paired_ct_cohort(
        workbook_path=workbook,
        ct_root=ct_root,
        mapping=_mapping(),
        pseudonymizer=HMACPseudonymizer(b"s" * 32),
        mode=DataMode.REAL_IMAGES,
        metadata_cache_root=tmp_path / "cache",
        split_seed=1,
        prediction_horizon_days=365.25,
    )

    assert len(result.input_cohort.patients) == 1
    reasons = {reason for exclusion in result.exclusions for reason in exclusion.reasons}
    assert "death_date_missing" in reasons
    assert "censored_status_with_death_date" in reasons
    assert "ct_link_shared_across_rows" in reasons
    assert len(result.exclusions) == 4


def test_equivalent_duplicate_directory_is_deduplicated_by_study_uid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workbook = tmp_path / "clinical.xlsx"
    _workbook(
        workbook,
        [
            {
                "A": "patient",
                "E": datetime(2020, 1, 1),
                "BS": 0,
                "BV": datetime(2023, 1, 1),
                "BW": "duplicate-base",
                "BZ": "post",
            }
        ],
    )
    ct_root = tmp_path / "ct"
    _asset_tree(ct_root, ("duplicate-base", "post"), second_batch=("duplicate-base",))
    _patch_index(
        monkeypatch,
        {"duplicate-base": date(2019, 12, 20), "post": date(2020, 6, 1)},
        duplicate_study_ids={
            ("batch-a", "duplicate-base"): "STUDY-SAME",
            ("batch-b", "duplicate-base"): "STUDY-SAME",
        },
    )
    result = build_paired_ct_cohort(
        workbook_path=workbook,
        ct_root=ct_root,
        mapping=_mapping(),
        pseudonymizer=HMACPseudonymizer(b"d" * 32),
        mode=DataMode.REAL_IMAGES,
        metadata_cache_root=tmp_path / "cache",
        split_seed=2,
        prediction_horizon_days=365.25,
    )
    assert len(result.input_cohort.patients) == 1
    assert result.equivalent_replica_link_count == 1
    baseline = next(binding for binding in result.bindings if binding.role == "baseline_ct")
    assert baseline.replica_count == 2
    assert Path(baseline.local_path).parent.name == "batch-b"


def test_private_outputs_hold_paths_while_aggregate_is_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workbook = tmp_path / "clinical.xlsx"
    _workbook(
        workbook,
        [
            {
                "A": "patient",
                "E": datetime(2020, 1, 1),
                "BS": 0,
                "BV": datetime(2023, 1, 1),
                "BW": "baseline",
                "BZ": "post",
            }
        ],
    )
    ct_root = tmp_path / "ct"
    _asset_tree(ct_root, ("baseline", "post"))
    _patch_index(
        monkeypatch,
        {"baseline": date(2019, 12, 20), "post": date(2020, 6, 1)},
    )
    result = build_paired_ct_cohort(
        workbook_path=workbook,
        ct_root=ct_root,
        mapping=_mapping(),
        pseudonymizer=HMACPseudonymizer(b"p" * 32),
        mode=DataMode.REAL_IMAGES,
        metadata_cache_root=tmp_path / "cache",
        split_seed=3,
        prediction_horizon_days=365.25,
    )
    payload = write_paired_ct_artifacts(result, output_root=tmp_path / "output", split_seed=3)
    aggregate = json.loads(Path(payload["artifact"]).read_text(encoding="utf-8"))
    assert "local_path" not in json.dumps(aggregate)
    assert aggregate["contains_identifiers_or_paths"] is False
    assert aggregate["phase_selection_status"] == "frozen"
    assert aggregate["phase_selection_pair_counts"] == {
        "fallback_no_common_confident_phase": 1
    }
    assert aggregate["selected_phase_study_counts_by_role"] == {
        "baseline_ct": {"unknown": 1},
        "post_treatment_ct": {"unknown": 1},
    }
    assert aggregate["selected_phase_basis_counts"] == {"unknown": 2}
    assert aggregate["phase_timing_audit_version"] == "paired-ct-phase-timing-v2"
    assert aggregate["studies_with_confident_phase_candidate"] == 0
    assert aggregate["top_rank_unresolved_after_timing_count"] == 2
    assert aggregate["studies_with_any_plausible_series_acquisition_time_count"] == 0
    assert aggregate["studies_with_complete_plausible_series_acquisition_time_count"] == 0
    assert aggregate["studies_with_plausible_series_bolus_delay_count"] == 0
    assert aggregate["plausible_series_bolus_delay_bucket_counts"] == {}
    assert aggregate["explicit_phase_bolus_delay_bucket_counts"] == {}
    assert aggregate["plausible_temporal_cluster_count_study_counts"] == {"0": 2}
    assert aggregate["explicit_phase_anchor_order_counts"] == {}
    assert aggregate["explicit_phase_cluster_position_counts"] == {}
    assert aggregate["explicit_phase_plausible_cluster_position_counts"] == {}
    assert aggregate["explicit_phase_position_by_sequence_signature_counts"] == {}
    assert aggregate["temporal_gap_signature_counts_by_evidence"] == {
        "no_confident_phase": {"0_clusters": 2}
    }
    assert aggregate["paired_confident_phase_counts"] == {
        "at_least_one_exam_unresolved": 1,
        "both_have_arterial": 0,
        "both_have_same_confident_phase": 0,
        "both_have_venous_or_portal": 0,
        "neither_exam_resolved": 1,
    }
    restricted = tmp_path / "output/data/restricted"
    asset_manifest = restricted / "asset_bindings.json"
    assert str(ct_root.resolve()) in asset_manifest.read_text(encoding="utf-8")
    private_assets = json.loads(asset_manifest.read_text(encoding="utf-8"))
    assert private_assets["phase_selection_status"] == "frozen"
    assert all(
        sum(bool(series["selected"]) for series in binding["candidate_series"]) == 1
        for binding in private_assets["bindings"]
    )
    for path in restricted.rglob("*.json"):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(restricted.stat().st_mode) == 0o700


def test_phase_timing_aggregate_records_plausible_sequence_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workbook = tmp_path / "clinical.xlsx"
    _workbook(
        workbook,
        [
            {
                "A": "patient",
                "E": datetime(2020, 1, 1),
                "BS": 0,
                "BV": datetime(2023, 1, 1),
                "BW": "baseline",
                "BZ": "post",
            }
        ],
    )
    ct_root = tmp_path / "ct"
    _asset_tree(ct_root, ("baseline", "post"))

    def fake_index(
        directories: object,
        *,
        cache_root: Path,
        tokenize: object,
        workers: int,
    ) -> dict[Path, DICOMDirectorySummary]:
        del cache_root, tokenize, workers
        result = {}
        for raw_path in directories:  # type: ignore[union-attr]
            path = Path(raw_path)
            acquired = date(2019, 12, 20) if path.name == "baseline" else date(2020, 6, 1)
            result[path] = _timed_summary(path, acquired)
        return result

    monkeypatch.setattr("stageworld.data.paired_ct.build_dicom_metadata_index", fake_index)
    result = build_paired_ct_cohort(
        workbook_path=workbook,
        ct_root=ct_root,
        mapping=_mapping(),
        pseudonymizer=HMACPseudonymizer(b"q" * 32),
        mode=DataMode.REAL_IMAGES,
        metadata_cache_root=tmp_path / "cache",
        split_seed=3,
        prediction_horizon_days=365.25,
    )
    payload = write_paired_ct_artifacts(result, output_root=tmp_path / "output", split_seed=3)
    aggregate = json.loads(Path(payload["artifact"]).read_text(encoding="utf-8"))

    signature = "3_clusters:46_to_90_seconds+46_to_90_seconds"
    assert aggregate["phase_selection_status"] == "frozen"
    assert aggregate["phase_selection_policy_version"] == PHASE_SELECTION_POLICY_VERSION
    assert aggregate["phase_selection_pair_counts"] == {
        "matched_venous_or_portal": 1
    }
    assert aggregate["selected_phase_study_counts_by_role"] == {
        "baseline_ct": {"venous_or_portal": 1},
        "post_treatment_ct": {"venous_or_portal": 1},
    }
    assert aggregate["selected_phase_basis_counts"] == {"explicit_dicom_text": 2}
    assert aggregate["studies_with_any_plausible_series_acquisition_time_count"] == 2
    assert aggregate["studies_with_complete_plausible_series_acquisition_time_count"] == 2
    assert aggregate["studies_with_plausible_series_bolus_delay_count"] == 2
    assert aggregate["plausible_series_bolus_delay_bucket_counts"] == {
        "10_to_45_seconds": 2,
        "50_to_100_seconds": 2,
    }
    assert aggregate["explicit_phase_bolus_delay_bucket_counts"] == {
        "arterial:10_to_45_seconds": 2,
        "venous_or_portal:50_to_100_seconds": 2,
    }
    assert aggregate["phase_timing_conflict_series_count"] == 0
    assert aggregate["explicit_phase_anchor_order_counts"] == {
        "arterial_before_venous_46_to_90_seconds": 2
    }
    assert aggregate["explicit_phase_plausible_cluster_position_counts"] == {
        "arterial:2_of_3": 2,
        "venous_or_portal:3_of_3": 2,
    }
    assert aggregate["temporal_gap_signature_counts_by_evidence"] == {
        "explicit_arterial_and_venous": {signature: 2}
    }
    assert aggregate["explicit_phase_position_by_sequence_signature_counts"] == {
        f"{signature}|arterial:2_of_3": 2,
        f"{signature}|venous_or_portal:3_of_3": 2,
    }


def test_validated_unnamed_sequence_selects_matched_portal_venous_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workbook = tmp_path / "clinical.xlsx"
    _workbook(
        workbook,
        [
            {
                "A": "patient",
                "E": datetime(2020, 1, 1),
                "BS": 0,
                "BV": datetime(2023, 1, 1),
                "BW": "baseline",
                "BZ": "post",
            }
        ],
    )
    ct_root = tmp_path / "ct"
    _asset_tree(ct_root, ("baseline", "post"))

    def fake_index(
        directories: object,
        *,
        cache_root: Path,
        tokenize: object,
        workers: int,
    ) -> dict[Path, DICOMDirectorySummary]:
        del cache_root, tokenize, workers
        result = {}
        for raw_path in directories:  # type: ignore[union-attr]
            path = Path(raw_path)
            acquired = date(2019, 12, 20) if path.name == "baseline" else date(2020, 6, 1)
            result[path] = _timed_summary(
                path,
                acquired,
                header_phase_evidence=False,
            )
        return result

    monkeypatch.setattr("stageworld.data.paired_ct.build_dicom_metadata_index", fake_index)
    result = build_paired_ct_cohort(
        workbook_path=workbook,
        ct_root=ct_root,
        mapping=_mapping(),
        pseudonymizer=HMACPseudonymizer(b"v" * 32),
        mode=DataMode.REAL_IMAGES,
        metadata_cache_root=tmp_path / "cache",
        split_seed=3,
        prediction_horizon_days=365.25,
    )

    assert {observation.phase for observation in result.input_cohort.observations} == {
        "venous_or_portal"
    }
    assert {binding.pair_phase_status for binding in result.bindings} == {
        "matched_venous_or_portal"
    }
    assert {binding.selected_phase_basis for binding in result.bindings} == {
        "validated_sequence_signature"
    }
    for binding in result.bindings:
        selected = [
            series for series in binding.candidate_series if bool(series["selected"])
        ]
        assert len(selected) == 1
        assert selected[0]["sequence_phase_candidate"] == "venous_or_portal"
        assert selected[0]["sequence_phase_confidence"] == "moderate"


def test_first_acquisition_policy_selects_earliest_cluster_without_phase_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workbook = tmp_path / "clinical.xlsx"
    _workbook(
        workbook,
        [
            {
                "A": "patient",
                "E": datetime(2020, 1, 1),
                "BS": 0,
                "BV": datetime(2023, 1, 1),
                "BW": "baseline",
                "BZ": "post",
            }
        ],
    )
    ct_root = tmp_path / "ct"
    _asset_tree(ct_root, ("baseline", "post"))

    def fake_index(
        directories: object,
        *,
        cache_root: Path,
        tokenize: object,
        workers: int,
    ) -> dict[Path, DICOMDirectorySummary]:
        del cache_root, tokenize, workers
        result = {}
        for raw_path in directories:  # type: ignore[union-attr]
            path = Path(raw_path)
            acquired = date(2019, 12, 20) if path.name == "baseline" else date(2020, 6, 1)
            result[path] = _timed_summary(path, acquired)
        return result

    monkeypatch.setattr("stageworld.data.paired_ct.build_dicom_metadata_index", fake_index)
    result = build_paired_ct_cohort(
        workbook_path=workbook,
        ct_root=ct_root,
        mapping=_mapping(),
        pseudonymizer=HMACPseudonymizer(b"f" * 32),
        mode=DataMode.REAL_IMAGES,
        metadata_cache_root=tmp_path / "cache",
        split_seed=3,
        prediction_horizon_days=365.25,
        selection_policy_version=FIRST_ACQUISITION_SELECTION_POLICY_VERSION,
    )

    assert result.series_selection_policy_version == (
        FIRST_ACQUISITION_SELECTION_POLICY_VERSION
    )
    assert {observation.phase for observation in result.input_cohort.observations} == {
        "first_acquisition_unknown"
    }
    assert {binding.pair_phase_status for binding in result.bindings} == {
        "paired_first_acquisition"
    }
    assert {binding.selected_phase_basis for binding in result.bindings} == {
        "earliest_plausible_temporal_cluster"
    }
    for binding in result.bindings:
        selected = [
            series for series in binding.candidate_series if bool(series["selected"])
        ]
        assert len(selected) == 1
        assert selected[0]["temporal_cluster_index"] == 0
        assert selected[0]["selected_phase"] == "first_acquisition_unknown"

    payload = write_paired_ct_artifacts(
        result,
        output_root=tmp_path / "output",
        split_seed=3,
    )
    aggregate = json.loads(Path(payload["artifact"]).read_text(encoding="utf-8"))
    assert aggregate["phase_selection_status"] == "frozen_exploratory"
    assert aggregate["phase_selection_policy_version"] == (
        FIRST_ACQUISITION_SELECTION_POLICY_VERSION
    )
    assert aggregate["phase_selection_pair_counts"] == {
        "paired_first_acquisition": 1
    }
    assert aggregate["selected_phase_study_counts_by_role"] == {
        "baseline_ct": {"first_acquisition_unknown": 1},
        "post_treatment_ct": {"first_acquisition_unknown": 1},
    }


def test_signed_mapping_and_s0_s1_config_do_not_require_s2(monkeypatch: pytest.MonkeyPatch) -> None:
    mapping = load_paired_ct_mapping(ROOT / "configs/field_mapping.weiai-os-v1.yaml")
    assert mapping.columns == {
        "row_key": "A",
        "baseline_origin": "E",
        "os_status": "BS",
        "death_date": "BU",
        "censor_date": "BV",
        "baseline_ct_link": "BW",
        "post_treatment_ct_link": "BZ",
    }
    monkeypatch.setenv("STAGEWORLD_WEIAI_ROOT", str(ROOT / "tests"))
    config = load_config(ROOT / "configs/project.weiai-os-v1.yaml")
    assert config.clinical.development_stages == ("s0", "s1")
    assert config.clinical_blockers() == []
    assert "stage2_definition" not in config.clinical_blockers()
    assert "pathology_availability_basis" not in config.clinical_blockers()

    first_config = load_config(
        ROOT / "configs/project.weiai-os-v1-first-acquisition.yaml"
    )
    assert first_config.clinical.ct_series_selection_policy_version == (
        FIRST_ACQUISITION_SELECTION_POLICY_VERSION
    )
    assert first_config.output_root.name == "paired-ct-os-v1-first-acquisition"


def test_hmac_key_must_be_private_and_outside_project(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    inside = project / "key"
    inside.write_bytes(b"x" * 32)
    inside.chmod(0o600)
    with pytest.raises(ConfigurationError, match="outside the project"):
        HMACPseudonymizer.from_file(inside, project_root=project)

    outside = tmp_path / "outside-key"
    outside.write_bytes(b"x" * 32)
    outside.chmod(0o644)
    with pytest.raises(ConfigurationError, match="owner-only"):
        HMACPseudonymizer.from_file(outside, project_root=project)
    outside.chmod(0o600)
    pseudonymizer = HMACPseudonymizer.from_file(outside, project_root=project)
    assert pseudonymizer.token("patient", "raw", prefix="P").startswith("P-")
