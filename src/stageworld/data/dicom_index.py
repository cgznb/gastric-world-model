"""Restartable, de-identified DICOM metadata indexing for linked CT directories."""

from __future__ import annotations

import json
import math
import statistics
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any, Protocol

from stageworld.artifacts import atomic_write_private_json
from stageworld.errors import DataContractError, StageWorldError

DICOM_INDEX_SCHEMA = "paired-ct-dicom-index-v2"
_TEMPORAL_CLUSTER_TOLERANCE_SECONDS = 15.0
_PHASE_TIMING_WINDOWS_SECONDS = {
    "noncontrast": (-3600.0, -5.0),
    "arterial": (10.0, 45.0),
    "venous_or_portal": (50.0, 100.0),
    "delayed": (120.0, 600.0),
}


class Tokenizer(Protocol):
    def __call__(self, namespace: str, value: str, *, prefix: str) -> str: ...


@dataclass(frozen=True, slots=True)
class DirectorySourceState:
    file_count: int
    total_bytes: int
    latest_mtime_ns: int


@dataclass(frozen=True, slots=True)
class CTSeriesSummary:
    series_id: str
    file_count: int
    rows: int | None
    columns: int | None
    positioned_slice_count: int
    unique_position_count: int
    median_slice_thickness_mm: float | None
    phase_candidates: tuple[str, ...]
    contrast_tag_fraction: float
    plausible_volume: bool
    candidate_score: float
    acquisition_time_source: str | None = None
    acquisition_offset_seconds: float | None = None
    temporal_cluster_index: int | None = None
    temporal_cluster_count: int = 0
    contrast_bolus_delay_seconds: float | None = None
    timing_phase_candidate: str | None = None
    timing_phase_basis: str | None = None
    timing_phase_confidence: str = "none"
    series_number: int | None = None
    acquisition_number: int | None = None


@dataclass(frozen=True, slots=True)
class CTStudySummary:
    study_id: str
    file_count: int
    acquisition_dates: tuple[str, ...]
    series: tuple[CTSeriesSummary, ...]


@dataclass(frozen=True, slots=True)
class DICOMDirectorySummary:
    schema_version: str
    directory_id: str
    source_state: DirectorySourceState
    readable_file_count: int
    ct_file_count: int
    non_ct_file_count: int
    header_error_count: int
    missing_study_uid_count: int
    missing_series_uid_count: int
    studies: tuple[CTStudySummary, ...]
    header_warning_count: int = 0
    warning_capture_complete: bool = False
    fatal_error_code: str | None = None

    @property
    def single_study_date(self) -> date | None:
        if len(self.studies) != 1 or len(self.studies[0].acquisition_dates) != 1:
            return None
        try:
            return date.fromisoformat(self.studies[0].acquisition_dates[0])
        except ValueError:
            return None

    @property
    def study_ids(self) -> tuple[str, ...]:
        return tuple(study.study_id for study in self.studies)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DICOMDirectorySummary:
        studies = tuple(
            CTStudySummary(
                study_id=str(study["study_id"]),
                file_count=int(study["file_count"]),
                acquisition_dates=tuple(str(item) for item in study["acquisition_dates"]),
                series=tuple(
                    CTSeriesSummary(
                        series_id=str(series["series_id"]),
                        file_count=int(series["file_count"]),
                        rows=(None if series.get("rows") is None else int(series["rows"])),
                        columns=(
                            None if series.get("columns") is None else int(series["columns"])
                        ),
                        positioned_slice_count=int(series["positioned_slice_count"]),
                        unique_position_count=int(series["unique_position_count"]),
                        median_slice_thickness_mm=(
                            None
                            if series.get("median_slice_thickness_mm") is None
                            else float(series["median_slice_thickness_mm"])
                        ),
                        phase_candidates=tuple(
                            str(item) for item in series["phase_candidates"]
                        ),
                        contrast_tag_fraction=float(series["contrast_tag_fraction"]),
                        plausible_volume=bool(series["plausible_volume"]),
                        candidate_score=float(series["candidate_score"]),
                        acquisition_time_source=(
                            None
                            if series.get("acquisition_time_source") is None
                            else str(series["acquisition_time_source"])
                        ),
                        acquisition_offset_seconds=(
                            None
                            if series.get("acquisition_offset_seconds") is None
                            else float(series["acquisition_offset_seconds"])
                        ),
                        temporal_cluster_index=(
                            None
                            if series.get("temporal_cluster_index") is None
                            else int(series["temporal_cluster_index"])
                        ),
                        temporal_cluster_count=int(series.get("temporal_cluster_count", 0)),
                        contrast_bolus_delay_seconds=(
                            None
                            if series.get("contrast_bolus_delay_seconds") is None
                            else float(series["contrast_bolus_delay_seconds"])
                        ),
                        timing_phase_candidate=(
                            None
                            if series.get("timing_phase_candidate") is None
                            else str(series["timing_phase_candidate"])
                        ),
                        timing_phase_basis=(
                            None
                            if series.get("timing_phase_basis") is None
                            else str(series["timing_phase_basis"])
                        ),
                        timing_phase_confidence=str(
                            series.get("timing_phase_confidence", "none")
                        ),
                        series_number=(
                            None
                            if series.get("series_number") is None
                            else int(series["series_number"])
                        ),
                        acquisition_number=(
                            None
                            if series.get("acquisition_number") is None
                            else int(series["acquisition_number"])
                        ),
                    )
                    for series in study["series"]
                ),
            )
            for study in value["studies"]
        )
        source = value["source_state"]
        return cls(
            schema_version=str(value["schema_version"]),
            directory_id=str(value["directory_id"]),
            source_state=DirectorySourceState(
                file_count=int(source["file_count"]),
                total_bytes=int(source["total_bytes"]),
                latest_mtime_ns=int(source["latest_mtime_ns"]),
            ),
            readable_file_count=int(value["readable_file_count"]),
            ct_file_count=int(value["ct_file_count"]),
            non_ct_file_count=int(value["non_ct_file_count"]),
            header_error_count=int(value["header_error_count"]),
            missing_study_uid_count=int(value["missing_study_uid_count"]),
            missing_series_uid_count=int(value["missing_series_uid_count"]),
            studies=studies,
            header_warning_count=int(value.get("header_warning_count", -1)),
            warning_capture_complete=bool(value.get("warning_capture_complete", False)),
            fatal_error_code=(
                None if value.get("fatal_error_code") is None else str(value["fatal_error_code"])
            ),
        )


@dataclass(slots=True)
class _SeriesAccumulator:
    file_count: int = 0
    rows: Counter[int] = field(default_factory=Counter)
    columns: Counter[int] = field(default_factory=Counter)
    positions: list[float] = field(default_factory=list)
    thicknesses: list[float] = field(default_factory=list)
    phases: Counter[str] = field(default_factory=Counter)
    contrast_count: int = 0
    acquisition_moments: dict[str, float] = field(default_factory=dict)
    contrast_bolus_delay_seconds: float | None = None
    series_numbers: Counter[int] = field(default_factory=Counter)
    acquisition_numbers: Counter[int] = field(default_factory=Counter)


def _require_pydicom() -> Any:
    try:
        import pydicom  # type: ignore[import-untyped]
    except ImportError as exc:
        raise StageWorldError(
            code="dependency_missing",
            message="DICOM metadata indexing requires pydicom.",
            remediation="Install the ct optional dependency in the approved environment.",
            details={"dependency": "pydicom"},
        ) from exc
    return pydicom


def _directory_files(path: Path) -> tuple[Path, ...]:
    try:
        return tuple(sorted(item for item in path.iterdir() if item.is_file()))
    except OSError as exc:
        raise DataContractError(
            code="ct_directory_unreadable",
            message="A linked CT directory cannot be enumerated.",
        ) from exc


def directory_source_state(files: Iterable[Path]) -> DirectorySourceState:
    count = total_bytes = latest_mtime_ns = 0
    try:
        for path in files:
            status = path.stat()
            count += 1
            total_bytes += int(status.st_size)
            latest_mtime_ns = max(latest_mtime_ns, int(status.st_mtime_ns))
    except OSError as exc:
        raise DataContractError(
            code="ct_file_unreadable",
            message="A linked CT file cannot be inspected.",
        ) from exc
    return DirectorySourceState(count, total_bytes, latest_mtime_ns)


def _valid_number(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mode(values: Counter[int]) -> int | None:
    if not values:
        return None
    return values.most_common(1)[0][0]


def _valid_integer(value: object) -> int | None:
    number = _valid_number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _parse_dicom_date(value: object) -> date | None:
    raw = str(value).strip()
    if len(raw) < 8 or not raw[:8].isdigit():
        return None
    try:
        return date(int(raw[:4]), int(raw[4:6]), int(raw[6:8]))
    except ValueError:
        return None


def _parse_dicom_time_seconds(value: object) -> float | None:
    raw = str(value).strip()
    if len(raw) < 4 or not raw[:4].isdigit():
        return None
    try:
        hour = int(raw[:2])
        minute = int(raw[2:4])
        second = float(raw[4:]) if len(raw) > 4 else 0.0
    except ValueError:
        return None
    if not 0 <= hour <= 23 or not 0 <= minute <= 59 or not 0.0 <= second < 61.0:
        return None
    return hour * 3600.0 + minute * 60.0 + second


def _datetime_seconds(day: date, time_seconds: float) -> float:
    return day.toordinal() * 86400.0 + time_seconds


def _dicom_moment_candidates(dataset: Any) -> dict[str, float]:
    candidates: dict[str, float] = {}
    raw_datetime = str(getattr(dataset, "AcquisitionDateTime", "")).strip()
    if len(raw_datetime) >= 12:
        day = _parse_dicom_date(raw_datetime[:8])
        raw_time = raw_datetime[8:]
        for separator in ("+", "-"):
            raw_time = raw_time.split(separator, maxsplit=1)[0]
        seconds = _parse_dicom_time_seconds(raw_time)
        if day is not None and seconds is not None:
            candidates["acquisition_datetime"] = _datetime_seconds(day, seconds)

    fallback_day = _dicom_date(dataset)
    for source, date_field, time_field in (
        ("acquisition_time", "AcquisitionDate", "AcquisitionTime"),
        ("series_time", "SeriesDate", "SeriesTime"),
        ("content_time", "ContentDate", "ContentTime"),
        ("study_time", "StudyDate", "StudyTime"),
    ):
        day = _parse_dicom_date(getattr(dataset, date_field, "")) or fallback_day
        seconds = _parse_dicom_time_seconds(getattr(dataset, time_field, ""))
        if day is not None and seconds is not None:
            candidates[source] = _datetime_seconds(day, seconds)
    return candidates


def _preferred_moment(moments: Mapping[str, float]) -> tuple[str | None, float | None]:
    for source in (
        "acquisition_datetime",
        "acquisition_time",
        "series_time",
        "content_time",
        "study_time",
    ):
        if source in moments:
            return source, moments[source]
    return None, None


def _time_of_day_delta(moment: float, reference_seconds: float) -> float:
    delta = moment % 86400.0 - reference_seconds
    if delta < -43200.0:
        delta += 86400.0
    elif delta > 43200.0:
        delta -= 86400.0
    return delta


def _phase_from_bolus_delay(delay_seconds: float | None) -> str | None:
    if delay_seconds is None:
        return None
    for phase, (lower, upper) in _PHASE_TIMING_WINDOWS_SECONDS.items():
        if lower <= delay_seconds <= upper:
            return phase
    return None


def _dicom_date(dataset: Any) -> date | None:
    for field_name in ("StudyDate", "AcquisitionDate", "SeriesDate", "ContentDate"):
        parsed = _parse_dicom_date(getattr(dataset, field_name, ""))
        if parsed is not None:
            return parsed
    return _parse_dicom_date(str(getattr(dataset, "AcquisitionDateTime", ""))[:8])


def _phase_candidates(dataset: Any) -> tuple[str, ...]:
    text = " ".join(
        str(getattr(dataset, field_name, ""))
        for field_name in ("SeriesDescription", "ProtocolName", "ImageType")
    ).casefold()
    candidates: set[str] = set()
    keyword_groups = {
        "arterial": ("arterial", "artery", "art phase", "a phase", "\u52a8\u8109"),
        "venous_or_portal": (
            "venous",
            "portal",
            "pvp",
            "pv phase",
            "\u9759\u8109",
            "\u95e8\u8109",
        ),
        "delayed": ("delay", "equilibrium", "\u5ef6\u8fdf"),
        "noncontrast": (
            "noncontrast",
            "non-contrast",
            "unenhanced",
            "plain",
            "\u5e73\u626b",
        ),
    }
    for phase, keywords in keyword_groups.items():
        if any(keyword in text for keyword in keywords):
            candidates.add(phase)
    return tuple(sorted(candidates))


def _series_summary(
    series_id: str,
    value: _SeriesAccumulator,
    *,
    study_start_seconds: float | None,
) -> CTSeriesSummary:
    rows = _mode(value.rows)
    columns = _mode(value.columns)
    unique_positions = len({round(position, 3) for position in value.positions})
    plausible = (
        rows == 512
        and columns == 512
        and value.file_count >= 16
        and unique_positions >= 16
    )
    median_thickness = (
        float(statistics.median(value.thicknesses)) if value.thicknesses else None
    )
    phase_candidates = tuple(sorted(value.phases))
    contrast_fraction = value.contrast_count / value.file_count if value.file_count else 0.0
    score = (
        (10000.0 if plausible else 0.0)
        + min(float(unique_positions), 999.0)
        + min(float(value.file_count), 999.0) / 1000.0
    )
    time_source, acquisition_moment = _preferred_moment(value.acquisition_moments)
    acquisition_offset = (
        None
        if acquisition_moment is None or study_start_seconds is None
        else acquisition_moment - study_start_seconds
    )
    timing_phase = _phase_from_bolus_delay(value.contrast_bolus_delay_seconds)
    return CTSeriesSummary(
        series_id=series_id,
        file_count=value.file_count,
        rows=rows,
        columns=columns,
        positioned_slice_count=len(value.positions),
        unique_position_count=unique_positions,
        median_slice_thickness_mm=median_thickness,
        phase_candidates=phase_candidates,
        contrast_tag_fraction=contrast_fraction,
        plausible_volume=plausible,
        candidate_score=score,
        acquisition_time_source=time_source,
        acquisition_offset_seconds=acquisition_offset,
        contrast_bolus_delay_seconds=value.contrast_bolus_delay_seconds,
        timing_phase_candidate=timing_phase,
        timing_phase_basis=("bolus_delay" if timing_phase is not None else None),
        timing_phase_confidence=("high" if timing_phase is not None else "none"),
        series_number=_mode(value.series_numbers),
        acquisition_number=_mode(value.acquisition_numbers),
    )


def _annotate_temporal_clusters(
    series: tuple[CTSeriesSummary, ...],
) -> tuple[CTSeriesSummary, ...]:
    timed = sorted(
        (
            (item.acquisition_offset_seconds, item.series_id)
            for item in series
            if item.acquisition_offset_seconds is not None
        ),
        key=lambda item: (float(item[0]), item[1]),
    )
    cluster_anchors: list[float] = []
    cluster_by_series: dict[str, int] = {}
    for raw_offset, series_id in timed:
        offset = float(raw_offset)
        if (
            not cluster_anchors
            or offset - cluster_anchors[-1] > _TEMPORAL_CLUSTER_TOLERANCE_SECONDS
        ):
            cluster_anchors.append(offset)
        cluster_by_series[series_id] = len(cluster_anchors) - 1
    cluster_count = len(cluster_anchors)
    annotated = tuple(
        replace(
            item,
            temporal_cluster_index=cluster_by_series.get(item.series_id),
            temporal_cluster_count=cluster_count,
        )
        for item in series
    )

    phase_anchors: dict[int, set[str]] = defaultdict(set)
    for item in annotated:
        if not item.plausible_volume or item.temporal_cluster_index is None:
            continue
        if len(item.phase_candidates) == 1:
            phase_anchors[item.temporal_cluster_index].add(item.phase_candidates[0])
        if item.timing_phase_candidate is not None and item.timing_phase_confidence == "high":
            phase_anchors[item.temporal_cluster_index].add(item.timing_phase_candidate)

    propagated: list[CTSeriesSummary] = []
    for item in annotated:
        if (
            item.plausible_volume
            and not item.phase_candidates
            and item.timing_phase_candidate is None
            and item.temporal_cluster_index is not None
        ):
            candidates = phase_anchors.get(item.temporal_cluster_index, set())
            if len(candidates) == 1:
                item = replace(
                    item,
                    timing_phase_candidate=next(iter(candidates)),
                    timing_phase_basis="same_time_phase_anchor",
                    timing_phase_confidence="moderate",
                )
        propagated.append(item)

    arterial_clusters = sorted(
        cluster
        for cluster, phases in phase_anchors.items()
        if phases == {"arterial"}
    )
    venous_clusters = sorted(
        cluster
        for cluster, phases in phase_anchors.items()
        if phases == {"venous_or_portal"}
    )
    if not arterial_clusters or not venous_clusters:
        return tuple(propagated)
    first_arterial = arterial_clusters[0]
    last_venous = venous_clusters[-1]
    if first_arterial >= last_venous:
        return tuple(propagated)

    ordered: list[CTSeriesSummary] = []
    for item in propagated:
        cluster = item.temporal_cluster_index
        if (
            not item.plausible_volume
            or item.phase_candidates
            or item.timing_phase_candidate is not None
            or cluster is None
        ):
            ordered.append(item)
            continue
        if cluster < first_arterial:
            item = replace(
                item,
                timing_phase_candidate="noncontrast",
                timing_phase_basis="relative_order_to_phase_anchors",
                timing_phase_confidence="low",
            )
        elif (
            cluster > last_venous
            and item.acquisition_offset_seconds is not None
            and item.acquisition_offset_seconds - cluster_anchors[last_venous] >= 60.0
        ):
            item = replace(
                item,
                timing_phase_candidate="delayed",
                timing_phase_basis="relative_order_to_phase_anchors",
                timing_phase_confidence="low",
            )
        ordered.append(item)
    return tuple(ordered)


def inspect_dicom_directory(
    path: Path,
    *,
    directory_id: str,
    tokenize: Tokenizer,
) -> DICOMDirectorySummary:
    """Read headers only; no pixels or free-text DICOM fields are retained."""

    pydicom = _require_pydicom()
    files = _directory_files(path)
    source_state = directory_source_state(files)
    tags = [
        "Modality",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "StudyDate",
        "StudyTime",
        "AcquisitionDateTime",
        "AcquisitionDate",
        "AcquisitionTime",
        "SeriesDate",
        "SeriesTime",
        "ContentDate",
        "ContentTime",
        "Rows",
        "Columns",
        "ImagePositionPatient",
        "SliceThickness",
        "SeriesDescription",
        "ProtocolName",
        "ImageType",
        "ContrastBolusAgent",
        "ContrastBolusStartTime",
        "ContrastBolusStopTime",
        "TriggerTime",
        "SeriesNumber",
        "AcquisitionNumber",
    ]
    readable = non_ct = header_errors = header_warnings = missing_study = missing_series = 0
    study_counts: Counter[str] = Counter()
    study_dates: dict[str, set[date]] = defaultdict(set)
    series_study: dict[str, str] = {}
    series_values: dict[str, _SeriesAccumulator] = defaultdict(_SeriesAccumulator)

    for file_path in files:
        captured: list[Any] = []
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                dataset = pydicom.dcmread(
                    file_path,
                    stop_before_pixels=True,
                    specific_tags=tags,
                )
                modality = str(getattr(dataset, "Modality", "")).strip().upper()
                raw_study = str(getattr(dataset, "StudyInstanceUID", "")).strip()
                raw_series = str(getattr(dataset, "SeriesInstanceUID", "")).strip()
                acquired = _dicom_date(dataset)
                row_value = _valid_number(getattr(dataset, "Rows", None))
                column_value = _valid_number(getattr(dataset, "Columns", None))
                position = getattr(dataset, "ImagePositionPatient", None)
                z_position = (
                    _valid_number(position[2])
                    if position is not None and len(position) >= 3
                    else None
                )
                thickness = _valid_number(getattr(dataset, "SliceThickness", None))
                phases = _phase_candidates(dataset)
                moment_candidates = _dicom_moment_candidates(dataset)
                moment_source, preferred_moment = _preferred_moment(moment_candidates)
                bolus_start = _parse_dicom_time_seconds(
                    getattr(dataset, "ContrastBolusStartTime", "")
                )
                bolus_delay = (
                    _time_of_day_delta(preferred_moment, bolus_start)
                    if preferred_moment is not None
                    and bolus_start is not None
                    and moment_source != "study_time"
                    else None
                )
                series_number = _valid_integer(getattr(dataset, "SeriesNumber", None))
                acquisition_number = _valid_integer(
                    getattr(dataset, "AcquisitionNumber", None)
                )
                has_contrast = bool(
                    str(getattr(dataset, "ContrastBolusAgent", "")).strip()
                )
        except Exception:  # pydicom normalizes many malformed-header exceptions.
            header_warnings += len(captured)
            header_errors += 1
            continue
        header_warnings += len(captured)
        readable += 1
        if modality != "CT":
            non_ct += 1
            continue
        if not raw_study:
            missing_study += 1
            continue
        if not raw_series:
            missing_series += 1
            continue
        study_id = tokenize("dicom-study", raw_study, prefix="STUDY")
        series_id = tokenize("dicom-series", raw_series, prefix="SERIES")
        study_counts[study_id] += 1
        series_study[series_id] = study_id
        if acquired is not None:
            study_dates[study_id].add(acquired)
        accumulator = series_values[series_id]
        accumulator.file_count += 1
        if row_value is not None and row_value.is_integer():
            accumulator.rows[int(row_value)] += 1
        if column_value is not None and column_value.is_integer():
            accumulator.columns[int(column_value)] += 1
        if z_position is not None:
            accumulator.positions.append(z_position)
        if thickness is not None and thickness > 0:
            accumulator.thicknesses.append(thickness)
        accumulator.phases.update(phases)
        for source, moment in moment_candidates.items():
            current = accumulator.acquisition_moments.get(source)
            if current is None or moment < current:
                accumulator.acquisition_moments[source] = moment
        if bolus_delay is not None and (
            accumulator.contrast_bolus_delay_seconds is None
            or bolus_delay < accumulator.contrast_bolus_delay_seconds
        ):
            accumulator.contrast_bolus_delay_seconds = bolus_delay
        if series_number is not None:
            accumulator.series_numbers[series_number] += 1
        if acquisition_number is not None:
            accumulator.acquisition_numbers[acquisition_number] += 1
        if has_contrast:
            accumulator.contrast_count += 1

    studies = []
    for study_id in sorted(study_counts):
        study_series_values = {
            series_id: value
            for series_id, value in series_values.items()
            if series_study.get(series_id) == study_id
        }
        study_moments = [
            moment
            for value in study_series_values.values()
            for _, moment in (_preferred_moment(value.acquisition_moments),)
            if moment is not None
        ]
        study_start = min(study_moments) if study_moments else None
        annotated_series = _annotate_temporal_clusters(
            tuple(
                _series_summary(
                    series_id,
                    value,
                    study_start_seconds=study_start,
                )
                for series_id, value in study_series_values.items()
            )
        )
        series = tuple(
            sorted(
                annotated_series,
                key=lambda item: (-item.candidate_score, item.series_id),
            )
        )
        studies.append(
            CTStudySummary(
                study_id=study_id,
                file_count=study_counts[study_id],
                acquisition_dates=tuple(
                    value.isoformat() for value in sorted(study_dates.get(study_id, set()))
                ),
                series=series,
            )
        )
    return DICOMDirectorySummary(
        schema_version=DICOM_INDEX_SCHEMA,
        directory_id=directory_id,
        source_state=source_state,
        readable_file_count=readable,
        ct_file_count=sum(study_counts.values()),
        non_ct_file_count=non_ct,
        header_error_count=header_errors,
        missing_study_uid_count=missing_study,
        missing_series_uid_count=missing_series,
        studies=tuple(studies),
        header_warning_count=header_warnings,
        warning_capture_complete=True,
    )


def _load_cached_summary(
    path: Path,
    *,
    directory_id: str,
    source_state: DirectorySourceState,
) -> DICOMDirectorySummary | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            return None
        summary = DICOMDirectorySummary.from_dict(payload)
    except (OSError, ValueError, TypeError, KeyError):
        return None
    if (
        summary.schema_version != DICOM_INDEX_SCHEMA
        or summary.directory_id != directory_id
        or summary.source_state != source_state
        or not summary.warning_capture_complete
        or summary.header_warning_count < 0
        or summary.fatal_error_code is not None
    ):
        return None
    return summary


def build_dicom_metadata_index(
    directories: Iterable[Path],
    *,
    cache_root: Path,
    tokenize: Tokenizer,
    workers: int = 8,
) -> dict[Path, DICOMDirectorySummary]:
    """Index unique directories, writing each completed header summary immediately."""

    if workers < 1:
        raise DataContractError(
            code="invalid_dicom_index_workers",
            message="DICOM metadata indexing requires at least one worker.",
        )
    cache_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    cache_root.chmod(0o700)
    unique_paths = tuple(sorted({path.resolve() for path in directories}, key=str))
    results: dict[Path, DICOMDirectorySummary] = {}
    pending: dict[Any, tuple[Path, Path]] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for path in unique_paths:
            directory_id = tokenize("ct-directory", str(path), prefix="DIR")
            cache_path = cache_root / f"{directory_id}.json"
            files = _directory_files(path)
            source_state = directory_source_state(files)
            cached = _load_cached_summary(
                cache_path,
                directory_id=directory_id,
                source_state=source_state,
            )
            if cached is not None:
                results[path] = cached
                continue
            future = executor.submit(
                inspect_dicom_directory,
                path,
                directory_id=directory_id,
                tokenize=tokenize,
            )
            pending[future] = (path, cache_path)
        for future in as_completed(pending):
            path, cache_path = pending[future]
            summary = future.result()
            atomic_write_private_json(cache_path, summary.as_dict())
            results[path] = summary
    return results
