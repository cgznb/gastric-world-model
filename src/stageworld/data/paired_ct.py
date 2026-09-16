"""Clinician-signed paired-CT OS cohort adapter.

Raw workbook identifiers and local paths exist only while this adapter runs. The
returned model-facing cohort uses keyed pseudonyms, and outcome records remain a
separate collection from prefix inputs.
"""

from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml

import hashlib
import hmac
import math
import stat
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import yaml  # type: ignore[import-untyped]

from stageworld.artifacts import atomic_write_json, atomic_write_private_json, new_artifact_id
from stageworld.errors import ConfigurationError, DataContractError, StageWorldError

from .contracts import (
    AdjudicationStatus,
    AvailabilityBasis,
    Cohort,
    DataMode,
    EventType,
    Modality,
    Observation,
    ObservationRole,
    Outcome,
    Patient,
    QualityStatus,
    Query,
    QueryEligibility,
    SourceType,
    Stage,
    TimePrecision,
)
from .dicom_index import DICOMDirectorySummary, build_dicom_metadata_index
from .firewall import FeatureFirewall, FeaturePolicy
from .io import cohort_to_dict
from .outcomes import LandmarkBuilder, OutcomeBuilder, OutcomeDefinition
from .split import SplitAssignment, SplitManager

PAIRED_CT_COHORT_SCHEMA = "paired-ct-os-cohort-v2"
PAIRED_CT_ARTIFACT_SCHEMA = "paired-ct-os-build-v2"
PRIVATE_OUTCOME_SCHEMA = "private-os-labels-v1"
PRIVATE_ASSET_SCHEMA = "private-ct-asset-bindings-v2"
PRIVATE_EXCLUSION_SCHEMA = "private-cohort-exclusions-v1"
PRIVATE_SPLIT_SCHEMA = "private-patient-split-v1"
PHASE_SELECTION_POLICY_VERSION = "paired-ct-phase-selection-v1"
FIRST_ACQUISITION_SELECTION_POLICY_VERSION = "paired-ct-first-acquisition-selection-v1"
SUPPORTED_CT_SERIES_SELECTION_POLICIES = frozenset(
    {
        PHASE_SELECTION_POLICY_VERSION,
        FIRST_ACQUISITION_SELECTION_POLICY_VERSION,
    }
)

_VALIDATED_SEQUENCE_PHASE_RULES: dict[str, dict[int, str]] = {
    "3_clusters:46_to_90_seconds+16_to_45_seconds": {
        2: "arterial",
        3: "venous_or_portal",
    },
    "3_clusters:46_to_90_seconds+46_to_90_seconds": {
        2: "arterial",
        3: "venous_or_portal",
    },
    "3_clusters:over_120_seconds+46_to_90_seconds": {
        2: "arterial",
        3: "venous_or_portal",
    },
    "4_clusters:16_to_45_seconds+46_to_90_seconds+46_to_90_seconds": {
        3: "arterial",
        4: "venous_or_portal",
    },
}
_MATCHED_PHASE_PRIORITY = ("venous_or_portal", "arterial")

_EXPECTED_OS_V1_COLUMNS = {
    "row_key": "A",
    "baseline_origin": "E",
    "os_status": "BS",
    "death_date": "BU",
    "censor_date": "BV",
    "baseline_ct_link": "BW",
    "post_treatment_ct_link": "BZ",
}
_MISSING_TEXT = frozenset(
    {
        "",
        "#n/a",
        "#na",
        "n/a",
        "na",
        "nan",
        "nat",
        "none",
        "null",
        "missing",
        "unknown",
        "-",
        "--",
    }
)


@dataclass(frozen=True, slots=True)
class PairedCTMapping:
    sheet_index: int
    field_header_row: int
    data_start_row: int
    expected_columns: int
    columns: Mapping[str, str]
    event_code: int
    censor_code: int
    outcome_label_version: str
    cohort_version: str
    permit_real_supervised_build: bool
    require_complete_pair: bool
    elapsed_time_only: bool

    def validate(self) -> None:
        if self.sheet_index < 0 or self.field_header_row < 1:
            raise ConfigurationError(
                code="INVALID_PAIRED_CT_MAPPING",
                message="Workbook sheet and header positions must be nonnegative/positive.",
            )
        if self.data_start_row <= self.field_header_row or self.expected_columns < 1:
            raise ConfigurationError(
                code="INVALID_PAIRED_CT_MAPPING",
                message="Workbook row/column bounds are inconsistent.",
            )
        actual = {name: str(self.columns.get(name, "")).upper() for name in _EXPECTED_OS_V1_COLUMNS}
        if actual != _EXPECTED_OS_V1_COLUMNS:
            raise ConfigurationError(
                code="OS_V1_COLUMN_CONTRACT_MISMATCH",
                message="The paired-CT OS-v1 source columns differ from the signed contract.",
                details={"required_fields": sorted(_EXPECTED_OS_V1_COLUMNS)},
            )
        if self.event_code != 1 or self.censor_code != 0:
            raise ConfigurationError(
                code="OS_V1_STATUS_CONTRACT_MISMATCH",
                message="OS-v1 requires death=1 and alive/censored=0.",
            )
        if not self.outcome_label_version or not self.cohort_version:
            raise ConfigurationError(
                code="INVALID_PAIRED_CT_MAPPING",
                message="Outcome and cohort versions must be explicit.",
            )
        if not self.permit_real_supervised_build:
            raise ConfigurationError(
                code="REAL_COHORT_MAPPING_NOT_APPROVED",
                message="The selected field mapping is not approved for real cohort construction.",
            )
        if not self.require_complete_pair:
            raise ConfigurationError(
                code="PAIRED_COHORT_SCOPE_MISMATCH",
                message="paired-CT-v1 requires both baseline and post-treatment CT assets.",
            )
        if not self.elapsed_time_only:
            raise ConfigurationError(
                code="UNCONFIRMED_TREATMENT_CONDITIONING",
                message="This first paired-CT contract supports elapsed-time conditioning only.",
            )


@dataclass(frozen=True, slots=True)
class CohortExclusion:
    source_row: int
    patient_id: str | None
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CTAssetBinding:
    asset_id: str
    patient_id: str
    role: str
    local_path: str
    directory_id: str
    replica_count: int
    ct_file_count: int
    header_warning_count: int
    warning_capture_complete: bool
    study_id: str
    acquisition_date_local: str
    selected_series_id: str
    selected_phase: str
    selected_phase_basis: str
    pair_phase_status: str
    phase_selection_policy_version: str
    candidate_series: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class PairedCTBuildResult:
    input_cohort: Cohort
    outcomes: tuple[Outcome, ...]
    assignments: tuple[SplitAssignment, ...]
    bindings: tuple[CTAssetBinding, ...]
    exclusions: tuple[CohortExclusion, ...]
    source_record_count: int
    rows_with_both_links: int
    directory_matched_pair_count: int
    metadata_usable_pair_count: int
    linked_pair_count: int
    equivalent_replica_link_count: int
    stage_landmark_counts: Mapping[str, int]
    stage_event_counts: Mapping[str, int]
    cohort_version: str
    outcome_label_version: str
    series_selection_policy_version: str


@dataclass(frozen=True, slots=True)
class _StudySeriesSelection:
    candidate_series: tuple[Mapping[str, Any], ...]
    selected_series_id: str
    selected_phase: str
    selected_phase_basis: str


@dataclass(slots=True, repr=False)
class _SourceRow:
    source_row: int
    row_key: str | None
    origin_date: date | None
    origin_error: str | None
    status: int | None
    status_error: str | None
    death_date: date | None
    death_error: str | None
    censor_date: date | None
    censor_error: str | None
    baseline_link: str | None
    post_link: str | None


class HMACPseudonymizer:
    """Generate stable, namespace-separated local IDs without retaining source values."""

    def __init__(self, key: bytes) -> None:
        if len(key) < 32:
            raise ConfigurationError(
                code="HMAC_KEY_TOO_SHORT",
                message="The identity HMAC key must contain at least 32 bytes.",
            )
        self._key = key

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        project_root: str | Path | None = None,
    ) -> HMACPseudonymizer:
        source = Path(path).expanduser().resolve()
        if project_root is not None and source.is_relative_to(Path(project_root).resolve()):
            raise ConfigurationError(
                code="HMAC_KEY_INSIDE_PROJECT",
                message="The identity HMAC key must remain outside the project tree.",
            )
        try:
            status = source.stat()
            if not stat.S_ISREG(status.st_mode):
                raise OSError("not a regular file")
            if stat.S_IMODE(status.st_mode) & 0o077:
                raise ConfigurationError(
                    code="HMAC_KEY_PERMISSIONS_UNSAFE",
                    message="The identity HMAC key must have owner-only permissions.",
                    remediation="Set the key file mode to 0600.",
                )
            key = source.read_bytes()
        except ConfigurationError:
            raise
        except OSError as exc:
            raise ConfigurationError(
                code="HMAC_KEY_UNREADABLE",
                message="The configured identity HMAC key cannot be read.",
            ) from exc
        return cls(key)

    def token(self, namespace: str, value: str, *, prefix: str) -> str:
        if not namespace or not value or not prefix:
            raise DataContractError(
                code="INVALID_PSEUDONYM_INPUT",
                message="Pseudonym namespace, value, and prefix must be nonempty.",
            )
        payload = namespace.encode("ascii") + b"\x00" + value.encode("utf-8")
        digest = hmac.new(self._key, payload, hashlib.sha256).hexdigest()[:32]
        return f"{prefix}-{digest}"


def _mapping_section(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(
            code="INVALID_PAIRED_CT_MAPPING",
            message=f"Field mapping section '{name}' must be a mapping.",
        )
    return {str(key): item for key, item in value.items()}


def load_paired_ct_mapping(path: str | Path) -> PairedCTMapping:
    source = Path(path)
    try:
        payload = _release_yaml(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(
            code="PAIRED_CT_MAPPING_UNREADABLE",
            message="The paired-CT field mapping cannot be read.",
        ) from exc
    root = _mapping_section(payload, "root")
    workbook = _mapping_section(root.get("workbook"), "workbook")
    fields = _mapping_section(root.get("fields"), "fields")
    rules = _mapping_section(root.get("rules"), "rules")
    contract = _mapping_section(root.get("contract"), "contract")
    columns: dict[str, str] = {}
    for name in _EXPECTED_OS_V1_COLUMNS:
        field = _mapping_section(fields.get(name), f"fields.{name}")
        columns[name] = str(field.get("source_column", ""))
    try:
        result = PairedCTMapping(
            sheet_index=int(workbook.get("sheet_index", 0)),
            field_header_row=int(workbook["field_header_row"]),
            data_start_row=int(workbook["data_start_row"]),
            expected_columns=int(workbook["expected_columns"]),
            columns=columns,
            event_code=int(contract["event_code"]),
            censor_code=int(contract["censor_code"]),
            outcome_label_version=str(contract["outcome_label_version"]),
            cohort_version=str(contract["cohort_version"]),
            permit_real_supervised_build=bool(rules["permit_real_supervised_build"]),
            require_complete_pair=bool(rules["require_complete_pair"]),
            elapsed_time_only=bool(rules["elapsed_time_only"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(
            code="INVALID_PAIRED_CT_MAPPING",
            message="The paired-CT field mapping is incomplete or incorrectly typed.",
        ) from exc
    result.validate()
    return result


def _require_openpyxl() -> Any:
    try:
        import openpyxl  # type: ignore[import-untyped]
    except ImportError as exc:
        raise StageWorldError(
            code="dependency_missing",
            message="Paired cohort construction requires openpyxl.",
            remediation="Install the clinical-io optional dependency.",
            details={"dependency": "openpyxl"},
        ) from exc
    return openpyxl


def _column_index(letter: str) -> int:
    normalized = letter.strip().upper()
    if not normalized or any(character < "A" or character > "Z" for character in normalized):
        raise ConfigurationError(
            code="INVALID_EXCEL_COLUMN",
            message="Mapped Excel columns must use A-Z letter notation.",
        )
    value = 0
    for character in normalized:
        value = value * 26 + ord(character) - ord("A") + 1
    return value


def _identifier(value: object, data_type: str | None = None) -> str | None:
    if value is None or data_type == "e" or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return str(int(value)) if value.is_integer() else str(value).strip()
    text = str(value).strip()
    if not text or text.casefold() in _MISSING_TEXT:
        return None
    normalized_path = text.replace("\\", "/")
    candidate = PurePosixPath(normalized_path).name.strip()
    if candidate in {"", ".", ".."} or "\x00" in candidate:
        return None
    return candidate


def _binary_status(value: object, data_type: str | None) -> tuple[int | None, str | None]:
    if value is None:
        return None, "os_status_missing"
    if data_type == "e":
        return None, "os_status_excel_error"
    if isinstance(value, bool):
        return None, "os_status_invalid"
    if isinstance(value, int) and value in {0, 1}:
        return value, None
    if isinstance(value, float) and math.isfinite(value) and value in {0.0, 1.0}:
        return int(value), None
    text = str(value).strip()
    if text in {"0", "1"}:
        return int(text), None
    return None, "os_status_invalid"


def _date_value(
    value: object,
    data_type: str | None,
    *,
    epoch: object,
    field_name: str,
) -> tuple[date | None, str | None]:
    if value is None:
        return None, f"{field_name}_missing"
    if data_type == "e":
        return None, f"{field_name}_excel_error"
    parsed: date | None = None
    if isinstance(value, datetime):
        parsed = value.date()
    elif isinstance(value, date):
        parsed = value
    elif isinstance(value, bool):
        parsed = None
    elif isinstance(value, (int, float)) and math.isfinite(float(value)):
        try:
            from openpyxl.utils.datetime import from_excel  # type: ignore[import-untyped]

            converted = from_excel(value, epoch)
            if isinstance(converted, datetime):
                parsed = converted.date()
            elif isinstance(converted, date):
                parsed = converted
        except (TypeError, ValueError, OverflowError):
            parsed = None
    elif isinstance(value, str):
        text = value.strip()
        if text.casefold() not in _MISSING_TEXT:
            try:
                parsed = datetime.fromisoformat(text).date()
            except ValueError:
                for date_format in (
                    "%Y/%m/%d",
                    "%Y.%m.%d",
                    "%Y%m%d",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y/%m/%d %H:%M:%S",
                    "%Y\u5e74%m\u6708%d\u65e5",
                ):
                    try:
                        parsed = datetime.strptime(text, date_format).date()
                        break
                    except ValueError:
                        continue
    if parsed is None or not date(1900, 1, 1) <= parsed <= date(2100, 12, 31):
        return None, f"{field_name}_invalid"
    return parsed, None


def _load_rows(path: Path, mapping: PairedCTMapping) -> tuple[_SourceRow, ...]:
    openpyxl = _require_openpyxl()
    try:
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=False)
    except (OSError, ValueError) as exc:
        raise DataContractError(
            code="excel_unreadable",
            message="Unable to read the configured clinical workbook.",
        ) from exc
    if mapping.sheet_index >= len(workbook.worksheets):
        raise DataContractError(
            code="excel_sheet_missing",
            message="The configured worksheet index is absent.",
        )
    worksheet = workbook.worksheets[mapping.sheet_index]
    if worksheet.max_column != mapping.expected_columns:
        raise DataContractError(
            code="excel_schema_mismatch",
            message="Workbook column count differs from the signed paired-CT mapping.",
            details={"expected_columns": mapping.expected_columns},
        )
    column_indices = {name: _column_index(letter) for name, letter in mapping.columns.items()}
    max_column = max(column_indices.values())
    rows: list[_SourceRow] = []
    for source_row, cells in enumerate(
        worksheet.iter_rows(
            min_row=mapping.data_start_row,
            max_col=max_column,
            values_only=False,
        ),
        start=mapping.data_start_row,
    ):
        selected = {name: cells[index - 1] for name, index in column_indices.items()}
        origin, origin_error = _date_value(
            selected["baseline_origin"].value,
            selected["baseline_origin"].data_type,
            epoch=workbook.epoch,
            field_name="baseline_origin",
        )
        status, status_error = _binary_status(
            selected["os_status"].value, selected["os_status"].data_type
        )
        death, death_error = _date_value(
            selected["death_date"].value,
            selected["death_date"].data_type,
            epoch=workbook.epoch,
            field_name="death_date",
        )
        censor, censor_error = _date_value(
            selected["censor_date"].value,
            selected["censor_date"].data_type,
            epoch=workbook.epoch,
            field_name="censor_date",
        )
        rows.append(
            _SourceRow(
                source_row=source_row,
                row_key=_identifier(
                    selected["row_key"].value, selected["row_key"].data_type
                ),
                origin_date=origin,
                origin_error=origin_error,
                status=status,
                status_error=status_error,
                death_date=death,
                death_error=death_error,
                censor_date=censor,
                censor_error=censor_error,
                baseline_link=_identifier(
                    selected["baseline_ct_link"].value,
                    selected["baseline_ct_link"].data_type,
                ),
                post_link=_identifier(
                    selected["post_treatment_ct_link"].value,
                    selected["post_treatment_ct_link"].data_type,
                ),
            )
        )
    workbook.close()
    return tuple(rows)


def _discover_directories(root: Path, requested: set[str]) -> dict[str, tuple[Path, ...]]:
    resolved_root = root.expanduser().resolve()
    if not resolved_root.is_dir():
        raise DataContractError(
            code="ct_root_unreadable",
            message="The configured CT root is not a readable directory.",
        )
    found: dict[str, list[Path]] = defaultdict(list)
    try:
        for batch in resolved_root.iterdir():
            if not batch.is_dir():
                continue
            for candidate in batch.iterdir():
                if not candidate.is_dir() or candidate.name not in requested:
                    continue
                resolved = candidate.resolve()
                if not resolved.is_relative_to(resolved_root):
                    raise DataContractError(
                        code="ct_asset_outside_root",
                        message="A linked CT directory resolves outside the approved CT root.",
                    )
                found[candidate.name].append(resolved)
    except OSError as exc:
        raise DataContractError(
            code="ct_root_unreadable",
            message="The configured CT root cannot be enumerated.",
        ) from exc
    return {key: tuple(sorted(paths, key=str)) for key, paths in found.items()}


def _series_payload(summary: DICOMDirectorySummary) -> tuple[Mapping[str, Any], ...]:
    if len(summary.studies) != 1:
        return ()
    payload = tuple(
        {
            "series_id": series.series_id,
            "rank": rank,
            "file_count": series.file_count,
            "rows": series.rows,
            "columns": series.columns,
            "positioned_slice_count": series.positioned_slice_count,
            "unique_position_count": series.unique_position_count,
            "median_slice_thickness_mm": series.median_slice_thickness_mm,
            "phase_candidates": list(series.phase_candidates),
            "contrast_tag_fraction": series.contrast_tag_fraction,
            "acquisition_time_source": series.acquisition_time_source,
            "acquisition_offset_seconds": series.acquisition_offset_seconds,
            "temporal_cluster_index": series.temporal_cluster_index,
            "temporal_cluster_count": series.temporal_cluster_count,
            "contrast_bolus_delay_seconds": series.contrast_bolus_delay_seconds,
            "timing_phase_candidate": series.timing_phase_candidate,
            "timing_phase_basis": series.timing_phase_basis,
            "timing_phase_confidence": series.timing_phase_confidence,
            "series_number": series.series_number,
            "acquisition_number": series.acquisition_number,
            "plausible_volume": series.plausible_volume,
            "candidate_score": series.candidate_score,
            "selected": False,
        }
        for rank, series in enumerate(summary.studies[0].series, start=1)
    )
    return _annotate_validated_sequence_phases(payload)


def _header_confident_series_phase(series: Mapping[str, Any]) -> str | None:
    explicit = tuple(str(item) for item in series.get("phase_candidates", ()))
    timing = series.get("timing_phase_candidate")
    confidence = str(series.get("timing_phase_confidence", "none"))
    if len(explicit) == 1:
        if (
            timing is not None
            and confidence in {"high", "moderate"}
            and str(timing) != explicit[0]
        ):
            return None
        return explicit[0]
    if timing is not None and confidence in {"high", "moderate"}:
        return str(timing)
    return None


def _confident_series_phase(series: Mapping[str, Any]) -> str | None:
    header_phase = _header_confident_series_phase(series)
    if header_phase is not None:
        return header_phase
    sequence_phase = series.get("sequence_phase_candidate")
    sequence_confidence = str(series.get("sequence_phase_confidence", "none"))
    if sequence_phase is not None and sequence_confidence == "moderate":
        return str(sequence_phase)
    return None


def _series_phase_basis(series: Mapping[str, Any]) -> str:
    explicit = tuple(str(item) for item in series.get("phase_candidates", ()))
    header_phase = _header_confident_series_phase(series)
    if len(explicit) == 1 and header_phase is not None:
        return "explicit_dicom_text"
    timing = series.get("timing_phase_candidate")
    timing_confidence = str(series.get("timing_phase_confidence", "none"))
    if (
        header_phase is not None
        and timing is not None
        and timing_confidence in {"high", "moderate"}
    ):
        return str(series.get("timing_phase_basis", "dicom_timing"))
    if series.get("sequence_phase_candidate") is not None:
        return str(series.get("sequence_phase_basis", "validated_sequence_signature"))
    return "unknown"


def _temporal_gap_bucket(gap_seconds: float) -> str:
    if gap_seconds <= 15.0:
        return "0_to_15_seconds"
    if gap_seconds <= 45.0:
        return "16_to_45_seconds"
    if gap_seconds <= 90.0:
        return "46_to_90_seconds"
    if gap_seconds <= 120.0:
        return "91_to_120_seconds"
    return "over_120_seconds"


def _bolus_delay_bucket(delay_seconds: float) -> str:
    if delay_seconds < -3600.0:
        return "before_minus_3600_seconds"
    if delay_seconds < -5.0:
        return "minus_3600_to_minus_5_seconds"
    if delay_seconds < 10.0:
        return "minus_5_to_9_seconds"
    if delay_seconds <= 45.0:
        return "10_to_45_seconds"
    if delay_seconds < 50.0:
        return "46_to_49_seconds"
    if delay_seconds <= 100.0:
        return "50_to_100_seconds"
    if delay_seconds < 120.0:
        return "101_to_119_seconds"
    if delay_seconds <= 600.0:
        return "120_to_600_seconds"
    return "over_600_seconds"


def _plausible_cluster_profile(
    candidate_series: tuple[Mapping[str, Any], ...],
) -> tuple[list[int], dict[int, float], str]:
    cluster_offsets: dict[int, float] = {}
    for series in candidate_series:
        if not bool(series.get("plausible_volume")):
            continue
        cluster = series.get("temporal_cluster_index")
        offset = series.get("acquisition_offset_seconds")
        if cluster is None or offset is None:
            continue
        cluster_index = int(cluster)
        offset_seconds = float(offset)
        previous = cluster_offsets.get(cluster_index)
        if previous is None or offset_seconds < previous:
            cluster_offsets[cluster_index] = offset_seconds
    cluster_indices = sorted(cluster_offsets)
    ordered_offsets = [cluster_offsets[index] for index in cluster_indices]
    gap_signature = "+".join(
        _temporal_gap_bucket(later - earlier)
        for earlier, later in zip(ordered_offsets, ordered_offsets[1:], strict=False)
    )
    signature = f"{len(cluster_indices)}_clusters"
    if gap_signature:
        signature = f"{signature}:{gap_signature}"
    return cluster_indices, cluster_offsets, signature


def _annotate_validated_sequence_phases(
    candidate_series: tuple[Mapping[str, Any], ...],
) -> tuple[Mapping[str, Any], ...]:
    cluster_indices, _, signature = _plausible_cluster_profile(candidate_series)
    phase_by_position = _VALIDATED_SEQUENCE_PHASE_RULES.get(signature)
    if phase_by_position is None:
        return candidate_series
    position_by_cluster = {
        cluster: position for position, cluster in enumerate(cluster_indices, start=1)
    }
    for series in candidate_series:
        existing_phase = _header_confident_series_phase(series)
        cluster = series.get("temporal_cluster_index")
        if existing_phase not in _MATCHED_PHASE_PRIORITY or cluster is None:
            continue
        position = position_by_cluster.get(int(cluster))
        if position is None or phase_by_position.get(position) != existing_phase:
            return candidate_series

    annotated: list[Mapping[str, Any]] = []
    for raw_series in candidate_series:
        series = dict(raw_series)
        cluster = series.get("temporal_cluster_index")
        position = None if cluster is None else position_by_cluster.get(int(cluster))
        inferred_phase = None if position is None else phase_by_position.get(position)
        if (
            bool(series.get("plausible_volume"))
            and inferred_phase is not None
            and _header_confident_series_phase(series) is None
        ):
            series.update(
                {
                    "sequence_phase_candidate": inferred_phase,
                    "sequence_phase_basis": "validated_sequence_signature",
                    "sequence_phase_confidence": "moderate",
                    "sequence_signature": signature,
                }
            )
        annotated.append(series)
    return tuple(annotated)


def _best_series_for_phase(
    candidate_series: tuple[Mapping[str, Any], ...],
    phase: str,
) -> Mapping[str, Any] | None:
    return next(
        (
            series
            for series in candidate_series
            if bool(series.get("plausible_volume"))
            and _confident_series_phase(series) == phase
        ),
        None,
    )


def _preferred_study_series(
    candidate_series: tuple[Mapping[str, Any], ...],
) -> Mapping[str, Any]:
    for phase in _MATCHED_PHASE_PRIORITY:
        selected = _best_series_for_phase(candidate_series, phase)
        if selected is not None:
            return selected
    selected = next(
        (series for series in candidate_series if bool(series.get("plausible_volume"))),
        None,
    )
    if selected is None:
        raise DataContractError(
            code="NO_SELECTABLE_CT_SERIES",
            message="A retained CT study has no selectable plausible volume.",
        )
    return selected


def _mark_selected_series(
    candidate_series: tuple[Mapping[str, Any], ...],
    selected: Mapping[str, Any],
    *,
    pair_phase_status: str,
    selection_policy_version: str,
    selected_phase_override: str | None = None,
    selected_basis_override: str | None = None,
) -> _StudySeriesSelection:
    selected_id = str(selected["series_id"])
    selected_phase = selected_phase_override or _confident_series_phase(selected) or "unknown"
    selected_basis = selected_basis_override or _series_phase_basis(selected)
    annotated = tuple(
        {
            **series,
            "selected": str(series["series_id"]) == selected_id,
            "selected_phase": (
                selected_phase if str(series["series_id"]) == selected_id else None
            ),
            "selected_phase_basis": (
                selected_basis if str(series["series_id"]) == selected_id else None
            ),
            "pair_phase_status": pair_phase_status,
            "phase_selection_policy_version": selection_policy_version,
        }
        for series in candidate_series
    )
    return _StudySeriesSelection(
        candidate_series=annotated,
        selected_series_id=selected_id,
        selected_phase=selected_phase,
        selected_phase_basis=selected_basis,
    )


def _select_paired_series(
    baseline_series: tuple[Mapping[str, Any], ...],
    post_series: tuple[Mapping[str, Any], ...],
    *,
    selection_policy_version: str,
) -> tuple[_StudySeriesSelection, _StudySeriesSelection, str]:
    if selection_policy_version == FIRST_ACQUISITION_SELECTION_POLICY_VERSION:
        pair_phase_status = "paired_first_acquisition"
        first_baseline = _first_acquisition_series(baseline_series)
        first_post = _first_acquisition_series(post_series)
        selection_kwargs = {
            "pair_phase_status": pair_phase_status,
            "selection_policy_version": selection_policy_version,
            "selected_phase_override": "first_acquisition_unknown",
            "selected_basis_override": "earliest_plausible_temporal_cluster",
        }
        return (
            _mark_selected_series(baseline_series, first_baseline, **selection_kwargs),
            _mark_selected_series(post_series, first_post, **selection_kwargs),
            pair_phase_status,
        )
    if selection_policy_version != PHASE_SELECTION_POLICY_VERSION:
        raise ConfigurationError(
            code="INVALID_CT_SERIES_SELECTION_POLICY",
            message="The requested CT series-selection policy is not supported.",
            details={"policy": selection_policy_version},
        )
    baseline_selected: Mapping[str, Any] | None = None
    post_selected: Mapping[str, Any] | None = None
    pair_phase_status = "fallback_no_common_confident_phase"
    for phase in _MATCHED_PHASE_PRIORITY:
        baseline_candidate = _best_series_for_phase(baseline_series, phase)
        post_candidate = _best_series_for_phase(post_series, phase)
        if baseline_candidate is not None and post_candidate is not None:
            baseline_selected = baseline_candidate
            post_selected = post_candidate
            pair_phase_status = f"matched_{phase}"
            break
    if baseline_selected is None or post_selected is None:
        baseline_selected = _preferred_study_series(baseline_series)
        post_selected = _preferred_study_series(post_series)
    return (
        _mark_selected_series(
            baseline_series,
            baseline_selected,
            pair_phase_status=pair_phase_status,
            selection_policy_version=selection_policy_version,
        ),
        _mark_selected_series(
            post_series,
            post_selected,
            pair_phase_status=pair_phase_status,
            selection_policy_version=selection_policy_version,
        ),
        pair_phase_status,
    )


def _first_acquisition_series(
    candidate_series: tuple[Mapping[str, Any], ...],
) -> Mapping[str, Any]:
    timed = [
        series
        for series in candidate_series
        if bool(series.get("plausible_volume"))
        and series.get("temporal_cluster_index") is not None
        and series.get("acquisition_offset_seconds") is not None
    ]
    if not timed:
        raise DataContractError(
            code="NO_TIMED_PLAUSIBLE_CT_SERIES",
            message=(
                "First-acquisition selection requires acquisition timing for at least one "
                "plausible CT volume."
            ),
        )
    earliest = min(
        timed,
        key=lambda series: (
            float(series["acquisition_offset_seconds"]),
            int(series["temporal_cluster_index"]),
        ),
    )
    earliest_cluster = int(earliest["temporal_cluster_index"])
    return next(
        series
        for series in candidate_series
        if bool(series.get("plausible_volume"))
        and series.get("temporal_cluster_index") is not None
        and int(series["temporal_cluster_index"]) == earliest_cluster
    )


def _paired_phase_counts(
    patient_phase_sets: Mapping[str, Mapping[str, set[str]]],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for phase_sets in patient_phase_sets.values():
        baseline = phase_sets.get(ObservationRole.BASELINE_CT.value, set())
        post = phase_sets.get(ObservationRole.POST_TREATMENT_CT.value, set())
        shared = baseline & post
        counts["both_have_same_confident_phase"] += int(bool(shared))
        counts["both_have_venous_or_portal"] += int("venous_or_portal" in shared)
        counts["both_have_arterial"] += int("arterial" in shared)
        counts["at_least_one_exam_unresolved"] += int(not baseline or not post)
        counts["neither_exam_resolved"] += int(not baseline and not post)
    return counts


def _resolve_asset(
    paths: tuple[Path, ...],
    metadata: Mapping[Path, DICOMDirectorySummary],
) -> tuple[Path | None, DICOMDirectorySummary | None, str | None, bool]:
    summaries = [(path, metadata[path]) for path in paths]
    for _, summary in summaries:
        if summary.fatal_error_code is not None:
            return None, None, "dicom_index_failure", False
        if summary.header_error_count:
            return None, None, "dicom_header_error", False
        if summary.missing_study_uid_count or summary.missing_series_uid_count:
            return None, None, "dicom_identity_missing", False
        if summary.ct_file_count == 0:
            return None, None, "no_ct_objects", False
        if len(summary.studies) != 1:
            return None, None, "ct_study_count_not_one", False
        if summary.single_study_date is None:
            return None, None, "ct_acquisition_date_ambiguous", False
        if not any(series.plausible_volume for series in summary.studies[0].series):
            return None, None, "no_plausible_ct_volume", False
    if len(summaries) == 1:
        return summaries[0][0], summaries[0][1], None, False
    study_ids = {summary.study_ids for _, summary in summaries}
    study_dates = {summary.single_study_date for _, summary in summaries}
    if len(study_ids) != 1 or len(study_dates) != 1:
        return None, None, "ambiguous_duplicate_ct_directory", False
    selected = sorted(summaries, key=lambda item: (-item[1].ct_file_count, str(item[0])))[0]
    return selected[0], selected[1], None, True


def _add_reason(reasons: dict[int, set[str]], row: _SourceRow, reason: str) -> None:
    reasons[row.source_row].add(reason)


def build_paired_ct_cohort(
    *,
    workbook_path: str | Path,
    ct_root: str | Path,
    mapping: PairedCTMapping,
    pseudonymizer: HMACPseudonymizer,
    mode: DataMode,
    metadata_cache_root: str | Path,
    split_seed: int,
    prediction_horizon_days: float,
    selection_policy_version: str = PHASE_SELECTION_POLICY_VERSION,
    workers: int = 8,
) -> PairedCTBuildResult:
    """Build an S0/S1 transition cohort without exposing outcome data to prefixes."""

    mapping.validate()
    if selection_policy_version not in SUPPORTED_CT_SERIES_SELECTION_POLICIES:
        raise ConfigurationError(
            code="INVALID_CT_SERIES_SELECTION_POLICY",
            message="The requested CT series-selection policy is not supported.",
            details={"policy": selection_policy_version},
        )
    if mode is DataMode.SYNTHETIC:
        raise ConfigurationError(
            code="REAL_MODE_REQUIRED",
            message="The paired clinical adapter cannot run in synthetic mode.",
        )
    if prediction_horizon_days <= 0 or not math.isfinite(prediction_horizon_days):
        raise ConfigurationError(
            code="INVALID_PREDICTION_HORIZON",
            message="The paired cohort prediction horizon must be positive and finite.",
        )
    rows = _load_rows(Path(workbook_path), mapping)
    reasons: dict[int, set[str]] = defaultdict(set)
    patient_ids: dict[int, str] = {}
    both_links = 0
    for row in rows:
        if row.row_key is None:
            _add_reason(reasons, row, "row_key_missing_or_invalid")
        else:
            patient_ids[row.source_row] = pseudonymizer.token(
                "patient", row.row_key, prefix="P"
            )
        if row.origin_date is None:
            _add_reason(reasons, row, row.origin_error or "baseline_origin_invalid")
        if row.status is None:
            _add_reason(reasons, row, row.status_error or "os_status_invalid")
        if row.baseline_link is None:
            _add_reason(reasons, row, "baseline_ct_link_missing_or_invalid")
        if row.post_link is None:
            _add_reason(reasons, row, "post_ct_link_missing_or_invalid")
        if row.baseline_link is not None and row.post_link is not None:
            both_links += 1
            if row.baseline_link == row.post_link:
                _add_reason(reasons, row, "baseline_and_post_ct_link_identical")
        if row.status == mapping.event_code:
            if row.death_date is None:
                _add_reason(reasons, row, row.death_error or "death_date_invalid")
        elif row.status == mapping.censor_code:
            if row.censor_date is None:
                _add_reason(reasons, row, row.censor_error or "censor_date_invalid")
            if row.death_date is not None:
                _add_reason(reasons, row, "censored_status_with_death_date")

    row_key_counts = Counter(row.row_key for row in rows if row.row_key is not None)
    for row in rows:
        if row.row_key is not None and row_key_counts[row.row_key] > 1:
            _add_reason(reasons, row, "duplicate_row_key")

    link_to_rows: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        for link in (row.baseline_link, row.post_link):
            if link is not None:
                link_to_rows[link].add(row.source_row)
    shared_links = {link for link, source_rows in link_to_rows.items() if len(source_rows) > 1}
    for row in rows:
        if row.baseline_link in shared_links or row.post_link in shared_links:
            _add_reason(reasons, row, "ct_link_shared_across_rows")

    requested_links = set(link_to_rows)
    directories = _discover_directories(Path(ct_root), requested_links)
    directory_matched_pairs = sum(
        row.baseline_link is not None
        and row.post_link is not None
        and row.baseline_link in directories
        and row.post_link in directories
        for row in rows
    )
    for row in rows:
        if row.baseline_link is not None and row.baseline_link not in directories:
            _add_reason(reasons, row, "baseline_ct_directory_missing")
        if row.post_link is not None and row.post_link not in directories:
            _add_reason(reasons, row, "post_ct_directory_missing")

    paths_to_index = {
        path
        for link, paths in directories.items()
        if link in requested_links
        for path in paths
    }
    metadata = build_dicom_metadata_index(
        paths_to_index,
        cache_root=Path(metadata_cache_root),
        tokenize=pseudonymizer.token,
        workers=workers,
    )
    resolved: dict[str, tuple[Path, DICOMDirectorySummary, bool]] = {}
    replica_count = 0
    for link, paths in directories.items():
        selected_path, summary, error, equivalent_replica = _resolve_asset(paths, metadata)
        if error is None and selected_path is not None and summary is not None:
            resolved[link] = (selected_path, summary, equivalent_replica)
            replica_count += int(equivalent_replica)
            continue
        for source_row in link_to_rows[link]:
            reasons[source_row].add(error or "ct_asset_resolution_failed")
    metadata_usable_pairs = sum(
        row.baseline_link is not None
        and row.post_link is not None
        and row.baseline_link in resolved
        and row.post_link in resolved
        for row in rows
    )

    patients: list[Patient] = []
    observations: list[Observation] = []
    outcomes: list[Outcome] = []
    queries: list[Query] = []
    bindings: list[CTAssetBinding] = []
    linked_pairs = 0
    for row in rows:
        if reasons[row.source_row]:
            continue
        assert row.row_key is not None
        assert row.origin_date is not None
        assert row.status is not None
        assert row.baseline_link is not None and row.post_link is not None
        assert row.baseline_link in resolved and row.post_link in resolved
        baseline_path, baseline_metadata, baseline_replica = resolved[row.baseline_link]
        post_path, post_metadata, post_replica = resolved[row.post_link]
        baseline_date = baseline_metadata.single_study_date
        post_date = post_metadata.single_study_date
        assert baseline_date is not None and post_date is not None
        if post_date <= baseline_date:
            _add_reason(reasons, row, "ct_pair_not_chronological")
            continue
        baseline_days = float((baseline_date - row.origin_date).days)
        post_days = float((post_date - row.origin_date).days)
        if post_days < 0:
            _add_reason(reasons, row, "post_ct_before_os_origin")
            continue
        terminal_date = (
            row.death_date if row.status == mapping.event_code else row.censor_date
        )
        assert terminal_date is not None
        terminal_days = float((terminal_date - row.origin_date).days)
        if terminal_days <= 0:
            _add_reason(reasons, row, "os_terminal_not_after_origin")
            continue
        if terminal_days <= post_days:
            _add_reason(reasons, row, "os_terminal_not_after_post_ct")
            continue

        patient_id = patient_ids[row.source_row]
        baseline_asset_id = pseudonymizer.token(
            "ct-asset", row.baseline_link, prefix="CT"
        )
        post_asset_id = pseudonymizer.token("ct-asset", row.post_link, prefix="CT")
        baseline_selection, post_selection, pair_phase_status = _select_paired_series(
            _series_payload(baseline_metadata),
            _series_payload(post_metadata),
            selection_policy_version=selection_policy_version,
        )
        baseline_available = max(0.0, baseline_days)
        patients.append(
            Patient(
                patient_id=patient_id,
                site_id="LOCAL-SITE-1",
                cohort_id="paired-ct-transition",
                eligibility_version=mapping.cohort_version,
                baseline_origin_local=0.0,
                index_event_type="os_origin_column_E",
            )
        )
        observations.extend(
            (
                Observation(
                    observation_id=f"{patient_id}-ct-s0",
                    patient_id=patient_id,
                    modality=Modality.CT,
                    role=ObservationRole.BASELINE_CT,
                    source_type=SourceType.OBSERVED,
                    acquired_at_days=baseline_days,
                    available_at_days=baseline_available,
                    time_precision=TimePrecision.DAY,
                    availability_basis=AvailabilityBasis.PROTOCOL_DEFINED_ORDER,
                    local_asset_id=baseline_asset_id,
                    quality_status=QualityStatus.UNKNOWN,
                    phase=baseline_selection.selected_phase,
                ),
                Observation(
                    observation_id=f"{patient_id}-ct-s1",
                    patient_id=patient_id,
                    modality=Modality.CT,
                    role=ObservationRole.POST_TREATMENT_CT,
                    source_type=SourceType.OBSERVED,
                    acquired_at_days=post_days,
                    available_at_days=post_days,
                    time_precision=TimePrecision.DAY,
                    availability_basis=AvailabilityBasis.PROTOCOL_DEFINED_ORDER,
                    local_asset_id=post_asset_id,
                    quality_status=QualityStatus.UNKNOWN,
                    phase=post_selection.selected_phase,
                ),
            )
        )
        outcomes.append(
            Outcome(
                patient_id=patient_id,
                endpoint_name="os",
                event_type=EventType.DEATH,
                source_status=row.status,
                event_date_days=(terminal_days if row.status == mapping.event_code else None),
                censor_date_days=(terminal_days if row.status == mapping.censor_code else None),
                adjudication_status=AdjudicationStatus.CONFIRMED,
                origin_definition="clinical_excel_column_E",
                reason_of_censoring=(
                    None
                    if row.status == mapping.event_code
                    else "alive_or_censored_at_column_BV"
                ),
                label_version=mapping.outcome_label_version,
            )
        )
        queries.extend(
            (
                Query(
                    query_id=f"{patient_id}-s0",
                    patient_id=patient_id,
                    stage=Stage.S0,
                    query_time_days=baseline_available,
                    prediction_horizon_days=prediction_horizon_days,
                    eligibility=QueryEligibility.ELIGIBLE,
                    eligibility_basis="retrospective_complete_pair_transition_cohort",
                    target_time_days=post_days,
                ),
                Query(
                    query_id=f"{patient_id}-s1",
                    patient_id=patient_id,
                    stage=Stage.S1,
                    query_time_days=post_days,
                    prediction_horizon_days=prediction_horizon_days,
                    eligibility=QueryEligibility.ELIGIBLE,
                    eligibility_basis="observed_post_neoadjuvant_ct",
                ),
            )
        )
        bindings.extend(
            (
                CTAssetBinding(
                    asset_id=baseline_asset_id,
                    patient_id=patient_id,
                    role=ObservationRole.BASELINE_CT.value,
                    local_path=str(baseline_path),
                    directory_id=baseline_metadata.directory_id,
                    replica_count=(len(directories[row.baseline_link]) if baseline_replica else 1),
                    ct_file_count=baseline_metadata.ct_file_count,
                    header_warning_count=baseline_metadata.header_warning_count,
                    warning_capture_complete=baseline_metadata.warning_capture_complete,
                    study_id=baseline_metadata.studies[0].study_id,
                    acquisition_date_local=baseline_date.isoformat(),
                    selected_series_id=baseline_selection.selected_series_id,
                    selected_phase=baseline_selection.selected_phase,
                    selected_phase_basis=baseline_selection.selected_phase_basis,
                    pair_phase_status=pair_phase_status,
                    phase_selection_policy_version=selection_policy_version,
                    candidate_series=baseline_selection.candidate_series,
                ),
                CTAssetBinding(
                    asset_id=post_asset_id,
                    patient_id=patient_id,
                    role=ObservationRole.POST_TREATMENT_CT.value,
                    local_path=str(post_path),
                    directory_id=post_metadata.directory_id,
                    replica_count=(len(directories[row.post_link]) if post_replica else 1),
                    ct_file_count=post_metadata.ct_file_count,
                    header_warning_count=post_metadata.header_warning_count,
                    warning_capture_complete=post_metadata.warning_capture_complete,
                    study_id=post_metadata.studies[0].study_id,
                    acquisition_date_local=post_date.isoformat(),
                    selected_series_id=post_selection.selected_series_id,
                    selected_phase=post_selection.selected_phase,
                    selected_phase_basis=post_selection.selected_phase_basis,
                    pair_phase_status=pair_phase_status,
                    phase_selection_policy_version=selection_policy_version,
                    candidate_series=post_selection.candidate_series,
                ),
            )
        )
        linked_pairs += 1

    input_cohort = Cohort(
        patients=tuple(patients),
        observations=tuple(observations),
        clinical_measurements=(),
        treatments=(),
        outcomes=(),
        queries=tuple(queries),
    )
    if not patients:
        raise DataContractError(
            code="NO_ELIGIBLE_PAIRED_PATIENTS",
            message="No row satisfies the signed paired-CT and OS-v1 contracts.",
        )
    outcome_definition = OutcomeDefinition(
        endpoint_name="os",
        event_type=EventType.DEATH,
        event_code=mapping.event_code,
        origin_definition="clinical_excel_column_E",
        label_version=mapping.outcome_label_version,
        mode=mode,
        status_mapping_confirmed=True,
        origin_confirmed=True,
        timeline_confirmed=True,
    )
    firewall = FeatureFirewall(
        input_cohort.patients,
        input_cohort.observations,
        (),
        (),
        FeaturePolicy(mode=mode),
    )
    landmarks = LandmarkBuilder(firewall, OutcomeBuilder(outcomes, outcome_definition)).build(
        input_cohort.queries
    )
    if landmarks.exclusions:
        raise DataContractError(
            code="PAIRED_COHORT_INTERNAL_LANDMARK_EXCLUSION",
            message="A validated paired row unexpectedly failed landmark construction.",
            details={"exclusion_count": len(landmarks.exclusions)},
        )
    expected_landmarks = len(patients) * 2
    if len(landmarks.landmarks) != expected_landmarks:
        raise DataContractError(
            code="PAIRED_COHORT_LANDMARK_COUNT_MISMATCH",
            message="Every included paired patient must have one S0 and one S1 landmark.",
        )
    for landmark in landmarks.landmarks:
        roles = tuple(item.role for item in landmark.prefix.observations)
        expected_roles = (
            (ObservationRole.BASELINE_CT,)
            if landmark.stage is Stage.S0
            else (ObservationRole.BASELINE_CT, ObservationRole.POST_TREATMENT_CT)
        )
        if (
            roles != expected_roles
            or landmark.prefix.clinical_measurements
            or landmark.prefix.treatments
        ):
            raise DataContractError(
                code="PAIRED_COHORT_PREFIX_LEAKAGE",
                message="A paired S0/S1 prefix differs from the signed CT-only input policy.",
            )

    assignments = SplitManager(seed=split_seed).assign(patient.patient_id for patient in patients)
    SplitManager.assert_records_assigned(input_cohort.observations, assignments)
    SplitManager.assert_records_assigned(input_cohort.queries, assignments)
    stage_counts = {stage: 0 for stage in ("s0", "s1")}
    stage_events = {stage: 0 for stage in stage_counts}
    for landmark in landmarks.landmarks:
        stage_counts[landmark.stage.value] += 1
        stage_events[landmark.stage.value] += int(landmark.label.event)
    exclusions = tuple(
        CohortExclusion(
            source_row=row.source_row,
            patient_id=patient_ids.get(row.source_row),
            reasons=tuple(sorted(reasons[row.source_row])),
        )
        for row in rows
        if reasons[row.source_row]
    )
    return PairedCTBuildResult(
        input_cohort=input_cohort,
        outcomes=tuple(outcomes),
        assignments=assignments,
        bindings=tuple(bindings),
        exclusions=exclusions,
        source_record_count=len(rows),
        rows_with_both_links=both_links,
        directory_matched_pair_count=directory_matched_pairs,
        metadata_usable_pair_count=metadata_usable_pairs,
        linked_pair_count=linked_pairs,
        equivalent_replica_link_count=replica_count,
        stage_landmark_counts=stage_counts,
        stage_event_counts=stage_events,
        cohort_version=mapping.cohort_version,
        outcome_label_version=mapping.outcome_label_version,
        series_selection_policy_version=selection_policy_version,
    )


def write_paired_ct_artifacts(
    result: PairedCTBuildResult,
    *,
    output_root: str | Path,
    split_seed: int,
) -> dict[str, Any]:
    selection_policy_version = result.series_selection_policy_version
    selection_status = (
        "frozen_exploratory"
        if selection_policy_version == FIRST_ACQUISITION_SELECTION_POLICY_VERSION
        else "frozen"
    )
    root = Path(output_root) / "data"
    restricted = root / "restricted"
    cohort_artifact_id = new_artifact_id("paired-ct-cohort")
    data_lineage_id = new_artifact_id("paired-ct-data")
    split_version = f"paired-ct-split-v1-seed-{split_seed}"
    input_path = restricted / "input_cohort.json"
    outcomes_path = restricted / "outcomes.json"
    assets_path = restricted / "asset_bindings.json"
    exclusions_path = restricted / "exclusions.json"
    split_path = restricted / "split_assignments.json"
    atomic_write_private_json(
        input_path,
        {
            **cohort_to_dict(result.input_cohort),
            "private_schema_version": PAIRED_CT_COHORT_SCHEMA,
            "data_lineage_id": data_lineage_id,
            "cohort_artifact_id": cohort_artifact_id,
            "contains_real_clinical_data": True,
        },
    )
    atomic_write_private_json(
        outcomes_path,
        {
            "schema_version": PRIVATE_OUTCOME_SCHEMA,
            "data_lineage_id": data_lineage_id,
            "cohort_artifact_id": cohort_artifact_id,
            "outcome_label_version": result.outcome_label_version,
            "outcomes": [asdict(item) for item in result.outcomes],
        },
    )
    atomic_write_private_json(
        assets_path,
        {
            "schema_version": PRIVATE_ASSET_SCHEMA,
            "data_lineage_id": data_lineage_id,
            "cohort_artifact_id": cohort_artifact_id,
            "phase_selection_status": selection_status,
            "phase_selection_policy_version": selection_policy_version,
            "bindings": [asdict(item) for item in result.bindings],
        },
    )
    atomic_write_private_json(
        exclusions_path,
        {
            "schema_version": PRIVATE_EXCLUSION_SCHEMA,
            "data_lineage_id": data_lineage_id,
            "cohort_artifact_id": cohort_artifact_id,
            "exclusions": [asdict(item) for item in result.exclusions],
        },
    )
    split_counts = {name: 0 for name in ("train", "validation", "test")}
    for assignment in result.assignments:
        split_counts[assignment.split.value] += 1
    atomic_write_private_json(
        split_path,
        {
            "schema_version": PRIVATE_SPLIT_SCHEMA,
            "data_lineage_id": data_lineage_id,
            "cohort_artifact_id": cohort_artifact_id,
            "split_version": split_version,
            "seed": split_seed,
            "assignments": [
                {
                    "patient_id": item.patient_id,
                    "split": item.split.value,
                    "fold": item.fold,
                }
                for item in result.assignments
            ],
        },
    )
    reason_counts: Counter[str] = Counter()
    for exclusion in result.exclusions:
        reason_counts.update(exclusion.reasons)
    series_candidate_count = plausible_series_count = 0
    studies_with_recognized_phase = top_rank_unknown_phase_count = 0
    studies_with_confident_phase = studies_with_header_confident_phase = 0
    top_rank_unresolved_after_timing = top_rank_unresolved_after_all_evidence = 0
    known_dicom_warning_count = legacy_warning_count_unavailable = 0
    phase_candidate_counts: Counter[str] = Counter()
    timing_phase_candidate_counts: Counter[str] = Counter()
    timing_phase_basis_counts: Counter[str] = Counter()
    timing_phase_confidence_counts: Counter[str] = Counter()
    bolus_delay_bucket_counts: Counter[str] = Counter()
    explicit_phase_bolus_delay_bucket_counts: Counter[str] = Counter()
    confident_phase_series_counts: Counter[str] = Counter()
    header_confident_phase_series_counts: Counter[str] = Counter()
    sequence_phase_candidate_counts: Counter[str] = Counter()
    acquisition_time_source_counts: Counter[str] = Counter()
    temporal_cluster_count_counts: Counter[str] = Counter()
    plausible_temporal_cluster_count_counts: Counter[str] = Counter()
    explicit_phase_anchor_order_counts: Counter[str] = Counter()
    explicit_phase_cluster_position_counts: Counter[str] = Counter()
    explicit_phase_plausible_cluster_position_counts: Counter[str] = Counter()
    explicit_phase_position_by_sequence_signature_counts: Counter[str] = Counter()
    temporal_gap_signature_counts_by_evidence: dict[str, Counter[str]] = defaultdict(
        Counter
    )
    phase_timing_conflict_count = series_with_time_count = series_with_bolus_delay_count = 0
    studies_with_any_time_count = studies_with_complete_time_count = 0
    studies_with_bolus_delay_count = 0
    studies_with_sequence_phase_count = 0
    phase_sets_by_role: dict[str, Counter[str]] = defaultdict(Counter)
    header_phase_sets_by_role: dict[str, Counter[str]] = defaultdict(Counter)
    patient_phase_sets: dict[str, dict[str, set[str]]] = defaultdict(dict)
    patient_header_phase_sets: dict[str, dict[str, set[str]]] = defaultdict(dict)
    selected_phase_counts_by_role: dict[str, Counter[str]] = defaultdict(Counter)
    selected_phase_basis_counts: Counter[str] = Counter()
    pair_phase_status_counts: Counter[str] = Counter()
    for binding in result.bindings:
        selected_candidates = [
            series for series in binding.candidate_series if bool(series.get("selected"))
        ]
        if (
            len(selected_candidates) != 1
            or str(selected_candidates[0].get("series_id")) != binding.selected_series_id
        ):
            raise DataContractError(
                code="INVALID_CT_PHASE_SELECTION",
                message="Each retained CT study must bind exactly one selected series.",
            )
        selected_phase_counts_by_role[binding.role][binding.selected_phase] += 1
        selected_phase_basis_counts[binding.selected_phase_basis] += 1
        if binding.role == ObservationRole.BASELINE_CT.value:
            pair_phase_status_counts[binding.pair_phase_status] += 1
        if not binding.warning_capture_complete or binding.header_warning_count < 0:
            legacy_warning_count_unavailable += 1
        else:
            known_dicom_warning_count += binding.header_warning_count
        series_candidate_count += len(binding.candidate_series)
        plausible_series_count += sum(
            bool(series["plausible_volume"]) for series in binding.candidate_series
        )
        recognized = {
            str(phase)
            for series in binding.candidate_series
            if bool(series["plausible_volume"])
            for phase in series["phase_candidates"]
        }
        phase_candidate_counts.update(recognized)
        studies_with_recognized_phase += int(bool(recognized))
        confident_phases: set[str] = set()
        header_confident_phases: set[str] = set()
        study_has_sequence_phase = False
        plausible_series = [
            series
            for series in binding.candidate_series
            if bool(series["plausible_volume"])
        ]
        timed_plausible_series_count = 0
        study_has_bolus_delay = False
        for series in plausible_series:
            time_source = series.get("acquisition_time_source")
            if time_source is not None:
                series_with_time_count += 1
                timed_plausible_series_count += 1
                acquisition_time_source_counts[str(time_source)] += 1
            bolus_delay = series.get("contrast_bolus_delay_seconds")
            if bolus_delay is not None:
                series_with_bolus_delay_count += 1
                study_has_bolus_delay = True
                bolus_delay_bucket_counts[_bolus_delay_bucket(float(bolus_delay))] += 1
            timing_phase = series.get("timing_phase_candidate")
            if timing_phase is not None:
                timing_phase_candidate_counts[str(timing_phase)] += 1
                timing_phase_basis_counts[
                    str(series.get("timing_phase_basis", "unknown"))
                ] += 1
                timing_phase_confidence_counts[
                    str(series.get("timing_phase_confidence", "none"))
                ] += 1
            explicit = tuple(str(item) for item in series.get("phase_candidates", ()))
            if len(explicit) == 1 and bolus_delay is not None:
                bucket = _bolus_delay_bucket(float(bolus_delay))
                explicit_phase_bolus_delay_bucket_counts[f"{explicit[0]}:{bucket}"] += 1
            if (
                len(explicit) == 1
                and timing_phase is not None
                and str(timing_phase) != explicit[0]
            ):
                phase_timing_conflict_count += 1
            header_confident_phase = _header_confident_series_phase(series)
            if header_confident_phase is not None:
                header_confident_phases.add(header_confident_phase)
                header_confident_phase_series_counts[header_confident_phase] += 1
            sequence_phase = series.get("sequence_phase_candidate")
            if sequence_phase is not None:
                study_has_sequence_phase = True
                sequence_phase_candidate_counts[str(sequence_phase)] += 1
            confident_phase = _confident_series_phase(series)
            if confident_phase is not None:
                confident_phases.add(confident_phase)
                confident_phase_series_counts[confident_phase] += 1
        studies_with_any_time_count += int(timed_plausible_series_count > 0)
        studies_with_complete_time_count += int(
            bool(plausible_series)
            and timed_plausible_series_count == len(plausible_series)
        )
        studies_with_bolus_delay_count += int(study_has_bolus_delay)
        studies_with_sequence_phase_count += int(study_has_sequence_phase)

        plausible_cluster_offsets: dict[int, float] = {}
        for series in plausible_series:
            cluster = series.get("temporal_cluster_index")
            offset = series.get("acquisition_offset_seconds")
            if cluster is None or offset is None:
                continue
            cluster_index = int(cluster)
            offset_seconds = float(offset)
            previous = plausible_cluster_offsets.get(cluster_index)
            if previous is None or offset_seconds < previous:
                plausible_cluster_offsets[cluster_index] = offset_seconds
        plausible_cluster_indices = sorted(plausible_cluster_offsets)
        plausible_temporal_cluster_count_counts[str(len(plausible_cluster_indices))] += 1
        explicit_arterial = [
            series
            for series in plausible_series
            if "arterial" in series.get("phase_candidates", ())
        ]
        explicit_venous = [
            series
            for series in plausible_series
            if "venous_or_portal" in series.get("phase_candidates", ())
        ]
        if explicit_arterial and explicit_venous:
            arterial_offsets = [
                float(series["acquisition_offset_seconds"])
                for series in explicit_arterial
                if series.get("acquisition_offset_seconds") is not None
            ]
            venous_offsets = [
                float(series["acquisition_offset_seconds"])
                for series in explicit_venous
                if series.get("acquisition_offset_seconds") is not None
            ]
            if not arterial_offsets or not venous_offsets:
                explicit_phase_anchor_order_counts["timing_unavailable"] += 1
            else:
                gap_seconds = min(venous_offsets) - min(arterial_offsets)
                if abs(gap_seconds) <= 15.0:
                    bucket = "same_time_within_15_seconds"
                elif gap_seconds < -15.0:
                    bucket = "venous_before_arterial_over_15_seconds"
                elif gap_seconds <= 45.0:
                    bucket = "arterial_before_venous_16_to_45_seconds"
                elif gap_seconds <= 90.0:
                    bucket = "arterial_before_venous_46_to_90_seconds"
                else:
                    bucket = "arterial_before_venous_over_90_seconds"
                explicit_phase_anchor_order_counts[bucket] += 1
        for phase, phase_series in (
            ("arterial", explicit_arterial),
            ("venous_or_portal", explicit_venous),
        ):
            positions = {
                int(series["temporal_cluster_index"])
                for series in phase_series
                if series.get("temporal_cluster_index") is not None
            }
            cluster_counts = {
                int(series.get("temporal_cluster_count", 0)) for series in phase_series
            }
            if len(cluster_counts) == 1:
                cluster_total = next(iter(cluster_counts))
                for position in positions:
                    key = f"{phase}:{position + 1}_of_{cluster_total}"
                    explicit_phase_cluster_position_counts[key] += 1
            plausible_positions = {
                plausible_cluster_indices.index(int(series["temporal_cluster_index"])) + 1
                for series in phase_series
                if series.get("temporal_cluster_index") is not None
                and int(series["temporal_cluster_index"]) in plausible_cluster_offsets
            }
            for position in plausible_positions:
                key = f"{phase}:{position}_of_{len(plausible_cluster_indices)}"
                explicit_phase_plausible_cluster_position_counts[key] += 1

        ordered_offsets = [
            plausible_cluster_offsets[index] for index in plausible_cluster_indices
        ]
        gap_signature = "+".join(
            _temporal_gap_bucket(later - earlier)
            for earlier, later in zip(ordered_offsets, ordered_offsets[1:], strict=False)
        )
        sequence_signature = f"{len(ordered_offsets)}_clusters"
        if gap_signature:
            sequence_signature = f"{sequence_signature}:{gap_signature}"
        for phase, phase_series in (
            ("arterial", explicit_arterial),
            ("venous_or_portal", explicit_venous),
        ):
            plausible_positions = {
                plausible_cluster_indices.index(int(series["temporal_cluster_index"])) + 1
                for series in phase_series
                if series.get("temporal_cluster_index") is not None
                and int(series["temporal_cluster_index"]) in plausible_cluster_offsets
            }
            for position in plausible_positions:
                key = (
                    f"{sequence_signature}|{phase}:"
                    f"{position}_of_{len(plausible_cluster_indices)}"
                )
                explicit_phase_position_by_sequence_signature_counts[key] += 1
        if explicit_arterial and explicit_venous:
            evidence_group = "explicit_arterial_and_venous"
        elif confident_phases:
            evidence_group = "other_confident_phase"
        else:
            evidence_group = "no_confident_phase"
        temporal_gap_signature_counts_by_evidence[evidence_group][
            sequence_signature
        ] += 1
        studies_with_confident_phase += int(bool(confident_phases))
        studies_with_header_confident_phase += int(bool(header_confident_phases))
        phase_signature = "+".join(sorted(confident_phases)) or "unknown"
        phase_sets_by_role[binding.role][phase_signature] += 1
        header_phase_signature = "+".join(sorted(header_confident_phases)) or "unknown"
        header_phase_sets_by_role[binding.role][header_phase_signature] += 1
        patient_phase_sets[binding.patient_id][binding.role] = confident_phases
        patient_header_phase_sets[binding.patient_id][binding.role] = header_confident_phases
        cluster_count = max(
            (int(series.get("temporal_cluster_count", 0)) for series in plausible_series),
            default=0,
        )
        temporal_cluster_count_counts[str(cluster_count)] += 1
        top_rank_unknown_phase_count += int(
            bool(binding.candidate_series)
            and not bool(binding.candidate_series[0]["phase_candidates"])
        )
        top_rank_unresolved_after_timing += int(
            bool(binding.candidate_series)
            and _header_confident_series_phase(binding.candidate_series[0]) is None
        )
        top_rank_unresolved_after_all_evidence += int(
            bool(binding.candidate_series)
            and _confident_series_phase(binding.candidate_series[0]) is None
        )
    paired_phase_counts = _paired_phase_counts(patient_phase_sets)
    paired_header_phase_counts = _paired_phase_counts(patient_header_phase_sets)
    build_path = root / "cohort_build.json"
    aggregate = {
        "schema_version": PAIRED_CT_ARTIFACT_SCHEMA,
        "status": "ok",
        "mode": "real_images",
        "cohort_scope": "retrospective_complete_paired_ct_transition_cohort",
        "development_stages": ["s0", "s1"],
        "treatment_conditioning": "elapsed_time_only",
        "future_ct_completion_used_as_s0_feature": False,
        "clinical_validation": False,
        "data_lineage_id": data_lineage_id,
        "cohort_artifact_id": cohort_artifact_id,
        "cohort_version": result.cohort_version,
        "outcome_label_version": result.outcome_label_version,
        "source_record_count": result.source_record_count,
        "rows_with_both_links": result.rows_with_both_links,
        "directory_matched_pair_count": result.directory_matched_pair_count,
        "metadata_usable_pair_count": result.metadata_usable_pair_count,
        "included_patient_count": len(result.input_cohort.patients),
        "excluded_record_count": len(result.exclusions),
        "exclusions_by_reason": dict(sorted(reason_counts.items())),
        "equivalent_replica_link_count": result.equivalent_replica_link_count,
        "stage_landmark_counts": dict(result.stage_landmark_counts),
        "stage_event_counts": dict(result.stage_event_counts),
        "split_version": split_version,
        "split_counts": split_counts,
        "phase_selection_status": selection_status,
        "phase_selection_policy_version": selection_policy_version,
        "phase_selection_pair_counts": dict(sorted(pair_phase_status_counts.items())),
        "selected_phase_study_counts_by_role": {
            role: dict(sorted(counts.items()))
            for role, counts in sorted(selected_phase_counts_by_role.items())
        },
        "selected_phase_basis_counts": dict(sorted(selected_phase_basis_counts.items())),
        "validated_sequence_minimum_explicit_study_support": 20,
        "validated_sequence_signatures": sorted(_VALIDATED_SEQUENCE_PHASE_RULES),
        "series_candidate_count": series_candidate_count,
        "plausible_volume_series_count": plausible_series_count,
        "studies_with_recognized_phase_candidate": studies_with_recognized_phase,
        "top_rank_unknown_phase_count": top_rank_unknown_phase_count,
        "phase_candidate_study_counts": dict(sorted(phase_candidate_counts.items())),
        "phase_timing_audit_version": "paired-ct-phase-timing-v2",
        "plausible_series_with_acquisition_time_count": series_with_time_count,
        "studies_with_any_plausible_series_acquisition_time_count": (
            studies_with_any_time_count
        ),
        "studies_with_complete_plausible_series_acquisition_time_count": (
            studies_with_complete_time_count
        ),
        "plausible_series_acquisition_time_source_counts": dict(
            sorted(acquisition_time_source_counts.items())
        ),
        "plausible_series_with_bolus_delay_count": series_with_bolus_delay_count,
        "studies_with_plausible_series_bolus_delay_count": (
            studies_with_bolus_delay_count
        ),
        "plausible_series_bolus_delay_bucket_counts": dict(
            sorted(bolus_delay_bucket_counts.items())
        ),
        "explicit_phase_bolus_delay_bucket_counts": dict(
            sorted(explicit_phase_bolus_delay_bucket_counts.items())
        ),
        "timing_phase_candidate_series_counts": dict(
            sorted(timing_phase_candidate_counts.items())
        ),
        "timing_phase_basis_counts": dict(sorted(timing_phase_basis_counts.items())),
        "timing_phase_confidence_counts": dict(
            sorted(timing_phase_confidence_counts.items())
        ),
        "phase_timing_conflict_series_count": phase_timing_conflict_count,
        "header_confident_phase_series_counts": dict(
            sorted(header_confident_phase_series_counts.items())
        ),
        "sequence_phase_candidate_series_counts": dict(
            sorted(sequence_phase_candidate_counts.items())
        ),
        "studies_with_validated_sequence_phase_candidate": (
            studies_with_sequence_phase_count
        ),
        "confident_phase_series_counts": dict(sorted(confident_phase_series_counts.items())),
        "studies_with_header_confident_phase_candidate": (
            studies_with_header_confident_phase
        ),
        "studies_with_confident_phase_candidate": studies_with_confident_phase,
        "top_rank_unresolved_after_timing_count": top_rank_unresolved_after_timing,
        "top_rank_unresolved_after_all_phase_evidence_count": (
            top_rank_unresolved_after_all_evidence
        ),
        "study_header_confident_phase_set_counts_by_role": {
            role: dict(sorted(counts.items()))
            for role, counts in sorted(header_phase_sets_by_role.items())
        },
        "study_confident_phase_set_counts_by_role": {
            role: dict(sorted(counts.items()))
            for role, counts in sorted(phase_sets_by_role.items())
        },
        "paired_header_confident_phase_counts": dict(
            sorted(paired_header_phase_counts.items())
        ),
        "paired_confident_phase_counts": dict(sorted(paired_phase_counts.items())),
        "temporal_cluster_count_study_counts": dict(
            sorted(temporal_cluster_count_counts.items(), key=lambda item: int(item[0]))
        ),
        "plausible_temporal_cluster_count_study_counts": dict(
            sorted(
                plausible_temporal_cluster_count_counts.items(),
                key=lambda item: int(item[0]),
            )
        ),
        "explicit_phase_anchor_order_counts": dict(
            sorted(explicit_phase_anchor_order_counts.items())
        ),
        "explicit_phase_cluster_position_counts": dict(
            sorted(explicit_phase_cluster_position_counts.items())
        ),
        "explicit_phase_plausible_cluster_position_counts": dict(
            sorted(explicit_phase_plausible_cluster_position_counts.items())
        ),
        "explicit_phase_position_by_sequence_signature_counts": dict(
            sorted(explicit_phase_position_by_sequence_signature_counts.items())
        ),
        "temporal_gap_signature_counts_by_evidence": {
            evidence: dict(sorted(counts.items()))
            for evidence, counts in sorted(
                temporal_gap_signature_counts_by_evidence.items()
            )
        },
        "known_dicom_header_warning_count": known_dicom_warning_count,
        "legacy_warning_count_unavailable_study_count": legacy_warning_count_unavailable,
        "encoder_status": "not_approved",
        "contains_identifiers_or_paths": False,
        "restricted_artifacts_owner_only": True,
    }
    atomic_write_json(build_path, aggregate)
    return {**aggregate, "artifact": str(build_path)}
