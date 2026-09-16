"""Outcome-blind automated QC for selected paired-CT volumes.

The routine reads DICOM pixels locally but persists neither pixels, raw UIDs,
absolute paths, free text, nor acquisition dates. Anatomical coverage and image
artifacts remain a manual-review responsibility.
"""

from __future__ import annotations

import hashlib
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from stageworld.artifacts import atomic_write_json, atomic_write_private_json, read_json
from stageworld.errors import ConfigurationError, DataContractError, StageWorldError

from .paired_ct import HMACPseudonymizer

SELECTED_CT_QC_SCHEMA = "paired-ct-selected-volume-qc-v1"
PRIVATE_SELECTED_CT_QC_SCHEMA = "private-paired-ct-selected-volume-qc-v1"

_HEADER_TAGS = (
    "Modality",
    "SeriesInstanceUID",
    "Rows",
    "Columns",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "PixelSpacing",
    "BurnedInAnnotation",
    "PhotometricInterpretation",
)


def _require_pydicom() -> Any:
    try:
        import pydicom  # type: ignore[import-untyped]
    except ImportError as exc:
        raise StageWorldError(
            code="dependency_missing",
            message="Selected-volume CT QC requires pydicom.",
            remediation="Install the ct optional dependency in the approved environment.",
            details={"dependency": "pydicom"},
        ) from exc
    return pydicom


def _finite_vector(value: object, length: int) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if vector.shape != (length,) or not np.isfinite(vector).all():
        return None
    return vector


def _valid_spacing(value: object) -> tuple[float, float] | None:
    vector = _finite_vector(value, 2)
    if vector is None or np.any(vector <= 0):
        return None
    return float(vector[0]), float(vector[1])


def _selected_sample(
    bindings: Sequence[Mapping[str, Any]],
    *,
    sample_per_role: int,
    sample_seed: int,
) -> tuple[Mapping[str, Any], ...]:
    by_role: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for binding in bindings:
        role = str(binding.get("role", ""))
        asset_id = str(binding.get("asset_id", ""))
        if not role or not asset_id:
            raise DataContractError(
                code="INVALID_CT_ASSET_MANIFEST",
                message="Every CT asset binding must identify an anonymous asset and role.",
            )
        by_role[role].append(binding)
    expected_roles = {"baseline_ct", "post_treatment_ct"}
    if set(by_role) != expected_roles:
        raise DataContractError(
            code="INVALID_CT_ASSET_MANIFEST",
            message="Paired CT QC requires baseline and post-treatment bindings.",
        )
    selected: list[Mapping[str, Any]] = []
    for role in sorted(expected_roles):
        ranked = sorted(
            by_role[role],
            key=lambda binding: hashlib.sha256(
                f"{sample_seed}:{binding['asset_id']}".encode()
            ).digest(),
        )
        selected.extend(ranked[: min(sample_per_role, len(ranked))])
    return tuple(selected)


def _slice_order(
    records: Sequence[tuple[Path, Any]], reference_normal: np.ndarray | None
) -> list[tuple[float, Path, Any]]:
    ordered: list[tuple[float, Path, Any]] = []
    for index, (path, dataset) in enumerate(records):
        position = _finite_vector(getattr(dataset, "ImagePositionPatient", None), 3)
        projection = (
            float(position @ reference_normal)
            if position is not None and reference_normal is not None
            else float(index)
        )
        ordered.append((projection, path, dataset))
    return sorted(ordered, key=lambda item: (item[0], str(item[1])))


def _sample_indices(length: int, count: int) -> tuple[int, ...]:
    if length <= 0:
        return ()
    if count == 1:
        return (length // 2,)
    raw = np.linspace(0, length - 1, num=min(count, length))
    return tuple(sorted({int(round(value)) for value in raw}))


def _pixel_check(pydicom: Any, paths: Sequence[Path]) -> dict[str, Any]:
    decoded = finite = dynamic = 0
    body_fractions: list[float] = []
    failure_types: Counter[str] = Counter()
    for path in paths:
        try:
            dataset = pydicom.dcmread(path)
            pixels = np.asarray(dataset.pixel_array, dtype=np.float32)
        except Exception as exc:  # Decoder errors vary by transfer syntax and plugin.
            failure_types[type(exc).__name__] += 1
            continue
        decoded += 1
        if pixels.ndim != 2 or not np.isfinite(pixels).all():
            continue
        finite += 1
        slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
        hu = pixels * slope + intercept
        low, high = np.percentile(hu, (1.0, 99.0))
        if math.isfinite(float(low)) and math.isfinite(float(high)) and high - low > 10.0:
            dynamic += 1
        body_fractions.append(float(np.mean((hu > -800.0) & (hu < 2000.0))))
    return {
        "requested_slice_count": len(paths),
        "decoded_slice_count": decoded,
        "finite_slice_count": finite,
        "nonconstant_slice_count": dynamic,
        "maximum_body_fraction": max(body_fractions, default=None),
        "decoder_failure_types": dict(sorted(failure_types.items())),
    }


def _inspect_binding(
    binding: Mapping[str, Any],
    *,
    approved_root: Path,
    pseudonymizer: HMACPseudonymizer,
    pixel_slices: int,
) -> dict[str, Any]:
    pydicom = _require_pydicom()
    raw_path = binding.get("local_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise DataContractError(
            code="INVALID_CT_ASSET_MANIFEST",
            message="A selected CT binding is missing its restricted local path.",
        )
    directory = Path(raw_path).expanduser().resolve()
    if not directory.is_relative_to(approved_root):
        raise ConfigurationError(
            code="PATH_OUTSIDE_APPROVED_ROOT",
            message="A selected CT asset is outside the approved data root.",
        )
    selected_series_id = str(binding.get("selected_series_id", ""))
    records: list[tuple[Path, Any]] = []
    header_errors = 0
    try:
        files = tuple(sorted(path for path in directory.iterdir() if path.is_file()))
    except OSError as exc:
        raise DataContractError(
            code="ct_directory_unreadable",
            message="A selected CT directory cannot be enumerated for QC.",
        ) from exc
    for path in files:
        try:
            dataset = pydicom.dcmread(
                path,
                stop_before_pixels=True,
                specific_tags=list(_HEADER_TAGS),
            )
        except Exception:
            header_errors += 1
            continue
        if str(getattr(dataset, "Modality", "")).strip().upper() != "CT":
            continue
        raw_series_id = str(getattr(dataset, "SeriesInstanceUID", "")).strip()
        if not raw_series_id:
            continue
        series_id = pseudonymizer.token("dicom-series", raw_series_id, prefix="SERIES")
        if series_id == selected_series_id:
            records.append((path, dataset))
    if not records:
        return {
            "asset_id": str(binding["asset_id"]),
            "role": str(binding["role"]),
            "selected_series_id": selected_series_id,
            "automatic_qc_status": "fail",
            "failure_reason": "selected_series_not_found",
            "manual_review_required": True,
        }

    orientations: list[np.ndarray] = []
    normal: np.ndarray | None = None
    positions = 0
    matrices: set[tuple[int, int]] = set()
    pixel_spacings: list[tuple[float, float]] = []
    burned_in_values: set[str] = set()
    photometric_values: set[str] = set()
    for _, dataset in records:
        try:
            matrices.add((int(dataset.Rows), int(dataset.Columns)))
        except (AttributeError, TypeError, ValueError):
            pass
        orientation = _finite_vector(
            getattr(dataset, "ImageOrientationPatient", None), 6
        )
        if orientation is not None:
            candidate_normal = np.cross(orientation[:3], orientation[3:])
            norm = float(np.linalg.norm(candidate_normal))
            if norm > 0:
                candidate_normal /= norm
                orientations.append(candidate_normal)
                if normal is None:
                    normal = candidate_normal
        if _finite_vector(getattr(dataset, "ImagePositionPatient", None), 3) is not None:
            positions += 1
        spacing = _valid_spacing(getattr(dataset, "PixelSpacing", None))
        if spacing is not None:
            pixel_spacings.append(spacing)
        burned = str(getattr(dataset, "BurnedInAnnotation", "")).strip().upper()
        burned_in_values.add(burned or "missing")
        photometric = str(getattr(dataset, "PhotometricInterpretation", "")).strip().upper()
        photometric_values.add(photometric or "missing")

    ordered = _slice_order(records, normal)
    projected_positions = [item[0] for item in ordered] if normal is not None else []
    unique_positions = sorted({round(position, 3) for position in projected_positions})
    z_span = (
        float(unique_positions[-1] - unique_positions[0])
        if len(unique_positions) >= 2
        else None
    )
    gaps = [
        later - earlier
        for earlier, later in zip(unique_positions, unique_positions[1:], strict=False)
        if later - earlier > 0.01
    ]
    median_gap = float(statistics.median(gaps)) if gaps else None
    maximum_gap = max(gaps, default=None)
    orientation_consistent = bool(orientations) and len(orientations) == len(records)
    if orientation_consistent and normal is not None:
        orientation_consistent = all(
            abs(float(candidate @ normal)) >= 0.999 for candidate in orientations
        )
    axial = bool(normal is not None and abs(float(normal[2])) >= 0.9)
    spacing_complete = len(pixel_spacings) == len(records)
    spacing_consistent = spacing_complete and len(
        {(round(row, 5), round(column, 5)) for row, column in pixel_spacings}
    ) == 1
    selected_paths = [
        ordered[index][1] for index in _sample_indices(len(ordered), pixel_slices)
    ]
    pixel = _pixel_check(pydicom, selected_paths)
    pixel_ok = (
        pixel["requested_slice_count"] > 0
        and pixel["decoded_slice_count"] == pixel["requested_slice_count"]
        and pixel["finite_slice_count"] == pixel["requested_slice_count"]
        and pixel["nonconstant_slice_count"] == pixel["requested_slice_count"]
    )
    geometry_ok = (
        len(records) >= 16
        and len(unique_positions) >= 16
        and matrices == {(512, 512)}
        and orientation_consistent
        and axial
        and spacing_consistent
        and z_span is not None
        and z_span >= 150.0
    )
    flags: list[str] = []
    if z_span is not None and z_span > 1200.0:
        flags.append("unusually_large_z_span")
    if "YES" in burned_in_values:
        flags.append("burned_in_annotation_declared")
    if maximum_gap is not None and median_gap is not None and maximum_gap > 3 * median_gap:
        flags.append("large_interslice_gap")
    return {
        "asset_id": str(binding["asset_id"]),
        "role": str(binding["role"]),
        "selected_series_id": selected_series_id,
        "automatic_qc_status": "pass" if geometry_ok and pixel_ok and not flags else "flag",
        "selected_file_count": len(records),
        "header_error_count": header_errors,
        "matrix_shapes": [list(shape) for shape in sorted(matrices)],
        "positioned_slice_count": positions,
        "unique_position_count": len(unique_positions),
        "axial_orientation": axial,
        "orientation_consistent": orientation_consistent,
        "pixel_spacing_complete_and_consistent": spacing_consistent,
        "pixel_spacing_mm": (
            list(pixel_spacings[0]) if spacing_consistent and pixel_spacings else None
        ),
        "z_span_mm": z_span,
        "median_interslice_spacing_mm": median_gap,
        "maximum_interslice_gap_mm": maximum_gap,
        "burned_in_annotation_values": sorted(burned_in_values),
        "photometric_values": sorted(photometric_values),
        "pixel_check": pixel,
        "flags": flags,
        "manual_review_required": True,
    }


def run_selected_ct_qc(
    *,
    asset_manifest_path: str | Path,
    approved_root: str | Path,
    output_root: str | Path,
    pseudonymizer: HMACPseudonymizer,
    sample_per_role: int = 20,
    pixel_slices: int = 3,
    sample_seed: int = 17,
) -> dict[str, Any]:
    """Run deterministic, outcome-blind selected-volume QC and write redacted artifacts."""

    if sample_per_role < 1 or pixel_slices < 1 or sample_seed < 0:
        raise ConfigurationError(
            code="INVALID_CT_QC_SETTINGS",
            message="CT QC sample sizes must be positive and its seed nonnegative.",
        )
    manifest = read_json(asset_manifest_path)
    bindings = manifest.get("bindings")
    if not isinstance(bindings, list) or not bindings:
        raise DataContractError(
            code="INVALID_CT_ASSET_MANIFEST",
            message="The selected CT asset manifest contains no bindings.",
        )
    policy = str(manifest.get("phase_selection_policy_version", ""))
    sampled = _selected_sample(
        bindings,
        sample_per_role=sample_per_role,
        sample_seed=sample_seed,
    )
    approved = Path(approved_root).expanduser().resolve()
    records = tuple(
        _inspect_binding(
            binding,
            approved_root=approved,
            pseudonymizer=pseudonymizer,
            pixel_slices=pixel_slices,
        )
        for binding in sampled
    )
    status_counts = Counter(str(record["automatic_qc_status"]) for record in records)
    role_counts = Counter(str(record["role"]) for record in records)
    flag_counts: Counter[str] = Counter()
    decoder_failure_counts: Counter[str] = Counter()
    z_spans: list[float] = []
    pixel_spacings: list[float] = []
    axial_count = orientation_consistent_count = 0
    pixel_decode_complete_count = 0
    burned_in_declared_count = 0
    for record in records:
        flag_counts.update(str(flag) for flag in record.get("flags", ()))
        axial_count += int(bool(record.get("axial_orientation")))
        orientation_consistent_count += int(bool(record.get("orientation_consistent")))
        z_span = record.get("z_span_mm")
        if isinstance(z_span, (int, float)) and math.isfinite(float(z_span)):
            z_spans.append(float(z_span))
        spacing = record.get("pixel_spacing_mm")
        if isinstance(spacing, list):
            pixel_spacings.extend(float(value) for value in spacing)
        burned_in_declared_count += int(
            "YES" in record.get("burned_in_annotation_values", ())
        )
        pixel = record.get("pixel_check")
        if not isinstance(pixel, Mapping):
            continue
        requested = int(pixel.get("requested_slice_count", 0))
        decoded = int(pixel.get("decoded_slice_count", 0))
        pixel_decode_complete_count += int(requested > 0 and decoded == requested)
        failures = pixel.get("decoder_failure_types", {})
        if isinstance(failures, Mapping):
            decoder_failure_counts.update(
                {str(name): int(count) for name, count in failures.items()}
            )

    root = Path(output_root) / "data"
    restricted_path = root / "restricted" / "selected_volume_qc.json"
    summary_path = root / "selected_volume_qc_summary.json"
    atomic_write_private_json(
        restricted_path,
        {
            "schema_version": PRIVATE_SELECTED_CT_QC_SCHEMA,
            "data_lineage_id": manifest.get("data_lineage_id"),
            "cohort_artifact_id": manifest.get("cohort_artifact_id"),
            "series_selection_policy_version": policy,
            "sample_seed": sample_seed,
            "outcome_data_read": False,
            "contains_raw_paths_uids_dates_or_pixels": False,
            "records": list(records),
        },
    )
    summary: dict[str, Any] = {
        "schema_version": SELECTED_CT_QC_SCHEMA,
        "status": "manual_review_required",
        "series_selection_policy_version": policy,
        "sample_seed": sample_seed,
        "sample_per_role_requested": sample_per_role,
        "pixel_slices_per_study_requested": pixel_slices,
        "sampled_study_count": len(records),
        "sampled_study_counts_by_role": dict(sorted(role_counts.items())),
        "automatic_qc_status_counts": dict(sorted(status_counts.items())),
        "axial_orientation_count": axial_count,
        "orientation_consistent_count": orientation_consistent_count,
        "pixel_decode_complete_study_count": pixel_decode_complete_count,
        "burned_in_annotation_declared_study_count": burned_in_declared_count,
        "qc_flag_counts": dict(sorted(flag_counts.items())),
        "decoder_failure_type_counts": dict(sorted(decoder_failure_counts.items())),
        "z_span_mm": {
            "minimum": min(z_spans, default=None),
            "median": statistics.median(z_spans) if z_spans else None,
            "maximum": max(z_spans, default=None),
            "below_150_mm_count": sum(value < 150.0 for value in z_spans),
        },
        "pixel_spacing_mm": {
            "minimum": min(pixel_spacings, default=None),
            "median": statistics.median(pixel_spacings) if pixel_spacings else None,
            "maximum": max(pixel_spacings, default=None),
        },
        "outcome_blind": True,
        "outcome_data_read": False,
        "pixels_or_images_persisted": False,
        "contains_identifiers_paths_uids_dates_or_free_text": False,
        "anatomic_gastric_coverage_verified": False,
        "motion_and_artifact_review_completed": False,
        "manual_review_required_count": len(records),
        "clinical_training_ready": False,
        "remaining_qc_requirements": [
            "clinician_review_of_gastric_and_abdominal_coverage",
            "clinician_review_of_motion_and_other_artifacts",
            "preprocessing_and_encoder_parity_approval",
        ],
    }
    atomic_write_json(summary_path, summary)
    return {
        **summary,
        "artifact": str(summary_path),
        "restricted_artifact": str(restricted_path),
    }


__all__ = [
    "PRIVATE_SELECTED_CT_QC_SCHEMA",
    "SELECTED_CT_QC_SCHEMA",
    "run_selected_ct_qc",
]
