"""Deterministic, local-only preprocessing for selected DICOM CT volumes."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from stageworld.encoders.base import CTGeometry
from stageworld.encoders.swinunetr import SWINUNETR_PREPROCESS_VERSION
from stageworld.errors import ArtifactError, ConfigurationError, DataContractError

from .paired_ct import HMACPseudonymizer

SWINUNETR_INPUT_SHAPE = (96, 96, 96)
SWINUNETR_SPACING_MM = (1.5, 1.5, 2.0)
SWINUNETR_HU_RANGE = (-1000.0, 1000.0)

_DICOM_TAGS = (
    "Modality",
    "SeriesInstanceUID",
    "Rows",
    "Columns",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "PhotometricInterpretation",
)


@dataclass(frozen=True)
class PreprocessedCT:
    image: Tensor
    geometry: CTGeometry
    selected_file_count: int
    source_shape_xyz: tuple[int, int, int]
    source_spacing_xyz_mm: tuple[float, float, float]
    quality_flags: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.image.shape != (1, *SWINUNETR_INPUT_SHAPE):
            raise DataContractError(
                code="INVALID_PREPROCESSED_CT_SHAPE",
                message="Preprocessed CT must have shape [1,96,96,96].",
            )
        if self.image.dtype != torch.float32 or not torch.isfinite(self.image).all():
            raise DataContractError(
                code="INVALID_PREPROCESSED_CT_VALUES",
                message="Preprocessed CT must contain finite float32 values.",
            )
        if float(self.image.min()) < 0.0 or float(self.image.max()) > 1.0:
            raise DataContractError(
                code="INVALID_PREPROCESSED_CT_RANGE",
                message="Preprocessed CT values must lie in [0,1].",
            )


def _finite_vector(value: object, length: int) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if vector.shape != (length,) or not np.isfinite(vector).all():
        return None
    return vector


def resolve_selected_series_files(
    binding: Mapping[str, Any],
    *,
    approved_root: str | Path,
    pseudonymizer: HMACPseudonymizer,
) -> tuple[Path, ...]:
    """Resolve one pseudonymous selected-series ID without returning its raw UID."""

    try:
        import pydicom  # type: ignore[import-untyped]
    except ImportError as error:
        raise ArtifactError(
            code="ENCODER_DEPENDENCY_MISSING",
            message="DICOM CT preprocessing requires pydicom.",
        ) from error
    raw_path = binding.get("local_path")
    selected_series_id = binding.get("selected_series_id")
    if not isinstance(raw_path, str) or not raw_path or not isinstance(selected_series_id, str):
        raise DataContractError(
            code="INVALID_CT_ASSET_MANIFEST",
            message="A CT asset binding lacks its restricted path or selected series ID.",
        )
    root = Path(approved_root).expanduser().resolve()
    directory = Path(raw_path).expanduser().resolve()
    if not directory.is_relative_to(root):
        raise ConfigurationError(
            code="PATH_OUTSIDE_APPROVED_ROOT",
            message="A selected CT asset is outside the approved data root.",
        )
    try:
        files = tuple(sorted(path for path in directory.iterdir() if path.is_file()))
    except OSError as error:
        raise DataContractError(
            code="CT_DIRECTORY_UNREADABLE",
            message="A selected CT directory cannot be enumerated.",
        ) from error

    records: list[tuple[float, Path]] = []
    reference_normal: np.ndarray | None = None
    reference_matrix: tuple[int, int] | None = None
    photometric_values: set[str] = set()
    for path in files:
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                dataset = pydicom.dcmread(
                    path,
                    stop_before_pixels=True,
                    specific_tags=list(_DICOM_TAGS),
                )
        except Exception:
            continue
        if str(getattr(dataset, "Modality", "")).strip().upper() != "CT":
            continue
        raw_series_id = str(getattr(dataset, "SeriesInstanceUID", "")).strip()
        if not raw_series_id:
            continue
        series_id = pseudonymizer.token("dicom-series", raw_series_id, prefix="SERIES")
        if series_id != selected_series_id:
            continue
        orientation = _finite_vector(
            getattr(dataset, "ImageOrientationPatient", None), 6
        )
        position = _finite_vector(getattr(dataset, "ImagePositionPatient", None), 3)
        if orientation is None or position is None:
            raise DataContractError(
                code="CT_GEOMETRY_INCOMPLETE",
                message="A selected CT slice lacks finite patient orientation or position.",
            )
        normal = np.cross(orientation[:3], orientation[3:])
        norm = float(np.linalg.norm(normal))
        if not math.isfinite(norm) or norm <= 0:
            raise DataContractError(
                code="CT_GEOMETRY_INVALID",
                message="A selected CT slice has an invalid orientation.",
            )
        normal /= norm
        if reference_normal is None:
            reference_normal = normal
        elif abs(float(normal @ reference_normal)) < 0.999:
            raise DataContractError(
                code="CT_ORIENTATION_INCONSISTENT",
                message="A selected CT series contains inconsistent slice orientations.",
            )
        try:
            matrix = (int(dataset.Rows), int(dataset.Columns))
        except (AttributeError, TypeError, ValueError) as error:
            raise DataContractError(
                code="CT_MATRIX_INVALID",
                message="A selected CT slice has no valid pixel matrix size.",
            ) from error
        if reference_matrix is None:
            reference_matrix = matrix
        elif matrix != reference_matrix:
            raise DataContractError(
                code="CT_MATRIX_INCONSISTENT",
                message="A selected CT series contains inconsistent pixel matrix sizes.",
            )
        photometric_values.add(
            str(getattr(dataset, "PhotometricInterpretation", "")).strip().upper()
        )
        records.append((float(position @ reference_normal), path))
    if len(records) < 16:
        raise DataContractError(
            code="SELECTED_CT_SERIES_NOT_FOUND",
            message="The selected CT series has fewer than 16 positioned slices.",
        )
    if photometric_values != {"MONOCHROME2"}:
        raise DataContractError(
            code="CT_PHOTOMETRIC_UNSUPPORTED",
            message="Only consistently MONOCHROME2 CT series are supported.",
        )
    records.sort(key=lambda item: (item[0], str(item[1])))
    rounded_positions = [round(item[0], 3) for item in records]
    if len(set(rounded_positions)) != len(rounded_positions):
        raise DataContractError(
            code="CT_DUPLICATE_SLICE_POSITION",
            message="A selected CT series contains duplicate slice positions.",
        )
    return tuple(path for _, path in records)


def _read_dicom_volume(files: Sequence[Path]) -> tuple[np.ndarray, np.ndarray, Any]:
    try:
        import SimpleITK as sitk  # type: ignore[import-untyped]
    except ImportError as error:
        raise ArtifactError(
            code="ENCODER_DEPENDENCY_MISSING",
            message="DICOM CT preprocessing requires SimpleITK.",
        ) from error
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(path) for path in files])
    try:
        image = reader.Execute()
        array_zyx = sitk.GetArrayFromImage(image)
    except RuntimeError as error:
        raise DataContractError(
            code="CT_PIXEL_DECODE_FAILED",
            message="The selected DICOM CT series could not be decoded as one volume.",
        ) from error
    if array_zyx.ndim != 3 or array_zyx.shape[0] != len(files):
        raise DataContractError(
            code="CT_VOLUME_SHAPE_MISMATCH",
            message="Decoded CT volume shape does not match its selected slices.",
        )
    array_xyz = np.transpose(array_zyx, (2, 1, 0)).astype(np.float32, copy=False)
    if not np.isfinite(array_xyz).all():
        raise DataContractError(
            code="CT_NONFINITE_PIXELS",
            message="Decoded CT contains nonfinite voxel values.",
        )
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    if (
        spacing.shape != (3,)
        or np.any(spacing <= 0)
        or not np.isfinite(direction).all()
        or not np.isfinite(origin).all()
    ):
        raise DataContractError(
            code="CT_GEOMETRY_INVALID",
            message="Decoded CT has invalid physical geometry.",
        )
    affine_lps = np.eye(4, dtype=np.float64)
    affine_lps[:3, :3] = direction @ np.diag(spacing)
    affine_lps[:3, 3] = origin
    lps_to_ras = np.diag((-1.0, -1.0, 1.0, 1.0))
    return array_xyz, lps_to_ras @ affine_lps, image


def preprocess_selected_ct(
    binding: Mapping[str, Any],
    *,
    approved_root: str | Path,
    pseudonymizer: HMACPseudonymizer,
) -> PreprocessedCT:
    """Read, orient, resample, window, foreground-crop, and center-crop one CT."""

    try:
        from monai.data import MetaTensor
        from monai.transforms import (
            Compose,
            CropForeground,
            Orientation,
            ResizeWithPadOrCrop,
            ScaleIntensityRange,
            Spacing,
        )
    except ImportError as error:
        raise ArtifactError(
            code="ENCODER_DEPENDENCY_MISSING",
            message="CT preprocessing requires MONAI.",
        ) from error

    files = resolve_selected_series_files(
        binding,
        approved_root=approved_root,
        pseudonymizer=pseudonymizer,
    )
    array_xyz, affine, image = _read_dicom_volume(files)
    source_shape = (
        int(array_xyz.shape[0]),
        int(array_xyz.shape[1]),
        int(array_xyz.shape[2]),
    )
    raw_spacing = image.GetSpacing()
    source_spacing = (
        float(raw_spacing[0]),
        float(raw_spacing[1]),
        float(raw_spacing[2]),
    )
    volume = MetaTensor(
        torch.from_numpy(array_xyz[None].copy()),
        affine=torch.from_numpy(affine),
    )
    transform = Compose(
        (
            Orientation(
                axcodes="RAS",
                labels=(("L", "R"), ("P", "A"), ("I", "S")),
            ),
            Spacing(
                pixdim=SWINUNETR_SPACING_MM,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            ),
            ScaleIntensityRange(
                a_min=SWINUNETR_HU_RANGE[0],
                a_max=SWINUNETR_HU_RANGE[1],
                b_min=0.0,
                b_max=1.0,
                clip=True,
                dtype=np.float32,
            ),
            CropForeground(
                select_fn=lambda value: value > 0,
                allow_smaller=False,
                mode="constant",
            ),
            ResizeWithPadOrCrop(
                spatial_size=SWINUNETR_INPUT_SHAPE,
                method="symmetric",
                mode="constant",
                value=0.0,
            ),
        )
    )
    try:
        transformed = transform(volume)
    except (RuntimeError, ValueError) as error:
        raise DataContractError(
            code="CT_PREPROCESSING_FAILED",
            message="The selected CT failed deterministic Swin UNETR preprocessing.",
        ) from error
    if not isinstance(transformed, MetaTensor):
        raise DataContractError(
            code="CT_PREPROCESSING_METADATA_LOST",
            message="CT preprocessing did not preserve physical affine metadata.",
        )
    tensor = transformed.as_tensor().contiguous().to(dtype=torch.float32)
    affine_out = transformed.affine.detach().to(dtype=torch.float64, device="cpu")
    linear = affine_out[:3, :3]
    spacing = torch.linalg.vector_norm(linear, dim=0)
    if torch.any(spacing <= 0) or not torch.isfinite(spacing).all():
        raise DataContractError(
            code="CT_PREPROCESSING_GEOMETRY_INVALID",
            message="Preprocessed CT affine has invalid voxel spacing.",
        )
    direction = linear / spacing[None, :]
    geometry = CTGeometry(
        spacing_mm=spacing.to(dtype=torch.float32)[None],
        origin_mm=affine_out[:3, 3].to(dtype=torch.float32)[None],
        direction=direction.to(dtype=torch.float32)[None],
        spatial_shape=torch.tensor([SWINUNETR_INPUT_SHAPE], dtype=torch.long),
    )
    return PreprocessedCT(
        image=tensor,
        geometry=geometry,
        selected_file_count=len(files),
        source_shape_xyz=source_shape,
        source_spacing_xyz_mm=source_spacing,
        quality_flags=(
            "first_acquisition_phase_unknown",
            "deterministic_center_crop_without_gastric_roi",
            "manual_anatomic_and_artifact_review_pending",
        ),
    )


__all__ = [
    "PreprocessedCT",
    "SWINUNETR_HU_RANGE",
    "SWINUNETR_INPUT_SHAPE",
    "SWINUNETR_PREPROCESS_VERSION",
    "SWINUNETR_SPACING_MM",
    "preprocess_selected_ct",
    "resolve_selected_series_files",
]
