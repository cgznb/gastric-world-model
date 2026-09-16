"""Offline stomach-organ localization with an independently versioned CT field of view."""

from __future__ import annotations

import contextlib
import importlib.metadata
import io
import os
import socket
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import torch

from stageworld.artifacts import atomic_write_private_json, read_json
from stageworld.encoders.base import CTGeometry
from stageworld.errors import ArtifactError, DataContractError

from .ct_preprocessing import (
    PreprocessedCT,
    _read_dicom_volume,
    resolve_selected_series_files,
)
from .paired_ct import HMACPseudonymizer

GASTRIC_ROI_VERSION = "totalseg-2.18.0-total297-resident-3mm-stomach-margin20-cube96-v1"
GASTRIC_MODEL_DIRECTORY = "Dataset297_TotalSegmentator_total_3mm_1559subj"
GASTRIC_TRAINER_DIRECTORY = "nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres"


@contextlib.contextmanager
def offline_network() -> Iterator[None]:
    """Fail closed on Python IPv4/IPv6 connects; permit local multiprocessing IPC."""

    original_connect = socket.socket.connect

    def connect(sock: socket.socket, address: Any) -> Any:
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise ArtifactError(code="NETWORK_FORBIDDEN", message="Clinical inference is offline.")
        return original_connect(sock, address)

    with patch.object(socket.socket, "connect", connect):
        yield


def stomach_field_of_view(
    mask: np.ndarray, affine: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep the entire organ plus 20 mm in a RAS isotropic cube, never a center truncation."""

    from scipy import ndimage  # type: ignore[import-untyped]

    if mask.ndim != 3 or affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise DataContractError(code="ROI_GEOMETRY_INVALID", message="Invalid stomach geometry.")
    binary = mask.astype(bool)
    count = int(binary.sum())
    volume_ml = count * abs(float(np.linalg.det(affine[:3, :3]))) / 1000
    if count == 0:
        raise DataContractError(code="ROI_EMPTY", message="No stomach was localized.")
    if not 5 <= volume_ml <= 4000:
        raise DataContractError(code="ROI_VOLUME_EXTREME", message="Stomach volume failed QC.")
    labels, components = ndimage.label(binary)
    sizes = np.bincount(labels.ravel())[1:]
    largest_fraction = float(sizes.max() / count)
    if largest_fraction < 0.8:
        raise DataContractError(
            code="ROI_FRAGMENTED", message="Stomach localization is fragmented."
        )
    positions = np.argwhere(binary)
    lower, upper = positions.min(axis=0), positions.max(axis=0)
    border = bool(np.any(lower == 0) or np.any(upper == np.asarray(mask.shape) - 1))
    if border:
        raise DataContractError(
            code="ROI_TRUNCATED", message="Stomach touches the CT volume boundary."
        )
    corners = (
        np.array(
            [
                [x, y, z, 1]
                for x in (lower[0] - 0.5, upper[0] + 0.5)
                for y in (lower[1] - 0.5, upper[1] + 0.5)
                for z in (lower[2] - 0.5, upper[2] + 0.5)
            ]
        )
        @ affine.T
    )
    low_mm, high_mm = corners[:, :3].min(0), corners[:, :3].max(0)
    extent = high_mm - low_mm
    if float(extent.max()) > 400:
        raise DataContractError(code="ROI_EXTENT_EXTREME", message="Stomach extent failed QC.")
    side_mm = max(192.0, float(extent.max()) + 40.0)
    output_affine = np.eye(4)
    output_affine[:3, :3] *= side_mm / 96
    output_affine[:3, 3] = (low_mm + high_mm) / 2 - side_mm / 2 + side_mm / 192
    return output_affine, {
        "stomach_volume_ml": volume_ml,
        "component_count": int(components),
        "largest_component_fraction": largest_fraction,
        "stomach_extent_mm": extent.tolist(),
        "field_of_view_mm": side_mm,
        "margin_mm": 20.0,
        "organ_not_tumor_segmentation": True,
        "expert_review_completed": False,
    }


class StomachSegmenter:
    """Use the pinned public 3-mm total model locally, with statistics disabled."""

    def __init__(self, home: Path, cache_root: Path, *, device: str = "cuda") -> None:
        for package, required in (("TotalSegmentator", "2.18.0"), ("nnunetv2", "2.8.1")):
            if importlib.metadata.version(package) != required:
                raise ArtifactError(
                    code="SEGMENTER_VERSION_MISMATCH", message="Unpinned segmenter."
                )
        model = home / "nnunet/results" / GASTRIC_MODEL_DIRECTORY / GASTRIC_TRAINER_DIRECTORY
        for relative in ("plans.json", "dataset.json", "fold_0/checkpoint_final.pth"):
            if not (model / relative).is_file():
                raise ArtifactError(
                    code="SEGMENTER_WEIGHTS_MISSING",
                    message="Download the approved public stomach model before offline inference.",
                )
        for directory in (home, cache_root, cache_root / "temporary"):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            directory.chmod(0o700)
        atomic_write_private_json(
            home / "config.json",
            {
                "totalseg_id": "stageworld-offline",
                "send_usage_stats": False,
                "prediction_counter": 0,
                "statistics_disclaimer_shown": True,
            },
        )
        os.environ["TOTALSEG_HOME_DIR"] = str(home)
        os.environ["nnUNet_compile"] = "false"
        os.environ["TMPDIR"] = str(cache_root / "temporary")
        tempfile.tempdir = str(cache_root / "temporary")
        self.cache_root = cache_root
        self.device = "gpu" if device == "cuda" else "cpu"
        self.model_directory = model
        self.predictor: Any = None

    def _predict_persistent(
        self,
        dir_in: Any,
        dir_out: Any,
        task_id: int,
        model: str = "3d_fullres",
        folds: Any = None,
        trainer: str = "nnUNetTrainer",
        tta: bool = False,
        num_threads_preprocessing: int = 3,
        num_threads_nifti_save: int = 2,
        **kwargs: Any,
    ) -> None:
        """Keep the upstream predictor resident; use its own I/O and array inference API."""
        import nibabel as nib
        from nnunetv2.inference.predict_from_raw_data import (  # type: ignore[import-untyped]
            nnUNetPredictor,
        )

        if task_id != 297 or tta or folds != [0] or model != "3d_fullres":
            raise ArtifactError(
                code="SEGMENTER_TASK_MISMATCH", message="Unexpected segmentation task."
            )
        torch.set_num_threads(1)
        if self.predictor is None:
            self.predictor = nnUNetPredictor(
                tile_step_size=kwargs.get("step_size", 0.5),
                use_gaussian=True,
                use_mirroring=False,
                perform_everything_on_device=True,
                device=torch.device("cuda" if self.device == "gpu" else "cpu"),
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=False,
            )
            self.predictor.initialize_from_trained_model_folder(
                str(self.model_directory),
                use_folds=[0],
                checkpoint_name="checkpoint_final.pth",
            )
        for source_path in sorted(Path(dir_in).glob("*_0000.nii.gz")):
            values, properties = (
                self.predictor.plans_manager.image_reader_writer_class().read_images(
                    [str(source_path)],
                )
            )
            segmentation = self.predictor.predict_single_npy_array(values, properties)
            original = cast(nib.Nifti1Image, nib.load(source_path))
            nib.save(
                nib.Nifti1Image(segmentation.transpose(2, 1, 0).astype(np.uint8), original.affine),
                Path(dir_out) / source_path.name.replace("_0000.nii.gz", ".nii.gz"),
            )

    def preprocess(
        self,
        binding: Mapping[str, Any],
        *,
        approved_root: str | Path,
        pseudonymizer: HMACPseudonymizer,
    ) -> PreprocessedCT:
        import nibabel as nib
        from nibabel.processing import resample_from_to
        from totalsegmentator.map_to_binary import class_map  # type: ignore[import-untyped]
        from totalsegmentator.python_api import totalsegmentator  # type: ignore[import-untyped]

        asset = str(binding["asset_id"])
        if not asset.replace("-", "").replace("_", "").isalnum():
            raise DataContractError(
                code="ROI_ASSET_INVALID", message="Invalid anonymous asset key."
            )
        files = resolve_selected_series_files(
            binding,
            approved_root=approved_root,
            pseudonymizer=pseudonymizer,
        )
        array, affine, source = _read_dicom_volume(files)
        image = nib.Nifti1Image(array, affine)
        mask_path = self.cache_root / f"{asset}.nii.gz"
        record_path = self.cache_root / f"{asset}.json"
        if mask_path.is_file() and record_path.is_file():
            record = read_json(record_path)
            if (
                record.get("preprocess_version") != GASTRIC_ROI_VERSION
                or record.get("selected_series_id") != binding["selected_series_id"]
            ):
                raise ArtifactError(code="ROI_CACHE_MISMATCH", message="Stale ROI mask cache.")
            stomach_image = cast(nib.Nifti1Image, nib.load(mask_path))
        else:
            # Upstream logs are not allowed to expose raw clinical paths or headers.
            captured = io.StringIO()
            from totalsegmentator.config import setup_nnunet  # type: ignore[import-untyped]

            setup_nnunet()
            import totalsegmentator.nnunet as total_runtime  # type: ignore[import-untyped]

            with (
                offline_network(),
                contextlib.redirect_stdout(captured),
                patch.object(total_runtime, "nnUNetv2_predict", self._predict_persistent),
            ):
                result = totalsegmentator(
                    image,
                    output=None,
                    task="total",
                    fast=True,
                    save_lowres=True,
                    device=self.device,
                    quiet=True,
                    nr_thr_resamp=2,
                    nr_thr_saving=1,
                    statistics=False,
                    radiomics=False,
                    preview=False,
                    skip_saving=True,
                )
            stomach_id = next(k for k, name in class_map["total"].items() if name == "stomach")
            binary = np.asarray(result.dataobj) == stomach_id
            stomach_image = nib.Nifti1Image(binary.astype(np.uint8), result.affine)
            nib.save(stomach_image, mask_path)
            mask_path.chmod(0o600)
            atomic_write_private_json(
                record_path,
                {
                    "preprocess_version": GASTRIC_ROI_VERSION,
                    "selected_series_id": binding["selected_series_id"],
                    "source_shape": list(array.shape),
                    "outcome_data_read": False,
                },
            )
        target_affine, qc = stomach_field_of_view(
            np.asarray(stomach_image.dataobj),
            stomach_image.affine,
        )
        resized = resample_from_to(image, ((96, 96, 96), target_affine), order=1, cval=-1000)
        values = np.clip((np.asarray(resized.dataobj, dtype=np.float32) + 1000) / 2000, 0, 1)
        atomic_write_private_json(
            self.cache_root / f"{asset}.qc.json",
            {
                "preprocess_version": GASTRIC_ROI_VERSION,
                "status": "passed_automated_qc",
                **qc,
            },
        )
        spacing = np.linalg.norm(target_affine[:3, :3], axis=0)
        geometry = CTGeometry(
            spacing_mm=torch.tensor(spacing[None], dtype=torch.float32),
            origin_mm=torch.tensor(target_affine[:3, 3][None], dtype=torch.float32),
            direction=torch.eye(3)[None],
            spatial_shape=torch.tensor([[96, 96, 96]]),
        )
        return PreprocessedCT(
            image=torch.from_numpy(values[None].copy()),
            geometry=geometry,
            selected_file_count=len(files),
            source_shape_xyz=tuple(array.shape),
            source_spacing_xyz_mm=tuple(source.GetSpacing()),
            quality_flags=(
                "first_acquisition_phase_unknown",
                "automatic_stomach_organ_roi",
                "20mm_margin_isotropic_adaptive_fov",
                "expert_review_pending",
            ),
        )
