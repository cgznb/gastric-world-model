"""Offline FLARE23 tumor candidates; anatomic association is not expert annotation."""

from __future__ import annotations

import contextlib
import filecmp
import importlib.metadata
import importlib.util
import io
import shutil
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch

from stageworld.artifacts import atomic_write_private_json, new_artifact_id, read_json
from stageworld.encoders.base import CTGeometry
from stageworld.errors import ArtifactError, DataContractError

from .ct_preprocessing import PreprocessedCT, _read_dicom_volume, resolve_selected_series_files
from .gastric_roi import offline_network
from .paired_ct import HMACPseudonymizer

TUMOR_ROI_VERSION = "flare23-stunet-gastric-candidate-near10-margin20-min192-cube96-v1"
COARSE_TUMOR_ROI_VERSION = "flare23-tumor-near40-margin30-stomachfallback40-min192-cube96-v1"
TUMOR_ROI_VERSIONS = (TUMOR_ROI_VERSION, COARSE_TUMOR_ROI_VERSION)
TUMOR_MODEL_FILES = (
    "dataset.json",
    "plans.json",
    "STUNetTrainer.py",
    "LICENSE",
    "fold_all/checkpoint_final.pth",
)
TUMOR_LABELS = {
    "background": 0,
    "Liver": 1,
    "Right Kidney": 2,
    "Spleen": 3,
    "Pancreas": 4,
    "Aorta": 5,
    "Inferior vena cava": 6,
    "Right adrenal gland": 7,
    "Left adrenal gland": 8,
    "Gallbladder": 9,
    "Esophagus": 10,
    "Stomach": 11,
    "Duodenum": 12,
    "Left Kidney": 13,
    "Tumor": 14,
}


def validate_tumor_model(model_directory: Path) -> dict[str, Any]:
    """Require the published pan-cancer task, including its separate tumor class."""
    if any(not (model_directory / name).is_file() for name in TUMOR_MODEL_FILES):
        raise ArtifactError(
            code="TUMOR_SEGMENTER_WEIGHTS_MISSING",
            message="Provide the published FLARE23 STU-Net base model and source snapshot.",
        )
    dataset = read_json(model_directory / "dataset.json")
    if (
        dataset.get("labels") != TUMOR_LABELS
        or dataset.get("channel_names") != {"0": "CT"}
        or dataset.get("file_ending") != ".nii.gz"
    ):
        raise ArtifactError(
            code="TUMOR_SEGMENTER_TASK_MISMATCH",
            message="Require FLARE23 single-channel CT, stomach label 11 and tumor label 14.",
        )
    plans = read_json(model_directory / "plans.json")
    if "3d_fullres" not in plans.get("configurations", {}):
        raise ArtifactError(
            code="TUMOR_SEGMENTER_PLAN_MISMATCH", message="Require nnU-Net 3d_fullres plans."
        )
    if (model_directory / TUMOR_MODEL_FILES[-1]).stat().st_size == 0:
        raise ArtifactError(
            code="TUMOR_SEGMENTER_WEIGHTS_MISSING", message="The tumor checkpoint is empty."
        )
    return {
        "task": "pan_cancer_gastric_candidates",
        "labels": TUMOR_LABELS,
        "configuration": "3d_fullres",
        "expert_review_completed": False,
    }


def bind_tumor_model(
    model_directory: Path, cache_root: Path, *, create: bool,
    preprocess_version: str = TUMOR_ROI_VERSION,
) -> str:
    """Bind caches to an exact private model copy without generating digests."""
    validate_tumor_model(model_directory)
    if preprocess_version not in TUMOR_ROI_VERSIONS:
        raise ArtifactError(code="TUMOR_PROTOCOL_INVALID", message="Unknown tumor ROI protocol.")
    filecmp.clear_cache()
    snapshot = cache_root / "model_snapshot"
    record_path = snapshot / "binding.json"
    if not record_path.is_file():
        if not create:
            raise ArtifactError(
                code="TUMOR_MODEL_BINDING_MISSING", message="Extract the tumor feature cache first."
            )
        for directory in (cache_root, snapshot, snapshot / "fold_all"):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            directory.chmod(0o700)
        for name in TUMOR_MODEL_FILES:
            source, target = model_directory / name, snapshot / name
            if target.exists() and not filecmp.cmp(source, target, shallow=False):
                raise ArtifactError(
                    code="TUMOR_MODEL_BINDING_MISMATCH",
                    message="An incomplete cache was created with different tumor weights.",
                )
            if not target.exists():
                shutil.copyfile(source, target)
                target.chmod(0o600)
        atomic_write_private_json(
            record_path,
            {
                "model_artifact_id": new_artifact_id("flare23-tumor-segmenter"),
                "preprocess_version": preprocess_version,
                "task": "pan_cancer_gastric_candidates",
            },
        )
    record = read_json(record_path)
    if (
        record.get("preprocess_version") != preprocess_version
        or record.get("task") != "pan_cancer_gastric_candidates"
        or not isinstance(record.get("model_artifact_id"), str)
        or any(
            not (snapshot / name).is_file()
            or not filecmp.cmp(model_directory / name, snapshot / name, shallow=False)
            for name in TUMOR_MODEL_FILES
        )
    ):
        raise ArtifactError(
            code="TUMOR_MODEL_BINDING_MISMATCH",
            message="Tumor weights/plans changed; use a separately versioned feature cache.",
        )
    return str(record["model_artifact_id"])


def reuse_tumor_mask_predictions(
    model_directory: Path, source_cache: Path, target_cache: Path,
) -> dict[str, int]:
    """Reuse full-label predictions only after exact model and source-record checks."""
    if source_cache.resolve() == target_cache.resolve():
        raise ArtifactError(code="TUMOR_CACHE_COLLISION", message="Use a separate coarse cache.")
    source_id = bind_tumor_model(model_directory, source_cache, create=False)
    target_id = bind_tumor_model(
        model_directory, target_cache, create=True, preprocess_version=COARSE_TUMOR_ROI_VERSION
    )
    imported = already_present = 0
    for record_path in sorted(source_cache.glob("*.json")):
        if record_path.name.endswith(".qc.json"):
            continue
        asset = record_path.stem
        if not asset.replace("-", "").replace("_", "").isalnum():
            raise ArtifactError(code="ROI_ASSET_INVALID", message="Invalid cached asset key.")
        record = read_json(record_path)
        if (
            record.get("model_artifact_id") != source_id
            or record.get("preprocess_version") != TUMOR_ROI_VERSION
            or record.get("outcome_data_read") is not False
        ):
            raise ArtifactError(code="ROI_CACHE_MISMATCH", message="Invalid source mask record.")
        source_mask = source_cache / f"{asset}.nii.gz"
        if not source_mask.is_file():
            continue
        target_record = target_cache / record_path.name
        target_mask = target_cache / source_mask.name
        expected = {
            **record, "model_artifact_id": target_id,
            "preprocess_version": COARSE_TUMOR_ROI_VERSION,
        }
        if target_mask.is_file() and target_record.is_file():
            if read_json(target_record) != expected:
                raise ArtifactError(code="ROI_CACHE_MISMATCH", message="Coarse mask cache differs.")
            already_present += 1
            continue
        temporary = target_mask.with_suffix(".importing")
        shutil.copyfile(source_mask, temporary)
        temporary.chmod(0o600)
        temporary.replace(target_mask)
        atomic_write_private_json(target_record, expected)
        imported += 1
    return {"imported_full_label_masks": imported, "already_present": already_present}


def gastric_tumor_candidates(
    labels: np.ndarray, affine: np.ndarray, *, distance_mm: float = 10.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Retain whole nearby tumor components; the historical rule defaults to 10 mm."""
    from scipy import ndimage  # type: ignore[import-untyped]

    if labels.ndim != 3 or not np.isin(labels, range(15)).all():
        raise DataContractError(code="TUMOR_LABELS_INVALID", message="Invalid FLARE23 label map.")
    if not np.isfinite(distance_mm) or distance_mm <= 0:
        raise DataContractError(code="TUMOR_DISTANCE_INVALID", message="Require positive distance.")
    if np.shape(affine) != (4, 4) or not np.isfinite(affine).all():
        raise DataContractError(code="TUMOR_MASK_GEOMETRY_INVALID", message="Invalid CT geometry.")
    axes = np.asarray(affine)[:3, :3]
    spacing = np.linalg.norm(axes, axis=0)
    if (
        (spacing <= 0).any()
        or not np.allclose(affine[3], (0, 0, 0, 1))
        or not np.allclose(axes.T @ axes, np.diag(spacing**2), atol=1e-4)
    ):
        raise DataContractError(code="TUMOR_MASK_GEOMETRY_INVALID", message="Invalid CT geometry.")
    tumor = labels == 14
    stomach = labels == 11
    components, count = ndimage.label(tumor, structure=np.ones((3, 3, 3), dtype=bool))
    candidate = np.zeros(labels.shape, dtype=np.uint8)
    selected: list[int] = []
    distances: list[float] = []
    # Work in each component's expanded box, so distance maps do not fill scan-sized RAM.
    padding = np.ceil(distance_mm / spacing).astype(int) + 1
    for index, bounds in enumerate(ndimage.find_objects(components), start=1):
        if bounds is None:
            continue
        expanded = tuple(
            slice(max(0, s.start - p), min(n, s.stop + p))
            for s, p, n in zip(bounds, padding, labels.shape, strict=True)
        )
        local_stomach = stomach[expanded]
        if not local_stomach.any():
            continue
        local_component = components[expanded] == index
        distance = ndimage.distance_transform_edt(~local_stomach, sampling=spacing)
        nearest = float(distance[local_component].min())
        if nearest <= distance_mm:
            candidate[expanded][local_component] = 1
            selected.append(index)
            distances.append(nearest)
    return candidate, {
        "pan_cancer_component_count": int(count),
        "gastric_candidate_component_count": len(selected),
        "stomach_detected": bool(stomach.any()),
        "gastric_candidate_min_distance_mm": min(distances) if distances else None,
        "association_rule": (
            f"whole_26connected_tumor_components_within_{distance_mm:g}mm_of_label11"
        ),
        "association_is_heuristic": True,
        "primary_tumor_identity_confirmed": False,
        "expert_review_completed": False,
    }


def tumor_field_of_view(mask: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep all tumor components; retain the prior crop scale and intensity protocol."""
    if (
        mask.ndim != 3
        or min(mask.shape) < 2
        or affine.shape != (4, 4)
        or not np.isfinite(affine).all()
        or not np.allclose(affine[3], (0, 0, 0, 1))
        or abs(float(np.linalg.det(affine[:3, :3]))) < 1e-8
        or not np.isfinite(mask).all()
        or not np.isin(mask, (0, 1)).all()
    ):
        raise DataContractError(
            code="TUMOR_MASK_GEOMETRY_INVALID",
            message="Require a binary tumor mask and valid affine.",
        )
    positions = np.argwhere(mask > 0)
    if len(positions) == 0:
        raise DataContractError(
            code="TUMOR_NOT_DETECTED_REVIEW_REQUIRED",
            message="An empty prediction is unresolved; do not equate it with complete response.",
        )
    lower, upper = positions.min(0), positions.max(0)
    if np.any(lower == 0) or np.any(upper == np.asarray(mask.shape) - 1):
        raise DataContractError(
            code="TUMOR_BOUNDARY_REVIEW_REQUIRED",
            message="Predicted tumor touches the acquisition boundary; review tumor coverage.",
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
    low, high = corners[:, :3].min(0), corners[:, :3].max(0)
    extent = high - low
    if extent.max() > 400:
        raise DataContractError(
            code="TUMOR_EXTENT_REVIEW_REQUIRED", message="Predicted tumor extent requires review."
        )
    side = max(192.0, float(extent.max()) + 40.0)
    target_affine = np.eye(4)
    target_affine[:3, :3] *= side / 96
    target_affine[:3, 3] = (low + high) / 2 - side / 2 + side / 192
    return target_affine, {
        "tumor_volume_ml": len(positions) * abs(float(np.linalg.det(affine[:3, :3]))) / 1000,
        "tumor_extent_mm": extent.tolist(),
        "field_of_view_mm": side,
        "margin_mm": 20.0,
        "segmentation_target": "gastric_associated_pan_cancer_candidate",
        "expert_review_completed": False,
    }


def coarse_tumor_field_of_view(
    labels: np.ndarray, affine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Tumor-guided contextual crop, with a separately recorded stomach fallback."""
    candidate, association = gastric_tumor_candidates(labels, affine, distance_mm=40.0)
    uses_candidate = bool(candidate.any())
    mask = candidate > 0 if uses_candidate else labels == 11
    positions = np.argwhere(mask)
    if len(positions) == 0:
        raise DataContractError(
            code="COARSE_ANATOMIC_ROI_UNAVAILABLE",
            message="Neither a stomach-associated candidate nor stomach localization is available.",
        )
    lower, upper = positions.min(0), positions.max(0)
    corners = np.array([
        [x, y, z, 1]
        for x in (lower[0] - 0.5, upper[0] + 0.5)
        for y in (lower[1] - 0.5, upper[1] + 0.5)
        for z in (lower[2] - 0.5, upper[2] + 0.5)
    ]) @ affine.T
    low, high = corners[:, :3].min(0), corners[:, :3].max(0)
    margin = 30.0 if uses_candidate else 40.0
    side = max(192.0, float((high - low).max()) + 2 * margin)
    target = np.eye(4)
    target[:3, :3] *= side / 96
    target[:3, 3] = (low + high) / 2 - side / 2 + side / 192
    return target, candidate, {
        **association,
        "roi_source": "tumor_candidate" if uses_candidate else "stomach_fallback",
        "fallback_used": not uses_candidate,
        "fallback_reason": None if uses_candidate else "no_tumor_component_within_40mm",
        "margin_mm": margin,
        "field_of_view_mm": side,
        "localization_extent_mm": (high - low).tolist(),
        "localization_touches_scan_boundary": bool(
            np.any(lower == 0) or np.any(upper == np.asarray(mask.shape) - 1)
        ),
        "segmentation_target": "tumor_guided_context_with_explicit_stomach_fallback",
        "mask_used_only_for_crop": True,
        "outside_mask_ct_preserved": True,
        "complete_response_inferred": False,
    }


class TumorSegmenter:
    """Published STU-Net weights with nnU-Net's physical-space reader and writer."""

    def __init__(
        self, home: Path, cache_root: Path, *, device: str = "cuda",
        preprocess_version: str = TUMOR_ROI_VERSION,
    ) -> None:
        if importlib.metadata.version("nnunetv2") != "2.8.1":
            raise ArtifactError(
                code="SEGMENTER_VERSION_MISMATCH", message="Require the existing nnunetv2 2.8.1."
            )
        self.preprocess_version = preprocess_version
        self.model_artifact_id = bind_tumor_model(
            home, cache_root, create=True, preprocess_version=preprocess_version
        )
        self.model_directory = cache_root / "model_snapshot"
        self.cache_root = cache_root
        self.device = torch.device(device)
        self.predictor: Any = None

    def initialize(self) -> None:
        if self.predictor is not None:
            return
        from nnunetv2.inference.predict_from_raw_data import (  # type: ignore[import-untyped]
            nnUNetPredictor,
        )
        from nnunetv2.utilities.plans_handling.plans_handler import (  # type: ignore[import-untyped]
            PlansManager,
        )

        with torch.serialization.safe_globals(
            [
                (
                    importlib.import_module("numpy._core.multiarray").scalar,
                    "numpy.core.multiarray.scalar",
                ),
                np.dtype,
                np.dtypes.Float64DType,
                np.dtypes.Float32DType,
            ]
        ):
            checkpoint = torch.load(
                self.model_directory / TUMOR_MODEL_FILES[-1], map_location="cpu", weights_only=True
            )
        if (
            checkpoint.get("trainer_name") != "STUNetTrainer_base_ep2k"
            or checkpoint.get("init_args", {}).get("configuration") != "3d_fullres"
        ):
            raise ArtifactError(
                code="TUMOR_SEGMENTER_PLAN_MISMATCH",
                message="Require the published STU-Net base full-resolution model.",
            )
        dataset = read_json(self.model_directory / "dataset.json")
        plans = read_json(self.model_directory / "plans.json")
        manager = PlansManager(plans)
        configuration = manager.get_configuration("3d_fullres")
        if self.preprocess_version == COARSE_TUMOR_ROI_VERSION:
            from .tumor_export import ParallelProbabilityConfiguration

            configuration = ParallelProbabilityConfiguration(configuration)
        spec = importlib.util.spec_from_file_location(
            "stageworld_flare23_upstream", self.model_directory / "STUNetTrainer.py"
        )
        if spec is None or spec.loader is None:
            raise ArtifactError(code="TUMOR_SOURCE_UNREADABLE", message="Missing STU-Net source.")
        upstream = importlib.util.module_from_spec(spec)
        previous_bytecode = sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = True
            spec.loader.exec_module(upstream)
        finally:
            sys.dont_write_bytecode = previous_bytecode
        network = upstream.STUNetTrainer_base.build_network_architecture(
            manager,
            dataset,
            SimpleNamespace(
                pool_op_kernel_sizes=plans["configurations"]["3d_fullres"]["pool_op_kernel_sizes"]
            ),
            1,
            enable_deep_supervision=False,
        )
        network.load_state_dict(checkpoint["network_weights"], strict=True)
        network.eval()
        self.predictor = nnUNetPredictor(
            tile_step_size=0.8,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=False,
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        self.predictor.manual_initialization(
            network=network,
            plans_manager=manager,
            configuration_manager=configuration,
            parameters=[checkpoint["network_weights"]],
            dataset_json=dataset,
            trainer_name="STUNetTrainer_base_ep2k",
            inference_allowed_mirroring_axes=None,
        )

    def _predict(self, image: Any, destination: Path) -> None:
        import nibabel as nib

        self.initialize()
        with tempfile.TemporaryDirectory(prefix="tumor-inference-", dir=self.cache_root) as tmp:
            source = Path(tmp) / "ct_0000.nii.gz"
            nib.save(image, source)
            source.chmod(0o600)
            reader = self.predictor.plans_manager.image_reader_writer_class()
            values, properties = reader.read_images([str(source)])
            result = self.predictor.predict_single_npy_array(values, properties)
            reader.write_seg(result, str(destination), properties)
            destination.chmod(0o600)

    def preprocess(
        self,
        binding: Mapping[str, Any],
        *,
        approved_root: str | Path,
        pseudonymizer: HMACPseudonymizer,
    ) -> PreprocessedCT:
        import nibabel as nib
        from nibabel.processing import resample_from_to

        asset = str(binding["asset_id"])
        if not asset.replace("-", "").replace("_", "").isalnum():
            raise DataContractError(
                code="ROI_ASSET_INVALID", message="Invalid anonymous asset key."
            )
        files = resolve_selected_series_files(
            binding, approved_root=approved_root, pseudonymizer=pseudonymizer
        )
        array, affine, source = _read_dicom_volume(files)
        image = nib.Nifti1Image(array, affine)
        mask_path = self.cache_root / f"{asset}.nii.gz"
        record_path = self.cache_root / f"{asset}.json"
        expected = {
            "preprocess_version": self.preprocess_version,
            "model_artifact_id": self.model_artifact_id,
            "selected_series_id": binding["selected_series_id"],
            "source_shape": list(array.shape),
            "source_affine": affine.tolist(),
            "outcome_data_read": False,
        }
        if mask_path.is_file() and record_path.is_file():
            if read_json(record_path) != expected:
                raise ArtifactError(code="ROI_CACHE_MISMATCH", message="Stale tumor mask cache.")
        else:
            captured = io.StringIO()
            with (
                offline_network(),
                contextlib.redirect_stdout(captured),
                contextlib.redirect_stderr(captured),
            ):
                self._predict(image, mask_path)
            atomic_write_private_json(record_path, expected)
        tumor_image = cast(nib.Nifti1Image, nib.load(mask_path))
        if tumor_image.shape != image.shape or not np.allclose(
            tumor_image.affine, image.affine, rtol=0, atol=1e-4
        ):
            raise DataContractError(
                code="TUMOR_MASK_IMAGE_MISMATCH", message="Tumor prediction must match CT geometry."
            )
        labels = np.asarray(tumor_image.dataobj)
        coarse = self.preprocess_version == COARSE_TUMOR_ROI_VERSION
        association: dict[str, Any]
        if coarse:
            target_affine, candidate, qc = coarse_tumor_field_of_view(labels, tumor_image.affine)
            association = {}
        else:
            candidate, association = gastric_tumor_candidates(labels, tumor_image.affine)
        candidate_path = self.cache_root / f"{asset}.candidate.nii.gz"
        nib.save(nib.Nifti1Image(candidate, tumor_image.affine), candidate_path)
        candidate_path.chmod(0o600)
        try:
            if not coarse:
                target_affine, qc = tumor_field_of_view(candidate, tumor_image.affine)
        except DataContractError as error:
            atomic_write_private_json(
                self.cache_root / f"{asset}.qc.json",
                {
                    "preprocess_version": self.preprocess_version,
                    "status": "review_required",
                    "error_code": error.code,
                    **association,
                },
            )
            raise
        resized = resample_from_to(image, ((96, 96, 96), target_affine), order=1, cval=-1000)
        if coarse:
            crop_path = self.cache_root / f"{asset}.crop.nii.gz"
            temporary_crop = self.cache_root / f"{asset}.crop.tmp.nii.gz"
            nib.save(resized, temporary_crop)
            temporary_crop.chmod(0o600)
            temporary_crop.replace(crop_path)
            qc["resampled_hu_crop_saved"] = True
        values = np.clip((np.asarray(resized.dataobj, dtype=np.float32) + 1000) / 2000, 0, 1)
        atomic_write_private_json(
            self.cache_root / f"{asset}.qc.json",
            {
                "preprocess_version": self.preprocess_version,
                "status": "coarse_crop" if coarse else "unreviewed_candidate",
                **qc,
                **association,
            },
        )
        return PreprocessedCT(
            image=torch.from_numpy(values[None].copy()),
            geometry=CTGeometry(
                spacing_mm=torch.tensor(
                    np.linalg.norm(target_affine[:3, :3], axis=0)[None], dtype=torch.float32
                ),
                origin_mm=torch.tensor(target_affine[:3, 3][None], dtype=torch.float32),
                direction=torch.eye(3)[None],
                spatial_shape=torch.tensor([[96, 96, 96]]),
            ),
            selected_file_count=len(files),
            source_shape_xyz=tuple(array.shape),
            source_spacing_xyz_mm=tuple(source.GetSpacing()),
            quality_flags=(
                "first_acquisition_phase_unknown",
                f"coarse_roi_source_{qc['roi_source']}" if coarse
                else "gastric_associated_pan_cancer_candidate_roi",
                "adaptive_fov_intact_ct_context" if coarse
                else "20mm_margin_isotropic_adaptive_fov",
                "expert_review_pending",
            ),
        )
