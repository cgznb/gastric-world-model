"""Bounded real paired-CT feature extraction and engineering-only pretraining."""

from __future__ import annotations

import random
import stat
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import Tensor

from stageworld.artifacts import (
    atomic_write_json,
    atomic_write_private_json,
    new_artifact_id,
    read_json,
)
from stageworld.cache import CacheProvenance, FeatureCache
from stageworld.config import RunMode, StageWorldConfig
from stageworld.data.ct_preprocessing import preprocess_selected_ct
from stageworld.data.gastric_roi import GASTRIC_ROI_VERSION, StomachSegmenter
from stageworld.data.paired_ct import (
    FIRST_ACQUISITION_SELECTION_POLICY_VERSION,
    PAIRED_CT_COHORT_SCHEMA,
    PRIVATE_ASSET_SCHEMA,
    PRIVATE_SPLIT_SCHEMA,
    HMACPseudonymizer,
)
from stageworld.data.tumor_roi import (
    COARSE_TUMOR_ROI_VERSION,
    TUMOR_ROI_VERSIONS,
    TumorSegmenter,
    bind_tumor_model,
)
from stageworld.encoders import (
    SWINUNETR_COMPONENT_VERSION,
    SWINUNETR_FEATURE_DIM,
    SWINUNETR_PREPROCESS_VERSION,
    SWINUNETR_SOURCE_VERSION,
    EncoderAccess,
    EncoderProvenance,
    ObservationTokens,
    SwinUNETREncoder,
    load_swinunetr_ssl_backbone,
)
from stageworld.errors import ArtifactError, ConfigurationError, DataContractError, ResourceError
from stageworld.inference import CheckpointContract
from stageworld.model import ActionTokens
from stageworld.synthetic_workflow import _config_lineage, build_model
from stageworld.training import (
    ExperimentRegistry,
    LocalEventLogger,
    LossWeights,
    StageWorldTrainer,
    TrainingPhase,
    WorldModelBatch,
    bounded_fit,
    checkpoint_payload_mismatches,
    checkpoint_snapshot_path,
    new_checkpoint_metadata,
)

REAL_CT_FEATURE_SCHEMA = "stageworld-real-paired-ct-features-v1"
REAL_CT_FEATURE_SUMMARY_SCHEMA = "stageworld-real-paired-ct-feature-summary-v1"
REAL_WORLD_PRETRAIN_SUMMARY_SCHEMA = "stageworld-real-world-pretrain-smoke-v1"
REAL_TIMELINE_CONTRACT_VERSION = "paired-ct-os-v1-s0-s1-timeline-v1"
NO_OUTCOME_WORLD_PRETRAIN_CONTRACT = "outcomes-not-read-world-pretrain-v1"
DISABLED_PATHOLOGY_ARTIFACT_ID = "pathology-disabled-s0-s1-v1"
CHECKPOINT_SIDECAR_SCHEMA = "stageworld-checkpoint-sidecar-v2"

_ROLE_ORDER = {"baseline_ct": 0, "post_treatment_ct": 1}


@dataclass(frozen=True)
class RealFeatureBundle:
    batches_by_split: Mapping[str, tuple[WorldModelBatch, ...]]
    data_lineage_id: str
    cohort_artifact_id: str
    split_version: str
    feature_artifact_id: str
    cache_provenance: CacheProvenance
    training_patient_count: int
    validation_patient_count: int
    held_out_test_patient_count: int
    treatment_snapshot: Mapping[str, Any] | None = None


def _feature_root(config: StageWorldConfig) -> Path:
    if config.paths.feature_root:
        return Path(config.paths.feature_root).expanduser().resolve()
    return (config.output_root / "features" / "restricted").resolve()


def _private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)


def _require_list(value: object, *, code: str, message: str) -> list[Any]:
    if not isinstance(value, list):
        raise ArtifactError(code=code, message=message)
    return value


def _require_text(value: object, *, code: str, message: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactError(code=code, message=message)
    return value


def _validate_lineage(artifacts: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    data_ids = {
        _require_text(
            artifact.get("data_lineage_id"),
            code="REAL_DATA_LINEAGE_MISSING",
            message="A real-data artifact has no data lineage ID.",
        )
        for artifact in artifacts
    }
    cohort_ids = {
        _require_text(
            artifact.get("cohort_artifact_id"),
            code="REAL_COHORT_LINEAGE_MISSING",
            message="A real-data artifact has no cohort artifact ID.",
        )
        for artifact in artifacts
    }
    if len(data_ids) != 1 or len(cohort_ids) != 1:
        raise ArtifactError(
            code="REAL_ARTIFACT_LINEAGE_MISMATCH",
            message="Real cohort, asset, and split artifacts do not share one lineage.",
        )
    return next(iter(data_ids)), next(iter(cohort_ids))


def _validate_swinunetr_config(config: StageWorldConfig) -> Path:
    expected = {
        "ct_encoder": (config.model.ct_encoder, "swinunetr"),
        "ct_input_dim": (config.model.ct_input_dim, SWINUNETR_FEATURE_DIM),
        "ct_source_version": (
            config.encoders.ct_source_version,
            SWINUNETR_SOURCE_VERSION,
        ),
        "ct_component_version": (
            config.encoders.ct_component_version,
            SWINUNETR_COMPONENT_VERSION,
        ),
        "ct_preprocess_version": (
            config.encoders.ct_preprocess_version,
            (
                config.encoders.ct_preprocess_version
                if config.encoders.ct_preprocess_version in (
                    GASTRIC_ROI_VERSION, *TUMOR_ROI_VERSIONS
                )
                else SWINUNETR_PREPROCESS_VERSION
            ),
        ),
    }
    changed = sorted(name for name, (actual, required) in expected.items() if actual != required)
    if changed:
        raise ConfigurationError(
            code="SWINUNETR_CONFIG_MISMATCH",
            message="The real CT run does not match the validated Swin UNETR contract.",
            details={"fields": changed},
        )
    if config.mode is not RunMode.REAL_IMAGES:
        raise ConfigurationError(
            code="IMAGE_MODE_REQUIRED",
            message="DICOM feature extraction requires mode=real_images.",
        )
    if config.permissions.allow_remote_code:
        raise ConfigurationError(
            code="REMOTE_CODE_FORBIDDEN",
            message="The approved Swin UNETR path must not enable remote code.",
        )
    if "Apache-2.0" not in config.permissions.approved_weight_licenses:
        raise ConfigurationError(
            code="ENCODER_LICENSE_NOT_APPROVED",
            message="The Swin UNETR Apache-2.0 terms are not recorded as approved.",
        )
    raw_weight = config.encoders.ct_weight_path
    if not raw_weight:
        raise ArtifactError(
            code="ENCODER_WEIGHTS_MISSING",
            message="Set the local Swin UNETR weight path; automatic download is disabled.",
        )
    weight = Path(raw_weight).expanduser().resolve()
    if not weight.is_file():
        raise ArtifactError(
            code="ENCODER_WEIGHTS_MISSING",
            message="The configured local Swin UNETR weight file is missing.",
        )
    return weight


def _resolve_device(requested: str) -> torch.device:
    if requested not in {"cpu", "cuda"}:
        raise ConfigurationError(
            code="INVALID_CT_DEVICE",
            message="CT extraction device must be cpu or cuda.",
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise ResourceError(
            code="CUDA_UNAVAILABLE",
            message="CUDA was requested for CT extraction but is unavailable.",
        )
    return torch.device(requested)


def _autocast_context(device: torch.device, precision: str):
    if precision == "off" or device.type != "cuda":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _cpu_observation(tokens: ObservationTokens) -> ObservationTokens:
    return ObservationTokens(
        values=tokens.values.detach().float().cpu(),
        valid=tokens.valid.detach().cpu(),
        modality=tokens.modality.detach().cpu(),
        acquired_time=tokens.acquired_time.detach().float().cpu(),
        available_time=tokens.available_time.detach().float().cpu(),
        provenance=tokens.provenance,
        source_id=tokens.source_id,
        modality_name=tokens.modality_name,
        coords=None if tokens.coords is None else tokens.coords.detach().float().cpu(),
        coordinate_system=tokens.coordinate_system,
        quality_flags=tokens.quality_flags,
    )


def _real_artifacts(config: StageWorldConfig) -> tuple[dict[str, Any], ...]:
    restricted = real_data_root(config)
    try:
        cohort = read_json(restricted / "input_cohort.json")
        assets = read_json(restricted / "asset_bindings.json")
        splits = read_json(restricted / "split_assignments.json")
    except (OSError, ValueError) as error:
        raise ArtifactError(
            code="REAL_COHORT_ARTIFACT_UNREADABLE",
            message="The signed paired-CT cohort artifacts are absent or unreadable.",
        ) from error
    if cohort.get("private_schema_version") != PAIRED_CT_COHORT_SCHEMA:
        raise ArtifactError(
            code="REAL_COHORT_SCHEMA_MISMATCH",
            message="The paired-CT input cohort schema is incompatible.",
        )
    if assets.get("schema_version") != PRIVATE_ASSET_SCHEMA:
        raise ArtifactError(
            code="REAL_ASSET_SCHEMA_MISMATCH",
            message="The paired-CT asset manifest schema is incompatible.",
        )
    if splits.get("schema_version") != PRIVATE_SPLIT_SCHEMA:
        raise ArtifactError(
            code="REAL_SPLIT_SCHEMA_MISMATCH",
            message="The patient split schema is incompatible.",
        )
    if (
        assets.get("phase_selection_policy_version") != FIRST_ACQUISITION_SELECTION_POLICY_VERSION
        or config.clinical.ct_series_selection_policy_version
        != FIRST_ACQUISITION_SELECTION_POLICY_VERSION
    ):
        raise ArtifactError(
            code="CT_SELECTION_POLICY_MISMATCH",
            message="Feature extraction requires the signed first-acquisition branch.",
        )
    _validate_lineage((cohort, assets, splits))
    return cohort, assets, splits


def real_data_root(config: StageWorldConfig) -> Path:
    if config.paths.data_artifact_root:
        return Path(config.paths.data_artifact_root)
    return config.output_root / "data" / "restricted"


def _select_feature_bindings(
    assets: Mapping[str, Any],
    splits: Mapping[str, Any],
    *,
    limit_per_split: int | None,
    include_test: bool,
) -> tuple[tuple[Mapping[str, Any], str], ...]:
    if limit_per_split is not None and limit_per_split <= 0:
        raise ConfigurationError(
            code="INVALID_EXTRACTION_LIMIT",
            message="limit_per_split must be positive when provided.",
        )
    assignments = _require_list(
        splits.get("assignments"),
        code="REAL_SPLIT_MANIFEST_INVALID",
        message="The real split manifest contains no assignments.",
    )
    split_by_patient: dict[str, str] = {}
    for raw in assignments:
        if not isinstance(raw, Mapping):
            raise ArtifactError(
                code="REAL_SPLIT_MANIFEST_INVALID",
                message="Every split assignment must be a mapping.",
            )
        patient_id = str(raw.get("patient_id", ""))
        split = str(raw.get("split", ""))
        if not patient_id or split not in {"train", "validation", "test"}:
            raise ArtifactError(
                code="REAL_SPLIT_MANIFEST_INVALID",
                message="A split assignment has an invalid anonymous patient or split.",
            )
        split_by_patient[patient_id] = split

    allowed_splits = (
        ("train", "validation", "test")
        if include_test
        else (
            "train",
            "validation",
        )
    )
    patients_by_split: dict[str, list[str]] = defaultdict(list)
    for patient_id, split in split_by_patient.items():
        if split in allowed_splits:
            patients_by_split[split].append(patient_id)
    selected_patients: set[str] = set()
    for split in allowed_splits:
        ranked = sorted(patients_by_split[split])
        if limit_per_split is not None:
            ranked = ranked[:limit_per_split]
        selected_patients.update(ranked)

    bindings = _require_list(
        assets.get("bindings"),
        code="REAL_ASSET_MANIFEST_INVALID",
        message="The real CT asset manifest contains no bindings.",
    )
    selected: list[tuple[Mapping[str, Any], str]] = []
    role_counts: Counter[tuple[str, str]] = Counter()
    for raw in bindings:
        if not isinstance(raw, Mapping):
            raise ArtifactError(
                code="REAL_ASSET_MANIFEST_INVALID",
                message="Every CT asset binding must be a mapping.",
            )
        patient_id = str(raw.get("patient_id", ""))
        if patient_id not in selected_patients:
            continue
        role = str(raw.get("role", ""))
        if role not in _ROLE_ORDER:
            raise ArtifactError(
                code="REAL_ASSET_MANIFEST_INVALID",
                message="A selected CT binding has an unsupported role.",
            )
        split = split_by_patient[patient_id]
        selected.append((raw, split))
        role_counts[(patient_id, role)] += 1
    invalid_pairs = [
        patient_id
        for patient_id in selected_patients
        if any(role_counts[(patient_id, role)] != 1 for role in _ROLE_ORDER)
    ]
    if invalid_pairs:
        raise ArtifactError(
            code="REAL_CT_PAIR_BINDING_MISMATCH",
            message="Every selected patient must have exactly one S0 and one S1 CT binding.",
            details={"invalid_patient_count": len(invalid_pairs)},
        )
    selected.sort(
        key=lambda item: (
            item[1],
            str(item[0]["patient_id"]),
            _ROLE_ORDER[str(item[0]["role"])],
        )
    )
    return tuple(selected)


def extract_real_ct_features(
    config: StageWorldConfig,
    *,
    pseudonymizer: HMACPseudonymizer,
    limit_per_split: int | None = None,
    include_test: bool = False,
    device: str = "cuda",
) -> dict[str, Any]:
    """Extract frozen Swin features for a deterministic, outcome-blind subset."""

    weight_path = _validate_swinunetr_config(config)
    if not config.paths.approved_data_root:
        raise ConfigurationError(
            code="APPROVED_DATA_ROOT_REQUIRED",
            message="Real CT extraction requires an approved data root.",
        )
    selected_device = _resolve_device(device)
    cohort, assets, splits = _real_artifacts(config)
    data_lineage_id, cohort_artifact_id = _validate_lineage((cohort, assets, splits))
    split_version = _require_text(
        splits.get("split_version"),
        code="REAL_SPLIT_VERSION_MISSING",
        message="The real split manifest has no split version.",
    )
    selected = _select_feature_bindings(
        assets,
        splits,
        limit_per_split=limit_per_split,
        include_test=include_test,
    )
    observations = _require_list(
        cohort.get("observations"),
        code="REAL_INPUT_COHORT_INVALID",
        message="The real input cohort contains no observations.",
    )
    observation_by_asset: dict[str, Mapping[str, Any]] = {}
    for raw in observations:
        if not isinstance(raw, Mapping):
            raise ArtifactError(
                code="REAL_INPUT_COHORT_INVALID",
                message="Every real cohort observation must be a mapping.",
            )
        asset_id = str(raw.get("local_asset_id", ""))
        if asset_id:
            observation_by_asset[asset_id] = raw

    load_result = load_swinunetr_ssl_backbone(weight_path, device=selected_device)
    access = EncoderAccess(
        mode=RunMode.REAL_IMAGES,
        source_version=SWINUNETR_SOURCE_VERSION,
        component_versions={"swinunetr": SWINUNETR_COMPONENT_VERSION},
        preprocess_version=config.encoders.ct_preprocess_version,
        version_approved=True,
        license_name="Apache-2.0",
        license_approved=True,
        weight_paths={"swinunetr": weight_path},
        network_enabled=False,
    )
    encoder = SwinUNETREncoder(
        access=access,
        backend=load_result.backend,
        feature_dim=SWINUNETR_FEATURE_DIM,
    )
    feature_root = _feature_root(config)
    _private_directory(feature_root)
    tumor_segmenter = None
    if config.encoders.ct_preprocess_version in TUMOR_ROI_VERSIONS:
        if not config.encoders.segmentation_home:
            raise ConfigurationError(
                code="SEGMENTER_HOME_REQUIRED", message="Configure the offline tumor model."
            )
        tumor_segmenter = TumorSegmenter(
            Path(config.encoders.segmentation_home), feature_root / "tumor_masks", device=device,
            preprocess_version=config.encoders.ct_preprocess_version,
        )
    cache_provenance = CacheProvenance.from_encoder(
        encoder.provenance,
        schema_version=REAL_CT_FEATURE_SCHEMA,
        patch_sampling_version=(
            f"{config.encoders.ct_preprocess_version}:{tumor_segmenter.model_artifact_id}"
            if tumor_segmenter is not None
            else GASTRIC_ROI_VERSION
            if config.encoders.ct_preprocess_version == GASTRIC_ROI_VERSION
            else "deterministic-center-crop-96-v1"
        ),
        split_version=split_version,
        target_transform_version="spatial-mean-768-v1",
        teacher_version=SWINUNETR_COMPONENT_VERSION,
    )
    segmenter: StomachSegmenter | TumorSegmenter | None = tumor_segmenter
    if config.encoders.ct_preprocess_version == GASTRIC_ROI_VERSION:
        if not config.encoders.segmentation_home:
            raise ConfigurationError(
                code="SEGMENTER_HOME_REQUIRED",
                message="Configure the offline segmenter weights.",
            )
        segmenter = StomachSegmenter(
            Path(config.encoders.segmentation_home),
            feature_root / "stomach_masks",
            device=device,
        )
    cache = FeatureCache(feature_root / "ct_entries")
    if selected_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected_device)
    started = time.monotonic()
    success_entries: list[dict[str, str]] = []
    failure_entries: list[dict[str, str]] = []
    failure_counts: Counter[str] = Counter()
    cache_hits = 0
    selected_slice_counts: list[int] = []
    roi_sources: Counter[str] = Counter()

    for binding, split in selected:
        asset_id = str(binding.get("asset_id", ""))
        patient_id = str(binding.get("patient_id", ""))
        role = str(binding.get("role", ""))
        observation = observation_by_asset.get(asset_id)
        if (
            not asset_id
            or not patient_id
            or observation is None
            or str(observation.get("patient_id", "")) != patient_id
            or str(observation.get("role", "")) != role
        ):
            raise ArtifactError(
                code="REAL_OBSERVATION_ASSET_MISMATCH",
                message="A selected CT asset does not match its input-cohort observation.",
            )
        try:
            decision = cache.begin(asset_id, cache_provenance)
            if decision.already_complete:
                cache.load(asset_id, cache_provenance)
                cache_hits += 1
            else:
                try:
                    preprocess = segmenter.preprocess if segmenter else preprocess_selected_ct
                    processed = preprocess(
                        binding,
                        approved_root=config.paths.approved_data_root,
                        pseudonymizer=pseudonymizer,
                    )
                    selected_slice_counts.append(processed.selected_file_count)
                    acquired = torch.tensor(
                        [float(observation["acquired_at_days"])], dtype=torch.float32
                    )
                    available = torch.tensor(
                        [float(observation["available_at_days"])], dtype=torch.float32
                    )
                    image = processed.image[None].to(selected_device)
                    geometry = processed.geometry
                    if selected_device.type == "cuda":
                        geometry = type(geometry)(
                            spacing_mm=geometry.spacing_mm.to(selected_device),
                            origin_mm=geometry.origin_mm.to(selected_device),
                            direction=geometry.direction.to(selected_device),
                            spatial_shape=geometry.spatial_shape.to(selected_device),
                        )
                    with (
                        torch.inference_mode(),
                        _autocast_context(selected_device, config.encoders.ct_inference_precision),
                    ):
                        encoded = encoder.encode(
                            image,
                            geometry=geometry,
                            source_ids=(asset_id,),
                            acquired_time=acquired.to(selected_device),
                            available_time=available.to(selected_device),
                            quality_flags=(processed.quality_flags,),
                        )
                    cache.store(
                        asset_id,
                        cache_provenance,
                        _cpu_observation(encoded.observations),
                    )
                except Exception as error:
                    failure_code = (
                        "CUDA_OUT_OF_MEMORY"
                        if isinstance(error, torch.OutOfMemoryError)
                        else str(getattr(error, "code", "ENCODER_EXTRACTION_FAILED"))
                    )
                    if not failure_code.isupper():
                        failure_code = "ENCODER_EXTRACTION_FAILED"
                    cache.mark_failed(
                        asset_id,
                        cache_provenance,
                        failure_code=failure_code,
                    )
                    raise
            roi_metadata: dict[str, str] = {}
            if config.encoders.ct_preprocess_version == COARSE_TUMOR_ROI_VERSION:
                qc = read_json(feature_root / "tumor_masks" / f"{asset_id}.qc.json")
                source = qc.get("roi_source")
                if source not in ("tumor_candidate", "stomach_fallback"):
                    raise ArtifactError(
                        code="COARSE_ROI_SOURCE_INVALID", message="Missing coarse ROI provenance."
                    )
                roi_metadata["roi_source"] = str(source)
                roi_sources[str(source)] += 1
            success_entries.append(
                {
                    "asset_id": asset_id,
                    "patient_id": patient_id,
                    "role": role,
                    "split": split,
                    "cache_entry_id": asset_id,
                    **roi_metadata,
                }
            )
        except torch.OutOfMemoryError as error:
            failure_counts["CUDA_OUT_OF_MEMORY"] += 1
            failure_entries.append(
                {
                    "asset_id": asset_id,
                    "patient_id": patient_id,
                    "role": role,
                    "split": split,
                    "failure_code": "CUDA_OUT_OF_MEMORY",
                }
            )
            raise ResourceError(
                code="CUDA_OUT_OF_MEMORY",
                message="Swin UNETR feature extraction exhausted device memory.",
            ) from error
        except (ArtifactError, ConfigurationError, DataContractError, ResourceError) as error:
            failure_counts[error.code] += 1
            failure_entries.append(
                {
                    "asset_id": asset_id,
                    "patient_id": patient_id,
                    "role": role,
                    "split": split,
                    "failure_code": error.code,
                }
            )
        atomic_write_json(
            config.output_root / "features" / "extraction_progress.json",
            {
                "requested_studies": len(selected),
                "complete_studies": len(success_entries),
                "failed_studies": len(failure_entries),
                "failure_counts": dict(failure_counts),
                "cache_hits": cache_hits,
                "roi_source_counts_before_pair_filter": dict(roi_sources),
                "elapsed_seconds": time.monotonic() - started,
                "preprocess_version": config.encoders.ct_preprocess_version,
            },
        )

    roles_by_patient: dict[str, set[str]] = defaultdict(set)
    split_by_patient: dict[str, str] = {}
    for entry in success_entries:
        roles_by_patient[entry["patient_id"]].add(entry["role"])
        split_by_patient[entry["patient_id"]] = entry["split"]
    complete_patients = {
        patient_id for patient_id, roles in roles_by_patient.items() if roles == set(_ROLE_ORDER)
    }
    successful_studies_before_pair_filter = len(success_entries)
    success_entries = [
        entry for entry in success_entries if entry["patient_id"] in complete_patients
    ]
    pair_counts = Counter(split_by_patient[patient_id] for patient_id in complete_patients)
    feature_artifact_id = new_artifact_id("real-ct-features")
    manifest = {
        "schema_version": REAL_CT_FEATURE_SCHEMA,
        "feature_artifact_id": feature_artifact_id,
        "data_lineage_id": data_lineage_id,
        "cohort_artifact_id": cohort_artifact_id,
        "split_version": split_version,
        "series_selection_policy_version": FIRST_ACQUISITION_SELECTION_POLICY_VERSION,
        "encoder_provenance": encoder.provenance.as_dict(),
        "cache_provenance": cache_provenance.as_dict(),
        "entries": success_entries,
        "failures": failure_entries,
        "outcome_data_read": False,
        "contains_raw_paths_uids_dates_or_pixels": False,
        "test_features_included": include_test,
        "limit_per_split": limit_per_split,
    }
    manifest_path = feature_root / "ct_manifest.json"
    atomic_write_private_json(manifest_path, manifest)
    elapsed = time.monotonic() - started
    summary = {
        "schema_version": REAL_CT_FEATURE_SUMMARY_SCHEMA,
        "status": "ok" if complete_patients else "failed",
        "mode": "real_images",
        "clinical_validation": False,
        "engineering_smoke_only": limit_per_split is not None,
        "feature_artifact_id": feature_artifact_id,
        "encoder": "swinunetr",
        "encoder_source_version": SWINUNETR_SOURCE_VERSION,
        "encoder_component_version": SWINUNETR_COMPONENT_VERSION,
        "preprocess_version": config.encoders.ct_preprocess_version,
        "backbone_loaded_tensor_count": len(load_result.state_dict_report.loaded_keys),
        "backbone_loaded_parameter_fraction": (
            load_result.state_dict_report.loaded_parameter_fraction
        ),
        "excluded_task_head_tensor_count": len(load_result.excluded_task_head_keys),
        "requested_study_count": len(selected),
        "complete_study_count": len(success_entries),
        "successful_studies_before_pair_filter": successful_studies_before_pair_filter,
        "roi_source_counts_before_pair_filter": dict(roi_sources),
        "roi_source_counts_in_complete_pairs": dict(Counter(
            entry["roi_source"] for entry in success_entries if "roi_source" in entry
        )),
        "complete_patient_pair_counts_by_split": dict(sorted(pair_counts.items())),
        "failure_counts": dict(sorted(failure_counts.items())),
        "cache_hit_count": cache_hits,
        "elapsed_seconds": elapsed,
        "selected_slice_count_range": (
            [min(selected_slice_counts), max(selected_slice_counts)]
            if selected_slice_counts
            else None
        ),
        "device": selected_device.type,
        "peak_cuda_memory_gib": (
            torch.cuda.max_memory_allocated(selected_device) / (1024**3)
            if selected_device.type == "cuda"
            else None
        ),
        "outcome_data_read": False,
        "test_features_included": include_test,
        "test_used_for_optimization_or_selection": False,
        "contains_identifiers_paths_uids_dates_or_pixels": False,
        "manual_anatomic_and_artifact_review_pending": True,
    }
    summary_path = config.output_root / "features" / "ct_summary.json"
    atomic_write_json(summary_path, summary)
    if not complete_patients:
        raise ArtifactError(
            code="NO_COMPLETE_REAL_CT_FEATURE_PAIRS",
            message="Feature extraction produced no complete S0/S1 patient pair.",
        )
    return {**summary, "summary": str(summary_path)}


def _stack_observations(values: Sequence[ObservationTokens]) -> ObservationTokens:
    if not values:
        raise DataContractError(
            code="EMPTY_REAL_FEATURE_BATCH",
            message="A real feature batch cannot be empty.",
        )
    first = values[0]
    for value in values:
        if (
            value.batch_size != 1
            or value.token_count != first.token_count
            or value.provenance != first.provenance
            or value.modality_name != first.modality_name
            or value.coordinate_system != first.coordinate_system
            or (value.coords is None) != (first.coords is None)
        ):
            raise DataContractError(
                code="INCOMPATIBLE_REAL_FEATURE_BATCH",
                message="Cached CT observations cannot be stacked under one feature contract.",
            )
    return ObservationTokens(
        values=torch.cat([value.values for value in values], dim=0),
        valid=torch.cat([value.valid for value in values], dim=0),
        modality=torch.cat([value.modality for value in values], dim=0),
        acquired_time=torch.cat([value.acquired_time for value in values], dim=0),
        available_time=torch.cat([value.available_time for value in values], dim=0),
        provenance=first.provenance,
        source_id=tuple(value.source_id[0] for value in values),
        modality_name=first.modality_name,
        coords=(
            None
            if first.coords is None
            else torch.cat([cast(Tensor, value.coords) for value in values], dim=0)
        ),
        coordinate_system=first.coordinate_system,
        quality_flags=tuple(
            value.quality_flags[0] if value.quality_flags else () for value in values
        ),
    )


def _masked_observation(
    modality: Literal["pathology", "clinical"],
    *,
    batch: int,
    feature_dim: int,
    times: Tensor,
) -> ObservationTokens:
    modality_id = 1 if modality == "pathology" else 2
    return ObservationTokens(
        values=torch.zeros(batch, 1, feature_dim),
        valid=torch.zeros(batch, 1, dtype=torch.bool),
        modality=torch.full((batch, 1), modality_id, dtype=torch.long),
        acquired_time=times[:, None].clone(),
        available_time=times[:, None].clone(),
        provenance=EncoderProvenance(
            encoder_name=f"disabled_{modality}_placeholder",
            source_version="paired-ct-s0-s1-scope-v1",
            component_versions=(("mask", "explicit-unavailable-v1"),),
            preprocess_version="no-values-enter-model-v1",
            feature_dim=feature_dim,
        ),
        source_id=tuple(("",) for _ in range(batch)),
        modality_name=modality,
        quality_flags=tuple(("outside_signed_s0_s1_scope",) for _ in range(batch)),
    )


def _paired_batch(
    config: StageWorldConfig,
    patient_ids: Sequence[str],
    features: Mapping[str, Mapping[str, ObservationTokens]],
    query_times: Mapping[tuple[str, str], float],
) -> WorldModelBatch:
    ids = tuple(patient_ids)
    ct0 = _stack_observations([features[patient_id]["baseline_ct"] for patient_id in ids])
    ct1 = _stack_observations([features[patient_id]["post_treatment_ct"] for patient_id in ids])
    batch = len(ids)
    s0_time = torch.tensor(
        [query_times[(patient_id, "s0")] for patient_id in ids], dtype=torch.float32
    )
    s1_time = torch.tensor(
        [query_times[(patient_id, "s1")] for patient_id in ids], dtype=torch.float32
    )
    ct1_acquisition = ct1.acquired_time[:, 0].clone()
    ct1_availability = ct1.available_time[:, 0].clone()
    if (
        (ct0.available_time > s0_time[:, None]).any()
        or (ct1_availability > s1_time).any()
        or (ct1_acquisition < s0_time).any()
    ):
        raise DataContractError(
            code="REAL_FEATURE_TIMELINE_INVALID",
            message="Cached paired CT times violate the signed S0/S1 prefix contract.",
        )
    clinical = _masked_observation(
        "clinical",
        batch=batch,
        feature_dim=config.model.clinical_input_dim,
        times=s0_time,
    )
    pathology = _masked_observation(
        "pathology",
        batch=batch,
        feature_dim=config.model.pathology_input_dim,
        times=s1_time,
    )
    ct_count = ct1.valid.sum(dim=1, keepdim=True)
    if (ct_count <= 0).any():
        raise DataContractError(
            code="EMPTY_REAL_CT_TARGET",
            message="Every real world-pretraining row requires a valid S1 CT target.",
        )
    future_ct = (
        (ct1.values * ct1.valid[..., None]).sum(dim=1, keepdim=True)
        / ct_count.clamp_min(1)[..., None]
    ).detach()
    empty_actions = ActionTokens.empty(
        batch_size=batch,
        value_dim=config.model.action_input_dim,
        device=torch.device("cpu"),
    )
    return WorldModelBatch(
        patient_ids=ids,
        ct0=ct0,
        clinical0=clinical,
        s0_time=s0_time,
        treatment_actions=empty_actions,
        ct1_acquisition_time=ct1_acquisition,
        ct1=ct1,
        s1_time=s1_time,
        surgery_actions=ActionTokens.empty(
            batch_size=batch,
            value_dim=config.model.action_input_dim,
            device=torch.device("cpu"),
        ),
        pathology_acquisition_time=s1_time.clone(),
        pathology=pathology,
        s2_time=s1_time.clone(),
        horizons=torch.tensor((0.0, *config.survival.report_horizons_years)),
        future_ct_target=future_ct,
        future_ct_valid=torch.ones(batch, 1, dtype=torch.bool),
        future_pathology_target=torch.zeros(batch, 1, config.model.pathology_input_dim),
        future_pathology_valid=torch.zeros(batch, 1, dtype=torch.bool),
        survival_durations=torch.zeros(batch, 3),
        survival_events=torch.zeros(batch, 3, dtype=torch.long),
        survival_valid=torch.zeros(batch, 3, dtype=torch.bool),
        ct1_availability_time=ct1_availability,
        ct1_unavailable_event_mask=torch.zeros(batch, dtype=torch.bool),
        pathology_availability_time=s1_time.clone(),
        pathology_unavailable_event_mask=torch.zeros(batch, dtype=torch.bool),
    )


def load_real_world_pretrain_batches(config: StageWorldConfig) -> RealFeatureBundle:
    """Load train/validation cached features without opening the outcome artifact."""

    _validate_swinunetr_config(config)
    cohort, assets, splits = _real_artifacts(config)
    del assets
    data_lineage_id, cohort_artifact_id = _validate_lineage((cohort, splits))
    feature_root = _feature_root(config)
    try:
        manifest = read_json(feature_root / "ct_manifest.json")
    except (OSError, ValueError) as error:
        raise ArtifactError(
            code="REAL_CT_FEATURE_MANIFEST_UNREADABLE",
            message="The real CT feature manifest is absent or unreadable.",
        ) from error
    if manifest.get("schema_version") != REAL_CT_FEATURE_SCHEMA:
        raise ArtifactError(
            code="REAL_CT_FEATURE_SCHEMA_MISMATCH",
            message="The real CT feature manifest schema is incompatible.",
        )
    manifest_data, manifest_cohort = _validate_lineage((manifest,))
    if (manifest_data, manifest_cohort) != (data_lineage_id, cohort_artifact_id):
        raise ArtifactError(
            code="REAL_CT_FEATURE_LINEAGE_MISMATCH",
            message="Cached CT features belong to a different paired cohort.",
        )
    if manifest.get("outcome_data_read") is not False:
        raise ArtifactError(
            code="WORLD_PRETRAIN_OUTCOME_ISOLATION_FAILED",
            message="World pretraining requires an outcome-blind feature artifact.",
        )
    if manifest.get("test_features_included") is not False:
        raise ArtifactError(
            code="WORLD_PRETRAIN_TEST_ISOLATION_FAILED",
            message="World pretraining requires a feature artifact that excludes the test split.",
        )
    expected_encoder = EncoderProvenance(
        encoder_name="swinunetr",
        source_version=SWINUNETR_SOURCE_VERSION,
        component_versions=(("swinunetr", SWINUNETR_COMPONENT_VERSION),),
        preprocess_version=config.encoders.ct_preprocess_version,
        feature_dim=SWINUNETR_FEATURE_DIM,
    )
    try:
        manifest_encoder = EncoderProvenance.from_dict(manifest["encoder_provenance"])
        cache_provenance = CacheProvenance.from_dict(manifest["cache_provenance"])
    except (KeyError, TypeError, ValueError, DataContractError) as error:
        raise ArtifactError(
            code="REAL_CT_FEATURE_PROVENANCE_INVALID",
            message="The real CT feature manifest has malformed encoder provenance.",
        ) from error
    if manifest_encoder != expected_encoder:
        raise ArtifactError(
            code="REAL_CT_FEATURE_PROVENANCE_MISMATCH",
            message="Cached CT features do not match the configured Swin UNETR contract.",
        )
    if config.encoders.ct_preprocess_version in TUMOR_ROI_VERSIONS:
        if not config.encoders.segmentation_home:
            raise ArtifactError(code="SEGMENTER_HOME_REQUIRED", message="Bind the tumor model.")
        model_id = bind_tumor_model(
            Path(config.encoders.segmentation_home), feature_root / "tumor_masks", create=False,
            preprocess_version=config.encoders.ct_preprocess_version,
        )
        if cache_provenance.patch_sampling_version != (
            f"{config.encoders.ct_preprocess_version}:{model_id}"
        ):
            raise ArtifactError(
                code="TUMOR_FEATURE_MODEL_MISMATCH",
                message="Tumor features must match the exact retained segmentation model.",
            )
    split_version = _require_text(
        splits.get("split_version"),
        code="REAL_SPLIT_VERSION_MISSING",
        message="The real split manifest has no split version.",
    )
    if cache_provenance.split_version != split_version:
        raise ArtifactError(
            code="REAL_CT_FEATURE_SPLIT_MISMATCH",
            message="Cached CT features do not match the fixed patient split.",
        )

    raw_assignments = _require_list(
        splits.get("assignments"),
        code="REAL_SPLIT_MANIFEST_INVALID",
        message="The real split manifest contains no assignments.",
    )
    split_by_patient = {
        str(item["patient_id"]): str(item["split"])
        for item in raw_assignments
        if isinstance(item, Mapping)
    }
    held_out_test_count = sum(value == "test" for value in split_by_patient.values())
    queries = _require_list(
        cohort.get("queries"),
        code="REAL_INPUT_COHORT_INVALID",
        message="The real input cohort contains no stage queries.",
    )
    query_times: dict[tuple[str, str], float] = {}
    for raw in queries:
        if not isinstance(raw, Mapping):
            raise ArtifactError(
                code="REAL_INPUT_COHORT_INVALID",
                message="Every real cohort query must be a mapping.",
            )
        stage = str(raw.get("stage", "")).lower()
        if stage in {"s0", "s1"}:
            query_times[(str(raw.get("patient_id", "")), stage)] = float(raw["query_time_days"])

    entries = _require_list(
        manifest.get("entries"),
        code="REAL_CT_FEATURE_MANIFEST_INVALID",
        message="The real CT feature manifest contains no entries.",
    )
    cache = FeatureCache(feature_root / "ct_entries")
    features: dict[str, dict[str, ObservationTokens]] = defaultdict(dict)
    for raw in entries:
        if not isinstance(raw, Mapping):
            raise ArtifactError(
                code="REAL_CT_FEATURE_MANIFEST_INVALID",
                message="Every real CT feature entry must be a mapping.",
            )
        patient_id = str(raw.get("patient_id", ""))
        split = str(raw.get("split", ""))
        role = str(raw.get("role", ""))
        entry_id = str(raw.get("cache_entry_id", ""))
        if (
            split not in {"train", "validation"}
            or split_by_patient.get(patient_id) != split
            or role not in _ROLE_ORDER
            or not entry_id
        ):
            raise ArtifactError(
                code="REAL_CT_FEATURE_MANIFEST_INVALID",
                message="A cached CT feature entry violates the fixed split or role contract.",
            )
        if role in features[patient_id]:
            raise ArtifactError(
                code="DUPLICATE_REAL_CT_FEATURE",
                message="A patient has duplicate cached CT features for one stage.",
            )
        features[patient_id][role] = cache.load(entry_id, cache_provenance)

    patients_by_split: dict[str, list[str]] = {"train": [], "validation": []}
    for patient_id, by_role in features.items():
        if set(by_role) != set(_ROLE_ORDER):
            continue
        if (patient_id, "s0") not in query_times or (patient_id, "s1") not in query_times:
            raise ArtifactError(
                code="REAL_QUERY_FEATURE_MISMATCH",
                message="A complete CT feature pair has no matching S0/S1 queries.",
            )
        patient_split = split_by_patient.get(patient_id)
        if patient_split is not None and patient_split in patients_by_split:
            patients_by_split[patient_split].append(patient_id)
    if not patients_by_split["train"] or not patients_by_split["validation"]:
        raise DataContractError(
            code="EMPTY_REAL_DEVELOPMENT_SPLIT",
            message="World pretraining requires cached pairs in train and validation splits.",
        )
    batches: dict[str, tuple[WorldModelBatch, ...]] = {}
    for split, patient_ids in patients_by_split.items():
        ordered = sorted(patient_ids)
        split_batches = []
        for start in range(0, len(ordered), config.training.patient_batch_size):
            split_batches.append(
                _paired_batch(
                    config,
                    ordered[start : start + config.training.patient_batch_size],
                    features,
                    query_times,
                )
            )
        batches[split] = tuple(split_batches)
    return RealFeatureBundle(
        batches_by_split=batches,
        data_lineage_id=data_lineage_id,
        cohort_artifact_id=cohort_artifact_id,
        split_version=split_version,
        feature_artifact_id=_require_text(
            manifest.get("feature_artifact_id"),
            code="REAL_CT_FEATURE_ARTIFACT_ID_MISSING",
            message="The real CT feature manifest has no artifact ID.",
        ),
        cache_provenance=cache_provenance,
        training_patient_count=len(patients_by_split["train"]),
        validation_patient_count=len(patients_by_split["validation"]),
        held_out_test_patient_count=held_out_test_count,
    )


def _secure_real_run_artifacts(paths: Sequence[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        if path.is_dir():
            path.chmod(0o700)
        elif stat.S_ISREG(path.stat().st_mode):
            path.chmod(0o600)


def run_real_world_pretraining_smoke(
    config: StageWorldConfig,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Run at most ten outcome-blind S0-to-S1 representation-learning steps."""

    if resume:
        raise ConfigurationError(
            code="REAL_SMOKE_RESUME_NOT_IMPLEMENTED",
            message="Real smoke resume is not enabled until its first checkpoint is audited.",
        )
    if tuple(config.clinical.development_stages) != ("s0", "s1"):
        raise ConfigurationError(
            code="REAL_SMOKE_STAGE_SCOPE_MISMATCH",
            message="This real smoke path is restricted to the signed S0/S1 scope.",
        )
    bundle = load_real_world_pretrain_batches(config)
    random.seed(config.training.seed)
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    model = build_model(config)
    config_lineage = _config_lineage(config)
    metadata = new_checkpoint_metadata(
        model=model,
        mode=config.mode.value,
        config_lineage_id=config_lineage,
        data_lineage_id=bundle.data_lineage_id,
        cohort_artifact_id=bundle.cohort_artifact_id,
        split_version=bundle.split_version,
        ct_feature_artifact_id=bundle.feature_artifact_id,
        pathology_feature_artifact_id=DISABLED_PATHOLOGY_ARTIFACT_ID,
        timeline_contract_version=REAL_TIMELINE_CONTRACT_VERSION,
        outcome_contract_version=NO_OUTCOME_WORLD_PRETRAIN_CONTRACT,
        training_seed=config.training.seed,
        source_schema_version=PRIVATE_ASSET_SCHEMA,
        cohort_schema_version=PAIRED_CT_COHORT_SCHEMA,
        feature_schema_version=REAL_CT_FEATURE_SCHEMA,
        phase=TrainingPhase.WORLD_PRETRAIN,
        selection_rule="engineering-smoke-validation-loss-not-model-selection-v1",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    max_steps = min(config.training.smoke_max_steps, config.training.world_pretrain_steps, 10)
    training_batch_count = len(bundle.batches_by_split["train"])
    validation_batch_count = len(bundle.batches_by_split["validation"])
    if max_steps < training_batch_count:
        raise ConfigurationError(
            code="REAL_SMOKE_COHORT_COVERAGE_INCOMPLETE",
            message=(
                "The bounded real smoke must expose every training patient at least once; "
                "increase the configured patient batch size within the validated memory budget."
            ),
            details={
                "training_batch_count": training_batch_count,
                "maximum_optimizer_steps": max_steps,
            },
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(max_steps, 1),
        eta_min=config.training.lr * 0.1,
    )
    phase_root = config.output_root / "runs" / TrainingPhase.WORLD_PRETRAIN.value
    _private_directory(phase_root)
    run_root = phase_root / metadata.checkpoint_id
    _private_directory(run_root)
    checkpoint_path = run_root / "checkpoint.pt"
    metadata_path = run_root / "checkpoint_metadata.json"
    summary_path = run_root / "summary.json"
    event_path = run_root / "events.jsonl"
    logger = LocalEventLogger(event_path)
    trainer = StageWorldTrainer(
        model,
        optimizer,
        scheduler=scheduler,
        device="cuda" if torch.cuda.is_available() else "cpu",
        mixed_precision=config.training.mixed_precision,
        grad_clip_norm=config.training.grad_clip_norm,
        survival_time_unit=config.survival.time_unit,
        survival_parameterization=config.survival.parameterization,
        survival_open_tail_interval=config.survival.open_tail_interval,
        event_logger=logger,
    )
    registry = ExperimentRegistry(config.output_root / "experiment_registry.csv")
    run_id = f"world_pretrain-{metadata.checkpoint_id}"
    started_at = time.time()
    registry.update(
        {
            "run_id": run_id,
            "status": "running",
            "mode": config.mode.value,
            "phase": TrainingPhase.WORLD_PRETRAIN.value,
            "seed": config.training.seed,
            "config_lineage_id": config_lineage,
            "data_lineage_id": bundle.data_lineage_id,
            "started_at_unix": started_at,
        }
    )
    weights = LossWeights(
        survival=0.0,
        future_ct=1.0,
        future_pathology=0.0,
        kl=0.01,
    )
    try:
        history = bounded_fit(
            trainer,
            bundle.batches_by_split["train"],
            phase=TrainingPhase.WORLD_PRETRAIN,
            max_steps=max_steps,
            max_minutes=config.training.smoke_max_minutes,
            weights=weights,
            kl_warmup_steps=max(1, max_steps // 3),
        )
        if trainer.state.optimizer_step != max_steps:
            raise ArtifactError(
                code="SMOKE_TIME_BUDGET_REACHED",
                message="Real world pretraining stopped before its bounded step target.",
            )
        validation = trainer.evaluate(
            bundle.batches_by_split["validation"],
            phase=TrainingPhase.WORLD_PRETRAIN,
            weights=weights,
        )
        trainer.save_checkpoint(
            checkpoint_path,
            metadata,
            sampler_state={"batch_cursor": 0},
        )
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, EOFError) as error:
            raise ArtifactError(
                code="CHECKPOINT_UNREADABLE",
                message="The saved real smoke checkpoint cannot be reopened.",
            ) from error
        if not isinstance(payload, Mapping):
            raise ArtifactError(
                code="CHECKPOINT_INVALID",
                message="The saved real smoke checkpoint root is invalid.",
            )
        contract = CheckpointContract.from_checkpoint_payload(payload)
        snapshot_path = checkpoint_snapshot_path(checkpoint_path, contract.weight_version)
        snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=True)
        if not isinstance(snapshot, Mapping) or checkpoint_payload_mismatches(payload, snapshot):
            raise ArtifactError(
                code="CHECKPOINT_POINTER_SNAPSHOT_MISMATCH",
                message="The real smoke checkpoint differs from its immutable snapshot.",
            )
        atomic_write_private_json(
            metadata_path,
            {
                "schema_version": CHECKPOINT_SIDECAR_SCHEMA,
                "active_weight_version": contract.weight_version,
                "metadata": asdict(metadata),
            },
        )
        summary = {
            "schema_version": REAL_WORLD_PRETRAIN_SUMMARY_SCHEMA,
            "status": "ok",
            "mode": config.mode.value,
            "phase": TrainingPhase.WORLD_PRETRAIN.value,
            "engineering_smoke_only": True,
            "clinical_validation": False,
            "manual_ct_review_pending": True,
            "optimizer_steps": trainer.state.optimizer_step,
            "initial_loss": history[0]["loss"] if history else None,
            "final_loss": history[-1]["loss"] if history else None,
            "validation": validation,
            "training_patient_count": bundle.training_patient_count,
            "validation_patient_count": bundle.validation_patient_count,
            "training_batch_count": training_batch_count,
            "validation_batch_count": validation_batch_count,
            "all_training_patients_seen_at_least_once": True,
            "held_out_test_patient_count": bundle.held_out_test_patient_count,
            "test_features_read": False,
            "test_used_for_optimization_or_selection": False,
            "outcome_data_read": False,
            "treatment_conditioning": "elapsed_time_only",
            "development_stages": ["s0", "s1"],
            "checkpoint": str(checkpoint_path),
            "checkpoint_snapshot": str(snapshot_path),
            "checkpoint_id": metadata.checkpoint_id,
            "weight_version": contract.weight_version,
            "ct_feature_artifact_id": bundle.feature_artifact_id,
            "pathology_feature_artifact_id": DISABLED_PATHOLOGY_ARTIFACT_ID,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "device": trainer.device.type,
        }
        atomic_write_json(summary_path, summary)
        registry.update(
            {
                "run_id": run_id,
                "status": "completed",
                "mode": config.mode.value,
                "phase": TrainingPhase.WORLD_PRETRAIN.value,
                "seed": config.training.seed,
                "config_lineage_id": config_lineage,
                "data_lineage_id": bundle.data_lineage_id,
                "started_at_unix": started_at,
                "finished_at_unix": time.time(),
            }
        )
        _secure_real_run_artifacts(
            (
                run_root,
                phase_root,
                run_root / "checkpoint_versions",
                checkpoint_path,
                snapshot_path,
                metadata_path,
                summary_path,
                event_path,
                config.output_root / "experiment_registry.csv",
            )
        )
        return {**summary, "summary": str(summary_path)}
    except Exception as error:
        failure_code = (
            "CUDA_OUT_OF_MEMORY"
            if isinstance(error, torch.OutOfMemoryError)
            else str(getattr(error, "code", type(error).__name__.upper()))
        )
        registry.update(
            {
                "run_id": run_id,
                "status": "failed",
                "mode": config.mode.value,
                "phase": TrainingPhase.WORLD_PRETRAIN.value,
                "seed": config.training.seed,
                "config_lineage_id": config_lineage,
                "data_lineage_id": bundle.data_lineage_id,
                "started_at_unix": started_at,
                "finished_at_unix": time.time(),
                "failure_code": failure_code,
            }
        )
        _secure_real_run_artifacts(
            (
                phase_root,
                run_root,
                event_path,
                config.output_root / "experiment_registry.csv",
            )
        )
        if isinstance(error, torch.OutOfMemoryError):
            raise ResourceError(
                code="CUDA_OUT_OF_MEMORY",
                message="The bounded real world-pretraining smoke exhausted CUDA memory.",
            ) from error
        raise


__all__ = [
    "DISABLED_PATHOLOGY_ARTIFACT_ID",
    "NO_OUTCOME_WORLD_PRETRAIN_CONTRACT",
    "REAL_CT_FEATURE_SCHEMA",
    "REAL_CT_FEATURE_SUMMARY_SCHEMA",
    "REAL_TIMELINE_CONTRACT_VERSION",
    "REAL_WORLD_PRETRAIN_SUMMARY_SCHEMA",
    "RealFeatureBundle",
    "extract_real_ct_features",
    "load_real_world_pretrain_batches",
    "run_real_world_pretraining_smoke",
]
