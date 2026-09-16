"""Executable synthetic prediction, evaluation, and report integration."""

from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml

import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]

from stageworld.artifacts import atomic_write_json, new_artifact_id, read_json
from stageworld.config import RunMode as ConfigRunMode
from stageworld.config import StageWorldConfig
from stageworld.data import (
    ClinicalMeasurement,
    FeatureFirewall,
    MissingCategory,
    Observation,
    ObservationRole,
    QualityStatus,
    Query,
    QueryEligibility,
    Stage,
    Treatment,
    TreatmentKind,
    TreatmentStatus,
    default_synthetic_policy,
    load_cohort_json,
)
from stageworld.encoders.base import EncoderProvenance, ObservationTokens
from stageworld.errors import ArtifactError, ConfigurationError, DataContractError
from stageworld.evaluation import (
    CalibrationBinSpec,
    CensoringDistribution,
    EvaluationArtifactWriter,
    EvaluationCohort,
    EvaluationProtocol,
    FigureLineage,
    PredictionArtifact,
    PredictionLineage,
    PredictionRecord,
    SplitRole,
    build_metric_lineage,
    calibration_curve,
    cumulative_dynamic_auc,
    integrated_brier_score,
    ipcw_brier_score,
    ipcw_concordance_index,
    read_prediction_artifact,
)
from stageworld.evaluation import (
    RunMode as EvaluationRunMode,
)
from stageworld.inference import (
    CHECKPOINT_SCHEMA_VERSION,
    FEATURE_INPUT_SCHEMA_VERSION,
    ActionFeature,
    CheckpointContract,
    FeatureInputContract,
    FeatureManifest,
    InferenceEngine,
    ModalityFeatureContract,
    PredictionOutput,
)
from stageworld.model import StageWorldModel
from stageworld.synthetic_workflow import (
    BUILD_SCHEMA,
    FEATURE_SCHEMA,
    SOURCE_SCHEMA,
    _load_tensor_artifact,
    _split_version,
    _validate_checkpoint_for_run,
    build_model,
    synthetic_data_root,
    synthetic_feature_root,
    validate_synthetic_feature_artifacts,
    validate_synthetic_source_config,
)
from stageworld.training import (
    TrainingPhase,
    checkpoint_payload_mismatches,
    checkpoint_snapshot_path,
    model_state_mismatches,
)

EVALUATION_SUMMARY_SCHEMA_VERSION = "stageworld-evaluation-summary-v3"
REPORT_MANIFEST_SCHEMA_VERSION = "stageworld-report-manifest-v3"
TRAINING_SUMMARY_SCHEMA_VERSION = "stageworld-training-summary-v3"


def _require_synthetic(config: StageWorldConfig, command: str) -> None:
    if config.mode is not ConfigRunMode.SYNTHETIC:
        raise ConfigurationError(
            code="SYNTHETIC_PIPELINE_MODE_REQUIRED",
            message=f"The current {command} integration is validated only for synthetic mode.",
        )


def _atomic_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _row_tokens(
    source: ObservationTokens,
    row: int,
    *,
    record_id: str,
    acquired: float,
    available: float,
    token_indices: tuple[int, ...] | None = None,
) -> ObservationTokens:
    indices = tuple(range(source.token_count)) if token_indices is None else token_indices
    if not indices:
        raise DataContractError(
            code="EMPTY_RECORD_TOKEN_SELECTION",
            message="A model-facing observation record must select at least one token.",
        )
    selection = torch.tensor(indices, dtype=torch.long, device=source.values.device)
    valid = source.valid[row : row + 1, selection]
    source_ids = (
        tuple(
            f"{record_id}-token-{index}" if bool(valid[0, index]) else ""
            for index in range(valid.shape[1])
        ),
    )
    return ObservationTokens(
        values=source.values[row : row + 1, selection].detach(),
        valid=valid.detach(),
        modality=source.modality[row : row + 1, selection].detach(),
        acquired_time=torch.full_like(source.acquired_time[row : row + 1, selection], acquired),
        available_time=torch.full_like(
            source.available_time[row : row + 1, selection], available
        ),
        provenance=source.provenance,
        source_id=source_ids,
        modality_name=source.modality_name,
        coords=(
            None
            if source.coords is None
            else source.coords[row : row + 1, selection].detach()
        ),
        coordinate_system=source.coordinate_system,
        quality_flags=(source.quality_flags[row],) if source.quality_flags else (),
    )


def _clinical_provenance(config: StageWorldConfig) -> EncoderProvenance:
    return EncoderProvenance(
        encoder_name="synthetic_structured_clinical",
        source_version="synthetic-v1",
        component_versions=(("field_tokenizer", "analytic-v1"),),
        preprocess_version="identity-v1",
        feature_dim=config.model.clinical_input_dim,
    )


def _feature_artifact_lineage(
    config: StageWorldConfig,
    source: Mapping[str, Any],
    ct_payload: Mapping[str, Any],
    pathology_payload: Mapping[str, Any],
) -> tuple[str, str]:
    expected = {
        "data_lineage_id": source.get("data_lineage_id"),
        "cohort_artifact_id": source.get("cohort_artifact_id"),
        "timeline_contract": source.get("timeline_contract"),
        "timeline_contract_version": source.get("timeline_contract_version"),
        "outcome_contract": source.get("outcome_contract"),
        "outcome_contract_version": source.get("outcome_contract_version"),
    }
    mismatched = sorted(
        f"{modality}.{key}"
        for modality, payload in (("ct", ct_payload), ("pathology", pathology_payload))
        for key, value in expected.items()
        if payload.get(key) != value
    )
    if ct_payload.get("modality") != "ct":
        mismatched.append("ct.modality")
    if pathology_payload.get("modality") != "pathology":
        mismatched.append("pathology.modality")
    if mismatched:
        raise ArtifactError(
            code="FEATURE_DATA_LINEAGE_MISMATCH",
            message=(
                "CT, pathology, and source artifacts do not share one data, cohort, "
                "timeline, and outcome lineage."
            ),
            details={"fields": sorted(set(mismatched))},
        )
    artifact_ids: list[str] = []
    for modality, payload in (("ct", ct_payload), ("pathology", pathology_payload)):
        artifact_id = payload.get("feature_artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise ArtifactError(
                code="FEATURE_ARTIFACT_ID_MISSING",
                message="Every feature tensor artifact requires its own immutable artifact ID.",
                details={"modality": modality},
            )
        artifact_ids.append(artifact_id)
    if artifact_ids[0] == artifact_ids[1]:
        raise ArtifactError(
            code="FEATURE_ARTIFACT_ID_COLLISION",
            message="CT and pathology feature tensor artifacts require distinct lineage IDs.",
        )
    validated_ids = validate_synthetic_feature_artifacts(
        config,
        source,
        ct_payload,
        pathology_payload,
    )
    if validated_ids != (artifact_ids[0], artifact_ids[1]):
        raise ArtifactError(
            code="FEATURE_ARTIFACT_ID_MISMATCH",
            message="Feature lineage IDs changed during synthetic contract validation.",
        )
    return validated_ids


def _clinical_token(
    values: torch.Tensor,
    row: int,
    token: int,
    *,
    record_id: str,
    acquired: float,
    available: float,
    provenance: EncoderProvenance,
) -> ObservationTokens:
    selected = values[row : row + 1, token : token + 1].detach()
    return ObservationTokens(
        values=selected,
        valid=torch.ones(1, 1, dtype=torch.bool),
        modality=torch.full((1, 1), 2, dtype=torch.long),
        acquired_time=torch.full((1, 1), acquired),
        available_time=torch.full((1, 1), available),
        provenance=provenance,
        source_id=((f"{record_id}-token",),),
        modality_name="clinical",
    )


def _load_joint_checkpoint(
    config: StageWorldConfig,
    data_lineage_id: str,
    *,
    cohort_artifact_id: str,
    ct_feature_artifact_id: str,
    pathology_feature_artifact_id: str,
    timeline_contract_version: str,
    outcome_contract_version: str,
    checkpoint_path: Path | None = None,
    split_version: str | None = None,
    require_snapshot_match: bool = True,
) -> tuple[StageWorldModel, CheckpointContract]:
    path = checkpoint_path or config.output_root / "runs" / "joint_survival" / "checkpoint.pt"
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError) as error:
        raise ArtifactError(
            code="JOINT_CHECKPOINT_REQUIRED",
            message="Prediction requires a complete joint_survival checkpoint.",
            remediation="Run both bounded synthetic training phases first.",
        ) from error
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
    ):
        raise ArtifactError(
            code="CHECKPOINT_SCHEMA_MISMATCH",
            message="Joint checkpoint schema is unsupported.",
        )
    model = build_model(config)
    contract = CheckpointContract.from_checkpoint_payload(payload)
    if require_snapshot_match:
        exact_snapshot_path = checkpoint_snapshot_path(path, contract.weight_version)
        if not exact_snapshot_path.is_file():
            raise ArtifactError(
                code="CHECKPOINT_SNAPSHOT_MISSING",
                message="Checkpoint pointer has no exact immutable snapshot.",
                details={"weight_version": contract.weight_version},
            )
        exact_payload = _load_tensor_artifact(
            exact_snapshot_path,
            CHECKPOINT_SCHEMA_VERSION,
        )
        pointer_differences = checkpoint_payload_mismatches(payload, exact_payload)
        if pointer_differences:
            raise ArtifactError(
                code="CHECKPOINT_POINTER_SNAPSHOT_MISMATCH",
                message="Checkpoint pointer differs from its exact immutable snapshot.",
                details={"fields": list(pointer_differences)},
            )
    _validate_checkpoint_for_run(
        contract,
        config,
        model,
        data_lineage_id=data_lineage_id,
        cohort_artifact_id=cohort_artifact_id,
        split_version=(
            split_version
            if split_version is not None
            else _split_version(config, data_lineage_id=data_lineage_id)
        ),
        ct_feature_artifact_id=ct_feature_artifact_id,
        pathology_feature_artifact_id=pathology_feature_artifact_id,
        timeline_contract_version=timeline_contract_version,
        outcome_contract_version=outcome_contract_version,
        phase=TrainingPhase.JOINT_SURVIVAL,
    )
    try:
        model.load_state_dict(payload["model_state"], strict=True)
    except (KeyError, RuntimeError) as error:
        raise ArtifactError(
            code="CHECKPOINT_STATE_INCOMPATIBLE",
            message="Checkpoint tensors do not match the configured model.",
        ) from error
    model.eval()
    return model, contract


def _checkpoint_lineage_fields(
    checkpoint: CheckpointContract,
) -> dict[str, str | int | None]:
    return {
        "checkpoint_schema_version": checkpoint.schema_version,
        "checkpoint_id": checkpoint.checkpoint_id,
        "weight_version": checkpoint.weight_version,
        "model_version": checkpoint.model_version,
        "checkpoint_endpoint": checkpoint.endpoint,
        "config_lineage_id": checkpoint.config_lineage_id,
        "data_lineage_id": checkpoint.data_lineage_id,
        "cohort_artifact_id": checkpoint.cohort_artifact_id,
        "split_version": checkpoint.split_version,
        "ct_feature_artifact_id": checkpoint.ct_feature_artifact_id,
        "pathology_feature_artifact_id": checkpoint.pathology_feature_artifact_id,
        "timeline_contract_version": checkpoint.timeline_contract_version,
        "outcome_contract_version": checkpoint.outcome_contract_version,
        "training_seed": checkpoint.training_seed,
        "source_schema_version": checkpoint.source_schema_version,
        "cohort_schema_version": checkpoint.cohort_schema_version,
        "feature_schema_version": checkpoint.feature_schema_version,
        "checkpoint_phase": checkpoint.phase,
        "checkpoint_step": checkpoint.step,
        "parent_checkpoint_id": checkpoint.parent_checkpoint_id,
        "parent_weight_version": checkpoint.parent_weight_version,
        "parent_phase": checkpoint.parent_phase,
        "parent_config_lineage_id": checkpoint.parent_config_lineage_id,
        "parent_data_lineage_id": checkpoint.parent_data_lineage_id,
        "parent_cohort_artifact_id": checkpoint.parent_cohort_artifact_id,
        "parent_split_version": checkpoint.parent_split_version,
        "parent_ct_feature_artifact_id": checkpoint.parent_ct_feature_artifact_id,
        "parent_pathology_feature_artifact_id": checkpoint.parent_pathology_feature_artifact_id,
        "parent_timeline_contract_version": checkpoint.parent_timeline_contract_version,
        "parent_outcome_contract_version": checkpoint.parent_outcome_contract_version,
    }


def _validate_prediction_checkpoint_lineage(
    artifact: PredictionArtifact, checkpoint: CheckpointContract
) -> None:
    expected_lineage = {
        **_checkpoint_lineage_fields(checkpoint),
        "model_artifact_id": checkpoint.weight_version,
        "config_version": checkpoint.config_lineage_id,
        "input_artifact_id": checkpoint.data_lineage_id,
        "run_mode": EvaluationRunMode(checkpoint.mode.value),
    }
    mismatched_lineage = {
        key: {"expected": expected, "actual": getattr(artifact.lineage, key)}
        for key, expected in expected_lineage.items()
        if getattr(artifact.lineage, key) != expected
    }
    if mismatched_lineage:
        raise ArtifactError(
            code="PREDICTION_CHECKPOINT_LINEAGE_MISMATCH",
            message="Prediction artifact is not derived from the validated checkpoint contract.",
            details={"fields": mismatched_lineage},
        )
    if any(
        record.seed != checkpoint.training_seed
        or record.endpoint.lower() != checkpoint.endpoint
        or record.input_artifact_id != checkpoint.data_lineage_id
        for record in artifact.records
    ):
        raise ArtifactError(
            code="PREDICTION_RECORD_LINEAGE_MISMATCH",
            message="Prediction records disagree with their checkpoint/data lineage.",
        )


def _load_exact_report_checkpoint(
    config: StageWorldConfig,
    *,
    root: Path,
    data_lineage_id: str,
    cohort_artifact_id: str,
    split_version: str,
    ct_feature_artifact_id: str,
    pathology_feature_artifact_id: str,
    timeline_contract_version: str,
    outcome_contract_version: str,
) -> tuple[CheckpointContract, Path, CheckpointContract, Path]:
    pointer_path = root / "runs" / "joint_survival" / "checkpoint.pt"
    pointer_model, pointer_contract = _load_joint_checkpoint(
        config,
        data_lineage_id,
        cohort_artifact_id=cohort_artifact_id,
        ct_feature_artifact_id=ct_feature_artifact_id,
        pathology_feature_artifact_id=pathology_feature_artifact_id,
        timeline_contract_version=timeline_contract_version,
        outcome_contract_version=outcome_contract_version,
        checkpoint_path=pointer_path,
        split_version=split_version,
    )
    snapshot_path = checkpoint_snapshot_path(pointer_path, pointer_contract.weight_version)
    if not snapshot_path.is_file():
        raise ArtifactError(
            code="CHECKPOINT_SNAPSHOT_MISSING",
            message="Report requires the append-only snapshot for the current weight version.",
            details={"weight_version": pointer_contract.weight_version},
        )
    snapshot_model, snapshot_contract = _load_joint_checkpoint(
        config,
        data_lineage_id,
        cohort_artifact_id=cohort_artifact_id,
        ct_feature_artifact_id=ct_feature_artifact_id,
        pathology_feature_artifact_id=pathology_feature_artifact_id,
        timeline_contract_version=timeline_contract_version,
        outcome_contract_version=outcome_contract_version,
        checkpoint_path=snapshot_path,
        split_version=split_version,
        require_snapshot_match=False,
    )
    if snapshot_contract != pointer_contract:
        raise ArtifactError(
            code="CHECKPOINT_SNAPSHOT_CONTRACT_MISMATCH",
            message="Checkpoint pointer and exact weight snapshot have different contracts.",
        )
    pointer_state = pointer_model.state_dict()
    snapshot_state = snapshot_model.state_dict()
    changed_tensors = sorted(
        name
        for name in set(pointer_state) | set(snapshot_state)
        if name not in pointer_state
        or name not in snapshot_state
        or not torch.equal(pointer_state[name], snapshot_state[name])
    )
    if changed_tensors:
        raise ArtifactError(
            code="CHECKPOINT_POINTER_SNAPSHOT_MISMATCH",
            message="Checkpoint pointer tensors differ from its exact weight snapshot.",
            details={"tensors": changed_tensors},
        )

    parent_weight_version = pointer_contract.parent_weight_version
    parent_ct_feature_artifact_id = pointer_contract.parent_ct_feature_artifact_id
    parent_pathology_feature_artifact_id = pointer_contract.parent_pathology_feature_artifact_id
    if (
        parent_weight_version is None
        or parent_ct_feature_artifact_id is None
        or parent_pathology_feature_artifact_id is None
    ):
        raise ArtifactError(
            code="PARENT_CHECKPOINT_LINEAGE_REQUIRED",
            message=(
                "Joint report checkpoint does not identify an exact world-pretraining "
                "parent and its feature artifacts."
            ),
        )
    parent_pointer_path = root / "runs" / "world_pretrain" / "checkpoint.pt"
    parent_snapshot_path = checkpoint_snapshot_path(parent_pointer_path, parent_weight_version)
    if not parent_snapshot_path.is_file():
        raise ArtifactError(
            code="PARENT_CHECKPOINT_SNAPSHOT_MISSING",
            message="Report requires the exact world-pretraining parent snapshot.",
            details={"parent_weight_version": parent_weight_version},
        )
    parent_payload = _load_tensor_artifact(parent_snapshot_path, CHECKPOINT_SCHEMA_VERSION)
    parent_contract = CheckpointContract.from_checkpoint_payload(parent_payload)
    parent_model = build_model(config)
    _validate_checkpoint_for_run(
        parent_contract,
        config,
        parent_model,
        data_lineage_id=data_lineage_id,
        cohort_artifact_id=cohort_artifact_id,
        split_version=split_version,
        ct_feature_artifact_id=parent_ct_feature_artifact_id,
        pathology_feature_artifact_id=parent_pathology_feature_artifact_id,
        timeline_contract_version=timeline_contract_version,
        outcome_contract_version=outcome_contract_version,
        phase=TrainingPhase.WORLD_PRETRAIN,
    )
    try:
        parent_model.load_state_dict(parent_payload["model_state"], strict=True)
    except (KeyError, RuntimeError) as error:
        raise ArtifactError(
            code="PARENT_CHECKPOINT_STATE_INCOMPATIBLE",
            message="World-pretraining parent snapshot tensors do not match the configured model.",
        ) from error
    expected_parent_lineage = {
        "checkpoint_id": pointer_contract.parent_checkpoint_id,
        "weight_version": pointer_contract.parent_weight_version,
        "phase": pointer_contract.parent_phase,
        "config_lineage_id": pointer_contract.parent_config_lineage_id,
        "data_lineage_id": pointer_contract.parent_data_lineage_id,
        "cohort_artifact_id": pointer_contract.parent_cohort_artifact_id,
        "split_version": pointer_contract.parent_split_version,
        "ct_feature_artifact_id": pointer_contract.parent_ct_feature_artifact_id,
        "pathology_feature_artifact_id": pointer_contract.parent_pathology_feature_artifact_id,
        "timeline_contract_version": pointer_contract.parent_timeline_contract_version,
        "outcome_contract_version": pointer_contract.parent_outcome_contract_version,
    }
    mismatched_parent_lineage = sorted(
        key
        for key, expected in expected_parent_lineage.items()
        if getattr(parent_contract, key) != expected
    )
    if mismatched_parent_lineage:
        raise ArtifactError(
            code="PARENT_CHECKPOINT_LINEAGE_MISMATCH",
            message="Joint checkpoint parent lineage differs from its exact snapshot.",
            details={"fields": mismatched_parent_lineage},
        )
    joint_payload = _load_tensor_artifact(snapshot_path, CHECKPOINT_SCHEMA_VERSION)
    embedded_parent_state = joint_payload.get("transfer_parent_model_state")
    exact_parent_state = parent_payload.get("model_state")
    if not isinstance(embedded_parent_state, Mapping) or not isinstance(
        exact_parent_state, Mapping
    ):
        raise ArtifactError(
            code="PARENT_CHECKPOINT_STATE_REQUIRED",
            message="Joint checkpoint cannot prove its transferred parent model state.",
        )
    changed_parent_tensors = model_state_mismatches(
        embedded_parent_state,
        exact_parent_state,
    )
    if changed_parent_tensors:
        raise ArtifactError(
            code="PARENT_CHECKPOINT_STATE_MISMATCH",
            message="Joint checkpoint embedded parent state differs from its exact snapshot.",
            details={"tensors": list(changed_parent_tensors)},
        )
    return pointer_contract, snapshot_path, parent_contract, parent_snapshot_path


def _query_file(path: Path, known_patients: set[str]) -> tuple[Query, ...]:
    payload = read_json(path)
    if payload.get("schema_version") != "stageworld-query-v1":
        raise DataContractError(
            code="QUERY_SCHEMA_MISMATCH",
            message="Query file schema is absent or unsupported.",
        )
    raw_queries = payload.get("queries")
    if not isinstance(raw_queries, list) or not raw_queries:
        raise DataContractError(
            code="EMPTY_QUERY_HISTORY", message="Query file must contain a nonempty list."
        )
    queries: list[Query] = []
    for raw in raw_queries:
        if not isinstance(raw, Mapping):
            raise DataContractError(
                code="INVALID_QUERY_RECORD", message="Each query must be an object."
            )
        patient_id = str(raw.get("patient_id", ""))
        if patient_id not in known_patients:
            raise DataContractError(
                code="QUERY_PATIENT_NOT_FOUND",
                message="Query references an unknown anonymous patient.",
            )
        try:
            stage = Stage(str(raw["stage"]).lower())
            query = Query(
                query_id=str(raw["query_id"]),
                patient_id=patient_id,
                stage=stage,
                query_time_days=float(raw["query_time_days"]),
                prediction_horizon_days=float(raw["prediction_horizon_days"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise DataContractError(
                code="INVALID_QUERY_RECORD", message="Query fields are incomplete or invalid."
            ) from error
        queries.append(query)
    return tuple(queries)


def _runtime_context(
    config: StageWorldConfig,
) -> tuple[InferenceEngine, FeatureFirewall, dict[str, FeatureManifest], tuple[Query, ...], str]:
    source = _load_tensor_artifact(synthetic_data_root(config) / "source_tensors.pt", SOURCE_SCHEMA)
    validate_synthetic_source_config(config, source)
    ct_payload = _load_tensor_artifact(synthetic_feature_root(config) / "ct.pt", FEATURE_SCHEMA)
    path_payload = _load_tensor_artifact(
        synthetic_feature_root(config) / "pathology.pt", FEATURE_SCHEMA
    )
    data_lineage = str(source["data_lineage_id"])
    ct_feature_artifact_id, pathology_feature_artifact_id = _feature_artifact_lineage(
        config, source, ct_payload, path_payload
    )
    cohort = load_cohort_json(synthetic_data_root(config) / "cohort.json")
    ct0 = ObservationTokens.from_cache_payload(ct_payload["ct0"])
    ct1 = ObservationTokens.from_cache_payload(ct_payload["ct1"])
    pathology = ObservationTokens.from_cache_payload(path_payload["pathology"])
    patient_ids = tuple(str(value) for value in source["patient_ids"])
    row_by_patient = {patient_id: index for index, patient_id in enumerate(patient_ids)}
    if set(row_by_patient) != {patient.patient_id for patient in cohort.patients}:
        raise ArtifactError(
            code="COHORT_FEATURE_PATIENT_MISMATCH",
            message="Synthetic cohort and feature patient sets differ.",
        )

    adjusted_observations: list[Observation] = []
    for observation in cohort.observations:
        row = row_by_patient[observation.patient_id]
        if observation.role is ObservationRole.POST_TREATMENT_CT and not bool(ct1.valid[row].any()):
            adjusted_observations.append(
                replace(
                    observation,
                    local_asset_id=None,
                    quality_status=QualityStatus.UNKNOWN,
                    missing_reason=MissingCategory.MISSING,
                )
            )
        else:
            adjusted_observations.append(observation)
    firewall = FeatureFirewall(
        cohort.patients,
        adjusted_observations,
        cohort.clinical_measurements,
        cohort.treatments,
        default_synthetic_policy(),
    )
    model, checkpoint = _load_joint_checkpoint(
        config,
        data_lineage,
        cohort_artifact_id=str(source["cohort_artifact_id"]),
        ct_feature_artifact_id=ct_feature_artifact_id,
        pathology_feature_artifact_id=pathology_feature_artifact_id,
        timeline_contract_version=str(source["timeline_contract_version"]),
        outcome_contract_version=str(source["outcome_contract_version"]),
    )
    clinical_provenance = _clinical_provenance(config)
    feature_contract = FeatureInputContract(
        schema_version=FEATURE_INPUT_SCHEMA_VERSION,
        modalities=(
            ModalityFeatureContract("ct", ct0.provenance, "synthetic-v1"),
            ModalityFeatureContract("pathology", pathology.provenance, "synthetic-v1"),
            ModalityFeatureContract("clinical", clinical_provenance),
        ),
        action_feature_version="synthetic-action-v1",
    )
    engine = InferenceEngine(
        model,
        checkpoint,
        feature_contract,
        survival_time_unit=config.survival.time_unit,
        survival_parameterization=config.survival.parameterization,
        survival_open_tail_interval=config.survival.open_tail_interval,
    )
    clinical_values = torch.as_tensor(source["clinical"]).float()
    treatment_values = torch.as_tensor(source["treatment_actions"]).float()
    treatment_valid = torch.as_tensor(source["treatment_valid"]).bool()
    surgery_values = torch.as_tensor(source["surgery_actions"]).float()
    surgery_valid = torch.as_tensor(source["surgery_valid"]).bool()
    clinical_field_tokens = {
        "age_years": 0,
        "baseline_stage": 1,
        "radiologic_response": 2,
        "yp_stage": 3,
    }
    if clinical_values.shape[1] != len(clinical_field_tokens):
        raise ArtifactError(
            code="SYNTHETIC_CLINICAL_SCHEMA_MISMATCH",
            message="Synthetic clinical source does not match the field-token contract.",
        )
    observations_by_patient: dict[str, list[Observation]] = {
        patient_id: [] for patient_id in patient_ids
    }
    for observation in adjusted_observations:
        observations_by_patient[observation.patient_id].append(observation)
    measurements_by_patient: dict[str, list[ClinicalMeasurement]] = {
        patient_id: [] for patient_id in patient_ids
    }
    for measurement in cohort.clinical_measurements:
        measurements_by_patient[measurement.patient_id].append(measurement)
    treatments_by_patient: dict[str, list[Treatment]] = {
        patient_id: [] for patient_id in patient_ids
    }
    for treatment in cohort.treatments:
        treatments_by_patient[treatment.patient_id].append(treatment)

    manifests: dict[str, FeatureManifest] = {}
    for patient_id, row in row_by_patient.items():
        observation_features: dict[str, ObservationTokens] = {}
        usable_pathology = sorted(
            (
                observation
                for observation in observations_by_patient[patient_id]
                if observation.role is ObservationRole.SURGICAL_PATHOLOGY
                and observation.local_asset_id is not None
                and observation.missing_reason is None
                and observation.quality_status is not QualityStatus.FAILED
                and observation.available_at_days is not None
            ),
            key=lambda observation: observation.observation_id,
        )
        pathology_token_indices: dict[str, tuple[int, ...]] = {}
        valid_pathology_tokens = [
            index for index in range(pathology.token_count) if bool(pathology.valid[row, index])
        ]
        if usable_pathology and len(valid_pathology_tokens) < len(usable_pathology):
            raise ArtifactError(
                code="SYNTHETIC_PATHOLOGY_TOKEN_RECORD_MISMATCH",
                message=(
                    "Each usable synthetic slide record requires at least one valid "
                    "patient-level pathology token."
                ),
            )
        start = 0
        for index, observation in enumerate(usable_pathology):
            remaining_records = len(usable_pathology) - index
            remaining_tokens = len(valid_pathology_tokens) - start
            width = max(1, (remaining_tokens + remaining_records - 1) // remaining_records)
            stop = min(len(valid_pathology_tokens), start + width)
            pathology_token_indices[observation.observation_id] = tuple(
                valid_pathology_tokens[start:stop]
            )
            start = stop
        for observation in observations_by_patient[patient_id]:
            if (
                observation.local_asset_id is None
                or observation.missing_reason is not None
                or observation.quality_status is QualityStatus.FAILED
                or observation.available_at_days is None
            ):
                continue
            if observation.role is ObservationRole.BASELINE_CT:
                source_tokens = ct0
            elif observation.role is ObservationRole.POST_TREATMENT_CT:
                source_tokens = ct1
            elif observation.role is ObservationRole.SURGICAL_PATHOLOGY:
                source_tokens = pathology
            else:
                continue
            observation_features[observation.observation_id] = _row_tokens(
                source_tokens,
                row,
                record_id=observation.observation_id,
                acquired=observation.acquired_at_days,
                available=observation.available_at_days,
                token_indices=pathology_token_indices.get(observation.observation_id),
            )
        clinical_features: dict[str, ObservationTokens] = {}
        for measurement in measurements_by_patient[patient_id]:
            if (
                measurement.available_at_days is None
                or measurement.field_name == "followup_duration"
            ):
                continue
            try:
                clinical_token = clinical_field_tokens[measurement.field_name]
            except KeyError as error:
                raise ArtifactError(
                    code="SYNTHETIC_CLINICAL_FIELD_UNMAPPED",
                    message="A permitted synthetic clinical field has no token mapping.",
                    details={"field": measurement.field_name},
                ) from error
            clinical_features[measurement.measurement_id] = _clinical_token(
                clinical_values,
                row,
                clinical_token,
                record_id=measurement.measurement_id,
                acquired=measurement.acquired_at_days,
                available=measurement.available_at_days,
                provenance=clinical_provenance,
            )
        action_features: dict[str, ActionFeature] = {}
        for treatment in treatments_by_patient[patient_id]:
            if treatment.planned_or_delivered is not TreatmentStatus.DELIVERED:
                continue
            if treatment.treatment_kind is TreatmentKind.SURGERY:
                if not bool(surgery_valid[row, 0]):
                    raise ArtifactError(
                        code="SYNTHETIC_ACTION_VALIDITY_MISMATCH",
                        message="Synthetic surgery record points to an invalid action feature.",
                    )
                values = surgery_values[row, 0]
            elif treatment.event_id.endswith("-post-ct-treatment"):
                if not bool(treatment_valid[row, 1]):
                    raise ArtifactError(
                        code="SYNTHETIC_ACTION_VALIDITY_MISMATCH",
                        message="Synthetic late treatment points to an invalid action feature.",
                    )
                values = treatment_values[row, 1]
            else:
                if not bool(treatment_valid[row, 0]):
                    raise ArtifactError(
                        code="SYNTHETIC_ACTION_VALIDITY_MISMATCH",
                        message="Synthetic systemic treatment points to an invalid action feature.",
                    )
                values = treatment_values[row, 0]
            action_features[treatment.event_id] = ActionFeature(
                values.detach(), "synthetic-action-v1", 1.0
            )
        manifests[patient_id] = FeatureManifest(
            schema_version=FEATURE_INPUT_SCHEMA_VERSION,
            manifest_lineage_id=f"{data_lineage}-{patient_id.lower()}",
            patient_id=patient_id,
            mode=ConfigRunMode.SYNTHETIC,
            observation_features=observation_features,
            clinical_features=clinical_features,
            action_features=action_features,
        )
    max_horizon = max(config.survival.report_horizons_years) * 365.25
    standard_queries = tuple(
        replace(query, prediction_horizon_days=max_horizon)
        for query in cohort.queries
        if query.eligibility is QueryEligibility.ELIGIBLE
        and query.query_id
        in {
            f"{query.patient_id}-s0",
            f"{query.patient_id}-s1",
            f"{query.patient_id}-s2",
        }
    )
    return engine, firewall, manifests, standard_queries, data_lineage


def run_synthetic_prediction(
    config: StageWorldConfig,
    *,
    query_file: Path | None = None,
) -> dict[str, Any]:
    _require_synthetic(config, "predict")
    engine, firewall, manifests, default_queries, data_lineage = _runtime_context(config)
    queries = _query_file(query_file, set(manifests)) if query_file is not None else default_queries
    by_patient: dict[str, list[Query]] = {}
    for query in queries:
        by_patient.setdefault(query.patient_id, []).append(query)
    horizon_days = tuple(value * 365.25 for value in config.survival.report_horizons_years)
    outputs: list[PredictionOutput] = []
    for patient_id, patient_queries in by_patient.items():
        outputs.extend(
            engine.predict_patient_history(
                firewall,
                manifests[patient_id],
                patient_queries,
                endpoint="os",
                horizons_days=horizon_days,
            )
        )
    if not outputs:
        raise DataContractError(
            code="EMPTY_PREDICTION_OUTPUT", message="No eligible query produced a prediction."
        )
    checkpoint = engine.checkpoint
    if any(
        output.checkpoint_id != checkpoint.checkpoint_id
        or output.weight_version != checkpoint.weight_version
        or output.model_version != checkpoint.model_version
        for output in outputs
    ):
        raise ArtifactError(
            code="PREDICTION_CHECKPOINT_LINEAGE_MISMATCH",
            message="Prediction outputs do not share the validated checkpoint lineage.",
        )
    output_root = config.output_root / "inference"
    history_path = output_root / "history_predictions.json"
    atomic_write_json(
        history_path,
        {
            "schema_version": "stageworld-history-predictions-v1",
            "mode": "synthetic",
            "clinical_validation": False,
            **_checkpoint_lineage_fields(checkpoint),
            "predictions": [output.as_dict() for output in outputs],
        },
    )
    first_patient = next(iter(by_patient))
    demo = [output.as_dict() for output in outputs if output.patient_id == first_patient]
    demo_path = output_root / "three_node_demo.json"
    atomic_write_json(
        demo_path,
        {
            "schema_version": "stageworld-three-node-demo-v1",
            "mode": "synthetic",
            "patient_id": first_patient,
            **_checkpoint_lineage_fields(checkpoint),
            "risk_may_increase_or_decrease": True,
            "predictions": demo,
        },
    )
    artifact_id = new_artifact_id("synthetic-predictions")
    lineage = PredictionLineage(
        artifact_id=artifact_id,
        checkpoint_schema_version=checkpoint.schema_version,
        checkpoint_id=checkpoint.checkpoint_id,
        weight_version=checkpoint.weight_version,
        model_artifact_id=checkpoint.weight_version,
        model_version=checkpoint.model_version,
        checkpoint_endpoint=checkpoint.endpoint,
        config_lineage_id=checkpoint.config_lineage_id,
        config_version=checkpoint.config_lineage_id,
        data_lineage_id=checkpoint.data_lineage_id,
        input_artifact_id=checkpoint.data_lineage_id,
        cohort_artifact_id=checkpoint.cohort_artifact_id,
        split_version=checkpoint.split_version,
        ct_feature_artifact_id=checkpoint.ct_feature_artifact_id,
        pathology_feature_artifact_id=checkpoint.pathology_feature_artifact_id,
        timeline_contract_version=checkpoint.timeline_contract_version,
        outcome_contract_version=checkpoint.outcome_contract_version,
        training_seed=checkpoint.training_seed,
        source_schema_version=checkpoint.source_schema_version,
        cohort_schema_version=checkpoint.cohort_schema_version,
        feature_schema_version=checkpoint.feature_schema_version,
        checkpoint_phase=checkpoint.phase,
        checkpoint_step=checkpoint.step,
        parent_checkpoint_id=checkpoint.parent_checkpoint_id,
        parent_weight_version=checkpoint.parent_weight_version,
        parent_phase=checkpoint.parent_phase,
        parent_config_lineage_id=checkpoint.parent_config_lineage_id,
        parent_data_lineage_id=checkpoint.parent_data_lineage_id,
        parent_cohort_artifact_id=checkpoint.parent_cohort_artifact_id,
        parent_split_version=checkpoint.parent_split_version,
        parent_ct_feature_artifact_id=checkpoint.parent_ct_feature_artifact_id,
        parent_pathology_feature_artifact_id=checkpoint.parent_pathology_feature_artifact_id,
        parent_timeline_contract_version=checkpoint.parent_timeline_contract_version,
        parent_outcome_contract_version=checkpoint.parent_outcome_contract_version,
        run_mode=EvaluationRunMode.SYNTHETIC,
    )
    records = tuple(
        PredictionRecord(
            patient_id=output.patient_id,
            stage=output.stage,
            query_time=output.query_time_days,
            endpoint=output.endpoint,
            horizon=horizon,
            risk=risk,
            fold=None,
            seed=checkpoint.training_seed,
            model_name="stageworld-gc",
            input_artifact_id=checkpoint.data_lineage_id,
            state_quality=(
                ",".join(output.quality_flags) if output.quality_flags else "observed_prefix"
            ),
        )
        for output in outputs
        for horizon, risk in zip(output.horizon_days[1:], output.risk[1:], strict=True)
    )
    artifact = PredictionArtifact(lineage=lineage, records=records)
    writer = EvaluationArtifactWriter(
        config.output_root / "evaluation", EvaluationRunMode.SYNTHETIC
    )
    evaluation_path = writer.write_predictions(artifact)
    index_path = output_root / "prediction_index.json"
    atomic_write_json(
        index_path,
        {
            "schema_version": "stageworld-prediction-index-v1",
            "mode": "synthetic",
            "prediction_artifact_id": artifact_id,
            **_checkpoint_lineage_fields(checkpoint),
            "evaluation_predictions": str(evaluation_path),
            "history_predictions": str(history_path),
            "three_node_demo": str(demo_path),
        },
    )
    return {
        "status": "ok",
        "mode": "synthetic",
        "clinical_validation": False,
        "patients": len(by_patient),
        "queries": len(outputs),
        "records": len(records),
        **_checkpoint_lineage_fields(checkpoint),
        "history_predictions": str(history_path),
        "evaluation_predictions": str(evaluation_path),
        "three_node_demo": str(demo_path),
        "index": str(index_path),
    }


def _load_protocol(config: StageWorldConfig, path: Path | None) -> tuple[str, tuple[float, ...]]:
    if path is None:
        return (
            f"synthetic-os-seed-{config.training.seed}-v1",
            tuple(value * 365.25 for value in config.survival.report_horizons_years),
        )
    try:
        payload = _release_yaml(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise DataContractError(
            code="EVALUATION_PROTOCOL_UNREADABLE",
            message="Evaluation protocol YAML cannot be read.",
        ) from error
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != "stageworld-evaluation-protocol-v1"
    ):
        raise DataContractError(
            code="EVALUATION_PROTOCOL_SCHEMA_MISMATCH",
            message="Evaluation protocol schema is absent or unsupported.",
        )
    if str(payload.get("endpoint", "")).lower() != "os":
        raise DataContractError(
            code="ENDPOINT_NOT_CONFIGURED", message="Only OS is enabled in this protocol."
        )
    raw_horizons = payload.get("horizons_days")
    if not isinstance(raw_horizons, list):
        raise DataContractError(
            code="INVALID_EVALUATION_HORIZONS",
            message="Protocol horizons_days must be a list.",
        )
    horizons = tuple(float(value) for value in raw_horizons)
    if not horizons or any(value <= 0 for value in horizons):
        raise DataContractError(
            code="INVALID_EVALUATION_HORIZONS",
            message="Protocol horizons must be positive.",
        )
    return str(payload.get("protocol_id", "")), horizons


def _prediction_risks(
    artifact: PredictionArtifact,
    patient_ids: Sequence[str],
    stage: str,
    horizon: float,
) -> tuple[float, ...]:
    lookup = {
        (record.patient_id, record.stage, record.horizon): record.risk
        for record in artifact.records
    }
    try:
        return tuple(lookup[(patient_id, stage, horizon)] for patient_id in patient_ids)
    except KeyError as error:
        raise ArtifactError(
            code="PREDICTION_PROTOCOL_MISMATCH",
            message="Prediction artifact does not cover the locked cohort/stage/horizon.",
        ) from error


def _make_evaluation_cohort(
    *,
    ids: tuple[str, ...],
    split: str,
    role: SplitRole,
    stage: str,
    stage_index: int,
    durations: np.ndarray,
    events: np.ndarray,
    row_by_patient: Mapping[str, int],
) -> EvaluationCohort:
    return EvaluationCohort(
        cohort_id=f"synthetic-{split}-{stage}-v1",
        split_role=role,
        endpoint="os",
        patient_ids=ids,
        times=tuple(
            float(durations[row_by_patient[patient_id], stage_index]) for patient_id in ids
        ),
        event_types=tuple(
            int(events[row_by_patient[patient_id], stage_index]) for patient_id in ids
        ),
    )


def _plot_calibration(
    path: Path,
    curves: Sequence[tuple[str, float, Any]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ConfigurationError(
            code="PLOTTING_DEPENDENCY_MISSING",
            message="Calibration plotting requires matplotlib.",
        ) from error
    figure, axis = plt.subplots(figsize=(6.4, 5.0))
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="ideal")
    plotted = False
    for stage, horizon, curve in curves:
        valid = [point for point in curve.points if point.status == "ok"]
        if not valid:
            continue
        axis.plot(
            [point.predicted_risk for point in valid],
            [point.observed_risk for point in valid],
            marker="o",
            label=f"{stage} / {horizon / 365.25:g} y",
        )
        plotted = True
    if not plotted:
        axis.text(0.5, 0.5, "No estimable calibration bins", ha="center", va="center")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted OS risk", ylabel="Observed OS risk")
    axis.set_title("Synthetic StageWorld-GC calibration smoke")
    axis.legend(loc="best")
    axis.grid(alpha=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def run_synthetic_evaluation(
    config: StageWorldConfig,
    *,
    predictions: Path | None = None,
    protocol_path: Path | None = None,
) -> dict[str, Any]:
    _require_synthetic(config, "evaluate")
    if predictions is None:
        index = read_json(config.output_root / "inference" / "prediction_index.json")
        predictions = Path(str(index["evaluation_predictions"]))
    artifact = read_prediction_artifact(predictions)
    if artifact.lineage.run_mode is not EvaluationRunMode.SYNTHETIC:
        raise ArtifactError(
            code="EVALUATION_MODE_MISMATCH",
            message="Synthetic evaluation cannot read real prediction artifacts.",
        )
    protocol_id, horizons = _load_protocol(config, protocol_path)
    source = _load_tensor_artifact(synthetic_data_root(config) / "source_tensors.pt", SOURCE_SCHEMA)
    validate_synthetic_source_config(config, source)
    ct_payload = _load_tensor_artifact(synthetic_feature_root(config) / "ct.pt", FEATURE_SCHEMA)
    pathology_payload = _load_tensor_artifact(
        synthetic_feature_root(config) / "pathology.pt", FEATURE_SCHEMA
    )
    ct_feature_artifact_id, pathology_feature_artifact_id = _feature_artifact_lineage(
        config, source, ct_payload, pathology_payload
    )
    data_lineage_id = str(source["data_lineage_id"])
    (
        checkpoint,
        checkpoint_snapshot,
        _parent_checkpoint,
        parent_checkpoint_snapshot,
    ) = _load_exact_report_checkpoint(
        config,
        root=config.output_root,
        data_lineage_id=data_lineage_id,
        cohort_artifact_id=str(source["cohort_artifact_id"]),
        split_version=_split_version(config, data_lineage_id=data_lineage_id),
        ct_feature_artifact_id=ct_feature_artifact_id,
        pathology_feature_artifact_id=pathology_feature_artifact_id,
        timeline_contract_version=str(source["timeline_contract_version"]),
        outcome_contract_version=str(source["outcome_contract_version"]),
    )
    _validate_prediction_checkpoint_lineage(artifact, checkpoint)
    build = read_json(synthetic_data_root(config) / "cohort_build.json")
    if build.get("schema_version") != BUILD_SCHEMA:
        raise ArtifactError(
            code="COHORT_BUILD_SCHEMA_MISMATCH",
            message="Cohort split artifact schema is incompatible.",
        )
    if build.get("data_lineage_id") != data_lineage_id:
        raise ArtifactError(
            code="COHORT_BUILD_DATA_LINEAGE_MISMATCH",
            message="Cohort split and evaluation outcomes do not share a data lineage.",
        )
    if build.get("cohort_artifact_id") != source.get("cohort_artifact_id"):
        raise ArtifactError(
            code="COHORT_BUILD_COHORT_LINEAGE_MISMATCH",
            message="Cohort split and evaluation source do not share a cohort artifact.",
        )
    if (
        build.get("timeline_contract") != source.get("timeline_contract")
        or build.get("timeline_contract_version") != source.get("timeline_contract_version")
        or build.get("outcome_contract") != source.get("outcome_contract")
        or build.get("outcome_contract_version") != source.get("outcome_contract_version")
    ):
        raise ArtifactError(
            code="COHORT_BUILD_CONTRACT_LINEAGE_MISMATCH",
            message="Cohort split and evaluation source contracts disagree.",
        )
    raw_assignments = build.get("assignments")
    if not isinstance(raw_assignments, list):
        raise ArtifactError(
            code="COHORT_SPLIT_ASSIGNMENTS_MISSING",
            message="Cohort split assignments are missing from the build artifact.",
        )
    split_by_patient = {
        str(item["patient_id"]): str(item["split"])
        for item in raw_assignments
        if isinstance(item, Mapping)
    }
    patient_ids = tuple(str(value) for value in source["patient_ids"])
    if len(split_by_patient) != len(raw_assignments) or set(split_by_patient) != set(patient_ids):
        raise ArtifactError(
            code="COHORT_SPLIT_PATIENT_MISMATCH",
            message="Evaluation split assignments do not exactly cover source patients.",
        )
    row_by_patient = {patient_id: index for index, patient_id in enumerate(patient_ids)}
    durations = torch.as_tensor(source["survival_durations"]).float().numpy() * 365.25
    events = torch.as_tensor(source["survival_events"]).long().numpy()
    valid = torch.as_tensor(source["survival_valid"]).bool().numpy()
    writer = EvaluationArtifactWriter(
        config.output_root / "evaluation", EvaluationRunMode.SYNTHETIC
    )
    metric_paths: list[str] = []
    metric_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    curves: list[tuple[str, float, Any]] = []
    metric_ids: list[str] = []

    for stage_index, stage in enumerate(("s0", "s1", "s2")):
        partitions: dict[str, tuple[str, ...]] = {}
        for split in ("train", "validation", "test"):
            partitions[split] = tuple(
                patient_id
                for patient_id in patient_ids
                if split_by_patient[patient_id] == split
                and bool(valid[row_by_patient[patient_id], stage_index])
            )
        if not all(partitions.values()):
            continue

        train_cohort = _make_evaluation_cohort(
            ids=partitions["train"],
            split="train",
            role=SplitRole.TRAIN,
            stage=stage,
            stage_index=stage_index,
            durations=durations,
            events=events,
            row_by_patient=row_by_patient,
        )
        development = _make_evaluation_cohort(
            ids=partitions["validation"],
            split="validation",
            role=SplitRole.DEVELOPMENT,
            stage=stage,
            stage_index=stage_index,
            durations=durations,
            events=events,
            row_by_patient=row_by_patient,
        )
        test = _make_evaluation_cohort(
            ids=partitions["test"],
            split="test",
            role=SplitRole.TEST,
            stage=stage,
            stage_index=stage_index,
            durations=durations,
            events=events,
            row_by_patient=row_by_patient,
        )
        censoring_id = f"synthetic-censoring-{stage}-train-v1"
        censoring = CensoringDistribution.fit(train_cohort, artifact_id=censoring_id)
        protocol = EvaluationProtocol(
            protocol_id=protocol_id,
            endpoint="os",
            horizons=horizons,
            evaluation_role=SplitRole.TEST,
        )
        risk_matrix = np.column_stack(
            [_prediction_risks(artifact, test.patient_ids, stage, horizon) for horizon in horizons]
        )
        for horizon_index, horizon in enumerate(horizons):
            risks = risk_matrix[:, horizon_index]
            for metric_name, function in (
                ("cumulative_dynamic_auc", cumulative_dynamic_auc),
                ("ipcw_concordance_index", ipcw_concordance_index),
                ("ipcw_brier_score", ipcw_brier_score),
            ):
                lineage = build_metric_lineage(
                    (artifact,),
                    censoring_artifact_id=censoring_id,
                    protocol_id=protocol_id,
                    artifact_id=new_artifact_id(f"{metric_name}-{stage}"),
                )
                result = function(
                    test,
                    risks.tolist(),
                    horizon,
                    censoring,
                    protocol=protocol,
                    lineage=lineage,
                )
                path = writer.write_metric(result)
                metric_paths.append(str(path))
                metric_ids.append(lineage.artifact_id)
                metric_rows.append({"stage": stage, **result.as_dict()})
            dev_risk = _prediction_risks(artifact, development.patient_ids, stage, horizon)
            bins = CalibrationBinSpec.fit(
                development,
                dev_risk,
                bins=2,
                source_prediction_artifact_id=artifact.lineage.artifact_id,
            )
            curve = calibration_curve(
                test, risks.tolist(), horizon, censoring, bins, protocol=protocol
            )
            curves.append((stage, horizon, curve))
            calibration_rows.append({"stage": stage, **curve.as_dict()})
        lineage = build_metric_lineage(
            (artifact,),
            censoring_artifact_id=censoring_id,
            protocol_id=protocol_id,
            artifact_id=new_artifact_id(f"integrated-brier-{stage}"),
        )
        ibs = integrated_brier_score(
            test,
            risk_matrix.tolist(),
            horizons,
            censoring,
            protocol=protocol,
            lineage=lineage,
        )
        path = writer.write_metric(ibs)
        metric_paths.append(str(path))
        metric_ids.append(lineage.artifact_id)
        metric_rows.append({"stage": stage, **ibs.as_dict()})

    evaluation_root = config.output_root / "evaluation" / "synthetic"
    calibration_path = evaluation_root / "calibration" / "calibration.json"
    atomic_write_json(
        calibration_path,
        {
            "schema_version": "stageworld-calibration-results-v1",
            "mode": "synthetic",
            "prediction_artifact_id": artifact.lineage.artifact_id,
            "protocol_id": protocol_id,
            "results": calibration_rows,
        },
    )
    plot_path = evaluation_root / "figures" / "calibration.png"
    _plot_calibration(plot_path, curves)
    figure = FigureLineage(
        artifact_id=new_artifact_id("calibration-figure"),
        figure_kind="calibration_png",
        prediction_artifact_ids=(artifact.lineage.artifact_id,),
        metric_artifact_ids=tuple(metric_ids),
        protocol_id=protocol_id,
        run_mode=EvaluationRunMode.SYNTHETIC,
    )
    figure_lineage_path = writer.write_figure_lineage(figure)
    summary_path = evaluation_root / "evaluation_summary.json"
    atomic_write_json(
        summary_path,
        {
            "schema_version": EVALUATION_SUMMARY_SCHEMA_VERSION,
            "mode": "synthetic",
            "clinical_validation": False,
            "protocol": {
                "protocol_id": protocol_id,
                "endpoint": "os",
                "horizons_days": list(horizons),
            },
            "prediction_artifact": str(predictions),
            "prediction_artifact_id": artifact.lineage.artifact_id,
            "prediction_lineage": artifact.lineage.as_dict(),
            **_checkpoint_lineage_fields(checkpoint),
            "checkpoint_snapshot": str(checkpoint_snapshot),
            "parent_checkpoint_snapshot": str(parent_checkpoint_snapshot),
            "metric_artifacts": metric_paths,
            "metrics": metric_rows,
            "calibration": str(calibration_path),
            "calibration_plot": str(plot_path),
            "figure_lineage": str(figure_lineage_path),
        },
    )
    return {
        "status": "ok",
        "mode": "synthetic",
        "clinical_validation": False,
        "metric_count": len(metric_rows),
        "estimable_metric_count": sum(row["status"] == "ok" for row in metric_rows),
        "not_estimable_metric_count": sum(row["status"] == "not_estimable" for row in metric_rows),
        "summary": str(summary_path),
        "calibration_plot": str(plot_path),
        "figure_lineage": str(figure_lineage_path),
    }


def generate_synthetic_report(
    config: StageWorldConfig, *, run_dir: Path | None = None
) -> dict[str, Any]:
    _require_synthetic(config, "report")
    root = run_dir or config.output_root
    evaluation_path = root / "evaluation" / "synthetic" / "evaluation_summary.json"
    try:
        evaluation = read_json(evaluation_path)
    except (OSError, ValueError) as error:
        raise ArtifactError(
            code="REPORT_EVALUATION_SUMMARY_INVALID",
            message="Synthetic report requires a readable evaluation summary.",
        ) from error
    if (
        evaluation.get("schema_version") != EVALUATION_SUMMARY_SCHEMA_VERSION
        or evaluation.get("mode") != "synthetic"
    ):
        raise ArtifactError(
            code="REPORT_MODE_MISMATCH",
            message="Synthetic report refuses an unsupported or non-synthetic evaluation.",
        )

    source = _load_tensor_artifact(root / "data" / "source_tensors.pt", SOURCE_SCHEMA)
    validate_synthetic_source_config(config, source)
    data_lineage_id = str(source["data_lineage_id"])
    cohort_artifact_id = str(source["cohort_artifact_id"])
    timeline_contract_version = str(source["timeline_contract_version"])
    outcome_contract_version = str(source["outcome_contract_version"])
    ct_payload = _load_tensor_artifact(root / "features" / "ct.pt", FEATURE_SCHEMA)
    pathology_payload = _load_tensor_artifact(root / "features" / "pathology.pt", FEATURE_SCHEMA)
    ct_feature_artifact_id, pathology_feature_artifact_id = _feature_artifact_lineage(
        config, source, ct_payload, pathology_payload
    )
    try:
        cohort_build = read_json(root / "data" / "cohort_build.json")
    except (OSError, ValueError) as error:
        raise ArtifactError(
            code="COHORT_BUILD_SCHEMA_MISMATCH",
            message="Synthetic report requires a readable cohort split artifact.",
        ) from error
    if cohort_build.get("schema_version") != BUILD_SCHEMA:
        raise ArtifactError(
            code="COHORT_BUILD_SCHEMA_MISMATCH",
            message="Cohort split artifact schema is incompatible with reporting.",
        )
    if cohort_build.get("data_lineage_id") != data_lineage_id:
        raise ArtifactError(
            code="COHORT_BUILD_DATA_LINEAGE_MISMATCH",
            message="Cohort split and report source do not share a data lineage.",
        )
    if cohort_build.get("cohort_artifact_id") != cohort_artifact_id:
        raise ArtifactError(
            code="COHORT_BUILD_COHORT_LINEAGE_MISMATCH",
            message="Cohort split and report source do not share a cohort artifact.",
        )
    if (
        cohort_build.get("timeline_contract") != source.get("timeline_contract")
        or cohort_build.get("timeline_contract_version") != timeline_contract_version
        or cohort_build.get("outcome_contract") != source.get("outcome_contract")
        or cohort_build.get("outcome_contract_version") != outcome_contract_version
    ):
        raise ArtifactError(
            code="COHORT_BUILD_CONTRACT_LINEAGE_MISMATCH",
            message="Cohort split and report source contracts disagree.",
        )
    split_version = cohort_build.get("split_version")
    if not isinstance(split_version, str) or not split_version.strip():
        raise ArtifactError(
            code="COHORT_SPLIT_VERSION_MISSING",
            message="Cohort split artifact has no immutable split version.",
        )
    (
        checkpoint,
        checkpoint_snapshot,
        parent_checkpoint,
        parent_checkpoint_snapshot,
    ) = _load_exact_report_checkpoint(
        config,
        root=root,
        data_lineage_id=data_lineage_id,
        cohort_artifact_id=cohort_artifact_id,
        split_version=split_version,
        ct_feature_artifact_id=ct_feature_artifact_id,
        pathology_feature_artifact_id=pathology_feature_artifact_id,
        timeline_contract_version=timeline_contract_version,
        outcome_contract_version=outcome_contract_version,
    )

    prediction_path_value = evaluation.get("prediction_artifact")
    if not isinstance(prediction_path_value, str) or not prediction_path_value.strip():
        raise ArtifactError(
            code="REPORT_PREDICTION_ARTIFACT_INVALID",
            message="Evaluation summary does not link a prediction artifact.",
        )
    prediction_path = Path(prediction_path_value)
    try:
        prediction = read_prediction_artifact(prediction_path)
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ArtifactError(
            code="REPORT_PREDICTION_ARTIFACT_INVALID",
            message="Evaluation summary links an unreadable prediction artifact.",
        ) from error
    if prediction.lineage.run_mode is not EvaluationRunMode.SYNTHETIC:
        raise ArtifactError(
            code="REPORT_MODE_MISMATCH",
            message="Synthetic report refuses a non-synthetic prediction artifact.",
        )
    _validate_prediction_checkpoint_lineage(prediction, checkpoint)

    expected_evaluation_lineage: dict[str, object] = {
        **_checkpoint_lineage_fields(checkpoint),
        "checkpoint_snapshot": str(checkpoint_snapshot),
        "parent_checkpoint_snapshot": str(parent_checkpoint_snapshot),
        "prediction_artifact_id": prediction.lineage.artifact_id,
        "prediction_lineage": prediction.lineage.as_dict(),
    }
    evaluation_mismatches = sorted(
        key
        for key, expected in expected_evaluation_lineage.items()
        if evaluation.get(key) != expected
    )
    protocol = evaluation.get("protocol")
    metrics = evaluation.get("metrics")
    if not isinstance(protocol, Mapping) or not isinstance(metrics, list):
        evaluation_mismatches.append("protocol_or_metrics")
        protocol_id = None
    else:
        protocol_id = protocol.get("protocol_id")
        for index, row in enumerate(metrics):
            if not isinstance(row, Mapping) or not isinstance(row.get("lineage"), Mapping):
                evaluation_mismatches.append(f"metrics[{index}].lineage")
                continue
            metric_lineage = row["lineage"]
            expected_metric_lineage = {
                "prediction_artifact_ids": [prediction.lineage.artifact_id],
                "model_artifact_ids": [checkpoint.weight_version],
                "protocol_id": protocol_id,
                "run_mode": "synthetic",
            }
            evaluation_mismatches.extend(
                f"metrics[{index}].lineage.{key}"
                for key, expected in expected_metric_lineage.items()
                if metric_lineage.get(key) != expected
            )
    if evaluation_mismatches:
        raise ArtifactError(
            code="REPORT_EVALUATION_LINEAGE_MISMATCH",
            message="Evaluation artifacts are stale or disagree with current prediction lineage.",
            details={"fields": sorted(set(evaluation_mismatches))},
        )

    joint_summary = read_json(root / "runs" / "joint_survival" / "summary.json")
    expected_joint_summary = {
        "schema_version": TRAINING_SUMMARY_SCHEMA_VERSION,
        "mode": "synthetic",
        "phase": TrainingPhase.JOINT_SURVIVAL.value,
        "checkpoint": str(root / "runs" / "joint_survival" / "checkpoint.pt"),
        "checkpoint_snapshot": str(checkpoint_snapshot),
        "checkpoint_id": checkpoint.checkpoint_id,
        "weight_version": checkpoint.weight_version,
        "optimizer_steps": checkpoint.step,
        "config_lineage_id": checkpoint.config_lineage_id,
        "data_lineage_id": checkpoint.data_lineage_id,
        "cohort_artifact_id": checkpoint.cohort_artifact_id,
        "split_version": checkpoint.split_version,
        "ct_feature_artifact_id": checkpoint.ct_feature_artifact_id,
        "pathology_feature_artifact_id": checkpoint.pathology_feature_artifact_id,
        "timeline_contract_version": checkpoint.timeline_contract_version,
        "outcome_contract_version": checkpoint.outcome_contract_version,
        "training_seed": checkpoint.training_seed,
        "source_schema_version": checkpoint.source_schema_version,
        "cohort_schema_version": checkpoint.cohort_schema_version,
        "feature_schema_version": checkpoint.feature_schema_version,
        "parent_checkpoint_id": checkpoint.parent_checkpoint_id,
        "parent_weight_version": checkpoint.parent_weight_version,
        "parent_phase": checkpoint.parent_phase,
        "parent_config_lineage_id": checkpoint.parent_config_lineage_id,
        "parent_data_lineage_id": checkpoint.parent_data_lineage_id,
        "parent_cohort_artifact_id": checkpoint.parent_cohort_artifact_id,
        "parent_split_version": checkpoint.parent_split_version,
        "parent_ct_feature_artifact_id": checkpoint.parent_ct_feature_artifact_id,
        "parent_pathology_feature_artifact_id": checkpoint.parent_pathology_feature_artifact_id,
        "parent_timeline_contract_version": checkpoint.parent_timeline_contract_version,
        "parent_outcome_contract_version": checkpoint.parent_outcome_contract_version,
    }
    mismatched_joint_summary = sorted(
        key
        for key, expected in expected_joint_summary.items()
        if joint_summary.get(key) != expected
    )
    if mismatched_joint_summary:
        raise ArtifactError(
            code="REPORT_CHECKPOINT_SUMMARY_MISMATCH",
            message="Joint training summary disagrees with the current checkpoint.",
            details={"fields": mismatched_joint_summary},
        )
    estimable_metric_count = sum(row["status"] == "ok" for row in evaluation["metrics"])
    not_estimable_metric_count = sum(
        row["status"] == "not_estimable" for row in evaluation["metrics"]
    )
    lines = [
        "# StageWorld-GC Synthetic Smoke Report",
        "",
        "> This report contains only fictional synthetic data. It is not clinical validation.",
        "",
        "## Execution",
        "",
        f"- World-pretraining optimizer steps: {parent_checkpoint.step}",
        f"- Joint-survival optimizer steps: {joint_summary['optimizer_steps']}",
        f"- Joint checkpoint: `{checkpoint.checkpoint_id}`",
        f"- Joint weight version: `{checkpoint.weight_version}`",
        f"- Prediction artifact: `{prediction.lineage.artifact_id}`",
        "",
        "## Metrics",
        "",
        f"- Estimable metric results: {estimable_metric_count}",
        f"- Explicitly not-estimable metric results: {not_estimable_metric_count}",
        "",
        "| Stage | Metric | Horizon days | Status | Estimate / reason |",
        "|---|---|---:|---|---|",
    ]
    for row in evaluation["metrics"]:
        estimate = f"{float(row['estimate']):.6f}" if row["status"] == "ok" else str(row["reason"])
        horizon = "" if row["horizon"] is None else f"{float(row['horizon']):.2f}"
        lines.append(
            f"| {row['stage']} | {row['metric']} | {horizon} | {row['status']} | {estimate} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "Synthetic optimization traces and explicit metric-status records demonstrate "
            "executable plumbing only. They do not estimate gastric-cancer performance, "
            "treatment effects, or generalization.",
            "",
            f"Calibration plot: `{evaluation['calibration_plot']}`",
            "",
        ]
    )
    report_path = root / "report" / "SYNTHETIC_REPORT.md"
    _atomic_text(report_path, "\n".join(lines))
    manifest_path = root / "report" / "report_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": REPORT_MANIFEST_SCHEMA_VERSION,
            "mode": "synthetic",
            "clinical_validation": False,
            "report": str(report_path),
            "evaluation_summary": str(evaluation_path),
            "prediction_artifact": str(prediction_path),
            "prediction_artifact_id": prediction.lineage.artifact_id,
            "prediction_lineage": prediction.lineage.as_dict(),
            **_checkpoint_lineage_fields(checkpoint),
            "checkpoint_snapshot": str(checkpoint_snapshot),
            "parent_checkpoint_snapshot": str(parent_checkpoint_snapshot),
            "joint_checkpoint_id": checkpoint.checkpoint_id,
            "joint_weight_version": checkpoint.weight_version,
        },
    )
    return {
        "status": "ok",
        "mode": "synthetic",
        "clinical_validation": False,
        "report": str(report_path),
        "manifest": str(manifest_path),
    }


__all__ = [
    "generate_synthetic_report",
    "run_synthetic_evaluation",
    "run_synthetic_prediction",
]
