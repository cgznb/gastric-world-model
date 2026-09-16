"""Offline synthetic artifacts used for executable StageWorld smoke workflows."""

from __future__ import annotations

import json
import os
import random
import tempfile
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import Tensor

from stageworld.artifacts import atomic_write_json, new_artifact_id, read_json
from stageworld.config import RunMode, StageWorldConfig
from stageworld.data import (
    SYNTHETIC_TIMELINE,
    TREATMENT_KIND_ID,
    TREATMENT_STATUS_ID,
    ClinicalMeasurement,
    Cohort,
    DataMode,
    EventType,
    FeatureFirewall,
    LandmarkBuilder,
    LandmarkSet,
    MissingCategory,
    Observation,
    ObservationRole,
    OutcomeBuilder,
    OutcomeDefinition,
    QualityStatus,
    Query,
    QueryEligibility,
    SplitAssignment,
    SplitManager,
    SplitName,
    Stage,
    Treatment,
    TreatmentKind,
    TreatmentStatus,
    cohort_from_dict,
    cohort_to_dict,
    default_synthetic_policy,
    generate_synthetic_cohort,
    load_cohort_json,
)
from stageworld.encoders.base import EncoderProvenance, ObservationTokens
from stageworld.encoders.synthetic import SyntheticEncoder
from stageworld.errors import (
    ArtifactError,
    ConfigurationError,
    DataContractError,
    ResourceError,
)
from stageworld.inference import CheckpointContract
from stageworld.model import ActionTokens, StageWorldModel, StageWorldModelConfig
from stageworld.training import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointMetadata,
    ExperimentRegistry,
    LocalEventLogger,
    LossWeights,
    StageWorldTrainer,
    TrainingPhase,
    WorldModelBatch,
    bounded_fit,
    checkpoint_payload_mismatches,
    checkpoint_snapshot_path,
    model_state_mismatches,
    new_checkpoint_metadata,
)

SOURCE_SCHEMA = "stageworld-synthetic-source-v3"
FEATURE_SCHEMA = "stageworld-synthetic-features-v3"
BUILD_SCHEMA = "stageworld-cohort-build-v3"
CHECKPOINT_SIDECAR_SCHEMA = "stageworld-checkpoint-sidecar-v2"
TRAINING_SUMMARY_SCHEMA = "stageworld-training-summary-v3"
SYNTHETIC_COHORT_LINEAGE_SCHEMA = "stageworld-synthetic-cohort-lineage-v1"
SYNTHETIC_TIMELINE_CONTRACT_VERSION = "stageworld-synthetic-timeline-contract-v1"
SYNTHETIC_OUTCOME_CONTRACT_VERSION = "stageworld-synthetic-os-contract-v1"
SYNTHETIC_SPLIT_POLICY_VERSION = "stageworld-synthetic-split-60-20-v1"

_SURVIVAL_STAGE_ORDER = (Stage.S0, Stage.S1, Stage.S2)
_SURVIVAL_TIME_UNIT = "year"
_DAYS_PER_YEAR = 365.25
_SYNTHETIC_TRAIN_FRACTION = 0.6
_SYNTHETIC_VALIDATION_FRACTION = 0.2


def _require_synthetic(config: StageWorldConfig, command: str) -> None:
    if config.mode is not RunMode.SYNTHETIC:
        raise ConfigurationError(
            code="SYNTHETIC_WORKFLOW_MODE_REQUIRED",
            message=f"{command} is a synthetic-only workflow.",
            remediation="Use an explicit real feature/image command after its approvals are met.",
        )


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _load_tensor_artifact(path: Path, schema: str) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError) as error:
        raise ArtifactError(
            code="SYNTHETIC_ARTIFACT_UNREADABLE",
            message="A required synthetic artifact is absent or incomplete.",
            remediation="Run the preceding synthetic CLI stage again.",
            details={"artifact": path.name},
        ) from error
    if not isinstance(payload, dict) or payload.get("schema_version") != schema:
        raise ArtifactError(
            code="SYNTHETIC_ARTIFACT_SCHEMA_MISMATCH",
            message="Synthetic artifact schema is absent or incompatible.",
            details={"artifact": path.name, "expected": schema},
        )
    return payload


def synthetic_data_root(config: StageWorldConfig) -> Path:
    return config.output_root / "data"


def synthetic_feature_root(config: StageWorldConfig) -> Path:
    if config.paths.feature_root:
        return Path(config.paths.feature_root)
    return config.output_root / "features"


def _synthetic_outcome_definition() -> OutcomeDefinition:
    return OutcomeDefinition(
        endpoint_name="os",
        event_type=EventType.DEATH,
        event_code=1,
        origin_definition="synthetic_baseline_day_zero",
        label_version="synthetic-os-v1",
        mode=DataMode.SYNTHETIC,
        status_mapping_confirmed=True,
        origin_confirmed=True,
        timeline_confirmed=True,
    )


def _synthetic_outcome_contract() -> dict[str, Any]:
    definition = _synthetic_outcome_definition()
    return {
        "contract_version": SYNTHETIC_OUTCOME_CONTRACT_VERSION,
        "endpoint_name": definition.endpoint_name,
        "event_type": definition.event_type.value,
        "event_code": definition.event_code,
        "origin_definition": definition.origin_definition,
        "label_version": definition.label_version,
        "mode": definition.mode.value,
        "status_mapping_confirmed": definition.status_mapping_confirmed,
        "origin_confirmed": definition.origin_confirmed,
        "timeline_confirmed": definition.timeline_confirmed,
        "zero_time_policy": definition.zero_time_policy.value,
    }


def _synthetic_firewall(cohort: Cohort) -> FeatureFirewall:
    return FeatureFirewall(
        cohort.patients,
        cohort.observations,
        cohort.clinical_measurements,
        cohort.treatments,
        default_synthetic_policy(),
    )


def _canonical_stage_queries(cohort: Cohort, patient_ids: tuple[str, ...]) -> tuple[Query, ...]:
    queries_by_id: dict[str, list[Query]] = {}
    for query in cohort.queries:
        queries_by_id.setdefault(query.query_id, []).append(query)
    expected_times = {
        Stage.S0: SYNTHETIC_TIMELINE.s0_query,
        Stage.S1: SYNTHETIC_TIMELINE.s1_query,
        Stage.S2: SYNTHETIC_TIMELINE.s2_query,
    }
    ordered: list[Query] = []
    for patient_id in patient_ids:
        for stage in _SURVIVAL_STAGE_ORDER:
            query_id = f"{patient_id}-{stage.value}"
            matches = queries_by_id.get(query_id, [])
            if len(matches) != 1:
                raise ArtifactError(
                    code="SYNTHETIC_COHORT_TIMELINE_MISMATCH",
                    message="Each synthetic patient requires one canonical query per stage.",
                    details={"patient_id": patient_id, "stage": stage.value},
                )
            query = matches[0]
            if (
                query.patient_id != patient_id
                or query.stage is not stage
                or query.query_time_days != expected_times[stage]
            ):
                raise ArtifactError(
                    code="SYNTHETIC_COHORT_TIMELINE_MISMATCH",
                    message="A canonical synthetic query differs from the source timeline.",
                    details={"patient_id": patient_id, "stage": stage.value},
                )
            ordered.append(query)
    expected_query_ids = {
        f"{patient_id}-{stage.value}"
        for patient_id in patient_ids
        for stage in _SURVIVAL_STAGE_ORDER
    }
    expected_query_ids.add("SYN-0000-after-event")
    actual_query_ids = [query.query_id for query in cohort.queries]
    if len(actual_query_ids) != len(set(actual_query_ids)) or set(actual_query_ids) != (
        expected_query_ids
    ):
        raise ArtifactError(
            code="SYNTHETIC_COHORT_QUERY_UNIVERSE_MISMATCH",
            message=(
                "Synthetic cohort queries must contain exactly the canonical three-stage "
                "queries and the declared already-event audit query."
            ),
            details={
                "expected_query_count": len(expected_query_ids),
                "actual_query_count": len(actual_query_ids),
            },
        )
    return tuple(ordered)


def _build_synthetic_landmarks(cohort: Cohort, queries: tuple[Query, ...]) -> LandmarkSet:
    try:
        return LandmarkBuilder(
            _synthetic_firewall(cohort),
            OutcomeBuilder(cohort.outcomes, _synthetic_outcome_definition()),
        ).build(queries)
    except (ConfigurationError, DataContractError) as error:
        raise ArtifactError(
            code="SYNTHETIC_SURVIVAL_LABEL_MISMATCH",
            message="Synthetic outcomes cannot produce the configured legal OS landmarks.",
        ) from error


def _synthetic_survival_targets(
    cohort: Cohort, patient_ids: tuple[str, ...]
) -> tuple[Tensor, Tensor, Tensor]:
    queries = _canonical_stage_queries(cohort, patient_ids)
    landmark_set = _build_synthetic_landmarks(cohort, queries)
    row_by_patient = {patient_id: row for row, patient_id in enumerate(patient_ids)}
    stage_index = {stage: index for index, stage in enumerate(_SURVIVAL_STAGE_ORDER)}
    durations = torch.zeros(len(patient_ids), len(_SURVIVAL_STAGE_ORDER), dtype=torch.float32)
    events = torch.zeros(len(patient_ids), len(_SURVIVAL_STAGE_ORDER), dtype=torch.long)
    valid = torch.zeros(len(patient_ids), len(_SURVIVAL_STAGE_ORDER), dtype=torch.bool)
    represented_queries: set[str] = set()
    for landmark in landmark_set.landmarks:
        column = stage_index[landmark.stage]
        key = (row_by_patient[landmark.patient_id], column)
        if bool(valid[key]):
            raise ArtifactError(
                code="SYNTHETIC_SURVIVAL_LABEL_MISMATCH",
                message="A synthetic patient has duplicate OS landmarks for one stage.",
            )
        durations[key] = landmark.label.remaining_time_days / _DAYS_PER_YEAR
        events[key] = int(landmark.label.event)
        valid[key] = True
        represented_queries.add(landmark.prefix.query.query_id)
    represented_queries.update(item.query_id for item in landmark_set.exclusions)
    expected_query_ids = {query.query_id for query in queries}
    if represented_queries != expected_query_ids:
        raise ArtifactError(
            code="SYNTHETIC_SURVIVAL_LABEL_MISMATCH",
            message="Synthetic canonical queries were not uniquely labelled or excluded.",
        )
    return durations, events, valid


def _require_clock(
    *,
    patient_id: str,
    record_kind: str,
    acquired: float | None,
    available: float | None,
    expected_acquired: float,
    expected_available: float,
) -> None:
    if acquired != expected_acquired or available != expected_available:
        raise ArtifactError(
            code="SYNTHETIC_COHORT_TIMELINE_MISMATCH",
            message="A synthetic cohort record differs from the declared dual-clock timeline.",
            details={"patient_id": patient_id, "record_kind": record_kind},
        )


def _validate_synthetic_cohort_timeline(cohort: Cohort, patient_ids: tuple[str, ...]) -> None:
    canonical_queries = _canonical_stage_queries(cohort, patient_ids)
    query_by_patient_stage = {(query.patient_id, query.stage): query for query in canonical_queries}
    observations_by_patient: dict[str, list[Observation]] = {
        patient_id: [] for patient_id in patient_ids
    }
    measurements_by_patient: dict[str, list[ClinicalMeasurement]] = {
        patient_id: [] for patient_id in patient_ids
    }
    treatments_by_patient: dict[str, list[Treatment]] = {
        patient_id: [] for patient_id in patient_ids
    }
    for observation in cohort.observations:
        observations_by_patient[observation.patient_id].append(observation)
    for measurement in cohort.clinical_measurements:
        measurements_by_patient[measurement.patient_id].append(measurement)
    for treatment in cohort.treatments:
        treatments_by_patient[treatment.patient_id].append(treatment)

    expected_measurement_clocks = {
        "age_years": (SYNTHETIC_TIMELINE.baseline_acquired, SYNTHETIC_TIMELINE.s0_query),
        "baseline_stage": (
            SYNTHETIC_TIMELINE.baseline_acquired,
            SYNTHETIC_TIMELINE.s0_query,
        ),
        "radiologic_response": (
            SYNTHETIC_TIMELINE.ct1_acquired,
            SYNTHETIC_TIMELINE.ct1_available,
        ),
        "yp_stage": (
            SYNTHETIC_TIMELINE.pathology_acquired,
            SYNTHETIC_TIMELINE.pathology_available,
        ),
    }
    for patient_id in patient_ids:
        observations = observations_by_patient[patient_id]
        baseline = [item for item in observations if item.role is ObservationRole.BASELINE_CT]
        post_treatment = [
            item for item in observations if item.role is ObservationRole.POST_TREATMENT_CT
        ]
        pathology = [
            item for item in observations if item.role is ObservationRole.SURGICAL_PATHOLOGY
        ]
        if len(baseline) != 1 or len(post_treatment) != 1:
            raise ArtifactError(
                code="SYNTHETIC_COHORT_TIMELINE_MISMATCH",
                message="Synthetic patients require one baseline and one post-treatment CT record.",
                details={"patient_id": patient_id},
            )
        _require_clock(
            patient_id=patient_id,
            record_kind="baseline_ct",
            acquired=baseline[0].acquired_at_days,
            available=baseline[0].available_at_days,
            expected_acquired=SYNTHETIC_TIMELINE.baseline_acquired,
            expected_available=SYNTHETIC_TIMELINE.s0_query,
        )
        _require_clock(
            patient_id=patient_id,
            record_kind="post_treatment_ct",
            acquired=post_treatment[0].acquired_at_days,
            available=post_treatment[0].available_at_days,
            expected_acquired=SYNTHETIC_TIMELINE.ct1_acquired,
            expected_available=SYNTHETIC_TIMELINE.ct1_available,
        )
        for observation in pathology:
            _require_clock(
                patient_id=patient_id,
                record_kind="surgical_pathology",
                acquired=observation.acquired_at_days,
                available=observation.available_at_days,
                expected_acquired=SYNTHETIC_TIMELINE.pathology_acquired,
                expected_available=SYNTHETIC_TIMELINE.pathology_available,
            )

        treatments = treatments_by_patient[patient_id]
        systemic = [
            item
            for item in treatments
            if item.treatment_kind is TreatmentKind.SYSTEMIC
            and item.planned_or_delivered is TreatmentStatus.DELIVERED
        ]
        surgery = [
            item
            for item in treatments
            if item.treatment_kind is TreatmentKind.SURGERY
            and item.planned_or_delivered is TreatmentStatus.DELIVERED
        ]
        if len(systemic) != 1 or len(surgery) > 1:
            raise ArtifactError(
                code="SYNTHETIC_COHORT_TIMELINE_MISMATCH",
                message="Synthetic delivered treatment records do not match the timeline contract.",
                details={"patient_id": patient_id},
            )
        _require_clock(
            patient_id=patient_id,
            record_kind="delivered_systemic",
            acquired=systemic[0].end_days,
            available=systemic[0].available_at_days,
            expected_acquired=SYNTHETIC_TIMELINE.systemic_event,
            expected_available=SYNTHETIC_TIMELINE.systemic_available,
        )
        for treatment in treatments:
            if treatment.event_id.endswith("-post-ct-treatment"):
                _require_clock(
                    patient_id=patient_id,
                    record_kind="post_ct_treatment",
                    acquired=treatment.end_days,
                    available=treatment.available_at_days,
                    expected_acquired=SYNTHETIC_TIMELINE.post_ct_event,
                    expected_available=SYNTHETIC_TIMELINE.post_ct_available,
                )
        if surgery:
            _require_clock(
                patient_id=patient_id,
                record_kind="surgery",
                acquired=surgery[0].end_days,
                available=surgery[0].available_at_days,
                expected_acquired=SYNTHETIC_TIMELINE.surgery_event,
                expected_available=SYNTHETIC_TIMELINE.surgery_available,
            )
        if bool(surgery) != bool(pathology):
            raise ArtifactError(
                code="SYNTHETIC_COHORT_TIMELINE_MISMATCH",
                message="Synthetic surgery applicability and pathology records disagree.",
                details={"patient_id": patient_id},
            )
        expected_s2_eligibility = (
            QueryEligibility.ELIGIBLE if surgery else QueryEligibility.NOT_APPLICABLE
        )
        if (
            query_by_patient_stage[(patient_id, Stage.S2)].eligibility
            is not expected_s2_eligibility
        ):
            raise ArtifactError(
                code="SYNTHETIC_COHORT_TIMELINE_MISMATCH",
                message="Synthetic S2 applicability differs from the observed surgery path.",
                details={"patient_id": patient_id},
            )

        measurements = measurements_by_patient[patient_id]
        for field_name, (acquired, available) in expected_measurement_clocks.items():
            matches = [item for item in measurements if item.field_name == field_name]
            expected_count = 0 if field_name == "yp_stage" and not surgery else 1
            if len(matches) != expected_count:
                raise ArtifactError(
                    code="SYNTHETIC_COHORT_TIMELINE_MISMATCH",
                    message="Synthetic clinical records do not match the stage timeline.",
                    details={"patient_id": patient_id, "field": field_name},
                )
            for measurement in matches:
                _require_clock(
                    patient_id=patient_id,
                    record_kind=f"clinical:{field_name}",
                    acquired=measurement.acquired_at_days,
                    available=measurement.available_at_days,
                    expected_acquired=acquired,
                    expected_available=available,
                )

    all_landmarks = _build_synthetic_landmarks(cohort, cohort.queries)
    if not any(
        item.query_id == "SYN-0000-after-event" and item.reason == "event_already_occurred"
        for item in all_landmarks.exclusions
    ):
        raise ArtifactError(
            code="SYNTHETIC_COHORT_TIMELINE_MISMATCH",
            message="The synthetic already-event landmark exclusion is absent or changed.",
        )


def _source_patient_ids(source: Mapping[str, Any]) -> tuple[str, ...]:
    raw = source.get("patient_ids")
    if not isinstance(raw, (tuple, list)):
        raise ArtifactError(
            code="SYNTHETIC_SOURCE_COHORT_MISMATCH",
            message="Synthetic source patient_ids must be an ordered sequence.",
        )
    patient_ids = tuple(str(item) for item in raw)
    if not patient_ids or len(patient_ids) != len(set(patient_ids)):
        raise ArtifactError(
            code="SYNTHETIC_SOURCE_COHORT_MISMATCH",
            message="Synthetic source patient_ids are empty or duplicated.",
        )
    return patient_ids


def _validate_survival_targets(source: Mapping[str, Any], cohort: Cohort) -> None:
    patient_ids = _source_patient_ids(source)
    expected_query_ids = tuple(
        tuple(f"{patient_id}-{stage.value}" for stage in _SURVIVAL_STAGE_ORDER)
        for patient_id in patient_ids
    )
    if (
        tuple(source.get("survival_stage_order", ()))
        != tuple(stage.value for stage in _SURVIVAL_STAGE_ORDER)
        or source.get("survival_endpoint") != "os"
        or source.get("survival_label_version") != "synthetic-os-v1"
        or source.get("survival_time_unit") != _SURVIVAL_TIME_UNIT
        or source.get("survival_query_ids") != expected_query_ids
        or source.get("outcome_contract") != _synthetic_outcome_contract()
    ):
        raise ArtifactError(
            code="SYNTHETIC_SURVIVAL_LABEL_MISMATCH",
            message="Synthetic source survival metadata does not identify the canonical OS labels.",
        )
    expected_durations, expected_events, expected_valid = _synthetic_survival_targets(
        cohort, patient_ids
    )
    try:
        durations = torch.as_tensor(source["survival_durations"])
        events = torch.as_tensor(source["survival_events"])
        valid = torch.as_tensor(source["survival_valid"])
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ArtifactError(
            code="SYNTHETIC_SURVIVAL_LABEL_MISMATCH",
            message="Synthetic source survival tensors are absent or malformed.",
        ) from error
    if (
        durations.dtype != torch.float32
        or events.dtype != torch.long
        or valid.dtype != torch.bool
        or not torch.equal(durations, expected_durations)
        or not torch.equal(events, expected_events)
        or not torch.equal(valid, expected_valid)
    ):
        raise ArtifactError(
            code="SYNTHETIC_SURVIVAL_LABEL_MISMATCH",
            message="Synthetic source OS tensors differ from cohort-derived legal landmarks.",
            remediation="Regenerate the synthetic artifacts from one unchanged cohort.",
        )


def _bound_synthetic_cohort(config: StageWorldConfig, source: Mapping[str, Any]) -> Cohort:
    root = synthetic_data_root(config)
    try:
        manifest = read_json(root / "manifest.json")
        cohort_payload = read_json(root / "cohort.json")
    except (OSError, ValueError) as error:
        raise ArtifactError(
            code="SYNTHETIC_COHORT_BINDING_MISSING",
            message="Synthetic source validation requires its manifest and cohort JSON.",
            remediation="Regenerate the synthetic artifact chain.",
        ) from error
    data_lineage_id = source.get("data_lineage_id")
    cohort_artifact_id = source.get("cohort_artifact_id")
    if (
        not isinstance(data_lineage_id, str)
        or not data_lineage_id.strip()
        or not isinstance(cohort_artifact_id, str)
        or not cohort_artifact_id.strip()
    ):
        raise ArtifactError(
            code="SYNTHETIC_COHORT_LINEAGE_MISMATCH",
            message="Synthetic source lacks its data or cohort artifact identifier.",
        )
    manifest_expected = {
        "schema_version": SOURCE_SCHEMA,
        "data_lineage_id": data_lineage_id,
        "cohort_artifact_id": cohort_artifact_id,
        "seed": source.get("seed"),
        "timeline_contract": source.get("timeline_contract"),
        "timeline_contract_version": source.get("timeline_contract_version"),
        "outcome_contract": source.get("outcome_contract"),
        "outcome_contract_version": source.get("outcome_contract_version"),
        "cohort_file": "cohort.json",
        "source_tensor_file": "source_tensors.pt",
    }
    if any(manifest.get(key) != value for key, value in manifest_expected.items()):
        raise ArtifactError(
            code="SYNTHETIC_SOURCE_MANIFEST_MISMATCH",
            message="Synthetic manifest and source tensor contracts do not match.",
        )
    lineage = cohort_payload.get("synthetic_lineage")
    lineage_expected = {
        "schema_version": SYNTHETIC_COHORT_LINEAGE_SCHEMA,
        "source_schema_version": SOURCE_SCHEMA,
        "data_lineage_id": data_lineage_id,
        "cohort_artifact_id": cohort_artifact_id,
        "seed": source.get("seed"),
        "timeline_contract": source.get("timeline_contract"),
        "timeline_contract_version": source.get("timeline_contract_version"),
        "outcome_contract": source.get("outcome_contract"),
        "outcome_contract_version": source.get("outcome_contract_version"),
    }
    if not isinstance(lineage, Mapping) or any(
        lineage.get(key) != value for key, value in lineage_expected.items()
    ):
        raise ArtifactError(
            code="SYNTHETIC_COHORT_LINEAGE_MISMATCH",
            message="Synthetic cohort, manifest, and source lineage identifiers do not match.",
        )
    try:
        cohort = cohort_from_dict(cohort_payload)
    except (TypeError, ValueError, DataContractError) as error:
        raise ArtifactError(
            code="SYNTHETIC_COHORT_BINDING_MISSING",
            message="The bound synthetic cohort payload is invalid.",
        ) from error
    patient_ids = _source_patient_ids(source)
    if tuple(patient.patient_id for patient in cohort.patients) != patient_ids:
        raise ArtifactError(
            code="SYNTHETIC_SOURCE_COHORT_MISMATCH",
            message="Synthetic cohort and source tensor patient order differs.",
        )
    _validate_synthetic_cohort_timeline(cohort, patient_ids)
    _validate_survival_targets(source, cohort)
    return cohort


def _signal_tensor(
    patient_count: int, raw_dim: int, generator: torch.Generator
) -> dict[str, Tensor]:
    severity = torch.linspace(-1.25, 1.25, patient_count)
    token_axis = torch.arange(5, dtype=torch.float32)[None, :, None]
    feature_axis = torch.arange(raw_dim, dtype=torch.float32)[None, None, :]

    def noise(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=generator) * 0.03

    ct0 = torch.sin(severity[:, None, None] + token_axis * 0.31 + feature_axis * 0.17)
    ct0 = ct0 + noise(patient_count, 5, raw_dim)
    treatment_effect = (-0.35 + 0.12 * severity)[:, None, None]
    ct1 = torch.tanh(ct0[:, :4] + treatment_effect + feature_axis[:, :, :] * 0.01) + noise(
        patient_count, 4, raw_dim
    )
    pathology = torch.sin(
        ct1.mean(dim=1, keepdim=True)[:, :, :raw_dim]
        + torch.arange(3, dtype=torch.float32)[None, :, None] * 0.21
    ) + noise(patient_count, 3, raw_dim)
    return {"severity": severity, "ct0_raw": ct0, "ct1_raw": ct1, "pathology_raw": pathology}


def make_synthetic_artifacts(
    config: StageWorldConfig,
    *,
    out: Path | None = None,
    patient_count: int = 12,
) -> dict[str, Any]:
    """Create a fictional cohort plus raw numeric sources; no clinical rows are used."""

    _require_synthetic(config, "make-synthetic")
    if patient_count < 8:
        raise ConfigurationError(
            code="SYNTHETIC_COHORT_TOO_SMALL",
            message="At least eight fictional patients are required for edge cases.",
        )
    root = out or synthetic_data_root(config)
    root.mkdir(parents=True, exist_ok=True)
    cohort = generate_synthetic_cohort(patient_count=patient_count, seed=config.training.seed)
    patient_ids = tuple(patient.patient_id for patient in cohort.patients)
    _validate_synthetic_cohort_timeline(cohort, patient_ids)
    survival_durations, survival_events, survival_valid = _synthetic_survival_targets(
        cohort, patient_ids
    )
    data_lineage_id = new_artifact_id("synthetic-data")
    cohort_artifact_id = new_artifact_id("synthetic-cohort")
    cohort_path = root / "cohort.json"
    atomic_write_json(
        cohort_path,
        {
            **cohort_to_dict(cohort),
            "synthetic_lineage": {
                "schema_version": SYNTHETIC_COHORT_LINEAGE_SCHEMA,
                "source_schema_version": SOURCE_SCHEMA,
                "data_lineage_id": data_lineage_id,
                "cohort_artifact_id": cohort_artifact_id,
                "seed": config.training.seed,
                "timeline_contract": asdict(SYNTHETIC_TIMELINE),
                "timeline_contract_version": SYNTHETIC_TIMELINE_CONTRACT_VERSION,
                "outcome_contract": _synthetic_outcome_contract(),
                "outcome_contract_version": SYNTHETIC_OUTCOME_CONTRACT_VERSION,
            },
        },
    )

    generator = torch.Generator().manual_seed(config.training.seed)
    raw_dim = 6
    values = _signal_tensor(patient_count, raw_dim, generator)
    ct1_valid = torch.ones(patient_count, 4, dtype=torch.bool)
    ct1_valid[5] = False
    pathology_valid = torch.ones(patient_count, 3, dtype=torch.bool)
    pathology_valid[2:5] = False
    pathology_valid[::3, -1] = False
    clinical_axis = torch.arange(config.model.clinical_input_dim, dtype=torch.float32)
    clinical = torch.stack(
        [
            torch.sin(values["severity"][:, None] + clinical_axis[None] * 0.11),
            torch.cos(values["severity"][:, None] * 0.5 + clinical_axis[None] * 0.07),
            torch.tanh(values["severity"][:, None] - clinical_axis[None] * 0.03),
            torch.sin(values["severity"][:, None] * 0.7 - clinical_axis[None] * 0.05),
        ],
        dim=1,
    )
    action_axis = torch.arange(config.model.action_input_dim, dtype=torch.float32)
    treatment_actions = torch.stack(
        [
            torch.sin(values["severity"][:, None] + action_axis[None] * 0.13),
            torch.cos(values["severity"][:, None] * 0.4 + action_axis[None] * 0.09),
        ],
        dim=1,
    )
    surgery_actions = torch.tanh(
        values["severity"][:, None, None] * 0.2 + action_axis[None, None] * 0.04
    )
    treatment_valid = torch.zeros(patient_count, 2, dtype=torch.bool)
    treatment_valid[:, 0] = True
    treatment_valid[0, 1] = True
    surgery_valid = torch.ones(patient_count, 1, dtype=torch.bool)
    surgery_valid[3] = False

    source_path = root / "source_tensors.pt"
    _atomic_torch_save(
        source_path,
        {
            "schema_version": SOURCE_SCHEMA,
            "data_lineage_id": data_lineage_id,
            "cohort_artifact_id": cohort_artifact_id,
            "seed": config.training.seed,
            "timeline_contract": asdict(SYNTHETIC_TIMELINE),
            "timeline_contract_version": SYNTHETIC_TIMELINE_CONTRACT_VERSION,
            "outcome_contract": _synthetic_outcome_contract(),
            "outcome_contract_version": SYNTHETIC_OUTCOME_CONTRACT_VERSION,
            "patient_ids": patient_ids,
            **values,
            "ct0_valid": torch.ones(patient_count, 5, dtype=torch.bool),
            "ct1_valid": ct1_valid,
            "pathology_valid": pathology_valid,
            "clinical": clinical,
            "clinical_fields": (
                "age_years",
                "baseline_stage",
                "radiologic_response",
                "yp_stage",
            ),
            "treatment_actions": treatment_actions,
            "treatment_valid": treatment_valid,
            "treatment_action_roles": ("delivered_systemic", "post_ct_other"),
            "treatment_event_time": torch.tensor(
                [SYNTHETIC_TIMELINE.systemic_event, SYNTHETIC_TIMELINE.post_ct_event]
            ).repeat(patient_count, 1),
            "treatment_available_time": torch.tensor(
                [
                    SYNTHETIC_TIMELINE.systemic_available,
                    SYNTHETIC_TIMELINE.post_ct_available,
                ]
            ).repeat(patient_count, 1),
            "treatment_event_type": torch.tensor(
                [
                    TREATMENT_KIND_ID[TreatmentKind.SYSTEMIC],
                    TREATMENT_KIND_ID[TreatmentKind.OTHER],
                ]
            ).repeat(patient_count, 1),
            "treatment_status": torch.full(
                (patient_count, 2),
                TREATMENT_STATUS_ID[TreatmentStatus.DELIVERED],
            ),
            "surgery_actions": surgery_actions,
            "surgery_valid": surgery_valid,
            "surgery_event_time": torch.full((patient_count, 1), SYNTHETIC_TIMELINE.surgery_event),
            "surgery_available_time": torch.full(
                (patient_count, 1), SYNTHETIC_TIMELINE.surgery_available
            ),
            "surgery_event_type": torch.full(
                (patient_count, 1), TREATMENT_KIND_ID[TreatmentKind.SURGERY]
            ),
            "surgery_status": torch.full(
                (patient_count, 1), TREATMENT_STATUS_ID[TreatmentStatus.DELIVERED]
            ),
            "survival_durations": survival_durations,
            "survival_events": survival_events,
            "survival_valid": survival_valid,
            "survival_endpoint": "os",
            "survival_label_version": "synthetic-os-v1",
            "survival_time_unit": _SURVIVAL_TIME_UNIT,
            "survival_stage_order": tuple(stage.value for stage in _SURVIVAL_STAGE_ORDER),
            "survival_query_ids": tuple(
                tuple(f"{patient_id}-{stage.value}" for stage in _SURVIVAL_STAGE_ORDER)
                for patient_id in patient_ids
            ),
        },
    )
    manifest_path = root / "manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": SOURCE_SCHEMA,
            "artifact_kind": "explicitly_synthetic",
            "data_lineage_id": data_lineage_id,
            "cohort_artifact_id": cohort_artifact_id,
            "seed": config.training.seed,
            "patient_count": patient_count,
            "timeline_contract": asdict(SYNTHETIC_TIMELINE),
            "timeline_contract_version": SYNTHETIC_TIMELINE_CONTRACT_VERSION,
            "outcome_contract": _synthetic_outcome_contract(),
            "outcome_contract_version": SYNTHETIC_OUTCOME_CONTRACT_VERSION,
            "cohort_file": cohort_path.name,
            "source_tensor_file": source_path.name,
            "contains_real_clinical_data": False,
        },
    )
    return {
        "status": "ok",
        "mode": "synthetic",
        "patient_count": patient_count,
        "data_lineage_id": data_lineage_id,
        "cohort_artifact_id": cohort_artifact_id,
        "cohort": str(cohort_path),
        "source_tensors": str(source_path),
        "manifest": str(manifest_path),
    }


def validate_synthetic_source_config(config: StageWorldConfig, source: Mapping[str, Any]) -> None:
    if source.get("schema_version") != SOURCE_SCHEMA:
        raise ArtifactError(
            code="SYNTHETIC_ARTIFACT_SCHEMA_MISMATCH",
            message="Synthetic source schema is absent or incompatible.",
            details={"expected": SOURCE_SCHEMA},
        )
    seed = source.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed != config.training.seed:
        raise ArtifactError(
            code="SYNTHETIC_SOURCE_CONFIG_MISMATCH",
            message="Synthetic source seed differs from the active configuration.",
            remediation="Regenerate the synthetic chain with one unchanged configuration.",
        )
    if source.get("timeline_contract") != asdict(SYNTHETIC_TIMELINE):
        raise ArtifactError(
            code="SYNTHETIC_TIMELINE_CONTRACT_MISMATCH",
            message="Synthetic source clocks differ from the active workflow contract.",
            remediation="Regenerate the synthetic chain with the current source schema.",
        )
    if source.get("timeline_contract_version") != SYNTHETIC_TIMELINE_CONTRACT_VERSION:
        raise ArtifactError(
            code="SYNTHETIC_TIMELINE_CONTRACT_MISMATCH",
            message="Synthetic source has no supported timeline contract version.",
            remediation="Regenerate the synthetic chain with the current source schema.",
        )
    if (
        source.get("outcome_contract") != _synthetic_outcome_contract()
        or source.get("outcome_contract_version") != SYNTHETIC_OUTCOME_CONTRACT_VERSION
    ):
        raise ArtifactError(
            code="SYNTHETIC_OUTCOME_CONTRACT_MISMATCH",
            message="Synthetic source outcome definition differs from the active OS contract.",
            remediation="Regenerate the synthetic chain with the current source schema.",
        )
    _bound_synthetic_cohort(config, source)


def _canonical_split_version(seed: int) -> str:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ArtifactError(
            code="COHORT_SPLIT_POLICY_MISMATCH",
            message="Synthetic split seed must be a nonnegative integer.",
        )
    return (
        f"{SYNTHETIC_SPLIT_POLICY_VERSION}_build-{BUILD_SCHEMA}_seed-{seed}_"
        "train-0p6_validation-0p2"
    )


def _canonical_split_assignments(
    patient_ids: tuple[str, ...], seed: int
) -> tuple[SplitAssignment, ...]:
    return SplitManager(seed=seed).assign(
        patient_ids,
        train_fraction=_SYNTHETIC_TRAIN_FRACTION,
        validation_fraction=_SYNTHETIC_VALIDATION_FRACTION,
    )


def _split_assignment_payload(
    assignments: tuple[SplitAssignment, ...],
) -> list[dict[str, str | int | None]]:
    return [
        {"patient_id": item.patient_id, "split": item.split.value, "fold": item.fold}
        for item in assignments
    ]


def _validate_synthetic_split_build(
    config: StageWorldConfig,
    source: Mapping[str, Any],
    build: Mapping[str, Any],
    *,
    data_lineage_id: str | None = None,
) -> tuple[tuple[SplitAssignment, ...], str]:
    if build.get("schema_version") != BUILD_SCHEMA:
        raise ArtifactError(
            code="COHORT_BUILD_SCHEMA_MISMATCH",
            message="Run build-cohort with the current schema before training.",
        )
    expected_lineage = (
        str(source.get("data_lineage_id")) if data_lineage_id is None else data_lineage_id
    )
    if build.get("data_lineage_id") != expected_lineage:
        raise ArtifactError(
            code="COHORT_BUILD_DATA_LINEAGE_MISMATCH",
            message="Cohort split and source tensors do not share a data lineage.",
        )
    if build.get("cohort_artifact_id") != source.get("cohort_artifact_id"):
        raise ArtifactError(
            code="COHORT_BUILD_COHORT_LINEAGE_MISMATCH",
            message="Cohort split and source tensors do not share a cohort artifact.",
        )
    if (
        build.get("timeline_contract") != source.get("timeline_contract")
        or build.get("timeline_contract_version") != source.get("timeline_contract_version")
        or build.get("outcome_contract") != source.get("outcome_contract")
        or build.get("outcome_contract_version") != source.get("outcome_contract_version")
    ):
        raise ArtifactError(
            code="COHORT_BUILD_CONTRACT_LINEAGE_MISMATCH",
            message="Cohort split, timeline, and outcome contracts are not from one source.",
        )

    patient_ids = _source_patient_ids(source)
    expected_assignments = _canonical_split_assignments(patient_ids, config.training.seed)
    expected_version = _canonical_split_version(config.training.seed)
    expected_counts = {name.value: 0 for name in SplitName}
    for assignment in expected_assignments:
        expected_counts[assignment.split.value] += 1
    policy_fields: dict[str, object] = {
        "mode": "synthetic",
        "split_version": expected_version,
        "split_policy_version": SYNTHETIC_SPLIT_POLICY_VERSION,
        "split_seed": config.training.seed,
        "train_fraction": _SYNTHETIC_TRAIN_FRACTION,
        "validation_fraction": _SYNTHETIC_VALIDATION_FRACTION,
        "patient_count": len(patient_ids),
        "split_counts": expected_counts,
        "assignments": _split_assignment_payload(expected_assignments),
        "contains_real_clinical_data": False,
    }
    changed = sorted(
        field for field, expected in policy_fields.items() if build.get(field) != expected
    )
    if changed:
        raise ArtifactError(
            code="COHORT_SPLIT_POLICY_MISMATCH",
            message=(
                "Cohort split policy or assignments differ from the deterministic "
                "patient-level split."
            ),
            details={"fields": changed},
        )
    return expected_assignments, expected_version


def build_synthetic_cohort(config: StageWorldConfig) -> dict[str, Any]:
    _require_synthetic(config, "build-cohort")
    root = synthetic_data_root(config)
    source = _load_tensor_artifact(root / "source_tensors.pt", SOURCE_SCHEMA)
    validate_synthetic_source_config(config, source)
    cohort = load_cohort_json(root / "cohort.json")
    patient_ids = _source_patient_ids(source)
    canonical_queries = _canonical_stage_queries(cohort, patient_ids)
    landmarks = _build_synthetic_landmarks(cohort, canonical_queries)
    audit_landmarks = _build_synthetic_landmarks(cohort, cohort.queries)
    assignments = _canonical_split_assignments(patient_ids, config.training.seed)
    SplitManager.assert_records_assigned(landmarks.landmarks, assignments)
    stage_counts = {stage: 0 for stage in ("s0", "s1", "s2")}
    stage_events = {stage: 0 for stage in stage_counts}
    for landmark in landmarks.landmarks:
        stage = landmark.stage.value
        stage_counts[stage] += 1
        stage_events[stage] += int(landmark.label.event)
    source_valid = torch.as_tensor(source["survival_valid"])
    source_events = torch.as_tensor(source["survival_events"])
    expected_stage_counts = {
        stage.value: int(source_valid[:, index].sum())
        for index, stage in enumerate(_SURVIVAL_STAGE_ORDER)
    }
    expected_stage_events = {
        stage.value: int((source_events[:, index] * source_valid[:, index]).sum())
        for index, stage in enumerate(_SURVIVAL_STAGE_ORDER)
    }
    if stage_counts != expected_stage_counts or stage_events != expected_stage_events:
        raise ArtifactError(
            code="SYNTHETIC_SURVIVAL_LABEL_MISMATCH",
            message="Cohort-build audit counts differ from canonical source OS tensors.",
        )
    split_counts = {name: 0 for name in ("train", "validation", "test")}
    for assignment in assignments:
        split_counts[assignment.split.value] += 1
    build_path = root / "cohort_build.json"
    atomic_write_json(
        build_path,
        {
            "schema_version": BUILD_SCHEMA,
            "mode": "synthetic",
            "data_lineage_id": str(source["data_lineage_id"]),
            "cohort_artifact_id": str(source["cohort_artifact_id"]),
            "timeline_contract": source["timeline_contract"],
            "timeline_contract_version": source["timeline_contract_version"],
            "outcome_contract": source["outcome_contract"],
            "outcome_contract_version": source["outcome_contract_version"],
            "split_version": _canonical_split_version(config.training.seed),
            "split_policy_version": SYNTHETIC_SPLIT_POLICY_VERSION,
            "split_seed": config.training.seed,
            "train_fraction": _SYNTHETIC_TRAIN_FRACTION,
            "validation_fraction": _SYNTHETIC_VALIDATION_FRACTION,
            "patient_count": len(cohort.patients),
            "stage_landmark_counts": stage_counts,
            "stage_event_counts": stage_events,
            "exclusion_count": len(audit_landmarks.exclusions),
            "exclusions_by_reason": {
                reason: sum(item.reason == reason for item in audit_landmarks.exclusions)
                for reason in sorted({item.reason for item in audit_landmarks.exclusions})
            },
            "split_counts": split_counts,
            "assignments": _split_assignment_payload(assignments),
            "contains_real_clinical_data": False,
        },
    )
    return {
        "status": "ok",
        "mode": "synthetic",
        "data_lineage_id": str(source["data_lineage_id"]),
        "cohort_artifact_id": str(source["cohort_artifact_id"]),
        "patient_count": len(cohort.patients),
        "stage_landmark_counts": stage_counts,
        "stage_event_counts": stage_events,
        "exclusion_count": len(audit_landmarks.exclusions),
        "split_counts": split_counts,
        "artifact": str(build_path),
    }


def _source_ids(
    modality: str,
    valid: Tensor,
    patient_ids: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    if valid.ndim != 2 or valid.shape[0] != len(patient_ids):
        raise ArtifactError(
            code="SYNTHETIC_FEATURE_PATIENT_MISMATCH",
            message="Synthetic token masks must have one row per ordered patient.",
        )
    return tuple(
        tuple(
            (f"synthetic-{modality}-{patient_id}-{token}" if bool(valid[row, token]) else "")
            for token in range(valid.shape[1])
        )
        for row, patient_id in enumerate(patient_ids)
    )


def _encode_synthetic_feature_payloads(
    config: StageWorldConfig,
    source: Mapping[str, Any],
    modality: Literal["ct", "pathology"],
) -> tuple[SyntheticEncoder, dict[str, dict[str, Any]]]:
    patient_ids = _source_patient_ids(source)
    if modality == "ct":
        encoder = SyntheticEncoder(
            mode=config.mode,
            input_dim=int(source["ct0_raw"].shape[-1]),
            feature_dim=config.model.ct_input_dim,
        )
        results: dict[str, Any] = {}
        for name, acquired, available in (
            (
                "ct0",
                SYNTHETIC_TIMELINE.baseline_acquired,
                SYNTHETIC_TIMELINE.s0_query,
            ),
            (
                "ct1",
                SYNTHETIC_TIMELINE.ct1_acquired,
                SYNTHETIC_TIMELINE.ct1_available,
            ),
        ):
            raw = source[f"{name}_raw"]
            valid = source[f"{name}_valid"]
            batch, tokens, _ = raw.shape
            coords = (
                torch.stack(
                    torch.meshgrid(
                        torch.linspace(0, 1, tokens),
                        torch.zeros(1),
                        torch.zeros(1),
                        indexing="ij",
                    ),
                    dim=-1,
                )
                .reshape(1, tokens, 3)
                .repeat(batch, 1, 1)
            )
            encoded = encoder.encode(
                raw,
                valid=valid,
                modality=torch.zeros(batch, tokens, dtype=torch.long),
                acquired_time=torch.full((batch, tokens), acquired),
                available_time=torch.full((batch, tokens), available),
                source_id=_source_ids(name, valid, patient_ids),
                modality_name="ct",
                coords=coords,
                coordinate_system="synthetic_normalized_xyz",
            )
            results[name] = encoded.observations.as_cache_payload()
        return encoder, results

    encoder = SyntheticEncoder(
        mode=config.mode,
        input_dim=int(source["pathology_raw"].shape[-1]),
        feature_dim=config.model.pathology_input_dim,
    )
    raw = source["pathology_raw"]
    valid = source["pathology_valid"]
    batch, tokens, _ = raw.shape
    coords = (
        torch.stack((torch.arange(tokens), torch.zeros(tokens)), dim=-1)
        .float()[None]
        .repeat(batch, 1, 1)
    )
    encoded = encoder.encode(
        raw,
        valid=valid,
        modality=torch.ones(batch, tokens, dtype=torch.long),
        acquired_time=torch.full((batch, tokens), SYNTHETIC_TIMELINE.pathology_acquired),
        available_time=torch.full((batch, tokens), SYNTHETIC_TIMELINE.pathology_available),
        source_id=_source_ids("pathology", valid, patient_ids),
        modality_name="pathology",
        coords=coords,
        coordinate_system="synthetic_level0_xy",
    )
    return encoder, {"pathology": encoded.observations.as_cache_payload()}


def _synthetic_feature_payload(
    config: StageWorldConfig,
    source: Mapping[str, Any],
    modality: Literal["ct", "pathology"],
    *,
    feature_artifact_id: str,
) -> dict[str, Any]:
    encoder, encoded_payloads = _encode_synthetic_feature_payloads(config, source, modality)
    return {
        "schema_version": FEATURE_SCHEMA,
        "artifact_kind": "explicitly_synthetic_features",
        "mode": "synthetic",
        "feature_artifact_id": feature_artifact_id,
        "data_lineage_id": source["data_lineage_id"],
        "cohort_artifact_id": source["cohort_artifact_id"],
        "patient_ids": _source_patient_ids(source),
        "timeline_contract": source["timeline_contract"],
        "timeline_contract_version": source["timeline_contract_version"],
        "outcome_contract": source["outcome_contract"],
        "outcome_contract_version": source["outcome_contract_version"],
        "modality": modality,
        "encoder": encoder.provenance.as_dict(),
        **encoded_payloads,
    }


def validate_synthetic_feature_artifacts(
    config: StageWorldConfig,
    source: Mapping[str, Any],
    ct: Mapping[str, Any],
    pathology: Mapping[str, Any],
) -> tuple[str, str]:
    """Bind deterministic synthetic token rows to source patients and encoders."""

    artifact_ids: list[str] = []
    lineage_fields = {
        "data_lineage_id": source.get("data_lineage_id"),
        "cohort_artifact_id": source.get("cohort_artifact_id"),
        "timeline_contract": source.get("timeline_contract"),
        "timeline_contract_version": source.get("timeline_contract_version"),
        "outcome_contract": source.get("outcome_contract"),
        "outcome_contract_version": source.get("outcome_contract_version"),
    }
    for modality, artifact in (("ct", ct), ("pathology", pathology)):
        changed_lineage = sorted(
            field for field, expected in lineage_fields.items() if artifact.get(field) != expected
        )
        if changed_lineage:
            raise ArtifactError(
                code="FEATURE_DATA_LINEAGE_MISMATCH",
                message=(
                    "CT, pathology, and source tensors do not share one cohort, timeline, "
                    "and outcome contract."
                ),
                details={"modality": modality, "fields": changed_lineage},
            )
        artifact_id = artifact.get("feature_artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise ArtifactError(
                code="FEATURE_ARTIFACT_ID_MISSING",
                message="Every synthetic feature artifact requires an immutable artifact ID.",
                details={"modality": modality},
            )
        try:
            expected = _synthetic_feature_payload(
                config,
                source,
                cast(Literal["ct", "pathology"], modality),
                feature_artifact_id=artifact_id,
            )
        except (KeyError, TypeError, ValueError, RuntimeError, DataContractError) as error:
            raise ArtifactError(
                code="SYNTHETIC_FEATURE_SOURCE_INVALID",
                message="Synthetic source tensors cannot reproduce the declared features.",
                details={"modality": modality},
            ) from error
        mismatched = checkpoint_payload_mismatches(
            expected,
            artifact,
            path=f"{modality}_feature",
        )
        if mismatched:
            raise ArtifactError(
                code="SYNTHETIC_FEATURE_ARTIFACT_MISMATCH",
                message=(
                    "Synthetic feature tensors, patient rows, modality, or encoder provenance "
                    "differ from their deterministic source contract."
                ),
                details={"fields": list(mismatched)},
            )
        artifact_ids.append(artifact_id)
    if artifact_ids[0] == artifact_ids[1]:
        raise ArtifactError(
            code="FEATURE_ARTIFACT_ID_COLLISION",
            message="CT and pathology feature artifacts require distinct lineage IDs.",
        )
    return artifact_ids[0], artifact_ids[1]


def extract_synthetic_features(
    config: StageWorldConfig, modality: Literal["ct", "pathology"]
) -> dict[str, Any]:
    _require_synthetic(config, "extract-features")
    source = _load_tensor_artifact(synthetic_data_root(config) / "source_tensors.pt", SOURCE_SCHEMA)
    validate_synthetic_source_config(config, source)
    root = synthetic_feature_root(config)
    root.mkdir(parents=True, exist_ok=True)
    feature_artifact_id = new_artifact_id(f"synthetic-{modality}-features")
    payload = _synthetic_feature_payload(
        config,
        source,
        modality,
        feature_artifact_id=feature_artifact_id,
    )
    encoder = SyntheticEncoder(
        mode=config.mode,
        input_dim=int(
            source["ct0_raw"].shape[-1] if modality == "ct" else source["pathology_raw"].shape[-1]
        ),
        feature_dim=(
            config.model.ct_input_dim if modality == "ct" else config.model.pathology_input_dim
        ),
    )
    destination = root / f"{modality}.pt"
    _atomic_torch_save(destination, payload)
    return {
        "status": "ok",
        "mode": "synthetic",
        "modality": modality,
        "data_lineage_id": str(source["data_lineage_id"]),
        "feature_artifact_id": feature_artifact_id,
        "encoder": encoder.provenance.encoder_name,
        "feature_dim": encoder.provenance.feature_dim,
        "artifact": str(destination),
    }


def build_model(config: StageWorldConfig) -> StageWorldModel:
    return StageWorldModel(
        StageWorldModelConfig(
            hidden_dim=config.model.hidden_dim,
            state_tokens=config.model.state_tokens,
            stochastic_dim=config.model.stochastic_dim_per_token,
            use_stochastic_state=config.model.use_stochastic_state,
            attention_heads=config.model.attention_heads,
            transition_blocks=config.model.transition_blocks,
            observation_blocks=config.model.observation_blocks,
            resampler_blocks=config.model.resampler_blocks,
            dropout=config.model.dropout,
            action_input_dim=config.model.action_input_dim,
            modality_input_dims=(
                ("ct", config.model.ct_input_dim),
                ("pathology", config.model.pathology_input_dim),
                ("clinical", config.model.clinical_input_dim),
            ),
            resampled_tokens=(
                ("ct", config.model.ct_tokens),
                # The S0/S1 real branch keeps one explicitly masked placeholder token
                # because the shared three-stage module still owns a disabled S2 path.
                ("pathology", max(1, config.model.pathology_tokens)),
                ("clinical", min(4, config.model.state_tokens)),
            ),
            future_output_dims=(
                ("ct", config.model.ct_input_dim),
                ("pathology", config.model.pathology_input_dim),
            ),
            future_output_tokens=(("ct", 1), ("pathology", 1)),
            survival_cutpoints=config.survival.finite_cutpoints,
            survival_causes=config.survival.num_causes,
            model_version=f"stageworld-gc-{config.project.schema_version}",
        )
    )


def _slice_observation(value: ObservationTokens, indices: Tensor) -> ObservationTokens:
    rows = indices.tolist()
    return ObservationTokens(
        values=value.values[indices],
        valid=value.valid[indices],
        modality=value.modality[indices],
        acquired_time=value.acquired_time[indices],
        available_time=value.available_time[indices],
        provenance=value.provenance,
        source_id=tuple(value.source_id[index] for index in rows),
        modality_name=value.modality_name,
        coords=None if value.coords is None else value.coords[indices],
        coordinate_system=value.coordinate_system,
        quality_flags=(
            tuple(value.quality_flags[index] for index in rows) if value.quality_flags else ()
        ),
    )


def _select_observation_tokens(
    value: ObservationTokens, token_indices: tuple[int, ...]
) -> ObservationTokens:
    indices = torch.tensor(token_indices, dtype=torch.long, device=value.values.device)
    rows = indices.tolist()
    return ObservationTokens(
        values=value.values[:, indices],
        valid=value.valid[:, indices],
        modality=value.modality[:, indices],
        acquired_time=value.acquired_time[:, indices],
        available_time=value.available_time[:, indices],
        provenance=value.provenance,
        source_id=tuple(tuple(row[index] for index in rows) for row in value.source_id),
        modality_name=value.modality_name,
        coords=None if value.coords is None else value.coords[:, indices],
        coordinate_system=value.coordinate_system,
        quality_flags=value.quality_flags,
    )


def _select_action_tokens(value: ActionTokens, token_index: int) -> ActionTokens:
    selection = slice(token_index, token_index + 1)
    return ActionTokens(
        values=value.values[:, selection],
        valid=value.valid[:, selection],
        event_time=value.event_time[:, selection],
        available_time=value.available_time[:, selection],
        event_type=None if value.event_type is None else value.event_type[:, selection],
        planned_or_delivered=(
            None if value.planned_or_delivered is None else value.planned_or_delivered[:, selection]
        ),
        known_exposure=(
            None if value.known_exposure is None else value.known_exposure[:, selection]
        ),
        provenance=value.provenance,
    )


def _pooled_target(observation: ObservationTokens) -> tuple[Tensor, Tensor]:
    count = observation.valid.sum(dim=1, keepdim=True)
    valid = count > 0
    target = (observation.values * observation.valid[..., None]).sum(
        dim=1, keepdim=True
    ) / count.clamp_min(1)[..., None]
    return target.detach(), valid


def _typed_unavailable_event_mask(
    cohort: Cohort,
    patient_ids: tuple[str, ...],
    role: ObservationRole,
) -> Tensor:
    """Mark recorded missing/failed acquisitions without conflating absent stages."""

    row_by_patient = {patient_id: row for row, patient_id in enumerate(patient_ids)}
    mask = torch.zeros(len(patient_ids), dtype=torch.bool)
    typed_missing = {MissingCategory.MISSING, MissingCategory.FAILED_QC}
    for observation in cohort.observations:
        if observation.role is not role:
            continue
        if (
            observation.quality_status is QualityStatus.FAILED
            or observation.missing_reason in typed_missing
        ):
            try:
                mask[row_by_patient[observation.patient_id]] = True
            except KeyError as error:
                raise ArtifactError(
                    code="COHORT_FEATURE_PATIENT_MISMATCH",
                    message="An unavailable observation event references an unknown patient.",
                ) from error
    return mask


def load_synthetic_batches(
    config: StageWorldConfig,
) -> tuple[dict[SplitName, tuple[WorldModelBatch, ...]], str]:
    _require_synthetic(config, "training")
    source = _load_tensor_artifact(synthetic_data_root(config) / "source_tensors.pt", SOURCE_SCHEMA)
    validate_synthetic_source_config(config, source)
    ct = _load_tensor_artifact(synthetic_feature_root(config) / "ct.pt", FEATURE_SCHEMA)
    pathology = _load_tensor_artifact(
        synthetic_feature_root(config) / "pathology.pt", FEATURE_SCHEMA
    )
    lineage = str(source["data_lineage_id"])
    validate_synthetic_feature_artifacts(config, source, ct, pathology)
    ct0 = ObservationTokens.from_cache_payload(ct["ct0"])
    ct1 = ObservationTokens.from_cache_payload(ct["ct1"])
    path = ObservationTokens.from_cache_payload(pathology["pathology"])
    patient_ids = tuple(str(item) for item in source["patient_ids"])
    cohort = load_cohort_json(synthetic_data_root(config) / "cohort.json")
    if {patient.patient_id for patient in cohort.patients} != set(patient_ids):
        raise ArtifactError(
            code="COHORT_FEATURE_PATIENT_MISMATCH",
            message="Synthetic cohort and feature patient sets differ.",
        )
    ct1_unavailable_event_mask = _typed_unavailable_event_mask(
        cohort, patient_ids, ObservationRole.POST_TREATMENT_CT
    )
    pathology_unavailable_event_mask = _typed_unavailable_event_mask(
        cohort, patient_ids, ObservationRole.SURGICAL_PATHOLOGY
    )
    build = read_json(synthetic_data_root(config) / "cohort_build.json")
    assignments, _ = _validate_synthetic_split_build(
        config,
        source,
        build,
        data_lineage_id=lineage,
    )
    split_by_patient = {item.patient_id: item.split for item in assignments}
    clinical_values = torch.as_tensor(source["clinical"]).float()
    expected_clinical_fields = (
        "age_years",
        "baseline_stage",
        "radiologic_response",
        "yp_stage",
    )
    if (
        clinical_values.shape[1] != len(expected_clinical_fields)
        or tuple(source.get("clinical_fields", ())) != expected_clinical_fields
    ):
        raise ArtifactError(
            code="SYNTHETIC_CLINICAL_SCHEMA_MISMATCH",
            message="Synthetic clinical features do not match the time-scoped field contract.",
        )
    clinical_valid = torch.ones(clinical_values.shape[:2], dtype=torch.bool)
    clinical_valid[:, 3] = torch.as_tensor(source["surgery_valid"]).bool().squeeze(1)
    clinical_acquired = torch.tensor(
        (
            SYNTHETIC_TIMELINE.baseline_acquired,
            SYNTHETIC_TIMELINE.baseline_acquired,
            SYNTHETIC_TIMELINE.ct1_acquired,
            SYNTHETIC_TIMELINE.pathology_acquired,
        )
    ).repeat(len(patient_ids), 1)
    clinical_available = torch.tensor(
        (
            SYNTHETIC_TIMELINE.s0_query,
            SYNTHETIC_TIMELINE.s0_query,
            SYNTHETIC_TIMELINE.ct1_available,
            SYNTHETIC_TIMELINE.pathology_available,
        )
    ).repeat(len(patient_ids), 1)
    clinical = ObservationTokens(
        values=clinical_values,
        valid=clinical_valid,
        modality=torch.full(clinical_valid.shape, 2, dtype=torch.long),
        acquired_time=clinical_acquired,
        available_time=clinical_available,
        provenance=EncoderProvenance(
            encoder_name="synthetic_structured_clinical",
            source_version="synthetic-v1",
            component_versions=(("field_tokenizer", "analytic-v1"),),
            preprocess_version="identity-v1",
            feature_dim=config.model.clinical_input_dim,
        ),
        source_id=_source_ids("clinical", clinical_valid, patient_ids),
        modality_name="clinical",
    )
    clinical0 = _select_observation_tokens(clinical, (0, 1))
    clinical1 = _select_observation_tokens(clinical, (2,))
    clinical2 = _select_observation_tokens(clinical, (3,))
    if tuple(source.get("treatment_action_roles", ())) != (
        "delivered_systemic",
        "post_ct_other",
    ):
        raise ArtifactError(
            code="SYNTHETIC_ACTION_SCHEMA_MISMATCH",
            message="Synthetic treatment slots do not match the replay contract.",
        )
    treatment = ActionTokens(
        values=torch.as_tensor(source["treatment_actions"]).float(),
        valid=torch.as_tensor(source["treatment_valid"]).bool(),
        event_time=torch.as_tensor(source["treatment_event_time"]).float(),
        available_time=torch.as_tensor(source["treatment_available_time"]).float(),
        event_type=torch.as_tensor(source["treatment_event_type"]).long(),
        planned_or_delivered=torch.as_tensor(source["treatment_status"]).long(),
        known_exposure=torch.ones(len(patient_ids), 2),
        provenance=("synthetic-treatment-history-v1",),
    )
    treatment_before_ct = _select_action_tokens(treatment, 0)
    treatment_after_ct = _select_action_tokens(treatment, 1)
    surgery = ActionTokens(
        values=torch.as_tensor(source["surgery_actions"]).float(),
        valid=torch.as_tensor(source["surgery_valid"]).bool(),
        event_time=torch.as_tensor(source["surgery_event_time"]).float(),
        available_time=torch.as_tensor(source["surgery_available_time"]).float(),
        event_type=torch.as_tensor(source["surgery_event_type"]).long(),
        planned_or_delivered=torch.as_tensor(source["surgery_status"]).long(),
        known_exposure=torch.ones(len(patient_ids), 1),
        provenance=("synthetic-surgery-history-v1",),
    )
    ct_target, ct_target_valid = _pooled_target(ct1)
    path_target, path_target_valid = _pooled_target(path)
    batches: dict[SplitName, list[WorldModelBatch]] = {
        SplitName.TRAIN: [],
        SplitName.VALIDATION: [],
        SplitName.TEST: [],
    }
    size = config.training.patient_batch_size
    for split_name in SplitName:
        split_indices = [
            index
            for index, patient_id in enumerate(patient_ids)
            if split_by_patient[patient_id] is split_name
        ]
        for start in range(0, len(split_indices), size):
            indices = torch.tensor(split_indices[start : start + size], dtype=torch.long)
            rows = indices.tolist()
            batches[split_name].append(
                WorldModelBatch(
                    patient_ids=tuple(patient_ids[index] for index in rows),
                    ct0=_slice_observation(ct0, indices),
                    clinical0=_slice_observation(clinical0, indices),
                    s0_time=torch.full((len(rows),), SYNTHETIC_TIMELINE.s0_query),
                    treatment_actions=ActionTokens(
                        values=treatment_before_ct.values[indices],
                        valid=treatment_before_ct.valid[indices],
                        event_time=treatment_before_ct.event_time[indices],
                        available_time=treatment_before_ct.available_time[indices],
                        event_type=treatment_before_ct.event_type[indices]
                        if treatment_before_ct.event_type is not None
                        else None,
                        planned_or_delivered=(
                            treatment_before_ct.planned_or_delivered[indices]
                            if treatment_before_ct.planned_or_delivered is not None
                            else None
                        ),
                        known_exposure=(
                            treatment_before_ct.known_exposure[indices]
                            if treatment_before_ct.known_exposure is not None
                            else None
                        ),
                        provenance=treatment_before_ct.provenance,
                    ),
                    ct1_acquisition_time=torch.full((len(rows),), SYNTHETIC_TIMELINE.ct1_acquired),
                    ct1=_slice_observation(ct1, indices),
                    s1_time=torch.full((len(rows),), SYNTHETIC_TIMELINE.s1_query),
                    surgery_actions=ActionTokens(
                        values=surgery.values[indices],
                        valid=surgery.valid[indices],
                        event_time=surgery.event_time[indices],
                        available_time=surgery.available_time[indices],
                        event_type=surgery.event_type[indices]
                        if surgery.event_type is not None
                        else None,
                        planned_or_delivered=(
                            surgery.planned_or_delivered[indices]
                            if surgery.planned_or_delivered is not None
                            else None
                        ),
                        known_exposure=(
                            surgery.known_exposure[indices]
                            if surgery.known_exposure is not None
                            else None
                        ),
                        provenance=surgery.provenance,
                    ),
                    pathology_acquisition_time=torch.full(
                        (len(rows),), SYNTHETIC_TIMELINE.pathology_acquired
                    ),
                    pathology=_slice_observation(path, indices),
                    s2_time=torch.full((len(rows),), SYNTHETIC_TIMELINE.s2_query),
                    horizons=torch.tensor((0.0, *config.survival.report_horizons_years)),
                    future_ct_target=ct_target[indices],
                    future_ct_valid=ct_target_valid[indices],
                    future_pathology_target=path_target[indices],
                    future_pathology_valid=path_target_valid[indices],
                    survival_durations=torch.as_tensor(source["survival_durations"])[
                        indices
                    ].float(),
                    survival_events=torch.as_tensor(source["survival_events"])[indices].long(),
                    survival_valid=torch.as_tensor(source["survival_valid"])[indices].bool(),
                    ct1_availability_time=torch.full(
                        (len(rows),), SYNTHETIC_TIMELINE.ct1_available
                    ),
                    ct1_unavailable_event_mask=ct1_unavailable_event_mask[indices],
                    clinical1=_slice_observation(clinical1, indices),
                    s1_update_actions=ActionTokens(
                        values=treatment_after_ct.values[indices],
                        valid=treatment_after_ct.valid[indices],
                        event_time=treatment_after_ct.event_time[indices],
                        available_time=treatment_after_ct.available_time[indices],
                        event_type=(
                            None
                            if treatment_after_ct.event_type is None
                            else treatment_after_ct.event_type[indices]
                        ),
                        planned_or_delivered=(
                            None
                            if treatment_after_ct.planned_or_delivered is None
                            else treatment_after_ct.planned_or_delivered[indices]
                        ),
                        known_exposure=(
                            None
                            if treatment_after_ct.known_exposure is None
                            else treatment_after_ct.known_exposure[indices]
                        ),
                        provenance=treatment_after_ct.provenance,
                    ),
                    pathology_availability_time=torch.full(
                        (len(rows),), SYNTHETIC_TIMELINE.pathology_available
                    ),
                    pathology_unavailable_event_mask=pathology_unavailable_event_mask[indices],
                    clinical2=_slice_observation(clinical2, indices),
                )
            )
    empty_splits = [split.value for split, values in batches.items() if not values]
    if empty_splits:
        raise DataContractError(
            code="EMPTY_SYNTHETIC_SPLIT",
            message="Synthetic training requires nonempty train, validation, and test splits.",
            details={"splits": empty_splits},
        )
    return {split: tuple(values) for split, values in batches.items()}, lineage


def _config_lineage(config: StageWorldConfig) -> str:
    clinical = asdict(config.clinical)
    if clinical["treatment_summary_cutoff"] is None:
        clinical.pop("treatment_summary_cutoff")  # Preserve existing CT-only checkpoint lineage.
    snapshot = {
        "lineage_schema": "stageworld-config-lineage-v2",
        "contracts": {
            "source": SOURCE_SCHEMA,
            "cohort_build": BUILD_SCHEMA,
            "features": FEATURE_SCHEMA,
            "checkpoint": CHECKPOINT_SCHEMA_VERSION,
            "timeline": SYNTHETIC_TIMELINE_CONTRACT_VERSION,
            "outcome": SYNTHETIC_OUTCOME_CONTRACT_VERSION,
        },
        "project": asdict(config.project),
        "privacy": asdict(config.privacy),
        "permissions": asdict(config.permissions),
        "encoders": {
            "ct_weight_configured": config.encoders.ct_weight_path is not None,
            "ct_source_version": config.encoders.ct_source_version,
            "ct_component_version": config.encoders.ct_component_version,
            "ct_preprocess_version": config.encoders.ct_preprocess_version,
            "ct_inference_precision": config.encoders.ct_inference_precision,
        },
        "clinical": clinical,
        "model": asdict(config.model),
        "survival": asdict(config.survival),
        "training": asdict(config.training),
        "evaluation": asdict(config.evaluation),
    }
    if not (config.training.development_protocol or "").startswith("ct6-"):
        if config.model.dropout == 0.1:
            cast(dict[str, Any], snapshot["model"]).pop("dropout")
        for key in (
            "joint_backbone_lr", "lr_warmup_epochs", "lr_min_ratio", "development_min_delta"
        ):
            cast(dict[str, Any], snapshot["training"]).pop(key)
    return json.dumps(
        snapshot,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _split_version(config: StageWorldConfig, *, data_lineage_id: str | None = None) -> str:
    source = _load_tensor_artifact(synthetic_data_root(config) / "source_tensors.pt", SOURCE_SCHEMA)
    validate_synthetic_source_config(config, source)
    build = read_json(synthetic_data_root(config) / "cohort_build.json")
    _, split_version = _validate_synthetic_split_build(
        config,
        source,
        build,
        data_lineage_id=data_lineage_id,
    )
    return split_version


def _metadata_from_dict(value: Mapping[str, Any]) -> CheckpointMetadata:
    try:
        return CheckpointMetadata(**dict(value))
    except (TypeError, ValueError, ConfigurationError) as error:
        raise ArtifactError(
            code="CHECKPOINT_METADATA_MISSING",
            message="Checkpoint sidecar metadata is incomplete or malformed.",
        ) from error


def _load_checkpoint_sidecar(path: Path) -> tuple[CheckpointMetadata, str]:
    try:
        sidecar = read_json(path)
    except (OSError, ValueError) as error:
        raise ArtifactError(
            code="RESUME_METADATA_MISSING",
            message="Resume requested but checkpoint sidecar metadata is absent or unreadable.",
        ) from error
    metadata = sidecar.get("metadata")
    active_weight_version = sidecar.get("active_weight_version")
    if (
        sidecar.get("schema_version") != CHECKPOINT_SIDECAR_SCHEMA
        or not isinstance(metadata, Mapping)
        or not isinstance(active_weight_version, str)
        or not active_weight_version.strip()
    ):
        raise ArtifactError(
            code="CHECKPOINT_SIDECAR_INVALID",
            message="Checkpoint sidecar schema, metadata, or active weight version is invalid.",
        )
    return _metadata_from_dict(metadata), active_weight_version


def _load_checkpoint_payload(path: Path, *, missing_code: str) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError) as error:
        raise ArtifactError(
            code=missing_code,
            message="A required checkpoint payload is absent, unreadable, or incomplete.",
            details={"artifact": path.name},
        ) from error
    if not isinstance(payload, Mapping):
        raise ArtifactError(
            code="CHECKPOINT_INVALID",
            message="Checkpoint root must be a mapping.",
        )
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ArtifactError(
            code="CHECKPOINT_SCHEMA_MISMATCH",
            message="Checkpoint schema is absent or unsupported.",
        )
    return dict(payload)


def _load_resume_snapshot(
    checkpoint_path: Path,
    metadata_path: Path,
) -> tuple[CheckpointMetadata, dict[str, Any], Path]:
    metadata, active_weight_version = _load_checkpoint_sidecar(metadata_path)
    pointer_payload = _load_checkpoint_payload(
        checkpoint_path,
        missing_code="CHECKPOINT_UNREADABLE",
    )
    snapshot_path = checkpoint_snapshot_path(checkpoint_path, active_weight_version)
    snapshot_payload = _load_checkpoint_payload(
        snapshot_path,
        missing_code="CHECKPOINT_SNAPSHOT_MISSING",
    )
    if (
        pointer_payload.get("weight_version") != active_weight_version
        or snapshot_payload.get("weight_version") != active_weight_version
    ):
        raise ArtifactError(
            code="RESUME_ACTIVE_WEIGHT_MISMATCH",
            message="Checkpoint sidecar does not identify the pointer's exact active snapshot.",
        )
    mismatched = checkpoint_payload_mismatches(pointer_payload, snapshot_payload)
    if mismatched:
        raise ArtifactError(
            code="RESUME_CHECKPOINT_SNAPSHOT_MISMATCH",
            message="Mutable checkpoint pointer differs from its active immutable snapshot.",
            details={"fields": list(mismatched)},
        )
    contract = CheckpointContract.from_checkpoint_payload(snapshot_payload)
    if contract.weight_version != active_weight_version:
        raise ArtifactError(
            code="RESUME_ACTIVE_WEIGHT_MISMATCH",
            message="Checkpoint contract and sidecar identify different active weights.",
        )
    return metadata, snapshot_payload, snapshot_path


def _expected_survival_contract(config: StageWorldConfig, model: StageWorldModel) -> dict[str, Any]:
    return {
        "schema_version": "stageworld-survival-contract-v1",
        "endpoint": "os",
        "parameterization": config.survival.parameterization,
        "time_unit": config.survival.time_unit,
        "cutpoints": tuple(model.config.survival_cutpoints),
        "open_tail_interval": config.survival.open_tail_interval,
        "num_causes": model.config.survival_causes,
    }


def _validate_checkpoint_for_run(
    contract: CheckpointContract,
    config: StageWorldConfig,
    model: StageWorldModel,
    *,
    data_lineage_id: str,
    cohort_artifact_id: str,
    split_version: str,
    ct_feature_artifact_id: str,
    pathology_feature_artifact_id: str,
    timeline_contract_version: str,
    outcome_contract_version: str,
    phase: TrainingPhase,
) -> None:
    expected_survival = _expected_survival_contract(config, model)
    if dict(contract.survival_contract) != expected_survival:
        raise ArtifactError(
            code="CHECKPOINT_SURVIVAL_CONTRACT_MISMATCH",
            message="Checkpoint survival settings differ from the active configuration.",
        )
    expected_model = asdict(model.config)
    if dict(contract.model_config) != expected_model:
        raise ArtifactError(
            code="CHECKPOINT_MODEL_CONFIG_MISMATCH",
            message="Checkpoint model settings differ from the active configuration.",
        )
    expected = {
        "model_version": model.config.model_version,
        "endpoint": "os",
        "mode": config.mode,
        "config_lineage_id": _config_lineage(config),
        "data_lineage_id": data_lineage_id,
        "cohort_artifact_id": cohort_artifact_id,
        "split_version": split_version,
        "ct_feature_artifact_id": ct_feature_artifact_id,
        "pathology_feature_artifact_id": pathology_feature_artifact_id,
        "timeline_contract_version": timeline_contract_version,
        "outcome_contract_version": outcome_contract_version,
        "training_seed": config.training.seed,
        "source_schema_version": SOURCE_SCHEMA,
        "cohort_schema_version": BUILD_SCHEMA,
        "feature_schema_version": FEATURE_SCHEMA,
        "phase": phase.value,
    }
    mismatched = {
        key: {"expected": expected_value, "actual": getattr(contract, key)}
        for key, expected_value in expected.items()
        if getattr(contract, key) != expected_value
    }
    if mismatched:
        raise ArtifactError(
            code="CHECKPOINT_RUNTIME_CONTRACT_MISMATCH",
            message="Checkpoint lineage differs from the active data/configuration contract.",
            details={"fields": mismatched},
        )


def _load_transfer_weights(
    model: StageWorldModel,
    path: Path,
    *,
    config: StageWorldConfig,
    data_lineage_id: str,
    cohort_artifact_id: str,
    split_version: str,
    ct_feature_artifact_id: str,
    pathology_feature_artifact_id: str,
    timeline_contract_version: str,
    outcome_contract_version: str,
) -> tuple[CheckpointContract, dict[str, Any]]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError) as error:
        raise ArtifactError(
            code="WORLD_PRETRAIN_CHECKPOINT_REQUIRED",
            message="Joint survival training requires a complete world-pretraining checkpoint.",
            remediation="Run train --phase world_pretrain first.",
        ) from error
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
    ):
        raise ArtifactError(
            code="CHECKPOINT_SCHEMA_MISMATCH",
            message="World-pretraining checkpoint schema is incompatible.",
        )
    contract = CheckpointContract.from_checkpoint_payload(payload)
    _validate_checkpoint_for_run(
        contract,
        config,
        model,
        data_lineage_id=data_lineage_id,
        cohort_artifact_id=cohort_artifact_id,
        split_version=split_version,
        ct_feature_artifact_id=ct_feature_artifact_id,
        pathology_feature_artifact_id=pathology_feature_artifact_id,
        timeline_contract_version=timeline_contract_version,
        outcome_contract_version=outcome_contract_version,
        phase=TrainingPhase.WORLD_PRETRAIN,
    )
    try:
        model.load_state_dict(payload["model_state"], strict=True)
    except (KeyError, RuntimeError) as error:
        raise ArtifactError(
            code="CHECKPOINT_STATE_INCOMPATIBLE",
            message="World-pretraining model state does not match the configured architecture.",
        ) from error
    return contract, dict(payload)


def _validate_parent_metadata(parent: CheckpointContract, metadata: CheckpointMetadata) -> None:
    expected = {
        "checkpoint_id": metadata.parent_checkpoint_id,
        "weight_version": metadata.parent_weight_version,
        "phase": metadata.parent_phase,
        "config_lineage_id": metadata.parent_config_lineage_id,
        "data_lineage_id": metadata.parent_data_lineage_id,
        "cohort_artifact_id": metadata.parent_cohort_artifact_id,
        "split_version": metadata.parent_split_version,
        "ct_feature_artifact_id": metadata.parent_ct_feature_artifact_id,
        "pathology_feature_artifact_id": metadata.parent_pathology_feature_artifact_id,
        "timeline_contract_version": metadata.parent_timeline_contract_version,
        "outcome_contract_version": metadata.parent_outcome_contract_version,
    }
    mismatched = {
        key: {"expected": value, "actual": getattr(parent, key)}
        for key, value in expected.items()
        if getattr(parent, key) != value
    }
    if mismatched:
        raise ArtifactError(
            code="PARENT_CHECKPOINT_LINEAGE_MISMATCH",
            message="Joint checkpoint parent metadata does not identify its exact snapshot.",
            details={"fields": mismatched},
        )


def run_synthetic_training(
    config: StageWorldConfig,
    *,
    phase: TrainingPhase,
    resume: bool = False,
) -> dict[str, Any]:
    _require_synthetic(config, "train")
    batches_by_split, data_lineage = load_synthetic_batches(config)
    source = _load_tensor_artifact(synthetic_data_root(config) / "source_tensors.pt", SOURCE_SCHEMA)
    validate_synthetic_source_config(config, source)
    cohort_artifact_id = str(source["cohort_artifact_id"])
    ct_features = _load_tensor_artifact(synthetic_feature_root(config) / "ct.pt", FEATURE_SCHEMA)
    pathology_features = _load_tensor_artifact(
        synthetic_feature_root(config) / "pathology.pt", FEATURE_SCHEMA
    )
    ct_feature_artifact_id, pathology_feature_artifact_id = validate_synthetic_feature_artifacts(
        config,
        source,
        ct_features,
        pathology_features,
    )
    timeline_contract_version = str(source["timeline_contract_version"])
    outcome_contract_version = str(source["outcome_contract_version"])
    training_batches = batches_by_split[SplitName.TRAIN]
    validation_batches = batches_by_split[SplitName.VALIDATION]
    test_batches = batches_by_split[SplitName.TEST]
    split_version = _split_version(config, data_lineage_id=data_lineage)
    random.seed(config.training.seed)
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    model = build_model(config)
    run_root = config.output_root / "runs" / phase.value
    checkpoint_path = run_root / "checkpoint.pt"
    metadata_path = run_root / "checkpoint_metadata.json"
    config_lineage = _config_lineage(config)
    parent: CheckpointContract | None = None
    parent_payload: dict[str, Any] | None = None
    resume_payload: dict[str, Any] | None = None
    resume_snapshot_path: Path | None = None
    if phase is TrainingPhase.JOINT_SURVIVAL and not resume:
        world_checkpoint = (
            config.output_root / "runs" / TrainingPhase.WORLD_PRETRAIN.value / "checkpoint.pt"
        )
        pointer_parent, pointer_parent_payload = _load_transfer_weights(
            model,
            world_checkpoint,
            config=config,
            data_lineage_id=data_lineage,
            cohort_artifact_id=cohort_artifact_id,
            split_version=split_version,
            ct_feature_artifact_id=ct_feature_artifact_id,
            pathology_feature_artifact_id=pathology_feature_artifact_id,
            timeline_contract_version=timeline_contract_version,
            outcome_contract_version=outcome_contract_version,
        )
        parent_snapshot = checkpoint_snapshot_path(world_checkpoint, pointer_parent.weight_version)
        snapshot_parent, snapshot_parent_payload = _load_transfer_weights(
            model,
            parent_snapshot,
            config=config,
            data_lineage_id=data_lineage,
            cohort_artifact_id=cohort_artifact_id,
            split_version=split_version,
            ct_feature_artifact_id=ct_feature_artifact_id,
            pathology_feature_artifact_id=pathology_feature_artifact_id,
            timeline_contract_version=timeline_contract_version,
            outcome_contract_version=outcome_contract_version,
        )
        parent_differences = checkpoint_payload_mismatches(
            pointer_parent_payload,
            snapshot_parent_payload,
            path="parent_checkpoint",
        )
        if snapshot_parent != pointer_parent or parent_differences:
            raise ArtifactError(
                code="PARENT_CHECKPOINT_SNAPSHOT_MISMATCH",
                message=(
                    "World-pretraining pointer and exact weight snapshot disagree in "
                    "contract or serialized state."
                ),
                details={"fields": list(parent_differences)},
            )
        parent = snapshot_parent
        parent_payload = snapshot_parent_payload
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay
    )
    max_steps = min(
        config.training.smoke_max_steps,
        config.training.world_pretrain_steps
        if phase is TrainingPhase.WORLD_PRETRAIN
        else config.training.joint_survival_steps,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(max_steps, 1), eta_min=config.training.lr * 0.1
    )
    logger = LocalEventLogger(run_root / "events.jsonl")
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
    if resume:
        stored_metadata, resume_payload, resume_snapshot_path = _load_resume_snapshot(
            checkpoint_path,
            metadata_path,
        )
        if phase is TrainingPhase.JOINT_SURVIVAL:
            parent_weight_version = stored_metadata.parent_weight_version
            if parent_weight_version is None:
                raise ArtifactError(
                    code="PARENT_CHECKPOINT_LINEAGE_REQUIRED",
                    message="Joint resume metadata has no exact parent weight version.",
                )
            parent, parent_payload = _load_transfer_weights(
                model,
                checkpoint_snapshot_path(
                    config.output_root
                    / "runs"
                    / TrainingPhase.WORLD_PRETRAIN.value
                    / "checkpoint.pt",
                    parent_weight_version,
                ),
                config=config,
                data_lineage_id=data_lineage,
                cohort_artifact_id=cohort_artifact_id,
                split_version=split_version,
                ct_feature_artifact_id=ct_feature_artifact_id,
                pathology_feature_artifact_id=pathology_feature_artifact_id,
                timeline_contract_version=timeline_contract_version,
                outcome_contract_version=outcome_contract_version,
            )
            _validate_parent_metadata(parent, stored_metadata)
            embedded_parent_state = resume_payload.get("transfer_parent_model_state")
            exact_parent_state = parent_payload.get("model_state")
            if not isinstance(embedded_parent_state, Mapping) or not isinstance(
                exact_parent_state, Mapping
            ):
                raise ArtifactError(
                    code="PARENT_CHECKPOINT_STATE_REQUIRED",
                    message="Joint resume cannot compare its embedded and exact parent states.",
                )
            changed_parent_state = model_state_mismatches(
                cast(Mapping[str, Tensor], embedded_parent_state),
                cast(Mapping[str, Tensor], exact_parent_state),
            )
            if changed_parent_state:
                raise ArtifactError(
                    code="PARENT_CHECKPOINT_STATE_MISMATCH",
                    message=(
                        "Joint checkpoint was not initialized from the exact parent snapshot "
                        "identified by its lineage."
                    ),
                    details={"tensors": list(changed_parent_state)},
                )
        metadata = CheckpointMetadata(
            checkpoint_id=stored_metadata.checkpoint_id,
            model_version=model.config.model_version,
            endpoint="os",
            mode=config.mode.value,
            config_lineage_id=config_lineage,
            data_lineage_id=data_lineage,
            cohort_artifact_id=cohort_artifact_id,
            split_version=split_version,
            ct_feature_artifact_id=ct_feature_artifact_id,
            pathology_feature_artifact_id=pathology_feature_artifact_id,
            timeline_contract_version=timeline_contract_version,
            outcome_contract_version=outcome_contract_version,
            training_seed=config.training.seed,
            source_schema_version=SOURCE_SCHEMA,
            cohort_schema_version=BUILD_SCHEMA,
            feature_schema_version=FEATURE_SCHEMA,
            phase=phase.value,
            selection_rule=config.training.checkpoint_selection,
            parent_checkpoint_id=stored_metadata.parent_checkpoint_id,
            parent_weight_version=stored_metadata.parent_weight_version,
            parent_phase=stored_metadata.parent_phase,
            parent_config_lineage_id=stored_metadata.parent_config_lineage_id,
            parent_data_lineage_id=stored_metadata.parent_data_lineage_id,
            parent_cohort_artifact_id=stored_metadata.parent_cohort_artifact_id,
            parent_split_version=stored_metadata.parent_split_version,
            parent_ct_feature_artifact_id=stored_metadata.parent_ct_feature_artifact_id,
            parent_pathology_feature_artifact_id=(
                stored_metadata.parent_pathology_feature_artifact_id
            ),
            parent_timeline_contract_version=stored_metadata.parent_timeline_contract_version,
            parent_outcome_contract_version=stored_metadata.parent_outcome_contract_version,
        )
        if resume_snapshot_path is None:
            raise ArtifactError(
                code="CHECKPOINT_SNAPSHOT_MISSING",
                message="Resume snapshot resolution failed.",
            )
        trainer.load_checkpoint(resume_snapshot_path, expected=metadata)
    else:
        metadata = new_checkpoint_metadata(
            model=model,
            mode=config.mode.value,
            config_lineage_id=config_lineage,
            data_lineage_id=data_lineage,
            cohort_artifact_id=cohort_artifact_id,
            split_version=split_version,
            ct_feature_artifact_id=ct_feature_artifact_id,
            pathology_feature_artifact_id=pathology_feature_artifact_id,
            timeline_contract_version=timeline_contract_version,
            outcome_contract_version=outcome_contract_version,
            training_seed=config.training.seed,
            source_schema_version=SOURCE_SCHEMA,
            cohort_schema_version=BUILD_SCHEMA,
            feature_schema_version=FEATURE_SCHEMA,
            phase=phase,
            selection_rule=config.training.checkpoint_selection,
            parent_checkpoint_id=None if parent is None else parent.checkpoint_id,
            parent_weight_version=None if parent is None else parent.weight_version,
            parent_phase=None if parent is None else parent.phase,
            parent_config_lineage_id=None if parent is None else parent.config_lineage_id,
            parent_data_lineage_id=None if parent is None else parent.data_lineage_id,
            parent_cohort_artifact_id=None if parent is None else parent.cohort_artifact_id,
            parent_split_version=None if parent is None else parent.split_version,
            parent_ct_feature_artifact_id=(
                None if parent is None else parent.ct_feature_artifact_id
            ),
            parent_pathology_feature_artifact_id=(
                None if parent is None else parent.pathology_feature_artifact_id
            ),
            parent_timeline_contract_version=(
                None if parent is None else parent.timeline_contract_version
            ),
            parent_outcome_contract_version=(
                None if parent is None else parent.outcome_contract_version
            ),
        )
    registry = ExperimentRegistry(config.output_root / "experiment_registry.csv")
    run_id = f"{phase.value}-{metadata.checkpoint_id}"
    started = time.time()
    registry.update(
        {
            "run_id": run_id,
            "status": "running",
            "mode": config.mode.value,
            "phase": phase.value,
            "seed": config.training.seed,
            "config_lineage_id": config_lineage,
            "data_lineage_id": data_lineage,
            "started_at_unix": started,
        }
    )
    try:
        history = bounded_fit(
            trainer,
            training_batches,
            phase=phase,
            max_steps=max_steps,
            max_minutes=config.training.smoke_max_minutes,
            accumulation_steps=1,
            weights=LossWeights(),
            kl_warmup_steps=max(1, max_steps // 3),
        )
        if trainer.state.optimizer_step < max_steps:
            registry.update(
                {
                    "run_id": run_id,
                    "status": "cancelled",
                    "mode": config.mode.value,
                    "phase": phase.value,
                    "seed": config.training.seed,
                    "config_lineage_id": config_lineage,
                    "data_lineage_id": data_lineage,
                    "started_at_unix": started,
                    "finished_at_unix": time.time(),
                    "failure_code": "SMOKE_TIME_BUDGET_REACHED",
                }
            )
            raise ArtifactError(
                code="SMOKE_TIME_BUDGET_REACHED",
                message="Synthetic training stopped before its configured step budget.",
                remediation="Reduce model/steps or explicitly approve a larger budget.",
            )
        validation = trainer.evaluate(validation_batches, phase=phase, weights=LossWeights())
        transfer_parent_model_state: Mapping[str, Tensor] | None = None
        if phase is TrainingPhase.JOINT_SURVIVAL:
            if parent_payload is None or not isinstance(parent_payload.get("model_state"), Mapping):
                raise ArtifactError(
                    code="PARENT_CHECKPOINT_STATE_REQUIRED",
                    message="Joint checkpoint save requires the exact transfer-parent model state.",
                )
            transfer_parent_model_state = cast(Mapping[str, Tensor], parent_payload["model_state"])
        trainer.save_checkpoint(
            checkpoint_path,
            metadata,
            sampler_state={"batch_cursor": 0},
            transfer_parent_model_state=transfer_parent_model_state,
        )
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint_payload, Mapping):
            raise ArtifactError(
                code="CHECKPOINT_INVALID",
                message="Saved checkpoint root must be a mapping.",
            )
        saved_contract = CheckpointContract.from_checkpoint_payload(checkpoint_payload)
        snapshot_path = checkpoint_snapshot_path(checkpoint_path, saved_contract.weight_version)
        if not snapshot_path.is_file():
            raise ArtifactError(
                code="CHECKPOINT_SNAPSHOT_MISSING",
                message="Exact checkpoint snapshot was not retained after training.",
            )
        saved_snapshot_payload = _load_checkpoint_payload(
            snapshot_path,
            missing_code="CHECKPOINT_SNAPSHOT_MISSING",
        )
        pointer_differences = checkpoint_payload_mismatches(
            checkpoint_payload,
            saved_snapshot_payload,
        )
        if pointer_differences:
            raise ArtifactError(
                code="CHECKPOINT_POINTER_SNAPSHOT_MISMATCH",
                message="Saved checkpoint pointer differs from its immutable snapshot.",
                details={"fields": list(pointer_differences)},
            )
        atomic_write_json(
            metadata_path,
            {
                "schema_version": CHECKPOINT_SIDECAR_SCHEMA,
                "active_weight_version": saved_contract.weight_version,
                "metadata": asdict(metadata),
            },
        )
        summary = {
            "schema_version": TRAINING_SUMMARY_SCHEMA,
            "mode": "synthetic",
            "clinical_validation": False,
            "phase": phase.value,
            "optimizer_steps": trainer.state.optimizer_step,
            "initial_loss": history[0]["loss"] if history else None,
            "final_loss": history[-1]["loss"] if history else None,
            "validation": validation,
            "training_split": SplitName.TRAIN.value,
            "validation_split": SplitName.VALIDATION.value,
            "training_patient_count": sum(len(batch.patient_ids) for batch in training_batches),
            "validation_patient_count": sum(len(batch.patient_ids) for batch in validation_batches),
            "held_out_test_patient_count": sum(len(batch.patient_ids) for batch in test_batches),
            "test_used_for_optimization_or_selection": False,
            "checkpoint": str(checkpoint_path),
            "checkpoint_snapshot": str(snapshot_path),
            "checkpoint_id": metadata.checkpoint_id,
            "weight_version": saved_contract.weight_version,
            "config_lineage_id": metadata.config_lineage_id,
            "data_lineage_id": metadata.data_lineage_id,
            "cohort_artifact_id": metadata.cohort_artifact_id,
            "split_version": metadata.split_version,
            "ct_feature_artifact_id": metadata.ct_feature_artifact_id,
            "pathology_feature_artifact_id": metadata.pathology_feature_artifact_id,
            "timeline_contract_version": metadata.timeline_contract_version,
            "outcome_contract_version": metadata.outcome_contract_version,
            "training_seed": metadata.training_seed,
            "source_schema_version": metadata.source_schema_version,
            "cohort_schema_version": metadata.cohort_schema_version,
            "feature_schema_version": metadata.feature_schema_version,
            "parent_checkpoint_id": metadata.parent_checkpoint_id,
            "parent_weight_version": metadata.parent_weight_version,
            "parent_phase": metadata.parent_phase,
            "parent_config_lineage_id": metadata.parent_config_lineage_id,
            "parent_data_lineage_id": metadata.parent_data_lineage_id,
            "parent_cohort_artifact_id": metadata.parent_cohort_artifact_id,
            "parent_split_version": metadata.parent_split_version,
            "parent_ct_feature_artifact_id": metadata.parent_ct_feature_artifact_id,
            "parent_pathology_feature_artifact_id": (metadata.parent_pathology_feature_artifact_id),
            "parent_timeline_contract_version": metadata.parent_timeline_contract_version,
            "parent_outcome_contract_version": metadata.parent_outcome_contract_version,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "device": trainer.device.type,
        }
        summary_path = run_root / "summary.json"
        atomic_write_json(summary_path, summary)
        registry.update(
            {
                "run_id": run_id,
                "status": "completed",
                "mode": config.mode.value,
                "phase": phase.value,
                "seed": config.training.seed,
                "config_lineage_id": config_lineage,
                "data_lineage_id": data_lineage,
                "started_at_unix": started,
                "finished_at_unix": time.time(),
            }
        )
        return {"status": "ok", **summary, "summary": str(summary_path)}
    except Exception as error:
        failure_code = (
            "CUDA_OUT_OF_MEMORY"
            if isinstance(error, torch.OutOfMemoryError)
            else getattr(error, "code", type(error).__name__.upper())
        )
        if not isinstance(error, ArtifactError) or error.code != "SMOKE_TIME_BUDGET_REACHED":
            registry.update(
                {
                    "run_id": run_id,
                    "status": "failed",
                    "mode": config.mode.value,
                    "phase": phase.value,
                    "seed": config.training.seed,
                    "config_lineage_id": config_lineage,
                    "data_lineage_id": data_lineage,
                    "started_at_unix": started,
                    "finished_at_unix": time.time(),
                    "failure_code": failure_code,
                }
            )
        if isinstance(error, torch.OutOfMemoryError):
            raise ResourceError(
                code="CUDA_OUT_OF_MEMORY",
                message="The bounded training run exceeded available accelerator memory.",
                remediation=(
                    "Reduce the configured patient/token budget or use an approved mixed-"
                    "precision mode, then restart from the last complete checkpoint."
                ),
                details={"phase": phase.value},
            ) from error
        raise


__all__ = [
    "BUILD_SCHEMA",
    "FEATURE_SCHEMA",
    "SOURCE_SCHEMA",
    "build_model",
    "build_synthetic_cohort",
    "extract_synthetic_features",
    "load_synthetic_batches",
    "make_synthetic_artifacts",
    "run_synthetic_training",
    "synthetic_data_root",
    "synthetic_feature_root",
]
