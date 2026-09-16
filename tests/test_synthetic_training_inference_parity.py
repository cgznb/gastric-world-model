from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from stageworld.artifacts import read_json
from stageworld.config import StageWorldConfig, load_config
from stageworld.data import (
    SYNTHETIC_TIMELINE,
    FeatureFirewall,
    MissingCategory,
    ObservationRole,
    QualityStatus,
    Query,
    SplitName,
    Stage,
    TreatmentKind,
    TreatmentStatus,
    default_synthetic_policy,
    load_cohort_json,
)
from stageworld.encoders import ObservationTokens
from stageworld.inference import ActionFeature, FeatureManifest, InferenceEngine
from stageworld.model import ActionTokens, BeliefState, ThreeStageOutput
from stageworld.pipeline import _runtime_context
from stageworld.synthetic_workflow import (
    SOURCE_SCHEMA,
    _load_tensor_artifact,
    build_synthetic_cohort,
    extract_synthetic_features,
    load_synthetic_batches,
    make_synthetic_artifacts,
    run_synthetic_training,
    synthetic_data_root,
)
from stageworld.training import TrainingPhase, WorldModelBatch

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class _ParityContext:
    config: StageWorldConfig
    engine: InferenceEngine
    firewall: FeatureFirewall
    manifests: dict[str, FeatureManifest]
    queries: tuple[Query, ...]
    batches: dict[SplitName, tuple[WorldModelBatch, ...]]
    source: dict[str, Any]


@pytest.fixture(scope="module")
def parity_context(tmp_path_factory: pytest.TempPathFactory) -> _ParityContext:
    run_root = tmp_path_factory.mktemp("synthetic-training-inference-parity")
    base = load_config(ROOT / "configs/project.synthetic.yaml")
    config = replace(
        base,
        paths=replace(
            base.paths,
            output_root=str(run_root),
            feature_root=str(run_root / "features"),
        ),
        model=replace(
            base.model,
            hidden_dim=8,
            state_tokens=2,
            stochastic_dim_per_token=2,
            attention_heads=2,
            transition_blocks=1,
            observation_blocks=1,
            resampler_blocks=1,
            ct_tokens=2,
            pathology_tokens=2,
        ),
        training=replace(
            base.training,
            patient_batch_size=12,
            smoke_max_steps=1,
            smoke_max_minutes=2,
            world_pretrain_steps=1,
            joint_survival_steps=1,
        ),
    )
    config.validate(command="synthetic-parity-test", supervised=True)
    with patch("torch.cuda.is_available", return_value=False):
        make_synthetic_artifacts(config, patient_count=12)
        build_synthetic_cohort(config)
        extract_synthetic_features(config, "ct")
        extract_synthetic_features(config, "pathology")
        run_synthetic_training(config, phase=TrainingPhase.WORLD_PRETRAIN)
        run_synthetic_training(config, phase=TrainingPhase.JOINT_SURVIVAL)
    engine, firewall, manifests, queries, runtime_lineage = _runtime_context(config)
    batches, training_lineage = load_synthetic_batches(config)
    assert runtime_lineage == training_lineage
    source = _load_tensor_artifact(
        synthetic_data_root(config) / "source_tensors.pt", SOURCE_SCHEMA
    )
    return _ParityContext(
        config=config,
        engine=engine,
        firewall=firewall,
        manifests=manifests,
        queries=queries,
        batches=batches,
        source=source,
    )


def _batch_row(context: _ParityContext, patient_id: str) -> tuple[WorldModelBatch, int]:
    for split in SplitName:
        for batch in context.batches[split]:
            if patient_id in batch.patient_ids:
                return batch, batch.patient_ids.index(patient_id)
    raise AssertionError(f"Synthetic patient {patient_id} was not assigned to a batch")


def _query(context: _ParityContext, patient_id: str, stage: Stage) -> Query:
    return next(
        query
        for query in context.queries
        if query.patient_id == patient_id and query.stage is stage
    )


def _forward(
    model: torch.nn.Module,
    batch: WorldModelBatch,
    **overrides: ActionTokens | ObservationTokens | torch.Tensor | None,
) -> ThreeStageOutput:
    arguments: dict[str, Any] = {
        "ct0": batch.ct0,
        "clinical0": batch.clinical0,
        "s0_time": batch.s0_time,
        "treatment_actions": batch.treatment_actions,
        "ct1_acquisition_time": batch.ct1_acquisition_time,
        "ct1": batch.ct1,
        "s1_time": batch.s1_time,
        "surgery_actions": batch.surgery_actions,
        "pathology_acquisition_time": batch.pathology_acquisition_time,
        "pathology": batch.pathology,
        "s2_time": batch.s2_time,
        "horizons": batch.horizons,
        "ct1_availability_time": batch.ct1_availability_time,
        "ct1_unavailable_event_mask": batch.ct1_unavailable_event_mask,
        "clinical1": batch.clinical1,
        "s1_update_actions": batch.s1_update_actions,
        "pathology_availability_time": batch.pathology_availability_time,
        "pathology_unavailable_event_mask": batch.pathology_unavailable_event_mask,
        "clinical2": batch.clinical2,
        "s2_update_actions": batch.s2_update_actions,
        "deterministic": True,
    }
    arguments.update(overrides)
    model.eval()
    with torch.inference_mode():
        return model.forward_three_stage(**arguments)  # type: ignore[attr-defined]


def _without_observations(value: ObservationTokens) -> ObservationTokens:
    return replace(value, valid=torch.zeros_like(value.valid))


def _without_actions(value: ActionTokens | None) -> ActionTokens | None:
    if value is None:
        return None
    return replace(value, valid=torch.zeros_like(value.valid))


def _poison_invalid_observations(value: ObservationTokens) -> ObservationTokens:
    assert not value.valid.any()
    return replace(
        value,
        values=torch.full_like(value.values, float("nan")),
        modality=torch.full_like(value.modality, 999),
        acquired_time=torch.full_like(value.acquired_time, float("nan")),
        available_time=torch.full_like(value.available_time, float("nan")),
        coords=(
            None if value.coords is None else torch.full_like(value.coords, float("nan"))
        ),
    )


def _poison_invalid_actions(value: ActionTokens | None) -> ActionTokens | None:
    if value is None:
        return None
    assert not value.valid.any()
    return replace(
        value,
        values=torch.full_like(value.values, float("nan")),
        event_time=torch.full_like(value.event_time, float("nan")),
        available_time=torch.full_like(value.available_time, float("nan")),
        event_type=(
            None if value.event_type is None else torch.full_like(value.event_type, 999)
        ),
        planned_or_delivered=(
            None
            if value.planned_or_delivered is None
            else torch.full_like(value.planned_or_delivered, 999)
        ),
        known_exposure=(
            None
            if value.known_exposure is None
            else torch.full_like(value.known_exposure, float("nan"))
        ),
    )


def _stage_firewall(
    context: _ParityContext,
    patient_id: str,
    stage: Stage,
    *,
    unavailable: tuple[MissingCategory, QualityStatus] | None = None,
    include_actions: bool = True,
) -> FeatureFirewall:
    cohort = load_cohort_json(synthetic_data_root(context.config) / "cohort.json")
    target_role = {
        Stage.S1: ObservationRole.POST_TREATMENT_CT,
        Stage.S2: ObservationRole.SURGICAL_PATHOLOGY,
    }[stage]
    target_clinical_field = {
        Stage.S1: "radiologic_response",
        Stage.S2: "yp_stage",
    }[stage]
    observations = []
    unavailable_added = False
    for observation in cohort.observations:
        if observation.patient_id != patient_id:
            continue
        if observation.role is not target_role:
            observations.append(observation)
            continue
        if unavailable is None or unavailable_added:
            continue
        missing_reason, quality_status = unavailable
        observations.append(
            replace(
                observation,
                local_asset_id=None,
                quality_status=quality_status,
                missing_reason=missing_reason,
            )
        )
        unavailable_added = True
    measurements = tuple(
        measurement
        for measurement in cohort.clinical_measurements
        if measurement.patient_id == patient_id
        and measurement.field_name != target_clinical_field
    )
    treatments = tuple(
        treatment
        for treatment in cohort.treatments
        if treatment.patient_id == patient_id and include_actions
    )
    return FeatureFirewall(
        tuple(patient for patient in cohort.patients if patient.patient_id == patient_id),
        tuple(observations),
        measurements,
        treatments,
        default_synthetic_policy(),
    )


def _assert_numeric_state_row_close(
    expected: BeliefState,
    row: int,
    actual: BeliefState,
    *,
    actual_row: int = 0,
) -> None:
    """Compare prediction-bearing state tensors; audit metadata has a separate contract."""

    torch.testing.assert_close(
        actual.memory[actual_row : actual_row + 1],
        expected.memory[row : row + 1],
        rtol=1e-6,
        atol=3e-6,
    )
    torch.testing.assert_close(
        actual.query_time[actual_row : actual_row + 1],
        expected.query_time[row : row + 1],
        rtol=0,
        atol=1e-6,
    )
    for name in ("stochastic_mean", "stochastic_log_std", "sample"):
        expected_value = getattr(expected, name)
        actual_value = getattr(actual, name)
        assert (expected_value is None) == (actual_value is None)
        if expected_value is not None:
            assert actual_value is not None
            torch.testing.assert_close(
                actual_value[actual_row : actual_row + 1],
                expected_value[row : row + 1],
                rtol=1e-6,
                atol=3e-6,
            )


def _fresh_engine(context: _ParityContext) -> InferenceEngine:
    return InferenceEngine(
        context.engine.model,
        context.engine.checkpoint,
        context.engine.feature_contract,
        survival_time_unit=context.config.survival.time_unit,
        survival_parameterization=context.config.survival.parameterization,
        survival_open_tail_interval=context.config.survival.open_tail_interval,
    )


def test_training_batch_and_runtime_replay_have_identical_deterministic_numeric_states(
    parity_context: _ParityContext,
) -> None:
    patient_id = "SYN-0000"
    batch, row = _batch_row(parity_context, patient_id)
    training_output = _forward(parity_context.engine.model, batch)
    runtime_states = {
        stage: parity_context.engine.replay_query(
            parity_context.firewall,
            parity_context.manifests[patient_id],
            _query(parity_context, patient_id, stage),
        ).state
        for stage in (Stage.S0, Stage.S1, Stage.S2)
    }

    _assert_numeric_state_row_close(training_output.state_s0, row, runtime_states[Stage.S0])
    _assert_numeric_state_row_close(training_output.state_s1, row, runtime_states[Stage.S1])
    _assert_numeric_state_row_close(training_output.state_s2, row, runtime_states[Stage.S2])


def test_synthetic_batches_preserve_typed_unavailable_events_without_conflating_no_surgery(
    parity_context: _ParityContext,
) -> None:
    ct1_events: set[str] = set()
    pathology_events: set[str] = set()
    for split_batches in parity_context.batches.values():
        for batch in split_batches:
            assert batch.ct1_unavailable_event_mask is not None
            assert batch.pathology_unavailable_event_mask is not None
            ct1_events.update(
                patient_id
                for patient_id, unavailable in zip(
                    batch.patient_ids,
                    batch.ct1_unavailable_event_mask.tolist(),
                    strict=True,
                )
                if unavailable
            )
            pathology_events.update(
                patient_id
                for patient_id, unavailable in zip(
                    batch.patient_ids,
                    batch.pathology_unavailable_event_mask.tolist(),
                    strict=True,
                )
                if unavailable
            )

    assert ct1_events == {"SYN-0005"}
    assert pathology_events == {"SYN-0002", "SYN-0004"}
    assert "SYN-0003" not in pathology_events


def test_runtime_audit_metadata_is_patient_scoped_not_batch_metadata_parity(
    parity_context: _ParityContext,
) -> None:
    patient_id = "SYN-0000"
    batch, _ = _batch_row(parity_context, patient_id)
    training = _forward(parity_context.engine.model, batch).state_s1
    runtime = _fresh_engine(parity_context).replay_query(
        parity_context.firewall,
        parity_context.manifests[patient_id],
        _query(parity_context, patient_id, Stage.S1),
    )

    assert runtime.input_manifest
    assert all(reference.record_id.startswith(patient_id) for reference in runtime.input_manifest)
    assert runtime.state.provenance
    assert all(patient_id in source for source in runtime.state.provenance)
    assert "synthetic_input" in runtime.state.quality_flags
    assert "synthetic_input" not in training.quality_flags
    assert runtime.state.provenance != training.provenance


def test_all_invalid_stage_padding_is_inert_and_has_no_implicit_observation_boundary(
    parity_context: _ParityContext,
) -> None:
    patient_id = "SYN-0000"
    batch, row = _batch_row(parity_context, patient_id)
    assert batch.clinical1 is not None
    clean_batch = replace(
        batch,
        ct1=_without_observations(batch.ct1),
        clinical1=_without_observations(batch.clinical1),
        treatment_actions=_without_actions(batch.treatment_actions),
        s1_update_actions=_without_actions(batch.s1_update_actions),
        ct1_unavailable_event_mask=torch.zeros(len(batch.patient_ids), dtype=torch.bool),
    )
    training_output = _forward(parity_context.engine.model, clean_batch)
    poisoned_output = _forward(
        parity_context.engine.model,
        replace(
            clean_batch,
            ct1=_poison_invalid_observations(clean_batch.ct1),
            clinical1=_poison_invalid_observations(clean_batch.clinical1),
            treatment_actions=_poison_invalid_actions(clean_batch.treatment_actions),
            s1_update_actions=_poison_invalid_actions(clean_batch.s1_update_actions),
        ),
    )
    runtime = _fresh_engine(parity_context).replay_query(
        _stage_firewall(
            parity_context,
            patient_id,
            Stage.S1,
            include_actions=False,
        ),
        parity_context.manifests[patient_id],
        _query(parity_context, patient_id, Stage.S1),
    )

    _assert_numeric_state_row_close(training_output.state_s1, row, runtime.state)
    _assert_numeric_state_row_close(
        training_output.state_s1,
        row,
        poisoned_output.state_s1,
        actual_row=row,
    )
    assert not any(
        reference.record_id.endswith(("-ct1", "-response"))
        for reference in runtime.input_manifest
    )


def test_action_only_stage_replays_delivered_actions_at_their_own_boundaries(
    parity_context: _ParityContext,
) -> None:
    patient_id = "SYN-0000"
    batch, row = _batch_row(parity_context, patient_id)
    assert batch.clinical1 is not None
    observation_overrides = {
        "ct1": _without_observations(batch.ct1),
        "clinical1": _without_observations(batch.clinical1),
    }
    action_only = _forward(
        parity_context.engine.model,
        batch,
        **observation_overrides,
    )
    no_actions = _forward(
        parity_context.engine.model,
        batch,
        **observation_overrides,
        treatment_actions=_without_actions(batch.treatment_actions),
        s1_update_actions=_without_actions(batch.s1_update_actions),
    )
    runtime = _fresh_engine(parity_context).replay_query(
        _stage_firewall(parity_context, patient_id, Stage.S1),
        parity_context.manifests[patient_id],
        _query(parity_context, patient_id, Stage.S1),
    )

    _assert_numeric_state_row_close(action_only.state_s1, row, runtime.state)
    assert {
        reference.record_id
        for reference in runtime.input_manifest
        if reference.record_kind == "treatment"
    } == {
        f"{patient_id}-delivered-systemic",
        f"{patient_id}-post-ct-treatment",
    }
    assert not torch.allclose(
        action_only.state_s1.memory[row],
        no_actions.state_s1.memory[row],
        rtol=0,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    ("stage", "missing_reason", "quality_status"),
    (
        (Stage.S1, MissingCategory.MISSING, QualityStatus.UNKNOWN),
        (Stage.S1, MissingCategory.FAILED_QC, QualityStatus.FAILED),
        (Stage.S2, MissingCategory.MISSING, QualityStatus.UNKNOWN),
        (Stage.S2, MissingCategory.FAILED_QC, QualityStatus.FAILED),
    ),
)
def test_explicit_unavailable_event_boundaries_have_training_runtime_state_parity(
    parity_context: _ParityContext,
    stage: Stage,
    missing_reason: MissingCategory,
    quality_status: QualityStatus,
) -> None:
    patient_id = "SYN-0000"
    batch, row = _batch_row(parity_context, patient_id)
    unavailable_mask = torch.zeros(len(batch.patient_ids), dtype=torch.bool)
    unavailable_mask[row] = True
    if stage is Stage.S1:
        assert batch.clinical1 is not None
        explicit_batch = replace(
            batch,
            ct1=_without_observations(batch.ct1),
            clinical1=_without_observations(batch.clinical1),
            ct1_unavailable_event_mask=unavailable_mask,
        )
        without_boundary_batch = replace(
            explicit_batch,
            ct1_unavailable_event_mask=torch.zeros_like(unavailable_mask),
        )
        state_name = "state_s1"
    else:
        assert batch.clinical2 is not None
        explicit_batch = replace(
            batch,
            pathology=_without_observations(batch.pathology),
            clinical2=_without_observations(batch.clinical2),
            pathology_unavailable_event_mask=unavailable_mask,
        )
        without_boundary_batch = replace(
            explicit_batch,
            pathology_unavailable_event_mask=torch.zeros_like(unavailable_mask),
        )
        state_name = "state_s2"
    explicit = _forward(parity_context.engine.model, explicit_batch)
    without_boundary = _forward(parity_context.engine.model, without_boundary_batch)
    runtime = _fresh_engine(parity_context).replay_query(
        _stage_firewall(
            parity_context,
            patient_id,
            stage,
            unavailable=(missing_reason, quality_status),
        ),
        parity_context.manifests[patient_id],
        _query(parity_context, patient_id, stage),
    )
    explicit_state = getattr(explicit, state_name)
    no_boundary_state = getattr(without_boundary, state_name)

    _assert_numeric_state_row_close(explicit_state, row, runtime.state)
    assert "observation_unavailable" in runtime.quality_flags
    assert ("observation_failed_qc" in runtime.quality_flags) == (
        quality_status is QualityStatus.FAILED
    )
    assert not torch.allclose(
        explicit_state.memory[row],
        no_boundary_state.memory[row],
        rtol=0,
        atol=1e-7,
    )


def test_planned_treatment_features_never_enter_observed_risk(
    parity_context: _ParityContext,
) -> None:
    patient_id = "SYN-0000"
    query = _query(parity_context, patient_id, Stage.S1)
    original = parity_context.manifests[patient_id]
    planned_id = f"{patient_id}-planned-systemic"

    def with_planned_value(value: float, lineage: str) -> FeatureManifest:
        actions = dict(original.action_features)
        actions[planned_id] = ActionFeature(
            values=torch.full((parity_context.config.model.action_input_dim,), value),
            feature_version="synthetic-action-v1",
            known_exposure=1.0,
        )
        return replace(original, manifest_lineage_id=lineage, action_features=actions)

    low = _fresh_engine(parity_context).replay_query(
        parity_context.firewall,
        with_planned_value(-1000.0, "planned-feature-low"),
        query,
    )
    high = _fresh_engine(parity_context).replay_query(
        parity_context.firewall,
        with_planned_value(1000.0, "planned-feature-high"),
        query,
    )

    torch.testing.assert_close(low.state.memory, high.state.memory, rtol=0, atol=0)
    assert "planned_treatment_not_applied_to_observed_risk" in low.quality_flags
    assert planned_id not in {item.record_id for item in low.input_manifest}


def test_post_ct_action_changes_s1_but_not_ct_acquisition_prior(
    parity_context: _ParityContext,
) -> None:
    patient_id = "SYN-0000"
    batch, row = _batch_row(parity_context, patient_id)
    late_action = batch.s1_update_actions
    assert late_action is not None and bool(late_action.valid[row].any())
    no_late_action = replace(late_action, valid=torch.zeros_like(late_action.valid))

    with_late = _forward(parity_context.engine.model, batch)
    without_late = _forward(
        parity_context.engine.model,
        batch,
        s1_update_actions=no_late_action,
    )

    torch.testing.assert_close(
        with_late.prior_ct1.memory[row], without_late.prior_ct1.memory[row], rtol=0, atol=0
    )
    torch.testing.assert_close(
        with_late.future_ct.mean[row], without_late.future_ct.mean[row], rtol=0, atol=0
    )
    assert not torch.allclose(
        with_late.state_s1.memory[row], without_late.state_s1.memory[row], rtol=0, atol=1e-7
    )


@pytest.mark.parametrize(
    ("patient_id", "stage", "missing_flag"),
    (
        ("SYN-0005", Stage.S1, "observation_unavailable"),
        ("SYN-0002", Stage.S2, "observation_unavailable"),
    ),
)
def test_missing_observation_still_advances_runtime_state_to_query(
    parity_context: _ParityContext,
    patient_id: str,
    stage: Stage,
    missing_flag: str,
) -> None:
    query = _query(parity_context, patient_id, stage)
    replay = parity_context.engine.replay_query(
        parity_context.firewall,
        parity_context.manifests[patient_id],
        query,
    )

    torch.testing.assert_close(
        replay.state.query_time,
        torch.tensor([query.query_time_days], dtype=replay.state.query_time.dtype),
        rtol=0,
        atol=1e-6,
    )
    assert missing_flag in replay.quality_flags


def test_multislide_runtime_merge_matches_training_patient_tokens_without_duplicates(
    parity_context: _ParityContext,
) -> None:
    patient_id = "SYN-0000"
    query = _query(parity_context, patient_id, Stage.S2)
    prefix = parity_context.firewall.build_prefix(patient_id, query)
    feature_events, _, _, _, _ = parity_context.engine._collect_events(  # noqa: SLF001
        prefix, parity_context.manifests[patient_id]
    )
    pathology_events = [
        event for event in feature_events if event.tokens.modality_name == "pathology"
    ]
    assert len(pathology_events) == 2
    merged = parity_context.engine._merge_feature_events(pathology_events)  # noqa: SLF001
    assert len(merged) == 1
    runtime_pathology = merged[0]

    batch, row = _batch_row(parity_context, patient_id)
    training_values = batch.pathology.values[row][batch.pathology.valid[row]]
    runtime_values = runtime_pathology.values[0][runtime_pathology.valid[0]]
    torch.testing.assert_close(runtime_values, training_values, rtol=0, atol=0)
    runtime_sources = runtime_pathology.provenance_ids()[0]
    assert len(runtime_sources) == len(set(runtime_sources)) == training_values.shape[0]


def test_serialized_timeline_contract_matches_cohort_and_action_clocks(
    parity_context: _ParityContext,
) -> None:
    expected = asdict(SYNTHETIC_TIMELINE)
    assert parity_context.source["timeline_contract"] == expected
    manifest = read_json(synthetic_data_root(parity_context.config) / "manifest.json")
    assert manifest["timeline_contract"] == expected

    cohort = load_cohort_json(synthetic_data_root(parity_context.config) / "cohort.json")
    patient_id = "SYN-0000"
    observations = [item for item in cohort.observations if item.patient_id == patient_id]
    ct0 = next(item for item in observations if item.role is ObservationRole.BASELINE_CT)
    ct1 = next(item for item in observations if item.role is ObservationRole.POST_TREATMENT_CT)
    pathology = [
        item for item in observations if item.role is ObservationRole.SURGICAL_PATHOLOGY
    ]
    assert (ct0.acquired_at_days, ct0.available_at_days) == (
        expected["baseline_acquired"],
        expected["s0_query"],
    )
    assert (ct1.acquired_at_days, ct1.available_at_days) == (
        expected["ct1_acquired"],
        expected["ct1_available"],
    )
    assert pathology and all(
        (item.acquired_at_days, item.available_at_days)
        == (expected["pathology_acquired"], expected["pathology_available"])
        for item in pathology
    )

    treatments = [item for item in cohort.treatments if item.patient_id == patient_id]
    delivered_systemic = next(
        item
        for item in treatments
        if item.treatment_kind is TreatmentKind.SYSTEMIC
        and item.planned_or_delivered is TreatmentStatus.DELIVERED
    )
    post_ct = next(item for item in treatments if item.event_id.endswith("-post-ct-treatment"))
    surgery = next(item for item in treatments if item.treatment_kind is TreatmentKind.SURGERY)
    assert (delivered_systemic.end_days, delivered_systemic.available_at_days) == (
        expected["systemic_event"],
        expected["systemic_available"],
    )
    assert (post_ct.end_days, post_ct.available_at_days) == (
        expected["post_ct_event"],
        expected["post_ct_available"],
    )
    assert (surgery.end_days, surgery.available_at_days) == (
        expected["surgery_event"],
        expected["surgery_available"],
    )

    queries = [item for item in cohort.queries if item.patient_id == patient_id]
    query_times = {item.stage: item.query_time_days for item in queries[:3]}
    assert query_times == {
        Stage.S0: expected["s0_query"],
        Stage.S1: expected["s1_query"],
        Stage.S2: expected["s2_query"],
    }
    torch.testing.assert_close(
        torch.as_tensor(parity_context.source["treatment_event_time"])[0],
        torch.tensor([expected["systemic_event"], expected["post_ct_event"]]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        torch.as_tensor(parity_context.source["surgery_available_time"])[0],
        torch.tensor([expected["surgery_available"]]),
        rtol=0,
        atol=0,
    )
