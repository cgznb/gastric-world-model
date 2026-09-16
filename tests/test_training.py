from __future__ import annotations

import csv
import datetime
import time
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from stageworld.encoders.base import EncoderProvenance, ObservationTokens
from stageworld.errors import ArtifactError, ConfigurationError, DataContractError, ResourceError
from stageworld.losses import freeze_module
from stageworld.model import ActionTokens, StageWorldModel, StageWorldModelConfig
from stageworld.training import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointMetadata,
    ExperimentRegistry,
    LossWeights,
    StageWorldTrainer,
    TrainingPhase,
    WorldModelBatch,
    bounded_fit,
    compute_world_model_loss,
)


def _provenance(name: str, dim: int) -> EncoderProvenance:
    return EncoderProvenance(
        encoder_name=f"synthetic_{name}",
        source_version="synthetic-v1",
        component_versions=(("fixture", "v1"),),
        preprocess_version="identity-v1",
        feature_dim=dim,
    )


def _observation(
    name: str,
    values: torch.Tensor,
    valid: torch.Tensor,
    acquired: float,
    available: float,
) -> ObservationTokens:
    batch, tokens, dim = values.shape
    sources = tuple(
        tuple(f"syn-{name}-{row}-{token}" if valid[row, token] else "" for token in range(tokens))
        for row in range(batch)
    )
    return ObservationTokens(
        values=values,
        valid=valid,
        modality=torch.full((batch, tokens), {"ct": 0, "pathology": 1, "clinical": 2}[name]),
        acquired_time=torch.full((batch, tokens), acquired),
        available_time=torch.full((batch, tokens), available),
        provenance=_provenance(name, dim),
        source_id=sources,
        modality_name=name,
    )


def _actions(values: torch.Tensor, event_time: float, event_type: int) -> ActionTokens:
    batch, count, _ = values.shape
    return ActionTokens(
        values=values,
        valid=torch.ones(batch, count, dtype=torch.bool),
        event_time=torch.full((batch, count), event_time),
        available_time=torch.full((batch, count), event_time),
        event_type=torch.full((batch, count), event_type),
        planned_or_delivered=torch.full((batch, count), 2),
        known_exposure=torch.ones(batch, count),
        provenance=(f"synthetic-action-{event_type}",),
    )


def _model() -> StageWorldModel:
    return StageWorldModel(
        StageWorldModelConfig(
            hidden_dim=16,
            state_tokens=3,
            stochastic_dim=2,
            use_stochastic_state=False,
            attention_heads=4,
            transition_blocks=1,
            observation_blocks=1,
            resampler_blocks=1,
            dropout=0.0,
            action_input_dim=5,
            modality_input_dims=(("ct", 8), ("pathology", 6), ("clinical", 4)),
            resampled_tokens=(("ct", 2), ("pathology", 2), ("clinical", 2)),
            future_output_dims=(("ct", 8), ("pathology", 6)),
            future_output_tokens=(("ct", 1), ("pathology", 1)),
            survival_cutpoints=(0.0, 1.0, 3.0),
            model_version="training-test-v1",
        )
    )


def _batch(indices: tuple[int, ...] = (0, 1, 2, 3)) -> WorldModelBatch:
    index = torch.tensor(indices, dtype=torch.float32)
    batch = len(indices)
    base = index[:, None, None]
    ct_axis = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8) / 17
    clinical_axis = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4) / 11
    ct1_axis = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8) / 13
    path_axis = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6) / 9
    ct0_values = torch.sin(base + ct_axis)
    clinical_values = torch.cos(base * 0.3 + clinical_axis)
    ct1_values = torch.tanh(base * 0.4 + ct1_axis)
    pathology_values = torch.sin(base * 0.2 + path_axis)
    ct_valid = torch.ones(batch, 4, dtype=torch.bool)
    clinical_valid = torch.ones(batch, 2, dtype=torch.bool)
    ct1_valid = torch.ones(batch, 3, dtype=torch.bool)
    path_valid = torch.ones(batch, 2, dtype=torch.bool)
    action_values = torch.stack(
        (index, index.square(), index + 1, index * 0.2, torch.ones_like(index)), dim=-1
    )[:, None]
    surgery_values = torch.stack(
        (torch.ones_like(index), index * 0.1, index % 2, index + 0.5, index * 0), dim=-1
    )[:, None]
    future_ct = (ct1_values.mean(dim=1) + 0.1 * ct0_values.mean(dim=1))[:, None]
    future_path = pathology_values.mean(dim=1, keepdim=True)
    future_path_valid = torch.ones(batch, 1, dtype=torch.bool)
    survival_durations = torch.stack(
        (0.35 + index * 0.05, 0.25 + index * 0.04, 0.15 + index * 0.03), dim=1
    )
    survival_events = torch.stack(
        ((index.long() % 2), ((index.long() + 1) % 2), (index.long() % 2)), dim=1
    )
    survival_valid = torch.ones(batch, 3, dtype=torch.bool)
    return WorldModelBatch(
        patient_ids=tuple(f"SYN-{value:04d}" for value in indices),
        ct0=_observation("ct", ct0_values, ct_valid, 0, 0),
        clinical0=_observation("clinical", clinical_values, clinical_valid, 0, 0),
        s0_time=torch.zeros(batch),
        treatment_actions=_actions(action_values, 10, 1),
        ct1_acquisition_time=torch.full((batch,), 30.0),
        ct1=_observation("ct", ct1_values, ct1_valid, 30, 35),
        s1_time=torch.full((batch,), 35.0),
        surgery_actions=_actions(surgery_values, 40, 7),
        pathology_acquisition_time=torch.full((batch,), 45.0),
        pathology=_observation("pathology", pathology_values, path_valid, 45, 48),
        s2_time=torch.full((batch,), 48.0),
        horizons=torch.tensor([0.0, 1.0, 3.0]),
        future_ct_target=future_ct.detach(),
        future_ct_valid=torch.ones(batch, 1, dtype=torch.bool),
        future_pathology_target=future_path.detach(),
        future_pathology_valid=future_path_valid,
        survival_durations=survival_durations,
        survival_events=survival_events,
        survival_valid=survival_valid,
        ct1_availability_time=torch.full((batch,), 35.0),
        ct1_unavailable_event_mask=torch.zeros(batch, dtype=torch.bool),
        pathology_availability_time=torch.full((batch,), 48.0),
        pathology_unavailable_event_mask=torch.zeros(batch, dtype=torch.bool),
    )


def _distributed_missing_batch(indices: tuple[int, ...]) -> WorldModelBatch:
    source = _batch(indices)
    ct_valid = {0: True, 1: True, 2: False, 3: True}
    pathology_valid = {0: False, 1: False, 2: True, 3: True}
    survival_valid = {
        0: (True, False, False),
        1: (True, False, False),
        2: (False, True, True),
        3: (False, True, True),
    }
    return replace(
        source,
        future_ct_valid=torch.tensor([[ct_valid[index]] for index in indices]),
        future_pathology_valid=torch.tensor(
            [[pathology_valid[index]] for index in indices]
        ),
        survival_valid=torch.tensor([survival_valid[index] for index in indices]),
    )


def _distributed_training_worker(
    rank: int,
    world_size: int,
    rendezvous_path: str,
    output_root: str,
) -> None:
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        torch.manual_seed(83)
        model = _model()
        distributed_model = DistributedDataParallel(
            model,
            find_unused_parameters=True,
        )
        optimizer = torch.optim.SGD(distributed_model.parameters(), lr=3e-3)
        trainer = StageWorldTrainer(
            distributed_model,
            optimizer,
            grad_clip_norm=1e6,
        )
        local_indices = (0, 1) if rank == 0 else (2, 3)
        result = trainer.optimizer_step(
            [_distributed_missing_batch(local_indices)],
            phase=TrainingPhase.JOINT_SURVIVAL,
            weights=LossWeights(kl=0.0),
        )
        torch.save(
            {"model_state": model.state_dict(), "result": result},
            Path(output_root) / f"rank-{rank}.pt",
        )
    finally:
        torch.distributed.destroy_process_group()


def _metadata() -> CheckpointMetadata:
    return CheckpointMetadata(
        checkpoint_id="checkpoint-test-v1",
        model_version="training-test-v1",
        endpoint="os",
        mode="synthetic",
        config_lineage_id="config-test-v1",
        data_lineage_id="data-test-v1",
        cohort_artifact_id="cohort-artifact-test-v1",
        split_version="split-test-v1",
        ct_feature_artifact_id="ct-features-test-v1",
        pathology_feature_artifact_id="pathology-features-test-v1",
        timeline_contract_version="timeline-test-v1",
        outcome_contract_version="outcome-test-v1",
        training_seed=17,
        source_schema_version="source-test-v1",
        cohort_schema_version="cohort-test-v1",
        feature_schema_version="features-test-v1",
        phase="joint_survival",
        selection_rule="stage-average-validation-loss-v1",
        parent_checkpoint_id="parent-checkpoint-test-v1",
        parent_weight_version="parent-weights-test-v1",
        parent_phase="world_pretrain",
        parent_config_lineage_id="parent-config-test-v1",
        parent_data_lineage_id="parent-data-test-v1",
        parent_cohort_artifact_id="parent-cohort-artifact-test-v1",
        parent_split_version="parent-split-test-v1",
        parent_ct_feature_artifact_id="parent-ct-features-test-v1",
        parent_pathology_feature_artifact_id="parent-pathology-features-test-v1",
        parent_timeline_contract_version="parent-timeline-test-v1",
        parent_outcome_contract_version="parent-outcome-test-v1",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("checkpoint_id", None),
        ("ct_feature_artifact_id", None),
        ("pathology_feature_artifact_id", 17),
        ("selection_rule", "  "),
    ),
)
def test_checkpoint_metadata_rejects_non_text_required_lineage(
    field: str, value: object
) -> None:
    with pytest.raises(ConfigurationError) as error:
        replace(_metadata(), **{field: value})
    assert error.value.code == "INCOMPLETE_CHECKPOINT_METADATA"


def test_checkpoint_metadata_rejects_unknown_phase() -> None:
    with pytest.raises(ConfigurationError) as error:
        replace(_metadata(), phase="fine_tune")
    assert error.value.code == "INVALID_TRAINING_PHASE"


@pytest.mark.parametrize(
    "field",
    ("parent_ct_feature_artifact_id", "parent_pathology_feature_artifact_id"),
)
def test_checkpoint_metadata_requires_complete_parent_feature_lineage(field: str) -> None:
    with pytest.raises(ConfigurationError) as error:
        replace(_metadata(), **{field: None})
    assert error.value.code == "INCOMPLETE_PARENT_CHECKPOINT_LINEAGE"


def test_composite_loss_records_effective_counts_and_gradients() -> None:
    model = StageWorldModel(replace(_model().config, use_stochastic_state=True))
    report = compute_world_model_loss(
        model,
        _batch(),
        phase=TrainingPhase.JOINT_SURVIVAL,
        weights=LossWeights(kl=0.0),
    )
    assert set(report.components) == {
        "future_ct",
        "future_pathology",
        "kl_ct",
        "kl_pathology",
        "survival_s0",
        "survival_s1",
        "survival_s2",
    }
    assert report.effective_n["future_ct"] == 4
    assert report.effective_n["survival_s2"] == 4
    assert report.diagnostics["ct_target_variance"] > 0
    assert report.diagnostics["ct_prediction_variance"] > 0
    assert report.diagnostics["ct_target_norm"] > 0
    assert report.diagnostics["ct_prediction_norm"] > 0
    assert report.diagnostics["ct_valid_tokens"] == 4
    assert report.diagnostics["ct_valid_fraction"] == 1.0
    assert report.diagnostics["ct_effective_kl_dimensions"] >= 0
    assert report.diagnostics["ct_mean_kl_per_dimension"] >= 0
    assert report.diagnostics["ct_prediction_collapsed"] is False
    assert report.diagnostics["pathology_valid_tokens"] == 4
    assert report.diagnostics["pathology_effective_kl_dimensions"] >= 0
    report.total.backward()
    assert model.transition.blocks[0].ffn[0].weight.grad is not None
    assert model.future_decoders["ct"].output.weight.grad is not None
    assert model.survival_decoder.output[1].weight.grad is not None


def test_training_batch_moves_and_forwards_unavailable_event_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _batch()
    ct1_mask = torch.tensor([True, False, False, True])
    pathology_mask = torch.tensor([False, True, False, True])
    batch = replace(
        source,
        ct1_unavailable_event_mask=ct1_mask,
        pathology_unavailable_event_mask=pathology_mask,
    ).to("cpu")
    model = _model()
    original_forward = model.forward_three_stage
    captured: dict[str, torch.Tensor | None] = {}

    def record_forward(**kwargs: object):  # type: ignore[no-untyped-def]
        captured["ct1"] = kwargs.get("ct1_unavailable_event_mask")  # type: ignore[assignment]
        captured["pathology"] = kwargs.get(  # type: ignore[assignment]
            "pathology_unavailable_event_mask"
        )
        return original_forward(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(model, "forward_three_stage", record_forward)
    compute_world_model_loss(
        model,
        batch,
        phase=TrainingPhase.JOINT_SURVIVAL,
        weights=LossWeights(kl=0.0),
    )

    assert torch.equal(batch.ct1_unavailable_event_mask, ct1_mask)
    assert torch.equal(batch.pathology_unavailable_event_mask, pathology_mask)
    assert torch.equal(captured["ct1"], ct1_mask)
    assert torch.equal(captured["pathology"], pathology_mask)


@pytest.mark.parametrize(
    "field",
    ("ct1_unavailable_event_mask", "pathology_unavailable_event_mask"),
)
def test_training_batch_rejects_invalid_unavailable_event_mask(field: str) -> None:
    with pytest.raises(DataContractError) as error:
        replace(_batch(), **{field: torch.zeros(4)})
    assert error.value.code == "INVALID_UNAVAILABLE_EVENT_MASK"


def test_teacher_is_unchanged_after_two_optimizer_steps() -> None:
    model = _model()
    teacher = freeze_module(nn.Linear(4, 3))
    original = {name: value.clone() for name, value in teacher.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = StageWorldTrainer(model, optimizer, teacher=teacher)
    for _ in range(2):
        result = trainer.optimizer_step(
            [_batch()],
            phase=TrainingPhase.JOINT_SURVIVAL,
            weights=LossWeights(kl=0.0),
        )
        assert result["gradient_norm"] > 0
    for name, value in teacher.state_dict().items():
        assert torch.equal(value, original[name])


def test_future_target_tensors_are_unchanged_after_two_optimizer_steps() -> None:
    model = _model()
    batch = _batch()
    original_ct = batch.future_ct_target.clone()
    original_pathology = batch.future_pathology_target.clone()
    trainer = StageWorldTrainer(model, torch.optim.AdamW(model.parameters(), lr=1e-3))

    for _ in range(2):
        trainer.optimizer_step(
            [batch],
            phase=TrainingPhase.JOINT_SURVIVAL,
            weights=LossWeights(kl=0.0),
        )

    assert torch.equal(batch.future_ct_target, original_ct)
    assert torch.equal(batch.future_pathology_target, original_pathology)
    assert not batch.future_ct_target.requires_grad
    assert not batch.future_pathology_target.requires_grad


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    torch.manual_seed(23)
    first_model = _model()
    transfer_parent_model_state = {
        name: value.detach().clone() for name, value in first_model.state_dict().items()
    }
    first_optimizer = torch.optim.AdamW(first_model.parameters(), lr=2e-3)
    first_scheduler = torch.optim.lr_scheduler.ExponentialLR(first_optimizer, gamma=0.9)
    first = StageWorldTrainer(first_model, first_optimizer, scheduler=first_scheduler)
    first.optimizer_step(
        [_batch()], phase=TrainingPhase.JOINT_SURVIVAL, weights=LossWeights(kl=0.0)
    )
    checkpoint = tmp_path / "checkpoint.pt"
    first.save_checkpoint(
        checkpoint,
        _metadata(),
        sampler_state={"cursor": 4, "epoch": 2},
        transfer_parent_model_state=transfer_parent_model_state,
    )
    first.optimizer_step(
        [_batch()], phase=TrainingPhase.JOINT_SURVIVAL, weights=LossWeights(kl=0.0)
    )
    expected = {name: value.detach().clone() for name, value in first_model.state_dict().items()}

    torch.manual_seed(999)
    resumed_model = _model()
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=2e-3)
    resumed_scheduler = torch.optim.lr_scheduler.ExponentialLR(resumed_optimizer, gamma=0.9)
    resumed = StageWorldTrainer(resumed_model, resumed_optimizer, scheduler=resumed_scheduler)
    sampler = resumed.load_checkpoint(checkpoint, expected=_metadata())
    assert sampler == {"cursor": 4, "epoch": 2}
    assert resumed.state.optimizer_step == 1
    resumed.optimizer_step(
        [_batch()], phase=TrainingPhase.JOINT_SURVIVAL, weights=LossWeights(kl=0.0)
    )
    for name, value in resumed_model.state_dict().items():
        assert torch.equal(value, expected[name]), name
    assert resumed_scheduler.state_dict() == first_scheduler.state_dict()


def test_checkpoint_persists_actual_model_and_survival_contract(tmp_path: Path) -> None:
    assert CHECKPOINT_SCHEMA_VERSION == "stageworld-checkpoint-v4"
    model = _model()
    trainer = StageWorldTrainer(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        survival_time_unit="day",
    )
    checkpoint = trainer.save_checkpoint(
        tmp_path / "checkpoint.pt",
        _metadata(),
        transfer_parent_model_state=model.state_dict(),
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert isinstance(payload["weight_version"], str) and payload["weight_version"]
    snapshot = checkpoint.parent / "checkpoint_versions" / f"{payload['weight_version']}.pt"
    assert snapshot.is_file()
    snapshot_payload = torch.load(snapshot, map_location="cpu", weights_only=True)
    assert snapshot_payload["weight_version"] == payload["weight_version"]
    assert snapshot_payload["metadata"] == payload["metadata"]
    assert snapshot_payload["transfer_parent_model_state"] is not None
    assert payload["metadata"]["ct_feature_artifact_id"] == "ct-features-test-v1"
    assert (
        payload["metadata"]["parent_pathology_feature_artifact_id"]
        == "parent-pathology-features-test-v1"
    )
    assert payload["model_config"] == asdict(model.config)
    assert payload["survival_contract"] == {
        "schema_version": "stageworld-survival-contract-v1",
        "endpoint": "os",
        "parameterization": "piecewise_constant_hazard_rate",
        "time_unit": "day",
        "cutpoints": (0.0, 1.0, 3.0),
        "open_tail_interval": True,
        "num_causes": 1,
    }


def test_checkpoint_resave_preserves_append_only_weight_snapshots(tmp_path: Path) -> None:
    model = _model()
    trainer = StageWorldTrainer(model, torch.optim.AdamW(model.parameters(), lr=1e-3))
    checkpoint = tmp_path / "checkpoint.pt"

    trainer.save_checkpoint(
        checkpoint,
        _metadata(),
        transfer_parent_model_state=model.state_dict(),
    )
    first = torch.load(checkpoint, map_location="cpu", weights_only=True)
    first_snapshot = (
        checkpoint.parent / "checkpoint_versions" / f"{first['weight_version']}.pt"
    )
    trainer.save_checkpoint(
        checkpoint,
        _metadata(),
        transfer_parent_model_state=model.state_dict(),
    )
    second = torch.load(checkpoint, map_location="cpu", weights_only=True)
    second_snapshot = (
        checkpoint.parent / "checkpoint_versions" / f"{second['weight_version']}.pt"
    )

    assert first["weight_version"] != second["weight_version"]
    assert first_snapshot.is_file()
    assert second_snapshot.is_file()
    assert torch.load(first_snapshot, map_location="cpu", weights_only=True)[
        "weight_version"
    ] == first["weight_version"]
    assert torch.load(second_snapshot, map_location="cpu", weights_only=True)[
        "weight_version"
    ] == second["weight_version"]


def test_checkpoint_resume_rejects_compatible_shape_config_masquerade(tmp_path: Path) -> None:
    source_model = _model()
    source = StageWorldTrainer(
        source_model,
        torch.optim.AdamW(source_model.parameters(), lr=1e-3),
    )
    checkpoint = source.save_checkpoint(
        tmp_path / "checkpoint.pt",
        _metadata(),
        transfer_parent_model_state=source_model.state_dict(),
    )

    changed_model = StageWorldModel(replace(_model().config, max_rollout_days=999.0))
    resumed = StageWorldTrainer(
        changed_model,
        torch.optim.AdamW(changed_model.parameters(), lr=1e-3),
    )
    with pytest.raises(ArtifactError) as error:
        resumed.load_checkpoint(checkpoint, expected=_metadata())
    assert error.value.code == "CHECKPOINT_MODEL_CONFIG_MISMATCH"


def test_checkpoint_resume_rejects_time_unit_and_missing_weight_version(tmp_path: Path) -> None:
    model = _model()
    source = StageWorldTrainer(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        survival_time_unit="day",
    )
    checkpoint = source.save_checkpoint(
        tmp_path / "checkpoint.pt",
        _metadata(),
        transfer_parent_model_state=model.state_dict(),
    )

    runtime_model = _model()
    resumed = StageWorldTrainer(
        runtime_model,
        torch.optim.AdamW(runtime_model.parameters(), lr=1e-3),
        survival_time_unit="year",
    )
    with pytest.raises(ArtifactError) as error:
        resumed.load_checkpoint(checkpoint, expected=_metadata())
    assert error.value.code == "CHECKPOINT_SURVIVAL_CONTRACT_MISMATCH"

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload.pop("weight_version")
    missing_weight = tmp_path / "missing-weight-version.pt"
    torch.save(payload, missing_weight)
    with pytest.raises(ArtifactError) as error:
        resumed.load_checkpoint(missing_weight, expected=_metadata())
    assert error.value.code == "CHECKPOINT_WEIGHT_VERSION_MISSING"


def test_gradient_accumulation_matches_equivalent_patient_batch() -> None:
    torch.manual_seed(31)
    full_model = _model()
    split_model = _model()
    split_model.load_state_dict(full_model.state_dict())
    full_optimizer = torch.optim.SGD(full_model.parameters(), lr=3e-3)
    split_optimizer = torch.optim.SGD(split_model.parameters(), lr=3e-3)
    full = StageWorldTrainer(full_model, full_optimizer, grad_clip_norm=1e6)
    split = StageWorldTrainer(split_model, split_optimizer, grad_clip_norm=1e6)
    full.optimizer_step(
        [_batch((0, 1, 2, 3))],
        phase=TrainingPhase.JOINT_SURVIVAL,
        weights=LossWeights(kl=0.0),
    )
    split.optimizer_step(
        [_batch((0, 1)), _batch((2, 3))],
        phase=TrainingPhase.JOINT_SURVIVAL,
        weights=LossWeights(kl=0.0),
    )
    for left, right in zip(full_model.parameters(), split_model.parameters(), strict=True):
        assert torch.allclose(left, right, atol=2e-6, rtol=1e-5)


def test_two_process_ddp_matches_global_batch_with_rank_specific_missingness(
    tmp_path: Path,
) -> None:
    torch.manual_seed(83)
    global_model = _model()
    global_optimizer = torch.optim.SGD(global_model.parameters(), lr=3e-3)
    global_trainer = StageWorldTrainer(global_model, global_optimizer, grad_clip_norm=1e6)
    global_result = global_trainer.optimizer_step(
        [_distributed_missing_batch((0, 1, 2, 3))],
        phase=TrainingPhase.JOINT_SURVIVAL,
        weights=LossWeights(kl=0.0),
    )

    rendezvous_path = tmp_path / "gloo-rendezvous"
    output_root = tmp_path / "rank-results"
    output_root.mkdir()
    process_context = torch.multiprocessing.spawn(
        _distributed_training_worker,
        args=(2, str(rendezvous_path), str(output_root)),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 40.0
    try:
        while not process_context.join(timeout=1.0, grace_period=2.0):
            if time.monotonic() >= deadline:
                pytest.fail("Two-process DDP training did not finish within 40 seconds")
    finally:
        for process in process_context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=2.0)

    outputs: list[Mapping[str, Any]] = [
        torch.load(output_root / f"rank-{rank}.pt", map_location="cpu", weights_only=True)
        for rank in range(2)
    ]
    expected_effective_n = {
        "future_ct": 3,
        "future_pathology": 2,
        "kl_ct": 3,
        "kl_pathology": 2,
        "survival_s0": 2,
        "survival_s1": 2,
        "survival_s2": 2,
    }
    assert global_result["effective_n"] == expected_effective_n
    global_state = global_model.state_dict()
    for output in outputs:
        result = output["result"]
        assert result["effective_n"] == expected_effective_n
        assert result["loss"] == pytest.approx(global_result["loss"], rel=1e-6, abs=1e-7)
        for name, expected in global_result["components"].items():
            assert result["components"][name] == pytest.approx(
                expected,
                rel=1e-6,
                abs=1e-7,
            )
        for name, expected in global_state.items():
            torch.testing.assert_close(
                output["model_state"][name],
                expected,
                rtol=1e-5,
                atol=2e-6,
            )
    for name, first in outputs[0]["model_state"].items():
        torch.testing.assert_close(
            outputs[1]["model_state"][name],
            first,
            rtol=0.0,
            atol=0.0,
        )


def test_trainer_rejects_ddp_without_unused_parameter_detection(tmp_path: Path) -> None:
    rendezvous_path = tmp_path / "single-rank-gloo-rendezvous"
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_path}",
        rank=0,
        world_size=1,
    )
    try:
        model = _model()
        distributed_model = DistributedDataParallel(model)
        optimizer = torch.optim.SGD(distributed_model.parameters(), lr=3e-3)
        with pytest.raises(ConfigurationError) as error:
            StageWorldTrainer(
                distributed_model,
                optimizer,
                grad_clip_norm=1e6,
            )
        assert error.value.code == "DDP_UNUSED_PARAMETER_DETECTION_REQUIRED"
    finally:
        torch.distributed.destroy_process_group()


def test_validation_is_write_protected() -> None:
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = StageWorldTrainer(model, optimizer)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    reports = trainer.evaluate(
        [_batch()], phase=TrainingPhase.JOINT_SURVIVAL, weights=LossWeights(kl=0.0)
    )
    assert len(reports) == 1
    assert reports[0]["effective_n"]["survival_s0"] == 4
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_synthetic_signal_batch_can_be_overfit_within_bounded_budget() -> None:
    torch.manual_seed(47)
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=0)
    trainer = StageWorldTrainer(model, optimizer, grad_clip_norm=5.0)
    weights = LossWeights(survival=0.25, future_ct=1.0, future_pathology=1.0, kl=0.0)
    source = _batch()
    missing_batch = replace(
        source,
        future_ct_valid=torch.tensor([[True], [True], [False], [True]]),
        future_pathology_valid=torch.tensor([[True], [False], [True], [True]]),
        survival_valid=torch.tensor(
            [[True, True, True], [True, True, False], [True, True, True], [True, False, True]]
        ),
    )
    initial_report = trainer.evaluate(
        [missing_batch], phase=TrainingPhase.JOINT_SURVIVAL, weights=weights
    )[0]
    history = bounded_fit(
        trainer,
        [missing_batch],
        phase=TrainingPhase.JOINT_SURVIVAL,
        max_steps=24,
        # This is a fail-safe wall-clock guard; shared hosts can take over one minute.
        max_minutes=5,
        weights=weights,
    )
    final_report = trainer.evaluate(
        [missing_batch], phase=TrainingPhase.JOINT_SURVIVAL, weights=weights
    )[0]
    assert len(history) == 24
    initial_survival = sum(
        initial_report["components"][f"survival_s{index}"] for index in range(3)
    )
    final_survival = sum(
        final_report["components"][f"survival_s{index}"] for index in range(3)
    )
    assert initial_survival > 0
    assert final_survival < initial_survival * 0.9
    assert final_report["total"] < initial_report["total"] * 0.75


def test_no_effective_targets_hard_fails() -> None:
    source = _batch()
    empty = WorldModelBatch(
        **{
            **source.__dict__,
            "future_ct_valid": torch.zeros_like(source.future_ct_valid),
            "future_pathology_valid": torch.zeros_like(source.future_pathology_valid),
            "survival_valid": torch.zeros_like(source.survival_valid),
        }
    )
    model = _model()
    trainer = StageWorldTrainer(model, torch.optim.SGD(model.parameters(), lr=1e-3))
    with pytest.raises(DataContractError, match="no active") as error:
        trainer.optimizer_step([empty], phase=TrainingPhase.JOINT_SURVIVAL)
    assert error.value.code == "NO_EFFECTIVE_TRAINING_TARGETS"


def test_optimizer_converts_out_of_memory_to_structured_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    trainer = StageWorldTrainer(model, torch.optim.SGD(model.parameters(), lr=1e-3))

    def exhaust_memory(*args: object, **kwargs: object) -> None:
        raise torch.OutOfMemoryError("synthetic test OOM")

    monkeypatch.setattr("stageworld.training.compute_world_model_loss", exhaust_memory)
    with pytest.raises(ResourceError) as error:
        trainer.optimizer_step([_batch()], phase=TrainingPhase.JOINT_SURVIVAL)

    assert error.value.code == "CUDA_OUT_OF_MEMORY"
    assert error.value.remediation is not None
    assert "microbatch" in error.value.remediation
    assert all(parameter.grad is None for parameter in model.parameters())


def test_experiment_registry_retains_failed_run(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path / "experiment_registry.csv")
    base = {
        "run_id": "run-synthetic-1",
        "mode": "synthetic",
        "phase": "world_pretrain",
        "seed": 17,
        "config_lineage_id": "config-v1",
        "data_lineage_id": "data-v1",
        "started_at_unix": 1,
    }
    registry.update({**base, "status": "running"})
    registry.update({**base, "status": "failed", "finished_at_unix": 2, "failure_code": "OOM"})
    with (tmp_path / "experiment_registry.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["failure_code"] == "OOM"
