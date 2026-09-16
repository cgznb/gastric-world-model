"""Two-phase binary training with raw-best selection and complete-epoch recovery."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import torch

from stageworld.artifacts import atomic_write_private_json, new_artifact_id, read_json
from stageworld.binary_endpoints import (
    LABEL_CONTRACT,
    BinaryEndpointBatch,
    BinaryEndpointModel,
    BinaryEndpointTrainer,
    BinaryLossWeights,
    build_binary_model,
)
from stageworld.binary_evaluation import binary_bce, predict_binary_logits
from stageworld.config import StageWorldConfig
from stageworld.ct6_training import EarlyStopState, feature_metrics, verify_phase
from stageworld.errors import ArtifactError
from stageworld.generated_workflow import _seed
from stageworld.real_survival import _metadata
from stageworld.real_workflow import RealFeatureBundle, _private_directory
from stageworld.training import (
    CheckpointMetadata,
    LocalEventLogger,
    TrainingPhase,
    _replace_checkpoint_pointer,
)


def train_binary_phase(
    config: StageWorldConfig,
    bundle: RealFeatureBundle,
    snapshot: dict[str, Any],
    root: Path,
    *,
    phase: TrainingPhase,
    epochs: int,
    parent: dict[str, Any] | None = None,
    resume: bool = False,
) -> tuple[BinaryEndpointModel, dict[str, Any]]:
    _private_directory(root)
    _seed(config.training.seed)
    joint = phase is TrainingPhase.JOINT_ENDPOINTS
    if joint != (parent is not None) or phase not in (
        TrainingPhase.WORLD_PRETRAIN,
        TrainingPhase.JOINT_ENDPOINTS,
    ):
        raise ArtifactError(
            code="BINARY_PHASE_PARENT", message="Bind each joint phase to its own world parent."
        )
    model = build_binary_model(config)
    if parent is not None:
        if (
            parent["metadata"]["split_version"] != bundle.split_version
            or parent["metadata"]["training_seed"] != config.training.seed
            or parent["metadata"]["phase"] != TrainingPhase.WORLD_PRETRAIN.value
            or parent["metadata"]["endpoint"] != "pcr_recurrence"
        ):
            raise ArtifactError(
                code="BINARY_WRONG_PARENT", message="World parent must match this fold and seed."
            )
        model.load_state_dict(parent["model_state"], strict=True)
    train = cast(tuple[BinaryEndpointBatch, ...], bundle.batches_by_split["train"])
    validation = cast(tuple[BinaryEndpointBatch, ...], bundle.batches_by_split["validation"])
    BinaryEndpointTrainer._effective_totals((*train, *validation), phase)
    trainer = BinaryEndpointTrainer(
        model,
        torch.optim.AdamW(
            model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay
        ),
        device="cuda" if torch.cuda.is_available() else "cpu",
        mixed_precision=config.training.mixed_precision,
        grad_clip_norm=config.training.grad_clip_norm,
        event_logger=LocalEventLogger(root / f"{new_artifact_id('attempt')}.jsonl"),
    )
    metadata = replace(
        _metadata(config, bundle, model, phase, parent),
        endpoint="pcr_recurrence",
        outcome_contract_version=LABEL_CONTRACT["schema_version"]
        if joint
        else "no_outcome_world_pretrain",
        selection_rule="inner_validation_mean_endpoint_bce"
        if joint
        else "inner_validation_ct_huber_cosine",
        config_lineage_id=json.dumps(
            {
                "model": asdict(model.config),
                "training": asdict(config.training),
                "input_snapshot_id": snapshot["artifact_id"],
                "label_contract": LABEL_CONTRACT,
            },
            sort_keys=True,
        ),
    )
    if (root / "metadata.json").exists():
        if not resume:
            raise ArtifactError(
                code="BINARY_PHASE_EXISTS", message="Resume existing phases explicitly."
            )
        stored = CheckpointMetadata(**read_json(root / "metadata.json"))
        metadata = replace(metadata, checkpoint_id=stored.checkpoint_id)
        if metadata != stored:
            raise ArtifactError(code="BINARY_RESUME_CONTRACT", message="Resume metadata differs.")
    atomic_write_private_json(root / "metadata.json", asdict(metadata))
    patience = config.training.development_patience
    if patience is None or patience < 1:
        raise ValueError("An early-stopping patience is required")
    status: dict[str, Any] = {
        "phase": phase.value,
        "target_epochs": epochs,
        "steps_per_epoch": len(train),
        "seed": config.training.seed,
        "patience": patience,
        "min_delta": config.training.development_min_delta,
        "time_guard_minutes": config.training.development_max_minutes,
        "checkpoint_selection": metadata.selection_rule,
        "recovery_boundary": "complete_epoch",
        "label_contract": LABEL_CONTRACT,
        "test_used": False,
        "outer_fold_used_for_selection": False,
    }
    if (root / "schedule.json").exists() and read_json(root / "schedule.json") != status:
        raise ArtifactError(code="BINARY_SCHEDULE", message="Resume schedule differs.")
    atomic_write_private_json(root / "schedule.json", status)
    stopper, history, previous_elapsed = EarlyStopState(), [], 0.0
    if resume and (root / "latest.pt").exists():
        recovered = trainer.load_checkpoint(root / "latest.pt", expected=metadata)
        stopper = EarlyStopState(**recovered["early_stop"])
        history, previous_elapsed = recovered["history"], recovered["elapsed_seconds"]
        if stopper.selected_epoch == trainer.state.epoch:
            _replace_checkpoint_pointer(root / "latest.pt", root / "selected.pt")
    started = time.monotonic()
    weights = BinaryLossWeights(
        future_ct=0.1 if joint else 1,
        future_pathology=0,
        kl=0.001 if joint else 0.01,
        pcr=1 if joint else 0,
        recurrence=1 if joint else 0,
    )
    try:
        for epoch in range(trainer.state.epoch + 1, epochs + 1):
            if stopper.stale_epochs >= patience:
                break
            order = list(range(len(train)))
            random.shuffle(order)
            records = []
            for index in order:
                limit = config.training.development_max_minutes
                if (
                    limit is not None
                    and previous_elapsed + time.monotonic() - started >= limit * 60
                ):
                    raise ArtifactError(
                        code="BINARY_TIME_LIMIT", message="Phase time limit reached."
                    )
                records.append(
                    trainer.optimizer_step(
                        [train[index]],
                        phase=phase,
                        weights=weights,
                        kl_beta=min(
                            1.0,
                            (trainer.state.optimizer_step + 1) / (5 * len(train) if joint else 40),
                        ),
                    )
                )
            trainer.state.epoch = epoch
            row: dict[str, Any] = {
                "epoch": epoch,
                "optimizer_steps": trainer.state.optimizer_step,
                "training_loss": sum(
                    r["loss"] * train[i].batch_size for r, i in zip(records, order, strict=True)
                )
                / sum(b.batch_size for b in train),
                "gradient_norm_max": max(r["gradient_norm"] for r in records),
                "training_features": feature_metrics(model, train),
                "validation_features": feature_metrics(model, validation),
            }
            if joint:
                row["training_bce"] = binary_bce(predict_binary_logits(model, train), train)
                row["validation_bce"] = binary_bce(
                    predict_binary_logits(model, validation), validation
                )
            score = (
                sum(row["validation_bce"].values()) / 2
                if joint
                else row["validation_features"]["ct_loss"]
            )
            row["selection_score"] = score
            selected = stopper.update(score, epoch, config.training.development_min_delta)
            history.append(row)
            elapsed = previous_elapsed + time.monotonic() - started
            trainer.save_checkpoint(
                root / "latest.pt",
                metadata,
                sampler_state={
                    "early_stop": asdict(stopper),
                    "history": history,
                    "elapsed_seconds": elapsed,
                },
                transfer_parent_model_state=None if parent is None else parent["model_state"],
            )
            if selected:
                _replace_checkpoint_pointer(root / "latest.pt", root / "selected.pt")
            atomic_write_private_json(root / "history.json", {"epochs": history})
            atomic_write_private_json(
                root / "progress.json",
                {
                    **status,
                    **row,
                    "status": "running",
                    "completed_epochs": len(history),
                    "selected_epoch": stopper.selected_epoch,
                    "elapsed_seconds": elapsed,
                },
            )
        _replace_checkpoint_pointer(root / "latest.pt", root / "final.pt")
        summary = {
            **status,
            "status": "completed",
            "completed_epochs": len(history),
            "optimizer_steps": trainer.state.optimizer_step,
            "selected_epoch": stopper.selected_epoch,
            "selected_score": stopper.best,
            "stop_reason": "early_stopping" if stopper.stale_epochs >= patience else "epoch_limit",
            "elapsed_seconds": previous_elapsed + time.monotonic() - started,
        }
        atomic_write_private_json(root / "history.json", {"epochs": history})
        atomic_write_private_json(root / "progress.json", summary)
        summary["verification"] = verify_phase(root, parent)
        trainer.load_checkpoint(root / "selected.pt", expected=metadata, restore_rng=False)
        replay = (
            sum(binary_bce(predict_binary_logits(model, validation), validation).values()) / 2
            if joint
            else feature_metrics(model, validation)["ct_loss"]
        )
        if not math.isclose(replay, cast(float, stopper.best), rel_tol=1e-6, abs_tol=1e-7):
            raise ArtifactError(
                code="BINARY_SELECTION_REPLAY", message="Selected score differs on replay."
            )
        summary["verification"]["selected_score_replayed"] = True
        atomic_write_private_json(root / "progress.json", summary)
        return model, summary
    except BaseException as error:
        atomic_write_private_json(
            root / "progress.json",
            {
                **status,
                "status": "failed",
                "error_code": getattr(error, "code", type(error).__name__),
                "completed_epochs": trainer.state.epoch,
                "attempted_optimizer_steps": trainer.state.optimizer_step,
            },
        )
        raise
