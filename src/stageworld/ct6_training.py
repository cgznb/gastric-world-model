"""Bounded CT6 phases with independent selection, stopping and epoch recovery."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from stageworld.artifacts import atomic_write_private_json, new_artifact_id, read_json
from stageworld.config import StageWorldConfig
from stageworld.data.baseline_clinical import CT6_CLINICAL_SCHEMA
from stageworld.errors import ArtifactError
from stageworld.generated_evaluation import (
    nll_by_patient,
    predict_rates,
    predict_s1_rates,
    s1_nll_by_patient,
)
from stageworld.generated_training import (
    GeneratedS1Trainer,
    GeneratedTrainingBatch,
    S1OnlyTrainer,
    observed_supervision,
)
from stageworld.generated_workflow import _checkpoint_metadata, _seed, build_generated_model
from stageworld.losses import future_feature_loss
from stageworld.model.generated_s1 import GeneratedS1Model
from stageworld.real_workflow import RealFeatureBundle, _private_directory
from stageworld.training import (
    CheckpointMetadata,
    LocalEventLogger,
    LossWeights,
    TrainingPhase,
    _masked_state_kl,
    _replace_checkpoint_pointer,
    optimizer_parameter_report,
)

CT6_PROTOCOL = "ct6-drugs-generated-s1-regularized-v1"
CT6_REFERENCE_PROTOCOL = "ct6-drugs-generated-s1-reference-v1"
S1_ONLY_PROTOCOL = "ct6-s1-pred-only-v1"
WORLD_SELECTION = "generated_ct_validation_huber_cosine_v1"
JOINT_SELECTION = "s1_pred_validation_nll_v1"


@dataclass
class EarlyStopState:
    best: float | None = None
    significant_best: float | None = None
    selected_epoch: int = 0
    stale_epochs: int = 0

    def update(self, score: float, epoch: int, min_delta: float) -> bool:
        if not math.isfinite(score) or not math.isfinite(min_delta) or min_delta <= 0:
            raise ArtifactError(code="CT6_SCORE_INVALID", message="Nonfinite selection score.")
        selected = self.best is None or score < self.best
        if selected:
            self.best, self.selected_epoch = score, epoch
        if self.significant_best is None or self.significant_best - score >= min_delta:
            self.significant_best, self.stale_epochs = score, 0
        else:
            self.stale_epochs += 1
        return selected


def build_ct6_model(config: StageWorldConfig, mode: str) -> GeneratedS1Model:
    base = build_generated_model(config, mode)
    return GeneratedS1Model(
        replace(
            base.config,
            clinical_schema_version=CT6_CLINICAL_SCHEMA,
            model_version=(
                "stageworld-generated-s1-only-ct6-v1"
                if config.model.survival_task == "s1_pred_only"
                else "stageworld-generated-s1-ct6-drugs-v1"
            ),
        )
    )


def make_trainer(
    config: StageWorldConfig,
    model: GeneratedS1Model,
    root: Path,
    *,
    joint: bool,
    steps_per_epoch: int,
    max_epochs: int,
) -> GeneratedS1Trainer:
    reference = config.training.development_protocol in (CT6_REFERENCE_PROTOCOL, S1_ONLY_PROTOCOL)
    settings = config.training
    if reference:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay
        )
        scheduler = None
    else:
        decayed: set[int] = set()
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.MultiheadAttention)):
                decayed.update(
                    id(p)
                    for name, p in module.named_parameters(recurse=False)
                    if "weight" in name and p.ndim >= 2
                )
        grouped: dict[tuple[float, float], list[nn.Parameter]] = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            head = name.startswith(("survival_decoder.", "survival_source."))
            lr = settings.lr if not joint or head else settings.joint_backbone_lr
            wd = settings.weight_decay if id(p) in decayed else 0.0
            grouped.setdefault((lr, wd), []).append(p)
        optimizer = torch.optim.AdamW(
            [
                {"params": params, "lr": lr, "weight_decay": wd}
                for (lr, wd), params in grouped.items()
            ]
        )
        warmup = settings.lr_warmup_epochs * steps_per_epoch
        total = max(max_epochs * steps_per_epoch, warmup + 1)

        def multiplier(step: int) -> float:
            low = settings.lr_min_ratio
            if step < warmup:
                return low + (1 - low) * step / max(1, warmup)
            progress = min(1.0, (step - warmup) / (total - warmup))
            return low + (1 - low) * (1 + math.cos(math.pi * progress)) / 2

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    attempt = new_artifact_id("attempt")
    trainer_class = (
        S1OnlyTrainer if model.config.survival_task == "s1_pred_only" else GeneratedS1Trainer
    )
    return trainer_class(
        model,
        optimizer,
        scheduler=scheduler,
        device="cuda" if torch.cuda.is_available() else "cpu",
        mixed_precision=settings.mixed_precision,
        grad_clip_norm=settings.grad_clip_norm,
        survival_time_unit="year",
        event_logger=LocalEventLogger(root / f"{attempt}.jsonl"),
    )


@torch.inference_mode()
def feature_metrics(
    model: GeneratedS1Model, batches: Sequence[GeneratedTrainingBatch]
) -> dict[str, float]:
    model.eval()
    loss, kl, count = 0.0, 0.0, 0
    for batch in batches:
        moved = batch.to(next(model.parameters()).device)
        output = model(**moved.prediction_inputs(deterministic=True))
        supervised = observed_supervision(model, output, moved, deterministic=True)
        loss += (
            float(
                future_feature_loss(
                    supervised.future_ct, moved.future_ct_target, moved.future_ct_valid
                )
            )
            * moved.batch_size
        )
        kl += (
            float(
                _masked_state_kl(
                    supervised.post_update, supervised.pre_update, moved.future_ct_valid.any(1)
                )
            )
            * moved.batch_size
        )
        count += moved.batch_size
    result = {"ct_loss": loss / count, "kl": kl / count}
    if not all(math.isfinite(v) for v in result.values()):
        raise ArtifactError(code="CT6_FEATURE_METRIC_INVALID", message="Nonfinite CT metric.")
    return result


def phase_metadata(
    config: StageWorldConfig,
    bundle: RealFeatureBundle,
    snapshot: dict[str, Any],
    model: GeneratedS1Model,
    phase: TrainingPhase,
    parent: dict[str, Any] | None,
) -> CheckpointMetadata:
    return replace(
        _checkpoint_metadata(config, bundle, snapshot, model, phase, parent),
        selection_rule=JOINT_SELECTION if parent is not None else WORLD_SELECTION,
    )


def verify_phase(root: Path, parent: dict[str, Any] | None) -> dict[str, Any]:
    summary = read_json(root / "progress.json")
    history = read_json(root / "history.json")["epochs"]
    if not history or len(history) != summary["completed_epochs"]:
        raise ArtifactError(code="CT6_HISTORY_COUNT", message="History length differs.")
    stopper = EarlyStopState()
    for epoch, row in enumerate(history, 1):
        if row["epoch"] != epoch or row["optimizer_steps"] != epoch * summary["steps_per_epoch"]:
            raise ArtifactError(code="CT6_UPDATE_COUNT", message="Update count differs.")
        stopper.update(row["selection_score"], epoch, summary["min_delta"])
        if epoch < len(history) and stopper.stale_epochs >= summary["patience"]:
            raise ArtifactError(code="CT6_EARLY_STOP_MISSED", message="Training passed early stop.")
    if summary["stop_reason"] == "early_stopping" and stopper.stale_epochs < summary["patience"]:
        raise ArtifactError(code="CT6_EARLY_STOP_INVALID", message="Invalid stopping decision.")
    if summary["stop_reason"] == "epoch_limit" and len(history) != summary["target_epochs"]:
        raise ArtifactError(code="CT6_EPOCH_LIMIT_INVALID", message="Premature epoch limit.")
    if (
        stopper.selected_epoch != summary["selected_epoch"]
        or stopper.best != summary["selected_score"]
        or summary["optimizer_steps"] != len(history) * summary["steps_per_epoch"]
    ):
        raise ArtifactError(code="CT6_SELECTED_EPOCH", message="Selected epoch differs.")
    versions = {}
    for kind, epoch in (("selected", stopper.selected_epoch), ("final", len(history))):
        payload = torch.load(root / f"{kind}.pt", weights_only=True, map_location="cpu")
        if (
            payload["trainer_state"]["epoch"] != epoch
            or payload["trainer_state"]["optimizer_step"] != epoch * summary["steps_per_epoch"]
        ):
            raise ArtifactError(code="CT6_CHECKPOINT_COUNT", message="Checkpoint count differs.")
        if payload["sampler_state"]["history"] != history[:epoch]:
            raise ArtifactError(
                code="CT6_CHECKPOINT_HISTORY", message="Checkpoint history differs."
            )

        def finite(value: Any) -> bool:
            if isinstance(value, torch.Tensor):
                return bool(torch.isfinite(value).all())
            if isinstance(value, dict):
                return all(finite(v) for v in value.values())
            if isinstance(value, (list, tuple)):
                return all(finite(v) for v in value)
            return not isinstance(value, float) or math.isfinite(value)

        if not finite(payload["model_state"]) or not finite(payload["optimizer_state"]):
            raise ArtifactError(code="CT6_NONFINITE_CHECKPOINT", message="Nonfinite checkpoint.")
        if parent is not None:
            if (
                payload["metadata"]["parent_weight_version"] != parent["weight_version"]
                or payload["transfer_parent_model_state"].keys() != parent["model_state"].keys()
                or any(
                    not torch.equal(value, parent["model_state"][key])
                    for key, value in payload["transfer_parent_model_state"].items()
                )
            ):
                raise ArtifactError(code="CT6_PARENT_CHANGED", message="Parent state differs.")
        versions[kind] = payload["weight_version"]
    return {
        "status": "passed",
        "selection_and_stopping_recomputed": True,
        "model_optimizer_finite": True,
        "actual_training_counts_verified": True,
        "shared_parent_verified": parent is not None,
        "weight_versions": versions,
    }


def train_ct6_phase(
    config: StageWorldConfig,
    bundle: RealFeatureBundle,
    snapshot: dict[str, Any],
    root: Path,
    *,
    phase: TrainingPhase,
    mode: str,
    epochs: int,
    parent: dict[str, Any] | None = None,
    resume: bool = False,
) -> tuple[GeneratedS1Model, dict[str, Any]]:
    _private_directory(root)
    _seed(config.training.seed)
    model = build_ct6_model(config, mode)
    if parent is not None:
        model.load_state_dict(parent["model_state"], strict=True)
    joint = phase is TrainingPhase.JOINT_SURVIVAL
    s1_only = model.config.survival_task == "s1_pred_only"
    train = cast(tuple[GeneratedTrainingBatch, ...], bundle.batches_by_split["train"])
    validation = cast(tuple[GeneratedTrainingBatch, ...], bundle.batches_by_split["validation"])
    if not joint and any(b.survival_valid.any() for b in (*train, *validation)):
        raise ArtifactError(
            code="CT6_WORLD_HAS_OUTCOMES", message="World phase must be unlabelled."
        )
    trainer = make_trainer(
        config, model, root, joint=joint, steps_per_epoch=len(train), max_epochs=epochs
    )
    metadata = phase_metadata(config, bundle, snapshot, model, phase, parent)
    metadata_path = root / "metadata.json"
    if metadata_path.exists():
        if not resume:
            raise ArtifactError(
                code="CT6_PHASE_EXISTS", message="Resume existing phases explicitly."
            )
        stored = CheckpointMetadata(**read_json(metadata_path))
        metadata = replace(metadata, checkpoint_id=stored.checkpoint_id)
        if metadata != stored:
            raise ArtifactError(code="CT6_RESUME_CONTRACT", message="Resume configuration differs.")
    else:
        atomic_write_private_json(metadata_path, asdict(metadata))
    patience = config.training.development_patience
    if patience is None or patience < 1:
        raise ArtifactError(code="CT6_PATIENCE_REQUIRED", message="Enable bounded early stopping.")
    status = {
        "phase": phase.value,
        "readout_mode": mode,
        "survival_branch": "S1_pred",
        "survival_task": model.config.survival_task,
        "target_epochs": epochs,
        "steps_per_epoch": len(train),
        "seed": config.training.seed,
        "patient_batch_size": config.training.patient_batch_size,
        "early_stopping_enabled": True,
        "patience": patience,
        "min_delta": config.training.development_min_delta,
        "time_guard_minutes": config.training.development_max_minutes,
        "checkpoint_selection": metadata.selection_rule,
        "outcomes_used": joint,
        "ct1_input_to_survival": False,
        "test_used": False,
        "clinical_schema": CT6_CLINICAL_SCHEMA,
        "cycle_counts_used": False,
        "optimizer": optimizer_parameter_report(model, trainer.optimizer),
        "recovery_boundary": "complete_epoch",
        "partial_attempt_logs": "not_committed_history",
    }
    if (root / "schedule.json").exists() and read_json(root / "schedule.json") != status:
        raise ArtifactError(code="CT6_SCHEDULE_CHANGED", message="Phase schedule differs.")
    atomic_write_private_json(root / "schedule.json", status)
    stopper, history, previous_elapsed = EarlyStopState(), [], 0.0
    if resume and (root / "latest.pt").exists():
        recovery = trainer.load_checkpoint(root / "latest.pt", expected=metadata)
        stopper = EarlyStopState(**recovery["early_stop"])
        history = recovery["history"]
        previous_elapsed = recovery["elapsed_seconds"]
        if (root / "progress.json").exists():
            previous_elapsed = max(
                previous_elapsed, read_json(root / "progress.json").get("elapsed_seconds", 0)
            )
        if stopper.selected_epoch == trainer.state.epoch:
            _replace_checkpoint_pointer(root / "latest.pt", root / "selected.pt")
    started = time.monotonic()
    parent_state = None if parent is None else parent["model_state"]
    weights = LossWeights(
        survival=1 if joint else 0,
        future_ct=0.1 if joint else 1,
        future_pathology=0,
        kl=0.001 if joint else 0.01,
    )
    try:
        for epoch in range(trainer.state.epoch + 1, epochs + 1):
            if stopper.stale_epochs >= patience:
                break
            order = list(range(len(train)))
            random.shuffle(order)
            records = []
            for index in order:
                if (
                    config.training.development_max_minutes is not None
                    and previous_elapsed + time.monotonic() - started
                    >= config.training.development_max_minutes * 60
                ):
                    raise ArtifactError(
                        code="CT6_TIME_BUDGET", message="Phase time budget reached."
                    )
                record = trainer.optimizer_step(
                    [train[index]],
                    phase=phase,
                    weights=weights,
                    kl_beta=min(
                        1.0, (trainer.state.optimizer_step + 1) / (5 * len(train) if joint else 40)
                    ),
                )
                records.append(record)
            trainer.state.epoch = epoch
            row: dict[str, Any] = {
                "epoch": epoch,
                "optimizer_steps": trainer.state.optimizer_step,
                "training_loss": sum(
                    r["loss"] * train[i].batch_size for r, i in zip(records, order, strict=True)
                )
                / sum(b.batch_size for b in train),
                "learning_rates": [g["lr"] for g in trainer.optimizer.param_groups],
                "learning_rate_position": "next_optimizer_step",
                "gradient_norm_mean": sum(r["gradient_norm"] for r in records) / len(records),
                "gradient_norm_max": max(r["gradient_norm"] for r in records),
                "training_features": feature_metrics(model, train),
                "validation_features": feature_metrics(model, validation),
                "effective_patients": sum(b.batch_size for b in train),
            }
            if joint:
                for name, batches in (("training", train), ("validation", validation)):
                    if s1_only:
                        row[f"{name}_s1_pred_nll"] = float(
                            s1_nll_by_patient(
                                predict_s1_rates(model, batches),
                                batches,
                                model.config.survival_cutpoints,
                            ).mean()
                        )
                    else:
                        nll = nll_by_patient(
                            predict_rates(model, batches), batches, model.config.survival_cutpoints
                        ).mean(0)
                        row[f"{name}_s0_nll"], row[f"{name}_s1_pred_nll"] = map(float, nll)
            score = (
                row["validation_s1_pred_nll"] if joint else row["validation_features"]["ct_loss"]
            )
            row["selection_score"] = score
            selected = stopper.update(score, epoch, config.training.development_min_delta)
            committed_history = [*history, row]
            elapsed = previous_elapsed + time.monotonic() - started
            trainer.save_checkpoint(
                root / "latest.pt",
                metadata,
                sampler_state={
                    "epoch": epoch,
                    "branch": "S1_pred",
                    "readout_mode": mode,
                    "early_stop": asdict(stopper),
                    "history": committed_history,
                    "elapsed_seconds": elapsed,
                },
                transfer_parent_model_state=parent_state,
            )
            history = committed_history
            if selected:
                _replace_checkpoint_pointer(root / "latest.pt", root / "selected.pt")
            atomic_write_private_json(root / "history.json", {"epochs": history})
            atomic_write_private_json(
                root / "progress.json",
                {
                    **status,
                    "status": "running",
                    **row,
                    "completed_epochs": len(history),
                    "selected_epoch": stopper.selected_epoch,
                    "elapsed_seconds": elapsed,
                },
            )
        _replace_checkpoint_pointer(root / "latest.pt", root / "final.pt")
        summary: dict[str, Any] = {
            **status,
            "status": "completed",
            "completed_epochs": len(history),
            "optimizer_steps": trainer.state.optimizer_step,
            "selected_epoch": stopper.selected_epoch,
            "selected_score": stopper.best,
            "selected_validation_s1_pred_nll": stopper.best if joint else None,
            "stop_reason": "early_stopping" if stopper.stale_epochs >= patience else "epoch_limit",
            "elapsed_seconds": previous_elapsed + time.monotonic() - started,
        }
        atomic_write_private_json(root / "history.json", {"epochs": history})
        atomic_write_private_json(root / "progress.json", summary)
        summary["verification"] = verify_phase(root, parent)
        trainer.load_checkpoint(root / "selected.pt", expected=metadata, restore_rng=False)
        if joint and s1_only:
            replay_score = float(
                s1_nll_by_patient(
                    predict_s1_rates(model, validation), validation, model.config.survival_cutpoints
                ).mean()
            )
        elif joint:
            replay_score = float(
                nll_by_patient(
                    predict_rates(model, validation), validation, model.config.survival_cutpoints
                ).mean(0)[1]
            )
        else:
            replay_score = feature_metrics(model, validation)["ct_loss"]
        if not math.isclose(replay_score, cast(float, stopper.best), rel_tol=1e-6, abs_tol=1e-7):
            raise ArtifactError(code="CT6_SELECTION_REPLAY", message="Checkpoint score differs.")
        summary["verification"]["selected_score_replayed"] = True
        atomic_write_private_json(root / "progress.json", summary)
        return model, summary
    except BaseException as error:
        atomic_write_private_json(
            root / "progress.json",
            {
                **status,
                "status": "interrupted"
                if getattr(error, "code", "") == "CT6_TIME_BUDGET"
                else "failed",
                "error_code": getattr(error, "code", type(error).__name__),
                "completed_epochs": len(history),
                "optimizer_steps": len(history) * len(train),
                "attempted_optimizer_steps": trainer.state.optimizer_step,
                "elapsed_seconds": previous_elapsed + time.monotonic() - started,
            },
        )
        raise
