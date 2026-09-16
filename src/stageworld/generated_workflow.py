"""Isolated baseline-conditioned generated-state development experiments."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from stageworld.artifacts import atomic_write_private_json, new_artifact_id, read_json
from stageworld.config import StageWorldConfig
from stageworld.data.baseline_clinical import (
    BASELINE_CLINICAL_SCHEMA,
    encode_baseline,
    fit_clinical_transform,
    read_baseline_rows,
)
from stageworld.data.gastric_roi import offline_network
from stageworld.data.paired_ct import HMACPseudonymizer
from stageworld.data.treatment_regimens import COARSE_TUMOR_100EP_PROTOCOL
from stageworld.data.treatment_summary import attach_treatments
from stageworld.errors import ArtifactError, ConfigurationError
from stageworld.generated_evaluation import (
    evaluate_generated_rates,
    future_diagnostics,
    matched_baseline_inputs,
    nll_by_patient,
    paired_comparisons,
    predict_rates,
)
from stageworld.generated_inference import ct0_packet, export_inference_bundle
from stageworld.generated_training import GeneratedS1Trainer, GeneratedTrainingBatch
from stageworld.model.generated_s1 import READOUT_MODES, GeneratedS1Config, GeneratedS1Model
from stageworld.real_survival import _labelled_splits, _labels, _metadata, fit_ridge_baseline
from stageworld.real_workflow import (
    RealFeatureBundle,
    _private_directory,
    load_real_world_pretrain_batches,
)
from stageworld.synthetic_workflow import _atomic_torch_save, build_model
from stageworld.training import CheckpointMetadata, LocalEventLogger, LossWeights, TrainingPhase

GENERATED_PROTOCOL = "baseline19-generated-s1-os-100ep-v1"
SELECTION_RULE = "s1_pred_validation_nll_v1"


def build_generated_model(config: StageWorldConfig, mode: str) -> GeneratedS1Model:
    with torch.random.fork_rng(devices=[]):
        values = asdict(build_model(config).config)
    values.update(
        model_version="stageworld-generated-s1-v1",
        readout_mode=mode,
        survival_task=config.model.survival_task,
    )
    return GeneratedS1Model(GeneratedS1Config(**values))


def prepare_bundle(config: StageWorldConfig) -> tuple[RealFeatureBundle, dict[str, Any]]:
    compatibility = replace(
        config,
        training=replace(
            config.training,
            development_protocol=COARSE_TUMOR_100EP_PROTOCOL,
        ),
    )
    bundle = attach_treatments(compatibility, load_real_world_pretrain_batches(compatibility))
    ids = {
        p for batches in bundle.batches_by_split.values() for b in batches for p in b.patient_ids
    }
    train_ids = {p for b in bundle.batches_by_split["train"] for p in b.patient_ids}
    if not config.paths.clinical_excel or not config.paths.identity_hmac_key_file:
        raise ConfigurationError(
            code="BASELINE_BINDINGS_REQUIRED", message="Bind baseline workbook."
        )
    path = config.output_root / "clinical_snapshot.json"
    existing = read_json(path) if path.exists() else {}
    if existing.get("schema_version") == BASELINE_CLINICAL_SCHEMA:
        snapshot = existing
        if (
            set(snapshot["rows"]) != ids
            or snapshot["transform"] != fit_clinical_transform(snapshot["rows"], train_ids)
            or snapshot["feature_artifact_id"] != bundle.feature_artifact_id
            or snapshot["split_version"] != bundle.split_version
        ):
            raise ArtifactError(
                code="BASELINE_SNAPSHOT_MISMATCH", message="Baseline snapshot differs."
            )
    else:
        rows, audit = read_baseline_rows(
            Path(config.paths.clinical_excel),
            ids,
            HMACPseudonymizer.from_file(
                Path(config.paths.identity_hmac_key_file), project_root=Path.cwd()
            ),
        )
        snapshot = {
            "schema_version": BASELINE_CLINICAL_SCHEMA,
            "artifact_id": new_artifact_id("baseline19"),
            "rows": rows,
            "transform": fit_clinical_transform(rows, train_ids),
            "audit": audit,
            "feature_artifact_id": bundle.feature_artifact_id,
            "split_version": bundle.split_version,
        }
        atomic_write_private_json(path, snapshot)
        atomic_write_private_json(config.output_root / "baseline_audit.json", audit)
    batches = {
        split: tuple(
            GeneratedTrainingBatch.from_world(
                batch,
                encode_baseline(
                    [snapshot["rows"][p] for p in batch.patient_ids], snapshot["transform"]
                ),
            )
            for batch in values
        )
        for split, values in bundle.batches_by_split.items()
    }
    return replace(bundle, batches_by_split=batches), snapshot


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _trainer(config: StageWorldConfig, model: GeneratedS1Model, path: Path) -> GeneratedS1Trainer:
    path.touch(mode=0o600, exist_ok=False)
    return GeneratedS1Trainer(
        model,
        torch.optim.AdamW(
            model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay
        ),
        device="cuda" if torch.cuda.is_available() else "cpu",
        mixed_precision=config.training.mixed_precision,
        grad_clip_norm=config.training.grad_clip_norm,
        survival_time_unit="year",
        event_logger=LocalEventLogger(path),
    )


def _checkpoint_metadata(
    config: StageWorldConfig,
    bundle: RealFeatureBundle,
    snapshot: dict[str, Any],
    model: GeneratedS1Model,
    phase: TrainingPhase,
    parent: dict[str, Any] | None,
) -> CheckpointMetadata:
    metadata = _metadata(config, bundle, model, phase, parent)
    return replace(
        metadata,
        config_lineage_id=json.dumps(
            {
                "configuration": metadata.config_lineage_id,
                "readout_mode": model.config.readout_mode,
                "clinical_snapshot_id": snapshot["artifact_id"],
                "survival_branch": "S1_pred",
            },
            sort_keys=True,
        ),
        timeline_contract_version=f"{metadata.timeline_contract_version}:{snapshot['artifact_id']}",
        outcome_contract_version=(
            "os-remaining-years-from-generated-target-v1"
            if parent
            else metadata.outcome_contract_version
        ),
    )


def train_phase(
    config: StageWorldConfig,
    bundle: RealFeatureBundle,
    snapshot: dict[str, Any],
    root: Path,
    *,
    phase: TrainingPhase,
    mode: str,
    epochs: int,
    deadline: float,
    parent: dict[str, Any] | None = None,
) -> tuple[GeneratedS1Model, dict[str, Any]]:
    _private_directory(root)
    _private_directory(root / "checkpoint_versions")
    _seed(config.training.seed)
    model = build_generated_model(config, mode)
    if parent is not None:
        model.load_state_dict(parent["model_state"], strict=True)
    trainer = _trainer(config, model, root / "events.jsonl")
    metadata = _checkpoint_metadata(config, bundle, snapshot, model, phase, parent)
    joint = phase is TrainingPhase.JOINT_SURVIVAL
    weights = (
        LossWeights(survival=1, future_ct=0.1, future_pathology=0, kl=0.001)
        if joint
        else LossWeights(survival=0, future_ct=1, future_pathology=0, kl=0.01)
    )
    train = cast(tuple[GeneratedTrainingBatch, ...], bundle.batches_by_split["train"])
    val = cast(tuple[GeneratedTrainingBatch, ...], bundle.batches_by_split["validation"])
    best, selected_epoch, history = math.inf, 0, []
    parent_state = None if parent is None else parent["model_state"]
    started = time.monotonic()
    status = {
        "phase": phase.value,
        "readout_mode": mode,
        "survival_branch": "S1_pred",
        "target_epochs": epochs,
        "steps_per_epoch": len(train),
        "seed": config.training.seed,
        "patient_batch_size": config.training.patient_batch_size,
        "early_stopping_enabled": False,
        "time_guard_minutes": config.training.development_max_minutes,
        "checkpoint_selection": SELECTION_RULE if joint else "fixed_epoch_count",
        "outcomes_used": joint,
        "ct1_input_to_survival": False,
        "test_used": False,
    }
    atomic_write_private_json(root / "schedule.json", status)
    try:
        for epoch in range(1, epochs + 1):
            order = list(range(len(train)))
            random.shuffle(order)
            records = []
            for index in order:
                if time.monotonic() >= deadline:
                    raise ArtifactError(
                        code="GENERATED_TIME_BUDGET_REACHED", message="Time guard reached."
                    )
                record = trainer.optimizer_step(
                    [train[index]],
                    phase=phase,
                    weights=weights,
                    kl_beta=min(1.0, epoch / 5)
                    if joint
                    else min(1.0, (trainer.state.optimizer_step + 1) / 40),
                )
                records.append(record)
            trainer.state.epoch = epoch
            row = {
                "epoch": epoch,
                "optimizer_steps": trainer.state.optimizer_step,
                "training_loss": sum(
                    r["loss"] * train[i].batch_size for r, i in zip(records, order, strict=True)
                )
                / sum(b.batch_size for b in train),
            }
            if joint:
                rates = predict_rates(model, val)
                stage_nll = nll_by_patient(rates, val, model.config.survival_cutpoints).mean(0)
                score = float(stage_nll[1])
                row.update(validation_s0_nll=float(stage_nll[0]), validation_s1_pred_nll=score)
                if score < best:
                    best, selected_epoch = score, epoch
                    trainer.save_checkpoint(
                        root / "selected.pt",
                        metadata,
                        sampler_state={"epoch": epoch, "branch": "S1_pred", "readout_mode": mode},
                        transfer_parent_model_state=parent_state,
                    )
            history.append(row)
            if epoch % 10 == 0 or epoch == epochs:
                trainer.save_checkpoint(
                    root / "final.pt",
                    metadata,
                    sampler_state={"epoch": epoch, "branch": "S1_pred", "readout_mode": mode},
                    transfer_parent_model_state=parent_state,
                )
            atomic_write_private_json(
                root / "progress.json",
                {
                    **status,
                    "status": "running",
                    **row,
                    "selected_epoch": selected_epoch,
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
            atomic_write_private_json(root / "history.json", {"epochs": history})
        summary = {
            **status,
            "status": "completed",
            "completed_epochs": len(history),
            "optimizer_steps": trainer.state.optimizer_step,
            "selected_epoch": selected_epoch,
            "selected_validation_s1_pred_nll": best if joint else None,
            "elapsed_seconds": time.monotonic() - started,
            "stop_reason": "epoch_limit",
        }
        if joint:
            trainer.load_checkpoint(root / "selected.pt", expected=metadata, restore_rng=False)
        atomic_write_private_json(root / "progress.json", summary)
        return model, summary
    except BaseException as error:
        atomic_write_private_json(
            root / "progress.json",
            {
                **status,
                "status": "failed",
                "error_code": getattr(error, "code", type(error).__name__),
                "completed_epochs": len(history),
                "optimizer_steps": trainer.state.optimizer_step,
            },
        )
        raise


def _verify_arm(
    root: Path,
    model: GeneratedS1Model,
    batches: tuple[GeneratedTrainingBatch, ...],
    parent: dict[str, Any],
    rates: torch.Tensor,
    summary: dict[str, Any],
) -> dict[str, Any]:
    selected = torch.load(root / "selected.pt", weights_only=True, map_location="cpu")
    final = torch.load(root / "final.pt", weights_only=True, map_location="cpu")
    history = read_json(root / "history.json")["epochs"]
    chosen = min(history, key=lambda row: row["validation_s1_pred_nll"])
    if (
        chosen["epoch"] != summary["selected_epoch"]
        or selected["sampler_state"]["epoch"] != chosen["epoch"]
        or final["trainer_state"]["optimizer_step"] != summary["optimizer_steps"]
        or final["sampler_state"]["epoch"] != summary["completed_epochs"]
        or summary["optimizer_steps"] != summary["target_epochs"] * summary["steps_per_epoch"]
        or selected["metadata"]["parent_weight_version"] != parent["weight_version"]
        or any(
            not torch.equal(value, parent["model_state"][key])
            for key, value in selected["transfer_parent_model_state"].items()
        )
    ):
        raise ArtifactError(
            code="GENERATED_TRAINING_AUDIT_FAILED", message="Training count or parent differs."
        )
    replay = GeneratedS1Model(GeneratedS1Config(**selected["model_config"]))
    replay.load_state_dict(selected["model_state"], strict=True)
    replay.to(next(model.parameters()).device)
    replay_rates = predict_rates(replay, batches)
    torch.testing.assert_close(replay_rates, rates, rtol=0, atol=0)
    nll = nll_by_patient(replay_rates, batches, model.config.survival_cutpoints).mean(0)
    if abs(float(nll[1]) - summary["selected_validation_s1_pred_nll"]) > 1e-7:
        raise ArtifactError(
            code="GENERATED_SELECTION_REPLAY_FAILED", message="Selected NLL differs."
        )
    return {
        "status": "passed",
        "checkpoint_prediction_exact_replay": True,
        "selection_recomputed": True,
        "training_counts_verified": True,
        "shared_pretraining_weights_exact": True,
        "selected_weight_version": selected["weight_version"],
        "final_weight_version": final["weight_version"],
    }


def run_generated_study(config: StageWorldConfig, *, smoke: bool = False) -> dict[str, Any]:
    config.validate(command="generated-s1-development", supervised=True)
    if (
        config.training.development_protocol != GENERATED_PROTOCOL
        or config.training.checkpoint_selection != SELECTION_RULE
        or config.training.development_patience is not None
        or not config.permissions.allow_long_training
        or config.training.seed != 17
        or config.training.patient_batch_size != 32
        or config.training.development_max_epochs != 100
        or config.survival.time_unit != "year"
        or tuple(config.clinical.development_stages) != ("s0", "s1")
    ):
        raise ConfigurationError(
            code="GENERATED_PROTOCOL_MISMATCH", message="Use the approved protocol."
        )
    _private_directory(config.output_root)
    bundle, snapshot = prepare_bundle(config)
    if not smoke and (
        bundle.training_patient_count != 486 or bundle.validation_patient_count != 105
    ):
        raise ArtifactError(
            code="GENERATED_COHORT_CHANGED", message="Retained cohort count differs."
        )
    if smoke:
        bundle = replace(
            bundle, batches_by_split={k: v[:1] for k, v in bundle.batches_by_split.items()}
        )
    epochs = 1 if smoke else 100
    config = replace(
        config,
        training=replace(
            config.training,
            world_pretrain_steps=epochs * len(bundle.batches_by_split["train"]),
            joint_survival_steps=epochs * len(bundle.batches_by_split["train"]),
        ),
    )
    study_id = new_artifact_id("generated-smoke" if smoke else "generated-study")
    root = config.output_root / study_id
    _private_directory(root)
    atomic_write_private_json(
        config.output_root / ("smoke_current.json" if smoke else "current.json"),
        {"study_id": study_id, "status": "running"},
    )
    assert bundle.treatment_snapshot is not None
    atomic_write_private_json(root / "treatment_snapshot.json", bundle.treatment_snapshot)
    atomic_write_private_json(root / "clinical_snapshot.json", snapshot)
    minutes = config.training.development_max_minutes
    main_deadline = float("inf") if minutes is None else time.monotonic() + minutes * 60
    result: dict[str, Any] = {
        "study_id": study_id,
        "protocol": GENERATED_PROTOCOL,
        "status": "running",
        "smoke": smoke,
        "test_used": False,
        "arms": {},
        "cohort": {s: sum(b.batch_size for b in v) for s, v in bundle.batches_by_split.items()},
        "clinical_schema": BASELINE_CLINICAL_SCHEMA,
        "clinical_snapshot_id": snapshot["artifact_id"],
        "survival_branch": "S1_pred",
        "intermediate_update_workflow": False,
        "inference_condition": "baseline_information_plus_caller_declared_treatment_and_target",
        "training_condition": "retrospective_observed_interval_summary_not_baseline_known_fact",
    }
    try:
        with offline_network():
            world, pretraining = train_phase(
                config,
                bundle,
                snapshot,
                root / "world_pretrain",
                phase=TrainingPhase.WORLD_PRETRAIN,
                mode="history_generated",
                epochs=epochs,
                deadline=main_deadline,
            )
            train = cast(tuple[GeneratedTrainingBatch, ...], bundle.batches_by_split["train"])
            validation = cast(
                tuple[GeneratedTrainingBatch, ...], bundle.batches_by_split["validation"]
            )
            result["pretraining"] = {
                **pretraining,
                "future_ct": future_diagnostics(world, train, validation),
            }
            parent = torch.load(
                root / "world_pretrain/final.pt", map_location="cpu", weights_only=True
            )
            del world
            labelled = _labelled_splits(config, bundle)
            bundle = replace(bundle, batches_by_split=labelled)
            assert bundle.treatment_snapshot is not None
            train = cast(tuple[GeneratedTrainingBatch, ...], labelled["train"])
            validation = cast(tuple[GeneratedTrainingBatch, ...], labelled["validation"])
            result["events"] = {s: int(_labels(v)[1][:, 0].sum()) for s, v in labelled.items()}
            for batches in labelled.values():
                for batch in batches:
                    absolute_end = (
                        batch.survival_durations[:, :2]
                        + torch.stack((batch.s0_time, batch.s1_time), dim=1) / 365.25
                    )
                    torch.testing.assert_close(
                        absolute_end[:, 0], absolute_end[:, 1], rtol=1e-6, atol=1e-6
                    )
            result["landmark_labels_verified"] = True
            predictions = {}
            for mode in READOUT_MODES:
                arm_root = root / mode
                deadline = (
                    main_deadline
                    if mode == "history_generated"
                    else float("inf")
                    if minutes is None
                    else time.monotonic() + minutes * 60
                )
                model, arm = train_phase(
                    config,
                    bundle,
                    snapshot,
                    arm_root,
                    phase=TrainingPhase.JOINT_SURVIVAL,
                    mode=mode,
                    epochs=epochs,
                    deadline=deadline,
                    parent=parent,
                )
                rates = predict_rates(model, validation)
                predictions[mode] = rates
                _atomic_torch_save(
                    arm_root / "predictions.pt",
                    {
                        "branch": "S1_pred",
                        "readout_mode": mode,
                        "rates": rates,
                        "patient_ids": tuple(p for b in validation for p in b.patient_ids),
                    },
                )
                verification = _verify_arm(arm_root, model, validation, parent, rates, arm)
                arm.update(
                    evaluate_generated_rates(
                        rates,
                        train,
                        validation,
                        config.survival.finite_cutpoints,
                        method=mode,
                        prediction_id=new_artifact_id("prediction"),
                        replicates=0 if smoke else config.evaluation.bootstrap_replicates,
                    )
                )
                arm["future_ct"] = future_diagnostics(model, train, validation)
                arm["verification"] = verification
                export_inference_bundle(
                    model,
                    arm_root / "inference.pt",
                    clinical_snapshot=snapshot,
                    cycle_transform=bundle.treatment_snapshot["cycle_transform"],
                    encoder_provenance=validation[0].ct0.provenance,
                    weight_version=verification["selected_weight_version"],
                )
                atomic_write_private_json(arm_root / "summary.json", arm)
                result["arms"][mode] = arm
                atomic_write_private_json(root / "summary.json", result)
                del model
            baseline, ridge_snapshots = fit_ridge_baseline(
                train,
                validation,
                config.survival.finite_cutpoints,
                input_builder=matched_baseline_inputs,
            )
            predictions["ridge"] = baseline
            _atomic_torch_save(
                root / "ridge.pt",
                {
                    "rates": baseline,
                    "snapshots": ridge_snapshots,
                    "input_contract": "CT0_baseline19_declared_scenario_target_no_CT1",
                },
            )
            result["ridge"] = evaluate_generated_rates(
                baseline,
                train,
                validation,
                config.survival.finite_cutpoints,
                method="ridge",
                prediction_id=new_artifact_id("ridge"),
                replicates=0 if smoke else config.evaluation.bootstrap_replicates,
            )
            if not smoke:
                result["paired_comparisons"] = paired_comparisons(
                    predictions,
                    train,
                    validation,
                    config.survival.finite_cutpoints,
                    replicates=config.evaluation.bootstrap_replicates,
                )
            from stageworld.synthetic_workflow import _slice_observation

            sample = validation[0]
            packet = ct0_packet(_slice_observation(sample.ct0, torch.tensor([0])))
            _atomic_torch_save(root / "baseline_only_example.pt", packet)
            patient = sample.patient_ids[0]
            treatment = next(
                r for r in bundle.treatment_snapshot["rows"] if r["patient_id"] == patient
            )
            atomic_write_private_json(
                root / "baseline_only_query.json",
                {
                    "ct0_features": "baseline_only_example.pt",
                    "baseline_clinical": snapshot["rows"][patient],
                    "s0_time_days": float(sample.s0_time[0]),
                    "target_interval_days": float(sample.s1_time[0] - sample.s0_time[0]),
                    "horizons_years": list(config.survival.report_horizons_years),
                    "treatment_scenario": {
                        "description": "development_replay_declared_interval",
                        **{k: treatment[k] for k in ("methods", "named_mentions", "cycles")},
                    },
                },
            )
            result["status"] = "completed"
            atomic_write_private_json(root / "summary.json", result)
            atomic_write_private_json(
                config.output_root / ("smoke_current.json" if smoke else "current.json"),
                {"study_id": study_id, "status": "completed"},
            )
            return {
                "status": "completed",
                "study_id": study_id,
                "smoke": smoke,
                "pretrain_epochs": epochs,
                "joint_epochs_per_arm": epochs,
                "test_used": False,
            }
    except BaseException as error:
        result.update(status="failed", error_code=getattr(error, "code", type(error).__name__))
        atomic_write_private_json(root / "summary.json", result)
        raise
