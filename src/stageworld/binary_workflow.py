"""Serial three-seed, five-fold study of the two recorded binary endpoints."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import torch

from stageworld.artifacts import atomic_write_private_json, new_artifact_id
from stageworld.binary_data import (
    make_binary_folds,
    make_fold_data,
    prepare_binary_development,
    unlabelled_bundle,
)
from stageworld.binary_endpoints import (
    BINARY_PROTOCOL,
    ENDPOINTS,
    LABEL_CONTRACT,
    BinaryEndpointBatch,
    BinaryEndpointTrainer,
)
from stageworld.binary_evaluation import evaluate_binary, predict_binary_logits, prevalence_baseline
from stageworld.binary_inference import export_binary_bundle, verify_binary_portable
from stageworld.binary_training import train_binary_phase
from stageworld.config import StageWorldConfig
from stageworld.ct6_evaluation import persistence_diagnostics
from stageworld.ct6_workflow import snapshot_once
from stageworld.data.gastric_roi import offline_network
from stageworld.data.treatment_compact import treatment_fields
from stageworld.errors import ArtifactError, ConfigurationError
from stageworld.generated_evaluation import future_diagnostics
from stageworld.generated_inference import ct0_packet
from stageworld.real_workflow import RealFeatureBundle, _private_directory
from stageworld.synthetic_workflow import _atomic_torch_save, _slice_observation
from stageworld.training import TrainingPhase, checkpoint_payload_mismatches


def validate_binary_protocol(config: StageWorldConfig) -> None:
    config.validate(command="binary-endpoints-development", supervised=False)
    if (
        config.training.development_protocol != BINARY_PROTOCOL
        or config.model.survival_task != "disabled"
        or config.clinical.primary_endpoint != "pcr_recurrence"
        or not config.permissions.allow_long_training
        or (
            config.model.hidden_dim,
            config.model.state_tokens,
            config.model.stochastic_dim_per_token,
        )
        != (128, 8, 8)
        or (config.model.ct_tokens, config.model.attention_heads, config.model.transition_blocks)
        != (8, 4, 2)
        or config.model.dropout != 0.1
        or config.model.action_input_dim != 82
        or not config.model.freeze_foundation_encoders
        or config.training.comparison_seeds != (17, 43, 97)
        or config.training.patient_batch_size != 32
        or config.training.lr != 0.0002
        or config.training.weight_decay != 0.01
        or config.training.grad_clip_norm != 1
        or config.training.development_max_epochs != 100
        or config.training.development_patience != 15
        or config.training.development_min_delta != 0.0001
        or config.training.development_max_minutes is not None
        or config.clinical.pcr_definition != "BJ_1_yes_2_no_recorded_label"
        or config.clinical.recurrence_definition != "CG_0_1_recorded_status_no_time_window"
    ):
        raise ConfigurationError(
            code="BINARY_PROTOCOL", message="Use the recorded-label five-fold recipe."
        )


def verify_binary_recovery(
    config: StageWorldConfig,
    bundle: RealFeatureBundle,
    snapshot: dict[str, Any],
    root: Path,
    reference: Path,
) -> dict[str, Any]:
    original = BinaryEndpointTrainer.optimizer_step

    def interrupt(self: BinaryEndpointTrainer, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        if self.state.optimizer_step == 2:
            raise ArtifactError(
                code="BINARY_SMOKE_INTERRUPT", message="Intentional partial-epoch interruption."
            )
        return result

    if not (root / "latest.pt").exists():
        with patch.object(BinaryEndpointTrainer, "optimizer_step", interrupt):
            try:
                train_binary_phase(
                    config, bundle, snapshot, root, phase=TrainingPhase.WORLD_PRETRAIN, epochs=2
                )
            except ArtifactError as error:
                if error.code != "BINARY_SMOKE_INTERRUPT":
                    raise
            else:
                raise AssertionError("Recovery probe was not interrupted")
    recovered, _ = train_binary_phase(
        config, bundle, snapshot, root, phase=TrainingPhase.WORLD_PRETRAIN, epochs=2, resume=True
    )
    del recovered
    expected = torch.load(reference / "final.pt", weights_only=True, map_location="cpu")
    actual = torch.load(root / "final.pt", weights_only=True, map_location="cpu")
    for key in (
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "rng_state",
        "trainer_state",
    ):
        if checkpoint_payload_mismatches(expected[key], actual[key]):
            raise ArtifactError(
                code="BINARY_RECOVERY_REPLAY", message="Interrupted epoch recovery differs."
            )
    return {"status": "passed", "model_optimizer_rng_exact": True, "executed_optimizer_updates": 3}


def summarize_seeds(seeds: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for endpoint in ENDPOINTS:
        for metric in (
            "bce",
            "auroc",
            "auprc",
            "brier",
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
        ):
            values = [
                row["oof"]["endpoints"][endpoint]["metrics"][metric]["estimate"]
                for row in seeds.values()
            ]
            present = [value for value in values if value is not None]
            result.append(
                {
                    "endpoint": endpoint,
                    "metric": metric,
                    "n_seeds": len(present),
                    "mean": statistics.mean(present) if present else None,
                    "standard_deviation": statistics.stdev(present) if len(present) > 1 else None,
                }
            )
    return result


def run_binary_study(
    config: StageWorldConfig,
    *,
    smoke: bool = False,
    resume: Path | None = None,
    audit_only: bool = False,
) -> dict[str, Any]:
    validate_binary_protocol(config)
    _private_directory(config.output_root)
    with offline_network():
        data = prepare_binary_development(config)
    folds = snapshot_once(
        config.output_root / "folds.json", make_binary_folds(data.labels), "binary-folds"
    )
    audit = {
        "status": "audited_not_trained",
        "pool": folds["pool"],
        "folds": [{"fold": row["fold"], "counts": row["counts"]} for row in folds["folds"]],
        "label_contract": LABEL_CONTRACT,
        "test_used": False,
    }
    atomic_write_private_json(config.output_root / "input_audit.json", audit)
    if audit_only:
        return audit
    root = (
        config.output_root / new_artifact_id("binary-smoke" if smoke else "binary-study")
        if resume is None
        else resume.resolve()
    )
    if resume is not None and (root.parent != config.output_root.resolve() or not root.is_dir()):
        raise ArtifactError(
            code="BINARY_RESUME_PATH", message="Resume in this project's output root."
        )
    _private_directory(root)
    snapshot_once(
        root / "study_contract.json",
        json.loads(
            json.dumps(
                {
                    "protocol": BINARY_PROTOCOL,
                    "model": asdict(config.model),
                    "training": asdict(config.training),
                    "label_contract": LABEL_CONTRACT,
                    "fold_artifact_id": folds["artifact_id"],
                    "source_id": data.source_id,
                    "smoke": smoke,
                    "selection_policy": "inner_validation_only_selected_model_no_refit",
                }
            )
        ),
        "binary-study-contract",
    )
    summary: dict[str, Any] = {
        "study_id": root.name,
        "status": "running",
        "smoke": smoke,
        "protocol": BINARY_PROTOCOL,
        "label_contract": LABEL_CONTRACT,
        "pool": folds["pool"],
        "seeds": {},
        "fits": {},
        "test_used": False,
        "test_role": "previously_opened_internal_holdout_excluded",
        "outer_selection_used": False,
        "survival_enabled": False,
    }
    current = config.output_root / ("smoke_current.json" if smoke else "current.json")
    started = time.monotonic()

    def save() -> None:
        atomic_write_private_json(root / "summary.json", summary)
        atomic_write_private_json(
            current,
            {
                "study_id": root.name,
                "status": summary["status"],
                "active_fit": summary.get("active_fit"),
            },
        )

    try:
        with offline_network():
            for seed in (17,) if smoke else config.training.comparison_seeds:
                settings = replace(config, training=replace(config.training, seed=seed))
                fold_predictions = []
                for fold in folds["folds"][:1] if smoke else folds["folds"]:
                    key = f"seed-{seed}/fold-{fold['fold']}"
                    target = root / key
                    bundle, outer, snapshot = make_fold_data(
                        settings, data, fold, root / f"fold-{fold['fold']}", smoke=smoke
                    )
                    unlabelled = unlabelled_bundle(bundle)
                    summary["active_fit"] = f"{key}/world_pretrain"
                    save()
                    world, world_report = train_binary_phase(
                        settings,
                        unlabelled,
                        snapshot,
                        target / "world_pretrain",
                        phase=TrainingPhase.WORLD_PRETRAIN,
                        epochs=2 if smoke else 100,
                        resume=resume is not None,
                    )
                    del world
                    if smoke:
                        summary["recovery_probe"] = verify_binary_recovery(
                            settings,
                            unlabelled,
                            snapshot,
                            root / "recovery_probe",
                            target / "world_pretrain",
                        )
                        atomic_write_private_json(
                            root / "recovery_probe.json", summary["recovery_probe"]
                        )
                    parent = torch.load(
                        target / "world_pretrain/selected.pt", weights_only=True, map_location="cpu"
                    )
                    summary["active_fit"] = f"{key}/joint_endpoints"
                    save()
                    model, joint_report = train_binary_phase(
                        settings,
                        bundle,
                        snapshot,
                        target / "joint_endpoints",
                        phase=TrainingPhase.JOINT_ENDPOINTS,
                        epochs=2 if smoke else 100,
                        parent=parent,
                        resume=resume is not None,
                    )
                    train = cast(tuple[BinaryEndpointBatch, ...], bundle.batches_by_split["train"])
                    logits = predict_binary_logits(model, outer)
                    probabilities = logits.sigmoid()
                    labels = torch.cat([b.endpoint_labels for b in outer])
                    valid = torch.cat([b.endpoint_valid for b in outer])
                    ids = tuple(p for batch in outer for p in batch.patient_ids)
                    baseline = prevalence_baseline(train, len(ids))
                    prediction = {
                        "patient_ids": ids,
                        "logits": logits,
                        "probabilities": probabilities,
                        "labels": labels,
                        "valid": valid,
                        "prevalence_baseline": baseline,
                        "label_contract": LABEL_CONTRACT,
                        "fold": fold["fold"],
                        "seed": seed,
                    }
                    _atomic_torch_save(target / "outer_predictions.pt", prediction)
                    report = {
                        "world_pretrain": world_report,
                        "joint_endpoints": joint_report,
                        "outer": evaluate_binary(probabilities, labels, valid, replicates=0),
                        "prevalence_baseline": evaluate_binary(
                            baseline, labels, valid, replicates=0
                        ),
                        "future_ct": {
                            **future_diagnostics(model, train, outer),
                            **persistence_diagnostics(outer),
                        },
                    }
                    export_binary_bundle(
                        model,
                        target / "inference.pt",
                        snapshot=snapshot,
                        encoder_provenance=outer[0].ct0.provenance,
                        weight_version=joint_report["verification"]["weight_versions"]["selected"],
                    )
                    sample = outer[0]
                    patient = sample.patient_ids[0]
                    _atomic_torch_save(
                        target / "ct0.pt",
                        ct0_packet(_slice_observation(sample.ct0, torch.tensor([0]))),
                    )
                    atomic_write_private_json(
                        target / "query.json",
                        {
                            "ct0_features": "ct0.pt",
                            "baseline_clinical": data.clinical[patient],
                            "s0_time_days": float(sample.s0_time[0]),
                            "target_interval_days": float(sample.s1_time[0] - sample.s0_time[0]),
                            "treatment_scenario": {
                                "description": "declared_development_replay",
                                **treatment_fields(data.treatments[patient]),
                            },
                        },
                    )
                    report["portable_inference"] = verify_binary_portable(target, probabilities)
                    atomic_write_private_json(target / "summary.json", report)
                    summary["fits"][key] = report
                    fold_predictions.append(prediction)
                    save()
                    del model, parent
                patient_ids = tuple(
                    p for prediction in fold_predictions for p in prediction["patient_ids"]
                )
                if len(patient_ids) != len(set(patient_ids)) or (
                    not smoke and set(patient_ids) != set(data.labels)
                ):
                    raise ArtifactError(
                        code="BINARY_OOF_COVERAGE",
                        message="OOF patients must occur exactly once per seed.",
                    )
                pooled = {
                    name: torch.cat([prediction[name] for prediction in fold_predictions])
                    for name in (
                        "logits",
                        "probabilities",
                        "labels",
                        "valid",
                        "prevalence_baseline",
                    )
                }
                _atomic_torch_save(
                    root / f"seed-{seed}/oof_predictions.pt",
                    {**pooled, "patient_ids": patient_ids, "label_contract": LABEL_CONTRACT},
                )
                summary["seeds"][str(seed)] = {
                    "patients": len(patient_ids),
                    "oof_patient_coverage_verified": True,
                    "oof": evaluate_binary(
                        pooled["probabilities"],
                        pooled["labels"],
                        pooled["valid"],
                        replicates=0 if smoke else config.evaluation.bootstrap_replicates,
                    ),
                    "prevalence_baseline": evaluate_binary(
                        pooled["prevalence_baseline"],
                        pooled["labels"],
                        pooled["valid"],
                        replicates=0 if smoke else config.evaluation.bootstrap_replicates,
                    ),
                }
                save()
        summary["seed_metrics"] = summarize_seeds(summary["seeds"])
        summary["optimizer_updates"] = sum(
            report[phase]["optimizer_steps"]
            for report in summary["fits"].values()
            for phase in ("world_pretrain", "joint_endpoints")
        )
        summary["elapsed_seconds"] = time.monotonic() - started
        if smoke and (summary["optimizer_updates"] + 3 > 20 or summary["elapsed_seconds"] > 300):
            raise ArtifactError(
                code="BINARY_SMOKE_BUDGET", message="Real-feature smoke exceeded its budget."
            )
        expected = 1 if smoke else 15
        if len(summary["fits"]) != expected:
            raise ArtifactError(
                code="BINARY_FIT_COUNT", message="Required seed/fold fits are missing."
            )
        summary.update(status="completed", active_fit=None)
        save()
        return summary
    except BaseException as error:
        summary.update(status="failed", error_code=getattr(error, "code", type(error).__name__))
        save()
        raise
