"""Three independent original-capacity CT6 fits with only generated-S1 survival."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import torch

from stageworld.artifacts import atomic_write_private_json, new_artifact_id, read_json
from stageworld.config import StageWorldConfig
from stageworld.ct6_evaluation import aggregate_seed_metrics, persistence_diagnostics
from stageworld.ct6_training import S1_ONLY_PROTOCOL, train_ct6_phase
from stageworld.ct6_workflow import (
    prepare_ct6_bundle,
    snapshot_once,
    verify_portable,
    verify_smoke_recovery,
)
from stageworld.data.baseline_clinical import CT6_CLINICAL_SCHEMA
from stageworld.data.gastric_roi import offline_network
from stageworld.data.treatment_compact import COMPACT_TREATMENT_SCHEMA, treatment_fields
from stageworld.errors import ArtifactError, ConfigurationError
from stageworld.generated_evaluation import (
    evaluate_s1_rates,
    future_diagnostics,
    predict_s1_rates,
)
from stageworld.generated_inference import ct0_packet, export_inference_bundle
from stageworld.generated_training import GeneratedTrainingBatch
from stageworld.real_survival import _labelled_splits
from stageworld.real_workflow import _private_directory
from stageworld.synthetic_workflow import _atomic_torch_save, _slice_observation
from stageworld.training import TrainingPhase


def validate_protocol(config: StageWorldConfig) -> None:
    config.validate(command="generated-s1-development", supervised=True)
    model = config.model
    training = config.training
    if (
        training.development_protocol != S1_ONLY_PROTOCOL
        or model.survival_task != "s1_pred_only"
        or (model.hidden_dim, model.state_tokens, model.stochastic_dim_per_token) != (128, 8, 8)
        or (model.ct_tokens, model.attention_heads, model.transition_blocks) != (8, 4, 2)
        or (model.observation_blocks, model.resampler_blocks) != (1, 1)
        or not model.use_stochastic_state
        or model.dropout != 0.1
        or model.action_input_dim != 82
        or not model.freeze_foundation_encoders
        or model.ct_input_dim != 768
        or training.comparison_seeds != (17, 43, 97)
        or training.seed != 17
        or training.patient_batch_size != 32
        or training.development_max_epochs != 100
        or training.development_patience != 15
        or training.development_min_delta != 1e-4
        or training.lr != 2e-4
        or training.weight_decay != 0.01
        or training.development_max_minutes is not None
        or config.clinical.development_stages != ("s0", "s1")
        or config.survival.time_unit != "year"
        or not config.permissions.allow_long_training
    ):
        raise ConfigurationError(
            code="S1_ONLY_PROTOCOL", message="Use the approved S1-only recipe."
        )


def run_s1_only_study(
    config: StageWorldConfig, *, smoke: bool = False, resume: Path | None = None
) -> dict[str, Any]:
    validate_protocol(config)
    _private_directory(config.output_root)
    with offline_network():
        bundle, clinical = prepare_ct6_bundle(config)
    if bundle.training_patient_count != 486 or bundle.validation_patient_count != 105:
        raise ArtifactError(
            code="S1_ONLY_COHORT", message="Retain the original development cohort."
        )
    if smoke:
        bundle = replace(
            bundle, batches_by_split={k: v[:1] for k, v in bundle.batches_by_split.items()}
        )
        config = replace(config, training=replace(config.training, development_max_minutes=5))
    epochs = 2 if smoke else 100
    root = (
        config.output_root / new_artifact_id("s1-only-smoke" if smoke else "s1-only-study")
        if resume is None
        else resume.resolve()
    )
    if resume is not None and (root.parent != config.output_root.resolve() or not root.is_dir()):
        raise ArtifactError(
            code="S1_ONLY_RESUME", message="Resume inside the matching output root."
        )
    _private_directory(root)
    assert bundle.treatment_snapshot is not None
    treatment = dict(bundle.treatment_snapshot)
    snapshot_once(
        root / "study_contract.json",
        json.loads(
            json.dumps(
                {
                    "protocol": S1_ONLY_PROTOCOL,
                    "smoke": smoke,
                    "clinical_snapshot_id": clinical["artifact_id"],
                    "treatment_snapshot_id": treatment["artifact_id"],
                    "model_settings": asdict(config.model),
                    "training_settings": asdict(config.training),
                    "survival_settings": asdict(config.survival),
                    "evaluation_settings": asdict(config.evaluation),
                }
            )
        ),
        "s1-only-contract",
    )
    atomic_write_private_json(root / "clinical_snapshot.json", clinical)
    atomic_write_private_json(root / "treatment_snapshot.json", treatment)
    current = config.output_root / ("smoke_current.json" if smoke else "current.json")
    summary: dict[str, Any] = {
        "study_id": root.name,
        "protocol": S1_ONLY_PROTOCOL,
        "status": "running",
        "smoke": smoke,
        "survival_task": "s1_pred_only",
        "survival_branch": "S1_pred",
        "test_used": False,
        "clinical_schema": CT6_CLINICAL_SCHEMA,
        "treatment_schema": COMPACT_TREATMENT_SCHEMA,
        "cycle_counts_used": False,
        "ct1_input_to_survival": False,
        "cohort": {k: sum(b.batch_size for b in v) for k, v in bundle.batches_by_split.items()},
        "pretraining": {},
        "arms": {},
        "training_condition": "retrospective_interval_summary_not_baseline_known_fact",
        "test_role": "previously_opened_internal_holdout_excluded_from_this_study",
    }
    started = time.monotonic()
    labelled = None

    def save() -> None:
        atomic_write_private_json(root / "summary.json", summary)
        atomic_write_private_json(current, {"study_id": root.name, "status": summary["status"]})

    try:
        with offline_network():
            for seed in (17,) if smoke else config.training.comparison_seeds:
                settings = replace(config, training=replace(config.training, seed=seed))
                key = f"original_capacity/seed-{seed}"
                phase_root = root / key
                summary["active_phase"] = f"{key}/world_pretrain"
                save()
                world, report = train_ct6_phase(
                    settings,
                    bundle,
                    clinical,
                    phase_root / "world_pretrain",
                    phase=TrainingPhase.WORLD_PRETRAIN,
                    mode="history_generated",
                    epochs=epochs,
                    resume=resume is not None,
                )
                train = cast(tuple[GeneratedTrainingBatch, ...], bundle.batches_by_split["train"])
                validation = cast(
                    tuple[GeneratedTrainingBatch, ...], bundle.batches_by_split["validation"]
                )
                report["future_ct"] = {
                    **future_diagnostics(world, train, validation),
                    **persistence_diagnostics(validation),
                }
                summary["pretraining"][key] = report
                if smoke:
                    probe = root / "recovery_probe.json"
                    recovery = (
                        read_json(probe)
                        if probe.exists()
                        else verify_smoke_recovery(
                            settings,
                            bundle,
                            clinical,
                            root / "recovery_probe",
                            phase_root / "world_pretrain",
                        )
                    )
                    atomic_write_private_json(probe, recovery)
                    summary["recovery_probe"] = recovery
                parent = torch.load(
                    phase_root / "world_pretrain/selected.pt", weights_only=True, map_location="cpu"
                )
                del world
                if labelled is None:
                    labelled = replace(
                        bundle, batches_by_split=_labelled_splits(config, bundle, stages=(1,))
                    )
                    for batches in labelled.batches_by_split.values():
                        for batch in batches:
                            if (
                                batch.survival_valid[:, (0, 2)].any()
                                or not batch.survival_valid[:, 1].all()
                            ):
                                raise ArtifactError(
                                    code="S1_ONLY_LABELS", message="Invalid active stages."
                                )
                    summary["events"] = {
                        k: sum(int(b.survival_events[:, 1].sum()) for b in v)
                        for k, v in labelled.batches_by_split.items()
                    }
                    summary["landmark_labels_verified"] = True
                train = cast(tuple[GeneratedTrainingBatch, ...], labelled.batches_by_split["train"])
                validation = cast(
                    tuple[GeneratedTrainingBatch, ...], labelled.batches_by_split["validation"]
                )
                arm_key = f"{key}/history_generated"
                arm_root = root / arm_key
                summary["active_phase"] = arm_key
                save()
                model, report = train_ct6_phase(
                    settings,
                    labelled,
                    clinical,
                    arm_root,
                    phase=TrainingPhase.JOINT_SURVIVAL,
                    mode="history_generated",
                    epochs=epochs,
                    parent=parent,
                    resume=resume is not None,
                )
                rates = predict_s1_rates(model, validation)
                _atomic_torch_save(
                    arm_root / "predictions.pt",
                    {
                        "branch": "S1_pred",
                        "survival_task": "s1_pred_only",
                        "rates": rates,
                        "patient_ids": tuple(p for b in validation for p in b.patient_ids),
                    },
                )
                report.update(
                    evaluate_s1_rates(
                        rates,
                        train,
                        validation,
                        config.survival.finite_cutpoints,
                        method=arm_key,
                        prediction_id=new_artifact_id("s1-only-prediction"),
                        replicates=0 if smoke else config.evaluation.bootstrap_replicates,
                    )
                )
                report["future_ct"] = {
                    **future_diagnostics(model, train, validation),
                    **persistence_diagnostics(validation),
                }
                export_inference_bundle(
                    model,
                    arm_root / "inference.pt",
                    clinical_snapshot=clinical,
                    treatment_support=treatment["support"],
                    encoder_provenance=validation[0].ct0.provenance,
                    weight_version=report["verification"]["weight_versions"]["selected"],
                )
                sample = validation[0]
                _atomic_torch_save(
                    arm_root / "ct0.pt",
                    ct0_packet(_slice_observation(sample.ct0, torch.tensor([0]))),
                )
                row = next(r for r in treatment["rows"] if r["patient_id"] == sample.patient_ids[0])
                atomic_write_private_json(
                    arm_root / "query.json",
                    {
                        "ct0_features": "ct0.pt",
                        "baseline_clinical": clinical["rows"][sample.patient_ids[0]],
                        "s0_time_days": float(sample.s0_time[0]),
                        "target_interval_days": float(sample.s1_time[0] - sample.s0_time[0]),
                        "horizons_years": list(config.survival.report_horizons_years),
                        "treatment_scenario": {
                            "description": "declared_development_replay",
                            **treatment_fields(row),
                        },
                    },
                )
                report["independent_inference"] = verify_portable(arm_root, rates)
                atomic_write_private_json(arm_root / "summary.json", report)
                summary["arms"][arm_key] = report
                save()
                del model, parent
            expected = 1 if smoke else 3
            if len(summary["arms"]) != expected or len(summary["pretraining"]) != expected:
                raise ArtifactError(code="S1_ONLY_INCOMPLETE", message="Required fits are missing.")
            summary["seed_metrics"] = aggregate_seed_metrics(
                summary["arms"], profile_prefix="original_capacity/"
            )
            if not summary["seed_metrics"]:
                raise ArtifactError(code="S1_ONLY_AGGREGATE", message="Seed metrics are missing.")
            summary["optimizer_updates_total"] = sum(
                r["optimizer_steps"]
                for r in (*summary["pretraining"].values(), *summary["arms"].values())
            )
            summary["executed_optimizer_updates_total"] = summary[
                "optimizer_updates_total"
            ] + summary.get("recovery_probe", {}).get("executed_optimizer_updates", 0)
            summary["elapsed_seconds"] = time.monotonic() - started
            if smoke and (
                summary["executed_optimizer_updates_total"] > 20 or summary["elapsed_seconds"] > 300
            ):
                raise ArtifactError(code="S1_ONLY_SMOKE_BUDGET", message="Smoke budget exceeded.")
            summary.update(status="completed", active_phase=None)
            save()
            return summary
    except BaseException as error:
        summary.update(status="failed", error_code=getattr(error, "code", type(error).__name__))
        save()
        raise
