"""Isolated, resumable CT6 study using original paired features read-only."""

from __future__ import annotations

import builtins
import io
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import torch

from stageworld.artifacts import atomic_write_private_json, new_artifact_id, read_json
from stageworld.config import StageWorldConfig
from stageworld.ct6_evaluation import (
    aggregate_seed_metrics,
    calibration_report,
    persistence_diagnostics,
    seed_averaged_comparisons,
)
from stageworld.ct6_training import CT6_PROTOCOL, CT6_REFERENCE_PROTOCOL, train_ct6_phase
from stageworld.data.baseline_clinical import (
    CT6_CLINICAL_SCHEMA,
    encode_baseline,
    fit_clinical_transform,
    read_baseline_rows,
)
from stageworld.data.gastric_roi import offline_network
from stageworld.data.paired_ct import HMACPseudonymizer
from stageworld.data.treatment_compact import (
    COMPACT_ACTION_DIM,
    COMPACT_TREATMENT_SCHEMA,
    compact_actions,
    encode_compact,
    fit_name_support,
    read_compact_rows,
    summarize_compact,
    treatment_fields,
)
from stageworld.data.treatment_summary import require_treatment_cutoff
from stageworld.errors import ArtifactError, ConfigurationError
from stageworld.generated_evaluation import (
    evaluate_generated_rates,
    future_diagnostics,
    matched_baseline_inputs,
    paired_comparisons,
    predict_rates,
)
from stageworld.generated_inference import (
    ct0_packet,
    export_inference_bundle,
    predict_generated_query,
)
from stageworld.generated_training import GeneratedS1Trainer, GeneratedTrainingBatch
from stageworld.model.generated_s1 import READOUT_MODES
from stageworld.real_survival import _labelled_splits, _labels, fit_ridge_baseline
from stageworld.real_workflow import (
    RealFeatureBundle,
    _private_directory,
    load_real_world_pretrain_batches,
)
from stageworld.synthetic_workflow import _atomic_torch_save, _slice_observation
from stageworld.training import TrainingPhase


def snapshot_once(path: Path, value: dict[str, Any], prefix: str) -> dict[str, Any]:
    if path.exists():
        stored = read_json(path)
        if {k: v for k, v in stored.items() if k != "artifact_id"} != value:
            raise ArtifactError(code="CT6_SNAPSHOT_CHANGED", message="Bound data snapshot changed.")
        return stored
    result = {"artifact_id": new_artifact_id(prefix), **value}
    atomic_write_private_json(path, result)
    return result


def prepare_ct6_bundle(config: StageWorldConfig) -> tuple[RealFeatureBundle, dict[str, Any]]:
    require_treatment_cutoff(config)
    bundle = load_real_world_pretrain_batches(config)
    split_ids = {
        k: [p for b in v for p in b.patient_ids] for k, v in bundle.batches_by_split.items()
    }
    if set(split_ids) != {"train", "validation"} or set(split_ids["train"]) & set(
        split_ids["validation"]
    ):
        raise ArtifactError(code="CT6_SPLIT_INVALID", message="Use the original development split.")
    ids, training_ids = set(split_ids["train"] + split_ids["validation"]), set(split_ids["train"])
    if not config.paths.clinical_excel or not config.paths.identity_hmac_key_file:
        raise ConfigurationError(code="CT6_BINDINGS_REQUIRED", message="Bind private inputs.")
    pseudonymizer = HMACPseudonymizer.from_file(
        Path(config.paths.identity_hmac_key_file), project_root=Path.cwd()
    )
    rows, audit = read_baseline_rows(
        Path(config.paths.clinical_excel), ids, pseudonymizer, schema_version=CT6_CLINICAL_SCHEMA
    )
    binding = {
        "feature_artifact_id": bundle.feature_artifact_id,
        "split_version": bundle.split_version,
        "data_lineage_id": bundle.data_lineage_id,
        "cohort_artifact_id": bundle.cohort_artifact_id,
        "split_patient_ids": split_ids,
    }
    clinical = snapshot_once(
        config.output_root / "clinical_snapshot.json",
        {
            **binding,
            "schema_version": CT6_CLINICAL_SCHEMA,
            "rows": rows,
            "audit": audit,
            "transform": fit_clinical_transform(
                rows, training_ids, schema_version=CT6_CLINICAL_SCHEMA
            ),
        },
        "ct6-clinical",
    )
    treatments = read_compact_rows(Path(config.paths.clinical_excel), ids, pseudonymizer)
    support = fit_name_support(treatments, training_ids)
    treatment = snapshot_once(
        config.output_root / "treatment_snapshot.json",
        {
            **binding,
            "schema_version": COMPACT_TREATMENT_SCHEMA,
            "rows": treatments,
            "support": support,
            "aggregate": summarize_compact(treatments, support),
            "cutoff": config.clinical.treatment_summary_cutoff,
        },
        "ct6-treatment",
    )
    by_id = {r["patient_id"]: r for r in treatment["rows"]}
    batches = {}
    unseen_counts = {}
    for split, original in bundle.batches_by_split.items():
        converted = []
        unseen_counts[split] = 0
        for batch in original:
            values, flags = encode_compact([by_id[p] for p in batch.patient_ids], support)
            unseen_counts[split] += sum(bool(f) for f in flags)
            base = replace(
                batch,
                treatment_actions=compact_actions(
                    values,
                    batch.ct1_acquisition_time,
                    provenance="retrospective_observed_interval_condition_not_baseline_fact",
                ),
            )
            converted.append(
                GeneratedTrainingBatch.from_world(
                    base,
                    encode_baseline([rows[p] for p in batch.patient_ids], clinical["transform"]),
                )
            )
        batches[split] = tuple(converted)
    atomic_write_private_json(
        config.output_root / "input_audit.json",
        {
            "clinical": audit,
            "treatment": treatment["aggregate"],
            "clinical_transform": clinical["transform"],
            "treatment_support": support,
            "cohort": {k: len(v) for k, v in split_ids.items()},
            "patients_with_unseen_names": unseen_counts,
            "test_used": False,
            "original_patient_split_retained": True,
        },
    )
    return replace(bundle, batches_by_split=batches, treatment_snapshot=treatment), clinical


def verify_portable(root: Path, expected_rates: torch.Tensor) -> dict[str, Any]:
    allowed = {root / name for name in ("inference.json", "inference.pt", "query.json", "ct0.pt")}
    allowed = {p.resolve() for p in allowed}
    opened: set[Path] = set()

    def guarded(original: Any) -> Any:
        def open_file(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            if isinstance(file, (str, Path)) and "r" in mode:
                path = Path(file).resolve()
                if path not in allowed:
                    raise ArtifactError(
                        code="CT6_UNAPPROVED_INFERENCE_READ", message="Input outside allowlist."
                    )
                opened.add(path)
            return original(file, mode, *args, **kwargs)

        return open_file

    with (
        patch.object(builtins, "open", guarded(builtins.open)),
        patch.object(io, "open", guarded(io.open)),
    ):
        predict_generated_query(
            root / "inference.json", root / "query.json", root / "independent.json"
        )
    actual = torch.tensor(read_json(root / "independent.json")["rates"]).squeeze(-1)
    stage_index = 0 if expected_rates.shape[1] == 1 else 1
    expected = expected_rates[:1, stage_index]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    if opened != allowed:
        raise ArtifactError(code="CT6_INFERENCE_INPUT_COVERAGE", message="Portable inputs differ.")
    return {
        "status": "passed",
        "input_file_count": len(opened),
        "ct1_outcome_other_reads_denied": True,
        "maximum_rate_difference": float((actual - expected).abs().max()),
    }


def experiment_config(config: StageWorldConfig, profile: str, seed: int) -> StageWorldConfig:
    config = replace(config, training=replace(config.training, seed=seed))
    if profile == "reference":
        config = replace(
            config,
            model=replace(
                config.model,
                hidden_dim=128,
                state_tokens=8,
                stochastic_dim_per_token=8,
                ct_tokens=8,
                transition_blocks=2,
                dropout=0.1,
            ),
            training=replace(
                config.training,
                development_protocol=CT6_REFERENCE_PROTOCOL,
                lr=2e-4,
                weight_decay=0.01,
            ),
        )
    return config


def verify_smoke_recovery(
    config: StageWorldConfig,
    bundle: RealFeatureBundle,
    clinical: dict[str, Any],
    root: Path,
    reference_root: Path,
) -> dict[str, Any]:
    original = GeneratedS1Trainer.optimizer_step
    attempted = 0

    def interrupt_after_update(self: GeneratedS1Trainer, *args: Any, **kwargs: Any) -> Any:
        nonlocal attempted
        result = original(self, *args, **kwargs)
        attempted += 1
        if self.state.optimizer_step == 2:
            raise ArtifactError(
                code="CT6_RECOVERY_PROBE", message="Intentional smoke interruption."
            )
        return result

    arguments: dict[str, Any] = {
        "phase": TrainingPhase.WORLD_PRETRAIN,
        "mode": "history_generated",
        "epochs": 2,
    }
    with patch.object(GeneratedS1Trainer, "optimizer_step", interrupt_after_update):
        try:
            train_ct6_phase(config, bundle, clinical, root, **arguments)
        except ArtifactError as error:
            if error.code != "CT6_RECOVERY_PROBE":
                raise
        else:
            raise ArtifactError(
                code="CT6_RECOVERY_PROBE_MISSED", message="Probe did not interrupt."
            )
    interrupted = read_json(root / "progress.json")
    if interrupted["completed_epochs"] != 1 or interrupted["attempted_optimizer_steps"] != 2:
        raise ArtifactError(code="CT6_RECOVERY_BOUNDARY", message="Incomplete epoch was committed.")
    model, report = train_ct6_phase(config, bundle, clinical, root, resume=True, **arguments)
    del model
    expected = torch.load(reference_root / "final.pt", weights_only=True, map_location="cpu")
    actual = torch.load(root / "final.pt", weights_only=True, map_location="cpu")

    def identical(left: Any, right: Any) -> bool:
        if isinstance(left, torch.Tensor):
            return isinstance(right, torch.Tensor) and torch.equal(left, right)
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(identical(left[k], right[k]) for k in left)
        if isinstance(left, (list, tuple)):
            return len(left) == len(right) and all(
                identical(a, b) for a, b in zip(left, right, strict=True)
            )
        return bool(left == right)

    checked = ("model_state", "optimizer_state", "scheduler_state", "rng_state", "trainer_state")
    if not all(identical(expected[k], actual[k]) for k in checked):
        raise ArtifactError(code="CT6_RECOVERY_REPLAY", message="Resumed training state differs.")
    if expected["sampler_state"]["history"] != actual["sampler_state"]["history"]:
        raise ArtifactError(code="CT6_RECOVERY_HISTORY", message="Resumed metrics differ.")
    return {
        "status": "passed",
        "checked_states": list(checked),
        "executed_optimizer_updates": attempted + report["optimizer_steps"] - 1,
    }


def run_ct6_study(
    config: StageWorldConfig, *, smoke: bool = False, resume: Path | None = None
) -> dict[str, Any]:
    config.validate(command="generated-s1-development", supervised=True)
    if (
        config.training.development_protocol != CT6_PROTOCOL
        or config.training.development_patience != 15
        or config.training.development_max_epochs != 100
        or config.model.action_input_dim != COMPACT_ACTION_DIM
        or tuple(config.training.comparison_seeds) != (17, 43, 97)
        or config.training.patient_batch_size != 32
        or config.training.seed != 17
        or config.survival.time_unit != "year"
        or not config.permissions.allow_long_training
        or tuple(config.clinical.development_stages) != ("s0", "s1")
    ):
        raise ConfigurationError(
            code="CT6_PROTOCOL_MISMATCH", message="Use the approved CT6 protocol."
        )
    _private_directory(config.output_root)
    with offline_network():
        bundle, clinical = prepare_ct6_bundle(config)
    if bundle.training_patient_count != 486 or bundle.validation_patient_count != 105:
        raise ArtifactError(code="CT6_COHORT_CHANGED", message="Original retained cohort changed.")
    if smoke:
        bundle = replace(
            bundle, batches_by_split={k: v[:1] for k, v in bundle.batches_by_split.items()}
        )
        config = replace(config, training=replace(config.training, development_max_minutes=5))
    epochs = 2 if smoke else 100
    if resume is None:
        root = config.output_root / new_artifact_id("ct6-smoke" if smoke else "ct6-study")
        _private_directory(root)
    else:
        root = resume.resolve()
        if root.parent != config.output_root.resolve() or not root.is_dir():
            raise ArtifactError(code="CT6_RESUME_PATH", message="Resume inside the study output.")
    assert bundle.treatment_snapshot is not None
    treatment = dict(bundle.treatment_snapshot)
    contract = {
        "protocol": CT6_PROTOCOL,
        "smoke": smoke,
        "clinical_snapshot_id": clinical["artifact_id"],
        "treatment_snapshot_id": treatment["artifact_id"],
        "epochs_limit": epochs,
        "model_settings": asdict(config.model),
        "training_settings": asdict(config.training),
        "survival_settings": asdict(config.survival),
        "evaluation_settings": asdict(config.evaluation),
    }
    # JSON normalizes tuple-valued settings, so compare through the structured JSON representation.
    import json

    contract = json.loads(json.dumps(contract))
    snapshot_once(root / "study_contract.json", contract, "ct6-contract")
    atomic_write_private_json(root / "clinical_snapshot.json", clinical)
    atomic_write_private_json(root / "treatment_snapshot.json", treatment)
    current = config.output_root / ("smoke_current.json" if smoke else "current.json")
    atomic_write_private_json(current, {"study_id": root.name, "status": "running"})
    summary: dict[str, Any] = {
        "study_id": root.name,
        "protocol": CT6_PROTOCOL,
        "status": "running",
        "smoke": smoke,
        "test_used": False,
        "survival_branch": "S1_pred",
        "clinical_schema": CT6_CLINICAL_SCHEMA,
        "treatment_schema": COMPACT_TREATMENT_SCHEMA,
        "cycle_counts_used": False,
        "cohort": {k: sum(b.batch_size for b in v) for k, v in bundle.batches_by_split.items()},
        "pretraining": {},
        "arms": {},
        "training_condition": "retrospective_interval_summary_not_baseline_known_fact",
    }
    started = time.monotonic()
    all_predictions: dict[str, torch.Tensor] = {}
    seed_predictions: dict[int, dict[str, torch.Tensor]] = {}
    labelled_bundle = None
    jobs = [("regularized", 17), ("reference", 17)]
    if not smoke:
        jobs.extend(("regularized", s) for s in (43, 97))
    try:
        with offline_network():
            for profile, seed in jobs:
                settings = experiment_config(config, profile, seed)
                key = f"{profile}/seed-{seed}"
                phase_root = root / key
                summary["active_phase"] = f"{key}/world_pretrain"
                atomic_write_private_json(root / "summary.json", summary)
                if smoke and time.monotonic() - started > 300:
                    raise ArtifactError(code="CT6_SMOKE_BUDGET", message="Smoke budget reached.")
                world, pretrain = train_ct6_phase(
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
                pretrain["future_ct"] = {
                    **future_diagnostics(world, train, validation),
                    **persistence_diagnostics(validation),
                }
                summary["pretraining"][key] = pretrain
                if smoke and profile == "regularized":
                    probe = root / "recovery_probe.json"
                    if probe.exists():
                        summary["recovery_probe"] = read_json(probe)
                    else:
                        summary["recovery_probe"] = verify_smoke_recovery(
                            settings,
                            bundle,
                            clinical,
                            root / "recovery_probe",
                            phase_root / "world_pretrain",
                        )
                        atomic_write_private_json(probe, summary["recovery_probe"])
                parent = torch.load(
                    phase_root / "world_pretrain/selected.pt", weights_only=True, map_location="cpu"
                )
                del world
                if labelled_bundle is None:
                    labelled_bundle = replace(
                        bundle, batches_by_split=_labelled_splits(config, bundle)
                    )
                    for batches in labelled_bundle.batches_by_split.values():
                        for batch in batches:
                            absolute = (
                                batch.survival_durations[:, :2]
                                + torch.stack((batch.s0_time, batch.s1_time), 1) / 365.25
                            )
                            torch.testing.assert_close(
                                absolute[:, 0], absolute[:, 1], rtol=1e-6, atol=1e-6
                            )
                    summary["landmark_labels_verified"] = True
                    summary["events"] = {
                        k: int(_labels(v)[1][:, 0].sum())
                        for k, v in labelled_bundle.batches_by_split.items()
                    }
                train = cast(
                    tuple[GeneratedTrainingBatch, ...], labelled_bundle.batches_by_split["train"]
                )
                validation = cast(
                    tuple[GeneratedTrainingBatch, ...],
                    labelled_bundle.batches_by_split["validation"],
                )
                modes = READOUT_MODES if profile == "regularized" else READOUT_MODES[:2]
                for mode in modes:
                    arm_key = f"{key}/{mode}"
                    arm_root = root / arm_key
                    summary["active_phase"] = arm_key
                    atomic_write_private_json(root / "summary.json", summary)
                    if smoke and time.monotonic() - started > 300:
                        raise ArtifactError(
                            code="CT6_SMOKE_BUDGET", message="Smoke budget reached."
                        )
                    model, report = train_ct6_phase(
                        settings,
                        labelled_bundle,
                        clinical,
                        arm_root,
                        phase=TrainingPhase.JOINT_SURVIVAL,
                        mode=mode,
                        epochs=epochs,
                        parent=parent,
                        resume=resume is not None,
                    )
                    rates = predict_rates(model, validation)
                    all_predictions[arm_key] = rates
                    if profile == "regularized":
                        seed_predictions.setdefault(seed, {})[mode] = rates
                    _atomic_torch_save(
                        arm_root / "predictions.pt",
                        {
                            "branch": "S1_pred",
                            "readout_mode": mode,
                            "rates": rates,
                            "patient_ids": tuple(p for b in validation for p in b.patient_ids),
                        },
                    )
                    report.update(
                        evaluate_generated_rates(
                            rates,
                            train,
                            validation,
                            config.survival.finite_cutpoints,
                            method=arm_key,
                            prediction_id=new_artifact_id("ct6-prediction"),
                            replicates=0 if smoke else config.evaluation.bootstrap_replicates,
                        )
                    )
                    report["calibration"] = calibration_report(
                        rates, train, validation, config.survival.finite_cutpoints
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
                    row = next(
                        r for r in treatment["rows"] if r["patient_id"] == sample.patient_ids[0]
                    )
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
                    atomic_write_private_json(root / "summary.json", summary)
                    del model
            summary["active_phase"] = "ridge_and_paired_evaluation"
            atomic_write_private_json(root / "summary.json", summary)
            ridge, ridge_snapshots = fit_ridge_baseline(
                train,
                validation,
                config.survival.finite_cutpoints,
                input_builder=matched_baseline_inputs,
            )
            _atomic_torch_save(
                root / "ridge.pt",
                {
                    "rates": ridge,
                    "snapshots": ridge_snapshots,
                    "input_contract": "CT0_CT6_regimen_drugs_target_no_cycles_no_CT1",
                },
            )
            summary["ridge"] = evaluate_generated_rates(
                ridge,
                train,
                validation,
                config.survival.finite_cutpoints,
                method="ridge",
                prediction_id=new_artifact_id("ct6-ridge"),
                replicates=0 if smoke else config.evaluation.bootstrap_replicates,
            )
            if not smoke:
                summary["paired_seed17"] = paired_comparisons(
                    {
                        **seed_predictions[17],
                        "ridge": ridge,
                    },
                    train,
                    validation,
                    config.survival.finite_cutpoints,
                    replicates=config.evaluation.bootstrap_replicates,
                )
                summary["paired_reference_seed17"] = []
                for mode in ("history_generated", "history_only"):
                    rows = paired_comparisons(
                        {
                            "history_generated": seed_predictions[17][mode],
                            "reference": all_predictions[f"reference/seed-17/{mode}"],
                        },
                        train,
                        validation,
                        config.survival.finite_cutpoints,
                        replicates=config.evaluation.bootstrap_replicates,
                    )
                    for row in rows:
                        row["comparison"] = f"regularized_minus_reference:{mode}"
                        row["training_seed"] = 17
                    summary["paired_reference_seed17"].extend(rows)
                summary["paired_mean_across_seeds"] = seed_averaged_comparisons(
                    seed_predictions,
                    train,
                    validation,
                    config.survival.finite_cutpoints,
                    replicates=config.evaluation.bootstrap_replicates,
                )
            summary["seed_metrics"] = aggregate_seed_metrics(summary["arms"])
            summary["optimizer_updates_total"] = sum(
                r["optimizer_steps"]
                for r in (*summary["pretraining"].values(), *summary["arms"].values())
            )
            summary["executed_optimizer_updates_total"] = summary["optimizer_updates_total"] + (
                summary.get("recovery_probe", {}).get("executed_optimizer_updates", 0)
            )
            summary["elapsed_seconds"] = time.monotonic() - started
            if smoke and (
                summary["executed_optimizer_updates_total"] > 20 or summary["elapsed_seconds"] > 300
            ):
                raise ArtifactError(
                    code="CT6_SMOKE_BUDGET", message="Smoke exceeded the declared budget."
                )
            if len(summary["pretraining"]) != (2 if smoke else 4) or len(summary["arms"]) != (
                5 if smoke else 11
            ):
                raise ArtifactError(
                    code="CT6_STUDY_INCOMPLETE", message="Study arms are incomplete."
                )
            summary.update(status="completed", active_phase=None)
            atomic_write_private_json(root / "summary.json", summary)
            atomic_write_private_json(
                root / "final_verification.json",
                {
                    "status": "passed",
                    "pretraining_count": len(summary["pretraining"]),
                    "joint_count": len(summary["arms"]),
                    "optimizer_updates_total": summary["optimizer_updates_total"],
                    "all_phase_audits_passed": True,
                    "independent_inference_all_arms_passed": True,
                    "landmark_labels_verified": True,
                    "test_used": False,
                },
            )
            atomic_write_private_json(current, {"study_id": root.name, "status": "completed"})
            return {
                "status": "completed",
                "study_id": root.name,
                "smoke": smoke,
                "optimizer_updates_total": summary["optimizer_updates_total"],
                "test_used": False,
            }
    except BaseException as error:
        interrupted = getattr(error, "code", "") in {"CT6_TIME_BUDGET", "CT6_SMOKE_BUDGET"}
        summary.update(
            status="interrupted" if interrupted else "failed",
            error_code=getattr(error, "code", type(error).__name__),
        )
        atomic_write_private_json(root / "summary.json", summary)
        atomic_write_private_json(current, {"study_id": root.name, "status": summary["status"]})
        raise
