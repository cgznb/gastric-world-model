"""Ordered baseline, residual, regularization and fixed-fold OOF experiments."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import torch

from stageworld.artifacts import atomic_write_private_json, new_artifact_id, read_json
from stageworld.binary_baselines import (
    baseline_predict,
    fit_baseline,
    operating_metrics,
    select_thresholds,
    targets,
)
from stageworld.binary_data import (
    BinaryDevelopmentData,
    make_binary_folds,
    make_fold_data,
    prepare_binary_development,
)
from stageworld.binary_endpoints import ENDPOINTS, LABEL_CONTRACT, BinaryEndpointBatch
from stageworld.binary_evaluation import evaluate_binary, prevalence_baseline
from stageworld.binary_improvement_diagnostics import (
    ct_controls,
    feature_diagnostics,
    geometry_audit,
    legacy_dependence,
    verify_model_contract,
    verify_recovery,
)
from stageworld.binary_improvement_evaluation import aggregate_study
from stageworld.binary_improvement_inference import export_bundle, verify_portable
from stageworld.binary_improvement_spec import ARMS, Arm, study_spec
from stageworld.binary_improvement_training import (
    Predictor,
    predict,
    summarize_predictions,
    train_phase,
)
from stageworld.binary_workflow import validate_binary_protocol
from stageworld.config import StageWorldConfig
from stageworld.ct6_workflow import snapshot_once
from stageworld.data.gastric_roi import offline_network
from stageworld.data.treatment_compact import treatment_fields
from stageworld.generated_inference import ct0_packet
from stageworld.synthetic_workflow import _atomic_torch_save, _slice_observation


def portable_check(
    model: Predictor | dict[str, Any],
    root: Path,
    data: BinaryDevelopmentData,
    snapshot: dict[str, Any],
    outer: tuple[BinaryEndpointBatch, ...],
    expected: torch.Tensor,
    thresholds: list[float],
    weights: list[float],
    version: str,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    sample = outer[0]
    patient = sample.patient_ids[0]
    export_bundle(
        model,
        root,
        snapshot=snapshot,
        provenance=sample.ct0.provenance,
        weight_version=version,
        weights=weights,
        thresholds=thresholds,
    )
    _atomic_torch_save(
        root / "ct0.pt", ct0_packet(_slice_observation(sample.ct0, torch.tensor([0])))
    )
    atomic_write_private_json(
        root / "query.json",
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
    if isinstance(model, dict):
        return verify_portable(root, expected)
    device = next(model.parameters()).device
    try:
        model.cpu()
        cpu = predict(model, [sample], correction=torch.tensor(weights))["probabilities"]
        # TF32 convolution and CPU FP32 have different accumulation precision.
        # Check the serialized input contract strictly on the deployment backend.
        torch.testing.assert_close(cpu, expected[: sample.batch_size], atol=1e-4, rtol=1e-4)
        report = verify_portable(root, cpu)
        report["training_backend_vs_cpu_max_difference"] = float(
            (cpu - expected[: sample.batch_size]).abs().max()
        )
        report["training_backend"] = str(device)
        report["cross_backend_absolute_tolerance"] = 1e-4
        report["deployment_reference"] = "CPU_FP32_full_model_same_inputs"
        return report
    finally:
        model.to(device)


def save_outer(
    root: Path,
    prediction: dict[str, torch.Tensor],
    train: tuple[BinaryEndpointBatch, ...],
    outer: tuple[BinaryEndpointBatch, ...],
    thresholds: list[float],
) -> dict[str, Any]:
    y, valid = targets(outer)
    ids = [p for b in outer for p in b.patient_ids]
    p = prediction["probabilities"]
    thresholds_tensor = torch.tensor(thresholds, dtype=p.dtype).repeat(len(ids), 1)
    payload = {
        "patient_ids": ids,
        "probabilities": p,
        "labels": y,
        "valid": valid,
        "prevalence": prevalence_baseline(train, len(ids)),
        "thresholds": thresholds_tensor,
        "label_contract": LABEL_CONTRACT,
    }
    if "ct_mean" in prediction:
        target = torch.cat([b.future_ct_target for b in outer])
        payload.update(
            ct_mean=prediction["ct_mean"],
            ct_target=target,
            ct_persistence=torch.cat([b.ct0.values.mean(1, keepdim=True) for b in outer]),
            ct_training_mean=torch.cat([b.future_ct_target for b in train])
            .mean(0, keepdim=True)
            .expand_as(target),
        )
    _atomic_torch_save(root / "outer_predictions.pt", payload)
    return {
        "outer": evaluate_binary(p, y, valid, replicates=0),
        "inner_selected_thresholds": thresholds,
        "operating_metrics": operating_metrics(p, y, valid, thresholds_tensor),
        "future_ct": ct_controls(train, outer, prediction.get("ct_mean")),
    }


def assert_reference(
    config: StageWorldConfig, data: BinaryDevelopmentData, reference: Path
) -> dict[str, Any]:
    generated = make_binary_folds(data.labels)
    original = read_json(reference.parent / "folds.json")
    if generated != {k: v for k, v in original.items() if k != "artifact_id"}:
        raise ValueError("The original fixed folds or endpoint composition changed")
    original_inputs = read_json(reference.parent / "raw_inputs_and_labels.json")
    for key, current in (("clinical_rows", data.clinical), ("labels", data.labels)):
        if original_inputs[key] != current:
            raise ValueError(f"The approved current input source differs: {key}")
    original_treatment = {row["patient_id"]: row for row in original_inputs["treatment_rows"]}
    if original_treatment != data.treatments:
        raise ValueError("The approved no-cycle treatment input changed")
    for key in ("feature_artifact_id", "cohort_artifact_id"):
        if original_inputs[key] != getattr(data.bundle, key):
            raise ValueError("Original frozen features or development cohort differ")
    return snapshot_once(config.output_root / "folds.json", generated, "binary-improvement-folds")


def run_improvement_study(
    config: StageWorldConfig, *, reference: Path, smoke: bool = False, resume: Path | None = None
) -> dict[str, Any]:
    validate_binary_protocol(config)
    config.output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    reference = reference.resolve()
    if not (reference / "summary.json").is_file():
        raise ValueError("Bind the completed original binary study")
    if read_json(reference / "summary.json")["status"] != "completed":
        raise ValueError("Original study is not complete")
    root = (
        config.output_root / new_artifact_id("improvement-smoke" if smoke else "improvement-study")
        if resume is None
        else resume.resolve()
    )
    if root.parent.resolve() != config.output_root.resolve():
        raise ValueError("Resume only this isolated experiment")
    if resume is not None and not root.is_dir():
        raise ValueError("Resume path does not exist")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with offline_network():
        data = prepare_binary_development(config)
        folds = assert_reference(config, data, reference)
    snapshot_once(
        root / "study_contract.json",
        json.loads(
            json.dumps(
                {
                    **study_spec(),
                    "smoke": smoke,
                    "reference_study": reference.name,
                    "source_id": data.source_id,
                    "fold_artifact_id": folds["artifact_id"],
                    "reference_model_config": asdict(config.model),
                    "reference_training_config": asdict(config.training),
                    "label_contract": LABEL_CONTRACT,
                }
            )
        ),
        "binary-improvement-contract",
    )
    summary = (
        read_json(root / "summary.json")
        if (root / "summary.json").exists()
        else {
            "study_id": root.name,
            "smoke": smoke,
            "status": "running",
            "fits": {},
            "world_fits": {},
            "baselines": {},
            "diagnostics": {},
            "pool": folds["pool"],
            "folds": [{"fold": row["fold"], "counts": row["counts"]} for row in folds["folds"]],
            "test_used": False,
            "test_role": "previously_opened_106_holdout_excluded",
            "outer_selection_used": False,
            "survival_enabled": False,
        }
    )
    summary["status"] = "running"
    summary.pop("error_code", None)
    current = config.output_root / ("smoke_current.json" if smoke else "current.json")
    started = time.monotonic()
    elapsed_before = summary.get("elapsed_seconds", 0)

    def save(active: str | None = None) -> None:
        summary["active_fit"] = active
        summary["elapsed_seconds"] = elapsed_before + time.monotonic() - started
        atomic_write_private_json(root / "summary.json", summary)
        atomic_write_private_json(
            current, {"study_id": root.name, "status": summary["status"], "active_fit": active}
        )

    selected_folds = folds["folds"][:1] if smoke else folds["folds"]
    seeds = [17] if smoke else [17, 43, 97]
    fold_data = {}
    try:
        save("input_geometry_audit")
        with offline_network():
            summary["geometry_audit"] = geometry_audit(config, data)
            atomic_write_private_json(root / "geometry_audit.json", summary["geometry_audit"])
            for fold in selected_folds:
                number = fold["fold"]
                bundle, outer, snapshot = make_fold_data(
                    config, data, fold, root / f"fold-{number}", smoke=smoke
                )
                train = cast(tuple[BinaryEndpointBatch, ...], bundle.batches_by_split["train"])
                inner = cast(tuple[BinaryEndpointBatch, ...], bundle.batches_by_split["validation"])
                if not all(b.future_ct_valid.all() for b in (*train, *inner, *outer)):
                    raise ValueError("The paired cohort requires complete frozen CT targets")
                fold_data[number] = train, inner, outer, snapshot
                features = feature_diagnostics(train)
                atomic_write_private_json(
                    root / f"fold-{number}/feature_diagnostics.json", features
                )
                summary["diagnostics"][f"fold-{number}-features"] = features
            # Linear fits and true-CT1 diagnostics precede every neural candidate.
            for mode in ("clinical", "ct0", "ct1_diagnostic"):
                for number, (train, inner, outer, snapshot) in fold_data.items():
                    key = f"{mode}/fold-{number}"
                    folder = root / "baselines" / key
                    if (folder / "completed.json").exists():
                        summary["baselines"][key] = read_json(folder / "completed.json")
                        continue
                    save(f"baselines/{key}")
                    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
                    payload, trace = fit_baseline(train, inner, mode)
                    version = new_artifact_id("binary-logistic")
                    _atomic_torch_save(folder / "selected.pt", {"artifact_id": version, **payload})
                    y, valid = targets(inner)
                    probability = baseline_predict(payload, inner)
                    thresholds = select_thresholds(probability, y, valid)
                    report: dict[str, Any] = {
                        "selection": trace,
                        "inner": evaluate_binary(probability, y, valid, replicates=0),
                        "outer_evaluation": mode != "ct1_diagnostic",
                        "status": "completed",
                    }
                    if mode != "ct1_diagnostic":
                        prediction = baseline_predict(payload, outer)
                        report.update(
                            save_outer(
                                folder, {"probabilities": prediction}, train, outer, thresholds
                            )
                        )
                        report["portable_inference"] = portable_check(
                            payload,
                            folder / "portable",
                            data,
                            snapshot,
                            outer,
                            prediction,
                            thresholds,
                            [1.0, 1.0],
                            version,
                        )
                    atomic_write_private_json(folder / "completed.json", report)
                    summary["baselines"][key] = report
                    save()
            # Candidates are locked before any OOF score is read; no adaptive arm choice.
            for arm in ARMS:
                for seed in seeds:
                    for number, (train, inner, outer, snapshot) in fold_data.items():
                        settings = replace(config, training=replace(config.training, seed=seed))
                        key = f"{arm.name}/seed-{seed}/fold-{number}"
                        folder = root / "models" / key
                        if (folder / "completed.json").exists():
                            summary["fits"][key] = read_json(folder / "completed.json")
                            continue
                        parent = None
                        if arm.architecture != "direct":
                            world_key = f"{arm.architecture}/seed-{seed}/fold-{number}"
                            world_folder = root / "world" / world_key
                            if world_key not in summary["world_fits"]:
                                save(f"world/{world_key}")
                                world_arm = Arm(f"world_{arm.architecture}", arm.architecture)
                                unlabelled = [
                                    tuple(
                                        replace(
                                            b,
                                            endpoint_labels=torch.zeros_like(b.endpoint_labels),
                                            endpoint_valid=torch.zeros_like(b.endpoint_valid),
                                        )
                                        for b in partition
                                    )
                                    for partition in (train, inner)
                                ]
                                world_model, world_report = train_phase(
                                    settings,
                                    world_arm,
                                    unlabelled[0],
                                    unlabelled[1],
                                    world_folder,
                                    seed=seed,
                                    snapshot_id=snapshot["artifact_id"],
                                    world=True,
                                    epochs=2 if smoke else 100,
                                    resume=(world_folder / "contract.json").exists(),
                                )
                                del world_model
                                summary["world_fits"][world_key] = world_report
                                if smoke and arm.architecture == "residual":
                                    summary["world_recovery"] = verify_recovery(
                                        settings,
                                        unlabelled[0],
                                        unlabelled[1],
                                        root / "world_recovery",
                                        world_folder,
                                        snapshot["artifact_id"],
                                        world=True,
                                        arm=world_arm,
                                    )
                                save()
                            parent = world_folder / "selected.pt"
                        save(f"models/{key}")
                        model, training_report = train_phase(
                            settings,
                            arm,
                            train,
                            inner,
                            folder,
                            seed=seed,
                            snapshot_id=snapshot["artifact_id"],
                            world=False,
                            parent=parent,
                            epochs=2 if smoke else 100,
                            resume=(folder / "contract.json").exists(),
                        )
                        if smoke and arm.name == "residual_joint":
                            summary["joint_recovery"] = verify_recovery(
                                settings,
                                train,
                                inner,
                                root / "joint_recovery",
                                folder,
                                snapshot["artifact_id"],
                                world=False,
                                arm=arm,
                                parent=parent,
                            )
                        weights = torch.tensor(training_report["positive_weights"])
                        inner_prediction = predict(model, inner, correction=weights)
                        y, valid = targets(inner)
                        report = {
                            "training": training_report,
                            "outer_evaluation": arm.outer_evaluation,
                            "status": "completed",
                            "contract_checks": verify_model_contract(model, inner[0]),
                        }
                        if arm.outer_evaluation:
                            thresholds = select_thresholds(
                                inner_prediction["probabilities"], y, valid
                            )
                            neural_prediction = predict(model, outer, correction=weights)
                            report.update(
                                save_outer(folder, neural_prediction, train, outer, thresholds)
                            )
                            selected = torch.load(
                                folder / "selected.pt", weights_only=True, map_location="cpu"
                            )
                            report["portable_inference"] = portable_check(
                                model,
                                folder / "portable",
                                data,
                                snapshot,
                                outer,
                                neural_prediction["probabilities"],
                                thresholds,
                                weights.tolist(),
                                selected["artifact_id"],
                            )
                        else:
                            report["inner_only"] = summarize_predictions(
                                inner_prediction, inner, endpoint=arm.endpoint
                            )
                        atomic_write_private_json(folder / "completed.json", report)
                        summary["fits"][key] = report
                        del model
                        save()
                if arm.name == "direct":
                    for seed in seeds:
                        for number, (_, inner, _, _) in fold_data.items():
                            diagnostic_key = f"original_dependence/seed-{seed}/fold-{number}"
                            if diagnostic_key not in summary["diagnostics"]:
                                save(diagnostic_key)
                                summary["diagnostics"][diagnostic_key] = legacy_dependence(
                                    config,
                                    inner,
                                    reference
                                    / f"seed-{seed}/fold-{number}/joint_endpoints/selected.pt",
                                )
                                save()
            save("pooled_OOF_and_paired_patient_bootstrap")
            expected_ids = {
                p for _, _, outer, _ in fold_data.values() for b in outer for p in b.patient_ids
            }
            summary["evaluation"] = aggregate_study(
                root,
                arms=[a.name for a in ARMS if a.outer_evaluation],
                seeds=seeds,
                folds=list(fold_data),
                patient_ids=expected_ids,
                reference_root=reference,
                smoke=smoke,
            )
        expected = len(ARMS) * len(seeds) * len(selected_folds)
        if len(summary["fits"]) != expected:
            raise ValueError("Missing prespecified candidate fits")
        summary["optimizer_updates"] = sum(
            r["training"]["optimizer_updates"] for r in summary["fits"].values()
        ) + sum(r["optimizer_updates"] for r in summary["world_fits"].values())
        summary["status"] = "completed"
        summary["multitask_diagnostic"] = {
            "policy": (
                "Compare each single-task inner BCE with residual_warm for the same "
                "fold and seed; unused heads are not evaluated."
            ),
            "rows": [
                {
                    "seed": seed,
                    "fold": fold,
                    "endpoint": endpoint,
                    "single_task_bce": summary["fits"][
                        f"single_{endpoint}_diagnostic/seed-{seed}/fold-{fold}"
                    ]["inner_only"]["endpoints"][endpoint]["bce"],
                    "shared_task_bce": summary["fits"][f"residual_warm/seed-{seed}/fold-{fold}"][
                        "training"
                    ]["selected_inner"]["endpoints"][endpoint]["bce"],
                }
                for seed in seeds
                for fold in fold_data
                for endpoint in ENDPOINTS
            ],
        }
        save()
        return summary
    except BaseException as error:
        summary.update(status="failed", error_code=getattr(error, "code", type(error).__name__))
        save(summary.get("active_fit"))
        raise
