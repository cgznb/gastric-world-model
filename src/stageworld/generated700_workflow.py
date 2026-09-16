"""Serial nested fivefold experiments with fresh refits and portable replay."""

from __future__ import annotations

import csv
import gc
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from stageworld.artifacts import atomic_write_private_json, read_json
from stageworld.binary700_statistics import (
    apply_operating,
    fit_operating,
    fit_statistical,
    logistic_anchor,
    statistical_predict,
)
from stageworld.generated700_data import Pool, fit_inputs
from stageworld.generated700_evaluation import evaluate
from stageworld.generated700_inference import export_bundle, predict_bundle
from stageworld.generated700_models import AnchoredClassifier, feature_set_loss, world_loss
from stageworld.generated700_spec import NEURAL, SEEDS, STATISTICAL, Candidate, specification
from stageworld.generated700_training import infer, train_phase
from stageworld.synthetic_workflow import _atomic_torch_save


def subset(pool: Pool, ids: list[str], count: int) -> list[str]:
    """Round-robin joint-label strata for a smoke with meaningful label support."""
    strata: dict[tuple, list[str]] = defaultdict(list)
    for p, row in zip(ids, pool.indices(ids).tolist(), strict=True):
        key = tuple(int(pool.labels[row, i]) if pool.valid[row, i] else -1 for i in range(2))
        strata[key].append(p)
    result: list[str] = []
    while len(result) < count:
        for group in strata.values():
            if group and len(result) < count:
                result.append(group.pop(0))
        if not any(strata.values()) and len(result) < count:
            raise ValueError("Insufficient smoke patients")
    return result


def publish_prediction(
    root: Path, pool: Pool, rows: torch.Tensor, logits: np.ndarray, rules: dict, contract: dict
) -> None:
    ids = [pool.ids[i] for i in rows.tolist()]
    actual = apply_operating(logits, rules)
    # Standalone export has no path to the cohort, CT1 targets or endpoint labels.
    replay = predict_bundle(
        root / "bundle",
        [pool.clinical[p] for p in ids],
        [pool.treatments[p] for p in ids],
        pool.interval[rows],
        pool.ct0[rows],
        pool.ct0_valid[rows],
    )
    maximum = float((replay["probabilities"] - actual["probabilities"]).abs().max())
    if not torch.allclose(replay["probabilities"], actual["probabilities"], atol=1e-5, rtol=1e-5):
        raise ValueError("Standalone CPU/GPU prediction replay differs")
    for key in ("raw_0.5", "inner_balanced", "inner_sensitivity_0.8", "calibrated_0.5"):
        if not torch.equal(replay[key], actual[key]):
            raise ValueError("Standalone operating decisions differ")
    _atomic_torch_save(
        root / "predictions.pt",
        {
            "patient_ids": ids,
            "logits": torch.from_numpy(logits),
            **actual,
            "labels": pool.labels[rows],
            "valid": pool.valid[rows],
            "contract": contract,
        },
    )
    atomic_write_private_json(
        root / "completed.json",
        {
            "status": "completed",
            "outer_patients": len(rows),
            "independent_inference_max_probability_error": maximum,
            "ct1_or_outcome_input_required": False,
        },
    )


def _statistical_group(
    pool: Pool,
    candidate: Candidate,
    root: Path,
    x_inner: torch.Tensor,
    x_refit: torch.Tensor,
    inner_snapshot: dict,
    refit_snapshot: dict,
    train: torch.Tensor,
    validation: torch.Tensor,
    refit: torch.Tensor,
    outer: torch.Tensor,
) -> tuple[list, list]:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (root / "completed.json").exists():
        return joblib.load(root / "inner.joblib"), joblib.load(root / "refit.joblib")
    models, selected = fit_statistical(
        candidate, x_inner, pool.labels, pool.valid, train, validation
    )
    inner_logits = statistical_predict(models, x_inner[validation].double().numpy())
    rules = fit_operating(inner_logits, pool.labels[validation], pool.valid[validation])
    fitted, details = fit_statistical(
        candidate, x_refit, pool.labels, pool.valid, refit, None, selected["choices"]
    )
    joblib.dump(models, root / "inner.joblib")
    joblib.dump(fitted, root / "refit.joblib")
    atomic_write_private_json(
        root / "selection.json",
        {
            "inner": selected,
            "refit": details,
            "rules": rules,
            "inner_snapshot_id": inner_snapshot["artifact_id"],
            "refit_snapshot_id": refit_snapshot["artifact_id"],
        },
    )
    export_bundle(root / "bundle", refit_snapshot, rules, candidate.family, models=fitted)
    publish_prediction(
        root,
        pool,
        outer,
        statistical_predict(fitted, x_refit[outer].double().numpy()),
        rules,
        {
            "inner_snapshot_id": inner_snapshot["artifact_id"],
            "refit_snapshot_id": refit_snapshot["artifact_id"],
        },
    )
    return models, fitted


def _ct_report(
    model: torch.nn.Module, x: torch.Tensor, pool: Pool, fitting: torch.Tensor, outer: torch.Tensor
) -> dict:
    rows = outer[(pool.ct0_valid & pool.ct1_valid)[outer]]
    fitted = fitting[(pool.ct0_valid & pool.ct1_valid)[fitting]]
    tokens = infer(model, x, pool, rows, world=True)
    prediction = tokens.mean(1)
    target = pool.ct1[rows]
    values = {
        "generated": prediction,
        "training_mean": pool.ct1[fitted].mean(0).expand_as(target),
        "copy_ct0": pool.ct0[rows].mean(1),
    }
    result = {
        name: {
            "n": len(rows),
            "mse": float((value - target).square().mean()),
            "smooth_l1_cosine": float(
                world_loss(value, target, torch.ones(len(rows), dtype=torch.bool))
            ),
        }
        for name, value in values.items()
    }
    result["generated"]["feature_set_loss"] = float(
        feature_set_loss(tokens, pool.ct1_tokens[rows], torch.ones(len(rows), dtype=torch.bool))
    )
    return result


@torch.inference_mode()
def select_residual_scales(
    model: AnchoredClassifier, x: torch.Tensor, pool: Pool, rows: torch.Tensor
) -> list[float]:
    model.residual_scale.fill_(1)
    full = infer(model, x, pool, rows).numpy()
    baseline = torch.nn.functional.linear(
        x[rows].float(), model.anchor_weight.cpu(), model.anchor_bias.cpu()
    ).numpy()
    selected = []
    for endpoint in range(2):
        mask = pool.valid[rows, endpoint].numpy()
        y = pool.labels[rows, endpoint].numpy()[mask]
        choices = []
        for scale in specification()["inner_residual_scales"]:
            logits = baseline[:, endpoint] + scale * (full[:, endpoint] - baseline[:, endpoint])
            choices.append((-average_precision_score(y, logits[mask]), scale))
        selected.append(float(min(choices)[1]))
    model.residual_scale.copy_(torch.tensor(selected, device=model.residual_scale.device))
    return selected


def run_study(
    pool: Pool, folds: dict, root: Path, *, smoke: bool = False, replicates: int = 1000
) -> dict:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    spec = {**specification(), "smoke": smoke, "pool_id": pool.artifact_id}
    if (root / "specification.json").exists() and read_json(root / "specification.json") != spec:
        raise ValueError("Cannot change a running experiment specification")
    atomic_write_private_json(root / "specification.json", spec)
    locked_folds = folds["folds"][:1] if smoke else folds["folds"]
    epochs = 2 if smoke else 100
    selections: list[dict[str, Any]] = []
    ct_reports = []
    for fold_index, fold in enumerate(locked_folds):
        groups = {k: subset(pool, v, 32) if smoke else v for k, v in fold["patient_ids"].items()}
        train, validation, outer = (
            pool.indices(groups[k]) for k in ("train", "validation", "outer")
        )
        refit_ids = groups["train"] + groups["validation"]
        refit = pool.indices(refit_ids)
        if set(refit_ids) & set(groups["outer"]) or len(set(refit_ids)) != len(refit_ids):
            raise ValueError("Outer patient leakage")
        fold_root = root / "partitions" / f"fold-{fold_index}"
        fold_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        atomic_write_private_json(fold_root / "groups.json", groups)
        x_inner, s_inner = fit_inputs(pool, groups["train"], fold_root / "inner.pt")
        x_refit, s_refit = fit_inputs(pool, refit_ids, fold_root / "refit.pt")
        anchors = None
        for candidate in STATISTICAL:
            print(f"fold={fold_index} arm={candidate.name} statistical", flush=True)
            target = root / "fits" / candidate.name / "deterministic" / f"fold-{fold_index}"
            models, fitted = _statistical_group(
                pool,
                candidate,
                target,
                x_inner,
                x_refit,
                s_inner,
                s_refit,
                train,
                validation,
                refit,
                outer,
            )
            if candidate.name == "logistic":
                anchors = logistic_anchor(models), logistic_anchor(fitted)
        assert anchors is not None
        for seed in (17,) if smoke else SEEDS:
            world_root = root / "world" / str(seed) / f"fold-{fold_index}"
            print(f"fold={fold_index} seed={seed} world inner/refit", flush=True)
            inner_world, inner_world_report = train_phase(
                pool,
                x_inner,
                s_inner,
                Candidate("world", "world"),
                train,
                validation,
                world_root / "inner",
                seed=seed,
                epochs=epochs,
            )
            del inner_world
            refit_world, refit_world_report = train_phase(
                pool,
                x_refit,
                s_refit,
                Candidate("world", "world"),
                refit,
                None,
                world_root / "refit",
                seed=seed,
                epochs=inner_world_report["selected_epoch"],
            )
            ct_reports.append(
                {
                    "fold": fold_index,
                    "seed": seed,
                    "metrics": _ct_report(refit_world, x_refit, pool, refit, outer),
                }
            )
            del refit_world
            for partition, report in (("inner", inner_world_report), ("refit", refit_world_report)):
                selections.append(
                    {
                        "fold": fold_index,
                        "seed": seed,
                        "arm": "world",
                        "partition": partition,
                        **report,
                    }
                )
            for candidate in NEURAL:
                print(f"fold={fold_index} seed={seed} arm={candidate.name} inner/refit", flush=True)
                target = root / "fits" / candidate.name / str(seed) / f"fold-{fold_index}"
                if not (target / "completed.json").exists():
                    parent = (
                        world_root / "inner/selected.pt"
                        if candidate.family.startswith("generated")
                        else None
                    )
                    model, inner_report = train_phase(
                        pool,
                        x_inner,
                        s_inner,
                        candidate,
                        train,
                        validation,
                        target / "inner",
                        seed=seed,
                        epochs=epochs,
                        anchor=anchors[0],
                        parent=parent,
                    )
                    assert isinstance(model, AnchoredClassifier)
                    scales = select_residual_scales(model, x_inner, pool, validation)
                    logits = infer(model, x_inner, pool, validation).numpy()
                    rules = fit_operating(logits, pool.labels[validation], pool.valid[validation])
                    del model
                    parent = (
                        world_root / "refit/selected.pt"
                        if candidate.family.startswith("generated")
                        else None
                    )
                    model, refit_report = train_phase(
                        pool,
                        x_refit,
                        s_refit,
                        candidate,
                        refit,
                        None,
                        target / "refit",
                        seed=seed,
                        epochs=inner_report["selected_epoch"],
                        anchor=anchors[1],
                        parent=parent,
                    )
                    assert isinstance(model, AnchoredClassifier)
                    model.residual_scale.copy_(
                        torch.tensor(scales, device=model.residual_scale.device)
                    )
                    atomic_write_private_json(
                        target / "selection.json",
                        {
                            "inner": inner_report,
                            "refit": refit_report,
                            "rules": rules,
                            "inner_residual_scales": scales,
                        },
                    )
                    export_bundle(
                        target / "bundle",
                        s_refit,
                        rules,
                        candidate.family,
                        model=model,
                        image_dim=pool.ct0.shape[-1],
                    )
                    atomic_write_private_json(
                        target / "ct_evaluation.json",
                        _ct_report(model.world, x_refit, pool, refit, outer),
                    )
                    publish_prediction(
                        target,
                        pool,
                        outer,
                        infer(model, x_refit, pool, outer).numpy(),
                        rules,
                        {
                            "inner_snapshot_id": s_inner["artifact_id"],
                            "refit_snapshot_id": s_refit["artifact_id"],
                        },
                    )
                    del model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                for partition in ("inner", "refit"):
                    report = read_json(target / partition / "completed.json")
                    selections.append(
                        {
                            "fold": fold_index,
                            "seed": seed,
                            "arm": candidate.name,
                            "partition": partition,
                            **report,
                        }
                    )
                atomic_write_private_json(
                    root / "progress.json",
                    {
                        "fold": fold_index,
                        "seed": seed,
                        "last_arm": candidate.name,
                        "status": "running",
                        "completed_neural_phases": len(selections),
                    },
                )
    atomic_write_private_json(
        root / "selections.json", {"phases": selections, "ct_predictions": ct_reports}
    )
    fields = (
        "fold",
        "seed",
        "arm",
        "partition",
        "selected_epoch",
        "selected_score",
        "completed_epochs",
        "updates",
        "seconds",
        "registered_parameters",
        "trainable_parameters",
    )
    with (root / "selections.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selections)
    result = evaluate(root, pool, expected=32 if smoke else 700, replicates=replicates)
    completion = {
        "status": "completed",
        "smoke": smoke,
        "neural_phases": len(selections),
        "epochs": sum(r["completed_epochs"] for r in selections),
        "updates": sum(r["updates"] for r in selections),
        "evaluated_patients": result["patients"],
    }
    atomic_write_private_json(root / "completed.json", completion)
    return completion
