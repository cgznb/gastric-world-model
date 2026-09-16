"""Checkpoint replay, patient membership, portable inference and per-seed audits."""

from __future__ import annotations

import builtins
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import joblib
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

from stageworld.artifacts import atomic_write_private_json, read_json
from stageworld.binary700_statistics import logistic_anchor
from stageworld.generated651_evaluation import METRICS
from stageworld.generated651_inference import predict_bundle
from stageworld.generated651_spec import NEURAL, RESIDUAL_SCALE, THRESHOLD, Candidate
from stageworld.generated651_workflow import fit_anchor
from stageworld.generated700_data import Pool, fit_inputs
from stageworld.generated700_training import build_model, score, train_phase
from stageworld.training import checkpoint_payload_mismatches


def verify_recovery(pool: Pool, root: Path) -> dict:
    groups = read_json(root / "partitions/fold-0/groups.json")
    x, snapshot = fit_inputs(pool, groups["train"], root / "partitions/fold-0/inputs.pt")
    train, validation = pool.indices(groups["train"]), pool.indices(groups["validation"])
    anchor = logistic_anchor(joblib.load(root / "partitions/fold-0/anchor.joblib"))
    results = {}
    for family in ("world", "generated"):
        base = root / "recovery_probes" / family
        parent = root / "world/17/fold-0/selected.pt" if family == "generated" else None
        options: dict[str, Any] = {"seed": 17, "epochs": 2, "anchor": anchor, "parent": parent}
        candidate = Candidate(family, family)
        model, _ = train_phase(
            pool, x, snapshot, candidate, train, validation, base / "reference", **options
        )
        del model
        if not (base / "recovered/completed.json").exists():
            try:
                train_phase(
                    pool,
                    x,
                    snapshot,
                    candidate,
                    train,
                    validation,
                    base / "recovered",
                    interrupt_after_update=len(train.split(32)) + 1,
                    **options,
                )
            except RuntimeError as error:
                if str(error) != "intentional_partial_epoch_interruption":
                    raise
            else:
                raise AssertionError("Recovery probe did not exercise a partial-epoch interruption")
        model, _ = train_phase(
            pool, x, snapshot, candidate, train, validation, base / "recovered", **options
        )
        del model
        reference = torch.load(base / "reference/final.pt", weights_only=True, map_location="cpu")
        recovered = torch.load(base / "recovered/final.pt", weights_only=True, map_location="cpu")
        for key in (
            "model_state",
            "optimizer_state",
            "rng_state",
            "epoch",
            "updates",
            "early_stop",
        ):
            if checkpoint_payload_mismatches(reference[key], recovered[key]):
                raise ValueError("Exact complete-epoch recovery failed")
        results[family] = {"model_optimizer_rng_exact": True, "updates": recovered["updates"]}
    atomic_write_private_json(root / "recovery_verification.json", results)
    return results


def _metrics(labels: np.ndarray, probability: np.ndarray) -> dict:
    decisions = probability >= THRESHOLD
    tn, fp, fn, tp = confusion_matrix(labels, decisions, labels=[0, 1]).ravel().tolist()
    sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
    return {
        "n": len(labels),
        "positive": tp + fn,
        "negative": tn + fp,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / len(labels),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": tp / (tp + fp) if tp + fp else None,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "auroc": roc_auc_score(labels, probability),
        "auprc": average_precision_score(labels, probability),
        "bce": log_loss(labels, np.clip(probability, 1e-7, 1 - 1e-7)),
        "brier": brier_score_loss(labels, np.clip(probability, 1e-7, 1 - 1e-7)),
    }


def verify_study(pool: Pool, root: Path) -> dict:
    spec = read_json(root / "specification.json")
    selection = read_json(root / "selections.json")
    folds = 1 if spec["smoke"] else 5
    expected_phases = folds * len(spec["active_seeds"]) * (1 + len(NEURAL))
    expected_bundles = folds * len(spec["active_seeds"]) * len(NEURAL)
    if len(selection["phases"]) != expected_phases:
        raise ValueError("Phase coverage differs from the locked protocol")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    partitions, phase_keys = {}, set()
    score_checks, parent_checks = 0, 0
    for entry in selection["phases"]:
        fold, seed, arm = (entry[key] for key in ("fold", "seed", "arm"))
        identity = (fold, seed, arm)
        if identity in phase_keys or seed not in spec["active_seeds"] or fold not in range(folds):
            raise ValueError("Unexpected phase identity")
        phase_keys.add(identity)
        fold_root = root / "partitions" / f"fold-{fold}"
        groups = read_json(fold_root / "groups.json")
        if set(groups["train"]) & set(groups["validation"]):
            raise ValueError("Training and validation patients overlap")
        if fold not in partitions:
            x, snapshot = fit_inputs(pool, groups["train"], fold_root / "inputs.pt")
            with TemporaryDirectory(dir=root) as temporary:
                fresh_x, fresh = fit_inputs(pool, groups["train"], Path(temporary) / "inputs.pt")
            if (
                not torch.equal(x, fresh_x)
                or snapshot["clinical"] != fresh["clinical"]
                or snapshot["support"] != fresh["support"]
            ):
                raise ValueError("Training-only input transformation does not replay")
            models = joblib.load(fold_root / "anchor.joblib")
            original = logistic_anchor(models)
            refitted = logistic_anchor(fit_anchor(pool, x, pool.indices(groups["train"])))
            if any(
                not torch.allclose(a, b, atol=1e-7, rtol=1e-6)
                for a, b in zip(original, refitted, strict=True)
            ):
                raise ValueError("Training-only fixed clinical anchor does not replay")
            partitions[fold] = (x, snapshot, original)
        x, snapshot, anchor = partitions[fold]
        base = (
            root / "world" / str(seed) if arm == "world" else root / "fits" / arm / str(seed)
        ) / f"fold-{fold}"
        final = torch.load(base / "final.pt", weights_only=True, map_location="cpu")
        latest = torch.load(base / "latest.pt", weights_only=True, map_location="cpu")
        if checkpoint_payload_mismatches(final, latest):
            raise ValueError("Latest and final checkpoints differ")
        selected = torch.load(base / "selected.pt", weights_only=True, map_location="cpu")
        contract = final["contract"]
        if (
            contract["train_ids"] != groups["train"]
            or contract["validation_ids"] != groups["validation"]
            or contract["snapshot_id"] != snapshot["artifact_id"]
            or contract["seed"] != seed
            or contract["fixed_refit"]
            or selected["contract"] != contract
        ):
            raise ValueError("Checkpoint training/validation identity differs")
        if selected["history"][-1]["score"] != min(row["score"] for row in final["history"]):
            raise ValueError("Selected checkpoint is not the raw validation optimum")
        candidate = Candidate(**contract["candidate"])
        if candidate.name != arm or (arm != "world" and candidate not in NEURAL):
            raise ValueError("Unexpected candidate checkpoint")
        parent = None
        if arm != "world":
            parent = torch.load(
                root / "world" / str(seed) / f"fold-{fold}/selected.pt",
                weights_only=True,
                map_location="cpu",
            )
            if (
                contract["parent_id"] != parent["artifact_id"]
                or parent["contract"]["snapshot_id"] != snapshot["artifact_id"]
                or parent["contract"]["seed"] != seed
            ):
                raise ValueError("Generator parent used a different partition or seed")
            parent_checks += 1
        model = build_model(
            candidate.family,
            x.shape[1],
            pool.ct0.shape[-1],
            None if arm == "world" else anchor,
            parent,
        ).to(device)
        validation = pool.indices(groups["validation"])
        for checkpoint in (selected, final):
            model.load_state_dict(checkpoint["model_state"])
            actual_score, _ = score(model, x, pool, validation, arm == "world")
            if abs(actual_score - checkpoint["history"][-1]["score"]) > 1e-6:
                raise ValueError("Selected/final validation score replay differs")
            if arm != "world":
                if not torch.equal(
                    checkpoint["model_state"]["residual_scale"], torch.full((2,), RESIDUAL_SCALE)
                ):
                    raise ValueError("Neural residual scale was retuned")
                for name, expected in zip(("anchor_weight", "anchor_bias"), anchor, strict=True):
                    if not torch.equal(checkpoint["model_state"][name], expected):
                        raise ValueError("Fixed clinical anchor changed during neural training")
            score_checks += 1
        del model
    original_open = builtins.open
    metric_groups: dict[tuple, list[dict]] = defaultdict(list)
    prediction_sets: dict[tuple, set[str]] = defaultdict(set)
    bundle_count = 0
    for path in sorted(root.glob("fits/*/*/fold-*/predictions.pt")):
        stored = torch.load(path, weights_only=True, map_location="cpu")
        arm, seed, fold = stored["arm"], stored["seed"], stored["fold"]
        if (fold, seed, arm) not in phase_keys:
            raise ValueError("Prediction does not belong to a trained phase")
        groups = read_json(root / "partitions" / f"fold-{fold}/groups.json")
        ids = stored["patient_ids"]
        if ids != groups["validation"] or prediction_sets[arm, seed] & set(ids):
            raise ValueError("Prediction patient membership differs or repeats")
        prediction_sets[arm, seed].update(ids)
        rows = pool.indices(ids)
        if not torch.equal(stored["labels"], pool.labels[rows]) or not torch.equal(
            stored["valid"], pool.valid[rows]
        ):
            raise ValueError("Exported endpoint labels differ")

        def guarded_open(file: Any, *args: Any, **kwargs: Any) -> Any:
            if any(
                word in str(file)
                for word in ("pool.pt", "tokens.pt", "bindings", ".xlsx", "ct1", "outcome")
            ):
                raise ValueError("Standalone inference tried to read source/future data")
            return original_open(file, *args, **kwargs)

        with patch("builtins.open", guarded_open), patch("io.open", guarded_open):
            replay = predict_bundle(
                path.parent / "bundle",
                [pool.clinical[p] for p in ids],
                [pool.treatments[p] for p in ids],
                pool.interval[rows],
                pool.ct0[rows],
            )
        if not torch.allclose(
            replay["probabilities"], stored["probabilities"], atol=1e-5, rtol=1e-5
        ):
            raise ValueError("Independent standalone probabilities differ")
        if not torch.equal(replay["decisions"], stored["decisions"]):
            raise ValueError("Independent standalone decisions differ")
        metrics = read_json(path.parent / "metrics.json")
        for endpoint, name in enumerate(("pcr", "recurrence")):
            actual_metrics = _metrics(
                stored["labels"][:, endpoint].numpy(), stored["probabilities"][:, endpoint].numpy()
            )
            for key, value in actual_metrics.items():
                expected = metrics["endpoints"][name][key]
                if (value is None) != (expected is None) or (
                    value is not None and abs(value - expected) > 1e-10
                ):
                    raise ValueError("Independent fold metric differs")
            metric_groups[arm, seed, name].append(actual_metrics)
        bundle_count += 1
    if bundle_count != expected_bundles:
        raise ValueError("Standalone prediction coverage is incomplete")
    expected_ids = (
        set(pool.ids)
        if not spec["smoke"]
        else set(read_json(root / "partitions/fold-0/groups.json")["validation"])
    )
    if any(ids != expected_ids for ids in prediction_sets.values()):
        raise ValueError("A model/seed prediction set does not cover the evaluation patients")
    summary = read_json(root / "evaluation/summary.json")
    if summary["across_seed_aggregation"] or summary["status"] != "complete":
        raise ValueError("Require complete separately reported seeds")
    if len(summary["per_seed_fold_summary"]) != len(metric_groups) * len(METRICS):
        raise ValueError("Per-seed metric summary coverage differs")
    summary_keys = set()
    for record in summary["per_seed_fold_summary"]:
        summary_identity = (record["arm"], record["seed"], record["endpoint"], record["metric"])
        if summary_identity in summary_keys or record["metric"] not in METRICS:
            raise ValueError("Duplicate or unexpected per-seed summary metric")
        summary_keys.add(summary_identity)
        values = [
            row[record["metric"]]
            for row in metric_groups[record["arm"], record["seed"], record["endpoint"]]
        ]
        supported = [value for value in values if value is not None]
        if record["fold_count"] != folds or record["supported_folds"] != len(supported):
            raise ValueError("Per-seed summary support count differs")
        mean = float(np.mean(supported)) if len(supported) == folds else None
        sd = float(np.std(supported, ddof=1)) if len(supported) == folds and folds > 1 else None
        for actual, expected in ((mean, record["mean"]), (sd, record["sample_standard_deviation"])):
            if (actual is None) != (expected is None) or (
                actual is not None and abs(actual - expected) > 1e-10
            ):
                raise ValueError("Fivefold mean/sample SD was not computed within the same seed")
    result = {
        "status": "passed",
        "phases": len(phase_keys),
        "checkpoint_score_replays": score_checks,
        "matching_generator_parents": parent_checks,
        "standalone_denied_source_bundles": bundle_count,
        "independent_input_and_anchor_refits": len(partitions),
        "independent_endpoint_metric_groups": bundle_count * 2,
        "per_seed_summary_checks": len(summary["per_seed_fold_summary"]),
        "optimizer_updates_during_audit": 0,
        "across_seed_aggregation": False,
    }
    atomic_write_private_json(root / "independent_verification.json", result)
    return result
