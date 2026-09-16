"""Pooled700 OOF reports with matched patient resampling across seeds."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from stageworld.artifacts import atomic_write_private_json
from stageworld.generated700_data import Pool

RULES = ("raw_0.5", "inner_balanced", "inner_sensitivity_0.8", "calibrated_0.5")


def points(probability: np.ndarray, labels: np.ndarray, decision: np.ndarray) -> dict[str, Any]:
    n = len(labels)
    if not n:
        return {
            k: None
            for k in (
                "accuracy",
                "sensitivity",
                "specificity",
                "precision",
                "balanced_accuracy",
                "auroc",
                "auprc",
                "bce",
                "brier",
            )
        }
    positive, negative = labels == 1, labels == 0
    tp, tn = int((decision & positive).sum()), int((~decision & negative).sum())
    fp, fn = int((decision & negative).sum()), int((~decision & positive).sum())
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    p = np.clip(probability, 1e-7, 1 - 1e-7)
    return {
        "n": n,
        "positive": int(positive.sum()),
        "negative": int(negative.sum()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / n,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": tp / (tp + fp) if tp + fp else None,
        "balanced_accuracy": (sensitivity + specificity) / 2
        if sensitivity is not None and specificity is not None
        else None,
        "auroc": float(roc_auc_score(labels, probability))
        if positive.any() and negative.any()
        else None,
        "auprc": float(average_precision_score(labels, probability)) if positive.any() else None,
        "bce": float(-(labels * np.log(p) + (1 - labels) * np.log1p(-p)).mean()),
        "brier": float(((probability - labels) ** 2).mean()),
    }


def evaluate(root: Path, pool: Pool, *, expected: int = 700, replicates: int = 1000) -> dict:
    output = root / "evaluation"
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    grouped: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for path in root.glob("fits/*/*/fold-*/predictions.pt"):
        grouped[path.parents[2].name][path.parents[1].name].append(path)
    if not grouped:
        raise ValueError("No completed OOF predictions")
    truth = {p: i for i, p in enumerate(pool.ids)}
    all_results: dict[str, Any] = {}
    csv_rows = []
    predictions: dict[str, list[dict[str, np.ndarray]]] = {}
    common_ids: list[str] | None = None
    for arm, seeds in sorted(grouped.items()):
        all_results[arm] = {}
        predictions[arm] = []
        for seed, files in sorted(seeds.items()):
            pieces = [torch.load(p, weights_only=True, map_location="cpu") for p in sorted(files)]
            ids = [p for item in pieces for p in item["patient_ids"]]
            if len(ids) != expected or len(set(ids)) != expected:
                raise ValueError("OOF patient coverage differs")
            order = sorted(range(len(ids)), key=lambda i: ids[i])
            sorted_ids = [ids[i] for i in order]
            if common_ids is None:
                common_ids = sorted_ids
            elif common_ids != sorted_ids:
                raise ValueError("Candidate OOF populations differ")
            indices = torch.tensor([truth[p] for p in sorted_ids])
            data = {
                k: torch.cat([item[k] for item in pieces])[order]
                for k in ("probabilities", "calibrated", *RULES, "labels", "valid")
            }
            if not torch.equal(data["labels"], pool.labels[indices]) or not torch.equal(
                data["valid"], pool.valid[indices]
            ):
                raise ValueError("OOF label or missing mask differs")
            result: dict[str, Any] = {}
            for i, endpoint in enumerate(("pcr", "recurrence")):
                mask = data["valid"][:, i].numpy()
                y = data["labels"][:, i].numpy()[mask]
                result[endpoint] = {}
                for rule in RULES:
                    pkey = "calibrated" if rule == "calibrated_0.5" else "probabilities"
                    metrics = points(
                        data[pkey][:, i].numpy()[mask], y, data[rule][:, i].numpy()[mask]
                    )
                    result[endpoint][rule] = metrics
                    csv_rows.append(
                        {"arm": arm, "seed": seed, "endpoint": endpoint, "rule": rule, **metrics}
                    )
            all_results[arm][seed] = result
            predictions[arm].append({k: v.numpy() for k, v in data.items()})
            torch.save({"patient_ids": sorted_ids, **data}, output / f"{arm}-{seed}-oof.pt")
    summary = []
    for arm, seeds in all_results.items():
        for endpoint in ("pcr", "recurrence"):
            for rule in RULES:
                for metric in (
                    "auroc",
                    "auprc",
                    "bce",
                    "brier",
                    "accuracy",
                    "sensitivity",
                    "specificity",
                    "precision",
                    "balanced_accuracy",
                ):
                    values = [
                        x[endpoint][rule][metric]
                        for x in seeds.values()
                        if x[endpoint][rule][metric] is not None
                    ]
                    summary.append(
                        {
                            "arm": arm,
                            "endpoint": endpoint,
                            "rule": rule,
                            "metric": metric,
                            "mean": float(np.mean(values)) if values else None,
                            "standard_deviation": float(np.std(values, ddof=1))
                            if len(values) > 1
                            else None,
                            "seeds": len(values),
                        }
                    )
    for filename, rows in (("per_seed_metrics.csv", csv_rows), ("seed_summary.csv", summary)):
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (output / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    if "logistic" not in predictions:
        raise ValueError("The fixed unweighted comparator is absent")
    rng = np.random.default_rng(17)
    draws = rng.integers(0, expected, (replicates, expected))
    baseline = predictions["logistic"][0]
    comparisons = {}
    for arm, values in predictions.items():
        if arm == "logistic":
            continue
        paired = {}
        for i, endpoint in enumerate(("pcr", "recurrence")):
            for rule in ("inner_balanced", "inner_sensitivity_0.8"):
                samples: dict[str, list[float]] = defaultdict(list)
                for draw in draws:
                    rows = draw[baseline["valid"][draw, i]]
                    y = baseline["labels"][rows, i]
                    reference = points(
                        baseline["probabilities"][rows, i], y, baseline[rule][rows, i]
                    )
                    actual = [
                        points(v["probabilities"][rows, i], y, v[rule][rows, i]) for v in values
                    ]
                    for metric in (
                        "auroc",
                        "auprc",
                        "sensitivity",
                        "specificity",
                        "balanced_accuracy",
                    ):
                        estimates = [m[metric] for m in actual]
                        if reference[metric] is not None and all(v is not None for v in estimates):
                            samples[metric].append(float(np.mean(estimates) - reference[metric]))
                paired[endpoint + "/" + rule] = {
                    k: {
                        "difference_ci95": np.quantile(v, [0.025, 0.975]).tolist()
                        if len(v) >= max(20, int(0.8 * replicates))
                        else None,
                        "valid_resamples": len(v),
                    }
                    for k, v in samples.items()
                }
                full_rows = np.flatnonzero(baseline["valid"][:, i])
                full_y = baseline["labels"][full_rows, i]
                full_reference = points(
                    baseline["probabilities"][full_rows, i], full_y, baseline[rule][full_rows, i]
                )
                full_actual = [
                    points(value["probabilities"][full_rows, i], full_y, value[rule][full_rows, i])
                    for value in values
                ]
                for metric, interval in paired[endpoint + "/" + rule].items():
                    estimates = [item[metric] for item in full_actual]
                    interval["difference"] = (
                        float(np.mean(estimates) - full_reference[metric])
                        if full_reference[metric] is not None
                        and all(v is not None for v in estimates)
                        else None
                    )
        comparisons[arm] = paired
    report = {
        "status": "completed",
        "patients": expected,
        "per_seed": all_results,
        "seed_summary": summary,
        "paired_vs_logistic": comparisons,
        "bootstrap_replicates": replicates,
        "bootstrap_unit": "patient_shared_across_seeds_no_refitting",
        "former_holdout_merged": True,
        "external_validation": False,
        "interpretation": "reused_development_data_nested_selection_no_external_validation",
        "all_prespecified_candidates_retained": True,
    }
    atomic_write_private_json(output / "summary.json", report)
    return report
