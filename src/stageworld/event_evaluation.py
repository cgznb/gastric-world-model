"""Development-fold metrics, reported separately for every random seed."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from stageworld.artifacts import atomic_write_private_json, read_json
from stageworld.generated651_evaluation import write_csv
from stageworld.generated700_evaluation import points

METRICS = (
    "auroc",
    "auprc",
    "accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "npv",
    "mcc",
    "balanced_accuracy",
    "bce",
    "brier",
)


def binary_metrics(probability: np.ndarray, labels: np.ndarray) -> dict:
    if len(probability) != len(labels) or not np.isfinite(probability).all():
        raise ValueError("Invalid probability rows")
    if not np.isin(labels, [0, 1]).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Invalid binary evaluation values")
    if not len(labels):
        return {
            "n": 0,
            "positive": 0,
            "negative": 0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            **dict.fromkeys(METRICS),
        }
    result = points(probability, labels, probability >= 0.5)
    tp, tn, fp, fn = (result[key] for key in ("tp", "tn", "fp", "fn"))
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    result.update(
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        npv=tn / (tn + fn) if tn + fn else None,
        mcc=(tp * tn - fp * fn) / denominator if denominator else None,
    )
    if not result["positive"] or not result["negative"]:
        result["auprc"] = None
    return result


def summarize(records: list[dict], expected_folds: int = 5) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in records:
        groups[row["seed"], row["endpoint"]].append(row)
    result = []
    for (seed, endpoint), rows in sorted(groups.items()):
        folds = [row["fold"] for row in rows]
        if len(set(folds)) != len(folds) or not set(folds) <= set(range(expected_folds)):
            raise ValueError("Unexpected or duplicated fold result")
        if len(rows) != expected_folds:
            continue
        for metric in METRICS:
            values = [row[metric] for row in rows if row[metric] is not None]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Nonfinite fold metric")
            complete = len(values) == expected_folds
            result.append(
                {
                    "seed": seed,
                    "endpoint": endpoint,
                    "metric": metric,
                    "fold_count": len(rows),
                    "supported_folds": len(values),
                    "mean": statistics.mean(values) if complete else None,
                    "sample_standard_deviation": statistics.stdev(values)
                    if complete and len(values) > 1
                    else None,
                }
            )
    return result


def evaluate(root: Path) -> dict:
    spec = read_json(root / "specification.json")
    records, identities, ct_records = [], set(), []
    for path in sorted(root.glob("fits/*/fold-*/metrics.json")):
        payload = read_json(path)
        identity = (payload["seed"], payload["fold"])
        if (
            identity in identities
            or identity[0] not in spec["seeds"]
            or identity[1] not in range(5)
        ):
            raise ValueError("Unexpected event fold result")
        identities.add(identity)
        for endpoint, values in payload["endpoints"].items():
            records.append(
                {
                    "seed": identity[0],
                    "fold": identity[1],
                    "endpoint": endpoint,
                    "selected_epoch": payload["selected_epoch"],
                    **values,
                }
            )
        for phase in ("pretrain", "joint"):
            folder = root / ("pretrain" if phase == "pretrain" else "fits")
            ct_path = folder / str(identity[0]) / f"fold-{identity[1]}" / "ct_evaluation.json"
            if ct_path.exists():
                report = read_json(ct_path)
                for method, values in report["methods"].items():
                    ct_records.append(
                        {
                            "seed": identity[0],
                            "fold": identity[1],
                            "phase": phase,
                            "method": method,
                            "n": report["n"],
                            **values,
                        }
                    )
    summaries = summarize(records)
    complete_seeds = [
        seed for seed in spec["seeds"] if all((seed, fold) in identities for fold in range(5))
    ]
    result = {
        "status": "complete" if len(identities) == len(spec["seeds"]) * 5 else "partial",
        "completed_folds": len(identities),
        "expected_folds": len(spec["seeds"]) * 5,
        "complete_seeds": complete_seeds,
        "fold_metrics": records,
        "per_seed_fold_summary": summaries,
        "across_seed_aggregation": False,
        "independent_evaluation": False,
        "threshold": 0.5,
        "recall_is_sensitivity": True,
        "calibrated": False,
    }
    output = root / "evaluation"
    atomic_write_private_json(output / "summary.json", result)
    write_csv(output / "fold_metrics.csv", records)
    write_csv(output / "per_seed_fold_summary.csv", summaries)
    write_csv(output / "generation_fold_metrics.csv", ct_records)
    lines = [
        "# Event Multistage Development Results",
        "",
        "Every seed has its own fivefold mean and sample SD. Seeds are never pooled.",
        "Validation selects checkpoints and supplies these development scores.",
        "The primary endpoint is recorded recurrence/metastasis 0/1, with no time horizon.",
        "pCR is an auxiliary target at the surgical state. Recall equals sensitivity.",
        "Classification uses threshold 0.5; weighted scores are not calibrated probabilities.",
        f"Completed folds: {len(identities)}/{len(spec['seeds']) * 5}.",
        "",
        "| Seed | Endpoint | Metric | Fivefold mean | Sample SD |",
        "| ---: | --- | --- | ---: | ---: |",
    ]
    for row in summaries:
        mean = "NA" if row["mean"] is None else f"{row['mean']:.5f}"
        sd = row["sample_standard_deviation"]
        lines.append(
            f"| {row['seed']} | {row['endpoint']} | {row['metric']} | "
            f"{mean} | {'NA' if sd is None else f'{sd:.5f}'} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return result
