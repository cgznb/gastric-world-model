"""Report fivefold mean and sample SD separately for every model and seed."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from stageworld.artifacts import atomic_write_private_json, read_json
from stageworld.generated651_spec import NEURAL

METRICS = (
    "auroc",
    "auprc",
    "accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "balanced_accuracy",
    "bce",
    "brier",
)


def summarize_folds(records: list[dict], *, expected_folds: int = 5) -> list[dict]:
    grouped: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["arm"], record["seed"], record["endpoint"]].append(record)
    result = []
    for (arm, seed, endpoint), rows in sorted(grouped.items()):
        folds = [row["fold"] for row in rows]
        if len(set(folds)) != len(folds) or not set(folds) <= set(range(expected_folds)):
            raise ValueError("Duplicate or unexpected fold in per-seed results")
        if len(rows) != expected_folds:
            continue
        for metric in METRICS:
            values = [row[metric] for row in rows if row[metric] is not None]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Nonfinite reported metric")
            supported = len(values) == expected_folds
            result.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "endpoint": endpoint,
                    "metric": metric,
                    "fold_count": len(rows),
                    "supported_folds": len(values),
                    "mean": float(statistics.mean(values)) if supported else None,
                    "sample_standard_deviation": float(statistics.stdev(values))
                    if supported and len(values) > 1
                    else None,
                }
            )
    return result


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows and fields is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def metric_text(row: dict) -> str:
    if row["mean"] is None:
        return f"NA ({row['supported_folds']}/{row['fold_count']} supported)"
    scale = 1 if row["metric"] in ("auroc", "auprc", "bce", "brier") else 100
    value = f"{row['mean'] * scale:.3f}" if scale == 1 else f"{row['mean'] * scale:.1f}%"
    if row["sample_standard_deviation"] is not None:
        sd = row["sample_standard_deviation"] * scale
        value += f" +/- {sd:.3f}" if scale == 1 else f" +/- {sd:.1f}%"
    return value


def _render(output: Path, result: dict) -> None:
    rows = result["per_seed_fold_summary"]
    lookup = {(r["arm"], r["seed"], r["endpoint"], r["metric"]): r for r in rows}
    smoke = result["smoke"]
    lines = [
        "# Generated V2 Complete651: Separate Results For Each Seed",
        "",
        "SMOKE ONLY: one 32-patient validation subset; no fivefold SD."
        if smoke
        else "Each value is the arithmetic mean +/- sample SD across five patient folds",
        "WITHIN ONE seed. The ten seeds are never averaged together.",
        "Validation folds select checkpoints and supply the reported scores. These",
        "are development validation results, not independent performance estimates.",
        "CT1 is a training target only; prediction requires CT0/clinical/treatment/time.",
        "Fixed anchor C=1, residual alpha=1, decision threshold=0.5; no calibration.",
        "Weighted raw scores are not calibrated probabilities.",
        "",
        f"Completed model/fold predictions: {result['prediction_files']}/"
        f"{result['expected_prediction_files']}.",
        "Incomplete fivefold groups are not assigned a fivefold mean or SD.",
        "",
        "| Seed | Method | pCR AUROC | Recurrence AUROC | pCR AUPRC | Recurrence AUPRC |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for seed in result["complete_seeds"]:
        seed_lines = [
            f"# Seed {seed}: Smoke Subset Only"
            if smoke
            else f"# Seed {seed}: Fivefold Mean And Sample SD",
            "",
            "One validation subset only; this is a pipeline check, not a fivefold result."
            if smoke
            else "All means and SDs on this page are across this seed's five folds.",
            "The reported validation folds also selected the checkpoints.",
            "",
            "| Method | Endpoint | AUROC | AUPRC | Accuracy | Sensitivity | "
            "Specificity | Precision |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for candidate in NEURAL:
            arm = candidate.name
            cells = [
                metric_text(lookup[arm, seed, endpoint, metric])
                for metric in ("auroc", "auprc")
                for endpoint in ("pcr", "recurrence")
            ]
            lines.append("| " + " | ".join([str(seed), arm, *cells]) + " |")
            for endpoint in ("pcr", "recurrence"):
                values = [
                    metric_text(lookup[arm, seed, endpoint, metric])
                    for metric in (
                        "auroc",
                        "auprc",
                        "accuracy",
                        "sensitivity",
                        "specificity",
                        "precision",
                    )
                ]
                seed_lines.append("| " + " | ".join([arm, endpoint, *values]) + " |")
        seed_lines.extend(
            [
                "",
                "## Individual Fold Results",
                "",
                "| Method | Fold | Endpoint | Patients | Positives | Selected epoch | "
                "AUROC | AUPRC |",
                "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for record in result["fold_metrics"]:
            if record["seed"] == seed:
                seed_lines.append(
                    f"| {record['arm']} | {record['fold'] + 1} | {record['endpoint']} | "
                    f"{record['n']} | {record['positive']} | {record['selected_epoch']} | "
                    f"{record['auroc']:.4f} | {record['auprc']:.4f} |"
                )
        (output / "per_seed").mkdir(parents=True, exist_ok=True, mode=0o700)
        (output / "per_seed" / f"seed-{seed}.md").write_text("\n".join(seed_lines) + "\n")
    lines.extend(
        [
            "",
            "Individual fold metrics, selected epochs and support counts are retained in CSV/JSON.",
            "Fold SD describes fold-to-fold variation, "
            "not a confidence interval from independent samples.",
            "The population and selection protocol differ from the old700 study; their scores",
            "must not be compared as a controlled architecture improvement.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")


def evaluate(root: Path) -> dict:
    spec = read_json(root / "specification.json")
    fold_count = 1 if spec["smoke"] else 5
    seeds = spec["active_seeds"]
    arms = {candidate.name for candidate in NEURAL}
    records, identities = [], set()
    for path in sorted(root.glob("fits/*/*/fold-*/metrics.json")):
        payload = read_json(path)
        identity = (payload["arm"], payload["seed"], payload["fold"])
        if (
            identity in identities
            or identity[0] not in arms
            or identity[1] not in seeds
            or identity[2] not in range(fold_count)
        ):
            raise ValueError("Unexpected or repeated candidate/seed/fold result")
        if payload["reporting_partition"] != "model_selection_validation":
            raise ValueError("Report must identify validation selection")
        identities.add(identity)
        for endpoint, values in payload["endpoints"].items():
            records.append(
                {
                    "arm": payload["arm"],
                    "seed": payload["seed"],
                    "fold": payload["fold"],
                    "endpoint": endpoint,
                    "selected_epoch": payload["selected_epoch"],
                    **values,
                }
            )
    records.sort(key=lambda r: (r["seed"], r["arm"], r["fold"], r["endpoint"]))
    summaries = summarize_folds(records, expected_folds=fold_count)
    complete_seeds = [
        seed
        for seed in seeds
        if all((arm, seed, fold) in identities for arm in arms for fold in range(fold_count))
    ]
    expected = len(seeds) * len(arms) * fold_count
    result = {
        "schema": "generated651-per-seed-fold-report-v1",
        "status": "complete" if len(identities) == expected else "partial",
        "smoke": spec["smoke"],
        "prediction_files": len(identities),
        "expected_prediction_files": expected,
        "complete_seeds": complete_seeds,
        "aggregation": "within_each_seed_unweighted_fold_mean_and_sample_SD",
        "across_seed_aggregation": False,
        "independent_evaluation": False,
        "fold_metrics": records,
        "per_seed_fold_summary": summaries,
    }
    output = root / "evaluation"
    atomic_write_private_json(output / "summary.json", result)
    write_csv(output / "fold_metrics.csv", records)
    write_csv(output / "per_seed_fold_summary.csv", summaries)
    _render(output, result)
    return result
