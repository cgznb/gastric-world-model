"""Locked OOF comparisons; patient bootstrap keeps all seed fits together."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stageworld.artifacts import atomic_write_private_json, read_json
from stageworld.binary_baselines import operating_metrics
from stageworld.binary_endpoints import ENDPOINTS, LABEL_CONTRACT
from stageworld.binary_evaluation import _point, evaluate_binary
from stageworld.synthetic_workflow import _atomic_torch_save


def pool_predictions(paths: list[Path], expected_ids: set[str]) -> dict[str, Any]:
    folds = [torch.load(path, weights_only=True, map_location="cpu") for path in paths]
    ids = [p for fold in folds for p in fold["patient_ids"]]
    if len(ids) != len(set(ids)) or set(ids) != expected_ids:
        raise ValueError("Each OOF patient must occur exactly once")
    order = torch.tensor(sorted(range(len(ids)), key=lambda i: ids[i]))
    result = {"patient_ids": [ids[i] for i in order.tolist()], "label_contract": LABEL_CONTRACT}
    for name in ("probabilities", "labels", "valid", "prevalence", "thresholds"):
        result[name] = torch.cat([fold[name] for fold in folds])[order]
    if all("ct_mean" in fold for fold in folds):
        for name in ("ct_mean", "ct_target", "ct_persistence", "ct_training_mean"):
            result[name] = torch.cat([fold[name] for fold in folds])[order]
    return result


def paired_comparison(
    candidate: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    *,
    replicates: int,
) -> list[dict[str, Any]]:
    if len(candidate) != len(reference):
        raise ValueError("Pair the same model seeds")
    first = candidate[0]
    for row in (*candidate, *reference):
        if (
            row["patient_ids"] != first["patient_ids"]
            or not torch.equal(row["valid"], first["valid"])
            or not torch.equal(row["labels"][row["valid"]], first["labels"][first["valid"]])
        ):
            raise ValueError("Paired comparison requires identical patients and labels")
    y, valid = first["labels"].numpy(), first["valid"].numpy()
    p = np.stack([row["probabilities"].double().numpy() for row in candidate])
    q = np.stack([row["probabilities"].double().numpy() for row in reference])
    rng = np.random.default_rng(17)
    draws = rng.integers(0, len(y), size=(replicates, len(y)))
    result = []
    for index, endpoint in enumerate(ENDPOINTS):

        def difference(rows: np.ndarray, index: int = index) -> dict[str, float | None]:
            selected = rows[valid[rows, index]]
            values: dict[str, list[float]] = {k: [] for k in ("bce", "auroc", "auprc", "brier")}
            for seed_index in range(len(candidate)):
                a = _point(y[selected, index], p[seed_index, selected, index])
                b = _point(y[selected, index], q[seed_index, selected, index])
                for key in values:
                    a_value, b_value = a[key], b[key]
                    if a_value is not None and b_value is not None:
                        values[key].append(a_value - b_value)
            return {
                k: statistics.mean(v) if len(v) == len(candidate) else None
                for k, v in values.items()
            }

        estimate = difference(np.arange(len(y)))
        samples: dict[str, list[float]] = {k: [] for k in estimate}
        for draw in draws:
            for key, value in difference(draw).items():
                if value is not None:
                    samples[key].append(value)
        for key, value in estimate.items():
            supported = value is not None and len(samples[key]) >= max(20, int(0.8 * replicates))
            result.append(
                {
                    "endpoint": endpoint,
                    "metric": key,
                    "candidate_minus_reference": value,
                    "confidence_interval_95": np.quantile(samples[key], [0.025, 0.975]).tolist()
                    if supported
                    else None,
                    "bootstrap_valid_replicates": len(samples[key]),
                    "bootstrap_replicates": replicates,
                    "scope": "same_patient_resample_across_all_seeds_conditional_on_fitted_models",
                }
            )
    return result


def aggregate_study(
    root: Path,
    *,
    arms: list[str],
    seeds: list[int],
    folds: list[int],
    patient_ids: set[str],
    reference_root: Path,
    smoke: bool,
) -> dict[str, Any]:
    destination = root / "evaluation"
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    reports: dict[str, Any] = {}
    all_predictions: dict[str, list[dict[str, Any]]] = {}
    for name in ["clinical", "ct0", *arms]:
        fits = seeds if name in arms else [17]
        reports[name], all_predictions[name] = {}, []
        for seed in fits:
            folder = root / (f"models/{name}/seed-{seed}" if name in arms else f"baselines/{name}")
            pooled = pool_predictions(
                [folder / f"fold-{fold}/outer_predictions.pt" for fold in folds], patient_ids
            )
            _atomic_torch_save(folder / "oof_predictions.pt", pooled)
            all_predictions[name].append(pooled)
            report_path = destination / f"{name}-seed-{seed}.json"
            if report_path.exists():
                report = read_json(report_path)
            else:
                report = {
                    "oof": evaluate_binary(
                        pooled["probabilities"],
                        pooled["labels"],
                        pooled["valid"],
                        replicates=0 if smoke else 1000,
                    ),
                    "inner_selected_thresholds": operating_metrics(
                        pooled["probabilities"],
                        pooled["labels"],
                        pooled["valid"],
                        pooled["thresholds"],
                    ),
                    "patients": len(patient_ids),
                    "oof_patient_coverage_verified": True,
                }
                if "ct_mean" in pooled:
                    report["ct_mse"] = {
                        k: float((pooled[k] - pooled["ct_target"]).square().mean())
                        for k in ("ct_mean", "ct_persistence", "ct_training_mean")
                    }
                atomic_write_private_json(report_path, report)
            reports[name][str(seed)] = report
    seed_summary = []
    for name, seed_reports in reports.items():
        for endpoint in ENDPOINTS:
            for metric in (
                "bce",
                "auroc",
                "auprc",
                "brier",
                "sensitivity",
                "specificity",
                "balanced_accuracy",
                "accuracy",
            ):
                values = [
                    row["oof"]["endpoints"][endpoint]["metrics"][metric]["estimate"]
                    for row in seed_reports.values()
                ]
                present = [v for v in values if v is not None]
                seed_summary.append(
                    {
                        "arm": name,
                        "endpoint": endpoint,
                        "metric": metric,
                        "n_seeds": len(present),
                        "mean": statistics.mean(present) if present else None,
                        "standard_deviation": statistics.stdev(present)
                        if len(present) > 1
                        else None,
                    }
                )
    direct = all_predictions["direct"]
    prevalence = [{**row, "probabilities": row["prevalence"]} for row in direct]
    comparisons = {}
    reference = []
    if not smoke:
        for seed in seeds:
            row = torch.load(
                reference_root / f"seed-{seed}/oof_predictions.pt",
                weights_only=True,
                map_location="cpu",
            )
            index = {p: i for i, p in enumerate(row["patient_ids"])}
            order = torch.tensor([index[p] for p in direct[0]["patient_ids"]])
            reference.append(
                {
                    "patient_ids": direct[0]["patient_ids"],
                    **{k: row[k][order] for k in ("probabilities", "labels", "valid")},
                }
            )
    for name in ["clinical", "ct0", *arms]:
        candidate = all_predictions[name]
        if name not in arms:
            candidate = candidate * len(seeds)
        references = {"prevalence": prevalence}
        if name != "direct":
            references["direct"] = direct
        if reference:
            references["original_stochastic"] = reference
        for reference_name, compared in references.items():
            key = f"{name}-vs-{reference_name}"
            path = destination / f"{key}.json"
            if path.exists():
                comparisons[key] = read_json(path)["metrics"]
            else:
                comparisons[key] = paired_comparison(
                    candidate, compared, replicates=0 if smoke else 1000
                )
                atomic_write_private_json(path, {"metrics": comparisons[key]})
    result = {
        "status": "completed",
        "arms": reports,
        "seed_metrics": seed_summary,
        "paired_comparisons": comparisons,
        "test_used": False,
        "outer_selected_best_arm": False,
        "interpretation": (
            "Prespecified candidate comparisons on previously used development "
            "patients; CIs do not correct for model search or prior OOF inspection."
        ),
    }
    atomic_write_private_json(destination / "summary.json", result)
    lines = [
        "# Gastric Binary Improvement OOF Results",
        "",
        "Fixed five patient folds; model seeds 17, 43 and 97. Model and threshold selection "
        "uses only the corresponding inner validation. Logistic fits are deterministic.",
        "",
        "pCR is BJ recorded status. Recurrence/metastasis is CG0/1, with no fixed time window. "
        "The previously opened 106-person holdout is excluded.",
        "",
        "## Seed Metrics",
        "",
        "| Arm | Endpoint | Metric | Mean | Seed SD | Fits |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in seed_summary:
        mean = "" if row["mean"] is None else f"{row['mean']:.4f}"
        sd = "" if row["standard_deviation"] is None else f"{row['standard_deviation']:.4f}"
        lines.append(
            f"| {row['arm']} | {row['endpoint']} | {row['metric']} "
            f"| {mean} | {sd} | {row['n_seeds']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "Sensitivity, specificity and accuracy above use threshold 0.5. Inner-selected "
        "threshold operating metrics are reported separately in summary.json.",
        "",
        "The JSON report includes 1000-replicate paired patient bootstrap comparisons "
        "against direct fusion, training prevalence and the original stochastic model. "
        "Each patient resample keeps all seeds together; it does not retrain models or "
        "correct for earlier development-set inspection. Unsupported intervals are null.",
        "",
        "True CT1 and single-task classification diagnostics are inner-only. "
        "Tokenwise future supervision is disabled because paired anatomical registration "
        "and phase matching are unverified. See ../geometry_audit.json and ../summary.json.",
        "",
    ]
    if smoke:
        lines.insert(
            2,
            "SMOKE ONLY: 32/32/32 patients, one fold/seed, two epochs. "
            "Not a performance estimate.\n",
        )
    (destination / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result
