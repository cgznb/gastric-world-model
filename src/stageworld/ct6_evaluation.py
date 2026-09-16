"""CT6 controls, calibration and mean-across-seeds paired comparisons."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from torch import Tensor

from stageworld.evaluation import (
    CalibrationBinSpec,
    CensoringDistribution,
    EvaluationCohort,
    SplitRole,
    calibration_curve,
    cumulative_dynamic_auc,
    integrated_brier_score,
    ipcw_brier_score,
    ipcw_concordance_index,
    paired_patient_bootstrap,
)
from stageworld.generated_evaluation import annotate_paired_interval_support, nll_by_patient
from stageworld.real_survival import _labels
from stageworld.survival import risk_probability
from stageworld.training import WorldModelBatch


def stage_cohort(
    batches: Sequence[WorldModelBatch], stage: int, *, train: bool
) -> EvaluationCohort:
    times, events = _labels(batches)
    return EvaluationCohort(
        "ct6-train" if train else "ct6-validation",
        SplitRole.TRAIN if train else SplitRole.DEVELOPMENT,
        "os",
        tuple(p for b in batches for p in b.patient_ids),
        tuple(times[:, stage].tolist()),
        tuple(events[:, stage].tolist()),
    )


def calibration_report(
    rates: Tensor,
    train: Sequence[WorldModelBatch],
    validation: Sequence[WorldModelBatch],
    cuts: Sequence[float],
) -> list[dict[str, Any]]:
    result = []
    for stage in range(2):
        cohort = stage_cohort(validation, stage, train=False)
        censor = CensoringDistribution.fit(stage_cohort(train, stage, train=True))
        risks = risk_probability(rates[:, stage], torch.tensor([3.0]), tuple(cuts))[:, 0].tolist()
        bins = CalibrationBinSpec.fit(
            cohort, risks, bins=3, source_prediction_artifact_id="ct6-selected"
        )
        result.append(
            {
                "stage": "S0" if stage == 0 else "S1_pred",
                "horizon": 3.0,
                "bin_rule": "prediction_tertiles_no_outcome_fitting",
                "small_sample_descriptive": True,
                **asdict(calibration_curve(cohort, risks, 3.0, censor, bins)),
            }
        )
    return result


def persistence_diagnostics(validation: Sequence[WorldModelBatch]) -> dict[str, Any]:
    before, targets = [], []
    for batch in validation:
        valid = batch.ct0.valid[..., None]
        before.append(batch.ct0.values.masked_fill(~valid, 0).sum(1) / valid.sum(1).clamp_min(1))
        targets.append(batch.future_ct_target.float().flatten(1))
    prediction, target = torch.cat(before).float(), torch.cat(targets)
    return {
        "persistence_mse": float((prediction - target).square().mean()),
        "persistence_mae": float((prediction - target).abs().mean()),
        "persistence_cosine": float(
            torch.nn.functional.cosine_similarity(prediction, target).mean()
        ),
    }


def seed_averaged_comparisons(
    predictions: dict[int, dict[str, Tensor]],
    train: Sequence[WorldModelBatch],
    validation: Sequence[WorldModelBatch],
    cuts: Sequence[float],
    *,
    replicates: int,
) -> list[dict[str, Any]]:
    seeds = sorted(predictions)
    ids = tuple(p for b in validation for p in b.patient_ids)
    n = len(ids)
    all_times, all_events = _labels(validation)
    results = []
    for stage in range(2):
        censor = CensoringDistribution.fit(stage_cohort(train, stage, train=True))
        times, events = all_times[:, stage].numpy(), all_events[:, stage].numpy()
        for other in ("history_only", "generated_only"):
            ordered = [[predictions[s][m] for s in seeds] for m in ("history_generated", other)]
            nll = np.array(
                [
                    [nll_by_patient(r, validation, cuts)[:, stage].numpy() for r in group]
                    for group in ordered
                ]
            )
            grid = torch.linspace(0.25, 3.0, 12)
            curves = np.array(
                [
                    [risk_probability(r[:, stage], grid, tuple(cuts)).numpy() for r in group]
                    for group in ordered
                ]
            )
            specifications: list[tuple[str, float | None, Any]] = [("nll", None, None)]
            specifications += [
                (f.__name__, t, f)
                for t in (1.0, 3.0)
                for f in (
                    cumulative_dynamic_auc,
                    ipcw_concordance_index,
                    ipcw_brier_score,
                )
            ]
            specifications.append(("integrated_brier_score", None, integrated_brier_score))
            for metric, horizon, function in specifications:
                risks = (
                    None
                    if horizon is None
                    else np.array(
                        [
                            [
                                risk_probability(r[:, stage], torch.tensor([horizon]), tuple(cuts))[
                                    :, 0
                                ].numpy()
                                for r in group
                            ]
                            for group in ordered
                        ]
                    )
                )

                def statistic(
                    values: Any,
                    indices: Any,
                    *,
                    metric_name: str = metric,
                    t: float | None = horizon,
                    f: Any = function,
                    risk: Any = risks,
                    losses: Any = nll,
                    probability_curves: Any = curves,
                    durations: Any = times,
                    outcomes: Any = events,
                    c: Any = censor,
                    horizons: Any = grid,
                ) -> float:
                    group = int(values[0]) // n
                    rows = values.astype(int) % n
                    if metric_name == "nll":
                        return float(losses[group][:, rows].mean())
                    cohort = EvaluationCohort(
                        "ct6-seed-draw",
                        SplitRole.DEVELOPMENT,
                        "os",
                        tuple(f"draw-{i}" for i in range(len(rows))),
                        tuple(durations[indices].tolist()),
                        tuple(outcomes[indices].tolist()),
                    )
                    estimates = []
                    for s in range(len(seeds)):
                        score = (
                            f(
                                cohort,
                                probability_curves[group, s, rows].tolist(),
                                horizons.tolist(),
                                c,
                            )
                            if t is None
                            else f(cohort, risk[group, s, rows].tolist(), t, c)
                        )
                        if not score.estimable:
                            return float("nan")
                        estimates.append(score.estimate)
                    return float(np.mean(estimates))

                estimate = paired_patient_bootstrap(
                    ids,
                    list(range(n)),
                    list(range(n, 2 * n)),
                    statistic,
                    n_resamples=replicates,
                    seed=17,
                )
                row = {
                    "stage": "S0" if stage == 0 else "S1_pred",
                    "metric": metric,
                    "horizon": horizon,
                    "comparison": f"history_generated_minus_{other}",
                    "seeds": seeds,
                    "aggregation": "mean_of_seed_metric_differences",
                    "interval_scope": (
                        "patient_sampling_conditional_on_selected_seed_models_not_selection_adjusted"
                    ),
                    **asdict(estimate),
                }
                annotate_paired_interval_support(row)
                results.append(row)
    return results


def aggregate_seed_metrics(
    arms: dict[str, dict[str, Any]], *, profile_prefix: str = "regularized/"
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, Any], list[float]] = {}
    for key, report in arms.items():
        if not key.startswith(profile_prefix):
            continue
        mode = report["readout_mode"]
        for stage, value in report["validation_nll"].items():
            groups.setdefault((mode, stage, "nll", None), []).append(value)
        for row in report["metrics"]:
            if row["status"] == "ok":
                groups.setdefault(
                    (mode, row["stage"], row["metric"], row.get("horizon")), []
                ).append(row["estimate"])
    return [
        {
            "readout_mode": mode,
            "stage": stage,
            "metric": metric,
            "horizon": horizon,
            "n_seeds_estimable": len(values),
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        }
        for (mode, stage, metric, horizon), values in groups.items()
    ]
