"""Matched-input controls and paired development evaluation for generated S1."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from torch import Tensor

from stageworld.errors import DataContractError
from stageworld.evaluation import (
    CensoringDistribution,
    EvaluationCohort,
    SplitRole,
    cumulative_dynamic_auc,
    integrated_brier_score,
    ipcw_brier_score,
    ipcw_concordance_index,
    make_patient_bootstrap_plan,
    paired_patient_bootstrap,
)
from stageworld.generated_training import GeneratedTrainingBatch
from stageworld.model.generated_s1 import GeneratedS1Model
from stageworld.real_survival import _labels, evaluate_rates
from stageworld.survival import piecewise_exponential_nll, risk_probability
from stageworld.training import WorldModelBatch


@torch.inference_mode()
def predict_rates(model: GeneratedS1Model, batches: Sequence[GeneratedTrainingBatch]) -> Tensor:
    model.eval()
    device = next(model.parameters()).device
    result = []
    for batch in batches:
        output = model(**batch.to(device).prediction_inputs(deterministic=True))
        if output.survival_s0 is None:
            raise DataContractError(code="S1_ONLY_EVALUATOR", message="Use the S1-only evaluator.")
        result.append(
            torch.stack((output.survival_s0.rates, output.survival_s1_pred.rates), dim=1).cpu()
        )
    rates = torch.cat(result).float().squeeze(-1)
    if rates.ndim != 3 or not torch.isfinite(rates).all() or (rates <= 0).any():
        raise DataContractError(code="GENERATED_RATES_INVALID", message="Invalid survival rates.")
    return rates


@torch.inference_mode()
def predict_s1_rates(model: GeneratedS1Model, batches: Sequence[GeneratedTrainingBatch]) -> Tensor:
    model.eval()
    device = next(model.parameters()).device
    result = []
    for batch in batches:
        output = model(**batch.to(device).prediction_inputs(deterministic=True))
        if output.survival_s0 is not None:
            raise DataContractError(code="S1_ONLY_MODEL", message="Use the S1-only task.")
        result.append(output.survival_s1_pred.rates.cpu().squeeze(-1).unsqueeze(1))
    rates = torch.cat(result).float()
    if (
        rates.ndim != 3
        or rates.shape[1] != 1
        or not torch.isfinite(rates).all()
        or (rates <= 0).any()
    ):
        raise DataContractError(code="S1_ONLY_RATES", message="Invalid S1 rates.")
    return rates


def s1_nll_by_patient(
    rates: Tensor, batches: Sequence[WorldModelBatch], cuts: Sequence[float]
) -> Tensor:
    if (
        rates.ndim != 3
        or rates.shape[1] != 1
        or any(
            b.survival_valid[:, (0, 2)].any() or not b.survival_valid[:, 1].all() for b in batches
        )
    ):
        raise DataContractError(
            code="S1_ONLY_LABELS", message="Bind one valid S1 label per patient."
        )
    durations = torch.cat([b.survival_durations[:, 1] for b in batches])
    events = torch.cat([b.survival_events[:, 1] for b in batches])
    return piecewise_exponential_nll(
        rates[:, 0], durations, events, tuple(cuts), reduction="none", zero_time_policy="allow"
    )


def evaluate_s1_rates(
    rates: Tensor,
    train: Sequence[WorldModelBatch],
    validation: Sequence[WorldModelBatch],
    cuts: Sequence[float],
    *,
    method: str,
    prediction_id: str,
    replicates: int,
) -> dict[str, Any]:
    nll = s1_nll_by_patient(rates, validation, cuts)
    metrics = evaluate_rates(
        rates,
        train,
        validation,
        cuts,
        method=method,
        prediction_id=prediction_id,
        bootstrap_replicates=replicates,
        stages=(1,),
    )
    for row in metrics:
        row.update(stage="S1_pred", ct1_input=False)
    return {
        "validation_nll": {"S1_pred": float(nll.mean())},
        "metrics": metrics,
        "test_used": False,
    }


def nll_by_patient(
    rates: Tensor,
    batches: Sequence[WorldModelBatch],
    cuts: Sequence[float],
) -> Tensor:
    durations, events = _labels(batches)
    return torch.stack(
        [
            piecewise_exponential_nll(
                rates[:, stage],
                durations[:, stage],
                events[:, stage],
                tuple(cuts),
                reduction="none",
                zero_time_policy="allow",
            )
            for stage in range(2)
        ],
        dim=1,
    )


def matched_baseline_inputs(batches: Sequence[WorldModelBatch], stage: int) -> Tensor:
    rows = []
    for batch in batches:
        if not isinstance(batch, GeneratedTrainingBatch):
            raise DataContractError(code="RIDGE_BASELINE_MISSING", message="Bind baseline fields.")
        valid = batch.ct0.valid[..., None]
        mean_ct = batch.ct0.values.masked_fill(~valid, 0).sum(1) / valid.sum(1).clamp_min(1)
        features = [mean_ct, batch.baseline.ridge_features(), batch.s0_time[:, None] / 365.25]
        if stage == 1:
            features.extend(
                (
                    batch.treatment_actions.values.masked_fill(
                        ~batch.treatment_actions.valid[..., None], 0
                    ).flatten(1),
                    batch.s1_time[:, None] / 365.25,
                )
            )
        rows.append(torch.cat(features, dim=1))
    return torch.cat(rows).float()


@torch.inference_mode()
def future_diagnostics(
    model: GeneratedS1Model,
    train: Sequence[GeneratedTrainingBatch],
    validation: Sequence[GeneratedTrainingBatch],
) -> dict[str, Any]:
    model.eval()
    target = torch.cat([b.future_ct_target for b in validation]).float().flatten(1)
    mean = torch.cat([b.future_ct_target for b in train]).float().mean(0).flatten()
    predictions, variance = [], []
    for batch in validation:
        moved = batch.to(next(model.parameters()).device)
        inputs = moved.prediction_inputs(deterministic=True)
        inputs["target_time"] = moved.ct1_acquisition_time
        output = model(**inputs)
        predictions.append(output.future_ct.mean.cpu().float().flatten(1))
        variance.append(output.future_ct.log_std.cpu().float().mul(2).exp().flatten(1))
    predicted = torch.cat(predictions)
    target_variance = target.var(0, unbiased=False).mean()
    predicted_variance = predicted.var(0, unbiased=False).mean()
    return {
        "branch": "S1_pred",
        "feature_target": "frozen_CT1_spatial_mean_at_acquisition",
        "mse": float((predicted - target).square().mean()),
        "mae": float((predicted - target).abs().mean()),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(predicted, target).mean()),
        "train_mean_mse": float((mean - target).square().mean()),
        "train_mean_mae": float((mean - target).abs().mean()),
        "prediction_between_patient_variance": float(predicted_variance),
        "target_between_patient_variance": float(target_variance),
        "variance_ratio": float(predicted_variance / target_variance.clamp_min(1e-12)),
        "predicted_distribution_variance_mean": float(torch.cat(variance).mean()),
        "future_ct_objective": "huber_cosine",
        "future_ct_variance_supervised": False,
        "future_ct_uncertainty_status": "not_trained_or_calibrated",
        "mean_baseline_fit_split": "train",
        "test_used": False,
    }


def evaluate_generated_rates(
    rates: Tensor,
    train: Sequence[WorldModelBatch],
    validation: Sequence[WorldModelBatch],
    cuts: Sequence[float],
    *,
    method: str,
    prediction_id: str,
    replicates: int,
) -> dict[str, Any]:
    metrics = evaluate_rates(
        rates,
        train,
        validation,
        cuts,
        method=method,
        prediction_id=prediction_id,
        bootstrap_replicates=replicates,
    )
    for row in metrics:
        row["stage"] = "S0" if row["stage"] == "s0" else "S1_pred"
        row["ct1_input"] = False
    nll = nll_by_patient(rates, validation, cuts).mean(0)
    return {
        "validation_nll": {"S0": float(nll[0]), "S1_pred": float(nll[1])},
        "metrics": metrics,
        "test_used": False,
    }


def paired_comparisons(
    predictions: dict[str, Tensor],
    train: Sequence[WorldModelBatch],
    validation: Sequence[WorldModelBatch],
    cuts: Sequence[float],
    *,
    replicates: int,
) -> list[dict[str, Any]]:
    ids = tuple(p for b in validation for p in b.patient_ids)
    train_ids = tuple(p for b in train for p in b.patient_ids)
    times, events = _labels(validation)
    train_times, train_events = _labels(train)
    plan = make_patient_bootstrap_plan(ids, n_resamples=replicates, seed=17)
    main = predictions["history_generated"]
    results = []
    for stage in range(2):
        censor = CensoringDistribution.fit(
            EvaluationCohort(
                "paired-train",
                SplitRole.TRAIN,
                "os",
                train_ids,
                tuple(train_times[:, stage].tolist()),
                tuple(train_events[:, stage].tolist()),
            )
        )
        stage_times, stage_events = times[:, stage].numpy(), events[:, stage].numpy()
        for other, rates in predictions.items():
            if other == "history_generated":
                continue
            left_nll = nll_by_patient(main, validation, cuts)[:, stage].tolist()
            right_nll = nll_by_patient(rates, validation, cuts)[:, stage].tolist()
            result = paired_patient_bootstrap(
                ids,
                left_nll,
                right_nll,
                lambda values, indices: float(np.mean(values)),
                n_resamples=replicates,
                seed=17,
                plan=plan,
            )
            results.append(
                {
                    "stage": "S0" if stage == 0 else "S1_pred",
                    "metric": "nll",
                    "comparison": f"history_generated_minus_{other}",
                    "lower_is_better": True,
                    **asdict(result),
                }
            )
            for horizon in (1.0, 3.0):
                left = risk_probability(main[:, stage], torch.tensor([horizon]), tuple(cuts))[:, 0]
                right = risk_probability(rates[:, stage], torch.tensor([horizon]), tuple(cuts))[
                    :, 0
                ]
                for function in (cumulative_dynamic_auc, ipcw_concordance_index, ipcw_brier_score):

                    def statistic(
                        values: Any,
                        indices: Any,
                        metric: Any = function,
                        t: float = horizon,
                        c: Any = censor,
                        durations: Any = stage_times,
                        outcomes: Any = stage_events,
                    ) -> Any:
                        sampled = EvaluationCohort(
                            "paired-draw",
                            SplitRole.DEVELOPMENT,
                            "os",
                            tuple(f"draw-{i}" for i in range(len(indices))),
                            tuple(durations[indices].tolist()),
                            tuple(outcomes[indices].tolist()),
                        )
                        return metric(sampled, values.tolist(), t, c)

                    result = paired_patient_bootstrap(
                        ids,
                        left.tolist(),
                        right.tolist(),
                        statistic,
                        n_resamples=replicates,
                        seed=17,
                        plan=plan,
                    )
                    results.append(
                        {
                            "stage": "S0" if stage == 0 else "S1_pred",
                            "metric": function.__name__,
                            "horizon": horizon,
                            "comparison": f"history_generated_minus_{other}",
                            "lower_is_better": function is ipcw_brier_score,
                            **asdict(result),
                        }
                    )
            # IBS needs whole curves. Encode row selectors as the paired scalar values.
            grid = torch.linspace(0.25, 3.0, 12)
            curves = np.concatenate(
                [
                    risk_probability(main[:, stage], grid, tuple(cuts)).numpy(),
                    risk_probability(rates[:, stage], grid, tuple(cuts)).numpy(),
                ]
            )

            def ibs_statistic(
                values: Any,
                indices: Any,
                probabilities: Any = curves,
                c: Any = censor,
                durations: Any = stage_times,
                outcomes: Any = stage_events,
                horizons: Any = grid,
            ) -> Any:
                sampled = EvaluationCohort(
                    "paired-ibs",
                    SplitRole.DEVELOPMENT,
                    "os",
                    tuple(f"draw-{i}" for i in range(len(indices))),
                    tuple(durations[indices].tolist()),
                    tuple(outcomes[indices].tolist()),
                )
                return integrated_brier_score(
                    sampled, probabilities[values.astype(int)].tolist(), horizons.tolist(), c
                )

            result = paired_patient_bootstrap(
                ids,
                list(range(len(ids))),
                list(range(len(ids), 2 * len(ids))),
                ibs_statistic,
                n_resamples=replicates,
                seed=17,
                plan=plan,
            )
            results.append(
                {
                    "stage": "S0" if stage == 0 else "S1_pred",
                    "metric": "integrated_brier_score",
                    "comparison": f"history_generated_minus_{other}",
                    "lower_is_better": True,
                    **asdict(result),
                }
            )
    for row in results:
        row["interval_scope"] = (
            "patient_sampling_conditional_on_selected_models_not_selection_adjusted"
        )
        annotate_paired_interval_support(row)
    return results


def annotate_paired_interval_support(row: dict[str, Any]) -> None:
    minimum = max(20, int(0.8 * row["n_resamples"]))
    row["minimum_valid_bootstrap_replicates"] = minimum
    if row["status"] != "ok":
        row["confidence_interval_status"] = "point_metric_not_estimable"
    elif row["n_estimable_resamples"] < minimum:
        row["confidence_lower"] = None
        row["confidence_upper"] = None
        row["confidence_interval_status"] = "insufficient_valid_resamples"
    else:
        row["confidence_interval_status"] = "ok"
