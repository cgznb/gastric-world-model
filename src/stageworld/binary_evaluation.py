"""Binary recorded-endpoint metrics with one row per outer-fold patient."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore[import-untyped]
from torch import Tensor

from stageworld.binary_endpoints import ENDPOINTS, BinaryEndpointBatch, BinaryEndpointModel
from stageworld.errors import DataContractError


@torch.inference_mode()
def predict_binary_logits(
    model: BinaryEndpointModel, batches: Sequence[BinaryEndpointBatch]
) -> Tensor:
    model.eval()
    device = next(model.parameters()).device
    logits = torch.cat(
        [
            model(**batch.to(device).prediction_inputs(deterministic=True)).logits.cpu()
            for batch in batches
        ]
    ).float()
    if logits.shape != (sum(b.batch_size for b in batches), 2) or not torch.isfinite(logits).all():
        raise DataContractError(code="BINARY_PREDICTIONS", message="Invalid endpoint logits.")
    return logits


def binary_bce(logits: Tensor, batches: Sequence[BinaryEndpointBatch]) -> dict[str, float]:
    labels = torch.cat([b.endpoint_labels for b in batches])
    valid = torch.cat([b.endpoint_valid for b in batches])
    if logits.shape != labels.shape:
        raise DataContractError(code="BINARY_SCORE_SHAPE", message="Predictions and labels differ.")
    result = {}
    for i, name in enumerate(ENDPOINTS):
        selected = valid[:, i]
        if not selected.any():
            raise DataContractError(
                code="BINARY_SCORE_SUPPORT", message="An endpoint has no labels."
            )
        result[name] = float(
            torch.nn.functional.binary_cross_entropy_with_logits(
                logits[selected, i], labels[selected, i].float()
            )
        )
    return result


def _point(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | None]:
    if not len(labels):
        return dict.fromkeys(
            (
                "bce",
                "auroc",
                "auprc",
                "brier",
                "accuracy",
                "balanced_accuracy",
                "sensitivity",
                "specificity",
            )
        )
    predicted = probabilities >= 0.5
    positive, negative = labels == 1, labels == 0
    sensitivity = float(predicted[positive].mean()) if positive.any() else None
    specificity = float((~predicted[negative]).mean()) if negative.any() else None
    both = positive.any() and negative.any()
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return {
        "bce": float(-(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)).mean()),
        "auroc": float(roc_auc_score(labels, probabilities)) if both else None,
        "auprc": float(average_precision_score(labels, probabilities)) if both else None,
        "brier": float(((probabilities - labels) ** 2).mean()),
        "accuracy": float((predicted == labels).mean()),
        "balanced_accuracy": (sensitivity + specificity) / 2
        if sensitivity is not None and specificity is not None
        else None,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def evaluate_binary(
    probabilities: Tensor, labels: Tensor, valid: Tensor, *, replicates: int = 1000, seed: int = 17
) -> dict[str, Any]:
    if probabilities.shape != labels.shape or valid.shape != labels.shape or labels.shape[1] != 2:
        raise DataContractError(code="BINARY_METRIC_SHAPE", message="Bind matching binary arrays.")
    p, y, mask = probabilities.double().numpy(), labels.numpy(), valid.numpy()
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise DataContractError(code="BINARY_PROBABILITY", message="Use finite probabilities.")
    report: dict[str, Any] = {}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(y), size=(replicates, len(y)))
    for i, name in enumerate(ENDPOINTS):
        selected = mask[:, i]
        point = _point(y[selected, i], p[selected, i])
        samples: dict[str, list[float]] = {k: [] for k in point}
        for draw in draws:
            rows = draw[mask[draw, i]]
            for key, value in _point(y[rows, i], p[rows, i]).items():
                if value is not None:
                    samples[key].append(value)
        report[name] = {
            "labelled_patients": int(selected.sum()),
            "positive": int(y[selected, i].sum()),
            "negative": int(selected.sum() - y[selected, i].sum()),
            "missing": int((~selected).sum()),
            "threshold": 0.5,
            "mean_probability": float(p[selected, i].mean()) if selected.any() else None,
            "metrics": {
                key: {
                    "estimate": value,
                    "status": "ok" if value is not None else "not_estimable",
                    "confidence_interval_95": np.quantile(samples[key], [0.025, 0.975]).tolist()
                    if value is not None and len(samples[key]) >= max(20, int(0.8 * replicates))
                    else None,
                    "bootstrap_valid_replicates": len(samples[key]),
                    "bootstrap_replicates": replicates,
                }
                for key, value in point.items()
            },
        }
    return {
        "endpoints": report,
        "time_window": None,
        "recurrence_interpretation": (
            "recorded_status_with_heterogeneous_followup_not_fixed_window_incidence"
        ),
        "bootstrap_scope": "patient_resampling_conditional_on_fitted_models_no_retraining",
        "test_used": False,
    }


def prevalence_baseline(train: Sequence[BinaryEndpointBatch], count: int) -> Tensor:
    y = torch.cat([b.endpoint_labels for b in train])
    mask = torch.cat([b.endpoint_valid for b in train])
    return torch.stack([y[mask[:, i], i].mean() for i in range(2)]).repeat(count, 1)
