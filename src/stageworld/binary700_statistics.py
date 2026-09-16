"""Fold-local statistical fitting, monotone calibration and operating rules."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss

from stageworld.binary700_spec import Candidate


def positive_weights(labels: torch.Tensor, valid: torch.Tensor, mode: str) -> torch.Tensor:
    result = []
    for i in range(2):
        y = labels[valid[:, i], i]
        positive, negative = int(y.sum()), int(len(y) - y.sum())
        if not positive or not negative:
            raise ValueError("Both classes must occur in each training endpoint")
        ratio = negative / positive
        if mode == "none":
            ratio = 1.0
        elif mode == "sqrt":
            ratio **= 0.5
        elif mode != "balanced":
            raise ValueError("Unknown class-weight rule")
        result.append(ratio)
    return torch.tensor(result, dtype=torch.float32)


def statistical_predict(models: list, x: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            model.decision_function(x)
            if isinstance(model, LogisticRegression)
            else logit(np.clip(model.predict_proba(x)[:, 1], 1e-7, 1 - 1e-7))
            for model in models
        ],
        axis=1,
    )


def fit_statistical(
    candidate: Candidate,
    x: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
    train: torch.Tensor,
    inner: torch.Tensor | None,
    selected: list[float] | None = None,
) -> tuple[list, dict[str, Any]]:
    array, y, mask = x.double().numpy(), labels.numpy(), valid.numpy()
    weights = positive_weights(labels[train], valid[train], candidate.weight).numpy()
    models, choices, traces = [], [], []
    for i in range(2):
        rows = train.numpy()[mask[train.numpy(), i]]
        choices_to_fit = (
            [selected[i]]
            if selected is not None
            else ([0.01, 0.1, 1.0, 10.0] if candidate.family == "logistic" else [7.0])
        )
        candidates = []
        for choice in choices_to_fit:
            model = (
                LogisticRegression(C=choice, max_iter=2000, solver="lbfgs", random_state=17)
                if candidate.family == "logistic"
                else HistGradientBoostingClassifier(
                    max_iter=100,
                    learning_rate=0.05,
                    max_leaf_nodes=int(choice),
                    min_samples_leaf=20,
                    l2_regularization=10,
                    early_stopping=False,
                    random_state=17,
                )
            )
            sample_weight = np.where(y[rows, i] == 1, weights[i], 1.0).astype(np.float64)
            sample_weight /= sample_weight.mean()
            model.fit(array[rows], y[rows, i], sample_weight=sample_weight)
            if isinstance(model, LogisticRegression) and model.n_iter_.max() >= 2000:
                raise ValueError("Logistic optimizer did not converge")
            if inner is not None:
                held = inner.numpy()[mask[inner.numpy(), i]]
                p = model.predict_proba(array[held])[:, 1]
                score = (-average_precision_score(y[held, i], p), log_loss(y[held, i], p), choice)
            else:
                score = (0.0, 0.0, choice)
            candidates.append((score, model))
        ranking, model = min(candidates, key=lambda item: item[0])
        models.append(model)
        choices.append(ranking[2])
        traces.append([{"selection_score": list(score)} for score, _ in candidates])
    return models, {"choices": choices, "positive_weights": weights.tolist(), "trace": traces}


def logistic_anchor(models: list) -> tuple[torch.Tensor, torch.Tensor]:
    if not all(isinstance(m, LogisticRegression) for m in models):
        raise ValueError("The frozen anchor must be an unweighted clinical logistic model")
    return (
        torch.from_numpy(np.concatenate([m.coef_ for m in models], axis=0)).float(),
        torch.from_numpy(np.concatenate([m.intercept_ for m in models])).float(),
    )


def choose_thresholds(probability: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Inner thresholds need both classes")
    positive, negative = labels == 1, labels == 0
    candidates = np.unique(np.r_[0.0, 0.5, 1.0, probability])
    rows = []
    for threshold in candidates:
        decision = probability >= threshold
        sensitivity = float(decision[positive].mean())
        specificity = float((~decision[negative]).mean())
        rows.append((float(threshold), sensitivity, specificity))
    balanced = max(rows, key=lambda r: ((r[1] + r[2]) / 2, -abs(r[0] - 0.5), r[0]))
    sensitive = max((r for r in rows if r[1] >= 0.8), key=lambda r: (r[2], r[0]))
    return balanced[0], sensitive[0]


def fit_operating(logits: np.ndarray, labels: torch.Tensor, valid: torch.Tensor) -> dict:
    logits = np.asarray(logits, dtype=np.float64)
    result: dict[str, Any] = {"balanced": [], "sensitivity_0.8": [], "calibration": []}
    for i in range(2):
        selected = valid[:, i].numpy()
        y, z = labels[selected, i].numpy(), logits[selected, i]
        balanced, sensitivity = choose_thresholds(expit(z), y)
        result["balanced"].append(balanced)
        result["sensitivity_0.8"].append(sensitivity)
        calibrator = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        calibrator.fit(z[:, None], y)
        slope, intercept = float(calibrator.coef_[0, 0]), float(calibrator.intercept_[0])
        fallback = slope < 0
        if fallback:
            slope, intercept = 0.0, float(logit(y.mean()))
        result["calibration"].append(
            {"slope": slope, "intercept": intercept, "constant_fallback": fallback}
        )
    result["fit_source"] = "own_inner_validation_predictions_only"
    return result


def apply_operating(logits: np.ndarray, rules: dict) -> dict[str, torch.Tensor]:
    logits = np.asarray(logits, dtype=np.float64)
    p = torch.from_numpy(expit(logits)).double()
    calibrated = torch.from_numpy(
        np.stack(
            [
                expit(
                    logits[:, i] * rules["calibration"][i]["slope"]
                    + rules["calibration"][i]["intercept"]
                )
                for i in range(2)
            ],
            axis=1,
        )
    ).double()
    return {
        "probabilities": p,
        "calibrated": calibrated,
        "raw_0.5": p >= 0.5,
        "inner_balanced": p >= torch.tensor(rules["balanced"], dtype=torch.float64)[None],
        "inner_sensitivity_0.8": p
        >= torch.tensor(rules["sensitivity_0.8"], dtype=torch.float64)[None],
        "calibrated_0.5": calibrated >= 0.5,
    }
