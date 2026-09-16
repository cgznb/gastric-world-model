"""Train-partition-only sklearn baselines and inner-selected operating thresholds."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
from torch import Tensor

from stageworld.binary_endpoints import ENDPOINTS, BinaryEndpointBatch
from stageworld.binary_evaluation import _point
from stageworld.data.baseline_clinical import BaselineClinical
from stageworld.encoders.base import ObservationTokens
from stageworld.model.types import ActionTokens


def targets(batches: Sequence[BinaryEndpointBatch]) -> tuple[Tensor, Tensor]:
    return torch.cat([b.endpoint_labels for b in batches]), torch.cat(
        [b.endpoint_valid for b in batches]
    )


def query_features(
    ct: ObservationTokens,
    baseline: BaselineClinical,
    actions: ActionTokens,
    interval: Tensor,
    *,
    include_ct: bool,
) -> Tensor:
    tabular = torch.cat(
        (baseline.ridge_features(), actions.values.flatten(1), torch.log1p(interval[:, None] / 30)),
        -1,
    )
    if not include_ct:
        return tabular
    count = ct.valid.sum(1, keepdim=True).clamp_min(1)
    mean = (ct.values * ct.valid[..., None]).sum(1) / count
    variance = ((ct.values - mean[:, None]).square() * ct.valid[..., None]).sum(1) / count
    return torch.cat((mean, variance.sqrt(), tabular), -1)


def feature_matrix(batches: Sequence[BinaryEndpointBatch], mode: str) -> tuple[np.ndarray, int]:
    if mode not in ("clinical", "ct0", "ct1_diagnostic"):
        raise ValueError("Unknown baseline feature access")
    rows = []
    ct_width = 0
    for batch in batches:
        ct = batch.ct1 if mode == "ct1_diagnostic" else batch.ct0
        value = query_features(
            ct,
            batch.baseline,
            batch.treatment_actions,
            batch.s1_time - batch.s0_time,
            include_ct=mode != "clinical",
        )
        ct_width = 0 if mode == "clinical" else 2 * ct.values.shape[-1]
        rows.append(value.detach().cpu())
    result = torch.cat(rows).double().numpy()
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite baseline features")
    return result, ct_width


def fit_transform(x: np.ndarray, ct_width: int) -> tuple[np.ndarray, dict[str, Any]]:
    scaler = StandardScaler().fit(x)
    standardized = scaler.transform(x)
    state: dict[str, Any] = {
        "mean": torch.from_numpy(scaler.mean_),
        "scale": torch.from_numpy(scaler.scale_),
        "ct_width": ct_width,
        "training_count": len(x),
    }
    if ct_width:
        pca = PCA(
            n_components=min(32, len(x) - 1, ct_width), svd_solver="randomized", random_state=17
        )
        reduced = pca.fit_transform(standardized[:, :ct_width])
        state.update(
            {
                "pca_mean": torch.from_numpy(pca.mean_),
                "pca_components": torch.from_numpy(pca.components_),
                "pca_explained_variance_ratio": torch.from_numpy(pca.explained_variance_ratio_),
            }
        )
        standardized = np.concatenate((reduced, standardized[:, ct_width:]), -1)
    return standardized, state


def transform_features(x: np.ndarray, state: dict[str, Any]) -> np.ndarray:
    transformed = (x - state["mean"].numpy()) / state["scale"].numpy()
    width = state["ct_width"]
    if width:
        reduced = (transformed[:, :width] - state["pca_mean"].numpy()) @ state[
            "pca_components"
        ].numpy().T
        transformed = np.concatenate((reduced, transformed[:, width:]), -1)
    return transformed


def baseline_predict(payload: dict[str, Any], batches: Sequence[BinaryEndpointBatch]) -> Tensor:
    x, width = feature_matrix(batches, payload["mode"])
    if width != payload["transform"]["ct_width"]:
        raise ValueError("Baseline feature schema differs")
    x = transform_features(x, payload["transform"])
    logits = np.concatenate(
        [
            x @ head["coefficient"].numpy().T + head["intercept"].numpy()
            for head in payload["heads"]
        ],
        -1,
    )
    return torch.from_numpy(logits).sigmoid()


def fit_baseline(
    train: Sequence[BinaryEndpointBatch],
    inner: Sequence[BinaryEndpointBatch],
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if {p for b in train for p in b.patient_ids} & {p for b in inner for p in b.patient_ids}:
        raise ValueError("Baseline train/inner overlap")
    x, width = feature_matrix(train, mode)
    _, transform = fit_transform(x, width)
    # Use the exported transform for both fitting and replay, including randomized PCA.
    x = transform_features(x, transform)
    val, _ = feature_matrix(inner, mode)
    val = transform_features(val, transform)
    y, valid = targets(train)
    yv, validv = targets(inner)
    heads, trace = [], {}
    for index, endpoint in enumerate(ENDPOINTS):
        mask, maskv = valid[:, index].numpy(), validv[:, index].numpy()
        label, labelv = y[mask, index].numpy(), yv[maskv, index].numpy()
        if len(np.unique(label)) != 2 or not maskv.any():
            raise ValueError("Logistic fitting requires supported training/inner labels")
        candidates = []
        for strength in (0.01, 0.1, 1.0):
            estimator = LogisticRegression(
                C=strength, solver="lbfgs", max_iter=2000, random_state=17
            )
            estimator.fit(x[mask], label)
            if estimator.n_iter_.max() >= 2000:
                raise ValueError("Logistic regression did not converge")
            score = _point(labelv, estimator.predict_proba(val[maskv])[:, 1])
            candidates.append((score["bce"], strength, estimator, score))
        selected = min(candidates, key=lambda candidate: (candidate[0], candidate[1]))
        estimator = selected[2]
        heads.append(
            {
                "coefficient": torch.from_numpy(estimator.coef_),
                "intercept": torch.from_numpy(estimator.intercept_),
                "C": selected[1],
                "training_labels": int(mask.sum()),
            }
        )
        trace[endpoint] = {
            "selected_C": selected[1],
            "inner": selected[3],
            "candidates": [{"C": c, "inner": report} for _, c, _, report in candidates],
        }
    payload = {
        "schema_version": "ct6-logistic-baseline-v1",
        "mode": mode,
        "transform": transform,
        "heads": heads,
        "fit_patient_ids": [p for b in train for p in b.patient_ids],
    }
    return payload, trace


def select_thresholds(probabilities: Tensor, labels: Tensor, valid: Tensor) -> list[float]:
    result = []
    for index in range(2):
        p, y = probabilities[valid[:, index], index].numpy(), labels[valid[:, index], index].numpy()
        if len(np.unique(y)) != 2:
            result.append(0.5)
            continue
        candidates = np.unique(np.concatenate(([0, 0.5, 1], p)))

        def score(
            threshold: float, p: np.ndarray = p, y: np.ndarray = y
        ) -> tuple[float, float, float]:
            prediction = p >= threshold
            balanced = (prediction[y == 1].mean() + (~prediction[y == 0]).mean()) / 2
            return float(balanced), -abs(threshold - 0.5), -threshold

        result.append(float(max(candidates, key=score)))
    return result


def operating_metrics(p: Tensor, y: Tensor, valid: Tensor, thresholds: Tensor) -> dict[str, Any]:
    if thresholds.shape != p.shape:
        raise ValueError("Thresholds must be bound per patient and endpoint")
    result = {}
    for index, name in enumerate(ENDPOINTS):
        mask = valid[:, index]
        label = y[mask, index]
        classified = p[mask, index] >= thresholds[mask, index]
        positive, negative = label == 1, label == 0
        sensitivity = float(classified[positive].float().mean()) if positive.any() else None
        specificity = float((~classified[negative]).float().mean()) if negative.any() else None
        result[name] = {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "balanced_accuracy": None
            if sensitivity is None or specificity is None
            else (sensitivity + specificity) / 2,
            "accuracy": float((classified == label).float().mean()) if len(label) else None,
            "threshold_source": "own_inner_validation_only",
        }
    return result
