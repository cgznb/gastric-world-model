"""Standalone fold bundles: CT0/clinical/treatment inputs, no CT1 or endpoints."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from stageworld.binary700_models import AnchoredClassifier
from stageworld.binary700_statistics import apply_operating, statistical_predict
from stageworld.data.baseline_clinical import encode_baseline
from stageworld.data.treatment_compact import encode_compact
from stageworld.synthetic_workflow import _atomic_torch_save


def export_bundle(
    root: Path,
    snapshot: dict,
    rules: dict,
    family: str,
    *,
    model: AnchoredClassifier | None = None,
    models: list | None = None,
    image_dim: int = 768,
) -> None:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload: dict[str, Any] = {
        "schema": "binary700-inference-v1",
        "inputs": snapshot,
        "rules": rules,
        "family": family,
        "ct1_required": False,
        "outcomes_required": False,
        "clinical_fields": ["sex", "age", "bmi", "ct_stage", "cn_stage", "cm_stage"],
        "cycles_used": False,
        "tabular_dim": len(snapshot["mean"]),
        "image_dim": image_dim,
    }
    if model is not None:
        payload["model_state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    else:
        if models is None:
            raise ValueError("Bind fitted statistical estimators")
        joblib.dump(models, root / "estimators.joblib")
    _atomic_torch_save(root / "inference.pt", payload)


@torch.inference_mode()
def predict_bundle(
    root: Path,
    clinical: list[dict],
    treatments: list[dict],
    interval: torch.Tensor,
    ct0: torch.Tensor | None = None,
    ct0_valid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    bundle = torch.load(root / "inference.pt", weights_only=True, map_location="cpu")
    if (
        bundle["schema"] != "binary700-inference-v1"
        or bundle["ct1_required"]
        or bundle["outcomes_required"]
    ):
        raise ValueError("Unsupported inference bundle")
    snapshot, count = bundle["inputs"], len(clinical)
    if count != len(treatments) or interval.shape != (count,) or not (interval > 0).all():
        raise ValueError("Invalid query dimensions or target times")
    baseline = encode_baseline(clinical, snapshot["clinical"])
    action, _ = encode_compact(treatments, snapshot["support"])
    raw = torch.cat(
        (baseline.ridge_features(), action.flatten(1), torch.log1p(interval[:, None].float() / 30)),
        1,
    ).double()
    x = ((raw - snapshot["mean"]) / snapshot["scale"]).float()
    if not torch.isfinite(x).all():
        raise ValueError("Nonfinite query features")
    family = bundle["family"]
    if family in ("logistic", "histgb"):
        logits = statistical_predict(joblib.load(root / "estimators.joblib"), x.double().numpy())
    else:
        if ct0 is None:
            ct0 = torch.zeros(count, 27, bundle["image_dim"])
            ct0_valid = torch.zeros(count, dtype=torch.bool)
        if (
            ct0_valid is None
            or ct0.shape != (count, 27, bundle["image_dim"])
            or ct0_valid.shape != (count,)
        ):
            raise ValueError("Bind canonical CT0 tokens and their explicit availability")
        state = bundle["model_state"]
        model = AnchoredClassifier(
            bundle["tabular_dim"],
            family,
            state["anchor_weight"],
            state["anchor_bias"],
            image_dim=bundle["image_dim"],
        )
        model.load_state_dict(state, strict=True)
        model.eval()
        logits = model(x, ct0.float(), ct0_valid.bool()).numpy()
    return {
        "logits": torch.from_numpy(np.asarray(logits)),
        **apply_operating(logits, bundle["rules"]),
    }
