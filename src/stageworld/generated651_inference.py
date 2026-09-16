"""Standalone complete-case V2 scores with fixed residual scale and threshold."""

from pathlib import Path

import torch

from stageworld.data.baseline_clinical import encode_baseline
from stageworld.data.treatment_compact import encode_compact
from stageworld.generated651_spec import RESIDUAL_SCALE, THRESHOLD
from stageworld.generated700_models import AnchoredClassifier
from stageworld.synthetic_workflow import _atomic_torch_save


def export_bundle(root: Path, snapshot: dict, model: AnchoredClassifier) -> None:
    if not torch.equal(model.residual_scale.cpu(), torch.full((2,), RESIDUAL_SCALE)):
        raise ValueError("Require the prespecified fixed neural residual scale")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    _atomic_torch_save(
        root / "inference.pt",
        {
            "schema": "generated651-inference-v1",
            "inputs": {key: snapshot[key] for key in ("clinical", "support", "mean", "scale")},
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "family": "generated",
            "tabular_dim": len(snapshot["mean"]),
            "image_dim": model.world.input_mean.numel(),
            "ct1_required": False,
            "outcomes_required": False,
            "attention_fastpath": False,
            "clinical_fields": ["sex", "age", "bmi", "ct_stage", "cn_stage", "cm_stage"],
            "cycles_used": False,
            "threshold": THRESHOLD,
            "residual_scale": RESIDUAL_SCALE,
            "calibrated": False,
            "model_selection": "reported_validation_fold",
        },
    )


@torch.inference_mode()
def predict_bundle(
    root: Path,
    clinical: list[dict],
    treatments: list[dict],
    interval: torch.Tensor,
    ct0: torch.Tensor,
) -> dict[str, torch.Tensor]:
    bundle = torch.load(root / "inference.pt", weights_only=True, map_location="cpu")
    if (
        bundle["schema"] != "generated651-inference-v1"
        or bundle["ct1_required"]
        or bundle["outcomes_required"]
        or bundle["attention_fastpath"] is not False
        or bundle["threshold"] != THRESHOLD
        or bundle["residual_scale"] != RESIDUAL_SCALE
    ):
        raise ValueError("Unsupported standalone inference contract")
    torch.backends.mha.set_fastpath_enabled(False)
    count = len(clinical)
    if (
        count == 0
        or len(treatments) != count
        or interval.shape != (count,)
        or ct0.shape != (count, 27, bundle["image_dim"])
        or not torch.isfinite(interval).all()
        or not (interval > 0).all()
        or not torch.isfinite(ct0).all()
    ):
        raise ValueError("Bind finite complete CT0, clinical, treatment and positive target time")
    snapshot = bundle["inputs"]
    baseline = encode_baseline(clinical, snapshot["clinical"])
    action, _ = encode_compact(treatments, snapshot["support"])
    raw = torch.cat(
        (
            baseline.ridge_features(),
            action.flatten(1),
            torch.log1p(interval[:, None].float() / 30),
        ),
        1,
    ).double()
    x = ((raw - snapshot["mean"]) / snapshot["scale"]).float()
    if not torch.isfinite(x).all():
        raise ValueError("Nonfinite encoded query")
    state = bundle["model_state"]
    model = AnchoredClassifier(
        bundle["tabular_dim"],
        "generated",
        state["anchor_weight"],
        state["anchor_bias"],
        image_dim=bundle["image_dim"],
    ).eval()
    model.load_state_dict(state, strict=True)
    if not torch.equal(model.residual_scale, torch.full((2,), RESIDUAL_SCALE)):
        raise ValueError("Inference state changed the fixed residual scale")
    logits, features = [], []
    for rows in torch.arange(count).split(32):
        z, future = model.forward_with_features(
            x[rows],
            ct0[rows].float(),
            torch.ones(len(rows), dtype=torch.bool),
        )
        logits.append(z)
        features.append(future)
    scores = torch.cat(logits).double()
    probabilities = scores.sigmoid()
    return {
        "logits": scores,
        "probabilities": probabilities,
        "decisions": probabilities >= THRESHOLD,
        "ct_features": torch.cat(features),
    }
