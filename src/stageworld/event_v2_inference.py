"""Input-only bundles for the prespecified seed-specific holdout experiment."""

from pathlib import Path

import torch

from stageworld.artifacts import atomic_write_private_json
from stageworld.event_data import encode_inputs
from stageworld.event_models import EventInputs
from stageworld.event_spec import PRESENT
from stageworld.event_v2_spec import TASK
from stageworld.event_v2_training import Model, build_model
from stageworld.synthetic_workflow import _atomic_torch_save


def export_bundle(
    root: Path, snapshot: dict, model: Model, family: str, checkpoint_id: str
) -> None:
    contract = {
        "task": TASK,
        "family": family,
        "dimensions": model.dimensions(),
        "checkpoint_id": checkpoint_id,
        "ct1_required": False,
        "outcomes_required": False,
        "time_required": False,
        "cycles_used": False,
        "threshold": 0.5,
        "calibrated": False,
        "endpoint_order": ["pcr", "recurrence"],
        "event_order": ["neoadjuvant", "surgery", "postoperative"],
        "aggregation": "mean_member_probabilities",
        "uncertainty": "uncalibrated_member_disagreement",
    }
    _atomic_torch_save(
        root / "inference.pt",
        {
            **contract,
            "inputs": {k: v for k, v in snapshot.items() if k != "fit_ids"},
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        },
    )
    atomic_write_private_json(root / "contract.json", contract)


@torch.inference_mode()
def predict_bundle(
    root: Path,
    clinical: list[dict],
    treatments: list[dict],
    events: torch.Tensor,
    ct0: torch.Tensor,
) -> dict[str, torch.Tensor]:
    bundle = torch.load(root / "inference.pt", weights_only=True, map_location="cpu")
    if (
        bundle["task"] != TASK
        or bundle["ct1_required"]
        or bundle["outcomes_required"]
        or bundle["time_required"]
        or bundle["cycles_used"]
    ):
        raise ValueError("Unsupported event holdout inference contract")
    if not clinical or len(clinical) != len(treatments) or len(clinical) != len(ct0):
        raise ValueError("Require matched nonempty input rows")
    x = encode_inputs(clinical, treatments, bundle["inputs"])
    inputs = EventInputs(x, ct0.detach().float().cpu(), events.cpu())
    inputs.validate(bundle["dimensions"]["image_dim"])
    torch.backends.mha.set_fastpath_enabled(False)
    model = build_model(bundle["family"], bundle["dimensions"]).eval()
    model.load_state_dict(bundle["model_state"], strict=True)
    pieces: dict[str, list[torch.Tensor]] = {
        "ct1": [],
        "member_logits": [],
        "last_stage": [],
        "incomplete_history": [],
    }
    for rows in torch.arange(len(x)).split(32):
        output = model(EventInputs(inputs.x[rows], inputs.ct0[rows], inputs.events[rows]))
        logits = torch.stack((output.pcr_logit, output.recurrence_logit), 1)
        values = {
            "ct1": output.ct1,
            "member_logits": getattr(output, "member_logits", logits[..., None]),
            "last_stage": output.last_stage,
            "incomplete_history": output.incomplete_history,
        }
        for key, value in values.items():
            pieces[key].append(value.cpu())
    result = {key: torch.cat(values) for key, values in pieces.items()}
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite holdout inference output")
    member_probability = result["member_logits"].double().sigmoid()
    result["probabilities"] = member_probability.mean(-1)
    result["member_disagreement"] = member_probability.var(-1, unbiased=False)
    result["decisions"] = result["probabilities"] >= 0.5
    result["pcr_applicable"] = inputs.events[:, 1] == PRESENT
    return result
