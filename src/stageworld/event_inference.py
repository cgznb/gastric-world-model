"""Portable event-only prediction; no target or clinical-source file access."""

from pathlib import Path

import torch

from stageworld.artifacts import atomic_write_private_json
from stageworld.event_data import encode_inputs
from stageworld.event_models import EventInputs, EventModel
from stageworld.event_spec import PRESENT, TASK
from stageworld.synthetic_workflow import _atomic_torch_save


def export_bundle(root: Path, snapshot: dict, model: EventModel) -> None:
    contract = {
        "task": TASK,
        "dimensions": model.dimensions(),
        "ct1_required": False,
        "outcomes_required": False,
        "time_required": False,
        "clinical_fields": ["sex", "age", "bmi", "ct_stage", "cn_stage", "cm_stage"],
        "event_order": ["neoadjuvant", "surgery", "postoperative"],
        "event_codes": ["absent", "present", "unknown", "conflict"],
        "cycles_used": False,
        "threshold": 0.5,
        "calibrated": False,
        "primary_endpoint": "recorded_recurrence_metastasis_status",
        "endpoint_order": ["pcr", "recurrence"],
        "pcr_is_intermediate_auxiliary": True,
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
        raise ValueError("Unsupported event inference contract")
    if not clinical or len(clinical) != len(treatments) or len(clinical) != len(ct0):
        raise ValueError("Require matched nonempty input rows")
    torch.backends.mha.set_fastpath_enabled(False)
    x = encode_inputs(clinical, treatments, bundle["inputs"])
    inputs = EventInputs(x, ct0.float().cpu(), events.cpu())
    inputs.validate(bundle["dimensions"]["image_dim"])
    model = EventModel(**bundle["dimensions"]).eval()
    model.load_state_dict(bundle["model_state"], strict=True)
    pieces: dict[str, list[torch.Tensor]] = {
        "logits": [],
        "ct1": [],
        "last_stage": [],
        "incomplete_history": [],
    }
    for rows in torch.arange(len(x)).split(32):
        output = model(EventInputs(inputs.x[rows], inputs.ct0[rows], inputs.events[rows]))
        pieces["logits"].append(torch.stack((output.pcr_logit, output.recurrence_logit), 1))
        pieces["ct1"].append(output.ct1)
        pieces["last_stage"].append(output.last_stage)
        pieces["incomplete_history"].append(output.incomplete_history)
    result = {key: torch.cat(value) for key, value in pieces.items()}
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite inference output")
    result["probabilities"] = result["logits"].double().sigmoid()
    result["decisions"] = result["probabilities"] >= 0.5
    result["recurrence_probability"] = result["probabilities"][:, 1]
    result["pcr_applicable"] = inputs.events[:, 1] == PRESENT
    return result
