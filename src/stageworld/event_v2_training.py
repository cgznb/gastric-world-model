"""Train/validation-only fitting with reproducible epoch recovery for both architectures."""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from sklearn.metrics import average_precision_score

from stageworld.artifacts import atomic_write_private_json, new_artifact_id, read_json
from stageworld.ct6_training import EarlyStopState
from stageworld.event_data import EventPool
from stageworld.event_models import EventModel, masked_bce
from stageworld.event_training import has_supervision, recurrence_weight
from stageworld.event_v2_models import EventV2Model
from stageworld.event_v2_spec import FAMILIES, TASK, model_dimensions, specification
from stageworld.generated700_models import feature_set_loss
from stageworld.generated_workflow import _seed
from stageworld.synthetic_workflow import _atomic_torch_save
from stageworld.training import capture_rng_state, restore_rng_state

Model = EventModel | EventV2Model


def build_model(family: str, dimensions: dict) -> Model:
    if family not in FAMILIES:
        raise ValueError("Unknown event holdout model family")
    return EventModel(**dimensions) if family == "event_v1" else EventV2Model(**dimensions)


def objective(output, targets: dict, phase: str, positive_weight: float) -> tuple:
    if phase not in ("pretrain", "joint"):
        raise ValueError("Unknown holdout training phase")
    members = getattr(output, "member_logits", None)
    if members is None:
        members = torch.stack((output.pcr_logit, output.recurrence_logit), 1)[..., None]
    ct = feature_set_loss(output.ct1, targets["ct1"], targets["ct_valid"])
    pcr = torch.stack(
        [
            masked_bce(member, targets["labels"][:, 0], targets["valid"][:, 0])
            for member in members[:, 0].unbind(1)
        ]
    ).mean()
    recurrence = output.recurrence_logit.new_zeros(())
    if phase == "joint":
        recurrence = torch.stack(
            [
                masked_bce(member, targets["labels"][:, 1], targets["valid"][:, 1], positive_weight)
                for member in members[:, 1].unbind(1)
            ]
        ).mean()
    loss = ct + 0.5 * pcr if phase == "pretrain" else recurrence + 0.5 * pcr + 0.1 * ct
    return loss, {"ct_loss": ct, "pcr_bce": pcr, "recurrence_weighted_bce": recurrence}


@torch.inference_mode()
def infer(model: Model, x: torch.Tensor, pool: EventPool, rows: torch.Tensor) -> dict:
    if not len(rows):
        raise ValueError("Cannot evaluate an empty partition")
    model.eval()
    device = next(model.parameters()).device
    pieces: dict[str, list[torch.Tensor]] = {
        "ct1": [],
        "logits": [],
        "member_logits": [],
        "last_stage": [],
        "incomplete_history": [],
    }
    for batch in rows.split(32):
        output = model(pool.inputs(x, batch).to(device))
        logits = torch.stack((output.pcr_logit, output.recurrence_logit), 1)
        values = {
            "ct1": output.ct1,
            "logits": logits,
            "member_logits": getattr(output, "member_logits", logits[..., None]),
            "last_stage": output.last_stage,
            "incomplete_history": output.incomplete_history,
        }
        for name, value in values.items():
            pieces[name].append(value.detach().cpu())
    result = {name: torch.cat(values) for name, values in pieces.items()}
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite holdout prediction")
    probabilities = result["member_logits"].double().sigmoid()
    result["probabilities"] = probabilities.mean(-1)
    result["member_disagreement"] = probabilities.var(-1, unbiased=False)
    return result


def score(
    model: Model,
    x: torch.Tensor,
    pool: EventPool,
    rows: torch.Tensor,
    phase: str,
    positive_weight: float,
) -> tuple[float, dict]:
    from types import SimpleNamespace

    predicted = infer(model, x, pool, rows)
    targets = pool.targets(rows, torch.device("cpu"))
    output = SimpleNamespace(
        ct1=predicted["ct1"],
        pcr_logit=predicted["logits"][:, 0],
        recurrence_logit=predicted["logits"][:, 1],
        member_logits=predicted["member_logits"],
    )
    loss, components = objective(output, targets, phase, positive_weight)
    metrics = {name: float(value) for name, value in components.items()}
    metrics["loss"] = float(loss)
    if phase == "pretrain":
        if not has_supervision(targets, phase):
            raise ValueError("No validation supervision")
        return float(loss), metrics
    valid = targets["valid"][:, 1]
    labels = targets["labels"][valid, 1]
    if labels.unique().numel() != 2:
        raise ValueError("Validation requires both recurrence classes")
    auprc = float(
        average_precision_score(labels.numpy(), predicted["probabilities"][valid, 1].numpy())
    )
    metrics["recurrence_auprc"] = auprc
    return -auprc, metrics


def train_phase(
    pool: EventPool,
    x: torch.Tensor,
    snapshot: dict,
    train: torch.Tensor,
    validation: torch.Tensor,
    root: Path,
    *,
    family: str,
    phase: str,
    seed: int,
    epochs: int = 100,
    parent: Path | None = None,
    dimensions: dict | None = None,
    interrupt_after_update: int | None = None,
) -> tuple[Model, dict]:
    if phase not in ("pretrain", "joint") or family not in FAMILIES or epochs < 1:
        raise ValueError("Invalid holdout phase configuration")
    if x.shape != (len(pool.base.ids), 360) or not torch.isfinite(x).all():
        raise ValueError("Input transform must align exactly with the development pool")
    if any(
        rows.ndim != 1
        or rows.dtype != torch.long
        or (rows < 0).any()
        or (rows >= len(pool.base.ids)).any()
        for rows in (train, validation)
    ):
        raise ValueError("Development row indices must be one-dimensional valid integer indices")
    train_ids = [pool.base.ids[i] for i in train.tolist()]
    val_ids = [pool.base.ids[i] for i in validation.tolist()]
    if (
        not train_ids
        or not val_ids
        or set(train_ids) & set(val_ids)
        or len(set(train_ids)) != len(train_ids)
        or len(set(val_ids)) != len(val_ids)
        or set(train_ids) | set(val_ids) != set(pool.base.ids)
        or snapshot["fit_ids"] != train_ids
        or snapshot["pool_id"] != pool.artifact_id
    ):
        raise ValueError("Trainer accepts exactly the development pool; test rows must be removed")
    dimensions = dimensions or model_dimensions(family, pool.base.ct0.shape[-1])
    parent_payload = (
        None if parent is None else torch.load(parent, weights_only=True, map_location="cpu")
    )
    if (phase == "joint") != (parent_payload is not None):
        raise ValueError("Joint fitting requires this run's selected pretraining checkpoint")
    if parent_payload is not None:
        pc = parent_payload["contract"]
        if (
            pc["task"] != TASK
            or pc["family"] != family
            or pc["phase"] != "pretrain"
            or pc["seed"] != seed
            or pc["snapshot_id"] != snapshot["artifact_id"]
            or pc["train_ids"] != train_ids
            or pc["validation_ids"] != val_ids
            or pc["dimensions"] != dimensions
            or parent_payload["epoch"] != parent_payload["early_stop"]["selected_epoch"]
        ):
            raise ValueError("Selected pretraining parent differs")
    contract = {
        "task": TASK,
        "family": family,
        "phase": phase,
        "seed": seed,
        "epochs": epochs,
        "pool_id": pool.artifact_id,
        "snapshot_id": snapshot["artifact_id"],
        "train_ids": train_ids,
        "validation_ids": val_ids,
        "dimensions": dimensions,
        "parent_id": None if parent_payload is None else parent_payload["artifact_id"],
        "protocol": specification(),
        "test_rows_available_to_trainer": False,
    }
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (root / "contract.json").exists() and read_json(root / "contract.json") != contract:
        raise ValueError("Holdout recovery contract differs")
    atomic_write_private_json(root / "contract.json", contract)
    _seed(seed)
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(family, dimensions).to(device)
    if parent_payload is None:
        model.world.fit_statistics(
            pool.base.ct0[train].to(device),
            pool.base.ct1_tokens[train].to(device),
            (pool.base.ct0_valid & pool.base.ct1_valid)[train].to(device),
        )
    else:
        model.load_state_dict(parent_payload["model_state"], strict=True)
    weight = recurrence_weight(pool, train) if phase == "joint" else 1.0
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.world.parameters()), "lr": 2e-5 if phase == "joint" else 2e-4},
            {
                "params": [*model.pcr_head.parameters(), *model.recurrence_head.parameters()],
                "lr": 2e-4,
            },
        ],
        weight_decay=0.01,
    )
    stopper, history, updates, first, elapsed_before = EarlyStopState(), [], 0, 1, 0.0
    if (root / "latest.pt").exists():
        saved = torch.load(root / "latest.pt", weights_only=True, map_location="cpu")
        if saved["contract"] != contract or saved["positive_weight"] != weight:
            raise ValueError("Checkpoint binding differs")
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        restore_rng_state(saved["rng_state"])
        stopper = EarlyStopState(**saved["early_stop"])
        history, updates, first = saved["history"], saved["updates"], saved["epoch"] + 1
        elapsed_before = saved["elapsed_seconds"]
        if stopper.selected_epoch == saved["epoch"]:
            selected_path = root / "selected.pt"
            previous = (
                torch.load(selected_path, weights_only=True, map_location="cpu")
                if selected_path.exists()
                else None
            )
            if previous is None or previous["artifact_id"] != saved["artifact_id"]:
                _atomic_torch_save(selected_path, saved)
    began = time.monotonic()
    use_mask = family not in ("event_v1", "event_v2_no_mask")
    for epoch in range(first, epochs + 1):
        if stopper.stale_epochs >= 15:
            break
        model.train()
        order = train[torch.randperm(len(train))]
        gradient_max, loss_sum, mask_sum, batches = 0.0, 0.0, 0.0, 0
        for rows in order.split(32):
            targets = pool.targets(rows, device)
            if not has_supervision(targets, phase):
                continue
            optimizer.zero_grad(set_to_none=True)
            inputs = pool.inputs(x, rows).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(inputs)
                loss, _ = objective(output, targets, phase, weight)
                auxiliary = loss.new_zeros(())
                if use_mask:
                    assert isinstance(model, EventV2Model)
                    mask = torch.rand(len(rows), 27, device=device) < 0.25
                    auxiliary = model.masked_source_loss(inputs, mask)
                    loss = loss + 0.05 * auxiliary
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite holdout training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            gradient_max = max(gradient_max, float(norm))
            optimizer.step()
            updates, batches = updates + 1, batches + 1
            loss_sum, mask_sum = (
                loss_sum + float(loss.detach()),
                mask_sum + float(auxiliary.detach()),
            )
            if interrupt_after_update is not None and updates == interrupt_after_update:
                raise RuntimeError("intentional_partial_epoch_interruption")
        if not batches:
            raise ValueError("No usable training supervision")
        value, metrics = score(model, x, pool, validation, phase, weight)
        selected = stopper.update(value, epoch, 0.0001)
        history.append(
            {
                "epoch": epoch,
                "updates": updates,
                "score": value,
                "metrics": metrics,
                "train_batch_mean_loss": loss_sum / batches,
                "train_masked_CT0_loss": mask_sum / batches,
                "gradient_norm_max": gradient_max,
                "selected": selected,
            }
        )
        payload = {
            "artifact_id": new_artifact_id("event-v2-checkpoint"),
            "contract": contract,
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "rng_state": capture_rng_state(),
            "epoch": epoch,
            "updates": updates,
            "early_stop": asdict(stopper),
            "history": history,
            "positive_weight": weight,
            "elapsed_seconds": elapsed_before + time.monotonic() - began,
        }
        _atomic_torch_save(root / "latest.pt", payload)
        if selected:
            _atomic_torch_save(root / "selected.pt", payload)
        atomic_write_private_json(
            root / "progress.json",
            {
                "status": "training",
                "family": family,
                "phase": phase,
                "seed": seed,
                "epoch": epoch,
                "updates": updates,
                "selected_epoch": stopper.selected_epoch,
                "validation_score": value,
                "device": str(device),
            },
        )
        print(
            f"seed={seed} family={family} phase={phase} epoch={epoch} "
            f"updates={updates} validation_score={value:.6f} selected={stopper.selected_epoch}",
            flush=True,
        )
    final = torch.load(root / "latest.pt", weights_only=True, map_location="cpu")
    if not (root / "final.pt").exists():
        _atomic_torch_save(root / "final.pt", final)
    elif (
        torch.load(root / "final.pt", weights_only=True, map_location="cpu")["artifact_id"]
        != final["artifact_id"]
    ):
        raise ValueError("Completed final checkpoint differs from latest")
    selected_payload = torch.load(root / "selected.pt", weights_only=True, map_location="cpu")
    if selected_payload["epoch"] != final["early_stop"]["selected_epoch"]:
        raise ValueError("Selected checkpoint epoch differs from final selection")
    for saved in (final, selected_payload):
        if saved["contract"] != contract or saved["positive_weight"] != weight:
            raise ValueError("Selected/final checkpoint binding differs")
        model.load_state_dict(saved["model_state"], strict=True)
        actual, _ = score(model, x, pool, validation, phase, weight)
        if not math.isclose(actual, saved["history"][-1]["score"], abs_tol=1e-7, rel_tol=1e-6):
            raise ValueError("Holdout checkpoint validation score does not replay")
    report = {
        "status": "completed",
        "family": family,
        "phase": phase,
        "seed": seed,
        "selected_epoch": selected_payload["epoch"],
        "selected_artifact_id": selected_payload["artifact_id"],
        "selected_score": selected_payload["history"][-1]["score"],
        "selected_metrics": selected_payload["history"][-1]["metrics"],
        "completed_epochs": final["epoch"],
        "updates": final["updates"],
        "seconds": final["elapsed_seconds"],
        "positive_weight": weight,
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": str(device),
        "checkpoint_score_replay": True,
        "test_rows_available_to_trainer": False,
    }
    atomic_write_private_json(root / "completed.json", report)
    atomic_write_private_json(root / "progress.json", report)
    return model.eval(), report
