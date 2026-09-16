"""Two-phase, fold-local event training with complete epoch recovery."""

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
from stageworld.event_models import EventModel, EventOutput, masked_bce
from stageworld.event_spec import TASK, specification
from stageworld.generated700_models import feature_set_loss
from stageworld.generated_workflow import _seed
from stageworld.synthetic_workflow import _atomic_torch_save
from stageworld.training import capture_rng_state, restore_rng_state


def has_supervision(targets: dict, phase: str) -> bool:
    return bool(
        targets["ct_valid"].any()
        or targets["valid"][:, 0].any()
        or (phase == "joint" and targets["valid"][:, 1].any())
    )


def objective(
    output: EventOutput, targets: dict, phase: str, positive_weight: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if phase not in ("pretrain", "joint"):
        raise ValueError("Unknown event training phase")
    ct = feature_set_loss(output.ct1, targets["ct1"], targets["ct_valid"])
    pcr = masked_bce(output.pcr_logit, targets["labels"][:, 0], targets["valid"][:, 0])
    recurrence = (
        masked_bce(
            output.recurrence_logit,
            targets["labels"][:, 1],
            targets["valid"][:, 1],
            positive_weight,
        )
        if phase == "joint"
        else output.recurrence_logit.new_zeros(())
    )
    loss = ct + 0.5 * pcr if phase == "pretrain" else recurrence + 0.5 * pcr + 0.1 * ct
    return loss, {"ct_loss": ct, "pcr_bce": pcr, "recurrence_weighted_bce": recurrence}


@torch.inference_mode()
def infer(model: EventModel, x: torch.Tensor, pool: EventPool, rows: torch.Tensor) -> dict:
    model.eval()
    device = next(model.parameters()).device
    pieces: dict[str, list[torch.Tensor]] = {
        "ct1": [],
        "logits": [],
        "last_stage": [],
        "incomplete_history": [],
    }
    for batch in rows.split(32):
        output = model(pool.inputs(x, batch).to(device))
        pieces["ct1"].append(output.ct1.cpu())
        pieces["logits"].append(torch.stack((output.pcr_logit, output.recurrence_logit), 1).cpu())
        pieces["last_stage"].append(output.last_stage.cpu())
        pieces["incomplete_history"].append(output.incomplete_history.cpu())
    result = {name: torch.cat(values) for name, values in pieces.items()}
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite event prediction")
    result["probabilities"] = result["logits"].double().sigmoid()
    return result


def score(
    model: EventModel,
    x: torch.Tensor,
    pool: EventPool,
    rows: torch.Tensor,
    phase: str,
    positive_weight: float,
) -> tuple[float, dict]:
    prediction = infer(model, x, pool, rows)
    targets = pool.targets(rows, torch.device("cpu"))
    output = EventOutput(
        torch.empty(0),
        prediction["ct1"],
        prediction["logits"][:, 0],
        prediction["logits"][:, 1],
        prediction["last_stage"],
        prediction["incomplete_history"],
    )
    loss, components = objective(output, targets, phase, positive_weight)
    metrics = {name: float(value) for name, value in components.items()}
    metrics["loss"] = float(loss)
    if phase == "pretrain":
        if not has_supervision(targets, phase):
            raise ValueError("No validation supervision for pretraining")
        return float(loss), metrics
    valid = targets["valid"][:, 1]
    y = targets["labels"][valid, 1]
    if y.unique().numel() != 2:
        raise ValueError("Joint checkpoint selection requires both recurrence classes")
    auprc = float(average_precision_score(y.numpy(), prediction["probabilities"][valid, 1].numpy()))
    metrics["recurrence_auprc"] = auprc
    return -auprc, metrics


def recurrence_weight(pool: EventPool, train: torch.Tensor) -> float:
    valid = pool.base.valid[train, 1]
    y = pool.base.labels[train, 1][valid]
    positives, negatives = int((y == 1).sum()), int((y == 0).sum())
    if not positives or not negatives:
        raise ValueError("Training recurrence weights require both classes")
    return negatives / positives


def train_phase(
    pool: EventPool,
    x: torch.Tensor,
    snapshot: dict,
    train: torch.Tensor,
    validation: torch.Tensor,
    root: Path,
    *,
    phase: str,
    seed: int,
    epochs: int = 100,
    parent: Path | None = None,
    hidden: int = 128,
    layers: int = 4,
    interrupt_after_update: int | None = None,
) -> tuple[EventModel, dict]:
    if phase not in ("pretrain", "joint") or epochs < 1:
        raise ValueError("Invalid event phase configuration")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    train_ids = [pool.base.ids[i] for i in train.tolist()]
    val_ids = [pool.base.ids[i] for i in validation.tolist()]
    if (
        not train_ids
        or not val_ids
        or set(train_ids) & set(val_ids)
        or train_ids != snapshot["fit_ids"]
        or snapshot["pool_id"] != pool.artifact_id
    ):
        raise ValueError("Phase partitions and training-only input fit differ")
    parent_payload = (
        None if parent is None else torch.load(parent, weights_only=True, map_location="cpu")
    )
    if (phase == "joint") != (parent_payload is not None):
        raise ValueError("Joint phase requires this seed's best pretraining checkpoint")
    if parent_payload is not None:
        pc = parent_payload["contract"]
        if (
            pc["task"] != TASK
            or pc["phase"] != "pretrain"
            or pc["seed"] != seed
            or pc["snapshot_id"] != snapshot["artifact_id"]
            or pc["train_ids"] != train_ids
            or pc["validation_ids"] != val_ids
            or parent_payload["epoch"] != parent_payload["early_stop"]["selected_epoch"]
        ):
            raise ValueError("Pretraining parent differs from this fold or seed")
    dimensions = {"image_dim": pool.base.ct0.shape[-1], "hidden": hidden, "layers": layers}
    contract = {
        "task": TASK,
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
    }
    if (root / "contract.json").exists() and read_json(root / "contract.json") != contract:
        raise ValueError("Epoch recovery contract differs")
    atomic_write_private_json(root / "contract.json", contract)
    _seed(seed)
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EventModel(**dimensions).to(device)
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
        if stopper.selected_epoch == saved["epoch"] and not (root / "selected.pt").exists():
            _atomic_torch_save(root / "selected.pt", saved)
        elif stopper.selected_epoch == saved["epoch"]:
            previous = torch.load(root / "selected.pt", weights_only=True, map_location="cpu")
            if previous["artifact_id"] != saved["artifact_id"]:
                _atomic_torch_save(root / "selected.pt", saved)
    began = time.monotonic()
    for epoch in range(first, epochs + 1):
        if stopper.stale_epochs >= 15:
            break
        model.train()
        order = train[torch.randperm(len(train))]
        gradient_max, loss_sum, batches, skipped = 0.0, 0.0, 0, 0
        for rows in order.split(32):
            targets = pool.targets(rows, device)
            if not has_supervision(targets, phase):
                skipped += 1
                continue
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(pool.inputs(x, rows).to(device))
                loss, _ = objective(output, targets, phase, weight)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            gradient_max = max(gradient_max, float(norm))
            optimizer.step()
            updates, batches, loss_sum = updates + 1, batches + 1, loss_sum + float(loss.detach())
            if interrupt_after_update is not None and updates == interrupt_after_update:
                raise RuntimeError("intentional_partial_epoch_interruption")
        if not batches:
            raise ValueError("No usable training supervision in this fold")
        value, metrics = score(model, x, pool, validation, phase, weight)
        selected = stopper.update(value, epoch, 0.0001)
        history.append(
            {
                "epoch": epoch,
                "updates": updates,
                "score": value,
                "metrics": metrics,
                "train_batch_mean_loss": loss_sum / batches,
                "skipped_batches": skipped,
                "gradient_norm_max": gradient_max,
                "selected": selected,
            }
        )
        payload = {
            "artifact_id": new_artifact_id("event-checkpoint"),
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
                "status": "running",
                "device": str(device),
                "phase": phase,
                "epoch": epoch,
                "updates": updates,
                "selected_epoch": stopper.selected_epoch,
                "score": value,
                "metrics": metrics,
            },
        )
        print(
            f"phase={phase} epoch={epoch} updates={updates} score={value:.6f} "
            f"selected={stopper.selected_epoch} device={device}",
            flush=True,
        )
    final = torch.load(root / "latest.pt", weights_only=True, map_location="cpu")
    _atomic_torch_save(root / "final.pt", final)
    selected_payload = torch.load(root / "selected.pt", weights_only=True, map_location="cpu")
    for saved in (final, selected_payload):
        model.load_state_dict(saved["model_state"], strict=True)
        actual, _ = score(model, x, pool, validation, phase, weight)
        if not math.isclose(actual, saved["history"][-1]["score"], abs_tol=1e-7, rel_tol=1e-6):
            raise ValueError("Checkpoint score does not replay")
    report = {
        "status": "completed",
        "phase": phase,
        "selected_epoch": selected_payload["epoch"],
        "selected_score": selected_payload["history"][-1]["score"],
        "selected_metrics": selected_payload["history"][-1]["metrics"],
        "completed_epochs": final["epoch"],
        "updates": final["updates"],
        "seconds": final["elapsed_seconds"],
        "positive_weight": weight,
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_score_replay": True,
        "world_jointly_adapted": phase == "joint",
    }
    atomic_write_private_json(root / "completed.json", report)
    atomic_write_private_json(root / "progress.json", report)
    return model.eval(), report
