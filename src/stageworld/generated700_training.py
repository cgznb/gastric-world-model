"""Restartable inner selection and fresh fixed-duration outer refitting."""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from stageworld.artifacts import atomic_write_private_json, new_artifact_id, read_json
from stageworld.binary700_statistics import positive_weights
from stageworld.ct6_training import EarlyStopState
from stageworld.generated700_data import Pool
from stageworld.generated700_models import (
    AnchoredClassifier,
    FutureWorld,
    endpoint_loss,
    feature_set_loss,
)
from stageworld.generated700_spec import Candidate
from stageworld.generated_workflow import _seed
from stageworld.synthetic_workflow import _atomic_torch_save
from stageworld.training import capture_rng_state, restore_rng_state


@torch.inference_mode()
def infer(
    model: torch.nn.Module,
    x: torch.Tensor,
    pool: Pool,
    indices: torch.Tensor,
    *,
    world: bool = False,
) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device
    values = []
    for rows in indices.split(32):
        current, ct = x[rows].to(device), pool.ct0[rows].to(device)
        if world:
            output = model(current, ct)[2]
        else:
            output = model(current, ct, pool.ct0_valid[rows].to(device))
        values.append(output.float().cpu())
    result = torch.cat(values)
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite prediction")
    return result


def score(
    model: torch.nn.Module, x: torch.Tensor, pool: Pool, indices: torch.Tensor, world: bool
) -> tuple[float, dict]:
    prediction = infer(model, x, pool, indices, world=world)
    if world:
        valid = (pool.ct0_valid & pool.ct1_valid)[indices]
        value = float(feature_set_loss(prediction, pool.ct1_tokens[indices], valid))
        return value, {
            "ct_loss": value,
            "ct_mse": float((prediction[valid].mean(1) - pool.ct1[indices][valid]).square().mean()),
        }
    probabilities = prediction.sigmoid().numpy()
    scores = []
    for i in range(2):
        mask = pool.valid[indices, i].numpy()
        scores.append(
            float(
                average_precision_score(
                    pool.labels[indices, i].numpy()[mask], probabilities[mask, i]
                )
            )
        )
    return -float(np.mean(scores)), {"auprc": scores}


def build_model(
    family: str,
    width: int,
    image_dim: int,
    anchor: tuple[torch.Tensor, torch.Tensor] | None,
    parent: dict | None = None,
) -> torch.nn.Module:
    torch.backends.mha.set_fastpath_enabled(False)
    if family == "world":
        return FutureWorld(width, image_dim)
    if anchor is None:
        raise ValueError("Bind the matching unweighted logistic anchor")
    model = AnchoredClassifier(width, family, *anchor, image_dim=image_dim)
    if family.startswith("generated"):
        if parent is None or model.world is None:
            raise ValueError("Generated classification requires a same-partition world parent")
        model.world.load_state_dict(parent["model_state"], strict=True)
    return model


def train_phase(
    pool: Pool,
    x: torch.Tensor,
    snapshot: dict,
    candidate: Candidate,
    train: torch.Tensor,
    validation: torch.Tensor | None,
    root: Path,
    *,
    seed: int,
    epochs: int = 100,
    anchor: tuple[torch.Tensor, torch.Tensor] | None = None,
    parent: Path | None = None,
    interrupt_after_update: int | None = None,
) -> tuple[torch.nn.Module, dict]:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    world = candidate.family == "world"
    train_ids = [pool.ids[i] for i in train.tolist()]
    val_ids = [] if validation is None else [pool.ids[i] for i in validation.tolist()]
    if set(train_ids) & set(val_ids) or set(train_ids) != set(snapshot["fit_ids"]):
        raise ValueError("Phase partitions and fitted preprocessing differ")
    parent_payload = (
        None if parent is None else torch.load(parent, weights_only=True, map_location="cpu")
    )
    if parent_payload is not None and (
        parent_payload["contract"]["snapshot_id"] != snapshot["artifact_id"]
        or parent_payload["contract"]["seed"] != seed
        or parent_payload["contract"]["candidate"]["family"] != "world"
    ):
        raise ValueError("World parent partition/seed differs")
    contract = {
        "schema": "generated700-phase-v2",
        "candidate": asdict(candidate),
        "seed": seed,
        "snapshot_id": snapshot["artifact_id"],
        "train_ids": train_ids,
        "validation_ids": val_ids,
        "fixed_refit": validation is None,
        "epochs": epochs,
        "parent_id": None if parent_payload is None else parent_payload["artifact_id"],
        "tabular_dim": x.shape[1],
        "image_dim": pool.ct0.shape[-1],
        "attention_fastpath": False,
    }
    if (root / "contract.json").exists() and read_json(root / "contract.json") != contract:
        raise ValueError("Phase recovery contract differs")
    atomic_write_private_json(root / "contract.json", contract)
    _seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        candidate.family, x.shape[1], pool.ct0.shape[-1], anchor, parent_payload
    ).to(device)
    if world:
        assert isinstance(model, FutureWorld)
        supervised = train[(pool.ct0_valid & pool.ct1_valid)[train]]
        if not len(supervised):
            raise ValueError("No usable paired CT training supervision")
        model.fit_statistics(
            pool.ct0[supervised].to(device), pool.ct1_tokens[supervised].to(device)
        )
        fitting = supervised
        weights = torch.ones(2, device=device)
    else:
        fitting = train
        weights = positive_weights(pool.labels[train], pool.valid[train], candidate.weight).to(
            device
        )
    trainable = [p for p in model.parameters() if p.requires_grad]
    if isinstance(model, AnchoredClassifier) and candidate.family == "generated":
        groups = [
            {"params": list(model.endpoints.parameters()), "lr": 2e-4},
            {"params": list(model.world.parameters()), "lr": 2e-5},
        ]
        optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=0.01)
    stopper, history, updates, first, elapsed_before = EarlyStopState(), [], 0, 1, 0.0
    if (root / "latest.pt").exists():
        saved = torch.load(root / "latest.pt", weights_only=True, map_location=device)
        if saved["contract"] != contract:
            raise ValueError("Recovered checkpoint does not match this run")
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        restore_rng_state(saved["rng_state"])
        stopper = EarlyStopState(**saved["early_stop"])
        history, updates, first = saved["history"], saved["updates"], saved["epoch"] + 1
        elapsed_before = saved["elapsed_seconds"]
        if stopper.selected_epoch == saved["epoch"]:
            _atomic_torch_save(root / "selected.pt", saved)
    began = time.monotonic()
    for epoch in range(first, epochs + 1):
        if validation is not None and stopper.stale_epochs >= 15:
            break
        model.train()
        order = fitting[torch.randperm(len(fitting))]
        gradient_max = 0.0
        for rows in order.split(32):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                if world:
                    future = model(x[rows].to(device), pool.ct0[rows].to(device))[2]
                    loss = feature_set_loss(
                        future, pool.ct1_tokens[rows].to(device), pool.ct1_valid[rows].to(device)
                    )
                else:
                    assert isinstance(model, AnchoredClassifier)
                    logits, future = model.forward_with_features(
                        x[rows].to(device),
                        pool.ct0[rows].to(device),
                        pool.ct0_valid[rows].to(device),
                    )
                    loss = endpoint_loss(
                        logits,
                        pool.labels[rows].to(device),
                        pool.valid[rows].to(device),
                        weights,
                        candidate.loss,
                    )
                    if candidate.family == "generated":
                        loss = loss + 0.1 * feature_set_loss(
                            future,
                            pool.ct1_tokens[rows].to(device),
                            (pool.ct0_valid & pool.ct1_valid)[rows].to(device),
                        )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite optimization loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            gradient_max = max(gradient_max, float(norm))
            optimizer.step()
            updates += 1
            if interrupt_after_update is not None and updates == interrupt_after_update:
                raise RuntimeError("intentional_partial_epoch_interruption")
        evaluation_rows = train if validation is None else validation
        value, metrics = score(model, x, pool, evaluation_rows, world)
        if validation is None:
            selected = True
            stopper.best, stopper.selected_epoch = value, epoch
        else:
            selected = stopper.update(value, epoch, 0.0001)
        history.append(
            {
                "epoch": epoch,
                "updates": updates,
                "score": value,
                "metrics": metrics,
                "gradient_norm_max": gradient_max,
                "selected": selected,
            }
        )
        payload = {
            "artifact_id": new_artifact_id("generated700-checkpoint"),
            "contract": contract,
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "rng_state": capture_rng_state(),
            "epoch": epoch,
            "updates": updates,
            "early_stop": asdict(stopper),
            "history": history,
            "positive_weights": weights.detach().cpu(),
            "elapsed_seconds": elapsed_before + time.monotonic() - began,
        }
        _atomic_torch_save(root / "latest.pt", payload)
        if selected:
            _atomic_torch_save(root / "selected.pt", payload)
        atomic_write_private_json(
            root / "progress.json", {"status": "running", "epoch": epoch, "updates": updates}
        )
    final = torch.load(root / "latest.pt", weights_only=True, map_location="cpu")
    _atomic_torch_save(root / "final.pt", final)
    selected = torch.load(root / "selected.pt", weights_only=True, map_location="cpu")
    evaluation_rows = train if validation is None else validation
    for payload in (final, selected):
        model.load_state_dict(payload["model_state"], strict=True)
        actual, _ = score(model, x, pool, evaluation_rows, world)
        expected = payload["history"][-1]["score"]
        if not math.isclose(actual, expected, abs_tol=1e-7, rel_tol=1e-6):
            raise ValueError("Saved checkpoint score does not replay")
    if parent_payload is not None and candidate.family == "generated_frozen":
        assert isinstance(model, AnchoredClassifier) and model.world is not None
        if any(
            not torch.equal(v.detach().cpu(), parent_payload["model_state"][k])
            for k, v in model.world.state_dict().items()
        ):
            raise ValueError("Frozen world weights changed during classification")
    report = {
        "status": "completed",
        "selected_epoch": selected["epoch"],
        "selected_score": selected["history"][-1]["score"],
        "completed_epochs": final["epoch"],
        "updates": final["updates"],
        "seconds": final["elapsed_seconds"],
        "registered_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "positive_weights": weights.detach().cpu().tolist(),
        "fixed_refit": validation is None,
        "checkpoint_replay_passed": True,
        "frozen_parent_verified": parent_payload is not None
        and candidate.family == "generated_frozen",
        "world_jointly_adapted": candidate.family == "generated",
    }
    atomic_write_private_json(root / "completed.json", report)
    return model.eval(), report
