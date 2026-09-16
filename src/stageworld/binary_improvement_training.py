"""Finite binary comparisons with raw-best selection and exact epoch recovery."""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from stageworld.artifacts import atomic_write_private_json, new_artifact_id, read_json
from stageworld.binary_endpoints import ENDPOINTS, BinaryEndpointBatch, build_binary_model
from stageworld.binary_evaluation import _point
from stageworld.binary_improvement_spec import PROTOCOL, Arm
from stageworld.config import StageWorldConfig
from stageworld.ct6_training import EarlyStopState
from stageworld.generated_workflow import _seed
from stageworld.model.compact_residual_binary import (
    CompactBinaryModel,
    CompactConfig,
    DeterministicLegacyModel,
    ForecastOutput,
)
from stageworld.synthetic_workflow import _atomic_torch_save
from stageworld.training import capture_rng_state, restore_rng_state

Predictor = CompactBinaryModel | DeterministicLegacyModel


def make_model(config: StageWorldConfig, arm: Arm) -> Predictor:
    if arm.architecture == "legacy_deterministic":
        return DeterministicLegacyModel(build_binary_model(config).config)
    return CompactBinaryModel(
        CompactConfig(architecture=arm.architecture, input_dim=config.model.ct_input_dim)
    )


def positive_weights(batches: Sequence[BinaryEndpointBatch], mode: str) -> Tensor:
    if mode == "none":
        return torch.ones(2)
    if mode != "sqrt":
        raise ValueError("Unknown class-weight rule")
    y = torch.cat([b.endpoint_labels for b in batches])
    mask = torch.cat([b.endpoint_valid for b in batches])
    result = []
    for i in range(2):
        values = y[mask[:, i], i]
        positive = int(values.sum())
        negative = len(values) - positive
        if not positive or not negative:
            raise ValueError("Class weighting requires both classes in training")
        result.append(math.sqrt(negative / positive))
    return torch.tensor(result)


def loss_components(
    output: ForecastOutput,
    batch: BinaryEndpointBatch,
    *,
    world: bool,
    weights: Tensor,
    endpoint: int | None = None,
) -> dict[str, Tensor]:
    zero = output.generated.sum() * 0
    components = {name: zero for name in (*ENDPOINTS, "ct")}
    if output.ct_mean is not None:
        valid = batch.future_ct_valid.any(1)
        if valid.any():
            prediction = output.ct_mean[valid]
            target = batch.future_ct_target[valid].detach()
            components["ct"] = (
                F.smooth_l1_loss(prediction, target)
                + (1 - F.cosine_similarity(prediction, target, dim=-1)).mean()
            )
    if not world:
        for i, name in enumerate(ENDPOINTS):
            valid = batch.endpoint_valid[:, i]
            if valid.any() and (endpoint is None or endpoint == i):
                components[name] = F.binary_cross_entropy_with_logits(
                    output.logits[valid, i],
                    batch.endpoint_labels[valid, i].float(),
                    pos_weight=weights[i],
                )
    components["total"] = components["ct"] * (1 if world else 0.1)
    if not world:
        components["total"] = components["total"] + sum(components[name] for name in ENDPOINTS)
    return components


@torch.inference_mode()
def predict(
    model: Predictor,
    batches: Sequence[BinaryEndpointBatch],
    *,
    correction: Tensor | None = None,
) -> dict[str, Tensor]:
    model.eval()
    device = next(model.parameters()).device
    outputs = [model(**b.to(device).prediction_inputs(deterministic=True)) for b in batches]
    logits = torch.cat([o.logits.cpu() for o in outputs]).float()
    if correction is not None:
        logits = logits - correction.cpu().log()
    result = {
        "logits": logits,
        "probabilities": logits.sigmoid(),
        "initial": torch.cat([o.initial.mean(1).cpu() for o in outputs]).float(),
        "generated": torch.cat([o.generated.mean(1).cpu() for o in outputs]).float(),
    }
    if outputs[0].ct_mean is not None:
        result["ct_mean"] = torch.cat([o.ct_mean.cpu() for o in outputs]).float()
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite model output")
    return result


def summarize_predictions(
    prediction: dict[str, Tensor],
    batches: Sequence[BinaryEndpointBatch],
    *,
    world: bool = False,
    endpoint: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not world:
        y = torch.cat([b.endpoint_labels for b in batches])
        valid = torch.cat([b.endpoint_valid for b in batches])
        result["endpoints"] = {
            name: _point(
                y[valid[:, i], i].numpy(), prediction["probabilities"][valid[:, i], i].numpy()
            )
            for i, name in enumerate(ENDPOINTS)
            if endpoint is None or endpoint == i
        }
    if "ct_mean" in prediction:
        target = torch.cat([b.future_ct_target for b in batches]).float()
        predicted = prediction["ct_mean"]
        result["ct_loss"] = float(
            F.smooth_l1_loss(predicted, target)
            + (1 - F.cosine_similarity(predicted, target, dim=-1)).mean()
        )
        result["ct_mse"] = float((predicted - target).square().mean())
        result["target_variance"] = float(target.var(0, unbiased=False).mean())
        result["prediction_variance"] = float(predicted.var(0, unbiased=False).mean())
    result["generated_variance"] = float(prediction["generated"].var(0, unbiased=False).mean())
    return result


def selection_score(report: dict[str, Any], world: bool) -> float:
    if world:
        return float(report["ct_loss"])
    scores = [row["bce"] for row in report["endpoints"].values()]
    if not scores or any(score is None for score in scores):
        raise ValueError("Inner endpoint selection lacks labels")
    return sum(scores) / len(scores)


def set_epoch_mode(model: Predictor, arm: Arm, *, world: bool, epoch: int) -> bool:
    head_ids = {id(p) for p in model.heads.parameters()}
    frozen = not world and (arm.policy == "frozen" or (arm.policy == "warm" and epoch <= 5))
    for parameter in model.parameters():
        parameter.requires_grad_(
            id(parameter) not in head_ids if world else (id(parameter) in head_ids or not frozen)
        )
    if frozen:
        model.eval()
        model.heads.train()
    else:
        model.train()
    return frozen


def train_phase(
    config: StageWorldConfig,
    arm: Arm,
    train: Sequence[BinaryEndpointBatch],
    inner: Sequence[BinaryEndpointBatch],
    root: Path,
    *,
    seed: int,
    snapshot_id: str,
    world: bool,
    parent: Path | None = None,
    epochs: int = 100,
    resume: bool = False,
    interrupt_after_update: int | None = None,
) -> tuple[Predictor, dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    _seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(config, arm).to(device)
    parent_payload = (
        None if parent is None else torch.load(parent, weights_only=True, map_location="cpu")
    )
    if world and parent is not None:
        raise ValueError("Pretraining must start from scratch")
    if parent_payload is not None:
        parent_contract = parent_payload["contract"]
        if (
            not parent_contract["world"]
            or parent_contract["snapshot_id"] != snapshot_id
            or parent_contract["seed"] != seed
            or parent_contract["arm"]["architecture"] != arm.architecture
        ):
            raise ValueError("World parent does not match this architecture/fold/seed")
        model.load_state_dict(parent_payload["model_state"], strict=True)
    elif not world and arm.architecture != "direct":
        raise ValueError("Generated-state classification requires its own pretraining")
    weights = positive_weights(train, "none" if world else arm.positive_weight)
    contract: dict[str, Any] = {
        "protocol": PROTOCOL,
        "arm": asdict(arm),
        "seed": seed,
        "snapshot_id": snapshot_id,
        "world": world,
        "epochs": epochs,
        "positive_weights": weights.tolist(),
        "parent_id": None if parent_payload is None else parent_payload["artifact_id"],
        "training_patients": [p for b in train for p in b.patient_ids],
        "inner_patients": [p for b in inner for p in b.patient_ids],
        "model_config": asdict(
            model.core.config if isinstance(model, DeterministicLegacyModel) else model.config
        ),
        "optimizer": {
            "head_lr": 2e-4,
            "backbone_lr": 2e-5 if arm.policy == "warm" and not world else 2e-4,
            "weight_decay": 0.01,
            "grad_clip": 1.0,
            "precision": config.training.mixed_precision,
        },
    }
    contract = json.loads(json.dumps(contract))
    if set(contract["training_patients"]) & set(contract["inner_patients"]):
        raise ValueError("Training/inner overlap")
    if (root / "contract.json").exists():
        if not resume or read_json(root / "contract.json") != contract:
            raise ValueError("Phase exists or resume contract differs")
    atomic_write_private_json(root / "contract.json", contract)
    head_ids = {id(p) for p in model.heads.parameters()}
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in model.parameters() if id(p) not in head_ids],
                "lr": contract["optimizer"]["backbone_lr"],
            },
            {"params": list(model.heads.parameters()), "lr": 2e-4},
        ],
        weight_decay=0.01,
    )
    stopper, history, updates, first_epoch, elapsed_before = EarlyStopState(), [], 0, 1, 0.0
    if resume and (root / "latest.pt").exists():
        saved = torch.load(root / "latest.pt", weights_only=True, map_location=device)
        if saved["contract"] != contract:
            raise ValueError("Checkpoint contract differs")
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        restore_rng_state(saved["rng_state"])
        stopper = EarlyStopState(**saved["early_stop"])
        history, updates = saved["history"], saved["updates"]
        first_epoch, elapsed_before = saved["epoch"] + 1, saved["elapsed_seconds"]
        if stopper.selected_epoch == saved["epoch"]:
            _atomic_torch_save(root / "selected.pt", saved)
    train_device = [batch.to(device) for batch in train]
    weights_device = weights.to(device)
    started = time.monotonic()
    for epoch in range(first_epoch, epochs + 1):
        if stopper.stale_epochs >= 15:
            break
        frozen = set_epoch_mode(model, arm, world=world, epoch=epoch)
        order = list(range(len(train)))
        random.shuffle(order)
        norms = []
        for index in order:
            batch = train_device[index]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and config.training.mixed_precision == "bf16",
            ):
                output = model(**batch.prediction_inputs(deterministic=True))
                losses = loss_components(
                    output, batch, world=world, weights=weights_device, endpoint=arm.endpoint
                )
            if not torch.isfinite(losses["total"]):
                raise ValueError("Nonfinite training loss")
            losses["total"].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            norms.append(float(norm))
            optimizer.step()
            updates += 1
            if interrupt_after_update is not None and updates == interrupt_after_update:
                raise RuntimeError("intentional_partial_epoch_interruption")
        train_report = summarize_predictions(
            predict(model, train, correction=weights), train, world=world, endpoint=arm.endpoint
        )
        inner_report = summarize_predictions(
            predict(model, inner, correction=weights), inner, world=world, endpoint=arm.endpoint
        )
        score = selection_score(inner_report, world)
        selected = stopper.update(score, epoch, 0.0001)
        row = {
            "epoch": epoch,
            "updates": updates,
            "training": train_report,
            "inner": inner_report,
            "selection_score": score,
            "backbone_frozen": frozen,
            "gradient_norm_max": max(norms),
            "selected": selected,
        }
        history.append(row)
        checkpoint = {
            "artifact_id": new_artifact_id("binary-improvement-checkpoint"),
            "contract": contract,
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "rng_state": capture_rng_state(),
            "epoch": epoch,
            "updates": updates,
            "early_stop": asdict(stopper),
            "history": history,
            "elapsed_seconds": elapsed_before + time.monotonic() - started,
        }
        _atomic_torch_save(root / "latest.pt", checkpoint)
        if selected:
            _atomic_torch_save(root / "selected.pt", checkpoint)
        atomic_write_private_json(root / "history.json", {"epochs": history})
        atomic_write_private_json(
            root / "progress.json",
            {
                "status": "running",
                "epoch": epoch,
                "updates": updates,
                "selected_epoch": stopper.selected_epoch,
                "selected_score": stopper.best,
            },
        )
    latest = torch.load(root / "latest.pt", weights_only=True, map_location="cpu")
    _atomic_torch_save(root / "final.pt", latest)
    final_payload = torch.load(root / "final.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(final_payload["model_state"], strict=True)
    final_report = summarize_predictions(
        predict(model, inner, correction=weights), inner, world=world, endpoint=arm.endpoint
    )
    if not math.isclose(
        selection_score(final_report, world),
        final_payload["history"][-1]["selection_score"],
        abs_tol=1e-7,
        rel_tol=1e-6,
    ):
        raise ValueError("Final checkpoint score does not replay")
    selected_payload = torch.load(root / "selected.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(selected_payload["model_state"], strict=True)
    selected_report = summarize_predictions(
        predict(model, inner, correction=weights), inner, world=world, endpoint=arm.endpoint
    )
    replay = selection_score(selected_report, world)
    if stopper.best is None or not math.isclose(replay, stopper.best, abs_tol=1e-7, rel_tol=1e-6):
        raise ValueError("Selected checkpoint score does not replay")
    report = {
        "status": "completed",
        "selected_epoch": stopper.selected_epoch,
        "selected_score": stopper.best,
        "completed_epochs": latest["epoch"],
        "optimizer_updates": latest["updates"],
        "elapsed_seconds": latest["elapsed_seconds"],
        "selected_inner": selected_report,
        "final_inner": final_report,
        "positive_weights": weights.tolist(),
        "registered_parameters": sum(p.numel() for p in model.parameters()),
        "verification": {
            "selected_score_replayed": True,
            "final_score_replayed": True,
            "parent_bound": parent_payload is not None,
            "recovery_boundary": "last_complete_epoch",
            "no_outer_selection": True,
        },
    }
    atomic_write_private_json(root / "progress.json", report)
    return model.eval(), report
