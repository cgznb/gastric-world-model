"""Development-only feature, geometry and frozen-model mechanism diagnostics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from torch import Tensor
from torch.nn import functional as F

from stageworld.artifacts import read_json
from stageworld.binary_data import BinaryDevelopmentData
from stageworld.binary_endpoints import BinaryEndpointBatch, build_binary_model
from stageworld.binary_evaluation import binary_bce, predict_binary_logits
from stageworld.binary_improvement_spec import Arm
from stageworld.binary_improvement_training import Predictor, loss_components, predict, train_phase
from stageworld.config import StageWorldConfig
from stageworld.model.compact_residual_binary import CompactBinaryModel, spatial_order
from stageworld.training import checkpoint_payload_mismatches


def distribution(values: Tensor) -> dict[str, float]:
    values = values.detach().float().flatten()
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p05": float(values.quantile(0.05)),
        "p95": float(values.quantile(0.95)),
    }


def geometry_audit(config: StageWorldConfig, data: BinaryDevelopmentData) -> dict[str, Any]:
    """Coordinate compatibility does not establish anatomical registration."""
    distances, ratios, matches = [], [], 0
    systems: Counter[str] = Counter()
    for roles in data.features.values():
        before, after = roles["baseline_ct"], roles["post_treatment_ct"]
        order0, order1 = spatial_order(before), spatial_order(after)
        assert before.coords is not None and after.coords is not None
        a = before.coords.gather(1, order0[..., None].expand_as(before.coords))
        b = after.coords.gather(1, order1[..., None].expand_as(after.coords))
        systems[str(before.coordinate_system)] += 1
        distances.append((a.mean(1) - b.mean(1)).norm(dim=-1))
        ratios.append((b.amax(1) - b.amin(1)) / (a.amax(1) - a.amin(1)))
        matches += int(torch.allclose(a, b, rtol=0, atol=1e-4))
    manifest = read_json(Path(str(config.paths.feature_root)) / "ct_manifest.json")
    allowed = set(data.labels)
    roi = Counter(
        str(row.get("roi_source", "unrecorded"))
        for row in manifest["entries"]
        if row["patient_id"] in allowed
    )
    return {
        "pairs": len(data.features),
        "complete_cartesian_27_token_grids": True,
        "coordinate_system_counts": dict(systems),
        "roi_source_counts": dict(roi),
        "same_coordinate_pairs": matches,
        "raw_coordinate_center_distance_mm": distribution(torch.cat(distances)),
        "center_distance_interpretation": (
            "Separate acquisition coordinate origins; not anatomical motion or tumor displacement."
        ),
        "axis_span_ratio_CT1_over_CT0": distribution(torch.cat(ratios)),
        "preprocess_version": config.encoders.ct_preprocess_version,
        "registration_verified": False,
        "phase_matching_verified": False,
        "spatial_supervision_enabled": False,
        "gate": "not_passed",
        "reason": (
            "Current producer crops each visit independently; no paired anatomical "
            "transform or phase adjudication is recorded."
        ),
        "next_step": (
            "Verify phase and CT0-defined paired registration before any "
            "tokenwise future target or multiscale re-extraction."
        ),
        "test_used": False,
    }


def feature_diagnostics(train: Sequence[BinaryEndpointBatch]) -> dict[str, Any]:
    before = torch.cat([b.ct0.values.mean(1) for b in train]).double()
    after = torch.cat([b.future_ct_target[:, 0] for b in train]).double()
    result: dict[str, Any] = {"scope": "gradient_training_partition_only", "patients": len(before)}
    for name, value in (("ct0", before), ("ct1", after), ("delta", after - before)):
        centered = value - value.mean(0)
        eigen = torch.linalg.svdvals(centered).square()
        fractions = eigen / eigen.sum().clamp_min(1e-20)
        result[name] = {
            "between_patient_variance": float(centered.square().mean()),
            "participation_rank": float(1 / fractions.square().sum().clamp_min(1e-20)),
            "top_component_variance_fraction": float(fractions[0]),
            "first_32_variance_fraction": float(fractions[:32].sum()),
        }
    result["paired_cosine"] = distribution(F.cosine_similarity(before, after))
    result["shuffled_pair_cosine"] = distribution(F.cosine_similarity(before, after.roll(1, 0)))
    result["common_mean_delta_energy_fraction"] = float(
        (after - before).mean(0).square().sum()
        / (after - before).square().sum(1).mean().clamp_min(1e-20)
    )
    return result


def ct_controls(
    train: Sequence[BinaryEndpointBatch],
    outer: Sequence[BinaryEndpointBatch],
    prediction: Tensor | None,
) -> dict[str, Any]:
    target = torch.cat([b.future_ct_target for b in outer])
    valid = torch.cat([b.future_ct_valid.any(1) for b in outer])
    training = torch.cat([b.future_ct_target for b in train])
    training_valid = torch.cat([b.future_ct_valid.any(1) for b in train])
    controls = {
        "persistence": torch.cat([b.ct0.values.mean(1, keepdim=True) for b in outer]),
        "training_mean": training[training_valid].mean(0, keepdim=True).expand_as(target),
    }
    if prediction is not None:
        controls["model"] = prediction
    return {
        name: {
            "mse": float((value[valid] - target[valid]).square().mean()),
            "ct_loss": float(
                F.smooth_l1_loss(value[valid], target[valid])
                + (1 - F.cosine_similarity(value[valid], target[valid], dim=-1)).mean()
            ),
            "between_patient_variance": float(value[valid].var(0, unbiased=False).mean()),
        }
        for name, value in controls.items()
    }


def verify_model_contract(model: Predictor, batch: BinaryEndpointBatch) -> dict[str, Any]:
    original = predict(model, [batch])["probabilities"]
    changed = replace(
        batch,
        ct1=replace(batch.ct1, values=batch.ct1.values + 37),
        future_ct_target=batch.future_ct_target + 51,
        endpoint_labels=1 - batch.endpoint_labels,
    )
    torch.testing.assert_close(original, predict(model, [changed])["probabilities"], rtol=0, atol=0)
    device = next(model.parameters()).device
    model.requires_grad_(True).eval()
    model.zero_grad(set_to_none=True)
    b = batch.to(device)
    output = model(**b.prediction_inputs(deterministic=True))
    losses = loss_components(output, b, world=False, weights=torch.ones(2, device=device))
    endpoint_loss = losses["pcr"] + losses["recurrence"]
    endpoint_loss.backward()
    gradients = {
        name: float(p.grad.detach().norm())
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    initial_prefix = (
        "ct_projection." if isinstance(model, CompactBinaryModel) else "core.input_projections.ct."
    )
    transition_prefix = "blocks." if isinstance(model, CompactBinaryModel) else "core.transition."
    initial_has_gradient = any(v > 0 for k, v in gradients.items() if k.startswith(initial_prefix))
    transition_has_gradient = any(
        v > 0 for k, v in gradients.items() if k.startswith(transition_prefix)
    )
    if not initial_has_gradient:
        raise ValueError("Endpoint gradient did not reach the baseline CT encoder")
    is_direct = isinstance(model, CompactBinaryModel) and model.config.architecture == "direct"
    if not is_direct and not transition_has_gradient:
        raise ValueError("Endpoint gradient did not reach the transition")
    if any("updater" in k or "observation_update" in k for k in gradients):
        raise ValueError("Endpoint gradient reached an observation updater")
    model.zero_grad(set_to_none=True)
    return {
        "status": "passed",
        "ct1_and_label_replacement_exact": True,
        "initial_gradient": initial_has_gradient,
        "transition_gradient": transition_has_gradient,
        "observation_updater_gradient": False,
        "future_target_requires_grad": b.future_ct_target.requires_grad,
        "gradient_probe_scope": "selected_model_with_gradients_enabled_for_structural_test",
    }


def verify_recovery(
    config: StageWorldConfig,
    train: Sequence[BinaryEndpointBatch],
    inner: Sequence[BinaryEndpointBatch],
    root: Path,
    reference: Path,
    snapshot_id: str,
    *,
    world: bool,
    arm: Arm,
    parent: Path | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(
        config=config,
        arm=arm,
        train=train,
        inner=inner,
        seed=17,
        snapshot_id=snapshot_id,
        world=world,
        epochs=2,
        parent=parent,
    )
    if not (root / "latest.pt").exists():
        try:
            train_phase(root=root, interrupt_after_update=2, **kwargs)
        except RuntimeError as error:
            if str(error) != "intentional_partial_epoch_interruption":
                raise
        else:
            raise ValueError("Recovery probe was not interrupted")
    model, _ = train_phase(root=root, resume=True, **kwargs)
    del model
    a = torch.load(reference / "final.pt", weights_only=True, map_location="cpu")
    b = torch.load(root / "final.pt", weights_only=True, map_location="cpu")
    for key in ("model_state", "optimizer_state", "rng_state", "epoch", "updates", "early_stop"):
        if checkpoint_payload_mismatches(a[key], b[key]):
            raise ValueError(f"Whole-epoch recovery differs: {key}")
    return {"status": "passed", "model_optimizer_rng_exact": True, "interrupted_update": 2}


def legacy_dependence(
    config: StageWorldConfig, inner: Sequence[BinaryEndpointBatch], checkpoint: Path
) -> dict[str, Any]:
    model = build_binary_model(config)
    payload = torch.load(checkpoint, weights_only=True, map_location="cpu")
    model.load_state_dict(payload["model_state"], strict=True)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    original = predict_binary_logits(model, inner)
    decode = model.endpoint_decoder.forward
    count = model.config.state_tokens

    def shuffled(tokens: Tensor, valid: Tensor, times: Tensor) -> Tensor:
        changed = tokens.clone()
        changed[:, -count:] = changed[:, -count:].roll(1, 0)
        return decode(changed, valid, times)

    with patch.object(model.endpoint_decoder, "forward", shuffled):
        altered = predict_binary_logits(model, inner)
    centered_batches = []
    ratios = []
    with torch.inference_mode():
        for batch in inner:
            observation = batch.to(next(model.parameters()).device).ct0
            assert observation.coords is not None and batch.ct0.coords is not None
            appearance = model.input_projections["ct"](observation.values)
            position = model.coordinate_projections["3"](observation.coords)
            ratios.append((position.norm(dim=-1) / appearance.norm(dim=-1).clamp_min(1e-8)).cpu())
            centered_batches.append(
                replace(
                    batch,
                    ct0=replace(
                        batch.ct0, coords=batch.ct0.coords - batch.ct0.coords.mean(1, keepdim=True)
                    ),
                )
            )
    centered = predict_binary_logits(model, centered_batches)
    return {
        "scope": "frozen_original_model_inner_only_posthoc_diagnostic",
        "original_bce": binary_bce(original, inner),
        "shuffled_generated_bce": binary_bce(altered, inner),
        "mean_absolute_probability_change": (original.sigmoid() - altered.sigmoid())
        .abs()
        .mean(0)
        .tolist(),
        "within_batch_patient_shuffle": True,
        "refitted": False,
        "coordinate_to_CT_projection_norm_ratio": distribution(torch.cat(ratios)),
        "centered_coordinate_bce": binary_bce(centered, inner),
        "centered_coordinate_mean_absolute_probability_change": (
            original.sigmoid() - centered.sigmoid()
        )
        .abs()
        .mean(0)
        .tolist(),
        "coordinate_probe_interpretation": (
            "Frozen-model sensitivity only, not a retrained ablation."
        ),
    }
