from __future__ import annotations

from dataclasses import fields, replace

import pytest
import torch
from test_binary_endpoints import binary_batch, binary_fixture
from test_ct6 import compact_row, ct6_rows

from stageworld.artifacts import atomic_write_private_json
from stageworld.binary_baselines import baseline_predict, fit_baseline, select_thresholds, targets
from stageworld.binary_improvement_diagnostics import legacy_dependence, verify_model_contract
from stageworld.binary_improvement_evaluation import paired_comparison
from stageworld.binary_improvement_inference import export_bundle, verify_portable
from stageworld.binary_improvement_spec import Arm
from stageworld.binary_improvement_training import (
    loss_components,
    make_model,
    positive_weights,
    predict,
    train_phase,
)
from stageworld.data.baseline_clinical import CT6_CLINICAL_SCHEMA, fit_clinical_transform
from stageworld.data.treatment_compact import fit_name_support, treatment_fields
from stageworld.errors import DataContractError
from stageworld.generated_inference import ct0_packet
from stageworld.model.compact_residual_binary import CompactBinaryModel, CompactConfig
from stageworld.synthetic_workflow import _atomic_torch_save, _slice_observation
from stageworld.training import checkpoint_payload_mismatches


def spatial_batch():
    batch = binary_batch()
    observations = {}
    for name in ("ct0", "ct1"):
        observation = getattr(batch, name)
        changes = {
            field.name: value[:, torch.zeros(27, dtype=torch.long)]
            for field in fields(observation)
            if isinstance(value := getattr(observation, field.name), torch.Tensor)
            and value.ndim >= 2
        }
        changes["values"] = torch.randn(batch.batch_size, 27, observation.values.shape[-1])
        grid = torch.cartesian_prod(torch.arange(3), torch.arange(3), torch.arange(3)).float()
        changes["coords"] = grid[None].repeat(batch.batch_size, 1, 1)
        changes["source_id"] = tuple((row[0],) * 27 for row in observation.source_id)
        changes["coordinate_system"] = "patient_ras_mm"
        observations[name] = replace(observation, **changes)
    return replace(batch, **observations)


@pytest.mark.parametrize("architecture", ["direct", "residual"])
def test_spatial_model_prefix_and_order_invariance(architecture):
    batch = spatial_batch()
    model = CompactBinaryModel(
        CompactConfig(architecture=architecture, input_dim=batch.ct0.values.shape[-1])
    ).eval()
    original = predict(model, [batch])["probabilities"]
    changed = replace(
        batch,
        ct1=replace(batch.ct1, values=batch.ct1.values + 100),
        future_ct_target=batch.future_ct_target + 100,
        endpoint_labels=torch.ones_like(batch.endpoint_labels),
    )
    assert torch.equal(original, predict(model, [changed])["probabilities"])
    permutation = torch.randperm(27)
    ct = replace(
        batch.ct0, values=batch.ct0.values[:, permutation], coords=batch.ct0.coords[:, permutation]
    )
    torch.testing.assert_close(
        original, predict(model, [replace(batch, ct0=ct)])["probabilities"], atol=1e-7, rtol=1e-6
    )
    bad = replace(batch.ct0, coords=torch.zeros_like(batch.ct0.coords))
    with pytest.raises(DataContractError):
        predict(model, [replace(batch, ct0=bad)])


def test_losses_gradient_flow_and_frozen_target():
    batch = spatial_batch()
    model = CompactBinaryModel(CompactConfig(input_dim=batch.ct0.values.shape[-1]))
    target = batch.future_ct_target.clone().requires_grad_()
    with pytest.raises(DataContractError, match="must be detached"):
        replace(batch, future_ct_target=target)
    # Probe loss-level detachment independently of the batch constructor's guard.
    target = batch.future_ct_target.requires_grad_()
    output = model(**batch.prediction_inputs(deterministic=True))
    components = loss_components(output, batch, world=False, weights=torch.ones(2))
    torch.testing.assert_close(
        components["total"], components["pcr"] + components["recurrence"] + 0.1 * components["ct"]
    )
    components["pcr"].backward(retain_graph=True)
    assert model.ct_projection[0].weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.blocks.parameters())
    assert target.grad is None and not hasattr(model, "survival_decoder")
    model.zero_grad(set_to_none=True)
    components["ct"].backward()
    assert target.grad is None
    world = loss_components(
        model(**batch.prediction_inputs(deterministic=True)),
        batch,
        world=True,
        weights=torch.ones(2),
    )
    torch.testing.assert_close(world["total"], world["ct"])


def test_cached_coordinate_rounding_preserves_grid_order():
    batch = spatial_batch()
    model = CompactBinaryModel(CompactConfig(input_dim=batch.ct0.values.shape[-1])).eval()
    original = predict(model, [batch])["probabilities"]
    coords = batch.ct0.coords * 64
    coords[coords == 128] = 128.5
    rounded = replace(batch, ct0=replace(batch.ct0, coords=coords))
    torch.testing.assert_close(original, predict(model, [rounded])["probabilities"], atol=0, rtol=0)
    coords = coords.clone()
    coords[:, 0, 0] += 1
    with pytest.raises(DataContractError, match="Cartesian"):
        predict(model, [replace(batch, ct0=replace(batch.ct0, coords=coords))])


def test_class_weights_mask_missing_and_thresholds():
    batch = spatial_batch()
    weight = positive_weights([batch], "sqrt")
    torch.testing.assert_close(weight, torch.tensor([2.0**0.5, 1.0]))
    y, mask = targets([batch])
    p = torch.tensor([[0.3, 0.1], [0.1, 0.4], [0.2, 0.2], [0.9, 0.5]])
    threshold = select_thresholds(p, y, mask)
    assert threshold == pytest.approx([0.3, 0.4])


def test_sklearn_transform_is_training_only_and_replay():
    train = spatial_batch()
    inner = replace(train, patient_ids=tuple(f"inner-{i}" for i in range(4)))
    first, _ = fit_baseline([train], [inner], "ct0")
    changed = replace(inner, ct0=replace(inner.ct0, values=inner.ct0.values + 100))
    other, _ = fit_baseline([train], [changed], "ct0")
    for key in ("mean", "scale", "pca_mean", "pca_components"):
        assert torch.equal(first["transform"][key], other["transform"][key])
    assert first["fit_patient_ids"] == list(train.patient_ids)
    prediction = baseline_predict(first, [inner])
    assert prediction.shape == (4, 2) and torch.isfinite(prediction).all()


def test_phase_recovery_and_freeze_preserve_backbone(tmp_path):
    torch.set_num_threads(1)
    config, _, _ = binary_fixture()
    train = spatial_batch()
    inner = replace(train, patient_ids=tuple(f"inner-{i}" for i in range(4)))
    arm = Arm("world", "residual")
    kwargs = dict(
        config=config,
        arm=arm,
        train=[train],
        inner=[inner],
        seed=17,
        snapshot_id="synthetic-fold",
        world=True,
        epochs=2,
    )
    train_phase(root=tmp_path / "reference", **kwargs)
    with pytest.raises(RuntimeError, match="intentional_partial"):
        train_phase(root=tmp_path / "recovered", interrupt_after_update=2, **kwargs)
    train_phase(root=tmp_path / "recovered", resume=True, **kwargs)
    expected = torch.load(tmp_path / "reference/final.pt", weights_only=True)
    recovered = torch.load(tmp_path / "recovered/final.pt", weights_only=True)
    for key in ("model_state", "optimizer_state", "rng_state", "epoch", "updates"):
        assert not checkpoint_payload_mismatches(expected[key], recovered[key])
    frozen = Arm("frozen", "residual", "frozen")
    train_phase(
        config,
        frozen,
        [train],
        [inner],
        tmp_path / "frozen",
        seed=17,
        snapshot_id="synthetic-fold",
        world=False,
        epochs=2,
        parent=tmp_path / "reference/selected.pt",
    )
    parent = torch.load(tmp_path / "reference/selected.pt", weights_only=True)
    child = torch.load(tmp_path / "frozen/final.pt", weights_only=True)
    for key, value in parent["model_state"].items():
        if not key.startswith("heads."):
            assert torch.equal(value, child["model_state"][key])
    assert any(
        not torch.equal(value, child["model_state"][key])
        for key, value in parent["model_state"].items()
        if key.startswith("heads.")
    )


@pytest.mark.parametrize(
    "architecture", ["direct", "residual", "legacy_deterministic", "clinical", "ct0"]
)
def test_portable_files_and_runtime_gradients(tmp_path, architecture):
    config, _, _ = binary_fixture()
    batch = spatial_batch()
    inner = replace(batch, patient_ids=tuple(f"inner-{i}" for i in range(4)))
    if architecture in ("clinical", "ct0"):
        model, _ = fit_baseline([batch], [inner], architecture)
        expected = baseline_predict(model, [batch])
    else:
        model = make_model(config, Arm(architecture, architecture))
        expected = predict(model, [batch])["probabilities"]
        assert verify_model_contract(model, batch)["status"] == "passed"
    rows = ct6_rows()
    snapshot = {
        "artifact_id": "synthetic-fold",
        "transform": fit_clinical_transform(rows, set(rows), schema_version=CT6_CLINICAL_SCHEMA),
        "treatment_support": fit_name_support(
            [compact_row(p) for p in batch.patient_ids], set(batch.patient_ids)
        ),
    }
    export_bundle(
        model,
        tmp_path,
        snapshot=snapshot,
        provenance=batch.ct0.provenance,
        weight_version="synthetic-test",
        weights=[1.0, 1.0],
        thresholds=[0.5, 0.5],
    )
    _atomic_torch_save(
        tmp_path / "ct0.pt", ct0_packet(_slice_observation(batch.ct0, torch.tensor([0])))
    )
    atomic_write_private_json(
        tmp_path / "query.json",
        {
            "ct0_features": "ct0.pt",
            "baseline_clinical": rows["person-0"],
            "s0_time_days": float(batch.s0_time[0]),
            "target_interval_days": float(batch.s1_time[0] - batch.s0_time[0]),
            "treatment_scenario": {"description": "synthetic", **treatment_fields(compact_row())},
        },
    )
    assert verify_portable(tmp_path, expected)["status"] == "passed"


def test_warm_policy_unfreezes_on_sixth_epoch(tmp_path):
    config, _, _ = binary_fixture()
    batch = spatial_batch()
    inner = replace(batch, patient_ids=tuple(f"inner-{i}" for i in range(4)))
    kwargs = dict(
        config=config, train=[batch], inner=[inner], seed=17, snapshot_id="synthetic-fold"
    )
    train_phase(
        arm=Arm("world", "residual"), root=tmp_path / "world", world=True, epochs=1, **kwargs
    )
    _, report = train_phase(
        arm=Arm("warm", "residual", "warm"),
        root=tmp_path / "warm",
        world=False,
        epochs=6,
        parent=tmp_path / "world/selected.pt",
        **kwargs,
    )
    final = torch.load(tmp_path / "warm/final.pt", weights_only=True)
    parent = torch.load(tmp_path / "world/selected.pt", weights_only=True)
    assert [r["backbone_frozen"] for r in final["history"]] == [True] * 5 + [False]
    assert not torch.equal(
        parent["model_state"]["blocks.0.film.weight"], final["model_state"]["blocks.0.film.weight"]
    )
    assert report["verification"]["final_score_replayed"]


def test_patient_bootstrap_preserves_seed_pairing():
    batch = binary_batch()
    y, valid = targets([batch])
    row = {
        "patient_ids": list(batch.patient_ids),
        "labels": y,
        "valid": valid,
        "probabilities": torch.tensor([[0.8, 0.2], [0.2, 0.8], [0.3, 0.3], [0.9, 0.9]]),
    }
    result = paired_comparison([row, row], [row, row], replicates=30)
    assert all(r["candidate_minus_reference"] == 0 for r in result)
    assert all(
        r["confidence_interval_95"] == [0.0, 0.0]
        for r in result
        if r["confidence_interval_95"] is not None
    )
    changed = {**row, "patient_ids": list(reversed(row["patient_ids"]))}
    with pytest.raises(ValueError, match="identical patients"):
        paired_comparison([row], [changed], replicates=0)


def test_original_model_coordinate_diagnostic_is_read_only(tmp_path):
    config, _, _ = binary_fixture()
    model = make_model(config, Arm("legacy", "legacy_deterministic"))
    path = tmp_path / "selected.pt"
    _atomic_torch_save(path, {"model_state": model.core.state_dict()})
    before = path.read_bytes()
    report = legacy_dependence(config, [spatial_batch()], path)
    assert path.read_bytes() == before
    assert report["refitted"] is False
    assert report["coordinate_to_CT_projection_norm_ratio"]["mean"] > 0
