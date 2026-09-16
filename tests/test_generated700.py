from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from test_ct6 import compact_row, ct6_rows

from stageworld.binary700_statistics import fit_operating, fit_statistical, logistic_anchor
from stageworld.generated700_data import Pool, fit_inputs
from stageworld.generated700_inference import export_bundle, predict_bundle
from stageworld.generated700_models import (
    AnchoredClassifier,
    FutureWorld,
    endpoint_loss,
    feature_set_loss,
)
from stageworld.generated700_spec import Candidate
from stageworld.generated700_training import infer, train_phase
from stageworld.generated700_workflow import select_residual_scales
from stageworld.training import checkpoint_payload_mismatches


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.set_num_threads(2)
    torch.manual_seed(17)
    clinical = ct6_rows(96)
    ids = list(clinical)
    labels = torch.tensor([[int(i % 4 == 0), int(i % 3 == 0)] for i in range(96)]).float()
    target = torch.randn(96, 27, 12)
    pool = Pool(
        ids,
        clinical,
        {p: compact_row(p) for p in ids},
        torch.full((96,), 60.0),
        torch.randn(96, 27, 12),
        torch.ones(96, dtype=torch.bool),
        target.mean(1),
        torch.ones(96, dtype=torch.bool),
        labels,
        torch.ones_like(labels, dtype=torch.bool),
        "synthetic-generated-v2",
        target,
    )
    pool.valid[5, 0] = False
    x, snapshot = fit_inputs(pool, ids[:32], tmp_path / "inputs.pt")
    train, inner, outer = torch.arange(32), torch.arange(32, 64), torch.arange(64, 96)
    estimators, _ = fit_statistical(
        Candidate("logistic", "logistic"), x, labels, pool.valid, train, inner
    )
    return pool, x, snapshot, train, inner, outer, logistic_anchor(estimators)


@pytest.mark.parametrize("family", ["generated", "generated_frozen"])
def test_future_invariance_gradient_paths_and_missing_images(fixture, family):
    pool, x, _, _, _, outer, anchor = fixture
    rows = outer[:4]
    model = AnchoredClassifier(361, family, *anchor, image_dim=12).eval()
    base = torch.nn.functional.linear(x[rows], *anchor)
    torch.testing.assert_close(model(x[rows], pool.ct0[rows], pool.ct0_valid[rows]), base)
    with torch.no_grad():
        for head in model.endpoints:
            head.body[-1].weight.normal_(std=0.02)
    original = infer(model, x, pool, rows)
    changed = replace(
        pool,
        ct1=pool.ct1 + 100,
        ct1_tokens=pool.ct1_tokens + 100,
        labels=1 - pool.labels,
        valid=~pool.valid,
    )
    assert torch.equal(original, infer(model, x, changed, rows))
    missing = torch.zeros(len(rows), dtype=torch.bool)
    torch.testing.assert_close(model(x[rows], pool.ct0[rows] * float("nan"), missing), base)
    z = model(x[rows], pool.ct0[rows], pool.ct0_valid[rows])
    z[:, 0].sum().backward()

    def has_gradient(module):
        return any(p.grad is not None and bool(p.grad.abs().sum()) for p in module.parameters())

    assert has_gradient(model.endpoints[0])
    assert not has_gradient(model.endpoints[1])
    assert has_gradient(model.world.blocks) == (family == "generated")
    assert has_gradient(model.world.image) == (family == "generated")
    assert model.anchor_weight.grad is None


def test_set_supervision_permutation_detachment_and_distribution():
    torch.manual_seed(17)
    p = torch.randn(4, 27, 12, requires_grad=True)
    target = torch.randn(4, 27, 12, requires_grad=True)
    valid = torch.tensor([True, True, False, True])
    value = feature_set_loss(p, target, valid)
    torch.testing.assert_close(
        value, feature_set_loss(p[:, torch.randperm(27)], target[:, torch.randperm(27)], valid)
    )
    value.backward()
    assert target.grad is None
    assert p.grad[2].abs().sum() == 0
    assert p.grad[0].abs().sum() > 0
    identity = feature_set_loss(target.detach(), target, valid)
    collapsed = feature_set_loss(target.mean(1, keepdim=True).expand_as(target), target, valid)
    assert identity.item() < 1e-6 and collapsed > identity + 0.01


def test_generator_condition_response_and_full_token_gradient(fixture):
    pool, x, _, train, _, _, _ = fixture
    model = FutureWorld(361, 12).eval()
    model.fit_statistics(pool.ct0[train], pool.ct1_tokens[train])
    assert torch.equal(model.input_mean, pool.ct0[train].mean((0, 1)))
    initial, generated, prediction = model(x[:4], pool.ct0[:4])
    assert initial.shape == generated.shape == (4, 27, 128)
    assert prediction.shape == (4, 27, 12)
    altered = x[:4].clone()
    altered[:, 32:360] += 1
    assert not torch.equal(generated, model(altered, pool.ct0[:4])[1])
    feature_set_loss(prediction, pool.ct1_tokens[:4], pool.ct1_valid[:4]).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.blocks.parameters())


@pytest.mark.parametrize("family", ["world", "generated", "generated_frozen"])
def test_exact_recovery(fixture, tmp_path, family):
    pool, x, snapshot, train, inner, _, anchor = fixture
    parent = None
    if family != "world":
        train_phase(
            pool,
            x,
            snapshot,
            Candidate("world", "world"),
            train,
            inner,
            tmp_path / "world",
            seed=17,
            epochs=1,
        )
        parent = tmp_path / "world/selected.pt"
    options = dict(seed=17, epochs=2, anchor=anchor, parent=parent)
    candidate = Candidate(family, family)
    train_phase(pool, x, snapshot, candidate, train, inner, tmp_path / "full", **options)
    with pytest.raises(RuntimeError, match="intentional_partial_epoch"):
        train_phase(
            pool,
            x,
            snapshot,
            candidate,
            train,
            inner,
            tmp_path / "resumed",
            interrupt_after_update=2,
            **options,
        )
    train_phase(pool, x, snapshot, candidate, train, inner, tmp_path / "resumed", **options)
    reference = torch.load(tmp_path / "full/final.pt", weights_only=True)
    actual = torch.load(tmp_path / "resumed/final.pt", weights_only=True)
    for key in ("model_state", "optimizer_state", "rng_state", "epoch", "updates", "early_stop"):
        assert not checkpoint_payload_mismatches(reference[key], actual[key])
    _, repeated = train_phase(
        pool, x, snapshot, candidate, train, inner, tmp_path / "resumed", **options
    )
    assert repeated["updates"] == actual["updates"]


def test_portable_predictions_and_inner_only_shrinkage(fixture, tmp_path, monkeypatch):
    pool, x, snapshot, _, inner, outer, anchor = fixture
    model = AnchoredClassifier(361, "generated", *anchor, image_dim=12).eval()
    with torch.no_grad():
        for head in model.endpoints:
            head.body[-1].weight.normal_(std=0.02)
    scales = select_residual_scales(model, x, pool, inner)
    changed = replace(pool, labels=pool.labels.clone())
    changed.labels[outer] = 1 - changed.labels[outer]
    assert scales == select_residual_scales(model, x, changed, inner)
    rules = fit_operating(
        infer(model, x, pool, inner).numpy(), pool.labels[inner], pool.valid[inner]
    )
    export_bundle(tmp_path / "bundle", snapshot, rules, "generated", model=model, image_dim=12)
    monkeypatch.setattr(
        "stageworld.generated700_data.load_pool", lambda *a: pytest.fail("source data read")
    )
    ids = [pool.ids[i] for i in outer]
    result = predict_bundle(
        tmp_path / "bundle",
        [pool.clinical[p] for p in ids],
        [pool.treatments[p] for p in ids],
        pool.interval[outer],
        pool.ct0[outer],
        pool.ct0_valid[outer],
    )
    np.testing.assert_allclose(result["logits"], infer(model, x, pool, outer), atol=1e-6)


def test_joint_formula_and_masked_target_gradients(fixture):
    pool, x, _, _, _, outer, anchor = fixture
    rows = outer[:4]
    model = AnchoredClassifier(361, "generated", *anchor, image_dim=12).eval()
    z, generated = model.forward_with_features(x[rows], pool.ct0[rows], pool.ct0_valid[rows])
    target = pool.ct1_tokens[rows].clone().requires_grad_()
    classification = endpoint_loss(z, pool.labels[rows], pool.valid[rows], torch.ones(2))
    auxiliary = feature_set_loss(generated, target, pool.ct1_valid[rows])
    loss = classification + 0.1 * auxiliary
    loss.backward()
    assert target.grad is None and torch.isfinite(loss)
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.world.decoder.parameters()
    )
