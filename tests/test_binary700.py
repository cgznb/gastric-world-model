from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy.special import expit
from test_ct6 import compact_row, ct6_rows

from stageworld.binary700_data import Pool, fit_inputs
from stageworld.binary700_evaluation import points
from stageworld.binary700_inference import export_bundle, predict_bundle
from stageworld.binary700_models import AnchoredClassifier, FutureWorld, endpoint_loss, world_loss
from stageworld.binary700_spec import Candidate
from stageworld.binary700_statistics import (
    apply_operating,
    choose_thresholds,
    fit_operating,
    fit_statistical,
    logistic_anchor,
    positive_weights,
    statistical_predict,
)
from stageworld.binary700_training import infer, train_phase
from stageworld.binary_data import make_binary_folds
from stageworld.ct6_training import EarlyStopState


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.set_num_threads(2)
    torch.manual_seed(17)
    clinical = ct6_rows(96)
    ids = list(clinical)
    y = torch.tensor([[int(i % 4 == 0), int(i % 3 == 0)] for i in range(96)]).float()
    valid = torch.ones_like(y, dtype=torch.bool)
    valid[5, 0] = False
    pool = Pool(
        ids,
        clinical,
        {p: compact_row(p) for p in ids},
        torch.full((96,), 60.0),
        torch.randn(96, 27, 12),
        torch.ones(96, dtype=torch.bool),
        torch.randn(96, 12),
        torch.ones(96, dtype=torch.bool),
        y,
        valid,
        "synthetic700",
    )
    x, snapshot = fit_inputs(pool, ids[:32], tmp_path / "inputs.pt")
    train, inner, outer = torch.arange(32), torch.arange(32, 64), torch.arange(64, 96)
    models, _ = fit_statistical(Candidate("logistic", "logistic"), x, y, valid, train, inner)
    return pool, x, snapshot, train, inner, outer, models


@pytest.mark.parametrize("family", ["tabular", "ct", "generated"])
def test_initial_anchor_future_invariance_and_separate_endpoints(fixture, family):
    pool, x, _, _, _, outer, estimators = fixture
    model = AnchoredClassifier(x.shape[1], family, *logistic_anchor(estimators), image_dim=12)
    model.eval()
    expected = torch.from_numpy(statistical_predict(estimators, x.double().numpy())).float()
    torch.testing.assert_close(model(x, pool.ct0, pool.ct0_valid), expected, atol=1e-6, rtol=1e-6)
    with torch.no_grad():
        for head in model.endpoints:
            head.body[-1].weight.fill_(0.1)
    first = infer(model, x, pool, outer)
    changed = replace(
        pool,
        ct1=pool.ct1 + 100,
        labels=1 - pool.labels,
        ct1_valid=~pool.ct1_valid,
        valid=~pool.valid,
    )
    assert torch.equal(first, infer(model, x, changed, outer))
    if family != "tabular":
        missing = torch.zeros(96, dtype=torch.bool)
        torch.testing.assert_close(
            model(x, pool.ct0 * 100, missing), expected, atol=1e-6, rtol=1e-6
        )
    output = model(x, pool.ct0, pool.ct0_valid)
    output[:, 0].sum().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.endpoints[0].parameters()
    )
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.endpoints[1].parameters())
    if model.world is not None:
        assert all(p.grad is None and not p.requires_grad for p in model.world.parameters())


@pytest.mark.parametrize("loss", ["bce", "focal"])
def test_masked_weights_and_loss_gradients(loss):
    z = torch.zeros(4, 2, requires_grad=True)
    y = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    valid = torch.ones_like(y, dtype=torch.bool)
    valid[3, 0] = False
    weight = positive_weights(y, valid, "balanced")
    assert weight.tolist() == [2.0, 3.0]
    actual = endpoint_loss(z, y, valid, weight, loss)
    assert actual.item() == pytest.approx(2 * np.log(2) * (0.25 if loss == "focal" else 1))
    actual.backward()
    assert z.grad[3, 0] == 0
    assert abs(z.grad[0, 0] / z.grad[1, 0]) == pytest.approx(2)


def test_world_targets_detached_and_transition_gradient():
    model = FutureWorld(5, 12)
    with torch.no_grad():
        model.decoder.weight.normal_()
    ct0 = torch.randn(4, 27, 12, requires_grad=True)
    ct1 = torch.randn(4, 12, requires_grad=True)
    world_loss(
        model(torch.randn(4, 5), ct0)[2], ct1, torch.tensor([True, True, False, True])
    ).backward()
    assert ct1.grad is None and ct0.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.blocks.parameters())
    assert ct0.grad[2].abs().sum() == 0


def test_inner_threshold_and_calibration():
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.5, 0.4, 0.3, 0.2, 0.05])
    balanced, sensitive = choose_thresholds(p, y)
    assert balanced == 0.6 and sensitive == 0.6
    logits = np.stack((np.log(p / (1 - p)), -np.log(p / (1 - p))), 1)
    labels = torch.tensor(np.stack((y, y), 1)).float()
    rules = fit_operating(logits, labels, torch.ones_like(labels, dtype=torch.bool))
    assert rules["calibration"][1]["constant_fallback"]
    result = apply_operating(logits, rules)
    assert result["probabilities"].dtype == torch.float64
    assert result["inner_sensitivity_0.8"][:5, 0].float().mean() == pytest.approx(0.8)
    assert np.all(np.diff(result["calibrated"][:, 0].numpy()[np.argsort(logits[:, 0])]) >= 0)


def test_fit_statistics_do_not_read_outer_values(fixture, tmp_path):
    pool, x, snapshot, train, inner, outer, estimators = fixture
    changed = copy.deepcopy(pool)
    for row in outer.tolist():
        changed.clinical[pool.ids[row]] = pool.clinical[pool.ids[0]]
        changed.treatments[pool.ids[row]] = compact_row(pool.ids[row], drugs={"oxaliplatin": 1})
    changed.interval[outer] = 10000
    new_x, new_snapshot = fit_inputs(changed, snapshot["fit_ids"], tmp_path / "other.pt")
    for key in ("mean", "scale"):
        assert torch.equal(snapshot[key], new_snapshot[key])
    assert snapshot["support"] == new_snapshot["support"]
    new_y = pool.labels.clone()
    new_y[outer] = 1 - new_y[outer]
    other, _ = fit_statistical(
        Candidate("logistic", "logistic"), new_x, new_y, pool.valid, train, inner
    )
    np.testing.assert_array_equal(
        statistical_predict(estimators, x[inner].double().numpy()),
        statistical_predict(other, new_x[inner].double().numpy()),
    )


@pytest.mark.parametrize("family", ["world", "tabular"])
def test_exact_recovery_and_fixed_refit(fixture, tmp_path, family):
    pool, x, snapshot, train, inner, _, estimators = fixture
    candidate = Candidate(family, family)
    kwargs = {"seed": 17, "epochs": 2, "anchor": logistic_anchor(estimators)}
    model, report = train_phase(
        pool, x, snapshot, candidate, train, inner, tmp_path / "full", **kwargs
    )
    with pytest.raises(RuntimeError, match="intentional_partial_epoch"):
        train_phase(
            pool,
            x,
            snapshot,
            candidate,
            train,
            inner,
            tmp_path / "recovery",
            interrupt_after_update=2,
            **kwargs,
        )
    recovered, resumed = train_phase(
        pool, x, snapshot, candidate, train, inner, tmp_path / "recovery", **kwargs
    )
    assert resumed["updates"] == report["updates"] == 2
    for key, value in model.state_dict().items():
        assert torch.equal(value, recovered.state_dict()[key])
    _, fixed = train_phase(pool, x, snapshot, candidate, train, None, tmp_path / "refit", **kwargs)
    assert fixed["selected_epoch"] == fixed["completed_epochs"] == 2
    _, again = train_phase(pool, x, snapshot, candidate, train, None, tmp_path / "refit", **kwargs)
    assert again["updates"] == fixed["updates"]
    bad = {**snapshot, "fit_ids": snapshot["fit_ids"] + [pool.ids[50]]}
    with pytest.raises(ValueError, match="partitions"):
        train_phase(pool, x, bad, candidate, train, inner, tmp_path / "bad", **kwargs)


@pytest.mark.parametrize("family", ["logistic", "tabular", "ct", "generated"])
def test_portable_without_future_files(fixture, tmp_path, family, monkeypatch):
    pool, x, snapshot, _, inner, outer, estimators = fixture
    rules = fit_operating(
        statistical_predict(estimators, x[inner].double().numpy()),
        pool.labels[inner],
        pool.valid[inner],
    )
    if family == "logistic":
        model = None
        expected = expit(statistical_predict(estimators, x[outer].double().numpy()))
    else:
        model = AnchoredClassifier(
            x.shape[1], family, *logistic_anchor(estimators), image_dim=12
        ).eval()
        expected = expit(infer(model, x, pool, outer).double().numpy())
    export_bundle(tmp_path, snapshot, rules, family, model=model, models=estimators, image_dim=12)
    ids = [pool.ids[i] for i in outer.tolist()]
    import builtins

    original = builtins.open

    def restricted(file, *args, **kwargs):
        if any(s in str(file) for s in ("pool.pt", "ct1", "labels", ".xlsx", "bindings")):
            raise AssertionError("Future/data file read during inference")
        return original(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", restricted)
    result = predict_bundle(
        tmp_path,
        [pool.clinical[p] for p in ids],
        [pool.treatments[p] for p in ids],
        pool.interval[outer],
        pool.ct0[outer],
        pool.ct0_valid[outer],
    )
    np.testing.assert_allclose(result["probabilities"], expected, atol=1e-6)


def test_all700_isolated_folds_and_raw_best():
    labels = {f"synthetic-{i}": [i % 2, (i // 2) % 2] for i in range(700)}
    folds = make_binary_folds(labels)
    seen = []
    for fold in folds["folds"]:
        groups = fold["patient_ids"]
        assert [len(groups[k]) for k in ("train", "validation", "outer")] == [448, 112, 140]
        assert not (set(groups["train"] + groups["validation"]) & set(groups["outer"]))
        seen.extend(groups["outer"])
    assert len(set(seen)) == 700
    stop = EarlyStopState()
    assert stop.update(1.0, 1, 0.0001)
    assert stop.update(0.99999, 2, 0.0001)
    assert stop.selected_epoch == 2 and stop.stale_epochs == 1


def test_metrics_distinguish_majority_accuracy_and_detection():
    y = np.r_[np.ones(2), np.zeros(8)]
    result = points(np.full(10, 0.1), y, np.zeros(10, dtype=bool))
    assert result["accuracy"] == 0.8 and result["sensitivity"] == 0
    assert result["specificity"] == 1 and result["precision"] is None
