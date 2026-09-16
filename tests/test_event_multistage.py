from __future__ import annotations

import copy
import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_ct6 import compact_row, ct6_rows

from stageworld.artifacts import read_json
from stageworld.ct6_training import EarlyStopState
from stageworld.event_data import EventPool, event_status, fit_inputs
from stageworld.event_evaluation import METRICS, binary_metrics, summarize
from stageworld.event_inference import export_bundle
from stageworld.event_models import EventInputs, EventModel, masked_bce
from stageworld.event_spec import ABSENT, CONFLICT, PRESENT, UNKNOWN
from stageworld.event_training import (
    has_supervision,
    infer,
    objective,
    recurrence_weight,
    train_phase,
)
from stageworld.event_verification import predict_without_sources, verify_study
from stageworld.event_workflow import run_study, validate_folds
from stageworld.generated651_data import make_folds
from stageworld.generated700_data import Pool
from stageworld.generated700_models import feature_set_loss


def synthetic_pool(n=80):
    clinical = ct6_rows(n)
    ids = list(clinical)
    labels = torch.tensor([[int(i % 4 == 0), int(i % 3 == 0)] for i in range(n)]).float()
    ct0 = torch.randn(n, 27, 8)
    ct1 = ct0 * 0.8 + torch.randn_like(ct0) * 0.2
    base = Pool(
        ids,
        clinical,
        {p: compact_row(p) for p in ids},
        torch.full((n,), 60.0),
        ct0,
        torch.ones(n, dtype=torch.bool),
        ct1.mean(1),
        torch.ones(n, dtype=torch.bool),
        labels,
        torch.ones_like(labels, dtype=torch.bool),
        "synthetic-event-base",
        ct1,
    )
    return EventPool(base, torch.full((n, 3), PRESENT, dtype=torch.long), "synthetic-event-input")


@pytest.fixture
def pool(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_num_threads(2)
    torch.manual_seed(17)
    return synthetic_pool()


def fitted(pool, path):
    train, validation = torch.arange(64), torch.arange(64, 80)
    x, snapshot = fit_inputs(pool, pool.base.ids[:64], path)
    return x, snapshot, train, validation


def test_prefix_invariance_and_event_identity(pool):
    model = EventModel(8, 16, 1).eval()
    inputs = EventInputs(torch.randn(4, 360), pool.base.ct0[:4], pool.events[:4].clone())
    first = model(inputs)
    changed = replace(inputs, events=inputs.events.clone())
    changed.events[:, 2] = ABSENT
    second = model(changed)
    assert torch.equal(first.states[:, :3], second.states[:, :3])
    assert torch.equal(first.pcr_logit, second.pcr_logit)
    assert torch.equal(first.ct1, second.ct1)
    assert not torch.equal(first.recurrence_logit, second.recurrence_logit)
    changed.events[:, 1:] = ABSENT
    third = model(changed)
    assert torch.equal(third.states[:, 1], third.states[:, 2])
    assert torch.equal(third.states[:, 2], third.states[:, 3])
    changed.x = inputs.x.clone()
    changed.x[:, 32:] += 100
    fourth = model(changed)
    assert torch.equal(third.states[:, 0], fourth.states[:, 0])
    assert not torch.equal(third.states[:, 1], fourth.states[:, 1])
    assert third.last_stage.tolist() == [1] * 4


def grad_norm(parameters):
    return sum(float(p.grad.abs().sum()) for p in parameters if p.grad is not None)


def test_terminal_and_intermediate_gradient_routes(pool):
    model = EventModel(8, 16, 1).eval()
    inputs = EventInputs(torch.randn(4, 360), pool.base.ct0[:4], pool.events[:4])
    model(inputs).recurrence_logit.sum().backward()
    assert grad_norm(model.world.image.parameters()) > 0
    assert grad_norm(model.world.blocks.parameters()) > 0
    assert all(grad_norm(embedding.parameters()) > 0 for embedding in model.world.event_codes)
    assert grad_norm(model.pcr_head.parameters()) == 0
    model.zero_grad(set_to_none=True)
    model(inputs).pcr_logit.sum().backward()
    assert grad_norm(model.world.event_codes[0].parameters()) > 0
    assert grad_norm(model.world.event_codes[1].parameters()) > 0
    assert grad_norm(model.world.event_codes[2].parameters()) == 0
    assert grad_norm(model.recurrence_head.parameters()) == 0


@pytest.mark.parametrize("status", [ABSENT, UNKNOWN, CONFLICT])
def test_missing_events_preserve_state_and_expose_incomplete_history(pool, status):
    model = EventModel(8, 16, 1).eval()
    events = torch.full((4, 3), status, dtype=torch.long)
    output = model(EventInputs(torch.randn(4, 360), pool.base.ct0[:4], events))
    assert all(torch.equal(output.states[:, 0], output.states[:, k]) for k in (1, 2, 3))
    assert not output.last_stage.any()
    assert output.incomplete_history.tolist() == [status in (UNKNOWN, CONFLICT)] * 4


def test_event_policy_and_masked_pathology(pool):
    assert event_status(0, 1) == (ABSENT, CONFLICT)
    assert event_status(None, 1) == (UNKNOWN, CONFLICT)
    assert event_status(1, 1) == (PRESENT, PRESENT)
    assert event_status(1, None) == (PRESENT, UNKNOWN)
    pool.events[0, 1:] = torch.tensor([ABSENT, CONFLICT])
    assert not pool.targets(torch.arange(2), torch.device("cpu"))["valid"][0, 0]
    invalid = EventInputs(
        torch.randn(2, 360), pool.base.ct0[:2], torch.tensor([[1, 0, 1], [1, 1, 1]])
    )
    with pytest.raises(ValueError, match="requires confirmed surgery"):
        EventModel(8, 16, 1)(invalid)


def test_loss_formulas_target_detach_and_missing_nans(pool):
    model = EventModel(8, 16, 1)
    output = model(EventInputs(torch.randn(4, 360), pool.base.ct0[:4], pool.events[:4]))
    targets = pool.targets(torch.arange(4), torch.device("cpu"))
    targets["ct1"][0] = torch.nan
    targets["ct_valid"][0] = False
    targets["labels"][1] = torch.nan
    targets["valid"][1] = False
    targets["ct1"].requires_grad_()
    targets["labels"].requires_grad_()
    pretrain, _ = objective(output, targets, "pretrain", 3.0)
    joint, pieces = objective(output, targets, "joint", 3.0)
    ct = feature_set_loss(output.ct1, targets["ct1"], targets["ct_valid"])
    pcr = masked_bce(output.pcr_logit, targets["labels"][:, 0], targets["valid"][:, 0])
    y = targets["labels"][targets["valid"][:, 1], 1].detach()
    z = output.recurrence_logit[targets["valid"][:, 1]]
    w = torch.where(y == 1, 3.0, 1.0)
    recurrence = (
        torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none") * w
    ).sum() / w.sum()
    assert torch.equal(pretrain, ct + 0.5 * pcr)
    assert torch.allclose(pieces["recurrence_weighted_bce"], recurrence)
    assert torch.equal(joint, recurrence + 0.5 * pcr + 0.1 * ct)
    joint.backward()
    assert targets["ct1"].grad is None and targets["labels"].grad is None
    targets["valid"][:] = False
    targets["ct_valid"][:] = False
    assert not has_supervision(targets, "joint")
    assert objective(output, targets, "joint", 3.0)[0] == 0


def test_train_only_statistics_and_weights(pool, tmp_path):
    x, snapshot, train, validation = fitted(pool, tmp_path / "first.pt")
    altered = copy.deepcopy(pool)
    for row in validation.tolist():
        altered.base.clinical[altered.base.ids[row]] = ct6_rows(1)["person-0"]
    altered.base.ct0[validation] += 100
    altered.base.ct1_tokens[validation] += 200
    altered.base.labels[validation] = 1
    changed_x, changed_snapshot = fit_inputs(altered, pool.base.ids[:64], tmp_path / "second.pt")
    assert torch.equal(x[train], changed_x[train])
    assert snapshot["clinical"] == changed_snapshot["clinical"]
    assert torch.equal(snapshot["mean"], changed_snapshot["mean"])
    assert recurrence_weight(pool, train) == recurrence_weight(altered, train)
    assert x.shape[1] == 360
    models = []
    for current in (pool, altered):
        torch.manual_seed(17)
        model = EventModel(8, 16, 1)
        model.world.fit_statistics(
            current.base.ct0[train], current.base.ct1_tokens[train], current.base.ct1_valid[train]
        )
        models.append(model)
    assert all(
        torch.equal(value, models[1].state_dict()[key])
        for key, value in models[0].state_dict().items()
    )


def test_input_only_bundle_and_target_replacement(pool, tmp_path):
    x, snapshot, _, validation = fitted(pool, tmp_path / "inputs.pt")
    model = EventModel(8, 16, 1).eval()
    original = infer(model, x, pool, validation)
    pool.base.ct1_tokens.fill_(torch.nan)
    pool.base.labels.fill_(torch.nan)
    pool.base.interval.fill_(torch.nan)
    altered = infer(model, x, pool, validation)
    assert torch.equal(original["logits"], altered["logits"])
    export_bundle(tmp_path / "bundle", snapshot, model)
    ids = pool.base.ids[64:]
    replay = predict_without_sources(
        tmp_path / "bundle",
        [pool.base.clinical[p] for p in ids],
        [pool.base.treatments[p] for p in ids],
        pool.events[validation],
        pool.base.ct0[validation],
    )
    assert torch.equal(replay["probabilities"], original["probabilities"])
    assert replay["pcr_applicable"].all()
    assert torch.equal(replay["recurrence_probability"], original["probabilities"][:, 1])


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_tree_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("phase", ["pretrain", "joint"])
def test_exact_partial_epoch_recovery_and_completed_resume(pool, tmp_path, phase):
    x, snapshot, train, validation = fitted(pool, tmp_path / "inputs.pt")
    kwargs = dict(phase=phase, seed=17, epochs=3, hidden=16, layers=1)
    if phase == "joint":
        train_phase(
            pool,
            x,
            snapshot,
            train,
            validation,
            tmp_path / "parent",
            phase="pretrain",
            seed=17,
            epochs=2,
            hidden=16,
            layers=1,
        )
        kwargs["parent"] = tmp_path / "parent/selected.pt"
    train_phase(pool, x, snapshot, train, validation, tmp_path / "whole", **kwargs)
    with pytest.raises(RuntimeError, match="intentional_partial_epoch"):
        train_phase(
            pool,
            x,
            snapshot,
            train,
            validation,
            tmp_path / "resumed",
            interrupt_after_update=3,
            **kwargs,
        )
    train_phase(pool, x, snapshot, train, validation, tmp_path / "resumed", **kwargs)
    whole = torch.load(tmp_path / "whole/latest.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed/latest.pt", weights_only=True)
    for key in ("model_state", "optimizer_state", "rng_state", "history", "early_stop", "updates"):
        assert_tree_equal(whole[key], resumed[key])
    selected_id = torch.load(tmp_path / "resumed/selected.pt", weights_only=True)["artifact_id"]
    _, report = train_phase(pool, x, snapshot, train, validation, tmp_path / "resumed", **kwargs)
    assert report["updates"] == resumed["updates"]
    assert (
        selected_id
        == torch.load(tmp_path / "resumed/selected.pt", weights_only=True)["artifact_id"]
    )
    assert report["checkpoint_score_replay"]


def test_early_stop_selection_and_missing_metric_support():
    stopper = EarlyStopState()
    assert stopper.update(0.5, 1, 0.0001)
    assert stopper.update(0.49999, 2, 0.0001)
    assert stopper.selected_epoch == 2 and stopper.stale_epochs == 1
    actual = binary_metrics(np.array([0.9, 0.7, 0.6, 0.1]), np.array([1, 0, 1, 0]))
    assert actual["sensitivity"] == 1 and actual["precision"] == 2 / 3
    assert actual["f1"] == 0.8 and actual["npv"] == 1
    empty = binary_metrics(np.array([]), np.array([]))
    assert empty["n"] == 0 and all(empty[k] is None for k in METRICS)
    records = [
        {"seed": seed, "endpoint": "recurrence", "fold": fold, **dict.fromkeys(METRICS, value)}
        for seed, values in ((17, [0.1, 0.2, 0.3, 0.4, 0.5]), (43, [0.8] * 5))
        for fold, value in enumerate(values)
    ]
    result = summarize(records)
    assert result[0]["mean"] == pytest.approx(0.3)
    assert result[0]["sample_standard_deviation"] == pytest.approx(math.sqrt(0.025))
    records[0]["precision"] = None
    assert (
        next(r for r in summarize(records) if r["seed"] == 17 and r["metric"] == "precision")[
            "mean"
        ]
        is None
    )


def test_saved_folds_are_reused_and_overlap_rejected(pool):
    folds = make_folds(pool.base)
    validate_folds(pool, folds)
    folds["folds"][0]["patient_ids"]["train"].append(
        folds["folds"][0]["patient_ids"]["validation"][0]
    )
    with pytest.raises(ValueError, match="disjoint"):
        validate_folds(pool, folds)


def test_synthetic_full_workflow_and_independent_verification(pool, tmp_path, monkeypatch):
    from stageworld import event_workflow

    # Exercise the formal path with artificial inputs; no clinical data or small real trial.
    artificial = synthetic_pool(651)
    original_spec = event_workflow.specification
    monkeypatch.setattr(event_workflow, "specification", lambda: {**original_spec(), "seeds": [17]})
    original_train = event_workflow.train_phase

    def two_epochs(*args, **kwargs):
        return original_train(*args, **kwargs, epochs=2, hidden=8, layers=1)

    monkeypatch.setattr(event_workflow, "train_phase", two_epochs)
    folds = make_folds(artificial.base)
    result = run_study(artificial, folds, tmp_path / "formal")
    assert result["phases"] == 10 and result["complete_seeds"] == [17]
    verification = verify_study(artificial, folds, tmp_path / "formal")
    assert verification["checkpoint_score_replays"] == 20
    assert verification["denied_source_bundles"] == 5
    summary = read_json(tmp_path / "formal/evaluation/summary.json")
    assert len(summary["per_seed_fold_summary"]) == len(METRICS) * 2
    assert not summary["across_seed_aggregation"]
