from __future__ import annotations

import builtins
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from test_event_multistage import assert_tree_equal, synthetic_pool
from torch.nn import functional as F

from stageworld import event_v2_workflow
from stageworld.artifacts import atomic_write_private_json, read_json
from stageworld.event_data import fit_inputs
from stageworld.event_models import EventModel
from stageworld.event_v2_inference import export_bundle, predict_bundle
from stageworld.event_v2_models import EventV2Model, mean_probability_logit
from stageworld.event_v2_spec import model_dimensions, specification
from stageworld.event_v2_splits import make_split
from stageworld.event_v2_training import build_model, infer, objective, train_phase
from stageworld.event_v2_verification import _compare_test_replay, verify_study
from stageworld.event_v2_workflow import (
    development_partition,
    evaluate_tests,
    freeze_selection,
    prepare_splits,
    run_study,
)
from stageworld.generated700_models import feature_set_loss


@pytest.fixture
def pool(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_num_threads(1)
    torch.manual_seed(17)
    return synthetic_pool(160)


def dimensions(family):
    result = {**model_dimensions(family, 8, hidden=16), "layers": 1}
    if family == "event_v2":
        result.update(rank=3, members=2)
    return result


def fitted_development(pool, path, split=None):
    split = make_split(pool, 17) if split is None else split
    groups = split["patient_ids"]
    dev = development_partition(pool, split)
    x, snapshot = fit_inputs(dev, groups["train"], path)
    train, validation = (dev.base.indices(groups[name]) for name in ("train", "validation"))
    return dev, x, snapshot, train, validation


@pytest.mark.parametrize("family", ["event_v1", "event_v2"])
@pytest.mark.parametrize("phase", ["pretrain", "joint"])
def test_partial_epoch_recovery_and_completed_resume(pool, tmp_path, monkeypatch, family, phase):
    dev, x, snapshot, train, validation = fitted_development(pool, tmp_path / "inputs.pt")
    kwargs = {
        "family": family,
        "phase": phase,
        "seed": 17,
        "epochs": 2,
        "dimensions": dimensions(family),
    }
    if phase == "joint":
        train_phase(
            dev,
            x,
            snapshot,
            train,
            validation,
            tmp_path / "parent",
            **{**kwargs, "phase": "pretrain", "epochs": 1},
        )
        kwargs["parent"] = tmp_path / "parent/selected.pt"
    train_phase(dev, x, snapshot, train, validation, tmp_path / "whole", **kwargs)
    with pytest.raises(RuntimeError, match="intentional_partial_epoch_interruption"):
        train_phase(
            dev,
            x,
            snapshot,
            train,
            validation,
            tmp_path / "resumed",
            interrupt_after_update=6,
            **kwargs,
        )
    interrupted = torch.load(tmp_path / "resumed/latest.pt", weights_only=True)
    assert interrupted["epoch"] == 1 and interrupted["updates"] == 4
    train_phase(dev, x, snapshot, train, validation, tmp_path / "resumed", **kwargs)
    whole = torch.load(tmp_path / "whole/latest.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed/latest.pt", weights_only=True)
    for key in ("model_state", "optimizer_state", "rng_state", "history", "early_stop", "updates"):
        assert_tree_equal(whole[key], resumed[key])
    checkpoints = [tmp_path / "resumed" / f"{name}.pt" for name in ("selected", "latest", "final")]
    original = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in checkpoints}

    def unexpected_update(*args, **kwargs):
        raise AssertionError("Completed recovery must not update the optimizer")

    monkeypatch.setattr(torch.optim.AdamW, "step", unexpected_update)
    _, report = train_phase(dev, x, snapshot, train, validation, tmp_path / "resumed", **kwargs)
    assert report["updates"] == 8 and report["checkpoint_score_replay"]
    assert not report["test_rows_available_to_trainer"]
    for path, expected in original.items():
        assert (path.stat().st_mtime_ns, path.read_bytes()) == expected


def test_member_bce_objective_and_detached_targets():
    members = torch.tensor(
        [[[0.0, 8.0], [1.0, -5.0]], [[-4.0, 1.0], [0.0, 7.0]]], requires_grad=True
    )
    output = SimpleNamespace(
        ct1=torch.randn(2, 27, 8, requires_grad=True),
        pcr_logit=mean_probability_logit(members[:, 0]),
        recurrence_logit=mean_probability_logit(members[:, 1]),
        member_logits=members,
    )
    labels = torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True)
    targets = {
        "ct1": torch.randn(2, 27, 8, requires_grad=True),
        "ct_valid": torch.ones(2, dtype=torch.bool),
        "labels": labels,
        "valid": torch.ones(2, 2, dtype=torch.bool),
    }
    joint, components = objective(output, targets, "joint", 3.0)
    ct = feature_set_loss(output.ct1, targets["ct1"], targets["ct_valid"])
    pcr = F.binary_cross_entropy_with_logits(members[:, 0], labels[:, 0, None].expand(-1, 2))
    recurrence_raw = F.binary_cross_entropy_with_logits(
        members[:, 1], labels[:, 1, None].expand(-1, 2), reduction="none"
    )
    weight = torch.tensor([3.0, 1.0])[:, None]
    recurrence = ((recurrence_raw * weight).sum(0) / weight.sum()).mean()
    torch.testing.assert_close(components["pcr_bce"], pcr)
    torch.testing.assert_close(components["recurrence_weighted_bce"], recurrence)
    torch.testing.assert_close(joint, recurrence + 0.5 * pcr + 0.1 * ct)
    assert not torch.allclose(
        pcr, F.binary_cross_entropy_with_logits(members[:, 0].mean(-1), labels[:, 0])
    )
    pretrain, _ = objective(output, targets, "pretrain", 3.0)
    torch.testing.assert_close(pretrain, ct + 0.5 * pcr)
    joint.backward()
    assert labels.grad is None and targets["ct1"].grad is None
    assert members.grad is not None and torch.isfinite(members.grad).all()
    with pytest.raises(ValueError, match="phase"):
        objective(output, targets, "test", 1.0)


def test_holdout_changes_cannot_reach_development_fit(pool, tmp_path):
    split = make_split(pool, 17)
    changed = copy.deepcopy(pool)
    rows = pool.base.indices(split["patient_ids"]["test"])
    changed.base.ct0[rows] += 1000
    changed.base.ct1_tokens[rows] -= 500
    changed.base.labels[rows] = 1 - changed.base.labels[rows]
    changed.base.interval[rows] = torch.nan
    runs = []
    for index, source in enumerate((pool, changed)):
        dev, x, snapshot, train, validation = fitted_development(
            source, tmp_path / f"inputs-{index}.pt", split
        )
        assert set(dev.base.ids).isdisjoint(split["patient_ids"]["test"])
        network, report = train_phase(
            dev,
            x,
            snapshot,
            train,
            validation,
            tmp_path / f"fit-{index}",
            family="event_v2",
            phase="pretrain",
            seed=17,
            epochs=1,
            dimensions=dimensions("event_v2"),
        )
        runs.append((x, snapshot, network, report))
    assert torch.equal(runs[0][0], runs[1][0])
    for key in ("clinical", "mean", "scale", "fit_ids"):
        if key in runs[0][1]:
            assert_tree_equal(runs[0][1][key], runs[1][1][key])
    assert_tree_equal(runs[0][2].state_dict(), runs[1][2].state_dict())
    assert runs[0][3]["selected_score"] == runs[1][3]["selected_score"]


def test_trainer_rejects_full_pool_and_invalid_checkpoint_parent(pool, tmp_path):
    split = make_split(pool, 17)
    dev, x, snapshot, train, validation = fitted_development(pool, tmp_path / "inputs.pt", split)
    groups = split["patient_ids"]
    with pytest.raises(ValueError, match="development pool"):
        train_phase(
            pool,
            x,
            snapshot,
            pool.base.indices(groups["train"]),
            pool.base.indices(groups["validation"]),
            tmp_path / "invalid",
            family="event_v2",
            phase="pretrain",
            seed=17,
            epochs=1,
            dimensions=dimensions("event_v2"),
        )
    with pytest.raises(ValueError, match="pretraining checkpoint"):
        train_phase(
            dev,
            x,
            snapshot,
            train,
            validation,
            tmp_path / "orphan",
            family="event_v2",
            phase="joint",
            seed=17,
            epochs=1,
            dimensions=dimensions("event_v2"),
        )
    train_phase(
        dev,
        x,
        snapshot,
        train,
        validation,
        tmp_path / "parent",
        family="event_v2",
        phase="pretrain",
        seed=17,
        epochs=1,
        dimensions=dimensions("event_v2"),
    )
    with pytest.raises(ValueError, match="parent differs"):
        train_phase(
            dev,
            x,
            snapshot,
            train,
            validation,
            tmp_path / "mismatched",
            family="event_v2",
            phase="joint",
            seed=43,
            epochs=1,
            dimensions=dimensions("event_v2"),
            parent=tmp_path / "parent/selected.pt",
        )


@pytest.mark.parametrize("family", ["event_v1", "event_v2"])
def test_bundle_inference_denies_sources_and_ignores_targets(pool, tmp_path, family):
    dev, x, snapshot, _, validation = fitted_development(pool, tmp_path / "inputs.pt")
    network = build_model(family, dimensions(family)).eval()
    expected = infer(network, x, dev, validation)
    dev.base.ct1_tokens.fill_(torch.nan)
    dev.base.labels.fill_(torch.nan)
    dev.base.interval.fill_(torch.nan)
    repeated = infer(network, x, dev, validation)
    assert_tree_equal(expected, repeated)
    root = tmp_path / "bundle"
    export_bundle(root, snapshot, network, family, "synthetic-selected-checkpoint")
    bundle = torch.load(root / "inference.pt", weights_only=True)
    assert "fit_ids" not in bundle["inputs"]
    assert not bundle["ct1_required"] and not bundle["outcomes_required"]
    ids = [dev.base.ids[i] for i in validation.tolist()]
    original_open = builtins.open
    allowed = (root / "inference.pt").resolve()
    accessed = []

    def only_bundle(path, *args, **kwargs):
        if not isinstance(path, (str, Path)) or Path(path).resolve() != allowed:
            raise PermissionError("Inference attempted to read an unapproved source")
        accessed.append(Path(path).resolve())
        return original_open(path, *args, **kwargs)

    with patch("builtins.open", only_bundle), patch("io.open", only_bundle):
        actual = predict_bundle(
            root,
            [dev.base.clinical[p] for p in ids],
            [dev.base.treatments[p] for p in ids],
            dev.events[validation],
            dev.base.ct0[validation],
        )
    assert accessed and all(path == allowed for path in accessed)
    for key in ("ct1", "probabilities", "member_logits", "member_disagreement"):
        assert torch.equal(actual[key], expected[key])
    assert torch.equal(actual["decisions"], expected["probabilities"] >= 0.5)
    assert actual["pcr_applicable"].all()
    with pytest.raises(ValueError, match="empty"):
        infer(network, x, dev, validation[:0])


def test_family_builders_and_invalid_family():
    assert isinstance(build_model("event_v1", dimensions("event_v1")), EventModel)
    for family in ("event_v2", "event_v2_no_adapter", "event_v2_no_mask", "event_v2_single_member"):
        network = build_model(family, model_dimensions(family, 8, hidden=16))
        assert isinstance(network, EventV2Model)
        if family == "event_v2_no_adapter":
            assert all(not block.adapters for block in network.world.blocks)
        if family == "event_v2_single_member":
            assert network.members == 1
    with pytest.raises(ValueError, match="family"):
        build_model("unknown", dimensions("event_v1"))


def test_final_test_gate_rejects_unfrozen_or_incomplete_study(pool, tmp_path):
    spec = {**specification(), "patients": 160, "seeds": [17, 43], "max_epochs": 1}
    prepare_splits(pool, tmp_path, spec)
    with pytest.raises(ValueError, match="not been frozen"):
        evaluate_tests(pool, tmp_path, spec)
    with pytest.raises(ValueError, match="not all development fits"):
        freeze_selection(tmp_path, spec)
    with pytest.raises(ValueError, match="entire locked experiment"):
        freeze_selection(tmp_path, {**spec, "seeds": [17], "families": ["event_v2"]})
    assert not (tmp_path / "selection_frozen.json").exists()
    assert not (tmp_path / "test").exists()


def replay_payload(member_logits):
    probabilities = member_logits.double().sigmoid()
    return {
        "member_logits": member_logits,
        "probabilities": probabilities.mean(-1),
        "member_disagreement": probabilities.var(-1, unbiased=False),
        "ct1": torch.zeros(len(member_logits), 27, 8),
        "last_stage": torch.full((len(member_logits),), 3, dtype=torch.long),
        "incomplete_history": torch.zeros(len(member_logits), dtype=torch.bool),
    }


def test_cross_device_replay_checks_probability_without_weakening_same_device_logits():
    saved = replay_payload(torch.full((1, 2, 1), 0.01))
    replay = replay_payload(saved["member_logits"] + 1.14e-5)
    with pytest.raises(ValueError, match="member_logits"):
        _compare_test_replay(replay, saved, cross_device=False)
    errors = _compare_test_replay(replay, saved, cross_device=True)
    assert 0 < errors["member_probabilities"] < 3e-6
    assert errors["ct1"] == 0


def test_cross_device_member_changes_cannot_cancel_in_average():
    saved = replay_payload(torch.tensor([[[-2.0, 2.0], [-1.0, 1.0]]]))
    replay = replay_payload(saved["member_logits"].flip(-1))
    assert torch.equal(replay["probabilities"], saved["probabilities"])
    assert torch.equal(replay["member_disagreement"], saved["member_disagreement"])
    with pytest.raises(ValueError, match="member_probabilities"):
        _compare_test_replay(replay, saved, cross_device=True)


@pytest.mark.parametrize("side", ["replay", "saved"])
@pytest.mark.parametrize("corruption", ["nan", "inf", "shape"])
def test_cross_device_member_replay_rejects_nonfinite_and_shape(side, corruption):
    saved = replay_payload(torch.zeros(1, 2, 2))
    replay = copy.deepcopy(saved)
    target = replay if side == "replay" else saved
    if corruption == "shape":
        target["member_logits"] = target["member_logits"][..., :1]
    else:
        target["member_logits"][0, 0, 0] = float(corruption)
    with pytest.raises(ValueError, match="invalid shapes or nonfinite"):
        _compare_test_replay(replay, saved, cross_device=True)


@pytest.mark.parametrize("key", ["probabilities", "member_disagreement", "ct1"])
def test_cross_device_replay_keeps_existing_float_thresholds_and_rejects_infinity(key):
    saved = replay_payload(torch.zeros(1, 2, 2))
    replay = copy.deepcopy(saved)
    replay[key].fill_(0.01)
    with pytest.raises(ValueError, match=key):
        _compare_test_replay(replay, saved, cross_device=True)
    saved[key].fill_(torch.inf)
    replay[key].fill_(torch.inf)
    with pytest.raises(ValueError, match=key):
        _compare_test_replay(replay, saved, cross_device=True)


def test_two_seed_full_workflow_freezes_every_fit_before_testing(pool, tmp_path, monkeypatch):
    root = tmp_path / "study"
    spec = {**specification(), "patients": 160, "seeds": [17, 43], "max_epochs": 1}
    original_fit, original_infer = event_v2_workflow.train_phase, event_v2_workflow.infer
    calls = {"fits": 0, "test_predictions": 0}

    def observed_fit(*args, **kwargs):
        assert not (root / "selection_frozen.json").exists()
        assert not (root / "test").exists()
        assert len(args[0].base.ids) == 144
        result = original_fit(*args, **kwargs)
        calls["fits"] += 1
        return result

    def observed_test(*args, **kwargs):
        assert calls["fits"] == 8
        frozen = read_json(root / "selection_frozen.json")
        assert len(frozen["selected_checkpoints"]) == 8
        assert len(args[2].base.ids) == 16
        calls["test_predictions"] += 1
        return original_infer(*args, **kwargs)

    monkeypatch.setattr(event_v2_workflow, "train_phase", observed_fit)
    monkeypatch.setattr(event_v2_workflow, "infer", observed_test)
    result = run_study(
        pool,
        root,
        spec=spec,
        dimensions={family: dimensions(family) for family in spec["families"]},
    )
    assert calls == {"fits": 8, "test_predictions": 4}
    assert result["status"] == "completed" and result["reporting_partition"] == "test"
    assert not result["across_seed_patient_pooling"] and not result["best_seed_selection"]
    assert not result["independent_external_validation"]
    assert len(result["test_metrics"]) == 16
    test_ids = []
    for seed in spec["seeds"]:
        groups = read_json(root / f"partitions/seed-{seed}/split.json")["patient_ids"]
        test_ids.append(groups["test"])
        for family in spec["families"]:
            predictions = torch.load(
                root / f"test/{family}/seed-{seed}/predictions.pt", weights_only=True
            )
            assert predictions["patient_ids"] == groups["test"]
            assert predictions["probabilities"].shape == (16, 2)
            assert predictions["reporting_partition"] == "test"
    assert test_ids[0] != test_ids[1]
    original_files = {
        path: (path.stat().st_mtime_ns, path.read_bytes()) for path in root.rglob("*.pt")
    }
    repeated = run_study(
        pool,
        root,
        spec=spec,
        dimensions={family: dimensions(family) for family in spec["families"]},
    )
    assert calls == {"fits": 8, "test_predictions": 8}
    assert repeated == result
    for path, original in original_files.items():
        assert (path.stat().st_mtime_ns, path.read_bytes()) == original
    verification = verify_study(pool, root)
    assert verification["status"] == "passed"
    assert verification["checkpoint_score_replays"] == 16
    assert verification["optimizer_updates_during_verification"] == 0
    assert verification["same_device_test_replays"] == 4
    assert verification["cross_device_member_probability_replays"] == 4
    assert all(value == 0 for value in verification["same_device_max_errors"].values())
    assert all(value == 0 for value in verification["cross_device_max_errors"].values())
    metrics_path = root / "test/event_v2/seed-17/metrics.json"
    altered = read_json(metrics_path)
    altered["endpoints"]["recurrence"]["tp"] += 1
    atomic_write_private_json(metrics_path, altered)
    with pytest.raises(ValueError, match="metric"):
        verify_study(pool, root)
