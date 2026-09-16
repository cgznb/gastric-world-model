from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from test_ct6 import compact_row, ct6_rows

from stageworld.artifacts import read_json
from stageworld.binary700_statistics import logistic_anchor
from stageworld.generated651_data import complete_cases, make_folds
from stageworld.generated651_evaluation import METRICS, summarize_folds
from stageworld.generated651_inference import export_bundle, predict_bundle
from stageworld.generated651_spec import NEURAL
from stageworld.generated651_verification import verify_study
from stageworld.generated651_workflow import fit_anchor, run_study
from stageworld.generated700_data import Pool, fit_inputs
from stageworld.generated700_models import AnchoredClassifier
from stageworld.generated700_training import infer


@pytest.fixture
def pool(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.set_num_threads(2)
    torch.manual_seed(17)
    clinical = ct6_rows(200)
    ids = list(clinical)
    labels = torch.tensor([[int(i % 4 == 0), int(i % 3 == 0)] for i in range(200)]).float()
    ct0 = torch.randn(200, 27, 12)
    ct1 = ct0 * 0.8 + torch.randn_like(ct0) * 0.2
    return Pool(
        ids,
        clinical,
        {p: compact_row(p) for p in ids},
        torch.full((200,), 60.0),
        ct0,
        torch.ones(200, dtype=torch.bool),
        ct1.mean(1),
        torch.ones(200, dtype=torch.bool),
        labels,
        torch.ones_like(labels, dtype=torch.bool),
        "synthetic-complete651-tests",
        ct1,
    )


def test_complete_case_intersection_counts_overlap_and_preserves_source(pool):
    pool.valid[0, 0] = False
    pool.ct0_valid[1] = False
    pool.ct1_valid[2] = False
    pool.valid[2, 1] = False
    original = pool.ct0.clone()
    filtered = complete_cases(pool, expected=197)
    assert filtered.ids == pool.ids[3:]
    assert filtered.valid.all() and filtered.ct0_valid.all() and filtered.ct1_valid.all()
    filtered.ct0.add_(1)
    assert torch.equal(original, pool.ct0)
    with pytest.raises(ValueError, match="count changed"):
        complete_cases(pool, expected=198)


def test_patient_folds_cover_every_complete_patient_once(pool):
    first, second = make_folds(pool), make_folds(pool)
    assert first == second
    all_validation = []
    for entry in first["folds"]:
        train, validation = (entry["patient_ids"][key] for key in ("train", "validation"))
        assert len(train) == 160 and len(validation) == 40
        assert not set(train) & set(validation)
        assert set(train) | set(validation) == set(pool.ids)
        all_validation.extend(validation)
        assert all(
            entry["counts"]["validation"][name]["positive"] > 0 for name in ("pcr", "recurrence")
        )
    assert sorted(all_validation) == sorted(pool.ids)
    pool.valid[0, 0] = False
    with pytest.raises(ValueError, match="complete"):
        make_folds(pool)


def metric_records():
    return [
        {
            "arm": "generated_v2_bce",
            "seed": seed,
            "fold": fold,
            "endpoint": "pcr",
            **dict.fromkeys(METRICS, value),
        }
        for seed, values in ((17, [0.1, 0.2, 0.3, 0.4, 0.5]), (43, [0.8] * 5))
        for fold, value in enumerate(values)
    ]


def test_mean_and_sample_sd_are_within_each_seed_not_between_seeds():
    results = summarize_folds(metric_records())
    lookup = {(r["seed"], r["metric"]): r for r in results}
    assert len(results) == 2 * len(METRICS)
    assert lookup[17, "auroc"]["mean"] == pytest.approx(0.3)
    assert lookup[17, "auroc"]["sample_standard_deviation"] == pytest.approx(
        np.std([0.1, 0.2, 0.3, 0.4, 0.5], ddof=1)
    )
    assert lookup[43, "auroc"]["mean"] == pytest.approx(0.8)
    assert lookup[43, "auroc"]["sample_standard_deviation"] == 0


def test_incomplete_folds_and_unsupported_metrics_are_not_fivefold_results():
    records = metric_records()
    assert all(row["seed"] == 43 for row in summarize_folds(records[1:]))
    records[0]["precision"] = None
    precision = next(
        row
        for row in summarize_folds(records)
        if row["seed"] == 17 and row["metric"] == "precision"
    )
    assert precision["mean"] is None and precision["sample_standard_deviation"] is None
    assert precision["supported_folds"] == 4
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_folds(records + [records[0]])


def test_anchor_and_preprocessing_ignore_validation_values(pool, tmp_path):
    train = torch.arange(100)
    x, snapshot = fit_inputs(pool, pool.ids[:100], tmp_path / "original.pt")
    anchor = logistic_anchor(fit_anchor(pool, x, train))
    changed = replace(
        pool,
        labels=pool.labels.clone(),
        clinical={p: dict(row) for p, row in pool.clinical.items()},
    )
    changed.labels[100:] = 1 - changed.labels[100:]
    for patient in pool.ids[100:]:
        changed.clinical[patient]["age"] = 999
    y, refit = fit_inputs(changed, pool.ids[:100], tmp_path / "changed.pt")
    assert snapshot["clinical"] == refit["clinical"]
    assert torch.equal(x[:100], y[:100])
    new_anchor = logistic_anchor(fit_anchor(changed, y, train))
    assert all(torch.equal(a, b) for a, b in zip(anchor, new_anchor, strict=True))


def test_standalone_fixed_scale_and_threshold_no_future_input(pool, tmp_path):
    train, held = torch.arange(100), torch.arange(100, 132)
    x, snapshot = fit_inputs(pool, pool.ids[:100], tmp_path / "inputs.pt")
    anchor = logistic_anchor(fit_anchor(pool, x, train))
    model = AnchoredClassifier(361, "generated", *anchor, image_dim=12).eval()
    with torch.no_grad():
        for head in model.endpoints:
            head.body[-1].weight.normal_(std=0.01)
    expected = infer(model, x, pool, held).double().sigmoid()
    export_bundle(tmp_path / "bundle", snapshot, model)
    ids = [pool.ids[index] for index in held.tolist()]
    actual = predict_bundle(
        tmp_path / "bundle",
        [pool.clinical[p] for p in ids],
        [pool.treatments[p] for p in ids],
        pool.interval[held],
        pool.ct0[held],
    )
    torch.testing.assert_close(actual["probabilities"], expected, atol=1e-6, rtol=1e-6)
    assert torch.equal(actual["decisions"], actual["probabilities"] >= 0.5)
    assert actual["ct_features"].shape == (32, 27, 12)
    model.residual_scale[0] = 0
    with pytest.raises(ValueError, match="fixed neural residual"):
        export_bundle(tmp_path / "altered", snapshot, model)


def test_real_workflow_paths_synthetic_smoke_and_zero_update_resume(pool, tmp_path):
    root = tmp_path / "smoke"
    folds = make_folds(pool)
    first = run_study(pool, folds, root, smoke=True)
    assert first["neural_phases"] == 4 and first["updates"] == 8 and first["classifiers"] == 3
    verification = verify_study(pool, root)
    assert verification["checkpoint_score_replays"] == 8
    assert verification["standalone_denied_source_bundles"] == 3
    assert verification["optimizer_updates_during_audit"] == 0
    selected = [root / "fits" / arm.name / "17/fold-0/selected.pt" for arm in NEURAL]
    identities = [torch.load(path, weights_only=True)["artifact_id"] for path in selected]
    repeated = run_study(pool, folds, root, smoke=True)
    assert repeated["updates"] == first["updates"]
    assert identities == [torch.load(path, weights_only=True)["artifact_id"] for path in selected]
    summary = read_json(root / "evaluation/summary.json")
    assert summary["complete_seeds"] == [17] and not summary["across_seed_aggregation"]
    assert (root / "evaluation/per_seed/seed-17.md").exists()
