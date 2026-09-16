from __future__ import annotations

from dataclasses import asdict, replace

import pytest
import torch
from test_ct6 import compact_row, ct6_batch, ct6_model, ct6_rows, tiny_phase_fixture

from stageworld.artifacts import atomic_write_private_json, read_json
from stageworld.binary_data import (
    BinaryDevelopmentData,
    make_binary_folds,
    make_fold_data,
    parse_binary_label,
    read_binary_labels,
    unlabelled_bundle,
)
from stageworld.binary_endpoints import (
    BINARY_PROTOCOL,
    BinaryEndpointBatch,
    BinaryEndpointConfig,
    BinaryEndpointModel,
    BinaryEndpointTrainer,
    BinaryLossWeights,
)
from stageworld.binary_evaluation import evaluate_binary, predict_binary_logits
from stageworld.binary_inference import export_binary_bundle, verify_binary_portable
from stageworld.binary_training import train_binary_phase
from stageworld.binary_workflow import verify_binary_recovery
from stageworld.data.baseline_clinical import (
    CT6_CLINICAL_SCHEMA,
    fit_clinical_transform,
    parse_baseline_fields,
)
from stageworld.data.paired_ct import HMACPseudonymizer
from stageworld.data.treatment_compact import fit_name_support, treatment_fields
from stageworld.errors import ArtifactError, DataContractError
from stageworld.generated_inference import ct0_packet, predict_generated_query
from stageworld.synthetic_workflow import _atomic_torch_save, _slice_observation
from stageworld.training import TrainingPhase


def binary_batch(*, unlabelled=False):
    base = ct6_batch(unlabelled=True)
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [float("nan"), 1.0]])
    valid = torch.tensor([[True, True], [True, True], [True, True], [False, True]])
    if unlabelled:
        labels, valid = torch.zeros_like(labels), torch.zeros_like(valid)
    return BinaryEndpointBatch.from_generated(base, labels, valid)


def binary_model():
    return BinaryEndpointModel(BinaryEndpointConfig(**asdict(ct6_model().config))).eval()


def binary_fixture():
    config, bundle, snapshot = tiny_phase_fixture()
    config = replace(
        config,
        training=replace(
            config.training,
            development_protocol=BINARY_PROTOCOL,
            development_max_minutes=None,
        ),
    )
    train = binary_batch()
    val = replace(train, patient_ids=tuple(f"validation-{i}" for i in range(4)))
    return (
        config,
        replace(bundle, batches_by_split={"train": (train,), "validation": (val,)}),
        snapshot,
    )


def test_recorded_column_encoding_and_missing_are_distinct():
    assert [parse_binary_label(x, positive=1, negative=2) for x in (1, 2, None, 0, "2", "bad")] == [
        1,
        0,
        None,
        None,
        0,
        None,
    ]


def test_label_reader_uses_latest_columns_and_excludes_other_patients(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.Workbook()
    sheet = book.active
    sheet["BJ2"] = "pCR(1=\u662f\uff0c2=\u5426)"
    sheet["CG2"] = "\u590d\u53d1\u8f6c\u79fb\u72b6\u6001"
    identity = HMACPseudonymizer(b"synthetic-endpoint-unit-test-identity")
    keys = [identity.token("patient", str(i), prefix="P") for i in range(3)]
    for i, (pcr, recurrence) in enumerate(((1, 0), (2, 1), (None, 1), (1, 1)), 3):
        sheet[f"A{i}"] = str(i - 3)
        sheet[f"BJ{i}"] = pcr
        sheet[f"CG{i}"] = recurrence
        sheet[f"CN{i}"] = "=unused_survival_or_followup()"
    path = tmp_path / "synthetic.xlsx"
    book.save(path)
    assert read_binary_labels(path, set(keys), identity) == {
        keys[0]: [1, 0],
        keys[1]: [0, 1],
        keys[2]: [None, 1],
    }


def test_fold_clinical_and_treatment_fitting_excludes_inner_and_outer_validation(tmp_path):
    config, bundle, _ = tiny_phase_fixture()
    features, times = {}, {}
    for batches in bundle.batches_by_split.values():
        for batch in batches:
            for i, patient in enumerate(batch.patient_ids):
                features[patient] = {
                    "baseline_ct": _slice_observation(batch.ct0, torch.tensor([i])),
                    "post_treatment_ct": _slice_observation(batch.ct1, torch.tensor([i])),
                }
                times[patient, "s0"], times[patient, "s1"] = (
                    float(batch.s0_time[i]),
                    float(batch.s1_time[i]),
                )
    ids = sorted(features)
    clinical = {
        p: parse_baseline_fields({"age": 20 + i}, schema_version=CT6_CLINICAL_SCHEMA)
        for i, p in enumerate(ids)
    }
    data = BinaryDevelopmentData(
        bundle,
        features,
        times,
        clinical,
        {p: compact_row(p) for p in ids},
        {p: [i % 2, i % 2] for i, p in enumerate(ids)},
        "synthetic-inputs",
    )
    fold = {"fold": 0, "patient_ids": {"train": ids[:4], "validation": ids[4:6], "outer": ids[6:]}}
    first, _, snapshot = make_fold_data(config, data, fold, tmp_path / "first")
    data.clinical = {
        p: {**row, "age": row["age"] if p in ids[:4] else 10000} for p, row in clinical.items()
    }
    data.treatments[ids[-1]] = compact_row(ids[-1], regimens={"SOX": 1})
    changed, _, changed_snapshot = make_fold_data(config, data, fold, tmp_path / "changed")
    assert snapshot["transform"] == changed_snapshot["transform"]
    assert snapshot["treatment_support"] == changed_snapshot["treatment_support"]
    assert snapshot["fit_patient_ids"] == ids[:4]
    assert torch.equal(
        first.batches_by_split["train"][0].baseline.values,
        changed.batches_by_split["train"][0].baseline.values,
    )
    assert [parse_binary_label(x, positive=1, negative=0) for x in (0, 1, None, 2)] == [
        0,
        1,
        None,
        None,
    ]


def test_binary_heads_do_not_read_ct1_or_outcomes_and_do_not_decode_survival(monkeypatch):
    model, batch = binary_model(), binary_batch()
    assert not hasattr(model, "survival_decoder")
    with pytest.raises(DataContractError, match="binary endpoints"):
        model.predict_survival(None)
    original = predict_binary_logits(model, [batch])
    assert original.shape == (4, 2)
    changed = replace(
        batch,
        ct1=replace(batch.ct1, values=batch.ct1.values + 1000),
        future_ct_target=batch.future_ct_target * 100,
        endpoint_labels=torch.ones_like(batch.endpoint_labels),
    )
    assert torch.equal(original, predict_binary_logits(model, [changed]))
    output = model(**batch.prediction_inputs(deterministic=True))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output.logits[:, 1], batch.endpoint_labels[:, 1]
    )
    loss.backward()
    for module in (
        model.initializer,
        model.transition,
        model.endpoint_decoder,
        model.baseline_clinical_encoder,
    ):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert all(p.grad is None for p in model.updater.parameters())
    assert batch.future_ct_target.grad is None


@pytest.mark.parametrize("joint", [False, True])
def test_binary_losses_have_correct_masks_weights_and_logs(joint):
    model, batch = binary_model(), binary_batch(unlabelled=not joint)
    trainer = BinaryEndpointTrainer(
        model, torch.optim.AdamW(model.parameters()), device="cpu", mixed_precision="off"
    )
    phase = TrainingPhase.JOINT_ENDPOINTS if joint else TrainingPhase.WORLD_PRETRAIN
    weights = BinaryLossWeights(future_ct=0.1 if joint else 1, kl=0.001 if joint else 0.01)
    report = trainer.compute_loss(batch, phase=phase, weights=weights, kl_beta=0.4)
    expected = (
        weights.future_ct * report.components["future_ct"]
        + weights.kl * 0.4 * report.components["kl_ct"]
    )
    if joint:
        expected += report.components["pcr"] + report.components["recurrence"]
        assert report.effective_n["pcr"] == 3 and report.effective_n["recurrence"] == 4
    torch.testing.assert_close(report.total, expected)
    record = trainer.optimizer_step([batch], phase=phase, weights=weights, kl_beta=0.4)
    assert set(record["components"]) == {"future_ct", "kl_ct", "pcr", "recurrence"}
    assert torch.isfinite(torch.tensor(record["loss"]))
    with pytest.raises(DataContractError, match="unlabelled"):
        trainer.optimizer_step(
            [binary_batch()], phase=TrainingPhase.WORLD_PRETRAIN, weights=weights
        )


def test_five_folds_are_jointly_stratified_isolated_and_reproducible():
    labels = {f"p{i:03}": [(None, 0, 1)[i % 3], (i // 3) % 2] for i in range(120)}
    folds = make_binary_folds(labels)
    assert folds == make_binary_folds(dict(reversed(list(labels.items()))))
    outer = []
    for fold in folds["folds"]:
        groups = fold["patient_ids"]
        assert set(groups["train"]).isdisjoint(groups["validation"] + groups["outer"])
        assert set(groups["validation"]).isdisjoint(groups["outer"])
        assert len(groups["outer"]) == 24
        outer.extend(groups["outer"])
    assert len(outer) == len(set(outer)) == len(labels)


def test_metric_missingness_and_absent_classes_are_not_zero_scores():
    batch = binary_batch()
    result = evaluate_binary(
        torch.full((4, 2), 0.5), batch.endpoint_labels, batch.endpoint_valid, replicates=25
    )
    assert result["endpoints"]["pcr"]["labelled_patients"] == 3
    assert result["endpoints"]["recurrence"]["labelled_patients"] == 4
    labels = torch.zeros(4, 2)
    empty_class = evaluate_binary(
        torch.full((4, 2), 0.5), labels, torch.ones_like(labels, dtype=torch.bool), replicates=0
    )
    assert empty_class["endpoints"]["pcr"]["metrics"]["auroc"]["estimate"] is None


def test_phase_recovery_parent_binding_and_binary_checkpoint_contract(tmp_path):
    config, labelled, snapshot = binary_fixture()
    world = unlabelled_bundle(labelled)
    model, _ = train_binary_phase(
        config, world, snapshot, tmp_path / "world", phase=TrainingPhase.WORLD_PRETRAIN, epochs=2
    )
    del model
    recovery = verify_binary_recovery(
        config, world, snapshot, tmp_path / "probe", tmp_path / "world"
    )
    assert recovery["model_optimizer_rng_exact"]
    parent = torch.load(tmp_path / "world/selected.pt", weights_only=True)
    assert "endpoint_contract" in parent and "survival_contract" not in parent
    model, report = train_binary_phase(
        config,
        labelled,
        snapshot,
        tmp_path / "joint",
        phase=TrainingPhase.JOINT_ENDPOINTS,
        epochs=2,
        parent=parent,
    )
    before = predict_binary_logits(model, labelled.batches_by_split["validation"])
    resumed, replay = train_binary_phase(
        config,
        labelled,
        snapshot,
        tmp_path / "joint",
        phase=TrainingPhase.JOINT_ENDPOINTS,
        epochs=2,
        parent=parent,
        resume=True,
    )
    assert torch.equal(
        before, predict_binary_logits(resumed, labelled.batches_by_split["validation"])
    )
    assert report["optimizer_steps"] == replay["optimizer_steps"] == 2
    wrong = {**parent, "metadata": {**parent["metadata"], "training_seed": 999}}
    with pytest.raises(ArtifactError, match="fold and seed"):
        train_binary_phase(
            config,
            labelled,
            snapshot,
            tmp_path / "wrong",
            phase=TrainingPhase.JOINT_ENDPOINTS,
            epochs=2,
            parent=wrong,
        )


def test_four_file_binary_inference_and_existing_dispatch(tmp_path):
    model, batch = binary_model(), binary_batch()
    rows = ct6_rows()
    snapshot = {
        "artifact_id": "test-clinical",
        "transform": fit_clinical_transform(rows, set(rows), schema_version=CT6_CLINICAL_SCHEMA),
        "treatment_support": fit_name_support(
            [compact_row(p) for p in batch.patient_ids], set(batch.patient_ids)
        ),
    }
    export_binary_bundle(
        model,
        tmp_path / "inference.pt",
        snapshot=snapshot,
        encoder_provenance=batch.ct0.provenance,
        weight_version="test-binary",
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
            "treatment_scenario": {
                "description": "declared_test",
                **treatment_fields(compact_row()),
            },
        },
    )
    report = verify_binary_portable(tmp_path, predict_binary_logits(model, [batch]).sigmoid())
    assert report["ct1_labels_other_reads_denied"]
    predict_generated_query(
        tmp_path / "inference.json", tmp_path / "query.json", tmp_path / "dispatched.json"
    )
    result = read_json(tmp_path / "dispatched.json")
    assert result["survival_enabled"] is False and "rates" not in result
    assert len(result["probabilities"][0]) == 2
