from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch
from test_ct6 import (
    assert_same_state,
    compact_row,
    ct6_batch,
    ct6_model,
    ct6_rows,
    tiny_phase_fixture,
)
from test_real_survival import _outcome

from stageworld.artifacts import atomic_write_private_json, read_json
from stageworld.config import load_config
from stageworld.ct6_evaluation import aggregate_seed_metrics
from stageworld.ct6_training import S1_ONLY_PROTOCOL, build_ct6_model, make_trainer, train_ct6_phase
from stageworld.ct6_workflow import verify_portable
from stageworld.data.baseline_clinical import CT6_CLINICAL_SCHEMA, fit_clinical_transform
from stageworld.data.outcomes import OutcomeBuilder
from stageworld.data.treatment_compact import fit_name_support, treatment_fields
from stageworld.errors import ArtifactError, DataContractError
from stageworld.generated_evaluation import (
    evaluate_s1_rates,
    predict_rates,
    predict_s1_rates,
    s1_nll_by_patient,
)
from stageworld.generated_inference import (
    ct0_packet,
    export_inference_bundle,
    predict_generated_query,
)
from stageworld.generated_training import S1OnlyTrainer
from stageworld.model.generated_s1 import GeneratedS1Model
from stageworld.real_survival import attach_os_labels
from stageworld.survival import piecewise_exponential_nll
from stageworld.synthetic_workflow import _atomic_torch_save, _slice_observation
from stageworld.training import LossWeights, TrainingPhase


def only_batch(*, unlabelled=False):
    batch = ct6_batch(unlabelled=unlabelled)
    valid = batch.survival_valid.clone()
    valid[:, (0, 2)] = False
    return replace(batch, survival_valid=valid)


def only_model():
    return GeneratedS1Model(replace(ct6_model().config, survival_task="s1_pred_only")).eval()


def only_phase_fixture():
    config, bundle, snapshot = tiny_phase_fixture()
    config = replace(
        config,
        model=replace(config.model, survival_task="s1_pred_only"),
        training=replace(
            config.training,
            development_protocol=S1_ONLY_PROTOCOL,
            development_max_minutes=None,
            lr=2e-4,
            weight_decay=0.01,
        ),
    )
    return config, bundle, snapshot


def test_approved_configuration_retains_capacity_and_new_clinical_inputs():
    config = load_config("configs/project.generated-s1-only-ct6.yaml")
    model = build_ct6_model(config, "history_generated")
    assert model.config.survival_task == "s1_pred_only"
    assert model.config.clinical_schema_version == CT6_CLINICAL_SCHEMA
    assert (model.config.hidden_dim, model.config.state_tokens, model.config.stochastic_dim) == (
        128,
        8,
        8,
    )
    assert model.config.transition_blocks == 2 and model.config.action_input_dim == 82
    assert config.training.comparison_seeds == (17, 43, 97)
    assert config.training.development_max_minutes is None


def test_three_original_capacity_seeds_are_aggregated_without_stage_padding():
    arms = {
        f"original_capacity/seed-{seed}/history_generated": {
            "readout_mode": "history_generated",
            "validation_nll": {"S1_pred": value},
            "metrics": [],
        }
        for seed, value in zip((17, 43, 97), (0.2, 0.3, 0.4), strict=True)
    }
    rows = aggregate_seed_metrics(arms, profile_prefix="original_capacity/")
    assert len(rows) == 1 and rows[0]["stage"] == "S1_pred"
    assert rows[0]["n_seeds_estimable"] == 3
    assert rows[0]["mean"] == pytest.approx(0.3)
    assert rows[0]["standard_deviation"] == pytest.approx(0.1)


def test_no_s0_decode_and_future_inputs_do_not_change_prediction(monkeypatch):
    model, batch = only_model(), only_batch()
    called = []
    original = model.survival_decoder.forward

    def guard(tokens, query_time, stage_index, *args):
        assert stage_index == 1
        called.append(stage_index)
        return original(tokens, query_time, stage_index, *args)

    monkeypatch.setattr(model.survival_decoder, "forward", guard)
    output = model(**batch.prediction_inputs(deterministic=True))
    assert output.survival_s0 is None and called == [1]
    changed = replace(
        batch,
        ct1=replace(batch.ct1, values=batch.ct1.values + 100),
        future_ct_target=batch.future_ct_target + 100,
        survival_durations=batch.survival_durations * 9,
    )
    assert torch.equal(predict_s1_rates(model, [batch]), predict_s1_rates(model, [changed]))
    with pytest.raises(DataContractError, match="S1-only evaluator"):
        predict_rates(model, [batch])


def test_only_s1_label_is_constructed(monkeypatch):
    batch = only_batch(unlabelled=True)
    original = OutcomeBuilder.build_label
    times = []

    def capture(self, patient_id, query_time):
        times.append(query_time)
        return original(self, patient_id, query_time)

    monkeypatch.setattr(OutcomeBuilder, "build_label", capture)
    (labelled,) = attach_os_labels(
        [batch],
        [_outcome(p) for p in batch.patient_ids],
        label_version="os-label-test-v1",
        stages=(1,),
    )
    assert times == batch.s1_time.tolist()
    assert labelled.survival_valid[:, 1].all() and not labelled.survival_valid[:, (0, 2)].any()
    assert not labelled.survival_durations[:, (0, 2)].any()
    torch.testing.assert_close(labelled.survival_durations[:, 1], (730.5 - batch.s1_time) / 365.25)


def test_s1_nll_alone_trains_initialization_and_transition_without_updater():
    model, batch = only_model(), only_batch()
    output = model(**batch.prediction_inputs(deterministic=True))
    loss = piecewise_exponential_nll(
        output.survival_s1_pred.rates.squeeze(-1),
        batch.survival_durations[:, 1],
        batch.survival_events[:, 1],
        model.config.survival_cutpoints,
    )
    loss.backward()
    for module in (
        model.initializer,
        model.transition,
        model.baseline_clinical_encoder,
        model.survival_decoder,
    ):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert all(p.grad is None for p in model.updater.parameters())
    assert batch.future_ct_target.grad is None


@pytest.mark.parametrize("phase", [TrainingPhase.WORLD_PRETRAIN, TrainingPhase.JOINT_SURVIVAL])
def test_exact_loss_weight_and_no_disabled_node_in_optimizer_logs(phase):
    model, batch = only_model(), only_batch(unlabelled=phase is TrainingPhase.WORLD_PRETRAIN)
    trainer = S1OnlyTrainer(
        model, torch.optim.AdamW(model.parameters(), lr=2e-4), device="cpu", mixed_precision="off"
    )
    joint = phase is TrainingPhase.JOINT_SURVIVAL
    weights = LossWeights(
        survival=int(joint),
        future_ct=0.1 if joint else 1,
        future_pathology=0,
        kl=0.001 if joint else 0.01,
    )
    report = trainer.compute_loss(batch, phase=phase, weights=weights, kl_beta=0.4)
    expected = (
        weights.future_ct * report.components["future_ct"]
        + weights.kl * 0.4 * report.components["kl_ct"]
    )
    if joint:
        expected = expected + report.components["survival_s1"]
    torch.testing.assert_close(report.total, expected)
    record = trainer.optimizer_step([batch], phase=phase, weights=weights, kl_beta=0.4)
    assert set(record["components"]) == {"future_ct", "kl_ct", "survival_s1"}
    assert set(record["effective_n"]) == {"future_ct", "kl_ct", "survival_s1"}
    with pytest.raises(DataContractError, match="Only S1 labels"):
        trainer.optimizer_step([ct6_batch()], phase=TrainingPhase.JOINT_SURVIVAL, weights=weights)


def test_s1_evaluation_matches_labels_and_ignores_disabled_slots():
    model, train = only_model(), only_batch()
    validation = replace(
        train, patient_ids=tuple(f"validation-{i}" for i in range(train.batch_size))
    )
    rates = predict_s1_rates(model, [validation])
    expected = piecewise_exponential_nll(
        rates[:, 0],
        validation.survival_durations[:, 1],
        validation.survival_events[:, 1],
        model.config.survival_cutpoints,
        reduction="none",
    )
    torch.testing.assert_close(
        s1_nll_by_patient(rates, [validation], model.config.survival_cutpoints), expected
    )
    report = evaluate_s1_rates(
        rates,
        [train],
        [validation],
        model.config.survival_cutpoints,
        method="test",
        prediction_id="test-s1",
        replicates=0,
    )
    assert set(report["validation_nll"]) == {"S1_pred"}
    assert len(report["metrics"]) == 7 and all(r["stage"] == "S1_pred" for r in report["metrics"])


def test_portable_s1_only_contract_and_four_file_inference(tmp_path):
    model, batch = only_model(), only_batch()
    rows = ct6_rows()
    treatment = compact_row(batch.patient_ids[0])
    export_inference_bundle(
        model,
        tmp_path / "inference.pt",
        clinical_snapshot={
            "artifact_id": "test-clinical",
            "transform": fit_clinical_transform(
                rows, set(rows), schema_version=CT6_CLINICAL_SCHEMA
            ),
        },
        treatment_support=fit_name_support(
            [compact_row(p) for p in batch.patient_ids], set(batch.patient_ids)
        ),
        encoder_provenance=batch.ct0.provenance,
        weight_version="test-weights",
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
            "horizons_years": [1.0, 3.0],
            "treatment_scenario": {"description": "test", **treatment_fields(treatment)},
        },
    )
    assert verify_portable(tmp_path, predict_s1_rates(model, [batch]))["status"] == "passed"
    config = read_json(tmp_path / "inference.json")
    assert config["survival_task"] == "s1_pred_only"
    config["survival_task"] = "s0_s1_pred"
    atomic_write_private_json(tmp_path / "inference.json", config)
    with pytest.raises(DataContractError, match="Survival task differs"):
        predict_generated_query(
            tmp_path / "inference.json", tmp_path / "query.json", tmp_path / "bad.json"
        )


@pytest.mark.parametrize("joint", [False, True])
def test_s1_only_epoch_recovery_no_time_cap_and_original_optimizer(tmp_path, monkeypatch, joint):
    config, bundle, snapshot = only_phase_fixture()
    args = dict(phase=TrainingPhase.WORLD_PRETRAIN, mode="history_generated", epochs=2)
    if joint:
        train_ct6_phase(config, bundle, snapshot, tmp_path / "parent", **args)
        parent = torch.load(tmp_path / "parent/selected.pt", weights_only=True)
        bundle = replace(
            bundle,
            batches_by_split={
                k: tuple(replace(b, survival_valid=only_batch().survival_valid) for b in v)
                for k, v in bundle.batches_by_split.items()
            },
        )
        args.update(phase=TrainingPhase.JOINT_SURVIVAL, parent=parent)
    model, report = train_ct6_phase(config, bundle, snapshot, tmp_path / "full", **args)
    trainer = make_trainer(
        config, model, tmp_path / "optimizer", joint=joint, steps_per_epoch=1, max_epochs=2
    )
    assert isinstance(trainer, S1OnlyTrainer) and trainer.scheduler is None
    assert all(
        g["lr"] == 2e-4 and g["weight_decay"] == 0.01 for g in trainer.optimizer.param_groups
    )
    assert report["time_guard_minutes"] is None
    assert "s0_nll" not in json.dumps(read_json(tmp_path / "full/history.json"))
    original = S1OnlyTrainer.optimizer_step

    def interrupted(self, *a, **kw):
        result = original(self, *a, **kw)
        if self.state.optimizer_step == 2:
            raise RuntimeError("test interrupted epoch")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(S1OnlyTrainer, "optimizer_step", interrupted)
        with pytest.raises(RuntimeError, match="interrupted epoch"):
            train_ct6_phase(config, bundle, snapshot, tmp_path / "resume", **args)
    train_ct6_phase(config, bundle, snapshot, tmp_path / "resume", resume=True, **args)
    expected = torch.load(tmp_path / "full/final.pt", weights_only=True)
    actual = torch.load(tmp_path / "resume/final.pt", weights_only=True)
    for field in (
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "rng_state",
        "trainer_state",
    ):
        assert_same_state(expected[field], actual[field])
    assert read_json(tmp_path / "full/history.json") == read_json(tmp_path / "resume/history.json")


def test_s1_only_plateau_selects_first_epoch_and_early_stops(tmp_path, monkeypatch):
    config, bundle, snapshot = only_phase_fixture()
    config = replace(config, training=replace(config.training, development_patience=2))
    monkeypatch.setattr(
        "stageworld.ct6_training.feature_metrics", lambda *a: {"ct_loss": 1.0, "kl": 0.0}
    )
    _, report = train_ct6_phase(
        config,
        bundle,
        snapshot,
        tmp_path / "phase",
        phase=TrainingPhase.WORLD_PRETRAIN,
        mode="history_generated",
        epochs=6,
    )
    assert report["selected_epoch"] == 1 and report["completed_epochs"] == 3
    assert report["stop_reason"] == "early_stopping"
    changed = replace(config, model=replace(config.model, survival_task="s0_s1_pred"))
    with pytest.raises(ArtifactError):
        train_ct6_phase(
            changed,
            bundle,
            snapshot,
            tmp_path / "phase",
            resume=True,
            phase=TrainingPhase.WORLD_PRETRAIN,
            mode="history_generated",
            epochs=6,
        )
