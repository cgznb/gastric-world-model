from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_training import _batch, _model

from stageworld.artifacts import atomic_write_private_json, read_json
from stageworld.cache import CacheProvenance
from stageworld.config import load_config
from stageworld.data.contracts import AdjudicationStatus, EventType
from stageworld.data.paired_ct import PRIVATE_OUTCOME_SCHEMA
from stageworld.errors import ArtifactError, ConfigurationError, DataContractError
from stageworld.real_survival import (
    _baseline_inputs,
    _validate_training_schedule,
    attach_os_labels,
    evaluate_rates,
    evaluate_real_os_development,
    fit_ridge_baseline,
    predict_rates,
    predict_real_os_development,
    run_real_os_development,
    stage_mean_nll,
    verify_real_os_development,
)
from stageworld.real_workflow import RealFeatureBundle


def _outcome(patient, *, death=True, terminal=730.5):
    return {
        "patient_id": patient,
        "endpoint_name": "os",
        "event_type": EventType.DEATH.value,
        "source_status": 1 if death else 0,
        "event_date_days": terminal if death else None,
        "censor_date_days": None if death else terminal,
        "adjudication_status": AdjudicationStatus.CONFIRMED.value,
        "origin_definition": "clinical_excel_column_E",
        "reason_of_censoring": None,
        "label_version": "os-label-test-v1",
    }


def test_real_label_attachment_converts_remaining_days_and_leaves_inputs_untouched():
    batch = _batch((0, 1))
    rows = [_outcome(batch.patient_ids[0]), _outcome(batch.patient_ids[1], death=False)]
    # A malformed held-out outcome must never enter validation/training label interpretation.
    rows.append({"patient_id": "test-patient", "source_status": "not-an-event"})
    (labelled,) = attach_os_labels((batch,), rows, label_version="os-label-test-v1")
    assert torch.allclose(labelled.survival_durations[:, 0], torch.tensor([2.0, 2.0]))
    assert torch.allclose(
        labelled.survival_durations[:, 1], torch.full((2,), (730.5 - 35) / 365.25)
    )
    assert labelled.survival_events[:, 0].tolist() == [1, 0]
    assert labelled.survival_valid.tolist() == [[True, True, False], [True, True, False]]
    assert labelled.ct0 is batch.ct0 and labelled.ct1 is batch.ct1
    assert labelled.future_ct_target is batch.future_ct_target


def test_real_labels_reject_nonpositive_remaining_followup():
    batch = _batch((0,))
    with pytest.raises(DataContractError) as error:
        attach_os_labels(
            (batch,),
            [_outcome(batch.patient_ids[0], terminal=35)],
            label_version="os-label-test-v1",
        )
    assert error.value.code == "zero_remaining_time"


def test_deterministic_os_predictions_ignore_outcomes_and_future_ct_at_s0():
    model = _model()
    batch = _batch()
    original = predict_rates(model, (batch,))
    changed = replace(
        batch,
        survival_durations=batch.survival_durations * 50,
        survival_events=1 - batch.survival_events,
        future_ct_target=batch.future_ct_target + 100,
    )
    assert torch.equal(original, predict_rates(model, (changed,)))
    changed = replace(batch, ct1=replace(batch.ct1, values=batch.ct1.values + 100))
    new = predict_rates(model, (changed,))
    assert torch.equal(original[:, 0], new[:, 0])
    assert not torch.equal(original[:, 1], new[:, 1])
    assert torch.equal(original, predict_rates(model, (batch,)))


def test_matched_baseline_has_no_future_feature_in_s0():
    batch = _batch()
    changed = replace(batch, ct1=replace(batch.ct1, values=batch.ct1.values + 100))
    assert torch.equal(_baseline_inputs((batch,), 0), _baseline_inputs((changed,), 0))
    assert not torch.equal(_baseline_inputs((batch,), 1), _baseline_inputs((changed,), 1))


def test_os_baseline_and_censored_metric_path_are_finite_and_split_matched():
    train, validation = (_batch(tuple(range(20))),), (_batch(tuple(range(20, 30))),)
    train = tuple(replace(b, survival_durations=b.survival_durations * 5) for b in train)
    validation = tuple(replace(b, survival_durations=b.survival_durations * 2) for b in validation)
    rates, snapshots = fit_ridge_baseline(train, validation, (0.0, 1.0, 3.0))
    assert rates.shape == (10, 2, 2) and torch.isfinite(rates).all()
    assert len(snapshots) == 2
    assert stage_mean_nll(rates, validation, (0.0, 1.0, 3.0)) > 0
    results = evaluate_rates(
        rates,
        train,
        validation,
        (0.0, 1.0, 3.0),
        method="synthetic-test-only",
        prediction_id="test-only",
    )
    assert len(results) == 14
    assert {row["stage"] for row in results} == {"s0", "s1"}
    assert all(row["status"] in {"ok", "not_estimable"} for row in results)
    assert any(row["status"] == "ok" for row in results)


@pytest.mark.parametrize(
    "with_treatments",
    [False, True, "regimens", "regimens_100ep", "tumor_candidates", "tumor_coarse"],
)
def test_synthetic_fixture_exercises_real_os_training_checkpoint_and_evaluation(
    tmp_path, monkeypatch, with_treatments
):
    import stageworld.real_survival as workflow

    config = load_config(
        Path(__file__).resolve().parents[1] / "configs/project.weiai-os-v1-gastric-roi.yaml"
    )
    config = replace(
        config,
        paths=replace(
            config.paths,
            approved_data_root=str(tmp_path),
            feature_root=str(tmp_path / "features"),
            data_artifact_root=str(tmp_path / "data"),
            output_root=str(tmp_path / "output"),
        ),
        survival=replace(config.survival, finite_cutpoints=(0.0, 1.0, 3.0)),
        training=replace(
            config.training,
            world_pretrain_steps=5,
            joint_survival_steps=5,
            development_max_epochs=1,
            mixed_precision="off",
        ),
    )
    if with_treatments:
        from stageworld.data.treatment_summary import CONFIRMED_CUTOFF, TREATMENT_PROTOCOL

        config = replace(
            config,
            paths=replace(
                config.paths, treatment_manifest=str(tmp_path / "treatments/manifest.json")
            ),
            clinical=replace(config.clinical, treatment_summary_cutoff=CONFIRMED_CUTOFF),
            training=replace(config.training, development_protocol=TREATMENT_PROTOCOL),
        )
        if with_treatments in ("regimens", "regimens_100ep", "tumor_candidates", "tumor_coarse"):
            from stageworld.data.treatment_regimens import (
                REGIMEN_100EP_PROTOCOL,
                REGIMEN_ACTION_DIM,
                REGIMEN_PROTOCOL,
            )

            config = replace(
                config,
                model=replace(config.model, action_input_dim=REGIMEN_ACTION_DIM),
                training=replace(config.training, development_protocol=REGIMEN_PROTOCOL),
            )
            if with_treatments in ("regimens_100ep", "tumor_candidates", "tumor_coarse"):
                config = replace(
                    config,
                    training=replace(
                        config.training,
                        development_protocol=REGIMEN_100EP_PROTOCOL,
                        world_pretrain_steps=500,
                        joint_survival_steps=500,
                        development_max_epochs=100,
                        development_patience=None,
                    ),
                )
                # Force a validation plateau to exercise disabled early stopping.
                monkeypatch.setattr(workflow, "stage_mean_nll", lambda *a, **kw: 0.5)
            if with_treatments == "tumor_candidates":
                from stageworld.data.treatment_regimens import TUMOR_REGIMEN_100EP_PROTOCOL
                from stageworld.data.tumor_roi import TUMOR_ROI_VERSION

                config = replace(
                    config,
                    encoders=replace(config.encoders, ct_preprocess_version=TUMOR_ROI_VERSION),
                    training=replace(
                        config.training, development_protocol=TUMOR_REGIMEN_100EP_PROTOCOL
                    ),
                )
            if with_treatments == "tumor_coarse":
                from stageworld.data.treatment_regimens import COARSE_TUMOR_100EP_PROTOCOL
                from stageworld.data.tumor_roi import COARSE_TUMOR_ROI_VERSION

                config = replace(
                    config,
                    encoders=replace(
                        config.encoders, ct_preprocess_version=COARSE_TUMOR_ROI_VERSION
                    ),
                    training=replace(
                        config.training, development_protocol=COARSE_TUMOR_100EP_PROTOCOL
                    ),
                )
    train = tuple(_batch(tuple(range(i, i + 20))) for i in range(0, 100, 20))
    validation = tuple(_batch(tuple(range(i, i + 15))) for i in range(100, 130, 15))
    if with_treatments:
        from test_treatment_summary import treatment_batch

        train = tuple(
            treatment_batch(tuple(range(i, i + 20)), config.model.action_input_dim)
            for i in range(0, 100, 20)
        )
        validation = tuple(
            treatment_batch(tuple(range(i, i + 15)), config.model.action_input_dim)
            for i in range(100, 130, 15)
        )
    provenance = CacheProvenance.from_encoder(
        train[0].ct0.provenance,
        schema_version="fixture-v1",
        patch_sampling_version="fixture-v1",
        split_version="fixture-split",
        target_transform_version="fixture-v1",
        teacher_version="fixture-v1",
    )
    bundle = RealFeatureBundle(
        batches_by_split={"train": train, "validation": validation},
        data_lineage_id="fixture-data",
        cohort_artifact_id="fixture-cohort",
        split_version="fixture-split",
        feature_artifact_id="fixture-features",
        cache_provenance=provenance,
        training_patient_count=100,
        validation_patient_count=30,
        held_out_test_patient_count=10,
    )
    if with_treatments:
        from test_treatment_summary import write_treatment_fixture

        if with_treatments in ("regimens", "regimens_100ep", "tumor_candidates", "tumor_coarse"):
            from test_treatment_regimens import write_regimen_fixture

            write_regimen_fixture(config, bundle)
        else:
            write_treatment_fixture(config, bundle)
    atomic_write_private_json(tmp_path / "features/ct_manifest.json", {"limit_per_split": None})
    atomic_write_private_json(
        tmp_path / "data/split_assignments.json",
        {
            "assignments": (
                [{"patient_id": p, "split": "train"} for b in train for p in b.patient_ids]
                + [
                    {"patient_id": p, "split": "validation"}
                    for b in validation
                    for p in b.patient_ids
                ]
                + [{"patient_id": f"test-{i}", "split": "test"} for i in range(10)]
            ),
        },
    )
    outcomes = [
        _outcome(p, death=i % 5 == 0, terminal=400 + (i % 50) * 50)
        for i, p in enumerate(p for b in (*train, *validation) for p in b.patient_ids)
    ]
    atomic_write_private_json(
        tmp_path / "data/outcomes.json",
        {
            "schema_version": PRIVATE_OUTCOME_SCHEMA,
            "data_lineage_id": "fixture-data",
            "cohort_artifact_id": "fixture-cohort",
            "outcome_label_version": "os-label-test-v1",
            "outcomes": outcomes,
        },
    )
    monkeypatch.setattr(workflow, "load_real_world_pretrain_batches", lambda _: bundle)
    if with_treatments:
        from stageworld.model import StageWorldModel

        monkeypatch.setattr(
            workflow,
            "build_model",
            lambda _: StageWorldModel(
                replace(_model().config, action_input_dim=config.model.action_input_dim)
            ),
        )
    else:
        monkeypatch.setattr(workflow, "build_model", lambda _: _model())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = run_real_os_development(config)
    assert result["status"] == "completed"
    expected_steps = 500 if with_treatments in (
        "regimens_100ep", "tumor_candidates", "tumor_coarse"
    ) else 5
    assert result["pretraining_steps"] == expected_steps
    assert result["joint_optimizer_steps"] == expected_steps
    assert len(result["metrics"]) == 28
    summary = read_json(config.output_root / "os_summary.json")
    assert summary["test_used"] is False and summary["clinical_validation"] is False
    assert summary["training_deaths"] == 20 and summary["validation_deaths"] == 6
    assert summary["future_ct_validation_phase"] == "selected_joint_survival"
    if with_treatments:
        assert summary["treatment_artifact_id"] == "treatment-interval-synthetic-fixture"
        assert summary["treatment_response_probe"]["s0_invariant"] is True
        if with_treatments in ("regimens", "regimens_100ep", "tumor_candidates", "tumor_coarse"):
            assert summary["regimen_response_probe"]["s0_invariant"] is True
            assert summary["treatment_inputs"]["drug_regimens_used"] is True
    payload = torch.load(summary["checkpoint"], weights_only=True)
    assert payload["metadata"]["phase"] == "joint_survival"
    assert payload["transfer_parent_model_state"] is not None
    final = torch.load(summary["final_checkpoint"], weights_only=True)
    assert final["trainer_state"]["optimizer_step"] == expected_steps
    assert final["sampler_state"]["epoch"] == expected_steps // 5
    assert summary["pretraining_completed_epochs"] == expected_steps // 5
    assert summary["pretraining_partial_epoch_steps"] == 0
    assert summary["joint_completed_epochs"] == expected_steps // 5
    assert summary["weight_version"] != summary["final_weight_version"]
    if with_treatments in ("regimens_100ep", "tumor_candidates", "tumor_coarse"):
        assert summary["selected_epoch"] == 1
        assert payload["trainer_state"]["optimizer_step"] == 5
        assert summary["stop_reason"] == "step_limit"
        assert [row["optimizer_steps"] for row in summary["training_history"]] == list(
            range(5, 501, 5)
        )
    predicted = predict_real_os_development(config)
    assert predicted["outcome_labels_used"] is False and predicted["patient_count"] == 30
    prediction_path = Path(predicted["predictions"])
    first = read_json(summary["predictions"])
    second = read_json(prediction_path)
    assert first["rates"] == second["rates"]
    evaluated = evaluate_real_os_development(config, predictions=prediction_path)
    assert evaluated["status"] == "ok" and len(evaluated["metrics"]) == 14
    verified = verify_real_os_development(config)
    assert verified["status"] == "ok"
    assert verified["immutable_parent_exact_match"] is True
    assert verified["maximum_rate_absolute_difference"] == 0.0
    assert verified["metrics_replayed"] is True
    assert verified["unsafe_permissions"] == 0
    assert verified["training_log_records"] == expected_steps * 2
    assert verified["training_duration_verified"] is True
    assert verified["final_checkpoint_verified"] is True
    assert verified["test_patient_overlap"] == verified["training_validation_overlap"] == 0
    assert read_json(config.output_root / "training_progress.json")["phase"] == "completed"
    report = Path(verified["report"]).read_text()
    assert "Future CT Prediction" in report and "different query times" in report
    assert "World model (world_pretrain)" in report
    assert "World model (selected_joint_survival)" in report
    assert "grouped treatment workbook does not replace labels" in report
    assert "Completed epochs:" in report
    if with_treatments == "tumor_candidates":
        assert "Tumor Candidate ROI" in report
        assert "detection-conditioned cohort" in report
        assert "Stomach-organ ROI, not tumor segmentation" not in report
        assert summary["explicit_tumor_growth_prediction"] is False
    if with_treatments == "tumor_coarse":
        assert "Tumor-Guided Coarse ROI" in report
        assert "including stomach fallbacks" in report
        assert "two nonempty tumor-candidate ROIs" not in report
        assert summary["roi_target"] == "tumor_guided_context_with_explicit_stomach_fallback"
        assert summary["explicit_tumor_growth_prediction"] is False

    changed_duration = {**summary, "joint_completed_epochs": 0}
    atomic_write_private_json(config.output_root / "os_summary.json", changed_duration)
    try:
        with pytest.raises(ArtifactError) as error:
            verify_real_os_development(config)
        assert error.value.code == "OS_AUDIT_DURATION_MISMATCH"
    finally:
        atomic_write_private_json(config.output_root / "os_summary.json", summary)

    events_path = Path(summary["predictions"]).parent / "world_events.jsonl"
    events_path.chmod(0o644)
    try:
        with pytest.raises(ArtifactError) as error:
            verify_real_os_development(config)
        assert error.value.code == "OS_AUDIT_PERMISSIONS"
    finally:
        events_path.chmod(0o600)

    corrupted = {**first, "rates": (torch.tensor(first["rates"]) + 1).tolist()}
    atomic_write_private_json(Path(summary["predictions"]), corrupted)
    try:
        with pytest.raises(ArtifactError) as error:
            verify_real_os_development(config)
        assert error.value.code == "OS_AUDIT_PREDICTION_MISMATCH"
    finally:
        atomic_write_private_json(Path(summary["predictions"]), first)

    second["weight_version"] = "wrong-weights"
    atomic_write_private_json(prediction_path, second)
    with pytest.raises(ArtifactError) as error:
        evaluate_real_os_development(config, predictions=prediction_path)
    assert error.value.code == "OS_PREDICTION_LINEAGE_MISMATCH"


@pytest.mark.parametrize(
    "changes,steps_per_epoch",
    [
        ({"world_pretrain_steps": 1099}, 11),
        ({"world_pretrain_steps": 1089}, 11),
        ({"joint_survival_steps": 960}, 11),
        ({"development_max_epochs": 60}, 11),
        ({"development_patience": 10}, 11),
        ({}, 12),
    ],
)
def test_long_protocol_rejects_partial_or_insufficient_epochs(changes, steps_per_epoch):
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs/project.weiai-os-v1-regimen-roi-100ep.yaml"
    )
    config = replace(config, training=replace(config.training, **changes))
    with pytest.raises(ConfigurationError) as error:
        _validate_training_schedule(config, steps_per_epoch)
    assert error.value.code == "OS_100_EPOCH_BUDGET_REQUIRED"


def test_long_protocol_changes_only_duration_and_experiment_identity():
    from dataclasses import asdict

    from stageworld.synthetic_workflow import _config_lineage

    configs = Path(__file__).resolve().parents[1] / "configs"
    old = load_config(configs / "project.weiai-os-v1-regimen-roi.yaml")
    new = load_config(configs / "project.weiai-os-v1-regimen-roi-100ep.yaml")
    _validate_training_schedule(new, 11)
    assert new.training.world_pretrain_steps == new.training.joint_survival_steps == 1100
    assert new.training.development_max_epochs == 100
    assert new.training.development_patience is None
    assert old.training.world_pretrain_steps == 160
    assert old.training.joint_survival_steps == 960
    assert old.training.development_patience == 10
    assert old.paths.output_root != new.paths.output_root
    allowed = {
        "world_pretrain_steps",
        "joint_survival_steps",
        "development_protocol",
        "development_max_epochs",
        "development_patience",
        "development_max_minutes",
    }
    assert {k: v for k, v in asdict(old.training).items() if k not in allowed} == {
        k: v for k, v in asdict(new.training).items() if k not in allowed
    }
    restored = replace(new, project=old.project, paths=old.paths, training=old.training)
    assert restored == old
    assert _config_lineage(restored) == _config_lineage(old)
    assert _config_lineage(new) != _config_lineage(old)
