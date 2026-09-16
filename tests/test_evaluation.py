from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from stageworld.evaluation import (
    CalibrationBinSpec,
    CensoringDistribution,
    CensoringRule,
    CompetingControlDefinition,
    EvaluationArtifactWriter,
    EvaluationCohort,
    EvaluationProtocol,
    FigureLineage,
    IPCWIsotonicCalibrator,
    MetricResult,
    PredictionArtifact,
    PredictionLineage,
    PredictionRecord,
    RunMode,
    ScoreDirection,
    SplitRole,
    build_metric_lineage,
    calibration_curve,
    cumulative_dynamic_auc,
    default_ablation_registry,
    integrated_brier_score,
    ipcw_brier_score,
    ipcw_concordance_index,
    make_patient_bootstrap_plan,
    match_prediction_artifacts,
    paired_patient_bootstrap,
    read_prediction_artifact,
    validate_censoring_policy,
)


def _cohort(
    cohort_id: str,
    role: SplitRole,
    *,
    times: tuple[float, ...] = (1.0, 2.0, 4.0, 5.0),
    events: tuple[int, ...] = (1, 1, 0, 1),
) -> EvaluationCohort:
    return EvaluationCohort(
        cohort_id=cohort_id,
        split_role=role,
        endpoint="OS",
        patient_ids=tuple(f"synthetic-{index}" for index in range(len(times))),
        times=times,
        event_types=events,
    )


def _protocol(role: SplitRole, *horizons: float) -> EvaluationProtocol:
    return EvaluationProtocol(
        protocol_id="protocol-v1",
        endpoint="OS",
        horizons=tuple(horizons),
        evaluation_role=role,
    )


def test_t40_risk_direction_ipcw_metrics_and_support() -> None:
    reference = _cohort(
        "development-reference",
        SplitRole.DEVELOPMENT,
        times=(1.0, 2.0, 6.0, 7.0, 8.0),
        events=(1, 0, 1, 0, 1),
    )
    evaluation = _cohort(
        "held-out",
        SplitRole.TEST,
        times=(1.0, 2.0, 4.0, 5.0),
        events=(1, 1, 0, 1),
    )
    censoring = CensoringDistribution.fit(reference, artifact_id="censoring-v1")
    protocol = _protocol(SplitRole.TEST, 3.0, 7.0)
    risk = (0.9, 0.8, 0.2, 0.1)

    auc = cumulative_dynamic_auc(evaluation, risk, 3.0, censoring, protocol=protocol)
    concordance = ipcw_concordance_index(evaluation, risk, 3.0, censoring, protocol=protocol)
    assert auc.status == "ok" and auc.estimate == pytest.approx(1.0)
    assert concordance.status == "ok" and concordance.estimate == pytest.approx(1.0)

    reverse = cumulative_dynamic_auc(
        evaluation,
        tuple(1.0 - value for value in risk),
        3.0,
        censoring,
        direction=ScoreDirection.HIGHER_RISK,
        protocol=protocol,
    )
    survival_direction = cumulative_dynamic_auc(
        evaluation,
        tuple(1.0 - value for value in risk),
        3.0,
        censoring,
        direction=ScoreDirection.HIGHER_SURVIVAL,
        protocol=protocol,
    )
    assert reverse.estimate == pytest.approx(0.0)
    assert survival_direction.estimate == pytest.approx(1.0)

    unsupported = cumulative_dynamic_auc(evaluation, risk, 9.0, censoring)
    assert unsupported.status == "not_estimable"
    assert unsupported.estimate is None
    assert unsupported.reason == "outside_censoring_support"


def test_t40_brier_and_ibs_match_uncensored_binary_reference() -> None:
    reference = _cohort(
        "train",
        SplitRole.TRAIN,
        times=(5.0, 6.0, 7.0, 8.0),
        events=(1, 1, 1, 1),
    )
    evaluation = _cohort(
        "test",
        SplitRole.TEST,
        times=(1.0, 2.0, 4.0, 5.0),
        events=(1, 1, 0, 1),
    )
    censoring = CensoringDistribution.fit(reference)
    risks_at_three = np.asarray([0.8, 0.7, 0.3, 0.2])
    result = ipcw_brier_score(evaluation, risks_at_three, 3.0, censoring)
    expected = np.mean(np.square(np.asarray([1.0, 1.0, 0.0, 0.0]) - risks_at_three))
    assert result.estimate == pytest.approx(expected)

    predictions = np.column_stack((risks_at_three * 0.5, risks_at_three))
    ibs = integrated_brier_score(evaluation, predictions, (1.5, 3.0), censoring)
    first = ipcw_brier_score(evaluation, predictions[:, 0], 1.5, censoring)
    assert ibs.status == "ok"
    assert ibs.estimate == pytest.approx((first.estimate + result.estimate) / 2.0)


def test_t40_censoring_source_rule_is_explicit_for_internal_and_external() -> None:
    test = _cohort("test", SplitRole.TEST)
    test_censoring = CensoringDistribution.fit(test)
    with pytest.raises(ValueError, match="train/development"):
        validate_censoring_policy(test_censoring, test, _protocol(SplitRole.TEST, 3.0))
    with pytest.raises(ValueError, match="explicit predeclared protocol"):
        cumulative_dynamic_auc(test, (0.9, 0.8, 0.2, 0.1), 3.0, test_censoring)

    external = _cohort("external", SplitRole.EXTERNAL)
    external_censoring = CensoringDistribution.fit(external)
    protocol = EvaluationProtocol(
        protocol_id="external-v1",
        endpoint="OS",
        horizons=(3.0,),
        evaluation_role=SplitRole.EXTERNAL,
        censoring_rule=CensoringRule.EXTERNAL_COHORT,
        external_censoring_predeclared=True,
    )
    validate_censoring_policy(external_censoring, external, protocol)
    with pytest.raises(ValueError, match="predeclaration"):
        EvaluationProtocol(
            protocol_id="invalid",
            endpoint="OS",
            horizons=(3.0,),
            evaluation_role=SplitRole.EXTERNAL,
            censoring_rule=CensoringRule.EXTERNAL_COHORT,
        )


def test_t40_competing_events_require_cif_cause_and_are_not_treated_as_censoring() -> None:
    reference = _cohort(
        "development-reference",
        SplitRole.DEVELOPMENT,
        times=(1.0, 2.0, 5.0, 6.0),
        events=(1, 2, 0, 1),
    )
    competing = _cohort(
        "competing-test",
        SplitRole.TEST,
        times=(1.0, 2.0, 5.0, 6.0),
        events=(1, 2, 0, 1),
    )
    censoring = CensoringDistribution.fit(reference)
    with pytest.raises(ValueError, match="explicit cause"):
        cumulative_dynamic_auc(competing, (0.9, 0.1, 0.2, 0.3), 3.0, censoring)

    auc = cumulative_dynamic_auc(
        competing,
        (0.9, 0.1, 0.2, 0.3),
        3.0,
        censoring,
        cause=1,
        control_definition=CompetingControlDefinition.NOT_CASE,
    )
    brier = ipcw_brier_score(competing, (0.9, 0.1, 0.2, 0.3), 3.0, censoring, cause=1)
    assert auc.status == "ok" and auc.estimate == pytest.approx(1.0)
    assert brier.status == "ok" and brier.n_events == 1


def test_t41_patient_bootstrap_keeps_stages_together_and_pairs_methods() -> None:
    patient_ids = ("p1", "p1", "p2", "p2", "p3", "p3")
    left = np.asarray([0.8, 0.7, 0.5, 0.4, 0.2, 0.1])
    right = left - 0.1
    plan = make_patient_bootstrap_plan(patient_ids, n_resamples=40, seed=13)
    for replicate in range(40):
        rows = plan.expand_rows(patient_ids, replicate)
        counts = np.bincount(rows, minlength=len(patient_ids))
        assert counts[0] == counts[1]
        assert counts[2] == counts[3]
        assert counts[4] == counts[5]

    seen: list[tuple[int, ...]] = []

    def statistic(values: np.ndarray, indices: np.ndarray) -> float:
        seen.append(tuple(int(index) for index in indices))
        return float(values.mean())

    result = paired_patient_bootstrap(
        patient_ids,
        left,
        right,
        statistic,
        n_resamples=40,
        plan=plan,
    )
    assert result.status == "ok"
    assert result.estimate_difference == pytest.approx(0.1)
    assert result.confidence_lower == pytest.approx(0.1)
    assert result.confidence_upper == pytest.approx(0.1)
    assert result.resampling_unit == "patient"
    assert result.seed_variability_included is False
    # Two point calls, followed by exactly the same index vector for each pair.
    for offset in range(2, len(seen), 2):
        assert seen[offset] == seen[offset + 1]


def test_t42_no_events_and_empty_comparisons_are_explicitly_not_estimable() -> None:
    reference = _cohort("reference", SplitRole.DEVELOPMENT, times=(3.0, 4.0, 5.0), events=(1, 0, 1))
    no_events = _cohort("subgroup", SplitRole.TEST, times=(2.0, 4.0, 5.0), events=(0, 0, 0))
    censoring = CensoringDistribution.fit(reference)
    auc = cumulative_dynamic_auc(no_events, (0.4, 0.3, 0.2), 2.5, censoring)
    concordance = ipcw_concordance_index(no_events, (0.4, 0.3, 0.2), 2.5, censoring)
    assert auc.status == "not_estimable" and auc.reason == "no_cases_by_horizon"
    assert concordance.status == "not_estimable"
    assert auc.estimate is None and concordance.estimate is None
    assert json.loads(json.dumps(auc.as_dict()))["estimate"] is None


def test_t43_calibration_is_censoring_aware_development_fit_and_immutable() -> None:
    reference = _cohort(
        "train",
        SplitRole.TRAIN,
        times=(2.0, 3.0, 5.0, 6.0, 7.0, 8.0),
        events=(1, 0, 1, 1, 0, 1),
    )
    development = _cohort(
        "development",
        SplitRole.DEVELOPMENT,
        times=(1.0, 2.0, 2.5, 4.0, 5.0, 6.0),
        events=(1, 0, 1, 0, 1, 0),
    )
    censoring = CensoringDistribution.fit(reference, artifact_id="censoring-train-v1")
    dev_risk = (0.8, 0.6, 0.7, 0.3, 0.4, 0.2)
    bins = CalibrationBinSpec.fit(
        development,
        dev_risk,
        bins=2,
        source_prediction_artifact_id="prediction-development-v1",
    )
    calibrator = IPCWIsotonicCalibrator.fit(
        development,
        dev_risk,
        3.0,
        censoring,
        source_prediction_artifact_id="prediction-development-v1",
    )
    before = tuple(calibrator.calibrated_values)
    transformed = calibrator.transform((0.1, 0.5, 0.9))
    assert np.all((0.0 <= transformed) & (transformed <= 1.0))
    assert tuple(calibrator.calibrated_values) == before
    with pytest.raises(FrozenInstanceError):
        calibrator.horizon = 4.0  # type: ignore[misc]

    test = _cohort(
        "test",
        SplitRole.TEST,
        times=(1.0, 2.0, 4.0, 5.0),
        events=(1, 1, 0, 1),
    )
    curve = calibration_curve(test, (0.8, 0.7, 0.3, 0.2), 3.0, censoring, bins)
    assert curve.status == "ok"
    assert curve.estimator == "kaplan_meier"
    assert curve.weighted_absolute_error is not None
    with pytest.raises(ValueError, match="development"):
        CalibrationBinSpec.fit(
            test,
            (0.8, 0.7, 0.3, 0.2),
            bins=2,
            source_prediction_artifact_id="forbidden-test-fit",
        )
    with pytest.raises(ValueError, match="development"):
        IPCWIsotonicCalibrator.fit(
            test,
            (0.8, 0.7, 0.3, 0.2),
            3.0,
            censoring,
            source_prediction_artifact_id="forbidden-test-fit",
        )


def _prediction_artifact(
    artifact_id: str,
    model_id: str,
    model_name: str,
    risks: tuple[float, ...],
    *,
    input_id: str = "synthetic-input-v1",
    split_id: str = "synthetic-split-v1",
    mode: RunMode = RunMode.SYNTHETIC,
) -> PredictionArtifact:
    lineage = PredictionLineage(
        artifact_id=artifact_id,
        checkpoint_schema_version="stageworld-checkpoint-test-v3",
        checkpoint_id=f"checkpoint-{model_id}",
        weight_version=model_id,
        model_artifact_id=model_id,
        model_version=f"{model_name}-v1",
        checkpoint_endpoint="os",
        config_lineage_id="config-v1",
        config_version="config-v1",
        data_lineage_id=input_id,
        input_artifact_id=input_id,
        cohort_artifact_id="synthetic-cohort-v1",
        split_version=split_id,
        ct_feature_artifact_id="synthetic-ct-features-v1",
        pathology_feature_artifact_id="synthetic-pathology-features-v1",
        timeline_contract_version="synthetic-timeline-v1",
        outcome_contract_version="synthetic-outcome-v1",
        training_seed=7,
        source_schema_version="synthetic-source-v1",
        cohort_schema_version="synthetic-cohort-build-v1",
        feature_schema_version="synthetic-features-v1",
        checkpoint_phase="baseline_evaluation",
        checkpoint_step=0,
        run_mode=mode,
    )
    records = tuple(
        PredictionRecord(
            patient_id=f"synthetic-{index}",
            stage="S1",
            query_time=100.0,
            endpoint="OS",
            horizon=365.0,
            risk=risk,
            fold=0,
            seed=7,
            model_name=model_name,
            input_artifact_id=input_id,
        )
        for index, risk in enumerate(risks)
    )
    return PredictionArtifact(lineage, records)


def test_matched_input_and_ablation_registry_contracts() -> None:
    main = _prediction_artifact("prediction-main", "model-main", "stageworld", (0.8, 0.2))
    baseline = _prediction_artifact(
        "prediction-baseline", "model-baseline", "direct-fusion", (0.7, 0.3)
    )
    matched = match_prediction_artifacts(main, baseline)
    assert matched.patient_ids == ("synthetic-0", "synthetic-1")
    assert matched.input_artifact_id == "synthetic-input-v1"
    unmatched = _prediction_artifact(
        "prediction-unmatched",
        "model-unmatched",
        "other",
        (0.7, 0.3),
        input_id="different-input-v1",
    )
    with pytest.raises(ValueError, match="identical input lineage"):
        match_prediction_artifacts(main, unmatched)
    other_split = _prediction_artifact(
        "prediction-other-split",
        "model-other-split",
        "other-split",
        (0.7, 0.3),
        split_id="synthetic-split-v2",
    )
    with pytest.raises(ValueError, match="cohort and split lineage"):
        match_prediction_artifacts(main, other_split)

    registry = default_ablation_registry()
    assert [entry.ablation_id for entry in registry] == [f"A{index}" for index in range(1, 11)]
    assert "real CT1 posterior update" in registry[0].held_constant
    assert "real S2 pathology posterior update" in registry[1].held_constant
    assert all(entry.status == "planned" for entry in registry)


def test_t49_prediction_metric_and_figure_lineage_is_traceable_and_namespaced(tmp_path) -> None:
    prediction = _prediction_artifact(
        "prediction-synthetic-v1", "model-synthetic-v1", "stageworld", (0.8, 0.2)
    )
    metric_lineage = build_metric_lineage(
        (prediction,),
        censoring_artifact_id="censoring-development-v1",
        protocol_id="protocol-v1",
        artifact_id="metric-synthetic-v1",
    )
    result = MetricResult(
        metric="ipcw_brier_score",
        status="ok",
        estimate=0.1,
        reason=None,
        horizon=365.0,
        n_patients=2,
        n_events=1,
        effective_n=2,
        lineage=metric_lineage,
    )
    figure = FigureLineage(
        artifact_id="calibration-figure-v1",
        figure_kind="calibration",
        prediction_artifact_ids=(prediction.lineage.artifact_id,),
        metric_artifact_ids=(metric_lineage.artifact_id,),
        protocol_id="protocol-v1",
        run_mode=RunMode.SYNTHETIC,
    )
    writer = EvaluationArtifactWriter(tmp_path, RunMode.SYNTHETIC)
    prediction_path = writer.write_predictions(prediction)
    metric_path = writer.write_metric(result)
    figure_path = writer.write_figure_lineage(figure)
    assert prediction_path.parent.name == "predictions"
    assert prediction_path.parents[1].name == "synthetic"
    assert metric_path.parents[1].name == "synthetic"
    assert figure_path.parents[1].name == "synthetic"
    assert read_prediction_artifact(prediction_path) == prediction
    assert prediction.lineage.input_artifact_id == "synthetic-input-v1"
    assert prediction.lineage.cohort_artifact_id == "synthetic-cohort-v1"
    assert prediction.lineage.split_version == "synthetic-split-v1"
    assert len(
        {
            prediction.lineage.input_artifact_id,
            prediction.lineage.cohort_artifact_id,
            prediction.lineage.split_version,
            prediction.lineage.ct_feature_artifact_id,
            prediction.lineage.pathology_feature_artifact_id,
        }
    ) == 5

    for legacy_version in ("stageworld.predictions.v1", "stageworld.predictions.v2"):
        legacy_prediction = prediction.as_dict()
        legacy_prediction["schema_version"] = legacy_version
        legacy_prediction["lineage"]["schema_version"] = legacy_version
        prediction_path.write_text(json.dumps(legacy_prediction), encoding="utf-8")
        with pytest.raises(ValueError, match="schema is unsupported"):
            read_prediction_artifact(prediction_path)

    incomplete_prediction = prediction.as_dict()
    del incomplete_prediction["lineage"]["ct_feature_artifact_id"]
    prediction_path.write_text(json.dumps(incomplete_prediction), encoding="utf-8")
    with pytest.raises(ValueError, match="ct_feature_artifact_id"):
        read_prediction_artifact(prediction_path)

    metric_payload = json.loads(metric_path.read_text(encoding="utf-8"))
    assert metric_payload["lineage"]["prediction_artifact_ids"] == ["prediction-synthetic-v1"]
    assert metric_payload["lineage"]["model_artifact_ids"] == ["model-synthetic-v1"]
    serialized = json.dumps(metric_payload).casefold()
    assert "checksum" not in serialized
    assert '"hash"' not in serialized

    real_writer = EvaluationArtifactWriter(tmp_path, RunMode.REAL)
    with pytest.raises(ValueError, match="run mode"):
        real_writer.write_predictions(prediction)
