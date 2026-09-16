from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml

import csv
import json
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from typer.testing import CliRunner

from stageworld.cli import app
from stageworld.evaluation import RunMode, read_prediction_artifact

ROOT = Path(__file__).resolve().parents[1]


def _compact_synthetic_config(tmp_path: Path) -> Path:
    payload = _release_yaml((ROOT / "configs/project.synthetic.yaml").read_text(encoding="utf-8"))
    run_root = tmp_path / "synthetic-run"
    payload["paths"]["output_root"] = str(run_root)
    payload["paths"]["feature_root"] = str(run_root / "features")
    payload["model"].update(
        {
            "hidden_dim": 8,
            "state_tokens": 2,
            "stochastic_dim_per_token": 2,
            "attention_heads": 2,
            "transition_blocks": 1,
            "observation_blocks": 1,
            "resampler_blocks": 1,
            "ct_tokens": 2,
            "pathology_tokens": 2,
        }
    )
    payload["training"].update(
        {
            "patient_batch_size": 12,
            "smoke_max_steps": 2,
            "smoke_max_minutes": 1,
            "world_pretrain_steps": 2,
            "joint_survival_steps": 2,
        }
    )
    config_path = tmp_path / "project.synthetic.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def _invoke(runner: CliRunner, *arguments: str) -> dict[str, Any]:
    result = runner.invoke(app, list(arguments))
    assert result.exit_code == 0, result.output
    try:
        payload = json.loads(result.output)
    except json.JSONDecodeError as error:
        pytest.fail(f"CLI did not emit one JSON document: {result.output!r}; {error}")
    assert payload["status"] == "ok"
    return payload


@pytest.mark.integration
def test_t45_synthetic_cli_chain_isolated_and_traceable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the complete offline synthetic release path from an empty directory."""

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config_path = _compact_synthetic_config(tmp_path)
    protocol_path = ROOT / "configs/evaluation.synthetic.yaml"
    run_root = tmp_path / "synthetic-run"
    runner = CliRunner()
    assert not run_root.exists()

    doctor = _invoke(runner, "doctor", "--config", str(config_path))
    generated = _invoke(
        runner,
        "make-synthetic",
        "--config",
        str(config_path),
        "--patient-count",
        "12",
    )
    audit = _invoke(runner, "audit", "--config", str(config_path))
    cohort = _invoke(runner, "build-cohort", "--config", str(config_path))
    ct = _invoke(
        runner,
        "extract-features",
        "--modality",
        "ct",
        "--config",
        str(config_path),
    )
    pathology = _invoke(
        runner,
        "extract-features",
        "--modality",
        "pathology",
        "--config",
        str(config_path),
    )
    world = _invoke(
        runner,
        "train",
        "--phase",
        "world_pretrain",
        "--config",
        str(config_path),
    )
    joint = _invoke(
        runner,
        "train",
        "--phase",
        "joint_survival",
        "--config",
        str(config_path),
    )
    prediction = _invoke(runner, "predict", "--config", str(config_path))
    evaluation = _invoke(
        runner,
        "evaluate",
        "--config",
        str(config_path),
        "--protocol",
        str(protocol_path),
    )
    report = _invoke(runner, "report", "--config", str(config_path))

    assert doctor["raw_paths_redacted"] is True
    assert generated["patient_count"] == 12
    assert audit["mode"] == cohort["mode"] == "synthetic"
    assert cohort["patient_count"] == 12
    assert sum(cohort["split_counts"].values()) == 12
    assert ct["data_lineage_id"] == pathology["data_lineage_id"]
    assert ct["data_lineage_id"] == generated["data_lineage_id"]
    assert world["optimizer_steps"] == joint["optimizer_steps"] == 2
    assert world["device"] == joint["device"] == "cpu"
    for training_summary in (world, joint):
        assert training_summary["schema_version"] == "stageworld-training-summary-v3"
        assert training_summary["training_split"] == "train"
        assert training_summary["validation_split"] == "validation"
        assert training_summary["training_patient_count"] == 7
        assert training_summary["validation_patient_count"] == 2
        assert training_summary["held_out_test_patient_count"] == 3
        assert training_summary["test_used_for_optimization_or_selection"] is False
    expected_parent_lineage = {
        "parent_checkpoint_id": world["checkpoint_id"],
        "parent_weight_version": world["weight_version"],
        "parent_phase": "world_pretrain",
        "parent_config_lineage_id": world["config_lineage_id"],
        "parent_data_lineage_id": world["data_lineage_id"],
        "parent_cohort_artifact_id": world["cohort_artifact_id"],
        "parent_split_version": world["split_version"],
        "parent_ct_feature_artifact_id": world["ct_feature_artifact_id"],
        "parent_pathology_feature_artifact_id": world["pathology_feature_artifact_id"],
        "parent_timeline_contract_version": world["timeline_contract_version"],
        "parent_outcome_contract_version": world["outcome_contract_version"],
    }
    assert {key: joint[key] for key in expected_parent_lineage} == expected_parent_lineage
    assert prediction["clinical_validation"] is False
    assert prediction["patients"] == 12
    assert prediction["queries"] == 35
    assert prediction["records"] == 70
    assert evaluation["clinical_validation"] is False
    assert evaluation["metric_count"] == 21
    assert report["clinical_validation"] is False

    source_manifest = json.loads(Path(generated["manifest"]).read_text(encoding="utf-8"))
    build_manifest = json.loads(Path(cohort["artifact"]).read_text(encoding="utf-8"))
    joint_sidecar = json.loads(
        (run_root / "runs/joint_survival/checkpoint_metadata.json").read_text(encoding="utf-8")
    )
    assert joint_sidecar["schema_version"] == "stageworld-checkpoint-sidecar-v2"
    joint_metadata = joint_sidecar["metadata"]
    joint_checkpoint = torch.load(
        run_root / "runs/joint_survival/checkpoint.pt", map_location="cpu", weights_only=True
    )
    assert joint_sidecar["active_weight_version"] == joint_checkpoint["weight_version"]
    prediction_artifact = read_prediction_artifact(Path(prediction["evaluation_predictions"]))
    evaluation_summary = json.loads(Path(evaluation["summary"]).read_text(encoding="utf-8"))
    report_manifest = json.loads(Path(report["manifest"]).read_text(encoding="utf-8"))

    expected_checkpoint_lineage = {
        "checkpoint_schema_version": joint_checkpoint["schema_version"],
        "checkpoint_id": joint_metadata["checkpoint_id"],
        "weight_version": joint_checkpoint["weight_version"],
        "model_version": joint_metadata["model_version"],
        "checkpoint_endpoint": joint_metadata["endpoint"],
        "config_lineage_id": joint_metadata["config_lineage_id"],
        "data_lineage_id": joint_metadata["data_lineage_id"],
        "cohort_artifact_id": joint_metadata["cohort_artifact_id"],
        "split_version": joint_metadata["split_version"],
        "ct_feature_artifact_id": joint_metadata["ct_feature_artifact_id"],
        "pathology_feature_artifact_id": joint_metadata["pathology_feature_artifact_id"],
        "timeline_contract_version": joint_metadata["timeline_contract_version"],
        "outcome_contract_version": joint_metadata["outcome_contract_version"],
        "training_seed": joint_metadata["training_seed"],
        "source_schema_version": joint_metadata["source_schema_version"],
        "cohort_schema_version": joint_metadata["cohort_schema_version"],
        "feature_schema_version": joint_metadata["feature_schema_version"],
        "checkpoint_phase": joint_metadata["phase"],
        "checkpoint_step": joint_checkpoint["trainer_state"]["optimizer_step"],
        **expected_parent_lineage,
    }

    data_lineage = source_manifest["data_lineage_id"]
    cohort_artifact_id = source_manifest["cohort_artifact_id"]
    split_version = build_manifest["split_version"]
    assert len({data_lineage, cohort_artifact_id, split_version}) == 3
    assert source_manifest["artifact_kind"] == "explicitly_synthetic"
    assert source_manifest["contains_real_clinical_data"] is False
    assert joint_metadata["data_lineage_id"] == data_lineage
    assert joint_metadata["split_version"] == build_manifest["split_version"]
    assert prediction_artifact.lineage.run_mode is RunMode.SYNTHETIC
    assert prediction_artifact.lineage.input_artifact_id == data_lineage
    assert prediction_artifact.lineage.cohort_artifact_id == cohort_artifact_id
    assert prediction_artifact.lineage.split_version == split_version
    assert prediction_artifact.lineage.ct_feature_artifact_id == ct["feature_artifact_id"]
    assert (
        prediction_artifact.lineage.pathology_feature_artifact_id
        == pathology["feature_artifact_id"]
    )
    assert prediction["checkpoint_id"] == joint["checkpoint_id"]
    assert prediction["weight_version"] == joint_checkpoint["weight_version"]
    assert prediction_artifact.lineage.model_artifact_id == joint_checkpoint["weight_version"]
    assert prediction_artifact.lineage.config_version == joint_metadata["config_lineage_id"]
    for key, expected in expected_checkpoint_lineage.items():
        assert getattr(prediction_artifact.lineage, key) == expected
        assert prediction[key] == expected
    assert evaluation_summary["prediction_artifact_id"] == (prediction_artifact.lineage.artifact_id)
    assert evaluation_summary["schema_version"] == "stageworld-evaluation-summary-v3"
    assert evaluation_summary["prediction_lineage"] == prediction_artifact.lineage.as_dict()
    assert evaluation_summary["checkpoint_id"] == joint["checkpoint_id"]
    assert evaluation_summary["weight_version"] == joint_checkpoint["weight_version"]
    assert evaluation_summary["config_lineage_id"] == joint_metadata["config_lineage_id"]
    assert evaluation_summary["data_lineage_id"] == data_lineage
    assert evaluation_summary["cohort_artifact_id"] == source_manifest["cohort_artifact_id"]
    assert evaluation_summary["split_version"] == build_manifest["split_version"]
    assert (
        evaluation_summary["timeline_contract_version"]
        == source_manifest["timeline_contract_version"]
    )
    assert (
        evaluation_summary["outcome_contract_version"]
        == source_manifest["outcome_contract_version"]
    )
    assert evaluation_summary["training_seed"] == joint_metadata["training_seed"]
    assert evaluation_summary["source_schema_version"] == joint_metadata["source_schema_version"]
    assert evaluation_summary["checkpoint_snapshot"] == joint["checkpoint_snapshot"]
    assert {
        key: evaluation_summary[key] for key in expected_parent_lineage
    } == expected_parent_lineage
    assert report_manifest["prediction_artifact_id"] == (prediction_artifact.lineage.artifact_id)
    assert report_manifest["schema_version"] == "stageworld-report-manifest-v3"
    assert report_manifest["joint_checkpoint_id"] == joint["checkpoint_id"]
    assert report_manifest["weight_version"] == joint_checkpoint["weight_version"]
    assert report_manifest["joint_weight_version"] == joint_checkpoint["weight_version"]
    assert report_manifest["prediction_lineage"] == prediction_artifact.lineage.as_dict()
    assert report_manifest["checkpoint_snapshot"] == joint["checkpoint_snapshot"]
    assert {key: report_manifest[key] for key in expected_parent_lineage} == expected_parent_lineage
    for carrier in (evaluation_summary, report_manifest):
        assert {key: carrier[key] for key in expected_checkpoint_lineage} == (
            expected_checkpoint_lineage
        )
    assert evaluation_summary["parent_checkpoint_snapshot"] == world["checkpoint_snapshot"]
    assert report_manifest["parent_checkpoint_snapshot"] == world["checkpoint_snapshot"]

    history = json.loads(Path(prediction["history_predictions"]).read_text(encoding="utf-8"))
    prediction_index = json.loads(Path(prediction["index"]).read_text(encoding="utf-8"))
    three_node_demo = json.loads(Path(prediction["three_node_demo"]).read_text(encoding="utf-8"))
    for carrier in (history, prediction_index, three_node_demo):
        assert {key: carrier[key] for key in expected_checkpoint_lineage} == (
            expected_checkpoint_lineage
        )
    assert history["checkpoint_id"] == prediction_index["checkpoint_id"] == joint["checkpoint_id"]
    assert (
        history["weight_version"]
        == prediction_index["weight_version"]
        == joint_checkpoint["weight_version"]
    )
    history_rows = history["predictions"]
    assert all(row["checkpoint_id"] == joint["checkpoint_id"] for row in history_rows)
    assert all(row["weight_version"] == joint_checkpoint["weight_version"] for row in history_rows)
    assert {
        stage: sum(row["stage"] == stage for row in history_rows) for stage in ("s0", "s1", "s2")
    } == {"s0": 12, "s1": 12, "s2": 11}
    assert all(row["simulated"] is False for row in history_rows)
    assert all(row["scenario"]["kind"] == "observed_history" for row in history_rows)
    assert all("synthetic_input" in row["quality_flags"] for row in history_rows)

    original_config = _release_yaml(config_path.read_text(encoding="utf-8"))
    changed_seed = dict(original_config)
    changed_seed["training"] = {**original_config["training"], "seed": 18}
    changed_seed_path = tmp_path / "changed-seed.synthetic.yaml"
    changed_seed_path.write_text(yaml.safe_dump(changed_seed, sort_keys=False), encoding="utf-8")
    rejected_seed = runner.invoke(app, ["predict", "--config", str(changed_seed_path)])
    assert rejected_seed.exit_code == 2
    assert json.loads(rejected_seed.output)["error"]["code"] == "SYNTHETIC_SOURCE_CONFIG_MISMATCH"

    changed_cutpoints = dict(original_config)
    changed_cutpoints["survival"] = {
        **original_config["survival"],
        "finite_cutpoints": [0.0, 1.5, 3.0, 5.0],
    }
    changed_cutpoints_path = tmp_path / "changed-cutpoints.synthetic.yaml"
    changed_cutpoints_path.write_text(
        yaml.safe_dump(changed_cutpoints, sort_keys=False), encoding="utf-8"
    )
    rejected_cutpoints = runner.invoke(
        app, ["predict", "--config", str(changed_cutpoints_path)]
    )
    assert rejected_cutpoints.exit_code == 2
    assert (
        json.loads(rejected_cutpoints.output)["error"]["code"]
        == "CHECKPOINT_SURVIVAL_CONTRACT_MISMATCH"
    )

    metric_paths = [Path(path) for path in evaluation_summary["metric_artifacts"]]
    assert metric_paths
    assert {
        stage: sum(row["stage"] == stage for row in evaluation_summary["metrics"])
        for stage in ("s0", "s1", "s2")
    } == {"s0": 7, "s1": 7, "s2": 7}
    for metric_path in metric_paths:
        metric = json.loads(metric_path.read_text(encoding="utf-8"))
        assert metric["lineage"]["prediction_artifact_ids"] == [
            prediction_artifact.lineage.artifact_id
        ]
        assert metric["lineage"]["protocol_id"] == "synthetic-os-fixed-horizons-v1"

    calibration_plot = Path(evaluation["calibration_plot"])
    assert calibration_plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    figure_lineage = json.loads(Path(evaluation["figure_lineage"]).read_text(encoding="utf-8"))
    assert figure_lineage["prediction_artifact_ids"] == [prediction_artifact.lineage.artifact_id]
    report_text = Path(report["report"]).read_text(encoding="utf-8")
    assert "fictional synthetic data" in report_text
    assert "not clinical validation" in report_text
    estimable_count = sum(row["status"] == "ok" for row in evaluation_summary["metrics"])
    not_estimable_count = sum(
        row["status"] == "not_estimable" for row in evaluation_summary["metrics"]
    )
    assert f"- Estimable metric results: {estimable_count}" in report_text
    assert (
        f"- Explicitly not-estimable metric results: {not_estimable_count}" in report_text
    )
    assert "estimable synthetic metrics demonstrate" not in report_text

    evaluation_summary_path = Path(evaluation["summary"])
    tampered_evaluation = dict(evaluation_summary)
    tampered_evaluation["parent_ct_feature_artifact_id"] = "tampered-parent-ct-features"
    evaluation_summary_path.write_text(
        json.dumps(tampered_evaluation), encoding="utf-8"
    )
    rejected_tampered_report = runner.invoke(
        app, ["report", "--config", str(config_path)]
    )
    assert rejected_tampered_report.exit_code == 2
    tampered_error = json.loads(rejected_tampered_report.output)["error"]
    assert tampered_error["code"] == "REPORT_EVALUATION_LINEAGE_MISMATCH"
    assert "parent_ct_feature_artifact_id" in tampered_error["details"]["fields"]
    evaluation_summary_path.write_text(json.dumps(evaluation_summary), encoding="utf-8")

    with (run_root / "experiment_registry.csv").open(newline="", encoding="utf-8") as handle:
        registry = list(csv.DictReader(handle))
    assert {(row["phase"], row["status"]) for row in registry} == {
        ("world_pretrain", "completed"),
        ("joint_survival", "completed"),
    }
    for path in (
        Path(doctor["report"]),
        Path(generated["cohort"]),
        Path(ct["artifact"]),
        Path(pathology["artifact"]),
        Path(world["checkpoint"]),
        Path(joint["checkpoint"]),
        Path(prediction["history_predictions"]),
        Path(evaluation["summary"]),
        Path(report["report"]),
    ):
        assert path.is_file()
        assert path.is_relative_to(run_root)

    original_snapshot = Path(joint["checkpoint_snapshot"])
    assert original_snapshot.is_file()
    joint_pointer_path = run_root / "runs/joint_survival/checkpoint.pt"
    original_pointer_payload = torch.load(
        joint_pointer_path, map_location="cpu", weights_only=True
    )
    tampered_pointer_payload = dict(original_pointer_payload)
    tampered_pointer_state = dict(original_pointer_payload["model_state"])
    tensor_name = next(iter(tampered_pointer_state))
    tampered_pointer_state[tensor_name] = tampered_pointer_state[tensor_name].clone()
    tampered_pointer_state[tensor_name].view(-1)[0] += 1.0
    tampered_pointer_payload["model_state"] = tampered_pointer_state
    versions_before_rejected_resume = set(original_snapshot.parent.glob("*.pt"))
    torch.save(tampered_pointer_payload, joint_pointer_path)
    rejected_pointer_resume = runner.invoke(
        app,
        [
            "train",
            "--phase",
            "joint_survival",
            "--resume",
            "--config",
            str(config_path),
        ],
    )
    assert rejected_pointer_resume.exit_code == 2
    assert (
        json.loads(rejected_pointer_resume.output)["error"]["code"]
        == "RESUME_CHECKPOINT_SNAPSHOT_MISMATCH"
    )
    rejected_pointer_prediction = runner.invoke(
        app, ["predict", "--config", str(config_path)]
    )
    assert rejected_pointer_prediction.exit_code == 2
    assert (
        json.loads(rejected_pointer_prediction.output)["error"]["code"]
        == "CHECKPOINT_POINTER_SNAPSHOT_MISMATCH"
    )
    assert set(original_snapshot.parent.glob("*.pt")) == versions_before_rejected_resume
    torch.save(original_pointer_payload, joint_pointer_path)

    parent_snapshot_path = Path(world["checkpoint_snapshot"])
    original_parent_payload = torch.load(
        parent_snapshot_path, map_location="cpu", weights_only=True
    )
    tampered_parent_payload = dict(original_parent_payload)
    tampered_parent_state = dict(original_parent_payload["model_state"])
    parent_tensor_name = next(iter(tampered_parent_state))
    tampered_parent_state[parent_tensor_name] = tampered_parent_state[
        parent_tensor_name
    ].clone()
    tampered_parent_state[parent_tensor_name].view(-1)[0] += 1.0
    tampered_parent_payload["model_state"] = tampered_parent_state
    torch.save(tampered_parent_payload, parent_snapshot_path)
    rejected_parent_report = runner.invoke(
        app, ["report", "--config", str(config_path)]
    )
    assert rejected_parent_report.exit_code == 2
    assert (
        json.loads(rejected_parent_report.output)["error"]["code"]
        == "PARENT_CHECKPOINT_STATE_MISMATCH"
    )
    rejected_parent_resume = runner.invoke(
        app,
        [
            "train",
            "--phase",
            "joint_survival",
            "--resume",
            "--config",
            str(config_path),
        ],
    )
    assert rejected_parent_resume.exit_code == 2
    assert (
        json.loads(rejected_parent_resume.output)["error"]["code"]
        == "PARENT_CHECKPOINT_STATE_MISMATCH"
    )
    torch.save(original_parent_payload, parent_snapshot_path)

    world_pointer_path = run_root / "runs/world_pretrain/checkpoint.pt"
    original_world_pointer = torch.load(
        world_pointer_path, map_location="cpu", weights_only=True
    )
    tampered_world_pointer = dict(original_world_pointer)
    tampered_world_state = dict(original_world_pointer["model_state"])
    world_tensor_name = next(iter(tampered_world_state))
    tampered_world_state[world_tensor_name] = tampered_world_state[world_tensor_name].clone()
    tampered_world_state[world_tensor_name].view(-1)[0] += 1.0
    tampered_world_pointer["model_state"] = tampered_world_state
    torch.save(tampered_world_pointer, world_pointer_path)
    rejected_parent_transfer = runner.invoke(
        app,
        [
            "train",
            "--phase",
            "joint_survival",
            "--config",
            str(config_path),
        ],
    )
    assert rejected_parent_transfer.exit_code == 2
    assert (
        json.loads(rejected_parent_transfer.output)["error"]["code"]
        == "PARENT_CHECKPOINT_SNAPSHOT_MISMATCH"
    )
    torch.save(original_world_pointer, world_pointer_path)

    resumed_joint = _invoke(
        runner,
        "train",
        "--phase",
        "joint_survival",
        "--resume",
        "--config",
        str(config_path),
    )
    assert resumed_joint["checkpoint_id"] == joint["checkpoint_id"]
    assert resumed_joint["weight_version"] != joint["weight_version"]
    assert original_snapshot.is_file()
    assert Path(resumed_joint["checkpoint_snapshot"]).is_file()
    assert {
        key: resumed_joint[key] for key in expected_parent_lineage
    } == expected_parent_lineage
    retained_payload = torch.load(original_snapshot, map_location="cpu", weights_only=True)
    retained_metadata = retained_payload["metadata"]
    assert retained_payload["weight_version"] == joint["weight_version"]
    assert {
        key: retained_metadata[key] for key in expected_parent_lineage
    } == expected_parent_lineage
    stale_report = runner.invoke(app, ["report", "--config", str(config_path)])
    assert stale_report.exit_code == 2
    stale_error = json.loads(stale_report.output)["error"]
    assert stale_error["code"] == "PREDICTION_CHECKPOINT_LINEAGE_MISMATCH"
    assert "model_artifact_id" in stale_error["details"]["fields"]
