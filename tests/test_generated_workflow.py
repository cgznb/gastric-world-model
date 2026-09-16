from __future__ import annotations

import builtins
import io
import json
from dataclasses import replace
from pathlib import Path

import torch
from test_generated_s1 import clinical_rows, generated_batch, generated_model
from typer.testing import CliRunner

from stageworld.cli import app
from stageworld.data.baseline_clinical import fit_clinical_transform
from stageworld.data.treatment_regimens import (
    CYCLE_FIELDS,
    MENTION_NAMES,
    encode_regimen_values,
    encode_regimens,
    regimen_summary_actions,
)
from stageworld.data.treatment_summary import METHOD_NAMES
from stageworld.generated_evaluation import matched_baseline_inputs, paired_comparisons
from stageworld.generated_inference import (
    ct0_packet,
    export_inference_bundle,
    predict_generated_query,
)
from stageworld.model.generated_s1 import GeneratedS1Model
from stageworld.synthetic_workflow import _atomic_torch_save


def scenario():
    return {
        "methods": dict.fromkeys(METHOD_NAMES),
        "named_mentions": dict.fromkeys(MENTION_NAMES),
        "cycles": dict.fromkeys(name for name, _, _ in CYCLE_FIELDS),
    }


def test_standalone_regimen_encoding_matches_original_adapter():
    batch = generated_batch((0,))
    row = scenario()
    row["cycles"]["chemotherapy"] = 4
    row["named_mentions"]["regimen_sox"] = 1
    transform = {name: {"log_mean": 0.4, "log_scale": 1.1} for name, _, _ in CYCLE_FIELDS}
    legacy = encode_regimens(batch, {batch.patient_ids[0]: row}, transform)
    direct = regimen_summary_actions(
        encode_regimen_values([row], transform),
        batch.ct1_acquisition_time,
        provenance="declared_scenario",
    )
    for name in ("values", "valid", "event_time", "available_time", "planned_or_delivered"):
        assert torch.equal(getattr(legacy.treatment_actions, name), getattr(direct, name))


def test_independent_query_and_cli_work_with_all_unapproved_input_files_denied(
    tmp_path, monkeypatch
):
    batch = generated_batch((0,))
    model = GeneratedS1Model(replace(generated_model().config, action_input_dim=45)).eval()
    transform = {name: {"log_mean": 0.0, "log_scale": 1.0} for name, _, _ in CYCLE_FIELDS}
    rows = clinical_rows(1)
    export_inference_bundle(
        model,
        tmp_path / "inference.pt",
        clinical_snapshot={
            "artifact_id": "synthetic-baseline",
            "transform": fit_clinical_transform(rows, set(rows)),
        },
        cycle_transform=transform,
        encoder_provenance=batch.ct0.provenance,
        weight_version="synthetic-weights",
    )
    _atomic_torch_save(tmp_path / "ct0.pt", ct0_packet(batch.ct0))
    query = {
        "ct0_features": "ct0.pt",
        "baseline_clinical": rows["person-0"],
        "s0_time_days": float(batch.s0_time[0]),
        "target_interval_days": 100,
        "horizons_years": [1.0, 3.0],
        "treatment_scenario": {"description": "specified", **scenario()},
    }
    (tmp_path / "query.json").write_text(json.dumps(query))
    allowed = {
        tmp_path / name for name in ("inference.pt", "inference.json", "query.json", "ct0.pt")
    }
    opened = []

    def guarded(original):
        def open_file(file, mode="r", *args, **kwargs):
            if isinstance(file, (str, Path)) and "r" in mode:
                path = Path(file).resolve()
                assert path in allowed, f"Unauthorized read: {path.name}"
                opened.append(path.name)
            return original(file, mode, *args, **kwargs)

        return open_file

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", guarded(builtins.open))
        patch.setattr(io, "open", guarded(io.open))
        result = predict_generated_query(
            tmp_path / "inference.json", tmp_path / "query.json", tmp_path / "output.json"
        )
    assert result["ct1_used"] is False
    assert set(opened) == {path.name for path in allowed}
    output = json.loads((tmp_path / "output.json").read_text())
    assert output["target_time_days"] == [float(batch.s0_time[0]) + 100]
    assert output["survival_origin"] == "target_S1_conditional_on_alive_at_target"
    assert output["future_ct_variance_supervised"] is False
    assert output["future_ct_uncertainty_status"] == "not_trained_or_calibrated"
    cli = CliRunner().invoke(
        app,
        [
            "predict-generated-s1",
            "--config",
            str(tmp_path / "inference.json"),
            "--query-file",
            str(tmp_path / "query.json"),
            "--output-file",
            str(tmp_path / "cli.json"),
        ],
    )
    assert cli.exit_code == 0, cli.output
    assert json.loads((tmp_path / "cli.json").read_text())["rates"] == output["rates"]


def test_ridge_has_same_baseline_permission_and_ignores_ct1():
    batch = generated_batch()
    modified = replace(batch, ct1=replace(batch.ct1, values=batch.ct1.values + 999))
    for stage in (0, 1):
        assert torch.equal(
            matched_baseline_inputs([batch], stage), matched_baseline_inputs([modified], stage)
        )


def test_identical_predictions_have_zero_paired_differences():
    train = generated_batch()
    validation = replace(generated_batch(), patient_ids=tuple(f"val-{i}" for i in range(4)))
    rates = torch.full((4, 2, 2), 0.2)
    rows = paired_comparisons(
        {"history_generated": rates, "history_only": rates.clone()},
        [train],
        [validation],
        (0.0, 1.0, 3.0),
        replicates=20,
    )
    assert rows
    for row in rows:
        if row["status"] == "ok":
            assert row["estimate_difference"] == 0


def test_one_event_paired_bootstrap_does_not_report_an_unsupported_interval():
    batch = generated_batch()
    durations = torch.tensor([[0.7, 0.5, 0.0], [2.2, 2.0, 0.0], [3.2, 3.0, 0.0], [4.2, 4.0, 0.0]])
    events = torch.zeros(4, 3, dtype=torch.long)
    events[0, :2] = 1
    train = replace(batch, survival_durations=durations, survival_events=events)
    validation = replace(train, patient_ids=tuple(f"val-{i}" for i in range(4)))
    rates = torch.full((4, 2, 2), 0.2)
    rows = paired_comparisons(
        {"history_generated": rates, "history_only": rates.clone()},
        [train],
        [validation],
        (0.0, 1.0, 3.0),
        replicates=20,
    )
    row = next(
        r
        for r in rows
        if r["stage"] == "S1_pred"
        and r["metric"] == "cumulative_dynamic_auc"
        and r["horizon"] == 1.0
    )
    assert row["estimate_difference"] == 0
    assert 0 < row["n_estimable_resamples"] < row["minimum_valid_bootstrap_replicates"]
    assert row["confidence_lower"] is None and row["confidence_upper"] is None
    assert row["confidence_interval_status"] == "insufficient_valid_resamples"
