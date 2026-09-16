"""Audit completed weights and clarify reporting without changing trained parameters."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from stageworld.artifacts import atomic_write_private_json, read_json
from stageworld.generated_evaluation import annotate_paired_interval_support

FEATURE_NOTES = {
    "future_ct_objective": "huber_cosine",
    "future_ct_variance_supervised": False,
    "future_ct_uncertainty_status": "not_trained_or_calibrated",
}


def preserve_json(path: Path) -> None:
    archive = path.with_name(f"{path.stem}.training-complete.json")
    if not archive.exists():
        shutil.copy2(path, archive)


def finite_tensors(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("A completed checkpoint contains a nonfinite tensor")
        return 1
    if isinstance(value, Mapping):
        return sum(finite_tensors(child) for child in value.values())
    if isinstance(value, (tuple, list)):
        return sum(finite_tensors(child) for child in value)
    return 0


def finalize(study: Path) -> dict[str, Any]:
    torch.set_num_threads(4)
    summary_path = study / "summary.json"
    summary = read_json(summary_path)
    inference = read_json(study / "independent_inference_verification.json")
    if summary["status"] != "completed" or inference["status"] != "passed":
        raise ValueError("Complete training and independent inference verification first")
    expected_epochs = 1 if summary["smoke"] else 100
    steps = summary["pretraining"]["steps_per_epoch"]
    parent = torch.load(study / "world_pretrain/final.pt", map_location="cpu", weights_only=True)
    checkpoints = {"world_pretrain/final": parent}
    for mode, arm in summary["arms"].items():
        if (
            arm["completed_epochs"] != expected_epochs
            or arm["optimizer_steps"] != expected_epochs * steps
            or arm["verification"]["status"] != "passed"
        ):
            raise ValueError("Joint training budget or checkpoint audit differs")
        history = read_json(study / mode / "history.json")["epochs"]
        chosen = min(history, key=lambda row: row["validation_s1_pred_nll"])
        if chosen["epoch"] != arm["selected_epoch"]:
            raise ValueError("Selected epoch does not minimize generated validation NLL")
        for kind in ("selected", "final"):
            payload = torch.load(study / mode / f"{kind}.pt", map_location="cpu", weights_only=True)
            epoch = arm["selected_epoch"] if kind == "selected" else expected_epochs
            if (
                payload["trainer_state"]["optimizer_step"] != epoch * steps
                or payload["sampler_state"]["epoch"] != epoch
                or payload["model_config"]["readout_mode"] != mode
                or payload["metadata"]["parent_weight_version"] != parent["weight_version"]
                or any(
                    not torch.equal(value, parent["model_state"][name])
                    for name, value in payload["transfer_parent_model_state"].items()
                )
            ):
                raise ValueError("Checkpoint step, readout or shared parent mismatch")
            checkpoints[f"{mode}/{kind}"] = payload
        saved = torch.load(study / mode / "predictions.pt", map_location="cpu", weights_only=True)
        if not torch.isfinite(saved["rates"]).all() or not (saved["rates"] > 0).all():
            raise ValueError("Saved prediction rates are invalid")
        arm["future_ct"].update(FEATURE_NOTES)
        preserve_json(study / mode / "summary.json")
        atomic_write_private_json(study / mode / "summary.json", arm)
        config_path = study / mode / "inference.json"
        config = read_json(config_path)
        config.update(FEATURE_NOTES)
        preserve_json(config_path)
        atomic_write_private_json(config_path, config)
    if (
        parent["trainer_state"]["optimizer_step"] != expected_epochs * steps
        or parent["sampler_state"]["epoch"] != expected_epochs
        or summary["pretraining"]["completed_epochs"] != expected_epochs
        or not summary["landmark_labels_verified"]
    ):
        raise ValueError("World epoch count or landmark-label verification differs")
    summary["pretraining"]["future_ct"].update(FEATURE_NOTES)
    for row in summary.get("paired_comparisons", []):
        annotate_paired_interval_support(row)
    ridge = torch.load(study / "ridge.pt", map_location="cpu", weights_only=True)
    if (
        not torch.isfinite(ridge["rates"]).all()
        or not (ridge["rates"] > 0).all()
        or ridge["input_contract"] != "CT0_baseline19_declared_scenario_target_no_CT1"
    ):
        raise ValueError("Matched ridge prediction contract differs")
    audit = {
        "status": "passed",
        "checkpoints_checked": len(checkpoints),
        "finite_tensors_per_checkpoint": {
            name: finite_tensors(payload) for name, payload in checkpoints.items()
        },
        "epochs_per_phase": expected_epochs,
        "updates_per_phase": expected_epochs * steps,
        "selected_epochs": {mode: arm["selected_epoch"] for mode, arm in summary["arms"].items()},
        "parent_parameter_identity_verified": True,
        "landmark_labels_verified": True,
        "independent_inference_verified": True,
        "test_used": False,
        "one_year_S1_event_support": next(
            row["n_events"]
            for row in summary["arms"]["history_generated"]["metrics"]
            if row["stage"] == "S1_pred"
            and row["metric"] == "cumulative_dynamic_auc"
            and row["horizon"] == 1.0
        ),
        "unsupported_paired_intervals_suppressed": sum(
            row["confidence_interval_status"] == "insufficient_valid_resamples"
            for row in summary.get("paired_comparisons", [])
        ),
        "trained_parameter_files_modified": False,
    }
    summary["final_verification"] = audit
    summary["reporting_clarification"] = {
        "feature_distribution_variance": "unsupervised_output_not_calibrated_uncertainty",
        "generation_diversity": "between_patient_variance_of_predicted_feature_means",
        "paired_intervals": "same_minimum_80_percent_valid_draw_rule_as_individual_metrics",
    }
    preserve_json(summary_path)
    atomic_write_private_json(summary_path, summary)
    atomic_write_private_json(study / "final_verification.json", audit)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("study", type=Path)
    print(json.dumps(finalize(parser.parse_args().study)))
