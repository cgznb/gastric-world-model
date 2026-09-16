"""Portable generated-S1 inference; no paired-cohort or outcome loader is used."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from stageworld.artifacts import atomic_write_private_json, read_json
from stageworld.data.baseline_clinical import (
    CT6_CLINICAL_SCHEMA,
    encode_baseline,
    parse_baseline_fields,
)
from stageworld.data.treatment_compact import (
    compact_actions,
    encode_compact,
    normalize_treatment,
    treatment_fields,
)
from stageworld.data.treatment_regimens import encode_regimen_values, regimen_summary_actions
from stageworld.encoders.base import EncoderProvenance, ObservationTokens
from stageworld.errors import DataContractError
from stageworld.model.generated_s1 import GeneratedS1Config, GeneratedS1Model
from stageworld.synthetic_workflow import _atomic_torch_save

INFERENCE_SCHEMA = "generated-s1-portable-inference-v1"
CT6_INFERENCE_SCHEMA = "generated-s1-portable-inference-v2"
CT0_PACKET_SCHEMA = "baseline-only-ct-feature-packet-v1"
QUERY_FIELDS = {
    "ct0_features",
    "baseline_clinical",
    "s0_time_days",
    "target_interval_days",
    "horizons_years",
    "treatment_scenario",
}


def export_inference_bundle(
    model: GeneratedS1Model,
    path: Path,
    *,
    clinical_snapshot: Mapping[str, Any],
    cycle_transform: Mapping[str, Any] | None = None,
    treatment_support: Mapping[str, Any] | None = None,
    encoder_provenance: EncoderProvenance,
    weight_version: str,
) -> None:
    compact = model.config.clinical_schema_version == CT6_CLINICAL_SCHEMA
    schema = CT6_INFERENCE_SCHEMA if compact else INFERENCE_SCHEMA
    if (compact and treatment_support is None) or (not compact and cycle_transform is None):
        raise DataContractError(code="INFERENCE_TRANSFORM_MISSING", message="Bind transforms.")
    _atomic_torch_save(
        path,
        {
            "schema_version": schema,
            "branch": "S1_pred",
            "survival_task": model.config.survival_task,
            "ct1_required": False,
            "future_ct_objective": "huber_cosine",
            "future_ct_variance_supervised": False,
            "weight_version": weight_version,
            "model_config": asdict(model.config),
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "clinical_transform": clinical_snapshot["transform"],
            "clinical_snapshot_id": clinical_snapshot["artifact_id"],
            **(
                {"treatment_support": dict(treatment_support or {})}
                if compact
                else {"cycle_transform": dict(cycle_transform or {})}
            ),
            "encoder_provenance": encoder_provenance.as_dict(),
            "time_unit": "year",
            "timeline_unit": "day",
        },
    )
    atomic_write_private_json(
        path.with_suffix(".json"),
        {
            "schema_version": schema,
            "checkpoint": path.name,
            "weight_version": weight_version,
            "branch": "S1_pred",
            "readout_mode": model.config.readout_mode,
            "survival_task": model.config.survival_task,
            "future_ct_objective": "huber_cosine",
            "future_ct_variance_supervised": False,
        },
    )


def ct0_packet(ct0: ObservationTokens) -> dict[str, Any]:
    return {
        "schema_version": CT0_PACKET_SCHEMA,
        "values": ct0.values.cpu(),
        "valid": ct0.valid.cpu(),
        "acquired_time": ct0.acquired_time.cpu(),
        "available_time": ct0.available_time.cpu(),
        "coords": None if ct0.coords is None else ct0.coords.cpu(),
        "coordinate_system": ct0.coordinate_system,
        "encoder_provenance": ct0.provenance.as_dict(),
    }


def load_ct0_packet(path: Path, expected: Mapping[str, Any]) -> ObservationTokens:
    packet = torch.load(path, weights_only=True, map_location="cpu")
    if (
        packet.get("schema_version") != CT0_PACKET_SCHEMA
        or packet.get("encoder_provenance") != expected
    ):
        raise DataContractError(
            code="CT0_PACKET_CONTRACT", message="Bind matching baseline features."
        )
    values, valid = packet["values"], packet["valid"]
    if values.shape[0] != 1:
        raise DataContractError(
            code="QUERY_BATCH_SIZE", message="Each query describes one patient."
        )
    return ObservationTokens(
        values=values,
        valid=valid,
        modality=torch.zeros_like(valid, dtype=torch.long),
        acquired_time=packet["acquired_time"],
        available_time=packet["available_time"],
        provenance=EncoderProvenance.from_dict(packet["encoder_provenance"]),
        source_id=(tuple("baseline_feature_packet" for _ in range(values.shape[1])),),
        modality_name="ct",
        coords=packet["coords"],
        coordinate_system=packet["coordinate_system"],
    )


@torch.inference_mode()
def predict_generated_query(
    config_file: Path,
    query_file: Path,
    output_file: Path,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    config, query = read_json(config_file), read_json(query_file)
    schema = config.get("schema_version")
    if schema == "generated-s1-binary-endpoints-inference-v1":
        from stageworld.binary_inference import predict_binary_query

        return predict_binary_query(config_file, query_file, output_file, device=device)
    if schema not in (INFERENCE_SCHEMA, CT6_INFERENCE_SCHEMA) or set(query) != QUERY_FIELDS:
        raise DataContractError(
            code="GENERATED_QUERY_SCHEMA", message="Use the baseline-only schema."
        )
    payload = torch.load(
        config_file.parent / config["checkpoint"], weights_only=True, map_location="cpu"
    )
    if (
        payload.get("schema_version") != schema
        or payload.get("branch") != "S1_pred"
        or payload.get("weight_version") != config.get("weight_version")
        or payload.get("time_unit") != "year"
        or payload.get("timeline_unit") != "day"
        or payload.get("ct1_required") is not False
    ):
        raise DataContractError(
            code="GENERATED_CHECKPOINT_CONTRACT", message="Invalid model bundle."
        )
    model = GeneratedS1Model(GeneratedS1Config(**payload["model_config"]))
    if (
        payload.get("survival_task", "s0_s1_pred") != model.config.survival_task
        or config.get("survival_task", "s0_s1_pred") != model.config.survival_task
    ):
        raise DataContractError(code="GENERATED_TASK_MISMATCH", message="Survival task differs.")
    compact = model.config.clinical_schema_version == CT6_CLINICAL_SCHEMA
    if compact != (schema == CT6_INFERENCE_SCHEMA) or (compact and "cycle_transform" in payload):
        raise DataContractError(code="GENERATED_SCHEMA_MISMATCH", message="Model schema differs.")
    if model.config.readout_mode != config.get("readout_mode"):
        raise DataContractError(code="GENERATED_READOUT_MISMATCH", message="Readout differs.")
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    ct0 = load_ct0_packet(query_file.parent / query["ct0_features"], payload["encoder_provenance"])
    from stageworld.training import _move_observation

    ct0 = _move_observation(ct0, torch.device(device))
    baseline = encode_baseline(
        [
            parse_baseline_fields(
                query["baseline_clinical"], schema_version=model.config.clinical_schema_version
            )
        ],
        payload["clinical_transform"],
    ).to(device)
    scenario = query["treatment_scenario"]
    if (
        not isinstance(scenario, dict)
        or set(scenario)
        != (
            {"description", "methods", "regimens", "drugs"}
            if compact
            else {"description", "methods", "named_mentions", "cycles"}
        )
        or not isinstance(scenario["description"], str)
        or not scenario["description"].strip()
    ):
        raise DataContractError(code="SCENARIO_REQUIRED", message="Declare the interval scenario.")
    s0 = torch.tensor([float(query["s0_time_days"])], device=device)
    interval = torch.tensor([float(query["target_interval_days"])], device=device)
    if not torch.isfinite(interval).all() or (interval <= 0).any():
        raise DataContractError(
            code="TARGET_INTERVAL_INVALID", message="Use a positive day interval."
        )
    target = s0 + interval
    support_flags: tuple[str, ...] = ()
    provenance = "caller_declared_scenario_assumed_completed_by_target_not_actual_dosing"
    if compact:
        row = normalize_treatment(treatment_fields(scenario))
        values, flags = encode_compact([row], payload["treatment_support"])
        support_flags = flags[0]
        actions = compact_actions(values.to(device), target, provenance=provenance)
    else:
        actions = regimen_summary_actions(
            encode_regimen_values([scenario], payload["cycle_transform"]).to(device),
            target,
            provenance=provenance,
        )
    output = model.predict_generated_s1(
        ct0=ct0,
        baseline=baseline,
        s0_time=s0,
        scenario_actions=actions,
        target_time=target,
        horizons=torch.tensor(query["horizons_years"], device=device, dtype=torch.float32),
        scenario=scenario["description"],
        deterministic=True,
    )
    state_path = output_file.with_suffix(".pt")
    if state_path == output_file:
        raise DataContractError(code="OUTPUT_PATH_INVALID", message="Use a JSON output filename.")
    _atomic_torch_save(
        state_path,
        {
            "schema_version": schema,
            "branch": "S1_pred",
            "memory": output.state_s1_pred.memory.cpu(),
            "stochastic_mean": None
            if output.state_s1_pred.stochastic_mean is None
            else output.state_s1_pred.stochastic_mean.cpu(),
            "stochastic_log_std": None
            if output.state_s1_pred.stochastic_log_std is None
            else output.state_s1_pred.stochastic_log_std.cpu(),
            "future_ct_mean": output.future_ct.mean.cpu(),
            "future_ct_log_std": output.future_ct.log_std.cpu(),
            "future_ct_variance_supervised": output.future_ct_variance_supervised,
            "target_time_days": target.cpu(),
        },
    )
    result = {
        "schema_version": schema,
        "branch": "S1_pred",
        "simulated": True,
        "ct1_used": False,
        "outcomes_used": False,
        "readout_mode": model.config.readout_mode,
        "weight_version": payload["weight_version"],
        "future_ct_objective": output.future_ct_objective,
        "future_ct_variance_supervised": output.future_ct_variance_supervised,
        "future_ct_uncertainty_status": "not_trained_or_calibrated",
        "s0_time_days": s0.tolist(),
        "target_time_days": target.tolist(),
        "target_interval_days": interval.tolist(),
        "horizons_years": query["horizons_years"],
        "survival_origin": "target_S1_conditional_on_alive_at_target",
        "scenario_semantics": "declared_condition_assumed_completed_by_target_not_causal_effect",
        "rates": output.survival_s1_pred.rates.cpu().tolist(),
        "survival": output.survival_s1_pred.survival.cpu().tolist(),
        "risk": output.survival_s1_pred.risk.cpu().tolist(),
        "state_artifact": state_path.name,
        "quality_flags": [*output.survival_s1_pred.quality_flags, *support_flags],
        **(
            {
                "clinical_schema": CT6_CLINICAL_SCHEMA,
                "treatment_schema": payload["treatment_support"]["schema_version"],
                "cycle_counts_used": False,
            }
            if compact
            else {}
        ),
    }
    atomic_write_private_json(output_file, result)
    return {
        "status": "predicted",
        "branch": "S1_pred",
        "ct1_used": False,
        "readout_mode": model.config.readout_mode,
        "query_count": 1,
    }
