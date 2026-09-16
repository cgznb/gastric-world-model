"""Portable pCR/recurrence inference from four baseline-only files."""

from __future__ import annotations

import builtins
import io
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

from stageworld.artifacts import atomic_write_private_json, read_json
from stageworld.binary_endpoints import LABEL_CONTRACT, BinaryEndpointConfig, BinaryEndpointModel
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
from stageworld.encoders.base import EncoderProvenance
from stageworld.errors import DataContractError
from stageworld.generated_inference import load_ct0_packet
from stageworld.synthetic_workflow import _atomic_torch_save
from stageworld.training import _move_observation

BINARY_INFERENCE_SCHEMA = "generated-s1-binary-endpoints-inference-v1"


def export_binary_bundle(
    model: BinaryEndpointModel,
    path: Path,
    *,
    snapshot: dict[str, Any],
    encoder_provenance: EncoderProvenance,
    weight_version: str,
) -> None:
    contract = {
        "schema_version": BINARY_INFERENCE_SCHEMA,
        "label_contract": LABEL_CONTRACT,
        "weight_version": weight_version,
        "clinical_schema": CT6_CLINICAL_SCHEMA,
        "cycle_counts_used": False,
        "ct1_required": False,
        "survival_enabled": False,
    }
    _atomic_torch_save(
        path,
        {
            **contract,
            "model_config": asdict(model.config),
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "clinical_transform": snapshot["transform"],
            "treatment_support": snapshot["treatment_support"],
            "encoder_provenance": encoder_provenance.as_dict(),
            "input_snapshot_id": snapshot["artifact_id"],
        },
    )
    atomic_write_private_json(path.with_suffix(".json"), {**contract, "checkpoint": path.name})


@torch.inference_mode()
def predict_binary_query(
    config_file: Path, query_file: Path, output_file: Path, *, device: str = "cpu"
) -> dict[str, Any]:
    config, query = read_json(config_file), read_json(query_file)
    if config.get("schema_version") != BINARY_INFERENCE_SCHEMA or set(query) != {
        "ct0_features",
        "baseline_clinical",
        "s0_time_days",
        "target_interval_days",
        "treatment_scenario",
    }:
        raise DataContractError(
            code="BINARY_QUERY_SCHEMA", message="Use the baseline-only binary query."
        )
    payload = torch.load(
        config_file.parent / config["checkpoint"], weights_only=True, map_location="cpu"
    )
    if (
        payload.get("schema_version") != BINARY_INFERENCE_SCHEMA
        or payload.get("label_contract") != LABEL_CONTRACT
        or config.get("label_contract") != LABEL_CONTRACT
        or payload.get("weight_version") != config.get("weight_version")
        or payload.get("ct1_required") is not False
        or payload.get("survival_enabled") is not False
    ):
        raise DataContractError(
            code="BINARY_INFERENCE_CONTRACT", message="Checkpoint contract differs."
        )
    model = BinaryEndpointModel(BinaryEndpointConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    ct0 = _move_observation(
        load_ct0_packet(query_file.parent / query["ct0_features"], payload["encoder_provenance"]),
        torch.device(device),
    )
    baseline = encode_baseline(
        [parse_baseline_fields(query["baseline_clinical"], schema_version=CT6_CLINICAL_SCHEMA)],
        payload["clinical_transform"],
    ).to(device)
    scenario = query["treatment_scenario"]
    if not isinstance(scenario, dict) or set(scenario) != {
        "description",
        "methods",
        "regimens",
        "drugs",
    }:
        raise DataContractError(
            code="BINARY_SCENARIO", message="Declare no-cycle treatment inputs."
        )
    interval = torch.tensor([float(query["target_interval_days"])], device=device)
    s0 = torch.tensor([float(query["s0_time_days"])], device=device)
    if not torch.isfinite(interval).all() or (interval <= 0).any():
        raise DataContractError(code="BINARY_INTERVAL", message="Use a positive target interval.")
    target = s0 + interval
    values, flags = encode_compact(
        [normalize_treatment(treatment_fields(scenario))], payload["treatment_support"]
    )
    output = model(
        ct0=ct0,
        baseline=baseline,
        s0_time=s0,
        scenario_actions=compact_actions(
            values.to(device), target, provenance="caller_declared_treatment_scenario"
        ),
        target_time=target,
        horizons=torch.empty(0, device=device),
        scenario=scenario["description"],
        deterministic=True,
    )
    result = {
        "schema_version": BINARY_INFERENCE_SCHEMA,
        "label_contract": LABEL_CONTRACT,
        "weight_version": payload["weight_version"],
        "ct1_used": False,
        "outcomes_used": False,
        "clinical_schema": CT6_CLINICAL_SCHEMA,
        "cycle_counts_used": False,
        "survival_enabled": False,
        "probabilities": output.probabilities.cpu().tolist(),
        "pcr_probability": float(output.probabilities[0, 0]),
        "recorded_recurrence_metastasis_probability": float(output.probabilities[0, 1]),
        "target_interval_days": interval.cpu().tolist(),
        "treatment_support_flags": list(flags[0]),
        "recurrence_interpretation": "recorded_status_not_fixed_window_incidence",
        "scenario_semantics": "declared_condition_not_causal_treatment_effect",
    }
    atomic_write_private_json(output_file, result)
    return {"status": "predicted", "query_count": 1, "ct1_used": False, "task": "pcr_recurrence"}


def verify_binary_portable(root: Path, expected: torch.Tensor) -> dict[str, Any]:
    allowed = {
        (root / name).resolve()
        for name in ("inference.json", "inference.pt", "query.json", "ct0.pt")
    }
    opened: set[Path] = set()

    def guard(original: Any) -> Any:
        def open_file(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            if isinstance(file, (str, Path)) and "r" in mode:
                path = Path(file).resolve()
                if path not in allowed:
                    raise DataContractError(
                        code="BINARY_INFERENCE_READ", message="File outside inference allowlist."
                    )
                opened.add(path)
            return original(file, mode, *args, **kwargs)

        return open_file

    with (
        patch.object(builtins, "open", guard(builtins.open)),
        patch.object(io, "open", guard(io.open)),
    ):
        predict_binary_query(
            root / "inference.json", root / "query.json", root / "independent.json"
        )
    actual = torch.tensor(read_json(root / "independent.json")["probabilities"])
    torch.testing.assert_close(actual, expected[:1], rtol=1e-5, atol=1e-7)
    if opened != allowed:
        raise DataContractError(
            code="BINARY_PORTABLE_COVERAGE", message="Portable input file set differs."
        )
    return {
        "status": "passed",
        "input_files": 4,
        "ct1_labels_other_reads_denied": True,
        "maximum_probability_difference": float((actual - expected[:1]).abs().max()),
    }
