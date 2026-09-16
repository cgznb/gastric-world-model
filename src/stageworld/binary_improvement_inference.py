"""Portable learned or linear classifiers with no post-treatment input files."""

from __future__ import annotations

import builtins
import io
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
from torch import Tensor

from stageworld.artifacts import atomic_write_private_json, read_json
from stageworld.binary_baselines import query_features, transform_features
from stageworld.binary_endpoints import LABEL_CONTRACT, BinaryEndpointConfig
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
from stageworld.generated_inference import load_ct0_packet
from stageworld.model.compact_residual_binary import (
    CompactBinaryModel,
    CompactConfig,
    DeterministicLegacyModel,
)
from stageworld.synthetic_workflow import _atomic_torch_save
from stageworld.training import _move_observation

SCHEMA = "ct6-binary-improvement-portable-v1"


def export_bundle(
    model: CompactBinaryModel | DeterministicLegacyModel | dict[str, Any],
    root: Path,
    *,
    snapshot: dict[str, Any],
    provenance: EncoderProvenance,
    weight_version: str,
    weights: list[float],
    thresholds: list[float],
) -> None:
    if isinstance(model, dict):
        if model["mode"] == "ct1_diagnostic":
            raise ValueError("The CT1 diagnostic is not a baseline inference model")
        kind, payload = (
            "linear",
            {"linear_state": {k: v for k, v in model.items() if k != "fit_patient_ids"}},
        )
    else:
        kind = "legacy" if isinstance(model, DeterministicLegacyModel) else "compact"
        model_config = (
            model.core.config if isinstance(model, DeterministicLegacyModel) else model.config
        )
        payload = {
            "model_config": asdict(model_config),
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        }
    contract = {
        "schema_version": SCHEMA,
        "model_kind": kind,
        "weight_version": weight_version,
        "label_contract": LABEL_CONTRACT,
        "clinical_schema": CT6_CLINICAL_SCHEMA,
        "ct1_required": False,
        "survival_enabled": False,
        "cycles_used": False,
        "positive_weights": weights,
        "thresholds": thresholds,
        "probability_correction": "logit_minus_log_positive_weight",
    }
    _atomic_torch_save(
        root / "inference.pt",
        {
            **contract,
            **payload,
            "clinical_transform": snapshot["transform"],
            "treatment_support": snapshot["treatment_support"],
            "encoder_provenance": provenance.as_dict(),
            "input_snapshot_id": snapshot["artifact_id"],
        },
    )
    atomic_write_private_json(root / "inference.json", {**contract, "checkpoint": "inference.pt"})


@torch.inference_mode()
def predict_query(config_file: Path, query_file: Path, output_file: Path) -> dict[str, Any]:
    config, query = read_json(config_file), read_json(query_file)
    if config.get("schema_version") != SCHEMA or set(query) != {
        "ct0_features",
        "baseline_clinical",
        "s0_time_days",
        "target_interval_days",
        "treatment_scenario",
    }:
        raise ValueError("Use the baseline-only improvement query schema")
    payload = torch.load(
        config_file.parent / config["checkpoint"], weights_only=True, map_location="cpu"
    )
    for key in (
        "schema_version",
        "model_kind",
        "weight_version",
        "label_contract",
        "clinical_schema",
        "ct1_required",
        "survival_enabled",
        "cycles_used",
        "positive_weights",
        "thresholds",
    ):
        if payload[key] != config[key]:
            raise ValueError("Portable bundle contract differs")
    if (
        payload["label_contract"] != LABEL_CONTRACT
        or payload["ct1_required"]
        or payload["survival_enabled"]
        or payload["model_kind"] not in ("linear", "compact", "legacy")
        or payload["clinical_schema"] != CT6_CLINICAL_SCHEMA
        or payload["cycles_used"]
    ):
        raise ValueError("Wrong endpoint or input contract")
    ct0 = _move_observation(
        load_ct0_packet(query_file.parent / query["ct0_features"], payload["encoder_provenance"]),
        torch.device("cpu"),
    )
    baseline = encode_baseline(
        [parse_baseline_fields(query["baseline_clinical"], schema_version=CT6_CLINICAL_SCHEMA)],
        payload["clinical_transform"],
    )
    scenario = query["treatment_scenario"]
    if not isinstance(scenario, dict) or set(scenario) != {
        "description",
        "methods",
        "regimens",
        "drugs",
    }:
        raise ValueError("Use a declared no-cycle treatment scenario")
    interval = torch.tensor([float(query["target_interval_days"])])
    s0 = torch.tensor([float(query["s0_time_days"])])
    if not torch.isfinite(interval).all() or (interval <= 0).any() or not torch.isfinite(s0).all():
        raise ValueError("Invalid query interval")
    if (ct0.available_time[ct0.valid] > s0.item()).any() or not scenario["description"].strip():
        raise ValueError("Invalid baseline availability or scenario")
    values, flags = encode_compact(
        [normalize_treatment(treatment_fields(scenario))], payload["treatment_support"]
    )
    actions = compact_actions(
        values, s0 + interval, provenance="caller_declared_treatment_scenario"
    )
    if payload["model_kind"] == "linear":
        state = payload["linear_state"]
        if state["mode"] not in ("clinical", "ct0"):
            raise ValueError("CT1 diagnostics cannot be deployed as baseline predictions")
        raw = (
            query_features(ct0, baseline, actions, interval, include_ct=state["mode"] == "ct0")
            .double()
            .numpy()
        )
        x = transform_features(raw, state["transform"])
        logits = torch.from_numpy(
            np.concatenate(
                [x @ h["coefficient"].numpy().T + h["intercept"].numpy() for h in state["heads"]],
                -1,
            )
        )
    else:
        model = (
            DeterministicLegacyModel(BinaryEndpointConfig(**payload["model_config"]))
            if payload["model_kind"] == "legacy"
            else CompactBinaryModel(CompactConfig(**payload["model_config"]))
        )
        model.load_state_dict(payload["model_state"], strict=True)
        model.eval()
        logits = model(
            ct0=ct0,
            baseline=baseline,
            s0_time=s0,
            scenario_actions=actions,
            target_time=s0 + interval,
            horizons=torch.empty(0),
            scenario=scenario["description"],
        ).logits
    probabilities = (logits - torch.tensor(payload["positive_weights"]).log()).sigmoid()
    result = {
        "schema_version": SCHEMA,
        "label_contract": LABEL_CONTRACT,
        "probabilities": probabilities.tolist(),
        "thresholds": payload["thresholds"],
        "pcr_probability": float(probabilities[0, 0]),
        "recorded_recurrence_metastasis_probability": float(probabilities[0, 1]),
        "ct1_used": False,
        "outcomes_used": False,
        "cycles_used": False,
        "survival_enabled": False,
        "clinical_schema": CT6_CLINICAL_SCHEMA,
        "treatment_support_flags": list(flags[0]),
        "recurrence_interpretation": "recorded_status_not_fixed_window_incidence",
        "scenario_semantics": "declared_condition_not_causal_treatment_effect",
    }
    atomic_write_private_json(output_file, result)
    return {"status": "predicted", "query_count": 1, "ct1_used": False}


def verify_portable(root: Path, expected: Tensor) -> dict[str, Any]:
    allowed = {
        (root / name).resolve()
        for name in ("inference.json", "inference.pt", "query.json", "ct0.pt")
    }
    opened: set[Path] = set()

    def guard(original: Any) -> Any:
        def guarded(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            if isinstance(file, (str, Path)) and "r" in mode:
                path = Path(file).resolve()
                if path not in allowed:
                    raise ValueError(
                        "Portable inference read a file outside its four-file contract"
                    )
                opened.add(path)
            return original(file, mode, *args, **kwargs)

        return guarded

    with (
        patch.object(builtins, "open", guard(builtins.open)),
        patch.object(io, "open", guard(io.open)),
    ):
        predict_query(root / "inference.json", root / "query.json", root / "independent.json")
    actual = torch.tensor(
        read_json(root / "independent.json")["probabilities"], dtype=expected.dtype
    )
    torch.testing.assert_close(actual, expected[:1], atol=1e-6, rtol=1e-5)
    if opened != allowed:
        raise ValueError("Portable inference did not use its declared file set")
    return {
        "status": "passed",
        "ct1_outcome_other_reads_denied": True,
        "maximum_probability_difference": float((actual - expected[:1]).abs().max()),
    }
