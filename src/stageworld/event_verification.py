"""Read-only replay of trained checkpoints, portable bundles and fold metrics."""

from __future__ import annotations

import builtins
import math
from pathlib import Path
from unittest.mock import patch

import torch

from stageworld.artifacts import atomic_write_private_json, read_json
from stageworld.event_data import EventPool, encode_inputs
from stageworld.event_inference import predict_bundle
from stageworld.event_models import EventModel
from stageworld.event_training import score
from stageworld.event_workflow import endpoint_metrics, validate_folds


def predict_without_sources(root: Path, *args) -> dict:
    original = builtins.open
    allowed = (root / "inference.pt").resolve()

    def only_bundle(path, *positional, **keywords):
        if not isinstance(path, (str, Path)) or Path(path).resolve() != allowed:
            raise PermissionError("Inference attempted to read an unapproved source")
        return original(path, *positional, **keywords)

    with patch("builtins.open", only_bundle), patch("io.open", only_bundle):
        return predict_bundle(root, *args)


def verify_study(pool: EventPool, folds: dict, root: Path) -> dict:
    validate_folds(pool, folds)
    spec = read_json(root / "specification.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phases, replays, bundles, metric_groups = 0, 0, 0, 0
    for seed in spec["seeds"]:
        for entry in folds["folds"]:
            fold, groups = entry["fold"], entry["patient_ids"]
            partition = root / "partitions" / f"fold-{fold}"
            snapshot = torch.load(partition / "inputs.pt", weights_only=True, map_location="cpu")
            if snapshot["fit_ids"] != groups["train"]:
                raise ValueError("Input fit patient membership differs")
            x = encode_inputs(
                [pool.base.clinical[p] for p in pool.base.ids],
                [pool.base.treatments[p] for p in pool.base.ids],
                snapshot,
            )
            rows = pool.base.indices(groups["validation"])
            parent_id = None
            for phase in ("pretrain", "joint"):
                target = (
                    root
                    / ("pretrain" if phase == "pretrain" else "fits")
                    / str(seed)
                    / f"fold-{fold}"
                )
                report = read_json(target / "completed.json")
                for name in ("final", "selected"):
                    saved = torch.load(target / f"{name}.pt", weights_only=True, map_location="cpu")
                    contract = saved["contract"]
                    if (
                        contract["seed"] != seed
                        or contract["phase"] != phase
                        or contract["train_ids"] != groups["train"]
                        or contract["validation_ids"] != groups["validation"]
                        or contract["snapshot_id"] != snapshot["artifact_id"]
                        or contract["parent_id"] != parent_id
                    ):
                        raise ValueError("Checkpoint data or parent binding differs")
                    model = EventModel(**contract["dimensions"]).to(device).eval()
                    model.load_state_dict(saved["model_state"], strict=True)
                    actual, _ = score(model, x, pool, rows, phase, saved["positive_weight"])
                    if not math.isclose(
                        actual, saved["history"][-1]["score"], abs_tol=1e-7, rel_tol=1e-6
                    ):
                        raise ValueError("Independent checkpoint score replay differs")
                    if name == "selected":
                        if saved["epoch"] != report["selected_epoch"]:
                            raise ValueError("Selected checkpoint epoch differs")
                        if phase == "pretrain":
                            parent_id = saved["artifact_id"]
                    replays += 1
                    del model
                phases += 1
            target = root / "fits" / str(seed) / f"fold-{fold}"
            predicted = torch.load(target / "predictions.pt", weights_only=True, map_location="cpu")
            ids = groups["validation"]
            truth = pool.targets(rows, torch.device("cpu"))
            if (
                predicted["patient_ids"] != ids
                or not torch.equal(predicted["labels"], truth["labels"])
                or not torch.equal(predicted["valid"], truth["valid"])
            ):
                raise ValueError("Prediction membership or masked targets differ")
            replay = predict_without_sources(
                target / "bundle",
                [pool.base.clinical[p] for p in ids],
                [pool.base.treatments[p] for p in ids],
                pool.events[rows],
                pool.base.ct0[rows],
            )
            if not torch.allclose(
                replay["probabilities"], predicted["probabilities"], atol=1e-5, rtol=1e-5
            ):
                raise ValueError("Input-only bundle replay differs")
            expected = read_json(target / "metrics.json")["endpoints"]
            if endpoint_metrics(pool, rows, predicted["probabilities"]) != expected:
                raise ValueError("Published binary metrics do not recompute")
            bundles, metric_groups = bundles + 1, metric_groups + 2
    result = {
        "status": "passed",
        "phases": phases,
        "checkpoint_score_replays": replays,
        "denied_source_bundles": bundles,
        "endpoint_metric_groups": metric_groups,
        "optimizer_updates_during_verification": 0,
    }
    atomic_write_private_json(root / "verification.json", result)
    return result
