"""Independent read-only replay of holdout memberships, fits and final predictions."""

from __future__ import annotations

import builtins
import math
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import torch

from stageworld.artifacts import atomic_write_private_json, read_json
from stageworld.event_data import EventPool, encode_inputs, fit_inputs
from stageworld.event_training import recurrence_weight
from stageworld.event_v2_inference import predict_bundle
from stageworld.event_v2_spec import TASK
from stageworld.event_v2_splits import validate_split
from stageworld.event_v2_training import build_model, score
from stageworld.event_v2_workflow import development_partition, freeze_selection, subset_pool
from stageworld.event_workflow import ct_report, endpoint_metrics


def _same(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_same(value, right[key]) for key, value in left.items())
        )
    if isinstance(left, (list, tuple)):
        return (
            isinstance(right, type(left))
            and len(left) == len(right)
            and all(_same(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def _load(path: Path) -> dict:
    return torch.load(path, weights_only=True, map_location="cpu")


def _close(actual: torch.Tensor, expected: torch.Tensor, name: str, atol: float = 1e-5) -> None:
    if actual.shape != expected.shape or not torch.allclose(actual, expected, atol=atol, rtol=1e-5):
        raise ValueError(f"Independent {name} replay differs")


def predict_without_sources(root: Path, *args) -> dict:
    original = builtins.open
    allowed = (root / "inference.pt").resolve()

    def only_bundle(path, *positional, **keywords):
        if not isinstance(path, (str, Path)) or Path(path).resolve() != allowed:
            raise PermissionError("Inference attempted to read an unapproved source")
        return original(path, *positional, **keywords)

    with patch("builtins.open", only_bundle), patch("io.open", only_bundle):
        return predict_bundle(root, *args)


def _predictions(path: Path, test: EventPool, freeze_time: float) -> dict:
    if path.stat().st_mtime + 0.001 < freeze_time:
        raise ValueError("Test predictions predate the global selection freeze")
    predicted = _load(path)
    targets = test.targets(torch.arange(len(test.base.ids)), torch.device("cpu"))
    if predicted["patient_ids"] != test.base.ids or any(
        not torch.equal(predicted[key], targets[key]) for key in ("labels", "valid")
    ):
        raise ValueError("Test prediction membership or supervision differs")
    return predicted


def _verify(pool: EventPool, root: Path) -> dict:
    spec = read_json(root / "specification.json")
    if spec["task"] != TASK or spec["pool_id"] != pool.artifact_id:
        raise ValueError("Holdout verification task or source pool differs")
    if not (root / "selection_frozen.json").exists():
        raise ValueError("Verification requires model selection to be frozen before test access")
    frozen = freeze_selection(root, spec)
    if (
        frozen["test_evaluated_before_freeze"]
        or frozen["status"] != "all_development_fits_complete"
    ):
        raise ValueError("Invalid global test-access gate")
    freeze_time = datetime.fromisoformat(frozen["time_utc"]).timestamp()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phases = replays = bundles = metric_groups = logistic_replays = 0
    records, generation = [], []
    for seed in spec["seeds"]:
        partition = root / "partitions" / f"seed-{seed}"
        split = read_json(partition / "split.json")
        validate_split(pool, split)
        if split["seed"] != seed:
            raise ValueError("Saved split seed differs from its registered location")
        groups = split["patient_ids"]
        dev = development_partition(pool, split)
        snapshot = _load(partition / "inputs.pt")
        if snapshot["fit_ids"] != groups["train"] or snapshot["pool_id"] != pool.artifact_id:
            raise ValueError("Input transform fitting membership differs")
        with tempfile.TemporaryDirectory(prefix="event-v2-transform-audit-") as temporary:
            x, refitted = fit_inputs(dev, groups["train"], Path(temporary) / "inputs.pt")
        if not _same(
            {key: value for key, value in snapshot.items() if key != "artifact_id"},
            {key: value for key, value in refitted.items() if key != "artifact_id"},
        ):
            raise ValueError("Training-only input transform does not refit exactly")
        train, validation = (dev.base.indices(groups[name]) for name in ("train", "validation"))
        expected_mean = dev.base.ct0[train].mean((0, 1))
        expected_scale = dev.base.ct0[train].std((0, 1), unbiased=False).clamp_min(0.05)
        test = subset_pool(pool, groups["test"])
        rows = torch.arange(len(test.base.ids))
        test_x = encode_inputs(
            [test.base.clinical[p] for p in test.base.ids],
            [test.base.treatments[p] for p in test.base.ids],
            snapshot,
        )
        for family in spec["families"]:
            parent_id = None
            selected = None
            for phase in ("pretrain", "joint"):
                folder = root / "fits" / family / f"seed-{seed}" / phase
                report, contract = (
                    read_json(folder / "completed.json"),
                    read_json(folder / "contract.json"),
                )
                expected_binding = {
                    "task": TASK,
                    "family": family,
                    "phase": phase,
                    "seed": seed,
                    "pool_id": pool.artifact_id,
                    "snapshot_id": snapshot["artifact_id"],
                    "train_ids": groups["train"],
                    "validation_ids": groups["validation"],
                    "parent_id": parent_id,
                    "test_rows_available_to_trainer": False,
                    "epochs": spec["max_epochs"],
                }
                if any(contract.get(key) != value for key, value in expected_binding.items()):
                    raise ValueError("Checkpoint patient, seed or parent contract differs")
                final, selected, latest = (
                    _load(folder / f"{name}.pt") for name in ("final", "selected", "latest")
                )
                if (
                    final["artifact_id"] != latest["artifact_id"]
                    or not _same(final["model_state"], latest["model_state"])
                    or selected["epoch"] != final["early_stop"]["selected_epoch"]
                    or report["selected_artifact_id"] != selected["artifact_id"]
                    or report["selected_epoch"] != selected["epoch"]
                    or report["updates"] != final["updates"]
                    or report["completed_epochs"] != final["epoch"]
                ):
                    raise ValueError("Final/selected checkpoint accounting differs")
                expected_epoch = min(final["history"], key=lambda entry: entry["score"])["epoch"]
                if selected["epoch"] != expected_epoch:
                    raise ValueError(
                        "Selected checkpoint is not the best recorded validation epoch"
                    )
                weight = recurrence_weight(dev, train) if phase == "joint" else 1.0
                for saved in (final, selected):
                    if (
                        saved["contract"] != contract
                        or saved["positive_weight"] != weight
                        or saved["history"] != final["history"][: saved["epoch"]]
                    ):
                        raise ValueError("Checkpoint contract, training weight or history differs")
                    _close(
                        saved["model_state"]["world.input_mean"], expected_mean, "CT0 mean", 1e-6
                    )
                    _close(
                        saved["model_state"]["world.input_scale"], expected_scale, "CT0 scale", 1e-6
                    )
                    model = build_model(family, contract["dimensions"]).to(device).eval()
                    model.load_state_dict(saved["model_state"], strict=True)
                    actual, _ = score(model, x, dev, validation, phase, weight)
                    if not math.isclose(
                        actual, saved["history"][-1]["score"], abs_tol=1e-7, rel_tol=1e-6
                    ):
                        raise ValueError("Independent validation checkpoint score replay differs")
                    replays += 1
                    del model
                parent_id = selected["artifact_id"]
                phases += 1
            assert selected is not None
            output = root / "test" / family / f"seed-{seed}"
            predicted = _predictions(output / "predictions.pt", test, freeze_time)
            if (
                predicted["seed"] != seed
                or predicted["family"] != family
                or predicted["checkpoint_id"] != selected["artifact_id"]
                or predicted["reporting_partition"] != "test"
            ):
                raise ValueError("Test prediction checkpoint binding differs")
            bundle = _load(output / "bundle/inference.pt")
            if (
                bundle["task"] != TASK
                or bundle["family"] != family
                or bundle["checkpoint_id"] != selected["artifact_id"]
                or bundle["dimensions"] != selected["contract"]["dimensions"]
                or not _same(bundle["model_state"], selected["model_state"])
                or not _same(
                    bundle["inputs"],
                    {key: value for key, value in snapshot.items() if key != "fit_ids"},
                )
            ):
                raise ValueError("Portable bundle differs from the frozen selected model")
            replay = predict_without_sources(
                output / "bundle",
                [test.base.clinical[p] for p in test.base.ids],
                [test.base.treatments[p] for p in test.base.ids],
                test.events,
                test.base.ct0,
            )
            for key in ("probabilities", "member_logits", "member_disagreement"):
                _close(replay[key], predicted[key], key)
            _close(replay["ct1"], predicted["ct1"], "CT features", 5e-5)
            if not torch.equal(replay["decisions"], predicted["probabilities"] >= 0.5):
                raise ValueError("Fixed-threshold test decisions differ")
            for key in ("last_stage", "incomplete_history"):
                if not torch.equal(replay[key], predicted[key]):
                    raise ValueError("Test event-history outputs differ")
            endpoints = endpoint_metrics(test, rows, predicted["probabilities"])
            metrics = read_json(output / "metrics.json")
            if metrics != {
                "seed": seed,
                "family": family,
                "endpoints": endpoints,
                "reporting_partition": "test",
                "selected_epoch": selected["epoch"],
                "checkpoint_id": selected["artifact_id"],
            }:
                raise ValueError("Test endpoint metrics do not recompute")
            for endpoint, values in endpoints.items():
                records.append({"seed": seed, "family": family, "endpoint": endpoint, **values})
            ct = ct_report(
                predicted["ct1"],
                pool,
                pool.base.indices(groups["train"]),
                pool.base.indices(groups["test"]),
            )
            if ct != read_json(output / "ct_evaluation.json"):
                raise ValueError("Training-mean and generated CT controls do not recompute")
            for method, values in ct["methods"].items():
                generation.append({"seed": seed, "family": family, "method": method, **values})
            bundles, metric_groups = bundles + 1, metric_groups + 2
        status = torch.nn.functional.one_hot(test.events, num_classes=4).flatten(1).float()
        features = torch.cat((test_x, status), 1).double()
        for name in ("logistic", "logistic_balanced"):
            fitted = _load(partition / f"{name}.pt")
            expected = {
                "snapshot_id": snapshot["artifact_id"],
                "train_ids": groups["train"],
                "C": 1.0,
                "class_weight": "balanced" if name.endswith("balanced") else None,
                "endpoints": ["pcr", "recurrence"],
            }
            if fitted["contract"] != expected:
                raise ValueError("Logistic training-only source contract differs")
            output = root / "test" / name / f"seed-{seed}"
            predicted = _predictions(output / "predictions.pt", test, freeze_time)
            if predicted["fit_id"] != fitted["artifact_id"]:
                raise ValueError("Logistic prediction fit identity differs")
            probabilities = (features @ fitted["weight"] + fitted["bias"]).sigmoid()
            _close(probabilities, predicted["probabilities"], "logistic probabilities", 1e-12)
            endpoints = endpoint_metrics(test, rows, predicted["probabilities"])
            if read_json(output / "metrics.json") != {
                "seed": seed,
                "family": name,
                "reporting_partition": "test",
                "endpoints": endpoints,
            }:
                raise ValueError("Logistic test metrics do not recompute")
            for endpoint, values in endpoints.items():
                records.append({"seed": seed, "family": name, "endpoint": endpoint, **values})
            logistic_replays, metric_groups = logistic_replays + 1, metric_groups + 2
    summary = read_json(root / "evaluation/summary.json")
    if (
        summary["test_metrics"] != records
        or summary["generation_metrics"] != generation
        or summary["across_seed_patient_pooling"]
        or summary["best_seed_selection"]
    ):
        raise ValueError("Final test summary differs from independently recomputed records")
    return {
        "status": "passed",
        "seeds": len(spec["seeds"]),
        "phases": phases,
        "checkpoint_score_replays": replays,
        "input_transform_refits": len(spec["seeds"]),
        "denied_source_bundles": bundles,
        "logistic_replays": logistic_replays,
        "endpoint_metric_groups": metric_groups,
        "optimizer_updates_during_verification": 0,
    }


def verify_study(pool: EventPool, root: Path) -> dict:
    paths = sorted(root.rglob("*.pt"))
    before = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in paths}
    original_train = torch.nn.Module.train

    def evaluation_only(module, mode=True):
        if mode:
            raise RuntimeError("Verification cannot enter model training mode")
        return original_train(module, False)

    torch.backends.mha.set_fastpath_enabled(False)
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with (
        torch.random.fork_rng(devices=devices),
        patch.object(torch.nn.Module, "train", evaluation_only),
        patch.object(
            torch.optim.AdamW,
            "step",
            side_effect=RuntimeError("Verification cannot update weights"),
        ),
        torch.no_grad(),
    ):
        result = _verify(pool, root)
    after = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in root.rglob("*.pt")}
    if before != after:
        raise ValueError("Checkpoint or prediction files changed during verification")
    result.update(unchanged_tensor_files=len(paths), checkpoint_files_unchanged=True)
    atomic_write_private_json(root / "verification.json", result)
    return result
