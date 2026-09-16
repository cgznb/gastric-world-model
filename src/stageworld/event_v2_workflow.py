"""Separate development fitting from a globally gated final test evaluation."""

from __future__ import annotations

import gc
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import torch
from sklearn.linear_model import LogisticRegression

from stageworld.artifacts import atomic_write_private_json, new_artifact_id, read_json
from stageworld.event_data import EventPool, encode_inputs, fit_inputs
from stageworld.event_v2_inference import export_bundle, predict_bundle
from stageworld.event_v2_spec import FAMILIES, specification
from stageworld.event_v2_splits import make_split, validate_split
from stageworld.event_v2_training import build_model, infer, train_phase
from stageworld.event_workflow import ct_report, endpoint_metrics
from stageworld.generated651_evaluation import write_csv
from stageworld.synthetic_workflow import _atomic_torch_save


def subset_pool(pool: EventPool, ids: list[str]) -> EventPool:
    if len(set(ids)) != len(ids) or not set(ids) <= set(pool.base.ids):
        raise ValueError("Subset requires distinct known patient IDs")
    rows = pool.base.indices(ids)
    fields = {
        name: getattr(pool.base, name)[rows].clone()
        for name in (
            "interval",
            "ct0",
            "ct0_valid",
            "ct1",
            "ct1_valid",
            "labels",
            "valid",
            "ct1_tokens",
        )
    }
    base = replace(
        pool.base,
        ids=list(ids),
        clinical={p: pool.base.clinical[p] for p in ids},
        treatments={p: pool.base.treatments[p] for p in ids},
        **fields,
    )
    return EventPool(base, pool.events[rows].clone(), pool.artifact_id)


def development_partition(pool: EventPool, split: dict) -> EventPool:
    groups = split["patient_ids"]
    return subset_pool(pool, groups["train"] + groups["validation"])


def _bound_json(path: Path, value: dict) -> None:
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Existing experiment binding differs: {path.name}")
    else:
        atomic_write_private_json(path, value)


def prepare_splits(pool: EventPool, root: Path, spec: dict) -> list[dict]:
    if len(pool.base.ids) != spec["patients"]:
        raise ValueError("Cohort size differs from the locked experiment")
    if not spec["seeds"] or len(set(spec["seeds"])) != len(spec["seeds"]):
        raise ValueError("Require unique model seeds")
    if (
        not spec["families"]
        or len(set(spec["families"])) != len(spec["families"])
        or not set(spec["families"]) <= set(FAMILIES)
    ):
        raise ValueError("Unknown or repeated model family")
    bound = {**spec, "pool_id": pool.artifact_id}
    _bound_json(root / "specification.json", bound)
    splits = []
    for seed in spec["seeds"]:
        split = make_split(pool, seed)
        _bound_json(root / "partitions" / f"seed-{seed}" / "split.json", split)
        validate_split(pool, split)
        splits.append(split)
    atomic_write_private_json(
        root / "split_audit.json",
        {
            "patients": len(pool.base.ids),
            "seed_count": len(splits),
            "seeds": [{"seed": split["seed"], "counts": split["counts"]} for split in splits],
            "test_policy": "selection_frozen_across_all_seeds_before_test",
            "across_seed_test_sets_independent": False,
        },
    )
    return splits


def fit_logistic(
    dev: EventPool, x: torch.Tensor, snapshot: dict, train: torch.Tensor, root: Path
) -> None:
    status = torch.nn.functional.one_hot(dev.events, num_classes=4).flatten(1).float()
    features = torch.cat((x, status), 1).double()
    for weighted in (False, True):
        name = "logistic_balanced" if weighted else "logistic"
        path = root / f"{name}.pt"
        contract = {
            "snapshot_id": snapshot["artifact_id"],
            "train_ids": snapshot["fit_ids"],
            "C": 1.0,
            "class_weight": "balanced" if weighted else None,
            "endpoints": ["pcr", "recurrence"],
        }
        if path.exists():
            if torch.load(path, weights_only=True, map_location="cpu")["contract"] != contract:
                raise ValueError("Logistic training-only binding differs")
            continue
        weights, biases = [], []
        for endpoint in range(2):
            model = LogisticRegression(
                C=1.0,
                max_iter=5000,
                solver="lbfgs",
                class_weight="balanced" if weighted else None,
                random_state=17,
            )
            model.fit(features[train].numpy(), dev.base.labels[train, endpoint].numpy())
            weights.append(torch.from_numpy(model.coef_[0]))
            biases.append(float(model.intercept_[0]))
        _atomic_torch_save(
            path,
            {
                "contract": contract,
                "artifact_id": new_artifact_id(name),
                "weight": torch.stack(weights, 1),
                "bias": torch.tensor(biases, dtype=torch.double),
            },
        )


def selection_registry(root: Path, spec: dict) -> list[dict]:
    if any(
        not (root / "fits" / family / f"seed-{seed}" / phase / "completed.json").exists()
        for seed in spec["seeds"]
        for family in spec["families"]
        for phase in ("pretrain", "joint")
    ):
        raise ValueError("Test access denied: not all development fits are complete")
    registry = []
    for seed in spec["seeds"]:
        partition = root / "partitions" / f"seed-{seed}"
        split = read_json(partition / "split.json")
        groups = split["patient_ids"]
        snapshot = torch.load(partition / "inputs.pt", weights_only=True, map_location="cpu")
        if snapshot["fit_ids"] != groups["train"] or snapshot["pool_id"] != split["pool_id"]:
            raise ValueError("Development preprocessing binding differs before test access")
        for family in spec["families"]:
            folder = root / "fits" / family / f"seed-{seed}"
            parent_id = None
            for phase in ("pretrain", "joint"):
                report_path = folder / phase / "completed.json"
                if not report_path.exists():
                    raise ValueError("Test access denied: not all development fits are complete")
                report = read_json(report_path)
                selected = torch.load(
                    folder / phase / "selected.pt", weights_only=True, map_location="cpu"
                )
                contract = selected["contract"]
                if (
                    report["status"] != "completed"
                    or report["selected_artifact_id"] != selected["artifact_id"]
                    or report["selected_epoch"] != selected["epoch"]
                    or contract["family"] != family
                    or contract["phase"] != phase
                    or contract["seed"] != seed
                    or contract["parent_id"] != parent_id
                    or contract["task"] != specification()["task"]
                    or contract["epochs"] != spec["max_epochs"]
                    or contract["pool_id"] != split["pool_id"]
                    or contract["snapshot_id"] != snapshot["artifact_id"]
                    or contract["train_ids"] != groups["train"]
                    or contract["validation_ids"] != groups["validation"]
                ):
                    raise ValueError("Development checkpoint selection changed")
                parent_id = selected["artifact_id"]
                registry.append(
                    {
                        "seed": seed,
                        "family": family,
                        "phase": phase,
                        "artifact_id": selected["artifact_id"],
                        "epoch": selected["epoch"],
                    }
                )
    return registry


def freeze_selection(root: Path, spec: dict) -> dict:
    bound = read_json(root / "specification.json")
    if {**spec, "pool_id": bound["pool_id"]} != bound:
        raise ValueError("Test gate specification differs from the entire locked experiment")
    registry = selection_registry(root, spec)
    path = root / "selection_frozen.json"
    if path.exists():
        frozen = read_json(path)
        if frozen["selected_checkpoints"] != registry or frozen["specification"] != read_json(
            root / "specification.json"
        ):
            raise ValueError("Frozen model selections or specification changed")
        return frozen
    frozen = {
        "status": "all_development_fits_complete",
        "time_utc": datetime.now(UTC).isoformat(),
        "selected_checkpoints": registry,
        "specification": read_json(root / "specification.json"),
        "test_evaluated_before_freeze": False,
    }
    atomic_write_private_json(path, frozen)
    return frozen


def evaluate_tests(pool: EventPool, root: Path, spec: dict) -> dict:
    if not (root / "selection_frozen.json").exists():
        raise ValueError("Test access denied: model selection has not been frozen")
    freeze_selection(root, spec)
    records, generation = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in spec["seeds"]:
        partition = root / "partitions" / f"seed-{seed}"
        split = read_json(partition / "split.json")
        validate_split(pool, split)
        groups = split["patient_ids"]
        snapshot = torch.load(partition / "inputs.pt", weights_only=True, map_location="cpu")
        if snapshot["fit_ids"] != groups["train"]:
            raise ValueError("Heldout prediction transform differs from training patients")
        ids = groups["test"]
        test = subset_pool(pool, ids)
        rows = torch.arange(len(ids))
        x = encode_inputs(
            [test.base.clinical[p] for p in ids], [test.base.treatments[p] for p in ids], snapshot
        )
        targets = test.targets(rows, torch.device("cpu"))
        for family in spec["families"]:
            fit = root / "fits" / family / f"seed-{seed}" / "joint"
            saved = torch.load(fit / "selected.pt", weights_only=True, map_location="cpu")
            model = build_model(family, saved["contract"]["dimensions"]).to(device).eval()
            model.load_state_dict(saved["model_state"], strict=True)
            prediction = infer(model, x, test, rows)
            output = root / "test" / family / f"seed-{seed}"
            if not (output / "bundle/inference.pt").exists():
                export_bundle(output / "bundle", snapshot, model, family, saved["artifact_id"])
            replay = predict_bundle(
                output / "bundle",
                [test.base.clinical[p] for p in ids],
                [test.base.treatments[p] for p in ids],
                test.events,
                test.base.ct0,
            )
            if not torch.allclose(
                replay["probabilities"], prediction["probabilities"], atol=1e-5, rtol=1e-5
            ):
                raise ValueError("Heldout CPU/GPU probability replay differs")
            if not torch.equal(replay["decisions"], prediction["probabilities"] >= 0.5):
                raise ValueError("Heldout CPU/GPU fixed decisions differ")
            if not torch.allclose(replay["ct1"], prediction["ct1"], atol=5e-5, rtol=1e-5):
                raise ValueError("Heldout CT feature replay differs")
            payload = {
                **prediction,
                "patient_ids": ids,
                "labels": targets["labels"],
                "valid": targets["valid"],
                "seed": seed,
                "family": family,
                "checkpoint_id": saved["artifact_id"],
                "reporting_partition": "test",
            }
            path = output / "predictions.pt"
            if path.exists():
                previous = torch.load(path, weights_only=True, map_location="cpu")
                if (
                    previous["checkpoint_id"] != saved["artifact_id"]
                    or previous["patient_ids"] != ids
                    or not torch.allclose(
                        previous["probabilities"], prediction["probabilities"], atol=1e-7, rtol=1e-6
                    )
                ):
                    raise ValueError("Previously evaluated test predictions changed")
            else:
                _atomic_torch_save(path, payload)
            endpoints = endpoint_metrics(test, rows, prediction["probabilities"])
            _bound_json(
                output / "metrics.json",
                {
                    "seed": seed,
                    "family": family,
                    "endpoints": endpoints,
                    "reporting_partition": "test",
                    "selected_epoch": saved["epoch"],
                    "checkpoint_id": saved["artifact_id"],
                },
            )
            for endpoint, values in endpoints.items():
                records.append({"seed": seed, "family": family, "endpoint": endpoint, **values})
            # The mean control uses training CT1; test targets are scoring-only.
            report = ct_report(
                prediction["ct1"], pool, pool.base.indices(groups["train"]), pool.base.indices(ids)
            )
            _bound_json(output / "ct_evaluation.json", report)
            for method, values in report["methods"].items():
                generation.append({"seed": seed, "family": family, "method": method, **values})
            atomic_write_private_json(
                output / "published.json",
                {
                    "status": "completed",
                    "checkpoint_id": saved["artifact_id"],
                    "cpu_gpu_probability_max_error": float(
                        (replay["probabilities"] - prediction["probabilities"]).abs().max()
                    ),
                },
            )
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        status = torch.nn.functional.one_hot(test.events, num_classes=4).flatten(1).float()
        features = torch.cat((x, status), 1).double()
        for name in ("logistic", "logistic_balanced"):
            fitted = torch.load(partition / f"{name}.pt", weights_only=True, map_location="cpu")
            probabilities = (features @ fitted["weight"] + fitted["bias"]).sigmoid()
            endpoints = endpoint_metrics(test, rows, probabilities)
            for endpoint, values in endpoints.items():
                records.append({"seed": seed, "family": name, "endpoint": endpoint, **values})
            output = root / "test" / name / f"seed-{seed}"
            _bound_json(
                output / "metrics.json",
                {
                    "seed": seed,
                    "family": name,
                    "reporting_partition": "test",
                    "endpoints": endpoints,
                },
            )
            if not (output / "predictions.pt").exists():
                _atomic_torch_save(
                    output / "predictions.pt",
                    {
                        "patient_ids": ids,
                        "probabilities": probabilities,
                        "labels": targets["labels"],
                        "valid": targets["valid"],
                        "fit_id": fitted["artifact_id"],
                    },
                )
    result = {
        "status": "completed",
        "seeds": spec["seeds"],
        "families": spec["families"],
        "test_metrics": records,
        "generation_metrics": generation,
        "model_selection_partition": "validation",
        "reporting_partition": "test",
        "across_seed_patient_pooling": False,
        "best_seed_selection": False,
        "previously_used_development_cohort": True,
        "independent_external_validation": False,
    }
    evaluation = root / "evaluation"
    atomic_write_private_json(evaluation / "summary.json", result)
    write_csv(evaluation / "test_metrics.csv", records)
    write_csv(evaluation / "generation_test_metrics.csv", generation)
    lines = [
        "# Event V2: Seed-Specific Internal Test Results",
        "",
        "Each seed has its own training/validation/test partition. No test-based model selection.",
        "This cohort was used in prior development. These are internal holdouts, "
        "not external validation.",
        "Seeds have overlapping patients. Do not pool rows or select a best seed.",
        "Recorded recurrence status is not incident risk at a specified horizon.",
        "",
        "| Seed | Family | Endpoint | N | AUROC | AUPRC | Recall | Precision | Brier |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in records:
        values = [
            "NA" if row[key] is None else f"{row[key]:.5f}"
            for key in ("auroc", "auprc", "sensitivity", "precision", "brier")
        ]
        lines.append(
            f"| {row['seed']} | {row['family']} | {row['endpoint']} | {row['n']} | "
            + " | ".join(values)
            + " |"
        )
    (evaluation / "report.md").write_text("\n".join(lines) + "\n")
    return result


def run_study(
    pool: EventPool,
    root: Path,
    *,
    spec: dict | None = None,
    dimensions: dict[str, dict] | None = None,
) -> dict:
    spec = specification() if spec is None else spec
    splits = prepare_splits(pool, root, spec)
    phases: list[dict] = []
    if not (root / "selection_frozen.json").exists():
        for split in splits:
            seed, groups = split["seed"], split["patient_ids"]
            partition = root / "partitions" / f"seed-{seed}"
            dev = development_partition(pool, split)
            x, snapshot = fit_inputs(dev, groups["train"], partition / "inputs.pt")
            train, validation = (dev.base.indices(groups[name]) for name in ("train", "validation"))
            fit_logistic(dev, x, snapshot, train, partition)
            for family in spec["families"]:
                target = root / "fits" / family / f"seed-{seed}"
                for phase in ("pretrain", "joint"):
                    atomic_write_private_json(
                        root / "progress.json",
                        {
                            "status": "training",
                            "seed": seed,
                            "family": family,
                            "phase": phase,
                            "completed_phases": len(phases),
                            "total_phases": 2 * len(splits) * len(spec["families"]),
                            "test_access_open": False,
                            "time_utc": datetime.now(UTC).isoformat(),
                        },
                    )
                    model, report = train_phase(
                        dev,
                        x,
                        snapshot,
                        train,
                        validation,
                        target / phase,
                        family=family,
                        phase=phase,
                        seed=seed,
                        epochs=spec["max_epochs"],
                        parent=None if phase == "pretrain" else target / "pretrain/selected.pt",
                        dimensions=None if dimensions is None else dimensions[family],
                    )
                    phases.append(report)
                    del model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            del dev, x
    frozen = freeze_selection(root, spec)
    atomic_write_private_json(
        root / "progress.json",
        {
            "status": "final_test_evaluation",
            "frozen_phases": len(frozen["selected_checkpoints"]),
            "test_access_open": True,
            "time_utc": datetime.now(UTC).isoformat(),
        },
    )
    result = evaluate_tests(pool, root, spec)
    atomic_write_private_json(
        root / "evaluation_complete.json",
        {
            "status": "completed",
            "seeds": spec["seeds"],
            "families": spec["families"],
            "frozen_phases": len(frozen["selected_checkpoints"]),
            "verification_pending": True,
        },
    )
    return result
