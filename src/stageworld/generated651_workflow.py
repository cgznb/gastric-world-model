"""One training and validation partition per fold; no inner split or refit."""

from __future__ import annotations

import gc
from datetime import UTC, datetime
from pathlib import Path

import joblib
import torch

from stageworld.artifacts import atomic_write_private_json, read_json
from stageworld.binary700_statistics import fit_statistical, logistic_anchor, statistical_predict
from stageworld.generated651_evaluation import evaluate, write_csv
from stageworld.generated651_inference import export_bundle, predict_bundle
from stageworld.generated651_spec import (
    ANCHOR_C,
    NEURAL,
    SEEDS,
    THRESHOLD,
    Candidate,
    specification,
)
from stageworld.generated700_data import Pool, fit_inputs
from stageworld.generated700_evaluation import points
from stageworld.generated700_models import AnchoredClassifier
from stageworld.generated700_training import infer, train_phase
from stageworld.generated700_workflow import _ct_report, subset
from stageworld.synthetic_workflow import _atomic_torch_save


def fit_anchor(pool: Pool, x: torch.Tensor, train: torch.Tensor) -> list:
    models, _ = fit_statistical(
        Candidate("logistic", "logistic"),
        x,
        pool.labels,
        pool.valid,
        train,
        None,
        selected=[ANCHOR_C, ANCHOR_C],
    )
    return models


def _endpoint_metrics(pool: Pool, rows: torch.Tensor, probability: torch.Tensor) -> dict:
    return {
        name: points(
            probability[:, endpoint].numpy(),
            pool.labels[rows, endpoint].numpy(),
            (probability[:, endpoint] >= THRESHOLD).numpy(),
        )
        for endpoint, name in enumerate(("pcr", "recurrence"))
    }


def publish_prediction(
    root: Path,
    model: AnchoredClassifier,
    snapshot: dict,
    pool: Pool,
    x: torch.Tensor,
    rows: torch.Tensor,
    *,
    arm: str,
    seed: int,
    fold: int,
    selected_epoch: int,
) -> None:
    ids = [pool.ids[index] for index in rows.tolist()]
    logits = infer(model, x, pool, rows).double()
    probability = logits.sigmoid()
    export_bundle(root / "bundle", snapshot, model)
    replay = predict_bundle(
        root / "bundle",
        [pool.clinical[p] for p in ids],
        [pool.treatments[p] for p in ids],
        pool.interval[rows],
        pool.ct0[rows],
    )
    maximum = float((replay["probabilities"] - probability).abs().max())
    if not torch.allclose(replay["probabilities"], probability, atol=1e-5, rtol=1e-5):
        raise ValueError("Standalone CPU/GPU probability replay differs")
    if not torch.equal(replay["decisions"], probability >= THRESHOLD):
        raise ValueError("Standalone fixed-threshold decisions differ")
    _atomic_torch_save(
        root / "predictions.pt",
        {
            "patient_ids": ids,
            "logits": logits,
            "probabilities": probability,
            "decisions": probability >= THRESHOLD,
            "labels": pool.labels[rows],
            "valid": pool.valid[rows],
            "arm": arm,
            "seed": seed,
            "fold": fold,
            "reporting_partition": "model_selection_validation",
        },
    )
    atomic_write_private_json(
        root / "metrics.json",
        {
            "arm": arm,
            "seed": seed,
            "fold": fold,
            "selected_epoch": selected_epoch,
            "reporting_partition": "model_selection_validation",
            "threshold": THRESHOLD,
            "endpoints": _endpoint_metrics(pool, rows, probability),
        },
    )
    atomic_write_private_json(
        root / "published.json",
        {
            "status": "completed",
            "patients": len(rows),
            "standalone_probability_max_error": maximum,
            "ct1_or_outcomes_required": False,
        },
    )


def run_study(pool: Pool, folds: dict, root: Path, *, smoke: bool = False) -> dict:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    seeds = [SEEDS[0]] if smoke else list(SEEDS)
    spec = {**specification(), "smoke": smoke, "active_seeds": seeds, "pool_id": pool.artifact_id}
    if (root / "specification.json").exists() and read_json(root / "specification.json") != spec:
        raise ValueError("Cannot change an existing experiment protocol")
    atomic_write_private_json(root / "specification.json", spec)
    active_folds = folds["folds"][:1] if smoke else folds["folds"]
    phases: list[dict] = []
    epochs = 2 if smoke else 100
    partitions = {}
    anchor_records: list[dict] = []
    for entry in active_folds:
        fold = entry["fold"]
        groups = {
            key: subset(pool, ids, 32) if smoke else ids
            for key, ids in entry["patient_ids"].items()
        }
        if set(groups) != {"train", "validation"} or set(groups["train"]) & set(
            groups["validation"]
        ):
            raise ValueError("Require exactly one disjoint training/validation partition")
        fold_root = root / "partitions" / f"fold-{fold}"
        if (fold_root / "groups.json").exists() and read_json(fold_root / "groups.json") != groups:
            raise ValueError("Existing fold membership changed")
        atomic_write_private_json(fold_root / "groups.json", groups)
        x, snapshot = fit_inputs(pool, groups["train"], fold_root / "inputs.pt")
        train, validation = pool.indices(groups["train"]), pool.indices(groups["validation"])
        contract = {
            "snapshot_id": snapshot["artifact_id"],
            "C": ANCHOR_C,
            "fit_ids": groups["train"],
        }
        if (fold_root / "anchor.joblib").exists():
            if read_json(fold_root / "anchor.json") != contract:
                raise ValueError("Existing clinical anchor binding differs")
            models = joblib.load(fold_root / "anchor.joblib")
        else:
            models = fit_anchor(pool, x, train)
            joblib.dump(models, fold_root / "anchor.joblib")
            atomic_write_private_json(fold_root / "anchor.json", contract)
        probability = torch.from_numpy(
            statistical_predict(models, x[validation].double().numpy())
        ).sigmoid()
        anchor_records.extend(
            {"fold": fold, "endpoint": name, **values}
            for name, values in _endpoint_metrics(pool, validation, probability).items()
        )
        partitions[fold] = (x, snapshot, train, validation, logistic_anchor(models))
    write_csv(root / "evaluation/clinical_anchor_fold_metrics.csv", anchor_records)
    for seed in seeds:
        for entry in active_folds:
            fold = entry["fold"]
            x, snapshot, train, validation, anchor = partitions[fold]
            world_root = root / "world" / str(seed) / f"fold-{fold}"
            print(f"seed={seed} fold={fold + 1} phase=world", flush=True)
            atomic_write_private_json(
                root / "progress.json",
                {
                    "status": "training",
                    "seed": seed,
                    "fold": fold + 1,
                    "phase": "world",
                    "completed_phases": len(phases),
                    "time_utc": datetime.now(UTC).isoformat(),
                },
            )
            world, report = train_phase(
                pool,
                x,
                snapshot,
                Candidate("world", "world"),
                train,
                validation,
                world_root,
                seed=seed,
                epochs=epochs,
            )
            atomic_write_private_json(
                world_root / "ct_evaluation.json", _ct_report(world, x, pool, train, validation)
            )
            phases.append({"fold": fold, "seed": seed, "arm": "world", **report})
            del world
            for candidate in NEURAL:
                target = root / "fits" / candidate.name / str(seed) / f"fold-{fold}"
                print(f"seed={seed} fold={fold + 1} phase={candidate.name}", flush=True)
                atomic_write_private_json(
                    root / "progress.json",
                    {
                        "status": "training",
                        "seed": seed,
                        "fold": fold + 1,
                        "phase": candidate.name,
                        "completed_phases": len(phases),
                        "time_utc": datetime.now(UTC).isoformat(),
                    },
                )
                model, report = train_phase(
                    pool,
                    x,
                    snapshot,
                    candidate,
                    train,
                    validation,
                    target,
                    seed=seed,
                    epochs=epochs,
                    anchor=anchor,
                    parent=world_root / "selected.pt",
                )
                assert isinstance(model, AnchoredClassifier)
                publish_prediction(
                    target,
                    model,
                    snapshot,
                    pool,
                    x,
                    validation,
                    arm=candidate.name,
                    seed=seed,
                    fold=fold,
                    selected_epoch=report["selected_epoch"],
                )
                atomic_write_private_json(
                    target / "ct_evaluation.json",
                    _ct_report(model.world, x, pool, train, validation),
                )
                phases.append({"fold": fold, "seed": seed, "arm": candidate.name, **report})
                atomic_write_private_json(root / "selections.json", {"phases": phases})
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            evaluate(root)
        print(f"seed={seed} all_folds_completed", flush=True)
    summary = evaluate(root)
    expected = len(active_folds) * len(seeds) * (1 + len(NEURAL))
    if len(phases) != expected or summary["status"] != "complete":
        raise ValueError("Incomplete prespecified training queue")
    atomic_write_private_json(root / "selections.json", {"phases": phases})
    scalar_rows = [
        {k: v for k, v in row.items() if not isinstance(v, (list, dict))} for row in phases
    ]
    write_csv(root / "evaluation/selections.csv", scalar_rows)
    result = {
        "status": "training_complete_verification_pending",
        "smoke": smoke,
        "neural_phases": len(phases),
        "world_pretrains": len(active_folds) * len(seeds),
        "classifiers": summary["prediction_files"],
        "epochs": sum(row["completed_epochs"] for row in phases),
        "updates": sum(row["updates"] for row in phases),
        "complete_seeds": summary["complete_seeds"],
        "reporting": "each_seed_separately_mean_and_sample_SD_across_five_folds",
    }
    atomic_write_private_json(root / "training_completed.json", result)
    return result
