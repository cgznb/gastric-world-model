"""Run the authorized651 event-only fivefold, ten-seed study serially."""

from __future__ import annotations

import gc
from datetime import UTC, datetime
from pathlib import Path

import torch
from sklearn.linear_model import LogisticRegression

from stageworld.artifacts import atomic_write_private_json, read_json
from stageworld.event_data import EventPool, fit_inputs
from stageworld.event_evaluation import binary_metrics, evaluate
from stageworld.event_inference import export_bundle, predict_bundle
from stageworld.event_models import EventModel
from stageworld.event_spec import specification
from stageworld.event_training import infer, train_phase
from stageworld.generated651_evaluation import write_csv
from stageworld.generated700_models import feature_set_loss
from stageworld.synthetic_workflow import _atomic_torch_save


def validate_folds(pool: EventPool, folds: dict) -> None:
    base_ids = set(pool.base.ids)
    if len(base_ids) != len(pool.base.ids) or len(folds["folds"]) != 5:
        raise ValueError("Require unique patients and five folds")
    held: set[str] = set()
    for number, entry in enumerate(folds["folds"]):
        groups = entry["patient_ids"]
        train, validation = set(groups["train"]), set(groups["validation"])
        if (
            entry["fold"] != number
            or not train
            or not validation
            or len(train) != len(groups["train"])
            or len(validation) != len(groups["validation"])
            or train & validation
            or train | validation != base_ids
            or validation & held
        ):
            raise ValueError("Saved patient folds do not form disjoint full partitions")
        held.update(validation)
    if held != base_ids:
        raise ValueError("Fivefold validation coverage differs")


def endpoint_metrics(pool: EventPool, rows: torch.Tensor, probabilities: torch.Tensor) -> dict:
    targets = pool.targets(rows, torch.device("cpu"))
    result = {}
    for index, name in enumerate(("pcr", "recurrence")):
        valid = targets["valid"][:, index]
        result[name] = binary_metrics(
            probabilities[valid, index].numpy(), targets["labels"][valid, index].numpy()
        )
        result[name]["missing"] = int((~valid).sum())
    return result


def ct_report(
    prediction: torch.Tensor, pool: EventPool, train: torch.Tensor, rows: torch.Tensor
) -> dict:
    base = pool.base
    valid = (base.ct0_valid & base.ct1_valid)[rows]
    fitting = train[(base.ct0_valid & base.ct1_valid)[train]]
    if not valid.any() or not len(fitting):
        return {"n": int(valid.sum()), "methods": {}}
    target = base.ct1_tokens[rows]
    average = base.ct1_tokens[fitting].mean((0, 1))[None, None].expand_as(prediction)
    outputs = {"generated": prediction, "copy_CT0": base.ct0[rows], "train_mean": average}
    methods = {}
    for name, predicted in outputs.items():
        p, y = predicted[valid], target[valid]
        methods[name] = {
            "set_loss": float(feature_set_loss(predicted, target, valid)),
            "global_mse": float((p.mean(1) - y.mean(1)).square().mean()),
            "global_cosine": float(
                torch.nn.functional.cosine_similarity(p.mean(1), y.mean(1)).mean()
            ),
        }
    return {"n": int(valid.sum()), "methods": methods, "spatial_correspondence_assumed": False}


def fit_logistic(
    pool: EventPool,
    x: torch.Tensor,
    train: torch.Tensor,
    validation: torch.Tensor,
    snapshot: dict,
    root: Path,
) -> dict:
    status = torch.nn.functional.one_hot(pool.events, num_classes=4).flatten(1).float()
    features = torch.cat((x, status), 1).double()
    selected = train[pool.base.valid[train, 1]]
    path = root / "logistic.pt"
    contract = {"snapshot_id": snapshot["artifact_id"], "C": 1.0, "endpoint": "recurrence"}
    if path.exists():
        payload = torch.load(path, weights_only=True, map_location="cpu")
        if payload["contract"] != contract:
            raise ValueError("Standalone logistic fit differs")
    else:
        model = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=17)
        model.fit(features[selected].numpy(), pool.base.labels[selected, 1].numpy())
        payload = {
            "contract": contract,
            "weight": torch.from_numpy(model.coef_[0]),
            "bias": torch.from_numpy(model.intercept_)[0],
        }
        _atomic_torch_save(path, payload)
    probability = (features[validation] @ payload["weight"] + payload["bias"]).sigmoid()
    valid = pool.base.valid[validation, 1]
    result = binary_metrics(
        probability[valid].numpy(), pool.base.labels[validation, 1][valid].numpy()
    )
    _atomic_torch_save(
        root / "logistic_predictions.pt",
        {
            "patient_ids": [pool.base.ids[i] for i in validation.tolist()],
            "probabilities": probability,
            "valid": valid,
        },
    )
    return result


def publish(
    model: EventModel,
    snapshot: dict,
    pool: EventPool,
    x: torch.Tensor,
    train: torch.Tensor,
    validation: torch.Tensor,
    root: Path,
    *,
    seed: int,
    fold: int,
    selected_epoch: int,
) -> None:
    predicted = infer(model, x, pool, validation)
    ids = [pool.base.ids[i] for i in validation.tolist()]
    export_bundle(root / "bundle", snapshot, model)
    replay = predict_bundle(
        root / "bundle",
        [pool.base.clinical[p] for p in ids],
        [pool.base.treatments[p] for p in ids],
        pool.events[validation],
        pool.base.ct0[validation],
    )
    if not torch.allclose(
        replay["probabilities"], predicted["probabilities"], atol=1e-5, rtol=1e-5
    ):
        raise ValueError("Portable CPU/GPU probability replay differs")
    if not torch.equal(replay["decisions"], predicted["probabilities"] >= 0.5):
        raise ValueError("Portable fixed-threshold decisions differ")
    if not torch.allclose(replay["ct1"], predicted["ct1"], atol=3e-5, rtol=1e-5):
        raise ValueError("Portable CT-feature replay differs")
    targets = pool.targets(validation, torch.device("cpu"))
    _atomic_torch_save(
        root / "predictions.pt",
        {
            **predicted,
            "patient_ids": ids,
            "labels": targets["labels"],
            "valid": targets["valid"],
            "seed": seed,
            "fold": fold,
            "reporting_partition": "model_selection_validation",
        },
    )
    atomic_write_private_json(
        root / "metrics.json",
        {
            "seed": seed,
            "fold": fold,
            "selected_epoch": selected_epoch,
            "endpoints": endpoint_metrics(pool, validation, predicted["probabilities"]),
            "reporting_partition": "model_selection_validation",
        },
    )
    atomic_write_private_json(
        root / "ct_evaluation.json", ct_report(predicted["ct1"], pool, train, validation)
    )
    atomic_write_private_json(
        root / "published.json",
        {
            "status": "completed",
            "standalone_probability_max_error": float(
                (replay["probabilities"] - predicted["probabilities"]).abs().max()
            ),
            "target_files_required": False,
        },
    )


def run_study(pool: EventPool, folds: dict, root: Path) -> dict:
    validate_folds(pool, folds)
    if len(pool.base.ids) != 651:
        raise ValueError("Formal study is bound to the authorized651 cohort")
    spec = {**specification(), "pool_id": pool.artifact_id}
    expected_phases = 2 * 5 * len(spec["seeds"])
    if (root / "specification.json").exists() and read_json(root / "specification.json") != spec:
        raise ValueError("Existing experiment specification differs")
    atomic_write_private_json(root / "specification.json", spec)
    partitions, baseline = {}, []
    for entry in folds["folds"]:
        fold, groups = entry["fold"], entry["patient_ids"]
        target = root / "partitions" / f"fold-{fold}"
        if (target / "groups.json").exists() and read_json(target / "groups.json") != groups:
            raise ValueError("Existing fold membership differs")
        atomic_write_private_json(target / "groups.json", groups)
        x, snapshot = fit_inputs(pool, groups["train"], target / "inputs.pt")
        train, validation = (pool.base.indices(groups[k]) for k in ("train", "validation"))
        baseline.append(
            {"fold": fold, **fit_logistic(pool, x, train, validation, snapshot, target)}
        )
        partitions[fold] = x, snapshot, train, validation
    write_csv(root / "evaluation/logistic_recurrence_fold_metrics.csv", baseline)
    phases: list[dict] = []
    for seed in spec["seeds"]:
        for fold in range(5):
            x, snapshot, train, validation = partitions[fold]
            pretrain = root / "pretrain" / str(seed) / f"fold-{fold}"
            for phase in ("pretrain", "joint"):
                target = (
                    pretrain if phase == "pretrain" else root / "fits" / str(seed) / f"fold-{fold}"
                )
                atomic_write_private_json(
                    root / "progress.json",
                    {
                        "status": "training",
                        "seed": seed,
                        "fold": fold + 1,
                        "phase": phase,
                        "completed_phases": len(phases),
                        "total_phases": expected_phases,
                        "time_utc": datetime.now(UTC).isoformat(),
                    },
                )
                print(
                    f"seed={seed} fold={fold + 1}/5 phase={phase} "
                    f"train={len(train)} validation={len(validation)}",
                    flush=True,
                )
                model, report = train_phase(
                    pool,
                    x,
                    snapshot,
                    train,
                    validation,
                    target,
                    phase=phase,
                    seed=seed,
                    parent=None if phase == "pretrain" else pretrain / "selected.pt",
                )
                if phase == "pretrain":
                    predicted = infer(model, x, pool, validation)
                    atomic_write_private_json(
                        target / "ct_evaluation.json",
                        ct_report(predicted["ct1"], pool, train, validation),
                    )
                    atomic_write_private_json(
                        target / "pcr_evaluation.json",
                        endpoint_metrics(pool, validation, predicted["probabilities"])["pcr"],
                    )
                    del predicted
                else:
                    publish(
                        model,
                        snapshot,
                        pool,
                        x,
                        train,
                        validation,
                        target,
                        seed=seed,
                        fold=fold,
                        selected_epoch=report["selected_epoch"],
                    )
                phases.append({"seed": seed, "fold": fold, **report})
                atomic_write_private_json(root / "selections.json", {"phases": phases})
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            evaluate(root)
    summary = evaluate(root)
    if len(phases) != expected_phases or summary["status"] != "complete":
        raise ValueError("Incomplete formal training queue")
    result = {
        "status": "training_complete_verification_pending",
        "phases": len(phases),
        "epochs": sum(p["completed_epochs"] for p in phases),
        "updates": sum(p["updates"] for p in phases),
        "complete_seeds": summary["complete_seeds"],
    }
    atomic_write_private_json(root / "training_completed.json", result)
    return result
