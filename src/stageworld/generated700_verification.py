"""Independent checkpoint, partition, frozen-parent and inference audits."""

from __future__ import annotations

import builtins
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import joblib
import torch

from stageworld.artifacts import atomic_write_private_json, read_json
from stageworld.binary700_statistics import logistic_anchor
from stageworld.generated700_data import Pool, fit_inputs
from stageworld.generated700_inference import predict_bundle
from stageworld.generated700_models import AnchoredClassifier
from stageworld.generated700_spec import NEURAL, SEEDS, STATISTICAL, Candidate
from stageworld.generated700_training import build_model, score, train_phase
from stageworld.generated700_workflow import select_residual_scales
from stageworld.training import checkpoint_payload_mismatches


def verify_recovery(pool: Pool, root: Path) -> dict:
    groups = read_json(root / "partitions/fold-0/groups.json")
    x, snapshot = fit_inputs(pool, groups["train"], root / "partitions/fold-0/inner.pt")
    train, inner = pool.indices(groups["train"]), pool.indices(groups["validation"])
    anchor = logistic_anchor(joblib.load(root / "fits/logistic/deterministic/fold-0/inner.joblib"))
    results = {}
    for family in ("world", "generated", "generated_frozen"):
        base = root / "recovery_probes" / family
        parent = (
            root / "world/17/fold-0/inner/selected.pt" if family.startswith("generated") else None
        )
        options: dict[str, Any] = {"seed": 17, "epochs": 2, "anchor": anchor, "parent": parent}
        candidate = Candidate(family, family)
        model, _ = train_phase(
            pool, x, snapshot, candidate, train, inner, base / "reference", **options
        )
        del model
        if not (base / "recovered/completed.json").exists():
            try:
                train_phase(
                    pool,
                    x,
                    snapshot,
                    candidate,
                    train,
                    inner,
                    base / "recovered",
                    interrupt_after_update=len(train.split(32)) + 1,
                    **options,
                )
            except RuntimeError as error:
                if str(error) != "intentional_partial_epoch_interruption":
                    raise
            else:
                raise AssertionError("The real partial-epoch interruption was not exercised")
        model, _ = train_phase(
            pool, x, snapshot, candidate, train, inner, base / "recovered", **options
        )
        del model
        expected = torch.load(base / "reference/final.pt", weights_only=True, map_location="cpu")
        actual = torch.load(base / "recovered/final.pt", weights_only=True, map_location="cpu")
        keys = ("model_state", "optimizer_state", "rng_state", "epoch", "updates", "early_stop")
        if any(checkpoint_payload_mismatches(expected[k], actual[k]) for k in keys):
            raise ValueError("Real-feature epoch recovery differs")
        results[family] = {"model_optimizer_rng_exact": True, "final_updates": actual["updates"]}
    atomic_write_private_json(root / "recovery_verification.json", results)
    return results


def verify_study(pool: Pool, root: Path) -> dict:
    selection = read_json(root / "selections.json")
    phase_count, score_count, frozen_count, bundle_count = 0, 0, 0, 0
    verified_partitions: set[tuple[int, str]] = set()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for entry in selection["phases"]:
        fold, seed, partition, arm = (entry[k] for k in ("fold", "seed", "partition", "arm"))
        fold_root = root / "partitions" / f"fold-{fold}"
        groups = read_json(fold_root / "groups.json")
        fitting = (
            groups["train"] if partition == "inner" else groups["train"] + groups["validation"]
        )
        if set(fitting) & set(groups["outer"]):
            raise ValueError("Outer patients entered fitting")
        x, snapshot = fit_inputs(pool, fitting, fold_root / f"{partition}.pt")
        if (fold, partition) not in verified_partitions:
            with TemporaryDirectory(dir=root) as temporary:
                recomputed, fresh = fit_inputs(pool, fitting, Path(temporary) / "inputs.pt")
            if (
                not torch.equal(x, recomputed)
                or snapshot["clinical"] != fresh["clinical"]
                or snapshot["support"] != fresh["support"]
            ):
                raise ValueError("Independent fitting-partition preprocessing differs")
            verified_partitions.add((fold, partition))
        base = (
            (root / "world" / str(seed) if arm == "world" else root / "fits" / arm / str(seed))
            / f"fold-{fold}"
            / partition
        )
        final = torch.load(base / "final.pt", weights_only=True, map_location="cpu")
        latest = torch.load(base / "latest.pt", weights_only=True, map_location="cpu")
        if checkpoint_payload_mismatches(final, latest):
            raise ValueError("Final/latest checkpoints differ")
        selected = torch.load(base / "selected.pt", weights_only=True, map_location="cpu")
        contract = final["contract"]
        if contract["train_ids"] != fitting or contract["snapshot_id"] != snapshot["artifact_id"]:
            raise ValueError("Checkpoint partition identity differs")
        expected_validation = groups["validation"] if partition == "inner" else []
        if contract["validation_ids"] != expected_validation:
            raise ValueError("Validation identity differs")
        if partition == "inner":
            minimum = min(row["score"] for row in final["history"])
            if selected["history"][-1]["score"] != minimum:
                raise ValueError("Raw best score was not selected")
        elif selected["epoch"] != final["epoch"] or final["epoch"] != contract["epochs"]:
            raise ValueError("Fixed-duration refit differs")
        candidate = Candidate(**contract["candidate"])
        anchor_file = root / "fits/logistic/deterministic" / f"fold-{fold}" / f"{partition}.joblib"
        anchor = None if arm == "world" else logistic_anchor(joblib.load(anchor_file))
        parent = None
        if candidate.family.startswith("generated"):
            parent = torch.load(
                root / "world" / str(seed) / f"fold-{fold}" / partition / "selected.pt",
                weights_only=True,
                map_location="cpu",
            )
            if (
                contract["parent_id"] != parent["artifact_id"]
                or parent["contract"]["snapshot_id"] != snapshot["artifact_id"]
                or parent["contract"]["seed"] != seed
            ):
                raise ValueError("Wrong world parent")
        model = build_model(candidate.family, x.shape[1], pool.ct0.shape[-1], anchor, parent).to(
            device
        )
        rows = pool.indices(groups["validation"] if partition == "inner" else fitting)
        for checkpoint in (selected, final):
            model.load_state_dict(checkpoint["model_state"])
            actual_score, _ = score(model, x, pool, rows, arm == "world")
            if abs(actual_score - checkpoint["history"][-1]["score"]) > 1e-6:
                raise ValueError("Independent checkpoint score differs")
            if parent is not None and candidate.family == "generated_frozen":
                state = {
                    k.removeprefix("world."): v
                    for k, v in checkpoint["model_state"].items()
                    if k.startswith("world.")
                }
                if checkpoint_payload_mismatches(state, parent["model_state"]):
                    raise ValueError("Frozen world changed")
                frozen_count += 1
            elif parent is not None:
                if torch.equal(
                    checkpoint["model_state"]["world.decoder.3.weight"],
                    parent["model_state"]["decoder.3.weight"],
                ):
                    raise ValueError("Joint generator was not updated")
            score_count += 1
        if parent is not None and partition == "inner":
            assert isinstance(model, AnchoredClassifier)
            model.load_state_dict(selected["model_state"])
            actual_scales = select_residual_scales(model, x, pool, rows)
            expected_scales = read_json(base.parent / "selection.json")["inner_residual_scales"]
            if actual_scales != expected_scales:
                raise ValueError("Residual scale selection does not replay on inner patients")
        del model
        phase_count += 1
    original_open = builtins.open
    for path in sorted(root.glob("fits/*/*/fold-*/predictions.pt")):
        stored = torch.load(path, weights_only=True, map_location="cpu")
        ids = stored["patient_ids"]
        rows = pool.indices(ids)
        arguments = (
            [pool.clinical[p] for p in ids],
            [pool.treatments[p] for p in ids],
            pool.interval[rows],
            pool.ct0[rows],
            pool.ct0_valid[rows],
        )
        replay = predict_bundle(path.parent / "bundle", *arguments)
        if not torch.allclose(
            replay["probabilities"], stored["probabilities"], atol=1e-5, rtol=1e-5
        ):
            raise ValueError("Independent bundle probabilities differ")
        for rule in ("raw_0.5", "inner_balanced", "inner_sensitivity_0.8", "calibrated_0.5"):
            if not torch.equal(replay[rule], stored[rule]):
                raise ValueError("Independent bundle operating decisions differ")

        def guarded_open(file: Any, *args: Any, **kwargs: Any) -> Any:
            if any(
                text in str(file)
                for text in ("pool.pt", "tokens.pt", "bindings", ".xlsx", "ct1", "outcome")
            ):
                raise ValueError("Standalone inference attempted a source-data read")
            return original_open(file, *args, **kwargs)

        with patch("builtins.open", guarded_open), patch("io.open", guarded_open):
            repeated = predict_bundle(path.parent / "bundle", *arguments)
        if not torch.equal(replay["probabilities"], repeated["probabilities"]):
            raise ValueError("Standalone replay is not deterministic")
        bundle_count += 1
    smoke = read_json(root / "specification.json")["smoke"]
    folds, seeds = (1, 1) if smoke else (5, len(SEEDS))
    expected = (
        folds * seeds * 2 * (1 + len(NEURAL)),
        folds * (len(STATISTICAL) + seeds * len(NEURAL)),
    )
    if (phase_count, bundle_count) != expected:
        raise ValueError("Prespecified phase or candidate coverage is incomplete")
    result = {
        "status": "passed",
        "phases": phase_count,
        "checkpoint_score_replays": score_count,
        "frozen_world_checks": frozen_count,
        "standalone_bundles": bundle_count,
        "optimizer_updates_during_audit": 0,
        "independently_refitted_input_partitions": len(verified_partitions),
    }
    atomic_write_private_json(root / "independent_verification.json", result)
    return result
