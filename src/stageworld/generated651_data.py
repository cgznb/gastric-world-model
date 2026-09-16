"""Intersect the existing cached cohort, then split complete patients once."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from stageworld.artifacts import atomic_write_private_json, new_artifact_id, read_json
from stageworld.generated651_spec import SPLIT_SEED
from stageworld.generated700_data import Pool, load_pool
from stageworld.synthetic_workflow import _atomic_torch_save


def complete_cases(pool: Pool, *, expected: int | None = 651) -> Pool:
    selected = pool.ct0_valid & pool.ct1_valid & pool.valid.all(1)
    rows = selected.nonzero().flatten()
    if expected is not None and len(rows) != expected:
        raise ValueError("Complete-case patient count changed")
    ids = [pool.ids[index] for index in rows.tolist()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Require unique nonempty complete patients")
    labels = pool.labels[rows].clone()
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("Recorded endpoints must be binary")
    result = Pool(
        ids,
        {p: pool.clinical[p] for p in ids},
        {p: pool.treatments[p] for p in ids},
        pool.interval[rows].clone(),
        pool.ct0[rows].clone(),
        pool.ct0_valid[rows].clone(),
        pool.ct1[rows].clone(),
        pool.ct1_valid[rows].clone(),
        labels,
        pool.valid[rows].clone(),
        new_artifact_id("generated651-inputs"),
        pool.ct1_tokens[rows].clone(),
    )
    for values in (result.interval, result.ct0, result.ct1, result.ct1_tokens):
        if not torch.isfinite(values).all():
            raise ValueError("Nonfinite complete-case input")
    return result


def counts(pool: Pool, ids: list[str]) -> dict:
    rows = pool.indices(ids)
    result: dict = {
        "patients": len(ids),
        "complete_CT_pairs": int((pool.ct0_valid[rows] & pool.ct1_valid[rows]).sum()),
    }
    for endpoint, name in enumerate(("pcr", "recurrence")):
        valid = pool.valid[rows, endpoint]
        positive = int(pool.labels[rows, endpoint][valid].sum())
        result[name] = {
            "positive": positive,
            "negative": int(valid.sum()) - positive,
            "missing": int((~valid).sum()),
        }
    return result


def make_folds(pool: Pool) -> dict:
    if not (pool.ct0_valid & pool.ct1_valid & pool.valid.all(1)).all():
        raise ValueError("Split requires complete paired images and both labels")
    labels = pool.labels.numpy().astype(int)
    strata = labels[:, 0] + 3 * labels[:, 1]
    if min(Counter(strata).values()) < 5:
        raise ValueError("Insufficient joint-label support for five folds")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
    folds = []
    held: set[str] = set()
    for fold, (train, validation) in enumerate(splitter.split(np.zeros(len(labels)), strata)):
        groups = {
            "train": [pool.ids[index] for index in train],
            "validation": [pool.ids[index] for index in validation],
        }
        if set(groups["train"]) & set(groups["validation"]) or held & set(groups["validation"]):
            raise ValueError("Patient partitions overlap")
        held.update(groups["validation"])
        folds.append(
            {
                "fold": fold,
                "patient_ids": groups,
                "counts": {name: counts(pool, ids) for name, ids in groups.items()},
            }
        )
    if held != set(pool.ids):
        raise ValueError("Validation folds do not cover the entire cohort")
    return {
        "schema": "generated651-validation-folds-v1",
        "split_seed": SPLIT_SEED,
        "selection_partition": "validation",
        "independent_evaluation": False,
        "pool": counts(pool, pool.ids),
        "folds": folds,
    }


def prepare_pool(reference: Path, output: Path) -> Pool:
    parent = load_pool(reference)
    if len(parent.ids) != 700 or parent.ct0.shape[1:] != (27, 768):
        raise ValueError("Require the original700 frozen feature pool")
    selected = complete_cases(parent)
    if selected.labels.sum(0).tolist() != [127, 147]:
        raise ValueError("Complete-case endpoint accounting changed")
    if (output / "pool.pt").exists():
        existing = load_pool(output)
        for name in ("ids", "clinical", "treatments"):
            if getattr(existing, name) != getattr(selected, name):
                raise ValueError("Existing complete-case inputs changed")
        for name in (
            "interval",
            "ct0",
            "ct1",
            "ct1_tokens",
            "ct0_valid",
            "ct1_valid",
            "labels",
            "valid",
        ):
            if not torch.equal(getattr(existing, name), getattr(selected, name)):
                raise ValueError("Existing complete-case tensors changed")
        if read_json(output / "audit.json")["parent_pool_id"] != parent.artifact_id:
            raise ValueError("Parent input lineage changed")
        selected = existing
    else:
        output.mkdir(parents=True, mode=0o700, exist_ok=True)
        _atomic_torch_save(output / "pool.pt", vars(selected))
    folds = make_folds(selected)
    if (output / "folds.json").exists() and read_json(output / "folds.json") != folds:
        raise ValueError("Complete-case fivefold assignment changed")
    atomic_write_private_json(output / "folds.json", folds)
    atomic_write_private_json(
        output / "audit.json",
        {
            **counts(selected, selected.ids),
            "parent_patients": len(parent.ids),
            "excluded_patients": len(parent.ids) - len(selected.ids),
            "parent_pool_id": parent.artifact_id,
            "pool_id": selected.artifact_id,
            "definition": "existing_valid_CT0_CT1_features_and_two_recorded_labels",
            "old_trained_weights_loaded": False,
        },
    )
    return selected
