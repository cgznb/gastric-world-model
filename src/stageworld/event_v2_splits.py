"""Patient-isolated, seed-specific joint-label train/validation/test partitions."""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from stageworld.event_data import EventPool
from stageworld.event_spec import PRESENT

SCHEMA = "event-v2-patient-holdout-v1"
GROUPS = ("train", "validation", "test")


def _cohort(pool: EventPool) -> tuple[list[str], np.ndarray]:
    base, n = pool.base, len(pool.base.ids)
    if (
        not n
        or any(not isinstance(patient, str) or not patient for patient in base.ids)
        or len(set(base.ids)) != n
    ):
        raise ValueError("Holdout requires unique nonempty patient identifiers")
    if (
        base.labels.shape != (n, 2)
        or base.valid.shape != (n, 2)
        or base.valid.dtype != torch.bool
        or not base.valid.all()
        or not torch.isfinite(base.labels).all()
        or not ((base.labels == 0) | (base.labels == 1)).all()
    ):
        raise ValueError("Holdout requires complete finite binary pCR and recurrence labels")
    if (
        pool.events.shape != (n, 3)
        or pool.events.dtype != torch.long
        or not (pool.events[:, 1] == PRESENT).all()
    ):
        raise ValueError("Complete pCR supervision requires confirmed surgery for every patient")
    if (
        base.ct0_valid.shape != (n,)
        or base.ct0_valid.dtype != torch.bool
        or not base.ct0_valid.all()
        or base.ct0.ndim != 3
        or base.ct0.shape[:2] != (n, 27)
        or base.ct0.shape[2] < 1
        or not torch.isfinite(base.ct0).all()
    ):
        raise ValueError("Holdout requires complete finite CT0 token inputs")
    if not isinstance(pool.artifact_id, str) or not pool.artifact_id:
        raise ValueError("Holdout requires a bound event pool artifact")
    ids = sorted(base.ids)
    labels = base.labels[base.indices(ids)].detach().cpu().numpy().astype(np.int64)
    if any(len(np.unique(labels[:, endpoint])) != 2 for endpoint in range(2)):
        raise ValueError("Each endpoint requires both classes in the cohort")
    return ids, labels


def _seed(seed: object) -> int:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Split seed must be an integer in the NumPy random-state range")
    return seed


def _partition_sizes(n: int) -> dict[str, int]:
    heldout = n // 10
    if not heldout:
        raise ValueError("Cohort is too small for nonempty 10-percent holdouts")
    return {"train": n - 2 * heldout, "validation": heldout, "test": heldout}


def _assign(labels: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    sizes = _partition_sizes(len(labels))
    strata = labels[:, 0] + 2 * labels[:, 1]
    if min(Counter(strata.tolist()).values()) < 2:
        raise ValueError("Insufficient joint-label stratum support for stratified holdout")
    try:
        train, heldout = next(
            StratifiedShuffleSplit(
                n_splits=1, test_size=2 * sizes["test"], random_state=seed
            ).split(np.zeros(len(labels)), strata)
        )
        validation, test = next(
            StratifiedShuffleSplit(
                n_splits=1, test_size=sizes["test"], random_state=seed
            ).split(np.zeros(len(heldout)), strata[heldout])
        )
    except ValueError as error:
        raise ValueError(
            "Insufficient joint-label support for the prespecified two-pass holdout; "
            "no alternate seed or fallback stratification was attempted"
        ) from error
    result = {"train": train, "validation": heldout[validation], "test": heldout[test]}
    for name, rows in result.items():
        if any(len(np.unique(labels[rows, endpoint])) != 2 for endpoint in range(2)):
            raise ValueError(f"Both endpoint classes are required in the {name} partition")
    return result


def _counts(labels: np.ndarray) -> dict:
    result: dict = {"patients": len(labels)}
    for endpoint, name in enumerate(("pcr", "recurrence")):
        positive = int(labels[:, endpoint].sum())
        result[name] = {"positive": positive, "negative": len(labels) - positive, "missing": 0}
    result["joint_labels"] = {
        f"pcr_{pcr}_recurrence_{recurrence}": int(
            ((labels[:, 0] == pcr) & (labels[:, 1] == recurrence)).sum()
        )
        for pcr in range(2)
        for recurrence in range(2)
    }
    return result


def _manifest(pool: EventPool, ids: list[str], labels: np.ndarray, seed: int) -> dict:
    partitions = _assign(labels, seed)
    return {
        "schema": SCHEMA,
        "scheme": "joint_stratified_80_10_10",
        "seed": seed,
        "pool_id": pool.artifact_id,
        "ratios": [0.8, 0.1, 0.1],
        "rounding": "floor_each_holdout_remainder_train",
        "stratification": "pcr_plus_2_times_recurrence",
        "algorithm": "sklearn_two_pass_stratified_shuffle_80_20_then_50_50",
        "selection_partition": "validation",
        "reporting_partition": "test",
        "patient_ids": {
            name: sorted(ids[index] for index in rows) for name, rows in partitions.items()
        },
        "counts": {name: _counts(labels[rows]) for name, rows in partitions.items()},
        "cohort_counts": _counts(labels),
    }


def make_split(pool: EventPool, seed: int) -> dict:
    """Make one fixed split without retries, cohort-dependent fitting or file writes."""
    seed = _seed(seed)
    ids, labels = _cohort(pool)
    return _manifest(pool, ids, labels, seed)


def validate_split(pool: EventPool, split: dict) -> None:
    """Reject invalid membership, changed labels/counts and nonreproducible assignments."""
    if not isinstance(split, dict):
        raise ValueError("Saved holdout must be a split manifest")
    seed = _seed(split.get("seed"))
    ids, labels = _cohort(pool)
    groups = split.get("patient_ids")
    if not isinstance(groups, dict) or set(groups) != set(GROUPS):
        raise ValueError("Saved holdout requires train, validation and test patient groups")
    sizes = _partition_sizes(len(ids))
    seen: set[str] = set()
    for name in GROUPS:
        patients = groups[name]
        if (
            not isinstance(patients, list)
            or any(not isinstance(patient, str) for patient in patients)
            or len(patients) != sizes[name]
            or len(set(patients)) != len(patients)
            or set(patients) & seen
        ):
            raise ValueError(
                "Holdout partitions must have exact sizes and disjoint unique patients"
            )
        seen.update(patients)
    if seen != set(ids):
        raise ValueError("Holdout patient groups must exhaust the bound cohort exactly")
    expected = _manifest(pool, ids, labels, seed)
    if split != expected:
        raise ValueError("Saved holdout metadata, counts or deterministic seed assignment differs")
