from __future__ import annotations

import copy
from dataclasses import fields

import pytest
import torch
from test_event_multistage import synthetic_pool

from stageworld.event_v2_splits import make_split, validate_split


@pytest.fixture
def pool():
    return synthetic_pool(160)


@pytest.mark.parametrize("n, expected", [(160, (128, 16, 16)), (651, (521, 65, 65))])
def test_exact_complete_disjoint_partitions(n, expected):
    pool = synthetic_pool(n)
    split = make_split(pool, 17)
    validate_split(pool, split)
    groups = split["patient_ids"]
    assert tuple(len(groups[name]) for name in ("train", "validation", "test")) == expected
    assert set().union(*(set(patients) for patients in groups.values())) == set(pool.base.ids)
    assert sum(len(set(patients)) for patients in groups.values()) == n
    for name, patients in groups.items():
        assert patients == sorted(patients)
        assert split["counts"][name]["patients"] == len(patients)
        for endpoint in ("pcr", "recurrence"):
            counts = split["counts"][name][endpoint]
            assert counts["positive"] > 0 and counts["negative"] > 0
            assert counts["missing"] == 0


def test_seed_replay_and_variation(pool):
    first = make_split(pool, 17)
    assert make_split(pool, 17) == first
    second = make_split(pool, 43)
    assert all(
        first["patient_ids"][key] != second["patient_ids"][key] for key in first["patient_ids"]
    )
    first["seed"] = 43
    with pytest.raises(ValueError, match="deterministic seed assignment"):
        validate_split(pool, first)


def test_storage_order_does_not_change_assignment(pool):
    original = make_split(pool, 17)
    reordered = copy.deepcopy(pool)
    order = torch.arange(len(pool.base.ids) - 1, -1, -1)
    reordered.base.ids = [pool.base.ids[index] for index in order.tolist()]
    for field in fields(pool.base):
        value = getattr(pool.base, field.name)
        if isinstance(value, torch.Tensor):
            setattr(reordered.base, field.name, value[order])
    reordered.events = pool.events[order]
    assert make_split(reordered, 17) == original
    validate_split(reordered, original)


@pytest.mark.parametrize("corruption", ["duplicate", "overlap", "unknown", "missing", "extra"])
def test_partition_corruption_rejected(pool, corruption):
    split = make_split(pool, 17)
    groups = split["patient_ids"]
    if corruption == "duplicate":
        groups["test"][0] = groups["test"][1]
    elif corruption == "overlap":
        groups["test"][0] = groups["train"][0]
    elif corruption == "unknown":
        groups["test"][0] = "not-in-artificial-cohort"
    elif corruption == "missing":
        groups["test"].pop()
    else:
        groups["unapproved"] = []
    with pytest.raises(ValueError):
        validate_split(pool, split)


@pytest.mark.parametrize("field", ["counts", "pool_id", "schema", "selection_partition"])
def test_binding_and_count_corruption_rejected(pool, field):
    split = make_split(pool, 17)
    if field == "counts":
        split[field]["test"]["pcr"]["positive"] += 1
    else:
        split[field] = "different"
    with pytest.raises(ValueError, match="metadata, counts"):
        validate_split(pool, split)


def test_equal_size_patient_swap_rejected(pool):
    split = make_split(pool, 17)
    groups = split["patient_ids"]
    groups["train"][0], groups["test"][0] = groups["test"][0], groups["train"][0]
    groups["train"].sort()
    groups["test"].sort()
    with pytest.raises(ValueError, match="deterministic seed assignment"):
        validate_split(pool, split)


@pytest.mark.parametrize("seed", [True, -1, 2**32, 1.5, "17", None])
def test_invalid_seed_rejected(pool, seed):
    with pytest.raises(ValueError, match="Split seed"):
        make_split(pool, seed)


@pytest.mark.parametrize(
    "corruption",
    [
        "duplicate_id", "missing_label", "nan_label", "nonbinary", "missing_ct0", "nan_ct0",
        "no_surgery",
    ],
)
def test_invalid_cohort_rejected(pool, corruption):
    if corruption == "duplicate_id":
        pool.base.ids[0] = pool.base.ids[1]
    elif corruption == "missing_label":
        pool.base.valid[0, 0] = False
    elif corruption == "nan_label":
        pool.base.labels[0, 0] = torch.nan
    elif corruption == "nonbinary":
        pool.base.labels[0, 0] = 2
    elif corruption == "missing_ct0":
        pool.base.ct0_valid[0] = False
    elif corruption == "nan_ct0":
        pool.base.ct0[0, 0, 0] = torch.nan
    else:
        pool.events[0, 1] = 0
    with pytest.raises(ValueError):
        make_split(pool, 17)


def test_single_class_and_rare_joint_strata_fail_without_retry(pool):
    pool.base.labels[:, 0] = 0
    with pytest.raises(ValueError, match="both classes"):
        make_split(pool, 17)
    pool.base.labels.zero_()
    pool.base.labels[0] = 1
    with pytest.raises(ValueError, match="joint-label stratum support"):
        make_split(pool, 17)
    pool.base.labels[1] = 1
    with pytest.raises(ValueError, match="two-pass holdout|Both endpoint classes"):
        make_split(pool, 17)


def test_cohort_changes_invalidate_saved_split(pool):
    split = make_split(pool, 17)
    pool.base.labels[0] = 1 - pool.base.labels[0]
    with pytest.raises(ValueError):
        validate_split(pool, split)
