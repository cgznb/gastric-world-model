from __future__ import annotations

import builtins

import pytest

from stageworld.data import (
    ClockKind,
    SplitAssignment,
    SplitManager,
    SplitName,
    generate_synthetic_cohort,
    load_cohort_json,
    load_cohort_npz,
    masked_patient_mean,
    patient_mean,
    patient_normalized_weights,
    save_cohort_json,
    save_cohort_npz,
    write_records_parquet,
)
from stageworld.errors import DataContractError, StageWorldError


def test_three_clocks_are_explicit_and_distinct() -> None:
    assert {clock.value for clock in ClockKind} == {
        "event_or_acquisition",
        "information_availability",
        "prediction_horizon",
    }


def test_t01_shared_patient_across_splits_or_folds_fails() -> None:
    assignments = (
        SplitAssignment("SYN-0", SplitName.TRAIN, fold=0),
        SplitAssignment("SYN-0", SplitName.VALIDATION, fold=1),
    )
    with pytest.raises(DataContractError) as error:
        SplitManager.validate(assignments)
    assert error.value.code == "patient_split_leakage"


def test_t01_every_asset_and_prefix_patient_has_one_assignment() -> None:
    cohort = generate_synthetic_cohort()
    assignments = SplitManager(seed=9).assign(patient.patient_id for patient in cohort.patients)
    SplitManager.assert_records_assigned(cohort.observations, assignments)
    SplitManager.assert_records_assigned(cohort.queries, assignments)
    assert len(assignments) == len(cohort.patients)
    assert len({item.patient_id for item in assignments}) == len(assignments)


def test_t01_record_without_assignment_fails() -> None:
    cohort = generate_synthetic_cohort()
    incomplete = (SplitAssignment("SYN-0000", SplitName.TRAIN),)
    with pytest.raises(DataContractError) as error:
        SplitManager.assert_records_assigned(cohort.observations, incomplete)
    assert error.value.code == "record_without_patient_split"


def test_t19_patient_normalized_weights_remove_patch_count_advantage() -> None:
    patient_ids = ["few", "many", "many", "many", "many"]
    weights = patient_normalized_weights(patient_ids)
    few_total = sum(
        weight for weight, patient in zip(weights, patient_ids, strict=True) if patient == "few"
    )
    many_total = sum(
        weight for weight, patient in zip(weights, patient_ids, strict=True) if patient == "many"
    )
    assert few_total == 1
    assert many_total == 1
    assert patient_mean([2.0, 1.0, 1.0, 1.0, 1.0], patient_ids) == pytest.approx(1.5)
    assert masked_patient_mean(
        [2.0, 1.0, 1.0, 999.0, 999.0],
        [True, True, True, False, False],
        patient_ids,
    ) == pytest.approx(1.5)


def test_synthetic_cohort_json_and_npz_round_trip_without_pickle(tmp_path) -> None:
    cohort = generate_synthetic_cohort(seed=3)
    json_path = tmp_path / "cohort.json"
    npz_path = tmp_path / "cohort.npz"
    save_cohort_json(cohort, json_path)
    save_cohort_npz(cohort, npz_path)
    assert load_cohort_json(json_path) == cohort
    assert load_cohort_npz(npz_path) == cohort


def test_npz_rejects_object_payload(tmp_path) -> None:
    import numpy as np

    path = tmp_path / "unsafe.npz"
    np.savez(path, payload=np.asarray({"unsafe": True}, dtype=object))
    with pytest.raises(DataContractError) as error:
        load_cohort_npz(path)
    assert error.value.code == "cohort_npz_invalid"


def test_missing_pyarrow_is_explicit_and_writes_no_fallback(tmp_path, monkeypatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ImportError("synthetic missing dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    target = tmp_path / "patients.parquet"
    with pytest.raises(StageWorldError) as error:
        write_records_parquet(generate_synthetic_cohort().patients, target)
    assert error.value.code == "dependency_missing"
    assert error.value.details == {"dependency": "pyarrow"}
    assert not target.exists()
