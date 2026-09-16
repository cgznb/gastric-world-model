"""Patient-level split contracts and patient-normalized weighting."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from stageworld.errors import DataContractError


class SplitName(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    patient_id: str
    split: SplitName
    fold: int | None = None


class SplitManager:
    def __init__(self, *, seed: int = 17) -> None:
        self.seed = seed

    def assign(
        self,
        patient_ids: Iterable[str],
        *,
        train_fraction: float = 0.7,
        validation_fraction: float = 0.15,
    ) -> tuple[SplitAssignment, ...]:
        ids = list(patient_ids)
        if len(ids) != len(set(ids)):
            raise DataContractError("duplicate_patient", "Cannot split duplicate patient IDs")
        if not ids:
            raise DataContractError("empty_split_input", "At least one patient is required")
        if not (0 < train_fraction < 1 and 0 <= validation_fraction < 1):
            raise DataContractError("invalid_split_fraction", "Split fractions are invalid")
        if train_fraction + validation_fraction >= 1:
            raise DataContractError(
                "invalid_split_fraction", "Train and validation fractions must sum to less than one"
            )
        random.Random(self.seed).shuffle(ids)
        train_end = int(len(ids) * train_fraction)
        validation_end = train_end + int(len(ids) * validation_fraction)
        assignments = []
        for index, patient_id in enumerate(ids):
            if index < train_end:
                split = SplitName.TRAIN
            elif index < validation_end:
                split = SplitName.VALIDATION
            else:
                split = SplitName.TEST
            assignments.append(SplitAssignment(patient_id, split))
        self.validate(assignments)
        return tuple(sorted(assignments, key=lambda item: item.patient_id))

    @staticmethod
    def validate(assignments: Iterable[SplitAssignment]) -> None:
        by_patient: dict[str, set[tuple[SplitName, int | None]]] = defaultdict(set)
        for assignment in assignments:
            if not assignment.patient_id:
                raise DataContractError("missing_patient_id", "Split patient_id is empty")
            if assignment.fold is not None and assignment.fold < 0:
                raise DataContractError("invalid_fold", "Fold index cannot be negative")
            by_patient[assignment.patient_id].add((assignment.split, assignment.fold))
        leaked = [
            patient_id for patient_id, placements in by_patient.items() if len(placements) > 1
        ]
        if leaked:
            raise DataContractError(
                "patient_split_leakage",
                "A patient appears in more than one split or fold",
                details={"patient_count": len(leaked)},
            )

    @staticmethod
    def assert_records_assigned(
        records: Iterable[object], assignments: Iterable[SplitAssignment]
    ) -> None:
        assignment_list = tuple(assignments)
        SplitManager.validate(assignment_list)
        known = {assignment.patient_id for assignment in assignment_list}
        missing = {
            getattr(record, "patient_id")  # noqa: B009
            for record in records
            if getattr(record, "patient_id") not in known  # noqa: B009
        }
        if missing:
            raise DataContractError(
                "record_without_patient_split",
                "A record belongs to a patient with no split assignment",
                details={"patient_count": len(missing)},
            )


def patient_normalized_weights(patient_ids: Sequence[str]) -> tuple[float, ...]:
    """Give each patient's collection total weight one, independent of patch count."""

    if not patient_ids:
        return ()
    counts = Counter(patient_ids)
    if "" in counts:
        raise DataContractError("missing_patient_id", "Cannot weight an empty patient_id")
    return tuple(1.0 / counts[patient_id] for patient_id in patient_ids)


def patient_mean(values: Sequence[float], patient_ids: Sequence[str]) -> float:
    """Average per-patient means, not individual patches or prefixes."""

    if len(values) != len(patient_ids):
        raise DataContractError("weight_length_mismatch", "Values and patient IDs must align")
    if not values:
        raise DataContractError("empty_patient_mean", "Cannot average an empty collection")
    sums: dict[str, float] = defaultdict(float)
    counts = Counter(patient_ids)
    for patient_id, value in zip(patient_ids, values, strict=True):
        sums[patient_id] += float(value)
    return sum(sums[key] / counts[key] for key in sums) / len(sums)


def masked_patient_mean(
    values: Sequence[float], valid_mask: Sequence[bool], patient_ids: Sequence[str]
) -> float:
    """Average valid records within patient, then average represented patients."""

    if not (len(values) == len(valid_mask) == len(patient_ids)):
        raise DataContractError(
            "mask_length_mismatch", "Values, mask, and patient IDs must align"
        )
    sums: dict[str, float] = defaultdict(float)
    counts: Counter[str] = Counter()
    for value, valid, patient_id in zip(values, valid_mask, patient_ids, strict=True):
        if valid:
            if not patient_id:
                raise DataContractError("missing_patient_id", "Masked record patient_id is empty")
            sums[patient_id] += float(value)
            counts[patient_id] += 1
    if not counts:
        raise DataContractError("empty_valid_mask", "No valid records remain after masking")
    return sum(sums[key] / counts[key] for key in counts) / len(counts)
