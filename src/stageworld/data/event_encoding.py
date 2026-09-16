"""Canonical categorical encodings shared by training and inference."""

from __future__ import annotations

from .contracts import TreatmentKind, TreatmentStatus

TREATMENT_KIND_ID: dict[TreatmentKind, int] = {
    TreatmentKind.SYSTEMIC: 0,
    TreatmentKind.CHEMOTHERAPY: 1,
    TreatmentKind.IMMUNOTHERAPY: 2,
    TreatmentKind.TARGETED: 3,
    TreatmentKind.SURGERY: 4,
    TreatmentKind.OTHER: 5,
}

TREATMENT_STATUS_ID: dict[TreatmentStatus, int] = {
    TreatmentStatus.PLANNED: 0,
    TreatmentStatus.DELIVERED: 1,
    TreatmentStatus.UNKNOWN: 2,
}


def treatment_kind_id(value: TreatmentKind) -> int:
    """Return the stable embedding row for a treatment kind."""

    return TREATMENT_KIND_ID[value]


def treatment_status_id(value: TreatmentStatus) -> int:
    """Return the stable embedding row for treatment knowledge status."""

    return TREATMENT_STATUS_ID[value]
