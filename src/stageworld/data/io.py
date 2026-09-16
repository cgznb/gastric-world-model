"""Explicit JSON/NPZ cohort I/O and dependency-gated Parquet helpers."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from stageworld.errors import DataContractError, StageWorldError

from .contracts import (
    AdjudicationStatus,
    AvailabilityBasis,
    ClinicalMeasurement,
    Cohort,
    DataIssue,
    EventType,
    MissingCategory,
    Modality,
    Observation,
    ObservationRole,
    Outcome,
    Patient,
    QualityStatus,
    Query,
    QueryEligibility,
    SourceType,
    Stage,
    TimePrecision,
    Treatment,
    TreatmentKind,
    TreatmentStatus,
)

SCHEMA_VERSION = "stageworld-cohort-v1"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def cohort_to_dict(cohort: Cohort) -> dict[str, Any]:
    payload = _jsonable(cohort)
    assert isinstance(payload, dict)
    return {"schema_version": SCHEMA_VERSION, **payload}


def _parse_observation(item: Mapping[str, Any]) -> Observation:
    values = dict(item)
    values.update(
        modality=Modality(values["modality"]),
        role=ObservationRole(values["role"]),
        source_type=SourceType(values["source_type"]),
        time_precision=TimePrecision(values["time_precision"]),
        availability_basis=AvailabilityBasis(values["availability_basis"]),
        quality_status=QualityStatus(values["quality_status"]),
        missing_reason=(
            MissingCategory(values["missing_reason"])
            if values.get("missing_reason") is not None
            else None
        ),
    )
    return Observation(**values)


def _parse_measurement(item: Mapping[str, Any]) -> ClinicalMeasurement:
    values = dict(item)
    values.update(
        source_type=SourceType(values["source_type"]),
        time_precision=TimePrecision(values["time_precision"]),
        availability_basis=AvailabilityBasis(values["availability_basis"]),
        missing_reason=(
            MissingCategory(values["missing_reason"])
            if values.get("missing_reason") is not None
            else None
        ),
    )
    return ClinicalMeasurement(**values)


def _parse_treatment(item: Mapping[str, Any]) -> Treatment:
    values = dict(item)
    values.update(
        treatment_kind=TreatmentKind(values["treatment_kind"]),
        standardized_components=tuple(values["standardized_components"]),
        planned_or_delivered=TreatmentStatus(values["planned_or_delivered"]),
        time_precision=TimePrecision(values["time_precision"]),
        availability_basis=AvailabilityBasis(values["availability_basis"]),
    )
    return Treatment(**values)


def _parse_outcome(item: Mapping[str, Any]) -> Outcome:
    values = dict(item)
    values.update(
        event_type=EventType(values["event_type"]),
        adjudication_status=AdjudicationStatus(values["adjudication_status"]),
    )
    return Outcome(**values)


def _parse_query(item: Mapping[str, Any]) -> Query:
    values = dict(item)
    values.update(
        stage=Stage(values["stage"]),
        time_precision=TimePrecision(values["time_precision"]),
        eligibility=QueryEligibility(values["eligibility"]),
    )
    return Query(**values)


def cohort_from_dict(payload: Mapping[str, Any]) -> Cohort:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise DataContractError(
            "cohort_schema_mismatch",
            "Cohort schema version is absent or unsupported",
            details={"expected": SCHEMA_VERSION},
        )
    return Cohort(
        patients=tuple(Patient(**item) for item in payload.get("patients", [])),
        observations=tuple(_parse_observation(item) for item in payload.get("observations", [])),
        clinical_measurements=tuple(
            _parse_measurement(item) for item in payload.get("clinical_measurements", [])
        ),
        treatments=tuple(_parse_treatment(item) for item in payload.get("treatments", [])),
        outcomes=tuple(_parse_outcome(item) for item in payload.get("outcomes", [])),
        queries=tuple(_parse_query(item) for item in payload.get("queries", [])),
        issues=tuple(DataIssue(**item) for item in payload.get("issues", [])),
    )


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def save_cohort_json(cohort: Cohort, path: str | Path) -> None:
    _atomic_text(Path(path), json.dumps(cohort_to_dict(cohort), indent=2, sort_keys=True) + "\n")


def load_cohort_json(path: str | Path) -> Cohort:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataContractError("cohort_json_invalid", "Unable to read cohort JSON") from exc
    if not isinstance(payload, dict):
        raise DataContractError("cohort_json_invalid", "Cohort JSON root must be an object")
    return cohort_from_dict(payload)


def save_cohort_npz(cohort: Cohort, path: str | Path) -> None:
    """Store canonical JSON inside NPZ without object arrays or pickle."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(cohort_to_dict(cohort), sort_keys=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name}.", suffix=".npz", delete=False
        ) as handle:
            temporary = Path(handle.name)
        np.savez_compressed(temporary, payload=np.asarray(content, dtype=np.str_))
        os.replace(temporary, target)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def load_cohort_npz(path: str | Path) -> Cohort:
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            if set(archive.files) != {"payload"}:
                raise DataContractError(
                    "cohort_npz_invalid", "NPZ must contain only the canonical payload"
                )
            content = str(archive["payload"].item())
        payload = json.loads(content)
    except DataContractError:
        raise
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        raise DataContractError("cohort_npz_invalid", "Unable to read cohort NPZ") from exc
    if not isinstance(payload, dict):
        raise DataContractError("cohort_npz_invalid", "NPZ payload root must be an object")
    return cohort_from_dict(payload)


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa  # type: ignore[import-not-found,import-untyped]  # Optional dependency.
        import pyarrow.parquet as pq  # type: ignore[import-not-found,import-untyped]
    except ImportError as exc:
        raise StageWorldError(
            "dependency_missing",
            "Parquet I/O requires pyarrow; no alternate format was written",
            remediation="Install the clinical-io optional dependencies in an approved environment",
            details={"dependency": "pyarrow"},
        ) from exc
    return pa, pq


def write_records_parquet(records: Iterable[object], path: str | Path) -> None:
    pa, pq = _require_pyarrow()
    rows = [_jsonable(record) for record in records]
    table = pa.Table.from_pylist(rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, target)


def read_records_parquet(path: str | Path) -> list[dict[str, Any]]:
    _, pq = _require_pyarrow()
    return pq.read_table(Path(path)).to_pylist()
