"""Small artifact helpers using lineage IDs rather than persisted checksums."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def new_artifact_id(prefix: str) -> str:
    clean = "".join(char if char.isalnum() or char in "-_" else "-" for char in prefix)
    return f"{clean}-{uuid.uuid4()}"


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def atomic_write_private_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically write a local restricted manifest with owner-only permissions."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    target.parent.chmod(0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        target.chmod(0o600)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("JSON artifact root must be an object")
    return value
