"""Atomic, version-exact cache for frozen observation features."""

from __future__ import annotations

import fcntl
import json
import os
import pickle
import re
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import torch

from stageworld.encoders.base import EncoderProvenance, ObservationTokens
from stageworld.encoders.gates import validate_named_version
from stageworld.errors import ArtifactError, DataContractError, StageWorldError

_SAFE_ENTRY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
CacheState = Literal["incomplete", "complete", "failed"]


@dataclass(frozen=True)
class CacheProvenance:
    schema_version: str
    encoder_name: str
    encoder_source_version: str
    component_versions: tuple[tuple[str, str], ...]
    preprocess_version: str
    patch_sampling_version: str
    split_version: str
    target_transform_version: str
    teacher_version: str
    feature_dim: int
    frozen_source: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "schema_version",
            "encoder_source_version",
            "preprocess_version",
            "patch_sampling_version",
            "split_version",
            "target_transform_version",
            "teacher_version",
        ):
            validate_named_version(str(getattr(self, field_name)), field_name=field_name)
        if not self.encoder_name:
            raise DataContractError(
                code="MISSING_ENCODER_IDENTITY", message="encoder_name is required."
            )
        if self.feature_dim <= 0:
            raise DataContractError(
                code="INVALID_FEATURE_DIMENSION", message="feature_dim must be positive."
            )
        names = [name for name, _ in self.component_versions]
        if not names or len(names) != len(set(names)):
            raise DataContractError(
                code="INVALID_COMPONENT_VERSIONS",
                message="Cache component versions must be named and unique.",
            )
        for name, version in self.component_versions:
            if not name:
                raise DataContractError(
                    code="INVALID_COMPONENT_VERSIONS", message="Component names cannot be empty."
                )
            validate_named_version(version, field_name=f"component:{name}")

    @classmethod
    def from_encoder(
        cls,
        encoder: EncoderProvenance,
        *,
        schema_version: str,
        patch_sampling_version: str,
        split_version: str,
        target_transform_version: str,
        teacher_version: str,
    ) -> CacheProvenance:
        return cls(
            schema_version=schema_version,
            encoder_name=encoder.encoder_name,
            encoder_source_version=encoder.source_version,
            component_versions=encoder.component_versions,
            preprocess_version=encoder.preprocess_version,
            patch_sampling_version=patch_sampling_version,
            split_version=split_version,
            target_transform_version=target_transform_version,
            teacher_version=teacher_version,
            feature_dim=encoder.feature_dim,
            frozen_source=encoder.frozen_source,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "encoder_name": self.encoder_name,
            "encoder_source_version": self.encoder_source_version,
            "component_versions": [
                {"name": name, "version": version} for name, version in self.component_versions
            ],
            "preprocess_version": self.preprocess_version,
            "patch_sampling_version": self.patch_sampling_version,
            "split_version": self.split_version,
            "target_transform_version": self.target_transform_version,
            "teacher_version": self.teacher_version,
            "feature_dim": self.feature_dim,
            "frozen_source": self.frozen_source,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CacheProvenance:
        components_raw = value.get("component_versions")
        if not isinstance(components_raw, list):
            raise DataContractError(
                code="INVALID_COMPONENT_VERSIONS",
                message="component_versions must be a list of named versions.",
            )
        components = tuple(
            (str(item["name"]), str(item["version"]))
            for item in components_raw
            if isinstance(item, Mapping)
        )
        if len(components) != len(components_raw):
            raise DataContractError(
                code="INVALID_COMPONENT_VERSIONS",
                message="Every component version entry must be a mapping.",
            )
        return cls(
            schema_version=str(value.get("schema_version", "")),
            encoder_name=str(value.get("encoder_name", "")),
            encoder_source_version=str(value.get("encoder_source_version", "")),
            component_versions=components,
            preprocess_version=str(value.get("preprocess_version", "")),
            patch_sampling_version=str(value.get("patch_sampling_version", "")),
            split_version=str(value.get("split_version", "")),
            target_transform_version=str(value.get("target_transform_version", "")),
            teacher_version=str(value.get("teacher_version", "")),
            feature_dim=int(value.get("feature_dim", 0)),
            frozen_source=bool(value.get("frozen_source", False)),
        )


@dataclass(frozen=True)
class CacheDecision:
    state: CacheState
    resume: bool
    already_complete: bool


class FeatureCache:
    """One-directory-per-entry cache with exact provenance and crash-safe writes."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self._private_directory(self.root)

    @staticmethod
    def _private_directory(path: Path) -> None:
        """Create a cache directory that is private to the current OS user."""

        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.chmod(0o700)

    def _entry_dir(self, entry_id: str) -> Path:
        if not _SAFE_ENTRY.fullmatch(entry_id) or entry_id in {".", ".."}:
            raise ArtifactError(
                code="INVALID_CACHE_ENTRY_ID",
                message="Cache entry identifiers may contain only safe filename characters.",
            )
        return self.root / entry_id

    @contextmanager
    def _locked(self, entry_id: str) -> Iterator[Path]:
        directory = self._entry_dir(entry_id)
        self._private_directory(directory)
        lock_path = directory / ".lock"
        lock_path.touch(mode=0o600, exist_ok=True)
        lock_path.chmod(0o600)
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield directory
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
        descriptor, temporary_name = tempfile.mkstemp(prefix=".status-", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(value, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _atomic_payload(path: Path, payload: Mapping[str, Any]) -> None:
        descriptor, temporary_name = tempfile.mkstemp(prefix=".tokens-", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                torch.save(dict(payload), handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _read_status(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactError(
                code="CORRUPT_CACHE_STATUS", message="Cache status is absent or malformed."
            ) from error
        if not isinstance(value, dict) or value.get("state") not in {
            "incomplete",
            "complete",
            "failed",
        }:
            raise ArtifactError(
                code="CORRUPT_CACHE_STATUS", message="Cache status has an invalid state."
            )
        return value

    @staticmethod
    def _status_payload(
        state: CacheState,
        provenance: CacheProvenance,
        *,
        failure_code: str | None = None,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "state": state,
            "updated_at": FeatureCache._utc_now(),
            "provenance": provenance.as_dict(),
        }
        if failure_code:
            value["failure_code"] = failure_code
        return value

    def begin(self, entry_id: str, provenance: CacheProvenance) -> CacheDecision:
        if not provenance.frozen_source:
            raise ArtifactError(
                code="ONLINE_FEATURE_CACHE_FORBIDDEN",
                message="Only frozen external encoder features may use this cache.",
            )
        with self._locked(entry_id) as directory:
            status_path = directory / "status.json"
            resume = False
            if status_path.exists():
                status = self._read_status(status_path)
                try:
                    previous = CacheProvenance.from_dict(status["provenance"])
                except (KeyError, TypeError, ValueError, StageWorldError) as error:
                    raise ArtifactError(
                        code="CORRUPT_CACHE_STATUS",
                        message="Cache status provenance is malformed.",
                    ) from error
                if previous == provenance and status["state"] == "complete":
                    if not (directory / "tokens.pt").is_file():
                        raise ArtifactError(
                            code="HALF_WRITTEN_CACHE",
                            message="Complete cache status has no payload.",
                        )
                    return CacheDecision("complete", resume=False, already_complete=True)
                resume = previous == provenance and status["state"] in {"incomplete", "failed"}
            self._atomic_json(status_path, self._status_payload("incomplete", provenance))
            return CacheDecision("incomplete", resume=resume, already_complete=False)

    def store(self, entry_id: str, provenance: CacheProvenance, tokens: ObservationTokens) -> None:
        if not provenance.frozen_source or not tokens.provenance.frozen_source:
            raise ArtifactError(
                code="ONLINE_FEATURE_CACHE_FORBIDDEN",
                message="Online/trainable encoder outputs cannot be cached as fixed targets.",
            )
        tensors = [
            tokens.values,
            tokens.valid,
            tokens.modality,
            tokens.acquired_time,
            tokens.available_time,
        ]
        if tokens.coords is not None:
            tensors.append(tokens.coords)
        if any(tensor.requires_grad for tensor in tensors):
            raise ArtifactError(
                code="ONLINE_FEATURE_CACHE_FORBIDDEN",
                message="Tensors requiring gradients cannot be written to the frozen cache.",
            )
        expected_encoder = (
            provenance.encoder_name,
            provenance.encoder_source_version,
            provenance.component_versions,
            provenance.preprocess_version,
            provenance.feature_dim,
        )
        actual_encoder = (
            tokens.provenance.encoder_name,
            tokens.provenance.source_version,
            tokens.provenance.component_versions,
            tokens.provenance.preprocess_version,
            tokens.provenance.feature_dim,
        )
        if expected_encoder != actual_encoder:
            raise ArtifactError(
                code="CACHE_ENCODER_PROVENANCE_MISMATCH",
                message="Observation features do not match cache encoder provenance.",
            )
        with self._locked(entry_id) as directory:
            status_path = directory / "status.json"
            if not status_path.exists():
                raise ArtifactError(
                    code="CACHE_NOT_BEGUN", message="Call begin() before storing a cache entry."
                )
            status = self._read_status(status_path)
            try:
                current = CacheProvenance.from_dict(status["provenance"])
            except (KeyError, TypeError, ValueError, StageWorldError) as error:
                raise ArtifactError(
                    code="CORRUPT_CACHE_STATUS", message="Cache provenance is malformed."
                ) from error
            if current != provenance or status["state"] != "incomplete":
                raise ArtifactError(
                    code="CACHE_WRITE_STATE_MISMATCH",
                    message="Cache entry is not incomplete under the requested provenance.",
                )
            self._atomic_payload(directory / "tokens.pt", tokens.as_cache_payload())
            self._atomic_json(status_path, self._status_payload("complete", provenance))

    def mark_failed(self, entry_id: str, provenance: CacheProvenance, *, failure_code: str) -> None:
        if not failure_code or not re.fullmatch(r"[A-Z][A-Z0-9_]*", failure_code):
            raise ArtifactError(
                code="INVALID_FAILURE_CODE",
                message="Cache failures require a non-sensitive structured code.",
            )
        with self._locked(entry_id) as directory:
            status_path = directory / "status.json"
            if not status_path.exists():
                raise ArtifactError(
                    code="CACHE_NOT_BEGUN", message="Call begin() before marking an entry failed."
                )
            status = self._read_status(status_path)
            try:
                current = CacheProvenance.from_dict(status["provenance"])
            except (KeyError, TypeError, ValueError, StageWorldError) as error:
                raise ArtifactError(
                    code="CORRUPT_CACHE_STATUS", message="Cache provenance is malformed."
                ) from error
            if current != provenance or status["state"] != "incomplete":
                raise ArtifactError(
                    code="CACHE_WRITE_STATE_MISMATCH",
                    message="Only the active incomplete cache entry may be marked failed.",
                )
            self._atomic_json(
                status_path,
                self._status_payload("failed", provenance, failure_code=failure_code),
            )

    def load(self, entry_id: str, provenance: CacheProvenance) -> ObservationTokens:
        with self._locked(entry_id) as directory:
            status_path = directory / "status.json"
            payload_path = directory / "tokens.pt"
            if not status_path.exists():
                if payload_path.exists():
                    raise ArtifactError(
                        code="HALF_WRITTEN_CACHE",
                        message="Cache payload exists without a status record.",
                    )
                raise ArtifactError(code="CACHE_MISS", message="Cache entry does not exist.")
            status = self._read_status(status_path)
            try:
                actual = CacheProvenance.from_dict(status["provenance"])
            except (KeyError, TypeError, ValueError, StageWorldError) as error:
                raise ArtifactError(
                    code="CORRUPT_CACHE_STATUS", message="Cache provenance is malformed."
                ) from error
            if actual != provenance:
                raise ArtifactError(
                    code="STALE_CACHE_PROVENANCE",
                    message="Cache provenance differs from the requested encoder/data versions.",
                )
            if status["state"] != "complete":
                raise ArtifactError(
                    code="CACHE_NOT_COMPLETE",
                    message=f"Cache entry is {status['state']} and cannot be read.",
                    details={"state": status["state"]},
                )
            if not payload_path.is_file():
                raise ArtifactError(
                    code="HALF_WRITTEN_CACHE", message="Complete cache status has no payload."
                )
            try:
                payload = torch.load(payload_path, map_location="cpu", weights_only=True)
                if not isinstance(payload, Mapping):
                    raise TypeError("payload is not a mapping")
                tokens = ObservationTokens.from_cache_payload(payload)
            except (
                OSError,
                RuntimeError,
                EOFError,
                KeyError,
                TypeError,
                ValueError,
                pickle.UnpicklingError,
                StageWorldError,
            ) as error:
                raise ArtifactError(
                    code="CORRUPT_CACHE_PAYLOAD",
                    message="Cache payload is malformed or incomplete.",
                ) from error
            expected_encoder = (
                provenance.encoder_name,
                provenance.encoder_source_version,
                provenance.component_versions,
                provenance.preprocess_version,
                provenance.feature_dim,
                provenance.frozen_source,
            )
            actual_encoder = (
                tokens.provenance.encoder_name,
                tokens.provenance.source_version,
                tokens.provenance.component_versions,
                tokens.provenance.preprocess_version,
                tokens.provenance.feature_dim,
                tokens.provenance.frozen_source,
            )
            if expected_encoder != actual_encoder:
                raise ArtifactError(
                    code="CACHE_ENCODER_PROVENANCE_MISMATCH",
                    message="Payload encoder identity differs from its cache status.",
                )
            return tokens

    def get_or_compute(
        self,
        entry_id: str,
        provenance: CacheProvenance,
        producer: Callable[[bool], ObservationTokens],
    ) -> ObservationTokens:
        decision = self.begin(entry_id, provenance)
        if decision.already_complete:
            return self.load(entry_id, provenance)
        try:
            tokens = producer(decision.resume)
            self.store(entry_id, provenance, tokens)
        except Exception as error:
            failure_code = str(getattr(error, "code", "ENCODER_EXTRACTION_FAILED"))
            if not re.fullmatch(r"[A-Z][A-Z0-9_]*", failure_code):
                failure_code = "ENCODER_EXTRACTION_FAILED"
            self.mark_failed(entry_id, provenance, failure_code=failure_code)
            raise
        return self.load(entry_id, provenance)


AtomicFeatureCache = FeatureCache
