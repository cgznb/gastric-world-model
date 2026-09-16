"""Typed tensor and geometry contracts shared by all encoder adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

from stageworld.errors import DataContractError


def _contract_error(code: str, message: str, **details: Any) -> DataContractError:
    return DataContractError(code=code, message=message, details=details)


def _require_shape(value: Tensor, shape: tuple[int | None, ...], name: str) -> None:
    if value.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(value.shape, shape, strict=True)
    ):
        raise _contract_error(
            "INVALID_TENSOR_SHAPE",
            f"{name} has an invalid shape.",
            field=name,
            actual=list(value.shape),
            expected=["*" if item is None else item for item in shape],
        )


@dataclass(frozen=True)
class EncoderProvenance:
    """Human-readable encoder lineage; no file digest is persisted."""

    encoder_name: str
    source_version: str
    component_versions: tuple[tuple[str, str], ...]
    preprocess_version: str
    feature_dim: int
    frozen_source: bool = True

    def __post_init__(self) -> None:
        if not self.encoder_name or not self.source_version or not self.preprocess_version:
            raise _contract_error(
                "MISSING_ENCODER_VERSION",
                "Encoder name and source/preprocess versions must be explicit.",
            )
        if self.feature_dim <= 0:
            raise _contract_error(
                "INVALID_FEATURE_DIMENSION",
                "Encoder feature_dim must be positive.",
                feature_dim=self.feature_dim,
            )
        names = [name for name, _ in self.component_versions]
        if not names or len(names) != len(set(names)):
            raise _contract_error(
                "INVALID_COMPONENT_VERSIONS",
                "Component versions must be non-empty and uniquely named.",
            )
        if any(not name or not version for name, version in self.component_versions):
            raise _contract_error(
                "INVALID_COMPONENT_VERSIONS",
                "Every encoder component needs a non-empty human-readable version.",
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "encoder_name": self.encoder_name,
            "source_version": self.source_version,
            "component_versions": [
                {"name": name, "version": version} for name, version in self.component_versions
            ],
            "preprocess_version": self.preprocess_version,
            "feature_dim": self.feature_dim,
            "frozen_source": self.frozen_source,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EncoderProvenance:
        raw_components = value.get("component_versions", [])
        if not isinstance(raw_components, list):
            raise _contract_error(
                "INVALID_COMPONENT_VERSIONS",
                "component_versions must be a list of named versions.",
            )
        components: list[tuple[str, str]] = []
        for item in raw_components:
            if not isinstance(item, Mapping):
                raise _contract_error(
                    "INVALID_COMPONENT_VERSIONS",
                    "Each component version must be a mapping.",
                )
            components.append((str(item.get("name", "")), str(item.get("version", ""))))
        return cls(
            encoder_name=str(value.get("encoder_name", "")),
            source_version=str(value.get("source_version", "")),
            component_versions=tuple(components),
            preprocess_version=str(value.get("preprocess_version", "")),
            feature_dim=int(value.get("feature_dim", 0)),
            frozen_source=bool(value.get("frozen_source", False)),
        )


@dataclass(frozen=True)
class ObservationTokens:
    """A batch of modality observations with explicit padding and availability."""

    values: Tensor
    valid: Tensor
    modality: Tensor
    acquired_time: Tensor
    available_time: Tensor
    provenance: EncoderProvenance
    source_id: tuple[tuple[str, ...], ...]
    modality_name: str
    coords: Tensor | None = None
    coordinate_system: str | None = None
    quality_flags: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        _require_shape(self.values, (None, None, self.provenance.feature_dim), "values")
        batch, tokens, _ = self.values.shape
        for name, tensor in (
            ("valid", self.valid),
            ("modality", self.modality),
            ("acquired_time", self.acquired_time),
            ("available_time", self.available_time),
        ):
            _require_shape(tensor, (batch, tokens), name)
            if tensor.device != self.values.device:
                raise _contract_error(
                    "TENSOR_DEVICE_MISMATCH",
                    f"{name} must be on the same device as values.",
                    field=name,
                )
        if self.valid.dtype is not torch.bool:
            raise _contract_error("INVALID_VALID_MASK", "valid must have torch.bool dtype.")
        if self.modality.dtype not in (torch.int32, torch.int64):
            raise _contract_error("INVALID_MODALITY_IDS", "modality_id must have an integer dtype.")
        if len(self.source_id) != batch or any(len(row) != tokens for row in self.source_id):
            raise _contract_error(
                "INVALID_SOURCE_IDS",
                "source_id must contain one string for every padded token.",
            )
        if self.modality_name not in {"ct", "pathology", "clinical"}:
            raise _contract_error(
                "INVALID_MODALITY_NAME",
                "modality_name must explicitly identify ct, pathology, or clinical tokens.",
                modality_name=self.modality_name,
            )
        if self.quality_flags and len(self.quality_flags) != batch:
            raise _contract_error(
                "INVALID_QUALITY_FLAGS",
                "quality_flags must be empty or contain one tuple per batch item.",
            )
        if self.coords is None:
            if self.coordinate_system is not None:
                raise _contract_error(
                    "COORDINATE_SYSTEM_WITHOUT_COORDS",
                    "coordinate_system cannot be set when coords are absent.",
                )
        else:
            if self.coords.ndim != 3 or self.coords.shape[:2] != (batch, tokens):
                raise _contract_error(
                    "INVALID_COORDINATE_SHAPE",
                    "coords must have shape [B, N, C].",
                    actual=list(self.coords.shape),
                )
            if self.coords.shape[-1] not in (2, 3) or not self.coordinate_system:
                raise _contract_error(
                    "INVALID_COORDINATE_SYSTEM",
                    "Coordinates require 2 or 3 axes and an explicit coordinate system.",
                )
            if self.coords.device != self.values.device:
                raise _contract_error(
                    "TENSOR_DEVICE_MISMATCH",
                    "coords must be on the same device as values.",
                    field="coords",
                )
        selected = self.valid
        if selected.any():
            valid_sources = self.provenance_ids()
            if any(not source for row in valid_sources for source in row):
                raise _contract_error(
                    "INVALID_SOURCE_IDS",
                    "Every valid observation token needs a non-empty source identifier.",
                )
            if not torch.isfinite(self.values[selected]).all():
                raise _contract_error(
                    "NONFINITE_OBSERVATION", "Valid observation values must be finite."
                )
            for name, tensor in (
                ("acquired_time", self.acquired_time),
                ("available_time", self.available_time),
            ):
                if not torch.isfinite(tensor[selected]).all():
                    raise _contract_error(
                        "NONFINITE_OBSERVATION_TIME", f"Valid {name} entries must be finite."
                    )
            if torch.any(self.available_time[selected] < self.acquired_time[selected]):
                raise _contract_error(
                    "AVAILABILITY_PRECEDES_ACQUISITION",
                    "Observation available_time cannot precede acquired_time.",
                )
            if self.coords is not None and not torch.isfinite(self.coords[selected]).all():
                raise _contract_error(
                    "NONFINITE_COORDINATES", "Valid observation coordinates must be finite."
                )
            if torch.any(self.modality[selected] < 0):
                raise _contract_error(
                    "INVALID_MODALITY_IDS", "Valid modality identifiers must be non-negative."
                )

    @property
    def encoder_version(self) -> str:
        return self.provenance.source_version

    @property
    def modality_id(self) -> Tensor:
        """Compatibility alias for code that names the categorical tensor explicitly."""

        return self.modality

    @property
    def source_ids(self) -> tuple[tuple[str, ...], ...]:
        """Compatibility alias emphasizing that identifiers follow the token axis."""

        return self.source_id

    def validate(self) -> None:
        """Compatibility hook; construction has already performed full validation."""

        return None

    def provenance_ids(self) -> tuple[tuple[str, ...], ...]:
        """Return source identifiers for valid tokens only, excluding padding."""

        mask = self.valid.detach().cpu()
        return tuple(
            tuple(source for source, keep in zip(row, mask[index].tolist(), strict=True) if keep)
            for index, row in enumerate(self.source_id)
        )

    @property
    def batch_size(self) -> int:
        return self.values.shape[0]

    @property
    def token_count(self) -> int:
        return self.values.shape[1]

    def as_cache_payload(self) -> dict[str, Any]:
        return {
            "values": self.values.detach().cpu(),
            "valid": self.valid.detach().cpu(),
            "modality": self.modality.detach().cpu(),
            "acquired_time": self.acquired_time.detach().cpu(),
            "available_time": self.available_time.detach().cpu(),
            "coords": None if self.coords is None else self.coords.detach().cpu(),
            "coordinate_system": self.coordinate_system,
            "modality_name": self.modality_name,
            "source_id": [list(row) for row in self.source_id],
            "quality_flags": [list(row) for row in self.quality_flags],
            "provenance": self.provenance.as_dict(),
        }

    @classmethod
    def from_cache_payload(cls, payload: Mapping[str, Any]) -> ObservationTokens:
        return cls(
            values=payload["values"],
            valid=payload["valid"],
            modality=payload["modality"],
            acquired_time=payload["acquired_time"],
            available_time=payload["available_time"],
            coords=payload.get("coords"),
            coordinate_system=payload.get("coordinate_system"),
            modality_name=str(payload["modality_name"]),
            source_id=tuple(tuple(str(item) for item in row) for row in payload["source_id"]),
            quality_flags=tuple(
                tuple(str(item) for item in row) for row in payload.get("quality_flags", [])
            ),
            provenance=EncoderProvenance.from_dict(payload["provenance"]),
        )


def merge_observation_tokens(values: Sequence[ObservationTokens]) -> ObservationTokens:
    """Concatenate records from one modality before a single resampling operation."""

    if not values:
        raise _contract_error("NO_OBSERVATIONS", "At least one observation token set is required.")
    first = values[0]
    for value in values:
        if value.batch_size != first.batch_size:
            raise _contract_error(
                "OBSERVATION_BATCH_MISMATCH", "Merged observations must share batch size."
            )
        if value.modality_name != first.modality_name or value.provenance != first.provenance:
            raise _contract_error(
                "OBSERVATION_SPACE_MISMATCH",
                "Merged observations must share modality and encoder provenance.",
            )
        if value.coordinate_system != first.coordinate_system:
            raise _contract_error(
                "COORDINATE_SYSTEM_MISMATCH",
                "Merged observations must share a coordinate system.",
            )
        if (value.coords is None) != (first.coords is None):
            raise _contract_error(
                "COORDINATE_PRESENCE_MISMATCH",
                "Merged observations must either all provide coordinates or all omit them.",
            )
    quality_flags = tuple(
        tuple(
            dict.fromkeys(
                flag
                for value in values
                for flag in (value.quality_flags[row] if value.quality_flags else ())
            )
        )
        for row in range(first.batch_size)
    )
    return ObservationTokens(
        values=torch.cat([value.values for value in values], dim=1),
        valid=torch.cat([value.valid for value in values], dim=1),
        modality=torch.cat([value.modality for value in values], dim=1),
        acquired_time=torch.cat([value.acquired_time for value in values], dim=1),
        available_time=torch.cat([value.available_time for value in values], dim=1),
        provenance=first.provenance,
        source_id=tuple(
            tuple(source for value in values for source in value.source_id[row])
            for row in range(first.batch_size)
        ),
        modality_name=first.modality_name,
        coords=(
            None
            if first.coords is None
            else torch.cat([value.coords for value in values if value.coords is not None], dim=1)
        ),
        coordinate_system=first.coordinate_system,
        quality_flags=quality_flags,
    )


@dataclass(frozen=True)
class EncoderOutput:
    observations: ObservationTokens
    global_embedding: Tensor | None = None

    def __post_init__(self) -> None:
        if self.global_embedding is not None:
            _require_shape(
                self.global_embedding,
                (self.observations.batch_size, None),
                "global_embedding",
            )
            if not torch.isfinite(self.global_embedding).all():
                raise _contract_error(
                    "NONFINITE_GLOBAL_EMBEDDING", "global_embedding must be finite."
                )


@runtime_checkable
class ObservationEncoder(Protocol):
    """Minimal protocol implemented by every StageWorld observation encoder."""

    name: str

    @property
    def provenance(self) -> EncoderProvenance: ...

    def encode(self, *args: Any, **kwargs: Any) -> EncoderOutput: ...


@dataclass(frozen=True)
class CTGeometry:
    """Physical geometry for a batch of channel-first CT volumes."""

    spacing_mm: Tensor
    origin_mm: Tensor
    direction: Tensor
    spatial_shape: Tensor

    def __post_init__(self) -> None:
        _require_shape(self.spacing_mm, (None, 3), "spacing_mm")
        batch = self.spacing_mm.shape[0]
        _require_shape(self.origin_mm, (batch, 3), "origin_mm")
        _require_shape(self.direction, (batch, 3, 3), "direction")
        _require_shape(self.spatial_shape, (batch, 3), "spatial_shape")
        if not all(
            torch.isfinite(item).all() for item in (self.spacing_mm, self.origin_mm, self.direction)
        ):
            raise _contract_error("INVALID_CT_GEOMETRY", "CT geometry must be finite.")
        if torch.any(self.spacing_mm <= 0) or torch.any(self.spatial_shape <= 0):
            raise _contract_error(
                "INVALID_CT_GEOMETRY", "CT spacing and spatial_shape must be positive."
            )

    @property
    def batch_size(self) -> int:
        return self.spacing_mm.shape[0]

    def feature_grid_centers(
        self, feature_shape: tuple[int, int, int], *, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        """Return feature-cell centers in physical millimetres."""

        if any(size <= 0 for size in feature_shape):
            raise _contract_error(
                "INVALID_FEATURE_GRID", "Every feature-grid axis must be positive."
            )
        spatial = self.spatial_shape.to(device=device, dtype=dtype)
        axes = []
        for axis, size in enumerate(feature_shape):
            position = torch.arange(size, device=device, dtype=dtype) + 0.5
            axes.append(position[None, :] * spatial[:, axis : axis + 1] / size - 0.5)
        batch_coords = []
        for batch_index in range(self.batch_size):
            grid = torch.stack(
                torch.meshgrid(
                    axes[0][batch_index],
                    axes[1][batch_index],
                    axes[2][batch_index],
                    indexing="ij",
                ),
                dim=-1,
            ).reshape(-1, 3)
            scaled = grid * self.spacing_mm[batch_index].to(device=device, dtype=dtype)
            rotated = scaled @ self.direction[batch_index].to(device=device, dtype=dtype).T
            batch_coords.append(
                rotated + self.origin_mm[batch_index].to(device=device, dtype=dtype)
            )
        return torch.stack(batch_coords)


@dataclass(frozen=True)
class WSIGeometry:
    """Level-0 WSI patch geometry with explicit physical pixel size."""

    coords_level0: Tensor
    valid: Tensor
    mpp: Tensor
    patch_size_level0: Tensor
    level0_size: Tensor | None = None

    def __post_init__(self) -> None:
        _require_shape(self.coords_level0, (None, None, 2), "coords_level0")
        batch, patches, _ = self.coords_level0.shape
        _require_shape(self.valid, (batch, patches), "valid")
        _require_shape(self.mpp, (batch, 2), "mpp")
        _require_shape(self.patch_size_level0, (batch,), "patch_size_level0")
        if self.valid.dtype is not torch.bool:
            raise _contract_error("INVALID_VALID_MASK", "WSI valid must have torch.bool dtype.")
        if not torch.isfinite(self.mpp).all() or torch.any(self.mpp <= 0):
            raise _contract_error("INVALID_MPP", "MPP must be finite and positive.")
        if torch.any(self.patch_size_level0 <= 0):
            raise _contract_error("INVALID_PATCH_SIZE", "patch_size_level0 must be positive.")
        selected = self.valid
        if selected.any():
            coords = self.coords_level0[selected]
            if not torch.isfinite(coords).all() or torch.any(coords < 0):
                raise _contract_error(
                    "INVALID_LEVEL0_COORDINATES",
                    "Valid level-0 coordinates must be finite and non-negative.",
                )
            if not torch.allclose(coords, coords.round()):
                raise _contract_error(
                    "INVALID_LEVEL0_COORDINATES",
                    "Level-0 coordinates must identify integer pixels.",
                )
        if self.level0_size is not None:
            _require_shape(self.level0_size, (batch, 2), "level0_size")
            if torch.any(self.level0_size <= 0):
                raise _contract_error("INVALID_WSI_SIZE", "level0_size must be positive.")
            patch = self.patch_size_level0[:, None, None]
            limit = self.level0_size[:, None, :]
            if torch.any(((self.coords_level0 + patch) > limit) & self.valid[..., None]):
                raise _contract_error(
                    "PATCH_OUTSIDE_WSI", "A valid patch extends beyond the level-0 slide bounds."
                )

    @property
    def batch_size(self) -> int:
        return self.coords_level0.shape[0]

    def physical_centers_mm(self) -> Tensor:
        patch = self.patch_size_level0.to(self.coords_level0)[:, None, None]
        centers_level0 = self.coords_level0 + patch / 2.0
        return centers_level0 * self.mpp.to(self.coords_level0)[:, None, :] / 1000.0


@dataclass(frozen=True)
class PathologyPatchBatch:
    features: Tensor
    geometry: WSIGeometry
    patch_encoder_name: str
    patch_encoder_version: str
    input_patch_size_px: int
    magnification: float
    slide_ids: tuple[str, ...]
    tissue_sources: tuple[str, ...]
    acquired_time: Tensor
    available_time: Tensor

    def __post_init__(self) -> None:
        _require_shape(self.features, (self.geometry.batch_size, None, None), "features")
        if self.features.shape[:2] != self.geometry.valid.shape:
            raise _contract_error(
                "FEATURE_COORDINATE_COUNT_MISMATCH",
                "Patch features and WSI coordinates must have identical [B, N] axes.",
            )
        batch = self.features.shape[0]
        if len(self.slide_ids) != batch or len(self.tissue_sources) != batch:
            raise _contract_error(
                "INVALID_SLIDE_METADATA", "Each batch item needs a slide and tissue source."
            )
        _require_shape(self.acquired_time, (batch,), "acquired_time")
        _require_shape(self.available_time, (batch,), "available_time")
        if torch.any(self.available_time < self.acquired_time):
            raise _contract_error(
                "AVAILABILITY_PRECEDES_ACQUISITION",
                "Slide available_time cannot precede acquired_time.",
            )
        if self.input_patch_size_px <= 0 or self.magnification <= 0:
            raise _contract_error(
                "INVALID_PATCH_PROTOCOL", "Patch input size and magnification must be positive."
            )
        if not self.patch_encoder_name or not self.patch_encoder_version:
            raise _contract_error(
                "MISSING_PATCH_ENCODER_VERSION", "Patch encoder identity/version is required."
            )
        if (
            self.geometry.valid.any()
            and not torch.isfinite(self.features[self.geometry.valid]).all()
        ):
            raise _contract_error(
                "NONFINITE_PATCH_FEATURES", "Valid patch features must be finite."
            )
