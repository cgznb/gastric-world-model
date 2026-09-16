"""Strict adapters around authorized, locally supplied medical encoders.

The adapters deliberately do not import model hubs or construct upstream models. A caller must
inject an already-audited backend, and image execution is gated before that backend is invoked.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Protocol

import torch
from torch import Tensor, nn

from stageworld.errors import ArtifactError, ConfigurationError, DataContractError

from .base import (
    CTGeometry,
    EncoderOutput,
    EncoderProvenance,
    ObservationTokens,
    PathologyPatchBatch,
)
from .gates import EncoderAccess


class EncoderBackend(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class StateDictLoadReport:
    loaded_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[str, ...]
    loaded_parameter_fraction: float


def _allowed(key: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch(key, pattern) for pattern in patterns)


def load_state_dict_strictly(
    module: nn.Module,
    state_dict: Mapping[str, Tensor],
    *,
    allowed_missing: tuple[str, ...] = (),
    allowed_unexpected: tuple[str, ...] = (),
    minimum_parameter_fraction: float = 1.0,
) -> StateDictLoadReport:
    """Preflight a state dict and load it only when explicit coverage rules pass."""

    if not 0.0 <= minimum_parameter_fraction <= 1.0:
        raise ConfigurationError(
            code="INVALID_STATE_DICT_THRESHOLD",
            message="minimum_parameter_fraction must be between zero and one.",
        )
    expected = module.state_dict()
    missing = tuple(sorted(set(expected) - set(state_dict)))
    unexpected = tuple(sorted(set(state_dict) - set(expected)))
    shape_mismatches = tuple(
        sorted(
            key
            for key in set(expected) & set(state_dict)
            if tuple(expected[key].shape) != tuple(state_dict[key].shape)
        )
    )
    loaded = tuple(
        sorted(key for key in set(expected) & set(state_dict) if key not in shape_mismatches)
    )
    parameter_keys = {name for name, _ in module.named_parameters()}
    total_parameters = sum(expected[key].numel() for key in parameter_keys)
    loaded_parameters = sum(expected[key].numel() for key in loaded if key in parameter_keys)
    fraction = loaded_parameters / total_parameters if total_parameters else 1.0
    report = StateDictLoadReport(
        loaded_keys=loaded,
        missing_keys=missing,
        unexpected_keys=unexpected,
        shape_mismatches=shape_mismatches,
        loaded_parameter_fraction=fraction,
    )
    disallowed_missing = tuple(key for key in missing if not _allowed(key, allowed_missing))
    disallowed_unexpected = tuple(
        key for key in unexpected if not _allowed(key, allowed_unexpected)
    )
    if (
        shape_mismatches
        or disallowed_missing
        or disallowed_unexpected
        or fraction < minimum_parameter_fraction
    ):
        raise ArtifactError(
            code="STATE_DICT_COVERAGE_FAILED",
            message="Encoder state dict does not satisfy the declared coverage contract.",
            details={
                "missing_keys": list(missing),
                "unexpected_keys": list(unexpected),
                "shape_mismatches": list(shape_mismatches),
                "loaded_parameter_fraction": fraction,
                "minimum_parameter_fraction": minimum_parameter_fraction,
            },
        )
    module.load_state_dict(dict(state_dict), strict=False)
    return report


def _provenance(access: EncoderAccess, name: str, feature_dim: int) -> EncoderProvenance:
    return EncoderProvenance(
        encoder_name=name,
        source_version=access.source_version,
        component_versions=tuple(sorted(access.component_versions.items())),
        preprocess_version=access.preprocess_version,
        feature_dim=feature_dim,
    )


def _require_backend(backend: EncoderBackend | None, name: str) -> EncoderBackend:
    if backend is None:
        raise ArtifactError(
            code="ENCODER_BACKEND_MISSING",
            message=f"{name} requires an explicitly injected local upstream backend.",
        )
    return backend


def _times_for_tokens(value: Tensor, tokens: int, field: str, *, device: torch.device) -> Tensor:
    if value.ndim != 1:
        raise DataContractError(
            code="INVALID_OBSERVATION_TIME",
            message=f"{field} must have shape [B].",
            details={"field": field, "actual": list(value.shape)},
        )
    return value.to(device=device)[:, None].expand(-1, tokens)


def _source_matrix(ids: tuple[str, ...], tokens: int) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(source for _ in range(tokens)) for source in ids)


class _MedicalAdapter:
    name: str
    required_components: tuple[str, ...] = ()
    modality_name: str

    def __init__(self, *, access: EncoderAccess, feature_dim: int) -> None:
        if feature_dim <= 0:
            raise ConfigurationError(
                code="INVALID_FEATURE_DIMENSION", message="feature_dim must be positive."
            )
        self.access = access
        self.feature_dim = feature_dim
        self._provenance = _provenance(access, self.name, feature_dim)

    @property
    def provenance(self) -> EncoderProvenance:
        return self._provenance

    def validate_precomputed(self, tokens: ObservationTokens) -> EncoderOutput:
        self.access.validate_feature_access(self.required_components)
        actual = tokens.provenance
        expected = self.provenance
        if tokens.modality_name != self.modality_name:
            raise DataContractError(
                code="ENCODER_MODALITY_MISMATCH",
                message="Precomputed features do not match the adapter modality.",
                details={
                    "expected_modality": self.modality_name,
                    "actual_modality": tokens.modality_name,
                },
            )
        if actual != expected:
            raise DataContractError(
                code="ENCODER_PROVENANCE_MISMATCH",
                message="Precomputed features do not match the selected encoder contract.",
                details={
                    "expected_encoder": expected.encoder_name,
                    "actual_encoder": actual.encoder_name,
                    "expected_feature_dim": expected.feature_dim,
                    "actual_feature_dim": actual.feature_dim,
                },
            )
        if not actual.frozen_source or tokens.values.requires_grad:
            raise DataContractError(
                code="ONLINE_FEATURES_NOT_ALLOWED",
                message="real_features mode accepts only frozen, detached encoder outputs.",
            )
        return EncoderOutput(observations=tokens)


class MerlinEncoder(_MedicalAdapter):
    """Merlin global baseline plus optional genuine spatial feature-map exposure."""

    name = "merlin"
    modality_name = "ct"
    required_components = ("merlin",)

    def __init__(
        self,
        *,
        access: EncoderAccess,
        backend: EncoderBackend | None = None,
        feature_dim: int = 2048,
        native_global_dim: int = 2048,
    ) -> None:
        super().__init__(access=access, feature_dim=feature_dim)
        self.backend = backend
        self.native_global_dim = native_global_dim

    def encode(
        self,
        images: Tensor,
        *,
        geometry: CTGeometry,
        source_ids: tuple[str, ...],
        acquired_time: Tensor,
        available_time: Tensor,
    ) -> EncoderOutput:
        self.access.validate_image_access(
            required_components=self.required_components,
            required_weights=("merlin",),
            required_dependencies=("torch",),
        )
        backend = _require_backend(self.backend, self.name)
        result = backend(images)
        global_embedding: object
        spatial: object
        if isinstance(result, Tensor):
            global_embedding, spatial = result, None
        elif isinstance(result, Mapping):
            global_embedding = result.get("global_embedding")
            spatial = result.get("spatial_features")
        else:
            raise DataContractError(
                code="INVALID_BACKEND_OUTPUT", message="Merlin backend returned an invalid type."
            )
        if not isinstance(global_embedding, Tensor) or global_embedding.ndim != 2:
            raise DataContractError(
                code="INVALID_MERLIN_GLOBAL", message="Merlin global output must be [B, D]."
            )
        batch = global_embedding.shape[0]
        if batch != geometry.batch_size or len(source_ids) != batch:
            raise DataContractError(
                code="ENCODER_BATCH_MISMATCH", message="Merlin batch metadata do not align."
            )
        if global_embedding.shape[1] != self.native_global_dim:
            raise DataContractError(
                code="INVALID_MERLIN_GLOBAL",
                message="Merlin native global output has an unexpected feature dimension.",
                details={
                    "actual": global_embedding.shape[1],
                    "expected": self.native_global_dim,
                },
            )
        if spatial is None:
            if global_embedding.shape[1] != self.feature_dim:
                raise DataContractError(
                    code="INVALID_FEATURE_DIMENSION",
                    message="Merlin global feature dimension does not match its contract.",
                    details={"actual": global_embedding.shape[1], "expected": self.feature_dim},
                )
            values = global_embedding[:, None, :]
            coords = None
            coordinate_system = None
        else:
            if not isinstance(spatial, Tensor) or spatial.ndim != 5:
                raise DataContractError(
                    code="INVALID_SPATIAL_FEATURE_MAP",
                    message="Merlin spatial output must be an actual [B, C, D, H, W] map.",
                )
            if spatial.shape[0] != batch or spatial.shape[1] != self.feature_dim:
                raise DataContractError(
                    code="INVALID_FEATURE_DIMENSION",
                    message="Merlin spatial feature map does not match the declared dimension.",
                )
            feature_shape = (
                int(spatial.shape[2]),
                int(spatial.shape[3]),
                int(spatial.shape[4]),
            )
            values = spatial.permute(0, 2, 3, 4, 1).reshape(batch, -1, self.feature_dim)
            coords = geometry.feature_grid_centers(
                feature_shape, device=values.device, dtype=values.dtype
            )
            coordinate_system = "patient_physical_mm"
        tokens = values.shape[1]
        observations = ObservationTokens(
            values=values,
            valid=torch.ones((batch, tokens), dtype=torch.bool, device=values.device),
            modality=torch.zeros((batch, tokens), dtype=torch.long, device=values.device),
            acquired_time=_times_for_tokens(
                acquired_time, tokens, "acquired_time", device=values.device
            ),
            available_time=_times_for_tokens(
                available_time, tokens, "available_time", device=values.device
            ),
            provenance=self.provenance,
            source_id=_source_matrix(source_ids, tokens),
            modality_name="ct",
            coords=coords,
            coordinate_system=coordinate_system,
        )
        return EncoderOutput(observations=observations, global_embedding=global_embedding)


class SwinUNETREncoder(_MedicalAdapter):
    """MONAI control adapter that only tokenizes genuine 3-D feature maps."""

    name = "swinunetr"
    modality_name = "ct"
    required_components = ("swinunetr",)

    def __init__(self, *, access: EncoderAccess, backend: EncoderBackend | None, feature_dim: int):
        super().__init__(access=access, feature_dim=feature_dim)
        self.backend = backend

    def encode(
        self,
        images: Tensor,
        *,
        geometry: CTGeometry,
        source_ids: tuple[str, ...],
        acquired_time: Tensor,
        available_time: Tensor,
        quality_flags: tuple[tuple[str, ...], ...] = (),
    ) -> EncoderOutput:
        self.access.validate_image_access(
            required_components=self.required_components,
            required_weights=("swinunetr",),
            required_dependencies=("monai",),
        )
        feature_map = _require_backend(self.backend, self.name)(images)
        if not isinstance(feature_map, Tensor) or feature_map.ndim != 5:
            raise DataContractError(
                code="INVALID_SPATIAL_FEATURE_MAP",
                message="SwinUNETR backend must return [B, C, D, H, W].",
            )
        batch, channels = feature_map.shape[:2]
        if (
            channels != self.feature_dim
            or batch != geometry.batch_size
            or len(source_ids) != batch
            or (quality_flags and len(quality_flags) != batch)
        ):
            raise DataContractError(
                code="INVALID_FEATURE_DIMENSION",
                message="SwinUNETR feature map does not match its batch/geometry contract.",
            )
        shape = (
            int(feature_map.shape[2]),
            int(feature_map.shape[3]),
            int(feature_map.shape[4]),
        )
        values = feature_map.permute(0, 2, 3, 4, 1).reshape(batch, -1, channels)
        coords = geometry.feature_grid_centers(shape, device=values.device, dtype=values.dtype)
        tokens = values.shape[1]
        observations = ObservationTokens(
            values=values,
            valid=torch.ones((batch, tokens), dtype=torch.bool, device=values.device),
            modality=torch.zeros((batch, tokens), dtype=torch.long, device=values.device),
            acquired_time=_times_for_tokens(
                acquired_time, tokens, "acquired_time", device=values.device
            ),
            available_time=_times_for_tokens(
                available_time, tokens, "available_time", device=values.device
            ),
            provenance=self.provenance,
            source_id=_source_matrix(source_ids, tokens),
            modality_name="ct",
            coords=coords,
            coordinate_system="patient_physical_mm",
            quality_flags=quality_flags,
        )
        return EncoderOutput(observations=observations)


class TITANCONCHEncoder(_MedicalAdapter):
    """TITAN slide adapter restricted to matching CONCH v1.5 patch features."""

    name = "titan_conch_v1_5"
    modality_name = "pathology"
    required_components = ("titan", "conch")

    def __init__(
        self,
        *,
        access: EncoderAccess,
        backend: EncoderBackend | None = None,
        feature_dim: int = 768,
    ) -> None:
        super().__init__(access=access, feature_dim=feature_dim)
        self.backend = backend

    def _validate_patches(self, patches: PathologyPatchBatch) -> None:
        conch_version = self.access.component_versions.get("conch", "").lower()
        if "1.5" not in conch_version and "v1_5" not in conch_version:
            raise ConfigurationError(
                code="INCOMPATIBLE_COMPONENT_VERSION",
                message="TITAN is restricted to the approved CONCH v1.5 feature protocol.",
            )
        if patches.patch_encoder_name.lower() not in {"conch_v1_5", "conch-v1.5"}:
            raise DataContractError(
                code="INCOMPATIBLE_PATCH_ENCODER",
                message="TITAN requires CONCH v1.5 patch features; UNI/other features are invalid.",
            )
        if patches.features.shape[-1] != 768:
            raise DataContractError(
                code="INVALID_FEATURE_DIMENSION",
                message="CONCH v1.5 patch features must have dimension 768.",
            )

    def encode(self, patches: PathologyPatchBatch) -> EncoderOutput:
        self.access.validate_image_access(
            required_components=self.required_components,
            required_weights=("titan", "conch"),
            required_dependencies=("torch",),
            requires_remote_code=True,
        )
        self._validate_patches(patches)
        backend = _require_backend(self.backend, self.name)
        slide = backend(
            patches.features,
            patches.geometry.coords_level0,
            patches.geometry.patch_size_level0,
            patches.geometry.valid,
        )
        if not isinstance(slide, Tensor) or slide.shape != (
            patches.geometry.batch_size,
            self.feature_dim,
        ):
            raise DataContractError(
                code="INVALID_TITAN_OUTPUT",
                message="TITAN slide output does not match [B, feature_dim].",
            )
        batch = slide.shape[0]
        observations = ObservationTokens(
            values=slide[:, None, :],
            valid=torch.ones((batch, 1), dtype=torch.bool, device=slide.device),
            modality=torch.ones((batch, 1), dtype=torch.long, device=slide.device),
            acquired_time=patches.acquired_time.to(slide.device)[:, None],
            available_time=patches.available_time.to(slide.device)[:, None],
            provenance=self.provenance,
            source_id=tuple((slide_id,) for slide_id in patches.slide_ids),
            modality_name="pathology",
            quality_flags=tuple((source,) for source in patches.tissue_sources),
        )
        return EncoderOutput(observations=observations, global_embedding=slide)


class UNI2HEncoder(_MedicalAdapter):
    """Independent UNI2-h patch-token branch; it is never routed through TITAN."""

    name = "uni2_h"
    modality_name = "pathology"
    required_components = ("uni2_h",)

    def __init__(self, *, access: EncoderAccess) -> None:
        super().__init__(access=access, feature_dim=1536)

    def encode(self, patches: PathologyPatchBatch) -> EncoderOutput:
        self.access.validate_feature_access(self.required_components)
        if patches.patch_encoder_name.lower() not in {"uni2_h", "uni2-h"}:
            raise DataContractError(
                code="INCOMPATIBLE_PATCH_ENCODER",
                message="UNI2-h branch requires UNI2-h patch features.",
            )
        if patches.features.shape[-1] != self.feature_dim:
            raise DataContractError(
                code="INVALID_FEATURE_DIMENSION",
                message="UNI2-h patch features must have dimension 1536.",
            )
        geometry = patches.geometry
        observations = ObservationTokens(
            values=patches.features,
            valid=geometry.valid.to(patches.features.device),
            modality=torch.ones(
                geometry.valid.shape, dtype=torch.long, device=patches.features.device
            ),
            acquired_time=_times_for_tokens(
                patches.acquired_time,
                patches.features.shape[1],
                "acquired_time",
                device=patches.features.device,
            ),
            available_time=_times_for_tokens(
                patches.available_time,
                patches.features.shape[1],
                "available_time",
                device=patches.features.device,
            ),
            provenance=self.provenance,
            source_id=_source_matrix(patches.slide_ids, patches.features.shape[1]),
            modality_name="pathology",
            coords=geometry.physical_centers_mm().to(
                device=patches.features.device, dtype=patches.features.dtype
            ),
            coordinate_system="wsi_level0_physical_mm",
            quality_flags=tuple((source,) for source in patches.tissue_sources),
        )
        return EncoderOutput(observations=observations)


class PRISM2Encoder(_MedicalAdapter):
    """Optional PRISM2 base branch with its published Virchow2 input protocol gated."""

    name = "prism2_base"
    modality_name = "pathology"
    required_components = ("prism2", "virchow2")

    def __init__(self, *, access: EncoderAccess, backend: EncoderBackend | None = None) -> None:
        super().__init__(access=access, feature_dim=2560)
        self.backend = backend

    def encode(self, patches: PathologyPatchBatch) -> EncoderOutput:
        self.access.validate_image_access(
            required_components=self.required_components,
            required_weights=("prism2", "virchow2"),
            required_dependencies=("torch",),
            requires_remote_code=True,
        )
        if patches.patch_encoder_name.lower() not in {
            "virchow2_cls",
            "virchow2_class_token",
        }:
            raise DataContractError(
                code="INCOMPATIBLE_PATCH_ENCODER",
                message="PRISM2 requires Virchow2 class-token-only patch features.",
            )
        if patches.features.shape[-1] != 1280:
            raise DataContractError(
                code="INVALID_FEATURE_DIMENSION",
                message="Virchow2 class-token input must have dimension 1280.",
            )
        if patches.input_patch_size_px != 224 or abs(patches.magnification - 20.0) > 1e-6:
            raise DataContractError(
                code="INVALID_PATCH_PROTOCOL",
                message="PRISM2 requires 224-pixel patches at 20x magnification.",
            )
        if not torch.allclose(
            patches.geometry.mpp,
            torch.full_like(patches.geometry.mpp, 0.5),
            rtol=0.0,
            atol=1e-6,
        ):
            raise DataContractError(
                code="INVALID_MPP", message="PRISM2 requires an explicit 0.5 MPP protocol."
            )
        slide = _require_backend(self.backend, self.name)(
            patches.features,
            patches.geometry.coords_level0,
            patches.geometry.valid,
        )
        if not isinstance(slide, Tensor) or slide.shape != (
            patches.geometry.batch_size,
            self.feature_dim,
        ):
            raise DataContractError(
                code="INVALID_PRISM2_OUTPUT",
                message="PRISM2 base output must have shape [B, 2560].",
            )
        batch = slide.shape[0]
        observations = ObservationTokens(
            values=slide[:, None, :],
            valid=torch.ones((batch, 1), dtype=torch.bool, device=slide.device),
            modality=torch.ones((batch, 1), dtype=torch.long, device=slide.device),
            acquired_time=patches.acquired_time.to(slide.device)[:, None],
            available_time=patches.available_time.to(slide.device)[:, None],
            provenance=self.provenance,
            source_id=tuple((slide_id,) for slide_id in patches.slide_ids),
            modality_name="pathology",
            quality_flags=tuple((source,) for source in patches.tissue_sources),
        )
        return EncoderOutput(observations=observations, global_embedding=slide)


# Compatibility aliases use the names found in configs and design documents.
MerlinAdapter = MerlinEncoder
SwinUNETRAdapter = SwinUNETREncoder
TITANAdapter = TITANCONCHEncoder
UNI2HAdapter = UNI2HEncoder
PRISM2Adapter = PRISM2Encoder
