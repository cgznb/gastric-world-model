"""Deterministic synthetic-only encoder used by contract tests and smoke runs."""

from __future__ import annotations

import torch
from torch import Tensor

from stageworld.config import RunMode
from stageworld.errors import ConfigurationError, DataContractError

from .base import EncoderOutput, EncoderProvenance, ObservationTokens


class SyntheticEncoder:
    name = "synthetic_deterministic"

    def __init__(self, *, mode: RunMode, input_dim: int, feature_dim: int) -> None:
        if mode is not RunMode.SYNTHETIC:
            raise ConfigurationError(
                code="SYNTHETIC_ENCODER_FORBIDDEN",
                message="The synthetic encoder is available only in mode=synthetic.",
            )
        if input_dim <= 0 or feature_dim <= 0:
            raise ConfigurationError(
                code="INVALID_ENCODER_DIMENSION",
                message="Synthetic encoder dimensions must be positive.",
            )
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self._provenance = EncoderProvenance(
            encoder_name=self.name,
            source_version="stageworld-synthetic-v1",
            component_versions=(("deterministic_projection", "analytic-v1"),),
            preprocess_version="identity-v1",
            feature_dim=feature_dim,
        )

    @property
    def provenance(self) -> EncoderProvenance:
        return self._provenance

    def encode(
        self,
        inputs: Tensor,
        *,
        valid: Tensor,
        modality: Tensor,
        acquired_time: Tensor,
        available_time: Tensor,
        source_id: tuple[tuple[str, ...], ...],
        modality_name: str,
        coords: Tensor | None = None,
        coordinate_system: str | None = None,
    ) -> EncoderOutput:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim:
            raise DataContractError(
                code="INVALID_SYNTHETIC_INPUT",
                message="Synthetic inputs must have shape [B, N, input_dim].",
                details={"actual": list(inputs.shape), "input_dim": self.input_dim},
            )
        work = inputs.to(dtype=torch.float32)
        in_axis = torch.arange(1, self.input_dim + 1, device=work.device, dtype=work.dtype)
        out_axis = torch.arange(1, self.feature_dim + 1, device=work.device, dtype=work.dtype)
        matrix = torch.sin(in_axis[:, None] * out_axis[None, :] * 0.173)
        values = torch.tanh(work @ matrix / self.input_dim**0.5)
        output_coords = None if coords is None else coords.to(device=values.device)
        observations = ObservationTokens(
            values=values,
            valid=valid.to(device=values.device),
            modality=modality.to(device=values.device),
            acquired_time=acquired_time.to(device=values.device),
            available_time=available_time.to(device=values.device),
            provenance=self.provenance,
            source_id=source_id,
            modality_name=modality_name,
            coords=output_coords,
            coordinate_system=coordinate_system,
        )
        denominator = valid.sum(dim=1, keepdim=True).clamp_min(1)
        global_embedding = (values * valid[..., None]).sum(dim=1) / denominator
        return EncoderOutput(observations=observations, global_embedding=global_embedding)
