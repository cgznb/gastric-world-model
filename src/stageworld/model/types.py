"""Tensor contracts for latent transition, observation update, and prediction."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import torch
from torch import Tensor

from stageworld.errors import DataContractError


def _expect_shape(name: str, tensor: Tensor, dims: int) -> None:
    if tensor.ndim != dims:
        raise DataContractError(
            code="INVALID_TENSOR_RANK",
            message=f"{name} must have rank {dims}; received {tensor.ndim}.",
        )


@dataclass
class ActionTokens:
    """Actions known to the caller for a specified transition target."""

    values: Tensor
    valid: Tensor
    event_time: Tensor
    available_time: Tensor
    event_type: Tensor | None = None
    planned_or_delivered: Tensor | None = None
    known_exposure: Tensor | None = None
    provenance: tuple[str, ...] = ()

    def validate(self) -> None:
        _expect_shape("ActionTokens.values", self.values, 3)
        batch, count, _ = self.values.shape
        for name, required_tensor in (
            ("valid", self.valid),
            ("event_time", self.event_time),
            ("available_time", self.available_time),
        ):
            if required_tensor.shape != (batch, count):
                raise DataContractError(
                    code="ACTION_SHAPE_MISMATCH",
                    message=f"ActionTokens.{name} must have shape [B,A].",
                )
            if required_tensor.device != self.values.device:
                raise DataContractError(
                    code="ACTION_DEVICE_MISMATCH",
                    message=f"ActionTokens.{name} must share the values device.",
                )
        if self.valid.dtype is not torch.bool:
            raise DataContractError(
                code="ACTION_MASK_DTYPE",
                message="ActionTokens.valid must be boolean.",
            )
        for name, metadata_tensor in (
            ("event_type", self.event_type),
            ("planned_or_delivered", self.planned_or_delivered),
            ("known_exposure", self.known_exposure),
        ):
            if metadata_tensor is not None and metadata_tensor.shape != (batch, count):
                raise DataContractError(
                    code="ACTION_METADATA_SHAPE",
                    message=f"ActionTokens.{name} must have shape [B,A].",
                )
            if metadata_tensor is not None and metadata_tensor.device != self.values.device:
                raise DataContractError(
                    code="ACTION_DEVICE_MISMATCH",
                    message=f"ActionTokens.{name} must share the values device.",
                )
        for name, integer_tensor in (
            ("event_type", self.event_type),
            ("planned_or_delivered", self.planned_or_delivered),
        ):
            if integer_tensor is not None and integer_tensor.dtype not in (
                torch.int32,
                torch.int64,
            ):
                raise DataContractError(
                    code="ACTION_METADATA_DTYPE",
                    message=f"ActionTokens.{name} must have an integer dtype.",
                )
        for name, bounded_tensor, upper in (
            ("event_type", self.event_type, 15),
            ("planned_or_delivered", self.planned_or_delivered, 3),
        ):
            if bounded_tensor is not None and (
                (bounded_tensor[self.valid] < 0).any()
                or (bounded_tensor[self.valid] > upper).any()
            ):
                raise DataContractError(
                    code="ACTION_METADATA_RANGE",
                    message=f"Valid ActionTokens.{name} values must be in [0, {upper}].",
                )
        if not torch.isfinite(self.values[self.valid]).all():
            raise DataContractError(
                code="NONFINITE_ACTION",
                message="Valid action values must be finite.",
            )
        for name, finite_tensor in (
            ("event_time", self.event_time),
            ("available_time", self.available_time),
            ("known_exposure", self.known_exposure),
        ):
            if finite_tensor is not None and not torch.isfinite(finite_tensor[self.valid]).all():
                raise DataContractError(
                    code="NONFINITE_ACTION_METADATA",
                    message=f"Valid ActionTokens.{name} values must be finite.",
                )

    @classmethod
    def empty(cls, *, batch_size: int, value_dim: int, device: torch.device) -> ActionTokens:
        return cls(
            values=torch.zeros(batch_size, 1, value_dim, device=device),
            valid=torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
            event_time=torch.zeros(batch_size, 1, device=device),
            available_time=torch.zeros(batch_size, 1, device=device),
        )


@dataclass
class BeliefState:
    """Deterministic token memory plus optional diagonal-Gaussian token state."""

    memory: Tensor
    query_time: Tensor
    stochastic_mean: Tensor | None = None
    stochastic_log_std: Tensor | None = None
    sample: Tensor | None = None
    state_kind: str = "posterior"
    provenance: tuple[str, ...] = ()
    quality_flags: tuple[str, ...] = ()

    def validate(self) -> None:
        _expect_shape("BeliefState.memory", self.memory, 3)
        _expect_shape("BeliefState.query_time", self.query_time, 1)
        if self.memory.shape[0] != self.query_time.shape[0]:
            raise DataContractError(
                code="STATE_BATCH_MISMATCH",
                message="State memory and query_time batch sizes differ.",
            )
        if self.query_time.device != self.memory.device:
            raise DataContractError(
                code="STATE_DEVICE_MISMATCH",
                message="State memory and query_time must share a device.",
            )
        if not torch.isfinite(self.memory).all() or not torch.isfinite(self.query_time).all():
            raise DataContractError(
                code="NONFINITE_STATE",
                message="State memory and query_time must be finite.",
            )
        if self.state_kind not in {"prior", "posterior", "mixed"}:
            raise DataContractError(
                code="INVALID_STATE_KIND",
                message="state_kind must be prior, posterior, or mixed.",
            )
        stochastic = (self.stochastic_mean, self.stochastic_log_std, self.sample)
        if any(item is None for item in stochastic) and not all(
            item is None for item in stochastic
        ):
            raise DataContractError(
                code="PARTIAL_STOCHASTIC_STATE",
                message="Stochastic state tensors must be all present or all absent.",
            )
        if self.stochastic_mean is not None:
            assert self.stochastic_log_std is not None and self.sample is not None
            if not (
                self.stochastic_mean.shape == self.stochastic_log_std.shape == self.sample.shape
            ):
                raise DataContractError(
                    code="STOCHASTIC_SHAPE_MISMATCH",
                    message="Stochastic mean, log_std, and sample shapes differ.",
                )
            if self.stochastic_mean.shape[:2] != self.memory.shape[:2]:
                raise DataContractError(
                    code="STOCHASTIC_MEMORY_MISMATCH",
                    message="Stochastic state must align with memory tokens.",
                )
            if any(item.device != self.memory.device for item in stochastic if item is not None):
                raise DataContractError(
                    code="STATE_DEVICE_MISMATCH",
                    message="Stochastic state tensors must share the memory device.",
                )
            if not all(torch.isfinite(item).all() for item in stochastic if item is not None):
                raise DataContractError(
                    code="NONFINITE_STATE",
                    message="Stochastic state tensors must be finite.",
                )

    def with_flag(self, flag: str) -> BeliefState:
        return replace(self, quality_flags=tuple(dict.fromkeys((*self.quality_flags, flag))))


@dataclass
class PredictionDistribution:
    modality: str
    mean: Tensor
    log_std: Tensor
    target_time: Tensor
    provenance: str
    scenario: str

    @property
    def std(self) -> Tensor:
        return self.log_std.exp()


@dataclass
class StagePrediction:
    stage: str
    query_time: Tensor
    endpoint: str
    horizon_grid: Tensor
    rates: Tensor
    survival: Tensor
    risk: Tensor
    cif: Tensor | None = None
    input_manifest: tuple[str, ...] = ()
    model_version: str = "unversioned"
    quality_flags: tuple[str, ...] = ()
    uncertainty_summary: dict[str, Any] = field(default_factory=dict)
    simulated: bool = False
