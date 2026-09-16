"""Simple comparators for held-out future representation prediction.

These predictors operate only in a frozen CT or pathology target-feature space.
They are deliberately separate from :mod:`stageworld.baselines`, whose models
predict survival.  None of the classes in this module consumes outcomes or
produces hazards, survival curves, or clinical risks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .data.split import SplitName
from .errors import DataContractError


@dataclass(frozen=True)
class FutureComparatorMetadata:
    """Immutable scope and provenance needed for a fair feature comparison."""

    name: str
    target_space_id: str
    fit_split: SplitName | None
    condition_schema_id: str | None = None
    uses_source_offset: bool = False
    task: str = "future_observation_representation"

    def __post_init__(self) -> None:
        if not self.name or not self.target_space_id:
            raise ValueError("comparator name and target_space_id must be nonempty")
        if self.condition_schema_id is not None and not self.condition_schema_id:
            raise ValueError("condition_schema_id must be nonempty when supplied")
        if self.fit_split is not None and not isinstance(self.fit_split, SplitName):
            object.__setattr__(self, "fit_split", SplitName(self.fit_split))


@dataclass(frozen=True)
class FutureFeaturePrediction:
    """Predicted frozen-target tokens and the positions the comparator supports."""

    mean: Tensor
    valid: Tensor
    metadata: FutureComparatorMetadata

    def __post_init__(self) -> None:
        if self.mean.ndim != 3:
            raise ValueError("future feature predictions must have shape [B,K,D]")
        if self.valid.shape != self.mean.shape[:2] or self.valid.dtype is not torch.bool:
            raise ValueError("prediction valid must be boolean with shape [B,K]")
        if self.valid.device != self.mean.device:
            raise ValueError("prediction values and valid mask must share a device")
        if not torch.isfinite(self.mean).all():
            raise ValueError("future feature predictions must be finite")


def _validate_feature_batch(values: Tensor, valid: Tensor, *, name: str) -> None:
    if values.ndim != 3 or not torch.is_floating_point(values):
        raise ValueError(f"{name} must be a floating tensor with shape [B,K,D]")
    if values.shape[0] == 0 or values.shape[1] == 0 or values.shape[2] == 0:
        raise ValueError(f"{name} dimensions must be nonzero")
    if valid.shape != values.shape[:2] or valid.dtype is not torch.bool:
        raise ValueError(f"{name}_valid must be boolean with shape [B,K]")
    if valid.device != values.device:
        raise ValueError(f"{name} and {name}_valid must share a device")
    if not torch.isfinite(values[valid]).all():
        raise ValueError(f"valid {name} values must be finite")


def _safe_features(values: Tensor, valid: Tensor) -> Tensor:
    return torch.where(valid.unsqueeze(-1), values, torch.zeros_like(values))


def _require_training_split(split: SplitName) -> None:
    try:
        normalized = SplitName(split)
    except ValueError as error:
        raise DataContractError(
            code="INVALID_COMPARATOR_FIT_SPLIT",
            message="Future-observation comparators require a recognized patient split.",
        ) from error
    if normalized is not SplitName.TRAIN:
        raise DataContractError(
            code="COMPARATOR_FIT_SPLIT_LEAKAGE",
            message="Future-observation comparators may only be fitted on the training split.",
        )


def _require_frozen_target(target: Tensor) -> None:
    if target.requires_grad:
        raise DataContractError(
            code="COMPARATOR_TARGET_REQUIRES_GRAD",
            message="Comparator targets must come from a frozen, detached target space.",
        )


def _check_identifier(actual: str, expected: str, *, name: str) -> None:
    if actual != expected:
        raise DataContractError(
            code="COMPARATOR_FEATURE_SPACE_MISMATCH",
            message=f"{name} does not match the comparator's fitted feature schema.",
            details={"expected": expected, "received": actual},
        )


class PersistenceFuturePredictor(nn.Module):
    """Copy the latest same-space observation into the future target space."""

    def __init__(self, *, target_space_id: str) -> None:
        super().__init__()
        self.metadata = FutureComparatorMetadata(
            name="persistence",
            target_space_id=target_space_id,
            fit_split=None,
        )

    def forward(
        self,
        source: Tensor,
        source_valid: Tensor,
        *,
        source_space_id: str,
    ) -> FutureFeaturePrediction:
        _check_identifier(source_space_id, self.metadata.target_space_id, name="source_space_id")
        _validate_feature_batch(source, source_valid, name="source")
        return FutureFeaturePrediction(
            mean=_safe_features(source, source_valid),
            valid=source_valid.clone(),
            metadata=self.metadata,
        )


class TrainingMeanFuturePredictor(nn.Module):
    """Predict the token-wise mean estimated from training targets only."""

    mean: Tensor
    support_counts: Tensor

    def __init__(
        self,
        mean: Tensor,
        support_counts: Tensor,
        *,
        target_space_id: str,
    ) -> None:
        super().__init__()
        if mean.ndim != 2 or not torch.is_floating_point(mean):
            raise ValueError("mean must be a floating tensor with shape [K,D]")
        if support_counts.shape != mean.shape[:1] or support_counts.dtype != torch.long:
            raise ValueError("support_counts must be int64 with shape [K]")
        if torch.any(support_counts < 0) or not torch.isfinite(mean).all():
            raise ValueError("mean and support_counts contain invalid values")
        if not torch.any(support_counts > 0):
            raise ValueError("at least one target token needs training support")
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("support_counts", support_counts.detach().clone())
        self.metadata = FutureComparatorMetadata(
            name="training_mean",
            target_space_id=target_space_id,
            fit_split=SplitName.TRAIN,
        )

    @classmethod
    def fit(
        cls,
        target: Tensor,
        target_valid: Tensor,
        *,
        target_space_id: str,
        split: SplitName,
    ) -> TrainingMeanFuturePredictor:
        _require_training_split(split)
        _require_frozen_target(target)
        _validate_feature_batch(target, target_valid, name="target")
        safe = _safe_features(target, target_valid)
        counts = target_valid.sum(dim=0)
        mean = safe.sum(dim=0) / counts.clamp_min(1).unsqueeze(-1)
        mean = torch.where(counts.unsqueeze(-1) > 0, mean, torch.zeros_like(mean))
        return cls(mean, counts.to(torch.long), target_space_id=target_space_id)

    def forward(
        self,
        batch_size: int,
        *,
        target_space_id: str,
    ) -> FutureFeaturePrediction:
        _check_identifier(target_space_id, self.metadata.target_space_id, name="target_space_id")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        valid = (self.support_counts > 0).unsqueeze(0).expand(batch_size, -1)
        mean = self.mean.unsqueeze(0).expand(batch_size, -1, -1)
        return FutureFeaturePrediction(mean=mean, valid=valid, metadata=self.metadata)


def _validate_conditions(
    elapsed_time: Tensor,
    treatment: Tensor,
    treatment_valid: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    if elapsed_time.ndim != 1 or not torch.is_floating_point(elapsed_time):
        raise ValueError("elapsed_time must be a floating tensor with shape [B]")
    if treatment.ndim != 2 or not torch.is_floating_point(treatment):
        raise ValueError("treatment must be a floating tensor with shape [B,C]")
    if treatment.shape[0] != elapsed_time.shape[0] or treatment.shape[1] == 0:
        raise ValueError("elapsed_time and nonempty treatment features must share batch size")
    if treatment.device != elapsed_time.device:
        raise ValueError("elapsed_time and treatment must share a device")
    if treatment_valid is None:
        mask = torch.ones_like(treatment, dtype=torch.bool)
    else:
        mask = treatment_valid
        if mask.shape != treatment.shape or mask.dtype is not torch.bool:
            raise ValueError("treatment_valid must be boolean with shape [B,C]")
        if mask.device != treatment.device:
            raise ValueError("treatment and treatment_valid must share a device")
    if not torch.isfinite(elapsed_time).all() or torch.any(elapsed_time < 0):
        raise ValueError("elapsed_time must be finite and nonnegative")
    if not torch.isfinite(treatment[mask]).all():
        raise ValueError("valid treatment values must be finite")
    return elapsed_time, _safe_treatment(treatment, mask), mask


def _safe_treatment(treatment: Tensor, valid: Tensor) -> Tensor:
    return torch.where(valid, treatment, torch.zeros_like(treatment))


def _training_condition_statistics(
    elapsed_time: Tensor,
    treatment: Tensor,
    treatment_valid: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    time_mean = elapsed_time.mean()
    time_scale = elapsed_time.std(unbiased=False).clamp_min(1e-6)
    counts = treatment_valid.sum(dim=0)
    treatment_mean = treatment.sum(dim=0) / counts.clamp_min(1)
    centered = torch.where(
        treatment_valid,
        treatment - treatment_mean,
        torch.zeros_like(treatment),
    )
    variance = centered.square().sum(dim=0) / counts.clamp_min(1)
    treatment_scale = variance.sqrt().clamp_min(1e-6)
    treatment_mean = torch.where(counts > 0, treatment_mean, torch.zeros_like(treatment_mean))
    treatment_scale = torch.where(counts > 0, treatment_scale, torch.ones_like(treatment_scale))
    return time_mean, time_scale, treatment_mean, treatment_scale


def _condition_design(
    elapsed_time: Tensor,
    treatment: Tensor,
    treatment_valid: Tensor,
    *,
    time_mean: Tensor,
    time_scale: Tensor,
    treatment_mean: Tensor,
    treatment_scale: Tensor,
) -> Tensor:
    standardized_time = (elapsed_time - time_mean) / time_scale
    standardized_treatment = torch.where(
        treatment_valid,
        (treatment - treatment_mean) / treatment_scale,
        torch.zeros_like(treatment),
    )
    return torch.cat(
        (
            torch.ones_like(standardized_time).unsqueeze(-1),
            standardized_time.unsqueeze(-1),
            standardized_treatment,
            treatment_valid.to(treatment.dtype),
        ),
        dim=-1,
    )


class TimeTreatmentFuturePredictor(nn.Module):
    """Training-only ridge model of feature change from time and treatment.

    When ``uses_source_offset`` is true, the regression target is future minus
    current same-space representation.  Without an offset it directly predicts
    a target such as pathology for which no prior same-modality observation
    exists.
    """

    coefficients: Tensor
    support_counts: Tensor
    time_mean: Tensor
    time_scale: Tensor
    treatment_mean: Tensor
    treatment_scale: Tensor

    def __init__(
        self,
        coefficients: Tensor,
        support_counts: Tensor,
        time_mean: Tensor,
        time_scale: Tensor,
        treatment_mean: Tensor,
        treatment_scale: Tensor,
        *,
        target_space_id: str,
        condition_schema_id: str,
        uses_source_offset: bool,
        ridge: float,
    ) -> None:
        super().__init__()
        if coefficients.ndim != 3 or not torch.is_floating_point(coefficients):
            raise ValueError("coefficients must be a floating tensor with shape [P,K,D]")
        if support_counts.shape != coefficients.shape[1:2] or support_counts.dtype != torch.long:
            raise ValueError("support_counts must be int64 with shape [K]")
        if treatment_mean.ndim != 1 or treatment_scale.shape != treatment_mean.shape:
            raise ValueError("treatment statistics must have shape [C]")
        expected_predictors = 2 + 2 * treatment_mean.numel()
        if coefficients.shape[0] != expected_predictors:
            raise ValueError("coefficient count does not match time/treatment design")
        if ridge <= 0 or not torch.isfinite(torch.as_tensor(ridge)):
            raise ValueError("ridge must be finite and positive")
        if not torch.any(support_counts > 0):
            raise ValueError("at least one target token needs training support")
        for name, tensor in (
            ("coefficients", coefficients),
            ("time_mean", time_mean),
            ("time_scale", time_scale),
            ("treatment_mean", treatment_mean),
            ("treatment_scale", treatment_scale),
        ):
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must be finite")
        if time_scale.numel() != 1 or torch.any(time_scale <= 0):
            raise ValueError("time_scale must be a positive scalar")
        if torch.any(treatment_scale <= 0):
            raise ValueError("treatment_scale must be positive")
        self.register_buffer("coefficients", coefficients.detach().clone())
        self.register_buffer("support_counts", support_counts.detach().clone())
        self.register_buffer("time_mean", time_mean.detach().reshape(()).clone())
        self.register_buffer("time_scale", time_scale.detach().reshape(()).clone())
        self.register_buffer("treatment_mean", treatment_mean.detach().clone())
        self.register_buffer("treatment_scale", treatment_scale.detach().clone())
        self.ridge = float(ridge)
        self.metadata = FutureComparatorMetadata(
            name="time_treatment_ridge",
            target_space_id=target_space_id,
            fit_split=SplitName.TRAIN,
            condition_schema_id=condition_schema_id,
            uses_source_offset=uses_source_offset,
        )

    @classmethod
    def fit(
        cls,
        target: Tensor,
        target_valid: Tensor,
        elapsed_time: Tensor,
        treatment: Tensor,
        *,
        target_space_id: str,
        condition_schema_id: str,
        split: SplitName,
        treatment_valid: Tensor | None = None,
        source: Tensor | None = None,
        source_valid: Tensor | None = None,
        source_space_id: str | None = None,
        ridge: float = 1e-3,
    ) -> TimeTreatmentFuturePredictor:
        _require_training_split(split)
        _require_frozen_target(target)
        _validate_feature_batch(target, target_valid, name="target")
        elapsed_time, treatment, treatment_mask = _validate_conditions(
            elapsed_time, treatment, treatment_valid
        )
        if elapsed_time.shape[0] != target.shape[0]:
            raise ValueError("conditions and target must share batch size")
        if ridge <= 0:
            raise ValueError("ridge must be positive")

        uses_source_offset = source is not None
        fit_valid = target_valid
        response = _safe_features(target, target_valid)
        if source is not None:
            if source_valid is None or source_space_id is None:
                raise ValueError("source_valid and source_space_id are required with source")
            _check_identifier(source_space_id, target_space_id, name="source_space_id")
            _validate_feature_batch(source, source_valid, name="source")
            if source.shape != target.shape:
                raise ValueError("source offset and target must have the same [B,K,D] shape")
            fit_valid = target_valid & source_valid
            response = _safe_features(target - source, fit_valid)
        elif source_valid is not None or source_space_id is not None:
            raise ValueError("source metadata cannot be supplied without source")

        statistics = _training_condition_statistics(elapsed_time, treatment, treatment_mask)
        design = _condition_design(
            elapsed_time,
            treatment,
            treatment_mask,
            time_mean=statistics[0],
            time_scale=statistics[1],
            treatment_mean=statistics[2],
            treatment_scale=statistics[3],
        )
        design64 = design.detach().to(torch.float64)
        response64 = response.detach().to(torch.float64)
        coefficients = torch.zeros(
            design.shape[1],
            target.shape[1],
            target.shape[2],
            dtype=torch.float64,
            device=target.device,
        )
        support_counts = fit_valid.sum(dim=0).to(torch.long)
        penalty = torch.eye(design.shape[1], dtype=torch.float64, device=target.device) * ridge
        penalty[0, 0] = 0.0
        for token_index in range(target.shape[1]):
            selected = fit_valid[:, token_index]
            if not torch.any(selected):
                continue
            token_design = design64[selected]
            token_response = response64[selected, token_index]
            gram = token_design.T @ token_design + penalty
            coefficients[:, token_index] = torch.linalg.solve(gram, token_design.T @ token_response)
        coefficients = coefficients.to(dtype=target.dtype)
        return cls(
            coefficients,
            support_counts,
            *statistics,
            target_space_id=target_space_id,
            condition_schema_id=condition_schema_id,
            uses_source_offset=uses_source_offset,
            ridge=ridge,
        )

    def forward(
        self,
        elapsed_time: Tensor,
        treatment: Tensor,
        *,
        target_space_id: str,
        condition_schema_id: str,
        treatment_valid: Tensor | None = None,
        source: Tensor | None = None,
        source_valid: Tensor | None = None,
        source_space_id: str | None = None,
    ) -> FutureFeaturePrediction:
        _check_identifier(target_space_id, self.metadata.target_space_id, name="target_space_id")
        assert self.metadata.condition_schema_id is not None
        _check_identifier(
            condition_schema_id,
            self.metadata.condition_schema_id,
            name="condition_schema_id",
        )
        elapsed_time, treatment, treatment_mask = _validate_conditions(
            elapsed_time, treatment, treatment_valid
        )
        if treatment.shape[1] != self.treatment_mean.numel():
            raise ValueError("treatment feature count differs from fitted schema")
        if elapsed_time.device != self.coefficients.device:
            raise ValueError("conditions and fitted comparator must share a device")
        design = _condition_design(
            elapsed_time,
            treatment,
            treatment_mask,
            time_mean=self.time_mean,
            time_scale=self.time_scale,
            treatment_mean=self.treatment_mean,
            treatment_scale=self.treatment_scale,
        ).to(self.coefficients.dtype)
        mean = torch.einsum("bp,pkd->bkd", design, self.coefficients)
        valid = (self.support_counts > 0).unsqueeze(0).expand(elapsed_time.shape[0], -1)

        if self.metadata.uses_source_offset:
            if source is None or source_valid is None or source_space_id is None:
                raise ValueError("this comparator requires a same-space source offset")
            _check_identifier(source_space_id, target_space_id, name="source_space_id")
            _validate_feature_batch(source, source_valid, name="source")
            if source.shape != mean.shape:
                raise ValueError("source offset shape differs from fitted target shape")
            if source.dtype != mean.dtype:
                raise ValueError("source offset dtype differs from fitted comparator")
            valid = valid & source_valid
            mean = mean + _safe_features(source, source_valid)
        elif source is not None or source_valid is not None or source_space_id is not None:
            raise ValueError("direct conditional comparator does not accept a source offset")

        mean = torch.where(valid.unsqueeze(-1), mean, torch.zeros_like(mean))
        return FutureFeaturePrediction(mean=mean, valid=valid, metadata=self.metadata)


__all__ = [
    "FutureComparatorMetadata",
    "FutureFeaturePrediction",
    "PersistenceFuturePredictor",
    "TimeTreatmentFuturePredictor",
    "TrainingMeanFuturePredictor",
]
