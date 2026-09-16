"""Exact piecewise-exponential survival mathematics.

Rates in this module are hazards per configured physical time unit.  This is
different from pycox ``PCHazard`` where ``softplus(phi)`` represents the
cumulative hazard of a complete interval.  Convert with ``rate * width``
before comparing the two parameterizations.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

Reduction = Literal["none", "mean", "sum"]
ZeroTimePolicy = Literal["error", "allow"]
TimeUnit = Literal["day", "year"]


def _as_float_tensor(value: Tensor | Sequence[float], *, like: Tensor) -> Tensor:
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def _validated_cuts(cuts: Tensor | Sequence[float], *, like: Tensor, intervals: int) -> Tensor:
    if intervals < 1:
        raise ValueError("at least one hazard interval is required")
    result = _as_float_tensor(cuts, like=like)
    if result.ndim != 1 or result.numel() != intervals + 1:
        raise ValueError(
            f"cuts must contain one more boundary than rates: expected {intervals + 1}"
        )
    if not torch.isfinite(result).all():
        raise ValueError("cuts must be finite")
    if not torch.isclose(result[0], result.new_tensor(0.0)):
        raise ValueError("cuts must start at zero")
    if not torch.all(result[1:] > result[:-1]):
        raise ValueError("cuts must be strictly increasing")
    return result


def _validated_rates(rates: Tensor, *, allow_causes: bool) -> tuple[Tensor, int]:
    if rates.ndim == 2:
        normalized = rates
        causes = 1
    elif allow_causes and rates.ndim == 3:
        normalized = rates
        causes = rates.shape[-1]
    elif not allow_causes and rates.ndim == 3 and rates.shape[-1] == 1:
        normalized = rates.squeeze(-1)
        causes = 1
    else:
        expected = (
            "[batch, intervals] or [batch, intervals, causes]"
            if allow_causes
            else "[batch, intervals]"
        )
        raise ValueError(f"rates must have shape {expected}")
    if normalized.shape[1] < 1 or causes < 1:
        raise ValueError("at least one interval and cause are required")
    if not normalized.is_floating_point():
        raise TypeError("rates must be floating point")
    if not torch.isfinite(normalized).all():
        raise ValueError("rates must be finite")
    if torch.any(normalized < 0):
        raise ValueError("hazard rates cannot be negative")
    return normalized, causes


def _valid_mask(mask: Tensor | None, reference: Tensor) -> Tensor:
    if mask is None:
        return torch.ones_like(reference, dtype=torch.bool)
    result = torch.as_tensor(mask, device=reference.device, dtype=torch.bool)
    if result.shape != reference.shape:
        raise ValueError("valid mask must have the same shape as durations")
    return result


def _validate_durations(
    durations: Tensor,
    valid: Tensor,
    *,
    cuts: Tensor,
    open_tail: bool,
    zero_time_policy: ZeroTimePolicy,
) -> None:
    selected = durations[valid]
    if selected.numel() == 0:
        raise ValueError("at least one valid survival label is required")
    if not torch.isfinite(selected).all():
        raise ValueError("valid durations must be finite")
    if torch.any(selected < 0):
        raise ValueError("negative remaining time is invalid and must not be clipped")
    if zero_time_policy == "error" and torch.any(selected == 0):
        raise ValueError("zero remaining time requires zero_time_policy='allow'")
    if zero_time_policy not in ("error", "allow"):
        raise ValueError(f"unknown zero-time policy: {zero_time_policy}")
    if not open_tail and torch.any(selected > cuts[-1]):
        raise ValueError("duration exceeds the closed administrative horizon")


def interval_exposure(
    times: Tensor,
    cuts: Tensor | Sequence[float],
    *,
    open_tail: bool = True,
) -> Tensor:
    """Return exact time spent in each piecewise-constant interval.

    ``cuts`` contains both zero and the finite interval ends.  Thus J rates
    require J+1 cuts.  Intervals are left-closed/right-open.  With an open
    tail, the last rate continues beyond the final cut; the last finite width
    remains useful for pycox conversion and configured reporting horizons.
    """

    if not times.is_floating_point():
        times = times.to(torch.get_default_dtype())
    boundaries = _validated_cuts(cuts, like=times, intervals=len(cuts) - 1)
    if not torch.isfinite(times).all() or torch.any(times < 0):
        raise ValueError("times must be finite and nonnegative")
    if not open_tail and torch.any(times > boundaries[-1]):
        raise ValueError("time exceeds the closed administrative horizon")

    starts = boundaries[:-1]
    widths = boundaries[1:] - starts
    elapsed = (times.unsqueeze(-1) - starts).clamp_min(0)
    finite = torch.minimum(elapsed, widths)
    if open_tail:
        finite = torch.cat((finite[..., :-1], elapsed[..., -1:]), dim=-1)
    return finite


def interval_index(
    times: Tensor,
    cuts: Tensor | Sequence[float],
    *,
    open_tail: bool = True,
) -> Tensor:
    """Map times to intervals; an internal boundary belongs to the next interval."""

    if not times.is_floating_point():
        times = times.to(torch.get_default_dtype())
    boundaries = _validated_cuts(cuts, like=times, intervals=len(cuts) - 1)
    if not torch.isfinite(times).all() or torch.any(times < 0):
        raise ValueError("times must be finite and nonnegative")
    if not open_tail and torch.any(times > boundaries[-1]):
        raise ValueError("time exceeds the closed administrative horizon")
    indices = torch.bucketize(times.contiguous(), boundaries[1:], right=True)
    return indices.clamp_max(boundaries.numel() - 2)


def cumulative_hazard(
    rates: Tensor,
    times: Tensor,
    cuts: Tensor | Sequence[float],
    *,
    open_tail: bool = True,
) -> Tensor:
    """Cumulative all-cause hazard for one time per batch row."""

    normalized, _ = _validated_rates(rates, allow_causes=True)
    total = normalized if normalized.ndim == 2 else normalized.sum(dim=-1)
    if times.ndim != 1 or times.shape[0] != total.shape[0]:
        raise ValueError("times must have shape [batch]")
    boundaries = _validated_cuts(cuts, like=total, intervals=total.shape[1])
    exposure = interval_exposure(times.to(total), boundaries, open_tail=open_tail)
    return (total * exposure).sum(dim=-1)


def survival_probability(
    rates: Tensor,
    horizons: Tensor,
    cuts: Tensor | Sequence[float],
    *,
    open_tail: bool = True,
) -> Tensor:
    """Return S(t) with shape ``[batch, horizons]``."""

    normalized, _ = _validated_rates(rates, allow_causes=True)
    total = normalized if normalized.ndim == 2 else normalized.sum(dim=-1)
    if horizons.ndim != 1:
        raise ValueError("horizons must be one-dimensional")
    boundaries = _validated_cuts(cuts, like=total, intervals=total.shape[1])
    exposure = interval_exposure(horizons.to(total), boundaries, open_tail=open_tail)
    integrated = torch.einsum("bj,hj->bh", total, exposure)
    return torch.exp(-integrated)


def risk_probability(
    rates: Tensor,
    horizons: Tensor,
    cuts: Tensor | Sequence[float],
    *,
    open_tail: bool = True,
) -> Tensor:
    """Return all-cause risk ``1 - S(t)`` stably."""

    normalized, _ = _validated_rates(rates, allow_causes=True)
    total = normalized if normalized.ndim == 2 else normalized.sum(dim=-1)
    if horizons.ndim != 1:
        raise ValueError("horizons must be one-dimensional")
    boundaries = _validated_cuts(cuts, like=total, intervals=total.shape[1])
    exposure = interval_exposure(horizons.to(total), boundaries, open_tail=open_tail)
    integrated = torch.einsum("bj,hj->bh", total, exposure)
    return -torch.expm1(-integrated)


def masked_patient_mean(
    losses: Tensor,
    valid_mask: Tensor | None = None,
    *,
    patient_ids: Sequence[Hashable] | Tensor | None = None,
) -> Tensor:
    """Average labels within patient, then average patients.

    A ``[patients, labels]`` tensor handles multiple prefixes/slides without
    giving patients with more labels extra weight.  Flat duplicated rows can
    provide ``patient_ids`` to obtain the same behavior.
    """

    if not losses.is_floating_point():
        raise TypeError("losses must be floating point")
    mask = (
        torch.ones_like(losses, dtype=torch.bool)
        if valid_mask is None
        else torch.as_tensor(valid_mask, device=losses.device, dtype=torch.bool)
    )
    if mask.shape != losses.shape:
        raise ValueError("valid_mask must have the same shape as losses")

    if patient_ids is not None:
        if losses.ndim != 1:
            raise ValueError("patient_ids are only accepted for flat losses")
        ids = patient_ids.tolist() if isinstance(patient_ids, Tensor) else list(patient_ids)
        if len(ids) != losses.numel():
            raise ValueError("patient_ids length must match flat losses")
        groups: dict[Hashable, list[Tensor]] = {}
        for index, patient_id in enumerate(ids):
            if bool(mask[index]):
                groups.setdefault(patient_id, []).append(losses[index])
        if not groups:
            raise ValueError("at least one valid patient is required")
        patient_losses = [torch.stack(values).mean() for values in groups.values()]
        return torch.stack(patient_losses).mean()

    if losses.ndim == 0:
        if not bool(mask):
            raise ValueError("at least one valid patient is required")
        return losses
    if losses.ndim == 1:
        if not mask.any():
            raise ValueError("at least one valid patient is required")
        return losses[mask].mean()

    flat_loss = losses.reshape(losses.shape[0], -1)
    flat_mask = mask.reshape(mask.shape[0], -1)
    counts = flat_mask.sum(dim=1)
    has_label = counts > 0
    if not has_label.any():
        raise ValueError("at least one valid patient is required")
    safe = torch.where(flat_mask, flat_loss, torch.zeros_like(flat_loss))
    per_patient = safe.sum(dim=1) / counts.clamp_min(1)
    return per_patient[has_label].mean()


def _reduce_losses(
    losses: Tensor,
    mask: Tensor,
    reduction: Reduction,
    patient_ids: Sequence[Hashable] | Tensor | None,
) -> Tensor:
    if reduction == "none":
        return torch.where(mask, losses, torch.zeros_like(losses))
    if reduction == "sum":
        return torch.where(mask, losses, torch.zeros_like(losses)).sum()
    if reduction == "mean":
        return masked_patient_mean(losses, mask, patient_ids=patient_ids)
    raise ValueError(f"unknown reduction: {reduction}")


def piecewise_exponential_nll(
    rates: Tensor,
    durations: Tensor,
    events: Tensor,
    cuts: Tensor | Sequence[float],
    *,
    valid_mask: Tensor | None = None,
    patient_ids: Sequence[Hashable] | Tensor | None = None,
    open_tail: bool = True,
    zero_time_policy: ZeroTimePolicy = "error",
    reduction: Reduction = "mean",
    log_epsilon: float = 1e-12,
) -> Tensor:
    """Exact event/censor NLL for physical-time piecewise hazard rates."""

    normalized, _ = _validated_rates(rates, allow_causes=False)
    if durations.ndim != 1 or durations.shape[0] != normalized.shape[0]:
        raise ValueError("durations must have shape [batch]")
    if events.shape != durations.shape:
        raise ValueError("events must have the same shape as durations")
    valid = _valid_mask(valid_mask, durations)
    boundaries = _validated_cuts(cuts, like=normalized, intervals=normalized.shape[1])
    durations = durations.to(normalized)
    event_values = events.to(normalized)
    if not torch.all((event_values[valid] == 0) | (event_values[valid] == 1)):
        raise ValueError("events must be binary on valid labels")
    _validate_durations(
        durations,
        valid,
        cuts=boundaries,
        open_tail=open_tail,
        zero_time_policy=zero_time_policy,
    )

    safe_duration = torch.where(valid, durations, torch.zeros_like(durations))
    safe_events = torch.where(valid, event_values, torch.zeros_like(event_values))
    exposure = interval_exposure(safe_duration, boundaries, open_tail=open_tail)
    cumulative = (normalized * exposure).sum(dim=-1)
    indices = interval_index(safe_duration, boundaries, open_tail=open_tail)
    event_rates = normalized.gather(1, indices.unsqueeze(1)).squeeze(1)
    losses = cumulative - safe_events * torch.log(event_rates.clamp_min(log_epsilon))
    return _reduce_losses(losses, valid, reduction, patient_ids)


def cause_specific_nll(
    rates: Tensor,
    durations: Tensor,
    event_types: Tensor,
    cuts: Tensor | Sequence[float],
    *,
    valid_mask: Tensor | None = None,
    patient_ids: Sequence[Hashable] | Tensor | None = None,
    open_tail: bool = True,
    zero_time_policy: ZeroTimePolicy = "error",
    reduction: Reduction = "mean",
    log_epsilon: float = 1e-12,
) -> Tensor:
    """Cause-specific NLL; event type 0 is censoring and 1..C selects a cause."""

    normalized, causes = _validated_rates(rates, allow_causes=True)
    if normalized.ndim != 3:
        raise ValueError("cause-specific rates must have shape [batch, intervals, causes]")
    if durations.ndim != 1 or durations.shape[0] != normalized.shape[0]:
        raise ValueError("durations must have shape [batch]")
    if event_types.shape != durations.shape:
        raise ValueError("event_types must have the same shape as durations")
    valid = _valid_mask(valid_mask, durations)
    boundaries = _validated_cuts(cuts, like=normalized, intervals=normalized.shape[1])
    durations = durations.to(normalized)
    types = event_types.to(device=normalized.device, dtype=torch.long)
    if torch.any(types[valid] < 0) or torch.any(types[valid] > causes):
        raise ValueError(f"event_types must be in [0, {causes}]")
    _validate_durations(
        durations,
        valid,
        cuts=boundaries,
        open_tail=open_tail,
        zero_time_policy=zero_time_policy,
    )

    safe_duration = torch.where(valid, durations, torch.zeros_like(durations))
    safe_types = torch.where(valid, types, torch.zeros_like(types))
    exposure = interval_exposure(safe_duration, boundaries, open_tail=open_tail)
    cumulative = (normalized.sum(dim=-1) * exposure).sum(dim=-1)
    indices = interval_index(safe_duration, boundaries, open_tail=open_tail)
    event_rows = safe_types > 0
    cause_indices = (safe_types - 1).clamp_min(0)
    batch_indices = torch.arange(normalized.shape[0], device=normalized.device)
    selected = normalized[batch_indices, indices, cause_indices]
    losses = cumulative - event_rows.to(normalized) * torch.log(selected.clamp_min(log_epsilon))
    return _reduce_losses(losses, valid, reduction, patient_ids)


@dataclass(frozen=True)
class CompetingRiskCurves:
    """Cause-specific cumulative incidence and all-cause survival."""

    survival: Tensor
    cif: Tensor
    risk: Tensor


@dataclass(frozen=True)
class LandmarkLabels:
    """Landmark-relative labels; invalid rows carry NaN rather than clipped time."""

    remaining_time: Tensor
    event: Tensor
    valid: Tensor


def build_landmark_labels(
    observed_times: Tensor,
    events: Tensor,
    query_times: Tensor,
    *,
    eligible_mask: Tensor | None = None,
) -> LandmarkLabels:
    """Build labels only for patients still observable after the query.

    observed_times is the event time when events is one and otherwise the
    censoring time, measured from the same origin as query_times. Events or
    censoring at or before the query are excluded. Invalid remaining times are
    NaN so a downstream caller cannot silently train on a clipped zero.
    """

    if observed_times.shape != events.shape or observed_times.shape != query_times.shape:
        raise ValueError("observed_times, events and query_times must have identical shapes")
    if observed_times.ndim != 1:
        raise ValueError("landmark label inputs must be one-dimensional")
    if not observed_times.is_floating_point():
        observed_times = observed_times.to(torch.get_default_dtype())
    query_times = query_times.to(observed_times)
    event_values = events.to(observed_times)
    if not torch.isfinite(observed_times).all() or not torch.isfinite(query_times).all():
        raise ValueError("landmark times must be finite")
    if torch.any(observed_times < 0) or torch.any(query_times < 0):
        raise ValueError("landmark times cannot be negative")
    if not torch.all((event_values == 0) | (event_values == 1)):
        raise ValueError("events must be binary")
    eligible = (
        torch.ones_like(event_values, dtype=torch.bool)
        if eligible_mask is None
        else torch.as_tensor(eligible_mask, dtype=torch.bool, device=observed_times.device)
    )
    if eligible.shape != events.shape:
        raise ValueError("eligible_mask must match event shape")
    raw_remaining = observed_times - query_times
    valid = eligible & (raw_remaining > 0)
    remaining = torch.where(
        valid,
        raw_remaining,
        torch.full_like(raw_remaining, float("nan")),
    )
    landmark_events = torch.where(valid, event_values, torch.zeros_like(event_values))
    return LandmarkLabels(remaining_time=remaining, event=landmark_events, valid=valid)


def competing_risk_curves(
    rates: Tensor,
    horizons: Tensor,
    cuts: Tensor | Sequence[float],
    *,
    open_tail: bool = True,
) -> CompetingRiskCurves:
    """Integrate piecewise cause-specific hazards with a stable zero-rate limit."""

    normalized, _ = _validated_rates(rates, allow_causes=True)
    if normalized.ndim != 3:
        raise ValueError("rates must have shape [batch, intervals, causes]")
    if horizons.ndim != 1:
        raise ValueError("horizons must be one-dimensional")
    boundaries = _validated_cuts(cuts, like=normalized, intervals=normalized.shape[1])
    exposure = interval_exposure(horizons.to(normalized), boundaries, open_tail=open_tail)
    total_rates = normalized.sum(dim=-1)
    interval_hazard = total_rates[:, None, :] * exposure[None, :, :]
    cumulative_before = interval_hazard.cumsum(dim=-1) - interval_hazard
    survival_before = torch.exp(-cumulative_before)
    event_probability = -torch.expm1(-interval_hazard)
    positive_total = total_rates > 0
    safe_total = torch.where(positive_total, total_rates, torch.ones_like(total_rates))
    proportions = torch.where(
        positive_total[..., None],
        normalized / safe_total[..., None],
        torch.zeros_like(normalized),
    )
    increments = (
        survival_before[..., None] * event_probability[..., None] * proportions[:, None, :, :]
    )
    cif = increments.sum(dim=2)
    integrated = interval_hazard.sum(dim=-1)
    survival = torch.exp(-integrated)
    risk = -torch.expm1(-integrated)
    return CompetingRiskCurves(survival=survival, cif=cif, risk=risk)


def convert_time(
    values: Tensor,
    *,
    from_unit: TimeUnit,
    to_unit: TimeUnit,
    days_per_year: float = 365.25,
) -> Tensor:
    """Convert physical durations without changing the represented time."""

    scale = _time_scale(from_unit, to_unit, days_per_year)
    return values * scale


def convert_hazard_rates(
    rates: Tensor,
    *,
    from_unit: TimeUnit,
    to_unit: TimeUnit,
    days_per_year: float = 365.25,
) -> Tensor:
    """Convert rates inversely to durations so integrated hazard is invariant."""

    scale = _time_scale(from_unit, to_unit, days_per_year)
    return rates / scale


def event_nll_unit_shift(
    *,
    from_unit: TimeUnit,
    to_unit: TimeUnit,
    days_per_year: float = 365.25,
) -> float:
    """Return ``event_NLL(to_unit) - event_NLL(from_unit)``.

    Event likelihood is a density and changes by this constant Jacobian term.
    Censoring likelihood and all predicted probabilities are unit invariant.
    """

    return math.log(_time_scale(from_unit, to_unit, days_per_year))


def _time_scale(from_unit: TimeUnit, to_unit: TimeUnit, days_per_year: float) -> float:
    if days_per_year <= 0 or not math.isfinite(days_per_year):
        raise ValueError("days_per_year must be finite and positive")
    if from_unit == to_unit:
        return 1.0
    if from_unit == "day" and to_unit == "year":
        return 1.0 / days_per_year
    if from_unit == "year" and to_unit == "day":
        return days_per_year
    raise ValueError(f"unsupported time-unit conversion: {from_unit} -> {to_unit}")


def rates_to_interval_hazards(rates: Tensor, cuts: Tensor | Sequence[float]) -> Tensor:
    """Convert physical-time rates to pycox-style complete-interval hazards."""

    normalized, _ = _validated_rates(rates, allow_causes=True)
    boundaries = _validated_cuts(cuts, like=normalized, intervals=normalized.shape[1])
    widths = boundaries[1:] - boundaries[:-1]
    if normalized.ndim == 2:
        return normalized * widths
    return normalized * widths[:, None]


@dataclass(frozen=True)
class PiecewiseHazardOutput:
    """Hazard rates plus their immutable interval semantics."""

    rates: Tensor
    cuts: Tensor
    open_tail: bool = True

    def survival(self, horizons: Tensor) -> Tensor:
        return survival_probability(self.rates, horizons, self.cuts, open_tail=self.open_tail)

    def risk(self, horizons: Tensor) -> Tensor:
        return risk_probability(self.rates, horizons, self.cuts, open_tail=self.open_tail)

    def competing_risks(self, horizons: Tensor) -> CompetingRiskCurves:
        return competing_risk_curves(self.rates, horizons, self.cuts, open_tail=self.open_tail)


class PiecewiseHazardHead(nn.Module):
    """Small output head shared by world-model and matched-input baselines."""

    cuts: Tensor

    def __init__(
        self,
        input_dim: int,
        cuts: Tensor | Sequence[float],
        *,
        num_causes: int = 1,
        min_rate: float = 1e-7,
        open_tail: bool = True,
    ) -> None:
        super().__init__()
        if input_dim < 1 or num_causes < 1:
            raise ValueError("input_dim and num_causes must be positive")
        if min_rate <= 0:
            raise ValueError("min_rate must be positive")
        boundary_tensor = torch.as_tensor(cuts, dtype=torch.float32)
        _validated_cuts(
            boundary_tensor,
            like=boundary_tensor,
            intervals=boundary_tensor.numel() - 1,
        )
        self.intervals = boundary_tensor.numel() - 1
        self.num_causes = num_causes
        self.min_rate = min_rate
        self.open_tail = open_tail
        self.projection = nn.Linear(input_dim, self.intervals * num_causes)
        self.register_buffer("cuts", boundary_tensor)

    def forward(self, features: Tensor) -> PiecewiseHazardOutput:
        if features.ndim != 2:
            raise ValueError("survival-head features must have shape [batch, input_dim]")
        raw = self.projection(features)
        rates = F.softplus(raw) + self.min_rate
        if self.num_causes == 1:
            rates = rates.reshape(features.shape[0], self.intervals)
        else:
            rates = rates.reshape(features.shape[0], self.intervals, self.num_causes)
        return PiecewiseHazardOutput(rates=rates, cuts=self.cuts, open_tail=self.open_tail)


__all__ = [
    "CompetingRiskCurves",
    "LandmarkLabels",
    "PiecewiseHazardHead",
    "PiecewiseHazardOutput",
    "build_landmark_labels",
    "cause_specific_nll",
    "competing_risk_curves",
    "convert_hazard_rates",
    "convert_time",
    "cumulative_hazard",
    "event_nll_unit_shift",
    "interval_exposure",
    "interval_index",
    "masked_patient_mean",
    "piecewise_exponential_nll",
    "rates_to_interval_hazards",
    "risk_probability",
    "survival_probability",
]
