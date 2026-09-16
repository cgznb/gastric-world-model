from __future__ import annotations

import math

import pytest
import torch

from stageworld.survival import (
    build_landmark_labels,
    cause_specific_nll,
    competing_risk_curves,
    convert_hazard_rates,
    convert_time,
    cumulative_hazard,
    event_nll_unit_shift,
    interval_exposure,
    interval_index,
    masked_patient_mean,
    piecewise_exponential_nll,
    rates_to_interval_hazards,
    risk_probability,
    survival_probability,
)


def test_t11_constant_hazard_event_and_censor_nll() -> None:
    rates = torch.tensor([[0.2], [0.2]], dtype=torch.float64)
    durations = torch.tensor([2.0, 2.0], dtype=torch.float64)
    events = torch.tensor([1, 0])
    losses = piecewise_exponential_nll(rates, durations, events, [0.0, 1.0], reduction="none")
    expected = torch.tensor([0.4 - math.log(0.2), 0.4], dtype=torch.float64)
    torch.testing.assert_close(losses, expected)


def test_t12_piecewise_hazard_exact_exposure() -> None:
    rates = torch.tensor([[0.1, 0.3], [0.1, 0.3]], dtype=torch.float64)
    durations = torch.tensor([2.0, 2.0], dtype=torch.float64)
    events = torch.tensor([1, 0])
    hazard = cumulative_hazard(rates, durations, [0.0, 1.0, 3.0])
    torch.testing.assert_close(hazard, torch.tensor([0.4, 0.4], dtype=torch.float64))
    losses = piecewise_exponential_nll(rates, durations, events, [0.0, 1.0, 3.0], reduction="none")
    expected = torch.tensor([0.4 - math.log(0.3), 0.4], dtype=torch.float64)
    torch.testing.assert_close(losses, expected)


def _pycox_parameterization_reference(
    rates: torch.Tensor,
    durations: torch.Tensor,
    events: torch.Tensor,
    cuts: torch.Tensor,
) -> torch.Tensor:
    """Reference pycox PCHazard formula converted back to physical density."""

    interval_hazards = rates_to_interval_hazards(rates, cuts)
    indices = interval_index(durations, cuts, open_tail=False)
    exposure = interval_exposure(durations, cuts, open_tail=False)
    widths = cuts[1:] - cuts[:-1]
    rows = torch.arange(rates.shape[0])
    current_hazard = interval_hazards[rows, indices]
    current_fraction = exposure[rows, indices] / widths[indices]
    interval_numbers = torch.arange(rates.shape[1]).unsqueeze(0)
    previous = torch.where(interval_numbers < indices.unsqueeze(1), interval_hazards, 0.0).sum(
        dim=1
    )
    pycox_nll = previous + current_hazard * current_fraction - events * torch.log(current_hazard)
    # pycox is an interval-mass density; add log(width) for physical-time rate density.
    return pycox_nll + events * torch.log(widths[indices])


def test_t13_pycox_parameterization_value_and_gradient_reference() -> None:
    cuts = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64)
    durations = torch.tensor([0.25, 2.0, 3.0], dtype=torch.float64)
    events = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
    rates = torch.tensor(
        [[0.2, 0.4], [0.3, 0.1], [0.5, 0.2]], dtype=torch.float64, requires_grad=True
    )
    actual = piecewise_exponential_nll(
        rates, durations, events, cuts, open_tail=False, reduction="mean"
    )
    reference = _pycox_parameterization_reference(rates, durations, events, cuts).mean()
    torch.testing.assert_close(actual, reference)
    actual_gradient = torch.autograd.grad(actual, rates, retain_graph=True)[0]
    reference_gradient = torch.autograd.grad(reference, rates)[0]
    torch.testing.assert_close(actual_gradient, reference_gradient)


def test_t15_survival_and_risk_are_valid_and_monotone() -> None:
    rates = torch.tensor([[0.1, 0.3], [0.0, 0.2]], dtype=torch.float64)
    horizons = torch.tensor([0.0, 0.5, 1.0, 2.0, 8.0], dtype=torch.float64)
    survival = survival_probability(rates, horizons, [0.0, 1.0, 3.0])
    risk = risk_probability(rates, horizons, [0.0, 1.0, 3.0])
    torch.testing.assert_close(survival[:, 0], torch.ones(2, dtype=torch.float64))
    assert torch.all(survival[:, 1:] <= survival[:, :-1])
    assert torch.all(risk[:, 1:] >= risk[:, :-1])
    assert torch.all((risk >= 0) & (risk <= 1))
    torch.testing.assert_close(survival + risk, torch.ones_like(survival))


def test_t16_competing_risk_cif_conserves_probability() -> None:
    rates = torch.tensor(
        [
            [[0.1, 0.2], [0.3, 0.1]],
            [[0.4, 0.0], [0.2, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    horizons = torch.tensor([0.0, 0.5, 1.0, 2.0, 10.0], dtype=torch.float64)
    curves = competing_risk_curves(rates, horizons, [0.0, 1.0, 3.0])
    assert torch.all(curves.cif >= 0)
    torch.testing.assert_close(
        curves.survival + curves.cif.sum(dim=-1), torch.ones_like(curves.survival)
    )
    torch.testing.assert_close(curves.risk, curves.cif.sum(dim=-1))
    torch.testing.assert_close(curves.cif[1, :, 1], torch.zeros_like(curves.cif[1, :, 1]))
    single_survival = survival_probability(rates[1:2, :, 0], horizons, [0.0, 1.0, 3.0])
    torch.testing.assert_close(curves.survival[1:2], single_survival)
    torch.testing.assert_close(curves.survival[2], torch.ones_like(curves.survival[2]))
    torch.testing.assert_close(curves.cif[2], torch.zeros_like(curves.cif[2]))


def test_cause_specific_event_and_censor_nll() -> None:
    rates = torch.tensor([[[0.1, 0.2]], [[0.1, 0.2]]], dtype=torch.float64)
    durations = torch.tensor([2.0, 2.0], dtype=torch.float64)
    event_types = torch.tensor([2, 0])
    losses = cause_specific_nll(rates, durations, event_types, [0.0, 1.0], reduction="none")
    expected = torch.tensor([0.6 - math.log(0.2), 0.6], dtype=torch.float64)
    torch.testing.assert_close(losses, expected)


def test_t17_day_year_conversion_preserves_probabilities_and_censor_nll() -> None:
    rate_year = torch.tensor([[0.2, 0.4]], dtype=torch.float64)
    cuts_year = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64)
    time_year = torch.tensor([2.0], dtype=torch.float64)
    rate_day = convert_hazard_rates(rate_year, from_unit="year", to_unit="day")
    cuts_day = convert_time(cuts_year, from_unit="year", to_unit="day")
    time_day = convert_time(time_year, from_unit="year", to_unit="day")

    survival_year = survival_probability(rate_year, time_year, cuts_year)
    survival_day = survival_probability(rate_day, time_day, cuts_day)
    torch.testing.assert_close(survival_year, survival_day)
    censor_year = piecewise_exponential_nll(rate_year, time_year, torch.tensor([0]), cuts_year)
    censor_day = piecewise_exponential_nll(rate_day, time_day, torch.tensor([0]), cuts_day)
    torch.testing.assert_close(censor_year, censor_day)

    event_year = piecewise_exponential_nll(rate_year, time_year, torch.tensor([1]), cuts_year)
    event_day = piecewise_exponential_nll(rate_day, time_day, torch.tensor([1]), cuts_day)
    expected_shift = event_nll_unit_shift(from_unit="year", to_unit="day")
    torch.testing.assert_close(event_day - event_year, event_year.new_tensor(expected_shift))


def test_t18_landmark_labels_exclude_prior_events_without_clipping() -> None:
    labels = build_landmark_labels(
        observed_times=torch.tensor([10.0, 5.0, 4.0, 12.0]),
        events=torch.tensor([1, 1, 0, 0]),
        query_times=torch.tensor([3.0, 5.0, 6.0, 2.0]),
        eligible_mask=torch.tensor([1, 1, 1, 0], dtype=torch.bool),
    )
    assert labels.valid.tolist() == [True, False, False, False]
    assert labels.remaining_time[0].item() == 7.0
    assert torch.isnan(labels.remaining_time[1:]).all()
    assert labels.event.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_t19_patient_normalization_prevents_duplicate_weighting() -> None:
    losses = torch.tensor([[1.0, 3.0, 99.0], [10.0, 99.0, 99.0]], requires_grad=True)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    reduced = masked_patient_mean(losses, mask)
    torch.testing.assert_close(reduced, torch.tensor(6.0))
    reduced.backward()
    torch.testing.assert_close(
        losses.grad,
        torch.tensor([[0.25, 0.25, 0.0], [0.5, 0.0, 0.0]]),
    )

    flat = masked_patient_mean(torch.tensor([1.0, 3.0, 10.0]), patient_ids=["p1", "p1", "p2"])
    torch.testing.assert_close(flat, torch.tensor(6.0))


def test_masked_invalid_labels_do_not_create_nan() -> None:
    rates = torch.tensor([[0.2], [0.3]], dtype=torch.float64, requires_grad=True)
    loss = piecewise_exponential_nll(
        rates,
        torch.tensor([2.0, float("nan")], dtype=torch.float64),
        torch.tensor([1, 9]),
        [0.0, 1.0],
        valid_mask=torch.tensor([True, False]),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(rates.grad).all()
    assert rates.grad[1].item() == 0.0


def test_invalid_cut_contract_rejected() -> None:
    rates = torch.ones(1, 2)
    with pytest.raises(ValueError, match="start at zero"):
        piecewise_exponential_nll(rates, torch.tensor([1.0]), torch.tensor([0]), [1, 2, 3])
    with pytest.raises(ValueError, match="strictly increasing"):
        piecewise_exponential_nll(rates, torch.tensor([1.0]), torch.tensor([0]), [0, 1, 1])
