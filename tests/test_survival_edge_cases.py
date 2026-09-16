from __future__ import annotations

import math

import pytest
import torch

from stageworld.survival import (
    PiecewiseHazardHead,
    cumulative_hazard,
    interval_exposure,
    interval_index,
    piecewise_exponential_nll,
)


def test_t14_internal_boundaries_and_open_tail_are_explicit() -> None:
    cuts = [0.0, 1.0, 3.0]
    times = torch.tensor([0.0, 1.0, 3.0, 5.0], dtype=torch.float64)
    assert interval_index(times, cuts).tolist() == [0, 1, 1, 1]
    expected_exposure = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [1.0, 4.0]], dtype=torch.float64
    )
    torch.testing.assert_close(interval_exposure(times, cuts), expected_exposure)
    hazard = cumulative_hazard(
        torch.tensor([[0.1, 0.3]], dtype=torch.float64),
        torch.tensor([5.0], dtype=torch.float64),
        cuts,
    )
    torch.testing.assert_close(hazard, torch.tensor([1.3], dtype=torch.float64))


def test_t14_closed_tail_rejects_only_times_beyond_administrative_end() -> None:
    rates = torch.tensor([[0.1, 0.3]], dtype=torch.float64)
    at_end = piecewise_exponential_nll(
        rates,
        torch.tensor([3.0], dtype=torch.float64),
        torch.tensor([0]),
        [0.0, 1.0, 3.0],
        open_tail=False,
    )
    torch.testing.assert_close(at_end, torch.tensor(0.7, dtype=torch.float64))
    with pytest.raises(ValueError, match="closed administrative horizon"):
        piecewise_exponential_nll(
            rates,
            torch.tensor([3.1], dtype=torch.float64),
            torch.tensor([0]),
            [0.0, 1.0, 3.0],
            open_tail=False,
        )


def test_t14_zero_duration_requires_explicit_allow_policy() -> None:
    rates = torch.tensor([[0.2], [0.2]], dtype=torch.float64)
    times = torch.tensor([0.0, 0.0], dtype=torch.float64)
    events = torch.tensor([1, 0])
    with pytest.raises(ValueError, match="zero remaining time"):
        piecewise_exponential_nll(rates, times, events, [0.0, 1.0], reduction="none")
    losses = piecewise_exponential_nll(
        rates,
        times,
        events,
        [0.0, 1.0],
        zero_time_policy="allow",
        reduction="none",
    )
    torch.testing.assert_close(losses, torch.tensor([-math.log(0.2), 0.0], dtype=torch.float64))


def test_t14_all_censored_and_single_event_batches_are_finite() -> None:
    rates = torch.full((3, 2), 0.2, dtype=torch.float64, requires_grad=True)
    durations = torch.tensor([0.5, 2.0, 7.0], dtype=torch.float64)
    all_censored = piecewise_exponential_nll(rates, durations, torch.zeros(3), [0.0, 1.0, 3.0])
    one_event = piecewise_exponential_nll(
        rates, durations, torch.tensor([0, 1, 0]), [0.0, 1.0, 3.0]
    )
    assert torch.isfinite(all_censored)
    assert torch.isfinite(one_event)
    (all_censored + one_event).backward()
    assert torch.isfinite(rates.grad).all()


def test_negative_duration_is_rejected_not_clipped() -> None:
    with pytest.raises(ValueError, match="negative remaining time"):
        piecewise_exponential_nll(
            torch.tensor([[0.2]]),
            torch.tensor([-0.1]),
            torch.tensor([0]),
            [0.0, 1.0],
        )


def test_shared_hazard_head_is_positive_and_differentiable() -> None:
    head = PiecewiseHazardHead(5, [0.0, 1.0, 3.0], num_causes=2)
    features = torch.randn(4, 5, requires_grad=True)
    output = head(features)
    assert output.rates.shape == (4, 2, 2)
    assert torch.all(output.rates > 0)
    loss = output.survival(torch.tensor([0.0, 1.0, 2.0])).sum()
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
