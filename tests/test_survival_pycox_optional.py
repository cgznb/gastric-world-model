from __future__ import annotations

import pytest
import torch

from stageworld.survival import (
    interval_exposure,
    interval_index,
    piecewise_exponential_nll,
    rates_to_interval_hazards,
)


@pytest.mark.integration
def test_optional_pycox_pc_hazard_matches_after_physical_rate_conversion() -> None:
    pycox_loss = pytest.importorskip("pycox.models.loss")
    cuts = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64)
    rates = torch.tensor(
        [[0.2, 0.4], [0.3, 0.1], [0.5, 0.2]], dtype=torch.float64, requires_grad=True
    )
    durations = torch.tensor([0.25, 2.0, 3.0], dtype=torch.float64)
    events = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
    indices = interval_index(durations, cuts, open_tail=False)
    exposure = interval_exposure(durations, cuts, open_tail=False)
    widths = cuts[1:] - cuts[:-1]
    fractions = exposure[torch.arange(rates.shape[0]), indices] / widths[indices]
    interval_hazards = rates_to_interval_hazards(rates, cuts)
    phi = torch.log(torch.expm1(interval_hazards))

    upstream = pycox_loss.nll_pc_hazard_loss(phi, indices, events, fractions, reduction="none")
    converted_upstream = upstream + events * torch.log(widths[indices])
    ours = piecewise_exponential_nll(
        rates, durations, events, cuts, open_tail=False, reduction="none"
    )
    torch.testing.assert_close(ours, converted_upstream)

    ours_gradient = torch.autograd.grad(ours.sum(), rates, retain_graph=True)[0]
    upstream_gradient = torch.autograd.grad(converted_upstream.sum(), rates)[0]
    torch.testing.assert_close(ours_gradient, upstream_gradient)
