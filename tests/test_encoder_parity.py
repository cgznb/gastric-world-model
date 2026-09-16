from __future__ import annotations

import pytest


@pytest.mark.requires_weights
@pytest.mark.skip(reason="Approved local Merlin weights and official preprocessing are unavailable")
def test_merlin_adapter_matches_official_global_output() -> None:
    """Reserved numerical parity test; a contract skip is not integration success."""


@pytest.mark.requires_weights
@pytest.mark.skip(reason="Approved TITAN/CONCH weights and audited remote code are unavailable")
def test_titan_adapter_matches_official_slide_output() -> None:
    """Reserved numerical parity test; a contract skip is not integration success."""
