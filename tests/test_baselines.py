from __future__ import annotations

import pytest
import torch

from stageworld.baselines import (
    CTSMambaFeatureAdaptation,
    DirectFusionSurvival,
    DynamicDeepHitAdaptation,
    GRUDynamicSurvival,
    LongitudinalTransformerSurvival,
)
from stageworld.survival import piecewise_exponential_nll

CUTS = [0.0, 1.0, 3.0]


def _assert_trainable_prediction(model: torch.nn.Module, output_rates: torch.Tensor) -> None:
    assert output_rates.shape == (3, 2)
    assert torch.all(output_rates > 0)
    loss = piecewise_exponential_nll(
        output_rates,
        torch.tensor([0.5, 1.5, 4.0]),
        torch.tensor([1, 0, 1]),
        CUTS,
    )
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert any(gradient is not None and torch.any(gradient != 0) for gradient in gradients)


def test_direct_fusion_is_trainable_and_ignores_masked_tokens() -> None:
    model = DirectFusionSurvival({"ct": 6, "clinical": 4}, CUTS, hidden_dim=12, dropout=0.0)
    model.eval()
    ct = torch.randn(3, 4, 6)
    clinical = torch.randn(3, 4)
    ct_valid = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
    output = model(
        {"ct": ct, "clinical": clinical},
        {"ct": ct_valid, "clinical": torch.tensor([1, 1, 0], dtype=torch.bool)},
    )
    changed = ct.clone()
    changed[~ct_valid] = 100_000.0
    output_changed = model(
        {"ct": changed, "clinical": clinical},
        {"ct": ct_valid, "clinical": torch.tensor([1, 1, 0], dtype=torch.bool)},
    )
    torch.testing.assert_close(output.rates, output_changed.rates)
    _assert_trainable_prediction(model, output.rates)


def test_longitudinal_transformer_padding_is_invariant_and_trainable() -> None:
    model = LongitudinalTransformerSurvival(
        5, CUTS, hidden_dim=16, num_heads=4, num_layers=1, dropout=0.0
    )
    model.eval()
    sequence = torch.randn(3, 4, 5)
    valid = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.bool)
    times = torch.tensor(
        [[0.0, 2.0, float("nan"), float("nan")], [0.0, 1.0, 4.0, 99.0], [3.0, 9.0, 9.0, 9.0]]
    )
    output = model(sequence, valid, times)
    changed = sequence.clone()
    changed[~valid] = float("nan")
    changed_times = times.clone()
    changed_times[~valid] = float("nan")
    output_changed = model(changed, valid, changed_times)
    torch.testing.assert_close(output.rates, output_changed.rates)
    _assert_trainable_prediction(model, output.rates)


def test_gru_prefix_mask_and_padding_behavior() -> None:
    model = GRUDynamicSurvival(5, CUTS, hidden_dim=10)
    model.eval()
    sequence = torch.randn(3, 4, 5)
    valid = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.bool)
    times = torch.tensor([[0.0, 2.0, 8.0, 8.0], [0.0, 1.0, 4.0, 8.0], [2.0, 8.0, 8.0, 8.0]])
    output = model(sequence, valid, times)
    changed = sequence.clone()
    changed[~valid] = 50_000.0
    output_changed = model(changed, valid, times + (~valid).to(times) * 50_000.0)
    torch.testing.assert_close(output.rates, output_changed.rates)
    _assert_trainable_prediction(model, output.rates)

    hole = valid.clone()
    hole[0] = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
    with pytest.raises(ValueError, match="contiguous prefixes"):
        model(sequence, hole, times)


def test_named_adaptations_explicitly_disclaim_upstream_reproduction() -> None:
    dynamic = DynamicDeepHitAdaptation(5, CUTS, hidden_dim=8)
    cts = CTSMambaFeatureAdaptation(6, CUTS, hidden_dim=12, num_heads=3, dropout=0.0)
    assert dynamic.metadata.exact_upstream_reproduction is False
    assert cts.metadata.exact_upstream_reproduction is False
    assert "not the upstream" in dynamic.metadata.implementation_scope
    assert "not the upstream" in cts.metadata.implementation_scope


def test_ctsmamba_feature_adaptation_is_paired_coattention_not_original_model() -> None:
    model = CTSMambaFeatureAdaptation(6, CUTS, hidden_dim=12, num_heads=3, dropout=0.0)
    model.eval()
    baseline = torch.randn(3, 3, 6)
    post = torch.randn(3, 4, 6)
    baseline_valid = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=torch.bool)
    post_valid = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.bool)
    output = model(baseline, post, baseline_valid, post_valid)
    changed_baseline = baseline.clone()
    changed_post = post.clone()
    changed_baseline[~baseline_valid] = 100_000.0
    changed_post[~post_valid] = -100_000.0
    output_changed = model(changed_baseline, changed_post, baseline_valid, post_valid)
    torch.testing.assert_close(output.rates, output_changed.rates)
    _assert_trainable_prediction(model, output.rates)

    invalid_post = post_valid.clone()
    invalid_post[0] = False
    with pytest.raises(ValueError, match="requires both CT observations"):
        model(baseline, post, baseline_valid, invalid_post)


def test_baselines_can_share_competing_risk_output_contract() -> None:
    model = DirectFusionSurvival({"state": 7}, CUTS, hidden_dim=8, num_causes=2)
    output = model({"state": torch.randn(3, 7)})
    assert output.rates.shape == (3, 2, 2)
    curves = output.competing_risks(torch.tensor([0.0, 1.0, 2.0]))
    torch.testing.assert_close(
        curves.survival + curves.cif.sum(dim=-1), torch.ones_like(curves.survival)
    )
