from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from stageworld.event_models import EventInputs
from stageworld.event_spec import ABSENT, CONFLICT, PRESENT, UNKNOWN
from stageworld.event_v2_models import EventV2Model, mean_probability_logit


@pytest.fixture
def inputs():
    torch.manual_seed(17)
    torch.set_num_threads(1)
    return EventInputs(
        torch.randn(4, 360),
        torch.randn(4, 27, 8),
        torch.full((4, 3), PRESENT, dtype=torch.long),
    )


def model(**kwargs):
    return EventV2Model(image_dim=8, hidden=16, layers=2, rank=3, **kwargs).eval()


def gradient_total(module):
    return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)


def test_dimensions_and_tensor_shapes(inputs):
    network = model()
    out = network(inputs)
    assert out.states.shape == (4, 4, 27, 16)
    assert out.stage_increments.shape == (4, 3, 27, 16)
    assert out.ct1.shape == (4, 27, 8)
    assert out.member_logits.shape == (4, 2, 4)
    assert out.pcr_logit.shape == out.recurrence_logit.shape == (4,)
    assert network.world.clinical(inputs.x).shape == (4, 6, 16)
    assert network.world.treatment(inputs.x[:, 32:].reshape(4, 4, 82)).shape == (4, 4, 16)
    assert out.last_stage.tolist() == [3] * 4
    assert not out.incomplete_history.any()
    assert all(torch.isfinite(value).all() for value in vars(out).values())
    restored = EventV2Model(**network.dimensions()).eval()
    restored.load_state_dict(network.state_dict())
    assert torch.equal(restored(inputs).member_logits, out.member_logits)


@pytest.mark.parametrize("status", [ABSENT, UNKNOWN, CONFLICT])
def test_exact_skip_and_incomplete_history(inputs, status):
    network = model()
    events = torch.full_like(inputs.events, status)
    events[0] = PRESENT
    out = network(replace(inputs, events=events))
    for stage in range(1, 4):
        assert torch.equal(out.states[1:, 0], out.states[1:, stage])
    assert torch.equal(out.stage_increments[1:], torch.zeros_like(out.stage_increments[1:]))
    assert out.last_stage.tolist() == [3, 0, 0, 0]
    assert out.incomplete_history.tolist() == [False] + [status in (UNKNOWN, CONFLICT)] * 3


def test_causal_prefix_and_action_exposure(inputs):
    network = model()
    full = network(inputs)
    events = inputs.events.clone()
    events[:, 2] = ABSENT
    without_postop = network(replace(inputs, events=events))
    assert torch.equal(full.states[:, :3], without_postop.states[:, :3])
    assert torch.equal(full.ct1, without_postop.ct1)
    assert torch.equal(full.pcr_logit, without_postop.pcr_logit)
    assert not torch.equal(full.recurrence_logit, without_postop.recurrence_logit)
    changed_x = inputs.x.clone()
    changed_x[:, 32:] += torch.randn_like(changed_x[:, 32:]) * 5
    changed = network(replace(inputs, x=changed_x))
    assert torch.equal(full.states[:, 0], changed.states[:, 0])
    assert not torch.equal(full.states[:, 1], changed.states[:, 1])
    events[:, 0] = ABSENT
    original = network(replace(inputs, events=events))
    changed = network(replace(inputs, x=changed_x, events=events))
    assert torch.equal(original.states, changed.states)
    assert torch.equal(original.member_logits, changed.member_logits)


def test_gradients_follow_stage_and_endpoint_boundaries(inputs):
    network = model()
    network(inputs).pcr_logit.sum().backward()
    for block in network.world.blocks:
        assert gradient_total(block.adapters[0]) > 0
        assert gradient_total(block.adapters[1]) > 0
        assert gradient_total(block.adapters[2]) == 0
    assert gradient_total(network.recurrence_head) == 0
    assert gradient_total(network.world.event_codes[2]) == 0
    assert gradient_total(network.world.decoder) == 0
    network.zero_grad(set_to_none=True)
    network(inputs).recurrence_logit.sum().backward()
    for block in network.world.blocks:
        assert all(gradient_total(adapter) > 0 for adapter in block.adapters)
    assert gradient_total(network.pcr_head) == 0
    assert gradient_total(network.world.image) > 0
    assert gradient_total(network.world.clinical) > 0


def test_eval_is_independent_of_batch_composition(inputs):
    network = model()
    with torch.no_grad():
        full = network(inputs)
        single = network(EventInputs(inputs.x[:1], inputs.ct0[:1], inputs.events[:1]))
    torch.testing.assert_close(single.states, full.states[:1], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(single.member_logits, full.member_logits[:1], atol=2e-6, rtol=2e-6)


def test_probability_aggregation_and_extreme_logits(inputs):
    network = model()
    out = network(inputs)
    for endpoint, aggregate in enumerate((out.pcr_logit, out.recurrence_logit)):
        expected = out.member_logits[:, endpoint].sigmoid().mean(-1)
        torch.testing.assert_close(aggregate.sigmoid(), expected)
    asymmetric = torch.tensor([[0.0, 0.0, 0.0, 8.0], [1000.0] * 4, [-1000.0] * 4])
    logits = mean_probability_logit(asymmetric)
    assert torch.isfinite(logits).all()
    assert not torch.allclose(logits[:1], asymmetric[:1].mean(-1))
    torch.testing.assert_close(logits[1:], torch.tensor([1000.0, -1000.0]))
    single = model(members=1)(inputs)
    torch.testing.assert_close(single.pcr_logit, single.member_logits[:, 0, 0])


def test_masked_source_target_detach_and_loss_formula(inputs):
    network = model()
    ct0 = inputs.ct0.clone().requires_grad_()
    current = replace(inputs, ct0=ct0)
    mask = torch.zeros(4, 27, dtype=torch.bool)
    mask[:, :7] = True
    loss = network.masked_source_loss(current, mask)
    state = network.world.encode_baseline(current, network.world.clinical(current.x), mask)
    expected = F.smooth_l1_loss(network.world.source_decoder(state)[mask], ct0.detach()[mask])
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert ct0.grad is not None
    assert torch.equal(ct0.grad[mask], torch.zeros_like(ct0.grad[mask]))
    assert ct0.grad[~mask].abs().sum() > 0
    assert gradient_total(network.world.source_decoder) > 0
    assert gradient_total(network.world.decoder) == 0
    assert torch.isfinite(loss)
    network.zero_grad(set_to_none=True)
    zero = network.masked_source_loss(inputs, torch.zeros_like(mask))
    assert zero.requires_grad and zero.item() == 0
    zero.backward()
    assert gradient_total(network.world.source_decoder) == 0


def test_masked_encoder_hides_masked_values(inputs):
    network = model()
    mask = torch.zeros(4, 27, dtype=torch.bool)
    mask[:, 2:8] = True
    changed = replace(inputs, ct0=inputs.ct0.clone())
    changed.ct0[mask] += 100
    clinical = network.world.clinical(inputs.x)
    first = network.world.encode_baseline(inputs, clinical, mask)
    second = network.world.encode_baseline(changed, clinical, mask)
    assert torch.equal(first, second)


def test_bfloat16_training_preserves_finite_float32_outputs(inputs):
    network = model().train()
    mask = torch.zeros(4, 27, dtype=torch.bool)
    mask[:, :6] = True
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = network(inputs)
        loss = out.member_logits.square().mean() + network.masked_source_loss(inputs, mask)
        loss = loss + out.ct1.square().mean()
    assert out.states.dtype == out.member_logits.dtype == out.ct1.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None)
    assert all(gradient_total(block.adapters[2]) > 0 for block in network.world.blocks)


def test_statistics_only_use_supplied_training_rows_and_valid_targets(inputs):
    network = model()
    ct0 = inputs.ct0[:2].clone().requires_grad_()
    ct1 = (ct0.detach() + 2).requires_grad_()
    ct1.data[1] = torch.nan
    valid = torch.tensor([True, False])
    network.world.fit_statistics(ct0, ct1, valid)
    torch.testing.assert_close(network.world.input_mean, ct0.detach().mean((0, 1)))
    scale = ct0.detach().std((0, 1), unbiased=False).clamp_min(0.05)
    torch.testing.assert_close(network.world.input_scale, scale)
    torch.testing.assert_close(network.world.decoder[-1].bias, 2 / scale)
    network(inputs).ct1.sum().backward()
    assert ct0.grad is None and ct1.grad is None
    network.world.fit_statistics(torch.ones_like(ct0), ct1, torch.zeros_like(valid))
    assert torch.equal(network.world.input_scale, torch.full((8,), 0.05))
    assert torch.equal(network.world.decoder[-1].bias, torch.zeros(8))
    assert network.world.source_decoder[-1].bias.grad is None


def test_ablation_variants(inputs):
    full = model()
    no_adapter = model(variant="no_stage_adapter")
    assert all(len(block.adapters) == 0 for block in no_adapter.world.blocks)
    assert sum(p.numel() for p in no_adapter.parameters()) < sum(
        p.numel() for p in full.parameters()
    )
    assert no_adapter(inputs).states.shape == full(inputs).states.shape
    static = model(variant="no_transition")
    out = static(inputs)
    assert all(torch.equal(out.states[:, 0], out.states[:, stage]) for stage in (1, 2, 3))
    changed = replace(inputs, x=inputs.x.clone(), events=torch.zeros_like(inputs.events))
    changed.x[:, 32:] += 10
    assert torch.equal(out.member_logits, static(changed).member_logits)


@pytest.mark.parametrize("problem", ["x", "ct0", "events", "nan", "postop", "empty"])
def test_invalid_forward_inputs_raise(inputs, problem):
    current = replace(inputs)
    if problem == "x":
        current.x = inputs.x[:, :359]
    elif problem == "ct0":
        current.ct0 = inputs.ct0[:, :26]
    elif problem == "events":
        current.events = inputs.events.float()
    elif problem == "nan":
        current.ct0 = inputs.ct0.clone()
        current.ct0[0, 0, 0] = torch.nan
    elif problem == "postop":
        current.events = inputs.events.clone()
        current.events[:, 1] = ABSENT
    else:
        current = EventInputs(inputs.x[:0], inputs.ct0[:0], inputs.events[:0])
    with pytest.raises(ValueError):
        model()(current)


def test_invalid_masks_and_configuration_raise(inputs):
    network = model()
    for mask in (torch.zeros(4, 27), torch.zeros(4, 26, dtype=torch.bool)):
        with pytest.raises(ValueError, match="mask"):
            network.masked_source_loss(inputs, mask)
    with pytest.raises(ValueError, match="divisible"):
        EventV2Model(hidden=15)
    with pytest.raises(ValueError, match="variant"):
        model(variant="unknown")
    with pytest.raises(ValueError, match="finite"):
        network.world.fit_statistics(
            inputs.ct0, torch.full_like(inputs.ct0, torch.nan), torch.ones(4, dtype=torch.bool)
        )
