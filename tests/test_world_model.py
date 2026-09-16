from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from stageworld.encoders.base import EncoderProvenance, ObservationTokens
from stageworld.errors import DataContractError
from stageworld.model import ActionTokens, StageWorldModel, StageWorldModelConfig


def _provenance(name: str, dim: int) -> EncoderProvenance:
    return EncoderProvenance(
        encoder_name=f"test_{name}",
        source_version="test-v1",
        component_versions=(("fixture", "v1"),),
        preprocess_version="identity-v1",
        feature_dim=dim,
    )


def _observation(
    name: str,
    values: torch.Tensor,
    *,
    acquired: float,
    available: float,
    valid: torch.Tensor | None = None,
    coords: torch.Tensor | None = None,
) -> ObservationTokens:
    batch, tokens, dim = values.shape
    if valid is None:
        valid = torch.ones(batch, tokens, dtype=torch.bool)
    source_id = tuple(
        tuple(f"test-{name}-{row}-{token}" for token in range(tokens)) for row in range(batch)
    )
    return ObservationTokens(
        values=values,
        valid=valid,
        modality=torch.full((batch, tokens), {"ct": 0, "pathology": 1, "clinical": 2}[name]),
        acquired_time=torch.full((batch, tokens), acquired),
        available_time=torch.full((batch, tokens), available),
        provenance=_provenance(name, dim),
        source_id=source_id,
        modality_name=name,
        coords=coords,
        coordinate_system=("normalized_physical_xyz" if coords is not None else None),
    )


def _actions(
    values: torch.Tensor,
    *,
    event_time: float,
    available_time: float,
    event_type: int,
) -> ActionTokens:
    batch, count, _ = values.shape
    return ActionTokens(
        values=values,
        valid=torch.ones(batch, count, dtype=torch.bool),
        event_time=torch.full((batch, count), event_time),
        available_time=torch.full((batch, count), available_time),
        event_type=torch.full((batch, count), event_type),
        planned_or_delivered=torch.full((batch, count), 2),
        known_exposure=torch.ones(batch, count),
        provenance=(f"test-action-{event_type}",),
    )


def _model(
    *,
    stochastic: bool = True,
    causes: int = 1,
    timeline_time_unit: str = "day",
) -> StageWorldModel:
    model = StageWorldModel(
        StageWorldModelConfig(
            hidden_dim=16,
            state_tokens=4,
            stochastic_dim=3,
            use_stochastic_state=stochastic,
            attention_heads=4,
            transition_blocks=2,
            observation_blocks=2,
            resampler_blocks=1,
            dropout=0.0,
            action_input_dim=5,
            modality_input_dims=(("ct", 8), ("pathology", 6), ("clinical", 4)),
            resampled_tokens=(("ct", 3), ("pathology", 2), ("clinical", 2)),
            future_output_dims=(("ct", 8), ("pathology", 6)),
            future_output_tokens=(("ct", 1), ("pathology", 1)),
            survival_cutpoints=(0.0, 1.0, 3.0),
            survival_causes=causes,
            max_rollout_days=100.0,
            timeline_time_unit=timeline_time_unit,
            model_version="test-model-v1",
        )
    )
    return model.eval()


def _trajectory_inputs(*, requires_grad: bool = False) -> dict[str, object]:
    batch = 2
    ct0_values = torch.randn(batch, 5, 8, requires_grad=requires_grad)
    clinical_values = torch.randn(batch, 3, 4, requires_grad=requires_grad)
    ct1_values = torch.randn(batch, 4, 8, requires_grad=requires_grad)
    path_values = torch.randn(batch, 3, 6, requires_grad=requires_grad)
    ct_coords = torch.rand(batch, 5, 3)
    path_coords = torch.rand(batch, 3, 2)
    treatment_values = torch.randn(batch, 2, 5, requires_grad=requires_grad)
    surgery_values = torch.randn(batch, 1, 5, requires_grad=requires_grad)
    return {
        "ct0": _observation("ct", ct0_values, acquired=0, available=0, coords=ct_coords),
        "clinical0": _observation("clinical", clinical_values, acquired=0, available=0),
        "s0_time": torch.zeros(batch),
        "treatment_actions": _actions(
            treatment_values, event_time=10, available_time=10, event_type=1
        ),
        "ct1_acquisition_time": torch.full((batch,), 30.0),
        "ct1": _observation("ct", ct1_values, acquired=30, available=35),
        "s1_time": torch.full((batch,), 35.0),
        "surgery_actions": _actions(surgery_values, event_time=50, available_time=50, event_type=7),
        "pathology_acquisition_time": torch.full((batch,), 60.0),
        "pathology": _observation(
            "pathology", path_values, acquired=60, available=65, coords=path_coords
        ),
        "s2_time": torch.full((batch,), 65.0),
        "horizons": torch.tensor([0.0, 1.0, 2.0, 4.0]),
        "deterministic": True,
    }


def test_main_model_has_distinct_transition_update_and_decoders() -> None:
    model = _model()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert parameter_count > 25_000
    assert model.transition is not model.updater
    assert set(model.future_decoders) == {"ct", "pathology"}


def test_three_stage_shapes_and_survival_monotonicity() -> None:
    output = _model().forward_three_stage(**_trajectory_inputs())
    assert output.state_s0.memory.shape == (2, 4, 16)
    assert output.future_ct.mean.shape == (2, 1, 8)
    assert output.future_pathology.mean.shape == (2, 1, 6)
    for prediction in (output.survival_s0, output.survival_s1, output.survival_s2):
        assert prediction.survival.shape == (2, 4)
        assert torch.all(prediction.survival[:, 1:] <= prediction.survival[:, :-1])
        assert torch.allclose(prediction.survival[:, 0], torch.ones(2))
        assert torch.all((prediction.risk >= 0) & (prediction.risk <= 1))


def test_three_stage_advances_from_acquisition_to_recorded_availability() -> None:
    model = _model(stochastic=False)
    inputs = _trajectory_inputs()
    inputs["s1_time"] = torch.full((2,), 40.0)
    inputs["s2_time"] = torch.full((2,), 80.0)

    output = model.forward_three_stage(**inputs)

    assert torch.equal(output.prior_ct1.query_time, torch.full((2,), 30.0))
    assert torch.equal(output.pre_ct1_update.query_time, torch.full((2,), 35.0))
    assert torch.equal(output.post_ct1_update.query_time, torch.full((2,), 35.0))
    assert torch.equal(output.state_s1.query_time, torch.full((2,), 40.0))
    assert torch.equal(output.prior_pathology.query_time, torch.full((2,), 60.0))
    assert torch.equal(output.pre_pathology_update.query_time, torch.full((2,), 65.0))
    assert torch.equal(output.post_pathology_update.query_time, torch.full((2,), 65.0))
    assert torch.equal(output.state_s2.query_time, torch.full((2,), 80.0))
    assert not torch.equal(output.prior_ct1.memory, output.pre_ct1_update.memory)
    assert not torch.equal(output.prior_pathology.memory, output.pre_pathology_update.memory)


def test_three_stage_rejects_declared_availability_that_differs_from_observation() -> None:
    inputs = _trajectory_inputs()
    inputs["s1_time"] = torch.full((2,), 40.0)
    inputs["ct1_availability_time"] = torch.full((2,), 36.0)
    with pytest.raises(DataContractError) as error:
        _model(stochastic=False).forward_three_stage(**inputs)
    assert error.value.code == "OBSERVATION_AVAILABILITY_TIME_MISMATCH"


def test_posterior_update_requires_state_at_observation_availability() -> None:
    model = _model(stochastic=False)
    inputs = _trajectory_inputs()
    state = model.initialize(
        [inputs["ct0"]], inputs["clinical0"], inputs["s0_time"], deterministic=True
    )
    with pytest.raises(DataContractError) as error:
        model.update_posterior(state, [inputs["ct1"]], inputs["s1_time"], deterministic=True)
    assert error.value.code == "UPDATE_STATE_TIME_MISMATCH"


def test_timeline_day_and_year_units_encode_the_same_physical_times() -> None:
    torch.manual_seed(23)
    day_model = _model(stochastic=False, timeline_time_unit="day")
    year_model = _model(stochastic=False, timeline_time_unit="year")
    year_model.load_state_dict(day_model.state_dict())

    baseline_values = torch.randn(2, 3, 8)
    day_baseline = _observation(
        "ct", baseline_values, acquired=100.0, available=110.0
    )
    day_action = _actions(
        torch.randn(2, 1, 5), event_time=200.0, available_time=200.0, event_type=1
    )
    days_per_year = 365.25
    year_baseline = replace(
        day_baseline,
        acquired_time=day_baseline.acquired_time / days_per_year,
        available_time=day_baseline.available_time / days_per_year,
    )
    year_action = replace(
        day_action,
        event_time=day_action.event_time / days_per_year,
        available_time=day_action.available_time / days_per_year,
    )

    day_state = day_model.initialize(
        [day_baseline], None, torch.full((2,), 110.0), deterministic=True
    )
    year_state = year_model.initialize(
        [year_baseline], None, torch.full((2,), 110.0 / days_per_year), deterministic=True
    )
    day_prior = day_model.predict_prior(
        day_state, day_action, torch.full((2,), 365.25), deterministic=True
    )
    year_prior = year_model.predict_prior(
        year_state, year_action, torch.full((2,), 1.0), deterministic=True
    )

    assert torch.allclose(day_state.memory, year_state.memory, atol=1e-6)
    assert torch.allclose(day_prior.memory, year_prior.memory, atol=1e-6)
    day_prediction = day_model.predict_survival(
        day_prior, "os", torch.tensor([0.0, 1.0]), stage="S1"
    )
    year_prediction = year_model.predict_survival(
        year_prior, "os", torch.tensor([0.0, 1.0]), stage="S1"
    )
    assert torch.allclose(day_prediction.rates, year_prediction.rates, atol=1e-6)


def test_full_three_stage_day_and_year_timeline_units_are_equivalent() -> None:
    day_model = _model(stochastic=False, timeline_time_unit="day")
    year_model = _model(stochastic=False, timeline_time_unit="year")
    year_model.load_state_dict(day_model.state_dict())
    day_inputs = _trajectory_inputs()
    days_per_year = day_model.config.days_per_year
    timeline_fields = {
        "s0_time",
        "ct1_acquisition_time",
        "ct1_availability_time",
        "s1_time",
        "pathology_acquisition_time",
        "pathology_availability_time",
        "s2_time",
    }
    year_inputs: dict[str, object] = {}
    for name, value in day_inputs.items():
        if isinstance(value, ObservationTokens):
            year_inputs[name] = replace(
                value,
                acquired_time=value.acquired_time / days_per_year,
                available_time=value.available_time / days_per_year,
            )
        elif isinstance(value, ActionTokens):
            year_inputs[name] = replace(
                value,
                event_time=value.event_time / days_per_year,
                available_time=value.available_time / days_per_year,
            )
        elif name in timeline_fields:
            assert isinstance(value, torch.Tensor)
            year_inputs[name] = value / days_per_year
        else:
            year_inputs[name] = value

    day_output = day_model.forward_three_stage(**day_inputs)
    year_output = year_model.forward_three_stage(**year_inputs)

    for field in (
        "state_s0",
        "prior_ct1",
        "pre_ct1_update",
        "post_ct1_update",
        "state_s1",
        "prior_pathology",
        "pre_pathology_update",
        "post_pathology_update",
        "state_s2",
    ):
        day_state = getattr(day_output, field)
        year_state = getattr(year_output, field)
        assert torch.allclose(day_state.memory, year_state.memory, atol=2e-6)
        assert torch.allclose(
            day_state.query_time,
            year_state.query_time * days_per_year,
            atol=1e-5,
        )
    for field in ("survival_s0", "survival_s1", "survival_s2"):
        assert torch.allclose(
            getattr(day_output, field).rates,
            getattr(year_output, field).rates,
            atol=2e-6,
        )


def test_future_tensors_are_structurally_disconnected_from_early_risk() -> None:
    inputs = _trajectory_inputs(requires_grad=True)
    output = _model().forward_three_stage(**inputs)
    ct0 = inputs["ct0"]
    ct1 = inputs["ct1"]
    pathology = inputs["pathology"]
    treatment = inputs["treatment_actions"]
    surgery = inputs["surgery_actions"]
    assert isinstance(ct0, ObservationTokens)
    assert isinstance(ct1, ObservationTokens)
    assert isinstance(pathology, ObservationTokens)
    assert isinstance(treatment, ActionTokens)
    assert isinstance(surgery, ActionTokens)
    early_gradients = torch.autograd.grad(
        output.survival_s0.risk.sum(),
        (ct1.values, pathology.values, treatment.values, surgery.values),
        allow_unused=True,
        retain_graph=True,
    )
    assert early_gradients == (None, None, None, None)
    preop_gradients = torch.autograd.grad(
        output.survival_s1.risk.sum(),
        (pathology.values, surgery.values),
        allow_unused=True,
        retain_graph=True,
    )
    assert preop_gradients == (None, None)
    visible_gradient = torch.autograd.grad(output.survival_s0.risk.sum(), ct0.values)[0]
    assert visible_gradient is not None and visible_gradient.abs().sum() > 0


def test_s1_risk_is_numerically_invariant_to_surgery_and_pathology() -> None:
    model = _model(stochastic=False)
    inputs = _trajectory_inputs()
    reference = model.forward_three_stage(**inputs)
    pathology = inputs["pathology"]
    surgery = inputs["surgery_actions"]
    assert isinstance(pathology, ObservationTokens)
    assert isinstance(surgery, ActionTokens)
    changed = {
        **inputs,
        "pathology": replace(pathology, values=pathology.values + 100.0),
        "surgery_actions": replace(surgery, values=surgery.values - 100.0),
    }

    perturbed = model.forward_three_stage(**changed)

    assert torch.equal(reference.survival_s1.rates, perturbed.survival_s1.rates)
    assert torch.equal(reference.survival_s1.risk, perturbed.survival_s1.risk)
    assert not torch.allclose(reference.survival_s2.rates, perturbed.survival_s2.rates)


def test_action_content_type_and_time_change_prior() -> None:
    model = _model(stochastic=False)
    inputs = _trajectory_inputs()
    state = model.initialize(
        [inputs["ct0"]], inputs["clinical0"], inputs["s0_time"], deterministic=True
    )
    action = inputs["treatment_actions"]
    assert isinstance(action, ActionTokens)
    prior = model.predict_prior(state, action, inputs["ct1_acquisition_time"], deterministic=True)
    changed = ActionTokens(
        values=action.values + 2.0,
        valid=action.valid,
        event_time=action.event_time + 3.0,
        available_time=action.available_time + 3.0,
        event_type=action.event_type + 6 if action.event_type is not None else None,
        planned_or_delivered=action.planned_or_delivered,
        known_exposure=action.known_exposure,
    )
    changed_prior = model.predict_prior(
        state, changed, inputs["ct1_acquisition_time"], deterministic=True
    )
    assert not torch.allclose(prior.memory, changed_prior.memory)


def test_current_observation_changes_posterior_but_padding_does_not() -> None:
    model = _model(stochastic=False)
    inputs = _trajectory_inputs()
    state = model.initialize(
        [inputs["ct0"]], inputs["clinical0"], inputs["s0_time"], deterministic=True
    )
    action = inputs["treatment_actions"]
    assert isinstance(action, ActionTokens)
    prior = model.predict_prior(state, action, inputs["ct1_acquisition_time"], deterministic=True)
    values = torch.randn(2, 3, 8)
    valid = torch.tensor([[True, True, False], [True, False, False]])
    first = _observation("ct", values, acquired=30, available=35, valid=valid)
    changed_padding = values.clone()
    changed_padding[~valid] = 10_000.0
    second = _observation("ct", changed_padding, acquired=30, available=35, valid=valid)
    empty = ActionTokens.empty(batch_size=2, value_dim=5, device=torch.device("cpu"))
    pre_update = model.predict_prior(prior, empty, inputs["s1_time"], deterministic=True)
    first_state = model.update_posterior(
        pre_update, [first], inputs["s1_time"], deterministic=True
    )
    second_state = model.update_posterior(
        pre_update, [second], inputs["s1_time"], deterministic=True
    )
    assert torch.allclose(first_state.memory, second_state.memory, atol=1e-6)
    visible_change = values.clone()
    visible_change[valid] += 5.0
    third = _observation("ct", visible_change, acquired=30, available=35, valid=valid)
    third_state = model.update_posterior(
        pre_update, [third], inputs["s1_time"], deterministic=True
    )
    assert not torch.allclose(first_state.memory, third_state.memory)


def test_empty_observation_preserves_prior() -> None:
    model = _model(stochastic=False)
    inputs = _trajectory_inputs()
    state = model.initialize(
        [inputs["ct0"]], inputs["clinical0"], inputs["s0_time"], deterministic=True
    )
    empty = _observation(
        "ct",
        torch.randn(2, 2, 8),
        acquired=30,
        available=35,
        valid=torch.zeros(2, 2, dtype=torch.bool),
    )
    empty_actions = ActionTokens.empty(batch_size=2, value_dim=5, device=torch.device("cpu"))
    advanced = model.predict_prior(
        state, empty_actions, torch.full((2,), 35.0), deterministic=True
    )
    result = model.update_posterior(
        advanced, [empty], torch.full((2,), 35.0), deterministic=True
    )
    assert torch.equal(result.memory, advanced.memory)
    assert torch.equal(result.query_time, torch.full((2,), 35.0))
    assert "empty_observation_update" in result.quality_flags


def test_pathology_set_is_permutation_invariant() -> None:
    model = _model(stochastic=False)
    inputs = _trajectory_inputs()
    output = model.forward_three_stage(**inputs)
    pathology = inputs["pathology"]
    assert isinstance(pathology, ObservationTokens)
    permutation = torch.tensor([2, 0, 1])
    permuted = ObservationTokens(
        values=pathology.values[:, permutation],
        valid=pathology.valid[:, permutation],
        modality=pathology.modality[:, permutation],
        acquired_time=pathology.acquired_time[:, permutation],
        available_time=pathology.available_time[:, permutation],
        provenance=pathology.provenance,
        source_id=tuple(
            tuple(row[index] for index in permutation.tolist()) for row in pathology.source_id
        ),
        modality_name="pathology",
        coords=pathology.coords[:, permutation] if pathology.coords is not None else None,
        coordinate_system=pathology.coordinate_system,
    )
    permuted_state = model.update_posterior(
        output.pre_pathology_update, [permuted], inputs["s2_time"], deterministic=True
    )
    assert torch.allclose(output.state_s2.memory, permuted_state.memory, atol=1e-6)


def test_ct_set_is_permutation_invariant_when_geometry_moves_with_tokens() -> None:
    model = _model(stochastic=False)
    inputs = _trajectory_inputs()
    output = model.forward_three_stage(**inputs)
    ct1 = inputs["ct1"]
    assert isinstance(ct1, ObservationTokens)
    permutation = torch.tensor([3, 1, 0, 2])
    permuted = ObservationTokens(
        values=ct1.values[:, permutation],
        valid=ct1.valid[:, permutation],
        modality=ct1.modality[:, permutation],
        acquired_time=ct1.acquired_time[:, permutation],
        available_time=ct1.available_time[:, permutation],
        provenance=ct1.provenance,
        source_id=tuple(
            tuple(row[index] for index in permutation.tolist()) for row in ct1.source_id
        ),
        modality_name="ct",
        coords=ct1.coords[:, permutation] if ct1.coords is not None else None,
        coordinate_system=ct1.coordinate_system,
    )
    permuted_state = model.update_posterior(
        output.pre_ct1_update, [permuted], inputs["s1_time"], deterministic=True
    )

    assert torch.allclose(output.post_ct1_update.memory, permuted_state.memory, atol=1e-6)


def test_stage_conditioning_does_not_impose_cross_stage_risk_monotonicity() -> None:
    model = _model(stochastic=False)
    inputs = _trajectory_inputs()
    state = model.initialize(
        [inputs["ct0"]], inputs["clinical0"], inputs["s0_time"], deterministic=True
    )
    horizons = torch.tensor([0.0, 1.0, 2.0, 4.0])
    with torch.no_grad():
        embedding = model.survival_decoder.stage_embedding.weight
        embedding[0].copy_(torch.linspace(-2.0, 2.0, embedding.shape[1]))
        embedding[1].copy_(torch.linspace(2.0, -2.0, embedding.shape[1]))
        first_s0 = model.predict_survival(state, "os", horizons, stage="S0").rates
        first_s1 = model.predict_survival(state, "os", horizons, stage="S1").rates
        swapped = embedding[[1, 0]].clone()
        embedding[0].copy_(swapped[0])
        embedding[1].copy_(swapped[1])
        second_s0 = model.predict_survival(state, "os", horizons, stage="S0").rates
        second_s1 = model.predict_survival(state, "os", horizons, stage="S1").rates

    assert not torch.allclose(first_s0, first_s1)
    assert torch.allclose(first_s0, second_s1)
    assert torch.allclose(first_s1, second_s0)


def test_stochastic_state_is_reproducible_with_fixed_generator() -> None:
    model = _model(stochastic=True)
    inputs = _trajectory_inputs()
    first_generator = torch.Generator().manual_seed(19)
    second_generator = torch.Generator().manual_seed(19)
    first = model.initialize(
        [inputs["ct0"]],
        inputs["clinical0"],
        inputs["s0_time"],
        generator=first_generator,
    )
    second = model.initialize(
        [inputs["ct0"]],
        inputs["clinical0"],
        inputs["s0_time"],
        generator=second_generator,
    )
    assert first.sample is not None and second.sample is not None
    assert torch.equal(first.sample, second.sample)


def test_long_rollout_is_flagged_and_future_action_rejected() -> None:
    model = _model(stochastic=False)
    inputs = _trajectory_inputs()
    state = model.initialize(
        [inputs["ct0"]], inputs["clinical0"], inputs["s0_time"], deterministic=True
    )
    empty = ActionTokens.empty(batch_size=2, value_dim=5, device=torch.device("cpu"))
    long_state = model.predict_prior(state, empty, torch.full((2,), 200.0), deterministic=True)
    assert "rollout_out_of_range" in long_state.quality_flags
    future_action = _actions(torch.randn(2, 1, 5), event_time=300, available_time=300, event_type=1)
    try:
        model.predict_prior(state, future_action, torch.full((2,), 30.0), deterministic=True)
    except DataContractError as error:
        assert error.code == "FUTURE_ACTION_IN_PRIOR"
    else:
        raise AssertionError("future action was accepted")


def test_future_observation_provenance_is_not_presented_as_observed() -> None:
    output = _model(stochastic=False).forward_three_stage(**_trajectory_inputs())
    assert output.future_ct.provenance == "predicted_not_observed"
    assert output.future_pathology.provenance == "predicted_not_observed"
    assert "training_pair" in output.future_ct.scenario
    assert "training_pair" in output.future_pathology.scenario
    assert all(source.startswith("test-") for source in output.state_s2.provenance)


def test_competing_risk_decoder_conserves_probability() -> None:
    model = _model(stochastic=False, causes=2)
    inputs = _trajectory_inputs()
    state = model.initialize(
        [inputs["ct0"]], inputs["clinical0"], inputs["s0_time"], deterministic=True
    )
    prediction = model.predict_survival(state, "os", torch.tensor([0.0, 0.5, 2.0, 4.0]), stage="S0")
    assert prediction.cif is not None
    assert torch.allclose(
        prediction.survival + prediction.cif.sum(dim=-1),
        torch.ones_like(prediction.survival),
        atol=1e-5,
    )


def test_future_predictions_are_computed_before_target_observations() -> None:
    inputs = _trajectory_inputs(requires_grad=True)
    output = _model().forward_three_stage(**inputs)
    ct1 = inputs["ct1"]
    pathology = inputs["pathology"]
    treatment = inputs["treatment_actions"]
    surgery = inputs["surgery_actions"]
    assert isinstance(ct1, ObservationTokens)
    assert isinstance(pathology, ObservationTokens)
    assert isinstance(treatment, ActionTokens)
    assert isinstance(surgery, ActionTokens)

    ct_gradients = torch.autograd.grad(
        output.future_ct.mean.sum(),
        (ct1.values, treatment.values),
        allow_unused=True,
        retain_graph=True,
    )
    assert ct_gradients[0] is None
    assert ct_gradients[1] is not None and ct_gradients[1].abs().sum() > 0

    pathology_gradients = torch.autograd.grad(
        output.future_pathology.mean.sum(),
        (pathology.values, surgery.values, ct1.values),
        allow_unused=True,
    )
    assert pathology_gradients[0] is None
    assert pathology_gradients[1] is not None and pathology_gradients[1].abs().sum() > 0
    assert pathology_gradients[2] is not None and pathology_gradients[2].abs().sum() > 0


@pytest.mark.parametrize("stochastic", [False, True])
def test_mixed_batch_empty_update_preserves_advanced_missing_patient(stochastic: bool) -> None:
    model = _model(stochastic=stochastic)
    inputs = _trajectory_inputs()
    state = model.initialize(
        [inputs["ct0"]], inputs["clinical0"], inputs["s0_time"], deterministic=True
    )
    action = inputs["treatment_actions"]
    assert isinstance(action, ActionTokens)
    prior = model.predict_prior(state, action, inputs["ct1_acquisition_time"], deterministic=True)
    empty = ActionTokens.empty(batch_size=2, value_dim=5, device=torch.device("cpu"))
    pre_update = model.predict_prior(prior, empty, inputs["s1_time"], deterministic=True)
    valid = torch.tensor([[True, True], [False, False]])
    observation = _observation("ct", torch.randn(2, 2, 8), acquired=30, available=35, valid=valid)
    result = model.update_posterior(
        pre_update, [observation], inputs["s1_time"], deterministic=True
    )

    assert not torch.equal(result.memory[0], pre_update.memory[0])
    assert torch.equal(result.memory[1], pre_update.memory[1])
    assert result.query_time[0].item() == 35.0
    assert result.query_time[1].item() == 35.0
    assert result.state_kind == "mixed"
    assert "partial_empty_observation_update" in result.quality_flags
    if stochastic:
        assert result.stochastic_mean is not None and pre_update.stochastic_mean is not None
        assert result.stochastic_log_std is not None and pre_update.stochastic_log_std is not None
        assert result.sample is not None and pre_update.sample is not None
        assert torch.equal(result.stochastic_mean[1], pre_update.stochastic_mean[1])
        assert torch.equal(result.stochastic_log_std[1], pre_update.stochastic_log_std[1])
        assert torch.equal(result.sample[1], pre_update.sample[1])


def test_mixed_missing_row_is_not_transitioned_twice_at_stage_query() -> None:
    model = _model(stochastic=False)
    inputs = _trajectory_inputs()
    ct1 = inputs["ct1"]
    assert isinstance(ct1, ObservationTokens)
    inputs["ct1"] = replace(
        ct1,
        valid=torch.tensor(
            [[True] * ct1.token_count, [False] * ct1.token_count], dtype=torch.bool
        ),
    )
    inputs["s1_time"] = torch.full((2,), 40.0)

    output = model.forward_three_stage(**inputs)

    assert torch.equal(output.pre_ct1_update.query_time, torch.tensor([35.0, 40.0]))
    assert torch.equal(output.post_ct1_update.query_time, torch.tensor([35.0, 40.0]))
    assert not torch.equal(output.state_s1.memory[0], output.post_ct1_update.memory[0])
    assert torch.equal(output.state_s1.memory[1], output.post_ct1_update.memory[1])


def test_stage_clinical_observation_resolves_update_time_when_ct_is_missing() -> None:
    model = _model(stochastic=False)
    inputs = _trajectory_inputs()
    ct1 = inputs["ct1"]
    assert isinstance(ct1, ObservationTokens)
    inputs["ct1"] = replace(ct1, valid=torch.zeros_like(ct1.valid))
    inputs["clinical1"] = _observation(
        "clinical",
        torch.randn(2, 2, 4),
        acquired=30,
        available=35,
    )
    inputs["s1_time"] = torch.full((2,), 40.0)

    output = model.forward_three_stage(**inputs)

    assert torch.equal(output.pre_ct1_update.query_time, torch.full((2,), 35.0))
    assert not torch.equal(output.post_ct1_update.memory, output.pre_ct1_update.memory)
    assert torch.equal(output.state_s1.query_time, torch.full((2,), 40.0))


def test_missing_stage_observations_still_advance_to_query_times() -> None:
    model = _model(stochastic=False)
    inputs = _trajectory_inputs()
    ct1 = inputs["ct1"]
    pathology = inputs["pathology"]
    assert isinstance(ct1, ObservationTokens)
    assert isinstance(pathology, ObservationTokens)
    inputs["ct1"] = replace(ct1, valid=torch.zeros_like(ct1.valid))
    inputs["pathology"] = replace(pathology, valid=torch.zeros_like(pathology.valid))
    inputs["s1_time"] = torch.full((2,), 40.0)
    inputs["s2_time"] = torch.full((2,), 80.0)

    output = model.forward_three_stage(**inputs)

    assert torch.equal(output.pre_ct1_update.query_time, torch.full((2,), 40.0))
    assert torch.equal(output.state_s1.query_time, torch.full((2,), 40.0))
    assert torch.equal(output.pre_pathology_update.query_time, torch.full((2,), 80.0))
    assert torch.equal(output.state_s2.query_time, torch.full((2,), 80.0))
    assert torch.equal(output.post_ct1_update.memory, output.pre_ct1_update.memory)
    assert torch.equal(output.post_pathology_update.memory, output.pre_pathology_update.memory)
    assert "empty_observation_update" in output.state_s1.quality_flags
    assert "empty_observation_update" in output.state_s2.quality_flags


def test_post_ct_action_changes_s1_but_not_ct_acquisition_prior() -> None:
    model = _model(stochastic=False)
    inputs = _trajectory_inputs()
    without_bridge = model.forward_three_stage(**inputs)
    bridge = _actions(
        torch.full((2, 1, 5), 2.0),
        event_time=32.0,
        available_time=32.0,
        event_type=5,
    )
    with_bridge = model.forward_three_stage(**inputs, s1_update_actions=bridge)

    assert torch.equal(without_bridge.prior_ct1.memory, with_bridge.prior_ct1.memory)
    assert not torch.allclose(
        without_bridge.pre_ct1_update.memory, with_bridge.pre_ct1_update.memory
    )
    assert not torch.allclose(without_bridge.state_s1.memory, with_bridge.state_s1.memory)


def test_stochastic_sample_conditions_next_transition() -> None:
    model = _model(stochastic=True)
    inputs = _trajectory_inputs()
    state = model.initialize(
        [inputs["ct0"]], inputs["clinical0"], inputs["s0_time"], deterministic=True
    )
    assert state.sample is not None
    action = inputs["treatment_actions"]
    assert isinstance(action, ActionTokens)
    changed_state = replace(state, sample=state.sample + 5.0)
    first = model.predict_prior(state, action, inputs["ct1_acquisition_time"], deterministic=True)
    second = model.predict_prior(
        changed_state, action, inputs["ct1_acquisition_time"], deterministic=True
    )
    assert not torch.allclose(first.memory, second.memory)


def test_action_time_assignment_changes_transition_but_token_order_does_not() -> None:
    model = _model(stochastic=False)
    inputs = _trajectory_inputs()
    state = model.initialize(
        [inputs["ct0"]], inputs["clinical0"], inputs["s0_time"], deterministic=True
    )
    values = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0, 0.0]],
        ]
    )
    times = torch.tensor([[5.0, 15.0], [5.0, 15.0]])
    action = ActionTokens(
        values=values,
        valid=torch.ones(2, 2, dtype=torch.bool),
        event_time=times,
        available_time=times,
        event_type=torch.tensor([[1, 2], [1, 2]]),
        planned_or_delivered=torch.full((2, 2), 2),
        known_exposure=torch.ones(2, 2),
    )
    reassigned = replace(
        action,
        event_time=times.flip(1),
        available_time=times.flip(1),
    )
    permuted = ActionTokens(
        values=values.flip(1),
        valid=action.valid.flip(1),
        event_time=times.flip(1),
        available_time=times.flip(1),
        event_type=action.event_type.flip(1) if action.event_type is not None else None,
        planned_or_delivered=(
            action.planned_or_delivered.flip(1) if action.planned_or_delivered is not None else None
        ),
        known_exposure=(
            action.known_exposure.flip(1) if action.known_exposure is not None else None
        ),
    )
    target = inputs["ct1_acquisition_time"]
    original_prior = model.predict_prior(state, action, target, deterministic=True)
    reassigned_prior = model.predict_prior(state, reassigned, target, deterministic=True)
    permuted_prior = model.predict_prior(state, permuted, target, deterministic=True)
    assert not torch.allclose(original_prior.memory, reassigned_prior.memory)
    assert torch.allclose(original_prior.memory, permuted_prior.memory, atol=1e-6)


def test_full_stochastic_trajectory_is_reproducible_with_fixed_generator() -> None:
    model = _model(stochastic=True)
    inputs = _trajectory_inputs()
    inputs["deterministic"] = False
    first = model.forward_three_stage(**inputs, generator=torch.Generator().manual_seed(71))
    second = model.forward_three_stage(**inputs, generator=torch.Generator().manual_seed(71))
    different_seed = model.forward_three_stage(
        **inputs, generator=torch.Generator().manual_seed(72)
    )
    assert first.state_s0.sample is not None and second.state_s0.sample is not None
    assert first.prior_ct1.sample is not None and second.prior_ct1.sample is not None
    assert first.state_s2.sample is not None and second.state_s2.sample is not None
    assert different_seed.state_s0.sample is not None
    assert torch.equal(first.state_s0.sample, second.state_s0.sample)
    assert torch.equal(first.prior_ct1.sample, second.prior_ct1.sample)
    assert torch.equal(first.state_s2.sample, second.state_s2.sample)
    assert torch.equal(first.survival_s2.risk, second.survival_s2.risk)
    assert not torch.equal(first.state_s0.sample, different_seed.state_s0.sample)
    assert not torch.equal(first.survival_s2.risk, different_seed.survival_s2.risk)


def test_full_deterministic_trajectory_is_stable_across_rng_states() -> None:
    model = _model(stochastic=True)
    inputs = _trajectory_inputs()
    inputs["deterministic"] = True

    first = model.forward_three_stage(
        **inputs,
        generator=torch.Generator().manual_seed(3),
    )
    torch.manual_seed(999)
    _ = torch.randn(100)
    second = model.forward_three_stage(
        **inputs,
        generator=torch.Generator().manual_seed(97),
    )

    for first_state, second_state in (
        (first.state_s0, second.state_s0),
        (first.prior_ct1, second.prior_ct1),
        (first.state_s1, second.state_s1),
        (first.prior_pathology, second.prior_pathology),
        (first.state_s2, second.state_s2),
    ):
        assert first_state.sample is not None and second_state.sample is not None
        assert first_state.stochastic_mean is not None
        assert second_state.stochastic_mean is not None
        assert torch.equal(first_state.sample, first_state.stochastic_mean)
        assert torch.equal(second_state.sample, second_state.stochastic_mean)
        assert torch.equal(first_state.memory, second_state.memory)
        assert torch.equal(first_state.stochastic_mean, second_state.stochastic_mean)
        assert torch.equal(first_state.sample, second_state.sample)
    assert torch.equal(first.future_ct.mean, second.future_ct.mean)
    assert torch.equal(first.future_pathology.mean, second.future_pathology.mean)
    assert torch.equal(first.survival_s0.risk, second.survival_s0.risk)
    assert torch.equal(first.survival_s1.risk, second.survival_s1.risk)
    assert torch.equal(first.survival_s2.risk, second.survival_s2.risk)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"attention_heads": 0}, "hidden_dim"),
        ({"survival_cutpoints": (0.0, 1.0, 1.0)}, "survival_cutpoints"),
        ({"transition_blocks": 0}, "block counts"),
        ({"resampled_tokens": (("ct", 3),)}, "resampled_tokens"),
        ({"future_output_tokens": (("ct", 1),)}, "future output"),
    ],
)
def test_invalid_model_configurations_fail_early(
    overrides: dict[str, object], message: str
) -> None:
    config = _model(stochastic=False).config
    with pytest.raises(ValueError, match=message):
        replace(config, **overrides)


def test_model_contracts_reject_ambiguous_or_nonfinite_inputs() -> None:
    model = _model(stochastic=False)
    inputs = _trajectory_inputs()
    ct0 = inputs["ct0"]
    assert isinstance(ct0, ObservationTokens)

    zero_token = _observation("ct", torch.empty(2, 0, 8), acquired=0, available=0)
    with pytest.raises(DataContractError) as zero_error:
        model.initialize([zero_token], None, inputs["s0_time"], deterministic=True)
    assert zero_error.value.code == "EMPTY_OBSERVATION_TOKEN_AXIS"

    wrong_modality = replace(ct0, modality=torch.ones_like(ct0.modality))
    with pytest.raises(DataContractError) as modality_error:
        model.initialize([wrong_modality], None, inputs["s0_time"], deterministic=True)
    assert modality_error.value.code == "OBSERVATION_MODALITY_MISMATCH"

    with pytest.raises(DataContractError) as time_error:
        model.initialize([ct0], None, torch.tensor([0.0, float("nan")]), deterministic=True)
    assert time_error.value.code == "NONFINITE_TIME"

    state = model.initialize([ct0], None, inputs["s0_time"], deterministic=True)
    bad_action = ActionTokens(
        values=torch.zeros(2, 1, 5),
        valid=torch.ones(2, 1, dtype=torch.bool),
        event_time=torch.tensor([[float("nan")], [1.0]]),
        available_time=torch.zeros(2, 1),
    )
    with pytest.raises(DataContractError) as action_error:
        model.predict_prior(state, bad_action, torch.full((2,), 3.0), deterministic=True)
    assert action_error.value.code == "NONFINITE_ACTION_METADATA"

    with pytest.raises(DataContractError) as forecast_error:
        model.predict_future_observation(
            state,
            "ct",
            scenario="test",
            target_time=torch.full((2,), 10.0),
        )
    assert forecast_error.value.code == "FUTURE_TARGET_REQUIRES_PRIOR"

    mismatched_inputs = _trajectory_inputs()
    mismatched_inputs["ct1_acquisition_time"] = torch.full((2,), 29.0)
    with pytest.raises(DataContractError) as target_error:
        model.forward_three_stage(**mismatched_inputs)
    assert target_error.value.code == "FUTURE_TARGET_TIME_MISMATCH"
