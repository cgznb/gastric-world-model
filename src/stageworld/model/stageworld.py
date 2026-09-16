"""StageWorld-GC model with explicit prior prediction and observation update."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from stageworld.encoders.base import ObservationTokens
from stageworld.errors import DataContractError
from stageworld.survival import competing_risk_curves, risk_probability, survival_probability

from .components import (
    ContinuousTimeEncoder,
    FutureObservationDecoder,
    GaussianStateHead,
    ObservationUpdater,
    SharedSurvivalDecoder,
    StateInitializer,
    StateTransition,
    TokenResampler,
)
from .types import ActionTokens, BeliefState, PredictionDistribution, StagePrediction


@dataclass(frozen=True)
class StageWorldModelConfig:
    hidden_dim: int = 256
    state_tokens: int = 24
    stochastic_dim: int = 16
    use_stochastic_state: bool = True
    attention_heads: int = 8
    transition_blocks: int = 4
    observation_blocks: int = 2
    resampler_blocks: int = 2
    dropout: float = 0.1
    action_input_dim: int = 8
    modality_input_dims: tuple[tuple[str, int], ...] = (
        ("ct", 2048),
        ("pathology", 768),
        ("clinical", 32),
    )
    resampled_tokens: tuple[tuple[str, int], ...] = (
        ("ct", 16),
        ("pathology", 8),
        ("clinical", 8),
    )
    future_output_dims: tuple[tuple[str, int], ...] = (("ct", 2048), ("pathology", 768))
    future_output_tokens: tuple[tuple[str, int], ...] = (("ct", 1), ("pathology", 1))
    survival_cutpoints: tuple[float, ...] = (0.0, 1.0, 3.0, 5.0)
    survival_causes: int = 1
    max_rollout_days: float = 730.5
    timeline_time_unit: str = "day"
    days_per_year: float = 365.25
    model_version: str = "stageworld-gc-0.1"

    def __post_init__(self) -> None:
        if (
            self.attention_heads <= 0
            or self.hidden_dim <= 0
            or self.hidden_dim % self.attention_heads
        ):
            raise ValueError("hidden_dim must be positive and divisible by attention_heads")
        if self.state_tokens <= 0 or self.action_input_dim <= 0:
            raise ValueError("state_tokens and action_input_dim must be positive")
        if self.use_stochastic_state and self.stochastic_dim <= 0:
            raise ValueError("stochastic_dim must be positive when stochastic state is enabled")
        if min(self.transition_blocks, self.observation_blocks, self.resampler_blocks) <= 0:
            raise ValueError("transition, observation, and resampler block counts must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if (
            len(self.survival_cutpoints) < 2
            or self.survival_cutpoints[0] != 0.0
            or any(not math.isfinite(value) for value in self.survival_cutpoints)
            or any(
                right <= left
                for left, right in zip(
                    self.survival_cutpoints,
                    self.survival_cutpoints[1:],
                    strict=False,
                )
            )
        ):
            raise ValueError("survival_cutpoints must contain zero and at least one interval")
        if (
            self.survival_causes <= 0
            or not math.isfinite(self.max_rollout_days)
            or self.max_rollout_days <= 0
        ):
            raise ValueError("survival_causes and max_rollout_days must be positive")
        if self.timeline_time_unit not in {"day", "year"}:
            raise ValueError("timeline_time_unit must be 'day' or 'year'")
        if not math.isfinite(self.days_per_year) or self.days_per_year <= 0:
            raise ValueError("days_per_year must be finite and positive")

        def validate_named_positive(
            values: tuple[tuple[str, int], ...], label: str
        ) -> dict[str, int]:
            names = [name for name, _ in values]
            if not names or len(names) != len(set(names)) or any(not name for name in names):
                raise ValueError(f"{label} must have non-empty unique names")
            if any(value <= 0 for _, value in values):
                raise ValueError(f"{label} values must be positive")
            return dict(values)

        inputs = validate_named_positive(self.modality_input_dims, "modality_input_dims")
        resampled = validate_named_positive(self.resampled_tokens, "resampled_tokens")
        outputs = validate_named_positive(self.future_output_dims, "future_output_dims")
        output_tokens = validate_named_positive(self.future_output_tokens, "future_output_tokens")
        if set(inputs) != set(resampled):
            raise ValueError("resampled_tokens must match configured input modalities")
        if set(outputs) != set(output_tokens) or not set(outputs).issubset(inputs):
            raise ValueError("future output dimensions/tokens must match configured modalities")
        if not self.model_version:
            raise ValueError("model_version must be non-empty")


@dataclass
class ThreeStageOutput:
    state_s0: BeliefState
    prior_ct1: BeliefState
    pre_ct1_update: BeliefState
    post_ct1_update: BeliefState
    future_ct: PredictionDistribution
    state_s1: BeliefState
    prior_pathology: BeliefState
    pre_pathology_update: BeliefState
    post_pathology_update: BeliefState
    future_pathology: PredictionDistribution
    state_s2: BeliefState
    survival_s0: StagePrediction
    survival_s1: StagePrediction
    survival_s2: StagePrediction


class StageWorldModel(nn.Module):
    """A structured token world model, not a static concatenation baseline."""

    survival_cutpoints: Tensor

    stage_to_index = {"S0": 0, "S1": 1, "S2": 2}
    modality_to_index = {"ct": 0, "pathology": 1, "clinical": 2}

    def __init__(self, config: StageWorldModelConfig):
        super().__init__()
        self.config = config
        dim = config.hidden_dim
        self.modality_names = tuple(name for name, _ in config.modality_input_dims)
        if not set(self.modality_names).issubset(self.modality_to_index):
            raise ValueError("modality_input_dims contains an unsupported modality")
        self.modality_index = self.modality_to_index
        self.input_projections = nn.ModuleDict(
            {name: nn.Linear(input_dim, dim) for name, input_dim in config.modality_input_dims}
        )
        token_counts = dict(config.resampled_tokens)
        self.resamplers = nn.ModuleDict(
            {
                name: TokenResampler(
                    dim,
                    token_counts[name],
                    config.attention_heads,
                    config.resampler_blocks,
                    config.dropout,
                )
                for name in self.modality_names
            }
        )
        self.modality_embedding = nn.Embedding(len(self.modality_to_index), dim)
        self.coordinate_projections = nn.ModuleDict(
            {"2": nn.Linear(2, dim), "3": nn.Linear(3, dim)}
        )
        self.observation_time = ContinuousTimeEncoder(dim)
        self.initializer = StateInitializer(
            dim,
            config.state_tokens,
            config.attention_heads,
            config.resampler_blocks,
            config.dropout,
        )
        self.action_projection = nn.Linear(config.action_input_dim, dim)
        self.action_type_embedding = nn.Embedding(16, dim)
        self.action_status_embedding = nn.Embedding(4, dim)
        self.exposure_projection = nn.Linear(1, dim)
        self.action_time = ContinuousTimeEncoder(dim)
        self.transition_time = ContinuousTimeEncoder(dim)
        self.transition = StateTransition(
            dim,
            config.attention_heads,
            config.transition_blocks,
            config.dropout,
        )
        self.updater = ObservationUpdater(
            dim,
            config.attention_heads,
            config.observation_blocks,
            config.dropout,
        )
        if config.use_stochastic_state:
            self.prior_head: GaussianStateHead | None = GaussianStateHead(
                dim, config.stochastic_dim
            )
            self.posterior_head: GaussianStateHead | None = GaussianStateHead(
                dim, config.stochastic_dim
            )
            self.stochastic_projection: nn.Linear | None = nn.Linear(config.stochastic_dim, dim)
        else:
            self.prior_head = None
            self.posterior_head = None
            self.stochastic_projection = None
        output_dims = dict(config.future_output_dims)
        output_tokens = dict(config.future_output_tokens)
        self.future_decoders = nn.ModuleDict(
            {
                modality: FutureObservationDecoder(
                    dim,
                    output_tokens[modality],
                    target_dim,
                    config.attention_heads,
                    config.dropout,
                )
                for modality, target_dim in output_dims.items()
            }
        )
        self.survival_decoder = SharedSurvivalDecoder(
            dim,
            len(config.survival_cutpoints) - 1,
            config.survival_causes,
            config.attention_heads,
            config.dropout,
        )
        self.register_buffer(
            "survival_cutpoints",
            torch.tensor(config.survival_cutpoints, dtype=torch.float32),
            persistent=True,
        )

    def _validate_observation(self, observation: ObservationTokens) -> None:
        observation.validate()
        if observation.modality_name not in self.input_projections:
            raise DataContractError(
                code="UNKNOWN_MODALITY",
                message=f"Unsupported observation modality: {observation.modality_name}",
            )
        expected = self.input_projections[observation.modality_name].in_features
        if observation.values.shape[-1] != expected:
            raise DataContractError(
                code="OBSERVATION_DIMENSION_MISMATCH",
                message=(
                    f"{observation.modality_name} tokens have dimension "
                    f"{observation.values.shape[-1]}; expected {expected}."
                ),
            )
        if observation.values.shape[1] == 0:
            raise DataContractError(
                code="EMPTY_OBSERVATION_TOKEN_AXIS",
                message="Observation tensors must retain at least one explicitly masked token.",
            )
        expected_modality = self.modality_index[observation.modality_name]
        if (
            observation.valid.any()
            and (observation.modality[observation.valid] != expected_modality).any()
        ):
            raise DataContractError(
                code="OBSERVATION_MODALITY_MISMATCH",
                message="Valid modality ids do not match modality_name.",
            )

    def _validate_time(self, name: str, value: Tensor, *, batch: int, device: torch.device) -> None:
        if value.shape != (batch,):
            raise DataContractError(
                code=f"{name.upper()}_SHAPE",
                message=f"{name} must have shape [B].",
            )
        if value.device != device:
            raise DataContractError(
                code="TIME_DEVICE_MISMATCH",
                message=f"{name} must share the state or observation device.",
            )
        if not torch.isfinite(value).all():
            raise DataContractError(
                code="NONFINITE_TIME",
                message=f"{name} must contain only finite values.",
            )

    def _timeline_to_days(self, value: Tensor) -> Tensor:
        """Map configured timeline values to the canonical unit used by time encoders."""

        years = (
            value / self.config.days_per_year
            if self.config.timeline_time_unit == "day"
            else value
        )
        return years * self.config.days_per_year

    def _relative_time_in_days(self, value: Tensor, reference: Tensor) -> Tensor:
        """Subtract canonical clocks so equivalent day/year inputs encode identically."""

        return self._timeline_to_days(value) - self._timeline_to_days(reference)

    def _times_match(self, left: Tensor, right: Tensor) -> Tensor:
        """Compare timeline values at a small, unit-independent day tolerance."""

        comparison_dtype = torch.promote_types(left.dtype, right.dtype)
        if not comparison_dtype.is_floating_point:
            comparison_dtype = torch.float32
        return torch.isclose(
            self._timeline_to_days(left).to(dtype=comparison_dtype),
            self._timeline_to_days(right).to(dtype=comparison_dtype),
            rtol=0.0,
            atol=1e-5,
        )

    def _time_precedes(self, left: Tensor, right: Tensor) -> Tensor:
        """Return whether ``left`` is materially earlier than ``right`` in physical time."""

        return self._timeline_to_days(left) < self._timeline_to_days(right) - 1e-5

    def _validate_state_compatibility(self, state: BeliefState) -> None:
        state.validate()
        expected = (self.config.state_tokens, self.config.hidden_dim)
        if state.memory.shape[1:] != expected:
            raise DataContractError(
                code="STATE_MODEL_SHAPE_MISMATCH",
                message=f"State memory must have trailing shape {expected}.",
            )
        has_stochastic = state.sample is not None
        if has_stochastic != self.config.use_stochastic_state:
            raise DataContractError(
                code="STATE_STOCHASTIC_CONFIG_MISMATCH",
                message="State stochastic tensors do not match the model configuration.",
            )
        if state.sample is not None and state.sample.shape[-1] != self.config.stochastic_dim:
            raise DataContractError(
                code="STATE_MODEL_SHAPE_MISMATCH",
                message="State stochastic dimension does not match the model configuration.",
            )

    def _encode_observation(
        self, observation: ObservationTokens, reference_time: Tensor
    ) -> tuple[Tensor, Tensor]:
        self._validate_observation(observation)
        values = self.input_projections[observation.modality_name](observation.values)
        modality_id = self.modality_index[observation.modality_name]
        values = values + self.modality_embedding.weight[modality_id][None, None, :]
        if observation.coords is not None:
            values = values + self.coordinate_projections[str(observation.coords.shape[-1])](
                observation.coords.to(values)
            )
        relative_acquisition = self._relative_time_in_days(
            observation.acquired_time, reference_time[:, None]
        )
        values = values + self.observation_time(relative_acquisition)
        values = values.masked_fill(~observation.valid.unsqueeze(-1), 0.0)
        return self.resamplers[observation.modality_name](values, observation.valid)

    def _combine_observations(
        self, observations: Sequence[ObservationTokens], reference_time: Tensor
    ) -> tuple[Tensor, Tensor, tuple[str, ...]]:
        if not observations:
            raise DataContractError(
                code="NO_OBSERVATIONS",
                message="At least one observation set is required.",
            )
        encoded: list[Tensor] = []
        valid: list[Tensor] = []
        provenance: list[str] = []
        batch = reference_time.shape[0]
        for observation in observations:
            if observation.values.shape[0] != batch:
                raise DataContractError(
                    code="OBSERVATION_BATCH_MISMATCH",
                    message="All observations and query_time must share batch size.",
                )
            values, mask = self._encode_observation(observation, reference_time)
            encoded.append(values)
            valid.append(mask)
            for row in observation.provenance_ids():
                provenance.extend(row)
        return torch.cat(encoded, dim=1), torch.cat(valid, dim=1), tuple(provenance)

    def _sample_state(
        self,
        memory: Tensor,
        query_time: Tensor,
        *,
        posterior: bool,
        deterministic: bool,
        generator: torch.Generator | None,
        provenance: tuple[str, ...],
        flags: tuple[str, ...] = (),
    ) -> BeliefState:
        head = self.posterior_head if posterior else self.prior_head
        if head is None:
            return BeliefState(
                memory=memory,
                query_time=query_time,
                state_kind="posterior" if posterior else "prior",
                provenance=provenance,
                quality_flags=flags,
            )
        mean, log_std, sample = head(memory, deterministic=deterministic, generator=generator)
        return BeliefState(
            memory=memory,
            query_time=query_time,
            stochastic_mean=mean,
            stochastic_log_std=log_std,
            sample=sample,
            state_kind="posterior" if posterior else "prior",
            provenance=provenance,
            quality_flags=flags,
        )

    def initialize(
        self,
        baseline_observations: Sequence[ObservationTokens],
        baseline_clinical: ObservationTokens | None,
        query_time: Tensor,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> BeliefState:
        observations = list(baseline_observations)
        if baseline_clinical is not None:
            observations.append(baseline_clinical)
        if not observations:
            raise DataContractError(
                code="NO_OBSERVATIONS",
                message="At least one baseline observation set is required.",
            )
        self._validate_time(
            "query_time",
            query_time,
            batch=observations[0].values.shape[0],
            device=observations[0].values.device,
        )
        for observation in observations:
            self._validate_observation(observation)
            if observation.values.shape[0] != query_time.shape[0]:
                raise DataContractError(
                    code="OBSERVATION_BATCH_MISMATCH",
                    message="All observations and query_time must share batch size.",
                )
            if observation.values.device != query_time.device:
                raise DataContractError(
                    code="TIME_DEVICE_MISMATCH",
                    message="query_time and observations must share a device.",
                )
            expected_query = query_time[:, None].expand_as(observation.available_time)
            if (
                observation.valid
                & self._time_precedes(expected_query, observation.available_time)
            ).any():
                raise DataContractError(
                    code="FUTURE_OBSERVATION_AT_INITIALIZATION",
                    message="Initialization received an observation unavailable at query_time.",
                )
        tokens, valid, provenance = self._combine_observations(observations, query_time)
        memory = self.initializer(tokens, valid)
        state = self._sample_state(
            memory,
            query_time,
            posterior=True,
            deterministic=deterministic,
            generator=generator,
            provenance=provenance,
        )
        state.validate()
        return state

    def predict_prior(
        self,
        state: BeliefState,
        known_actions: ActionTokens,
        target_time: Tensor,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> BeliefState:
        self._validate_state_compatibility(state)
        known_actions.validate()
        self._validate_time(
            "target_time",
            target_time,
            batch=state.memory.shape[0],
            device=state.memory.device,
        )
        if known_actions.values.shape[0] != state.memory.shape[0]:
            raise DataContractError(
                code="ACTION_BATCH_MISMATCH",
                message="Actions and state must share batch size.",
            )
        if known_actions.values.shape[-1] != self.config.action_input_dim:
            raise DataContractError(
                code="ACTION_DIMENSION_MISMATCH",
                message="Action feature dimension does not match the model configuration.",
            )
        if known_actions.values.device != state.memory.device:
            raise DataContractError(
                code="ACTION_DEVICE_MISMATCH",
                message="Actions and state must share a device.",
            )
        if self._time_precedes(target_time, state.query_time).any():
            raise DataContractError(
                code="BACKWARD_TRANSITION",
                message="predict_prior cannot move state backward in time.",
            )
        target_grid = target_time[:, None].expand_as(known_actions.event_time)
        forbidden = known_actions.valid & (
            self._time_precedes(target_grid, known_actions.event_time)
            | self._time_precedes(target_grid, known_actions.available_time)
        )
        if forbidden.any():
            raise DataContractError(
                code="FUTURE_ACTION_IN_PRIOR",
                message="Prior transition received an action not known/exposed by target_time.",
            )
        no_op_rows = self._times_match(target_time, state.query_time) & ~known_actions.valid.any(
            dim=1
        )
        if no_op_rows.all():
            return state

        action_values = self.action_projection(known_actions.values)
        if known_actions.event_type is not None:
            action_values = action_values + self.action_type_embedding(
                known_actions.event_type.long().clamp(min=0, max=15)
            )
        if known_actions.planned_or_delivered is not None:
            action_values = action_values + self.action_status_embedding(
                known_actions.planned_or_delivered.long().clamp(min=0, max=3)
            )
        if known_actions.known_exposure is not None:
            action_values = action_values + self.exposure_projection(
                known_actions.known_exposure.to(action_values).unsqueeze(-1)
            )
        relative_event_time = self._relative_time_in_days(
            known_actions.event_time, state.query_time[:, None]
        )
        action_values = action_values + self.action_time(relative_event_time)
        delta_days = self._relative_time_in_days(target_time, state.query_time)
        time_token = self.transition_time(delta_days)
        conditions = torch.cat((action_values, time_token[:, None, :]), dim=1)
        valid = torch.cat(
            (
                known_actions.valid,
                torch.ones(target_time.shape[0], 1, dtype=torch.bool, device=target_time.device),
            ),
            dim=1,
        )
        memory = self.transition(self._state_tokens(state), conditions, valid, time_token)
        flags = state.quality_flags
        if (delta_days > self.config.max_rollout_days).any():
            flags = tuple(dict.fromkeys((*flags, "rollout_out_of_range")))
        result = self._sample_state(
            memory,
            target_time,
            posterior=False,
            deterministic=deterministic,
            generator=generator,
            provenance=(*state.provenance, *known_actions.provenance),
            flags=flags,
        )
        if no_op_rows.any():
            keep = no_op_rows[:, None, None]
            result.memory = torch.where(keep, state.memory, result.memory)
            if result.stochastic_mean is not None:
                assert result.stochastic_log_std is not None
                assert result.sample is not None
                assert state.stochastic_mean is not None
                assert state.stochastic_log_std is not None
                assert state.sample is not None
                result.stochastic_mean = torch.where(
                    keep, state.stochastic_mean, result.stochastic_mean
                )
                result.stochastic_log_std = torch.where(
                    keep, state.stochastic_log_std, result.stochastic_log_std
                )
                result.sample = torch.where(keep, state.sample, result.sample)
            result.state_kind = "mixed"
        result.validate()
        return result

    def update_posterior(
        self,
        state: BeliefState,
        new_observations: Sequence[ObservationTokens],
        availability_time: Tensor,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> BeliefState:
        self._validate_state_compatibility(state)
        self._validate_time(
            "availability_time",
            availability_time,
            batch=state.memory.shape[0],
            device=state.memory.device,
        )
        if self._time_precedes(availability_time, state.query_time).any():
            raise DataContractError(
                code="BACKDATED_UPDATE",
                message="Observation update cannot precede the current state time.",
            )
        if not self._times_match(state.query_time, availability_time).all():
            raise DataContractError(
                code="UPDATE_STATE_TIME_MISMATCH",
                message=(
                    "Advance the prior state to the observation availability time before "
                    "the posterior update."
                ),
            )
        if not new_observations:
            return state.with_flag("no_observation_update")
        for observation in new_observations:
            self._validate_observation(observation)
            if observation.values.shape[0] != state.memory.shape[0]:
                raise DataContractError(
                    code="OBSERVATION_BATCH_MISMATCH",
                    message="All observations and state must share batch size.",
                )
            if observation.values.device != state.memory.device:
                raise DataContractError(
                    code="OBSERVATION_DEVICE_MISMATCH",
                    message="Observations and state must share a device.",
                )
            expected = availability_time[:, None].expand_as(observation.available_time)
            if (observation.valid & ~self._times_match(observation.available_time, expected)).any():
                raise DataContractError(
                    code="OBSERVATION_AVAILABILITY_TIME_MISMATCH",
                    message="Posterior observations must be updated exactly at available_time.",
                )
        tokens, valid, provenance = self._combine_observations(new_observations, availability_time)
        row_valid = valid.any(dim=1)
        if not row_valid.any():
            return state.with_flag("empty_observation_update")
        memory = self.updater(state.memory, tokens, valid)
        result = self._sample_state(
            memory,
            availability_time,
            posterior=True,
            deterministic=deterministic,
            generator=generator,
            provenance=(*state.provenance, *provenance),
            flags=state.quality_flags,
        )
        if not row_valid.all():
            keep = row_valid[:, None, None]
            result.memory = torch.where(keep, result.memory, state.memory)
            if result.stochastic_mean is not None:
                assert result.stochastic_log_std is not None
                assert result.sample is not None
                assert state.stochastic_mean is not None
                assert state.stochastic_log_std is not None
                assert state.sample is not None
                result.stochastic_mean = torch.where(
                    keep, result.stochastic_mean, state.stochastic_mean
                )
                result.stochastic_log_std = torch.where(
                    keep, result.stochastic_log_std, state.stochastic_log_std
                )
                result.sample = torch.where(keep, result.sample, state.sample)
            result.state_kind = "mixed"
            result.quality_flags = tuple(
                dict.fromkeys((*result.quality_flags, "partial_empty_observation_update"))
            )
        result.validate()
        return result

    def _state_tokens(self, state: BeliefState) -> Tensor:
        if state.sample is None:
            return state.memory
        assert self.stochastic_projection is not None
        return state.memory + self.stochastic_projection(state.sample)

    def predict_future_observation(
        self,
        state: BeliefState,
        target_modality: str,
        *,
        scenario: str,
        target_time: Tensor | None = None,
    ) -> PredictionDistribution:
        self._validate_state_compatibility(state)
        if target_modality not in self.future_decoders:
            raise DataContractError(
                code="UNSUPPORTED_FUTURE_MODALITY",
                message=f"No future decoder is configured for {target_modality}.",
            )
        if not scenario:
            raise DataContractError(
                code="MISSING_PREDICTION_SCENARIO",
                message="Future observation predictions require an explicit scenario.",
            )
        if target_time is not None:
            self._validate_time(
                "target_time",
                target_time,
                batch=state.memory.shape[0],
                device=state.memory.device,
            )
            if not self._times_match(target_time, state.query_time).all():
                raise DataContractError(
                    code="FUTURE_TARGET_REQUIRES_PRIOR",
                    message="Roll the state to target_time before decoding a future observation.",
                )
        mean, log_std = self.future_decoders[target_modality](self._state_tokens(state))
        return PredictionDistribution(
            modality=target_modality,
            mean=mean,
            log_std=log_std,
            target_time=state.query_time if target_time is None else target_time,
            provenance="predicted_not_observed",
            scenario=scenario,
        )

    def predict_survival(
        self,
        state: BeliefState,
        endpoint: str,
        horizons: Tensor,
        *,
        stage: str,
        simulated: bool = False,
    ) -> StagePrediction:
        self._validate_state_compatibility(state)
        if endpoint.lower() != "os":
            raise DataContractError(
                code="ENDPOINT_NOT_CONFIGURED",
                message="Only OS is configured in the initial StageWorld model.",
            )
        if stage not in self.stage_to_index:
            raise DataContractError(code="UNKNOWN_STAGE", message=f"Unknown stage: {stage}")
        if horizons.ndim != 1 or not torch.isfinite(horizons).all() or (horizons < 0).any():
            raise DataContractError(
                code="INVALID_HORIZONS",
                message="horizons must be a nonnegative rank-one tensor.",
            )
        rates = self.survival_decoder(
            self._state_tokens(state),
            self._timeline_to_days(state.query_time),
            self.stage_to_index[stage],
        )
        horizon_values = horizons.to(rates)
        if rates.shape[-1] == 1:
            single_rates = rates.squeeze(-1)
            survival = survival_probability(
                single_rates, horizon_values, self.survival_cutpoints, open_tail=True
            )
            risk = risk_probability(
                single_rates, horizon_values, self.survival_cutpoints, open_tail=True
            )
            cif = None
        else:
            curves = competing_risk_curves(
                rates, horizon_values, self.survival_cutpoints, open_tail=True
            )
            survival = curves.survival
            risk = curves.risk
            cif = curves.cif
        return StagePrediction(
            stage=stage,
            query_time=state.query_time,
            endpoint="os",
            horizon_grid=horizons,
            rates=rates,
            survival=survival,
            risk=risk,
            cif=cif,
            input_manifest=state.provenance,
            model_version=self.config.model_version,
            quality_flags=state.quality_flags,
            simulated=simulated,
        )

    def _validate_stage_times(
        self,
        name: str,
        *,
        acquisition_time: Tensor,
        availability_time: Tensor,
        query_time: Tensor,
    ) -> None:
        batch = acquisition_time.shape[0]
        device = acquisition_time.device
        self._validate_time(
            f"{name}_availability_time", availability_time, batch=batch, device=device
        )
        self._validate_time(f"{name}_query_time", query_time, batch=batch, device=device)
        if self._time_precedes(availability_time, acquisition_time).any():
            raise DataContractError(
                code="AVAILABILITY_PRECEDES_ACQUISITION",
                message=f"{name} availability cannot precede acquisition.",
            )
        if self._time_precedes(query_time, availability_time).any():
            raise DataContractError(
                code="QUERY_PRECEDES_OBSERVATION_AVAILABILITY",
                message=f"{name} stage query cannot precede observation availability.",
            )

    def _resolve_observation_availability(
        self,
        name: str,
        observations: Sequence[ObservationTokens],
        declared_time: Tensor | None,
        query_time: Tensor,
        unavailable_event_mask: Tensor,
    ) -> Tensor:
        """Resolve update clocks without treating invalid padding as an event."""

        if not observations:
            raise DataContractError(
                code="NO_OBSERVATIONS",
                message=f"At least one {name} observation tensor is required.",
            )
        for observation in observations:
            self._validate_observation(observation)
            if observation.values.shape[0] != query_time.shape[0]:
                raise DataContractError(
                    code="OBSERVATION_BATCH_MISMATCH",
                    message="All stage observations and query_time must share batch size.",
                )
            if observation.values.device != query_time.device:
                raise DataContractError(
                    code="TIME_DEVICE_MISMATCH",
                    message="Stage observations and query_time must share a device.",
                )
        row_has_observation = torch.stack(
            [observation.valid.any(dim=1) for observation in observations], dim=1
        ).any(dim=1)
        if declared_time is not None:
            self._validate_time(
                f"{name}_availability_time",
                declared_time,
                batch=query_time.shape[0],
                device=query_time.device,
            )
            resolved = torch.where(
                row_has_observation | unavailable_event_mask,
                declared_time,
                query_time,
            )
        else:
            if unavailable_event_mask.any():
                raise DataContractError(
                    code="UNAVAILABLE_EVENT_TIME_REQUIRED",
                    message=(
                        f"Explicit {name} missing/failed events require a declared "
                        "availability time."
                    ),
                )
            resolved = query_time.clone()
            for row in range(query_time.shape[0]):
                row_time_parts = [
                    observation.available_time[row, observation.valid[row]]
                    for observation in observations
                    if observation.valid[row].any()
                ]
                if not row_time_parts:
                    continue
                row_times = torch.cat(row_time_parts)
                if not self._times_match(row_times, row_times[0].expand_as(row_times)).all():
                    raise DataContractError(
                        code="AMBIGUOUS_OBSERVATION_AVAILABILITY",
                        message=(
                            f"Valid {name} tokens in one patient row must share available_time."
                        ),
                    )
                resolved[row] = row_times[0].to(resolved)

        for observation in observations:
            expected = resolved[:, None].expand_as(observation.available_time)
            if (observation.valid & ~self._times_match(observation.available_time, expected)).any():
                raise DataContractError(
                    code="OBSERVATION_AVAILABILITY_TIME_MISMATCH",
                    message=f"Declared {name} availability must match valid observation tokens.",
                )
        return resolved

    def _advance_state(
        self,
        state: BeliefState,
        target_time: Tensor,
        actions: ActionTokens | None,
        *,
        deterministic: bool,
        generator: torch.Generator | None,
    ) -> BeliefState:
        if actions is None and self._times_match(state.query_time, target_time).all():
            return state
        if actions is None:
            actions = ActionTokens.empty(
                batch_size=state.memory.shape[0],
                value_dim=self.config.action_input_dim,
                device=state.memory.device,
            )
        return self.predict_prior(
            state,
            actions,
            target_time,
            deterministic=deterministic,
            generator=generator,
        )

    def _optional_row_mask(
        self,
        name: str,
        value: Tensor | None,
        *,
        batch: int,
        device: torch.device,
    ) -> Tensor:
        if value is None:
            return torch.zeros(batch, dtype=torch.bool, device=device)
        if value.shape != (batch,) or value.dtype is not torch.bool:
            raise DataContractError(
                code="INVALID_UNAVAILABLE_EVENT_MASK",
                message=f"{name} must be a boolean tensor with shape [B].",
            )
        if value.device != device:
            raise DataContractError(
                code="TIME_DEVICE_MISMATCH",
                message=f"{name} must share the observation device.",
            )
        return value

    def _combine_action_groups(
        self,
        groups: Sequence[ActionTokens | None],
        *,
        batch: int,
        device: torch.device,
    ) -> ActionTokens:
        actions: list[ActionTokens] = []
        for value in groups:
            if value is None:
                continue
            value.validate()
            if value.values.shape[0] != batch:
                raise DataContractError(
                    code="ACTION_BATCH_MISMATCH",
                    message="All scheduled action groups must share the state batch size.",
                )
            if value.values.shape[-1] != self.config.action_input_dim:
                raise DataContractError(
                    code="ACTION_DIMENSION_MISMATCH",
                    message="Scheduled action features do not match the model configuration.",
                )
            if value.values.device != device:
                raise DataContractError(
                    code="ACTION_DEVICE_MISMATCH",
                    message="Scheduled actions and state must share a device.",
                )
            if value.valid.any():
                actions.append(value)
        if not actions:
            return ActionTokens.empty(
                batch_size=batch,
                value_dim=self.config.action_input_dim,
                device=device,
            )

        optional_names = ("event_type", "planned_or_delivered", "known_exposure")
        for name in optional_names:
            present = [getattr(value, name) is not None for value in actions]
            if any(present) and not all(present):
                raise DataContractError(
                    code="INCOMPATIBLE_ACTION_METADATA",
                    message=f"Scheduled action groups disagree on optional {name} metadata.",
                )

        def concatenate_optional(name: str) -> Tensor | None:
            values = [getattr(value, name) for value in actions]
            if values[0] is None:
                return None
            return torch.cat([value for value in values if value is not None], dim=1)

        return ActionTokens(
            values=torch.cat([value.values for value in actions], dim=1),
            valid=torch.cat([value.valid for value in actions], dim=1),
            event_time=torch.cat([value.event_time for value in actions], dim=1),
            available_time=torch.cat([value.available_time for value in actions], dim=1),
            event_type=concatenate_optional("event_type"),
            planned_or_delivered=concatenate_optional("planned_or_delivered"),
            known_exposure=concatenate_optional("known_exposure"),
            provenance=tuple(
                dict.fromkeys(item for value in actions for item in value.provenance)
            ),
        )

    @staticmethod
    def _mask_actions(actions: ActionTokens, valid: Tensor) -> ActionTokens:
        return ActionTokens(
            values=actions.values,
            valid=valid,
            event_time=actions.event_time,
            available_time=actions.available_time,
            event_type=actions.event_type,
            planned_or_delivered=actions.planned_or_delivered,
            known_exposure=actions.known_exposure,
            provenance=actions.provenance,
        )

    def _action_replay_times(self, actions: ActionTokens) -> Tensor:
        return torch.maximum(actions.event_time, actions.available_time)

    def _replay_to(
        self,
        state: BeliefState,
        action_groups: Sequence[ActionTokens | None],
        target_time: Tensor,
        *,
        additional_boundaries: Sequence[tuple[Tensor, Tensor]] = (),
        deterministic: bool,
        generator: torch.Generator | None,
    ) -> BeliefState:
        """Advance at canonical action/observation boundaries, then at ``target_time``."""

        self._validate_state_compatibility(state)
        self._validate_time(
            "target_time",
            target_time,
            batch=state.memory.shape[0],
            device=state.memory.device,
        )
        if self._time_precedes(target_time, state.query_time).any():
            raise DataContractError(
                code="BACKWARD_TRANSITION",
                message="A scheduled replay target cannot precede the current state.",
            )
        actions = self._combine_action_groups(
            action_groups,
            batch=state.memory.shape[0],
            device=state.memory.device,
        )
        replay_times = self._action_replay_times(actions)
        boundary_times = [replay_times, target_time[:, None]]
        boundary_valid = [
            actions.valid,
            torch.ones(
                target_time.shape[0],
                1,
                dtype=torch.bool,
                device=target_time.device,
            ),
        ]
        for times, valid in additional_boundaries:
            if times.shape != valid.shape or times.shape[0] != target_time.shape[0]:
                raise DataContractError(
                    code="BOUNDARY_SHAPE_MISMATCH",
                    message="Replay boundary times and masks must have shape [B,N].",
                )
            if valid.dtype is not torch.bool:
                raise DataContractError(
                    code="BOUNDARY_MASK_DTYPE",
                    message="Replay boundary masks must be boolean.",
                )
            if times.device != state.memory.device or valid.device != state.memory.device:
                raise DataContractError(
                    code="TIME_DEVICE_MISMATCH",
                    message="Replay boundaries and state must share a device.",
                )
            if valid.any() and not torch.isfinite(times[valid]).all():
                raise DataContractError(
                    code="NONFINITE_TIME",
                    message="Valid replay boundaries must be finite.",
                )
            boundary_times.append(times)
            boundary_valid.append(valid)

        times = torch.cat(boundary_times, dim=1)
        valid = torch.cat(boundary_valid, dim=1)
        state_grid = state.query_time[:, None].expand_as(times)
        target_grid = target_time[:, None].expand_as(times)
        if (valid & self._time_precedes(times, state_grid)).any():
            raise DataContractError(
                code="BOUNDARY_PRECEDES_STATE",
                message="A replay boundary cannot precede the current state.",
            )
        if (valid & self._time_precedes(target_grid, times)).any():
            raise DataContractError(
                code="BOUNDARY_AFTER_TARGET",
                message="A replay boundary cannot follow its stage target.",
            )

        unique_times = torch.sort(torch.unique(times[valid].detach())).values
        previous: Tensor | None = None
        result = state
        for replay_time in unique_times:
            if previous is not None and bool(
                self._times_match(replay_time.reshape(1), previous.reshape(1)).item()
            ):
                continue
            time_grid = replay_time.expand_as(times)
            active_rows = (valid & self._times_match(times, time_grid)).any(dim=1)
            target = torch.where(active_rows, replay_time.to(target_time), result.query_time)
            action_valid = (
                actions.valid
                & self._times_match(replay_times, replay_time.expand_as(replay_times))
                & active_rows[:, None]
            )
            result = self.predict_prior(
                result,
                self._mask_actions(actions, action_valid),
                target,
                deterministic=deterministic,
                generator=generator,
            )
            previous = replay_time
        return result

    def _latest_action_time(
        self,
        state: BeliefState,
        action_groups: Sequence[ActionTokens | None],
    ) -> Tensor:
        actions = self._combine_action_groups(
            action_groups,
            batch=state.memory.shape[0],
            device=state.memory.device,
        )
        replay_times = self._action_replay_times(actions)
        initial = state.query_time[:, None].expand_as(replay_times)
        if (actions.valid & self._time_precedes(replay_times, initial)).any():
            raise DataContractError(
                code="ACTION_PRECEDES_STATE",
                message="A stage action cannot precede the state from which it is replayed.",
            )
        return torch.where(actions.valid, replay_times, initial).max(dim=1).values

    def _merge_state_rows(
        self,
        preferred: BeliefState,
        fallback: BeliefState,
        prefer: Tensor,
    ) -> BeliefState:
        self._validate_state_compatibility(preferred)
        self._validate_state_compatibility(fallback)
        keep = prefer[:, None, None]
        stochastic_mean: Tensor | None = None
        stochastic_log_std: Tensor | None = None
        sample: Tensor | None = None
        if preferred.stochastic_mean is not None:
            assert preferred.stochastic_log_std is not None and preferred.sample is not None
            assert fallback.stochastic_mean is not None
            assert fallback.stochastic_log_std is not None and fallback.sample is not None
            stochastic_mean = torch.where(
                keep, preferred.stochastic_mean, fallback.stochastic_mean
            )
            stochastic_log_std = torch.where(
                keep, preferred.stochastic_log_std, fallback.stochastic_log_std
            )
            sample = torch.where(keep, preferred.sample, fallback.sample)
        if prefer.all():
            state_kind = preferred.state_kind
        elif not prefer.any():
            state_kind = fallback.state_kind
        else:
            state_kind = "mixed"
        result = BeliefState(
            memory=torch.where(keep, preferred.memory, fallback.memory),
            query_time=torch.where(prefer, preferred.query_time, fallback.query_time),
            stochastic_mean=stochastic_mean,
            stochastic_log_std=stochastic_log_std,
            sample=sample,
            state_kind=state_kind,
            provenance=tuple(dict.fromkeys((*preferred.provenance, *fallback.provenance))),
            quality_flags=tuple(
                dict.fromkeys((*preferred.quality_flags, *fallback.quality_flags))
            ),
        )
        result.validate()
        return result

    def _observed_stage_start(
        self,
        state: BeliefState,
        future_prior: BeliefState,
        pre_acquisition_actions: Sequence[ActionTokens | None],
        target_observation: ObservationTokens,
        unavailable_event_mask: Tensor,
        *,
        deterministic: bool,
        generator: torch.Generator | None,
    ) -> BeliefState:
        has_target_event = target_observation.valid.any(dim=1) | unavailable_event_mask
        latest_action_time = self._latest_action_time(state, pre_acquisition_actions)
        action_only = self._replay_to(
            state,
            pre_acquisition_actions,
            latest_action_time,
            deterministic=deterministic,
            generator=generator,
        )
        return self._merge_state_rows(future_prior, action_only, has_target_event)

    def _stage_update(
        self,
        state: BeliefState,
        observations: Sequence[ObservationTokens],
        update_actions: ActionTokens | None,
        acquisition_time: Tensor,
        availability_time: Tensor,
        query_time: Tensor,
        unavailable_event_mask: Tensor,
        *,
        deterministic: bool,
        generator: torch.Generator | None,
    ) -> tuple[BeliefState, BeliefState, BeliefState]:
        actions = self._combine_action_groups(
            (update_actions,),
            batch=state.memory.shape[0],
            device=state.memory.device,
        )
        replay_times = self._action_replay_times(actions)
        after_update = actions.valid & self._time_precedes(
            availability_time[:, None].expand_as(replay_times), replay_times
        )
        before_update = actions.valid & ~after_update
        boundaries = [(value.acquired_time, value.valid) for value in observations]
        boundaries.append((acquisition_time[:, None], unavailable_event_mask[:, None]))
        pre_update = self._replay_to(
            state,
            (self._mask_actions(actions, before_update),),
            availability_time,
            additional_boundaries=boundaries,
            deterministic=deterministic,
            generator=generator,
        )
        post_update = self.update_posterior(
            pre_update,
            observations,
            availability_time,
            deterministic=deterministic,
            generator=generator,
        )
        final = self._replay_to(
            post_update,
            (self._mask_actions(actions, after_update),),
            query_time,
            deterministic=deterministic,
            generator=generator,
        )
        return pre_update, post_update, final

    def forward_three_stage(
        self,
        *,
        ct0: ObservationTokens,
        clinical0: ObservationTokens,
        s0_time: Tensor,
        treatment_actions: ActionTokens,
        ct1_acquisition_time: Tensor,
        ct1: ObservationTokens,
        s1_time: Tensor,
        surgery_actions: ActionTokens,
        pathology_acquisition_time: Tensor,
        pathology: ObservationTokens,
        s2_time: Tensor,
        horizons: Tensor,
        ct1_availability_time: Tensor | None = None,
        ct1_unavailable_event_mask: Tensor | None = None,
        clinical1: ObservationTokens | None = None,
        s1_update_actions: ActionTokens | None = None,
        pathology_availability_time: Tensor | None = None,
        pathology_unavailable_event_mask: Tensor | None = None,
        clinical2: ObservationTokens | None = None,
        s2_update_actions: ActionTokens | None = None,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> ThreeStageOutput:
        expected_modalities: list[tuple[ObservationTokens, str, str]] = [
            (ct0, "ct", "ct0"),
            (clinical0, "clinical", "clinical0"),
            (ct1, "ct", "ct1"),
            (pathology, "pathology", "pathology"),
        ]
        if clinical1 is not None:
            expected_modalities.append((clinical1, "clinical", "clinical1"))
        if clinical2 is not None:
            expected_modalities.append((clinical2, "clinical", "clinical2"))
        for observation, expected_modality, field in expected_modalities:
            if observation.modality_name != expected_modality:
                raise DataContractError(
                    code="STAGE_MODALITY_MISMATCH",
                    message=f"{field} must contain {expected_modality} observations.",
                )
        for observation, target_time, field in (
            (ct1, ct1_acquisition_time, "ct1"),
            (pathology, pathology_acquisition_time, "pathology"),
        ):
            self._validate_time(
                f"{field}_acquisition_time",
                target_time,
                batch=observation.values.shape[0],
                device=observation.values.device,
            )
            expected_acquisition = target_time[:, None].expand_as(observation.acquired_time)
            if (
                observation.valid
                & ~self._times_match(observation.acquired_time, expected_acquisition)
            ).any():
                raise DataContractError(
                    code="FUTURE_TARGET_TIME_MISMATCH",
                    message=f"{field} target time must match its observed acquisition time.",
                )
        state_s0 = self.initialize(
            [ct0],
            clinical0,
            s0_time,
            deterministic=deterministic,
            generator=generator,
        )
        survival_s0 = self.predict_survival(state_s0, "os", horizons, stage="S0")

        ct1_unavailable_event_mask = self._optional_row_mask(
            "ct1_unavailable_event_mask",
            ct1_unavailable_event_mask,
            batch=ct1.values.shape[0],
            device=ct1.values.device,
        )
        prior_ct1 = self._replay_to(
            state_s0,
            (treatment_actions,),
            ct1_acquisition_time,
            deterministic=deterministic,
            generator=generator,
        )
        future_ct = self.predict_future_observation(
            prior_ct1,
            "ct",
            scenario="observed_treatment_exposure_for_training_pair",
            target_time=ct1_acquisition_time,
        )
        s1_observations = [ct1]
        if clinical1 is not None:
            s1_observations.append(clinical1)
        ct1_availability_time = self._resolve_observation_availability(
            "ct1",
            s1_observations,
            ct1_availability_time,
            s1_time,
            ct1_unavailable_event_mask,
        )
        self._validate_stage_times(
            "ct1",
            acquisition_time=ct1_acquisition_time,
            availability_time=ct1_availability_time,
            query_time=s1_time,
        )
        s1_start = self._observed_stage_start(
            state_s0,
            prior_ct1,
            (treatment_actions,),
            ct1,
            ct1_unavailable_event_mask,
            deterministic=deterministic,
            generator=generator,
        )
        pre_ct1_update, post_ct1_update, state_s1 = self._stage_update(
            s1_start,
            s1_observations,
            s1_update_actions,
            ct1_acquisition_time,
            ct1_availability_time,
            s1_time,
            ct1_unavailable_event_mask,
            deterministic=deterministic,
            generator=generator,
        )
        survival_s1 = self.predict_survival(state_s1, "os", horizons, stage="S1")

        pathology_unavailable_event_mask = self._optional_row_mask(
            "pathology_unavailable_event_mask",
            pathology_unavailable_event_mask,
            batch=pathology.values.shape[0],
            device=pathology.values.device,
        )
        prior_pathology = self._replay_to(
            state_s1,
            (surgery_actions,),
            pathology_acquisition_time,
            deterministic=deterministic,
            generator=generator,
        )
        future_pathology = self.predict_future_observation(
            prior_pathology,
            "pathology",
            scenario="known_surgery_event_for_training_pair",
            target_time=pathology_acquisition_time,
        )
        s2_observations = [pathology]
        if clinical2 is not None:
            s2_observations.append(clinical2)
        pathology_availability_time = self._resolve_observation_availability(
            "pathology",
            s2_observations,
            pathology_availability_time,
            s2_time,
            pathology_unavailable_event_mask,
        )
        self._validate_stage_times(
            "pathology",
            acquisition_time=pathology_acquisition_time,
            availability_time=pathology_availability_time,
            query_time=s2_time,
        )
        s2_start = self._observed_stage_start(
            state_s1,
            prior_pathology,
            (surgery_actions,),
            pathology,
            pathology_unavailable_event_mask,
            deterministic=deterministic,
            generator=generator,
        )
        pre_pathology_update, post_pathology_update, state_s2 = self._stage_update(
            s2_start,
            s2_observations,
            s2_update_actions,
            pathology_acquisition_time,
            pathology_availability_time,
            s2_time,
            pathology_unavailable_event_mask,
            deterministic=deterministic,
            generator=generator,
        )
        survival_s2 = self.predict_survival(state_s2, "os", horizons, stage="S2")
        return ThreeStageOutput(
            state_s0=state_s0,
            prior_ct1=prior_ct1,
            pre_ct1_update=pre_ct1_update,
            post_ct1_update=post_ct1_update,
            future_ct=future_ct,
            state_s1=state_s1,
            prior_pathology=prior_pathology,
            pre_pathology_update=pre_pathology_update,
            post_pathology_update=post_pathology_update,
            future_pathology=future_pathology,
            state_s2=state_s2,
            survival_s0=survival_s0,
            survival_s1=survival_s1,
            survival_s2=survival_s2,
        )

    def forward(self, **kwargs: Any) -> ThreeStageOutput:
        """Route module calls through the typed three-stage training path."""

        return self.forward_three_stage(**kwargs)
