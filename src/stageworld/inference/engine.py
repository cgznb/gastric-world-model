"""Feature-history replay and prediction orchestration for StageWorld inference."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace

import torch
from torch import Tensor

from stageworld.config import RunMode
from stageworld.data import (
    AvailabilityBasis,
    FeatureFirewall,
    MissingCategory,
    ObservationRole,
    Prefix,
    QualityStatus,
    Query,
    QueryEligibility,
    SourceType,
    Treatment,
    TreatmentStatus,
    treatment_kind_id,
    treatment_status_id,
)
from stageworld.encoders import ObservationTokens, merge_observation_tokens
from stageworld.errors import ArtifactError, ConfigurationError, DataContractError
from stageworld.model import ActionTokens, BeliefState, StageWorldModel

from .cache import (
    InferenceStateCache,
    PrefixEntry,
    PrefixIdentity,
    StateCacheKey,
    clone_inference_state,
)
from .contracts import (
    CHECKPOINT_SCHEMA_VERSION,
    ActionFeature,
    CheckpointContract,
    FeatureInputContract,
    FeatureManifest,
    FutureObservationOutput,
    InputReference,
    PredictionOutput,
    Scenario,
    ScenarioKind,
)


@dataclass(frozen=True, slots=True)
class _FeatureEvent:
    record_id: str
    acquired_at_days: float
    available_at_days: float
    tokens: ObservationTokens
    initializable: bool
    reference: InputReference


@dataclass(frozen=True, slots=True)
class _ActionEvent:
    record_id: str
    replay_at_days: float
    event_time_days: float
    available_at_days: float
    treatment: Treatment
    feature: ActionFeature
    reference: InputReference


@dataclass(frozen=True, slots=True)
class _UnavailableObservationBoundary:
    """Known clocks for an observation that cannot supply feature tokens."""

    record_id: str
    acquired_at_days: float
    available_at_days: float


@dataclass(frozen=True, slots=True)
class ReplayResult:
    state: BeliefState
    cache_key: StateCacheKey
    input_manifest: tuple[InputReference, ...]
    quality_flags: tuple[str, ...]
    cache_hit: bool
    population: str
    query: Query


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


class InferenceEngine:
    """Replay legal prefixes and produce conditional OS predictions.

    The engine never receives outcomes. Each query is independently passed through
    ``FeatureFirewall`` before any feature lookup, which prevents a prior S2 request or
    the mere existence of future feature entries from changing an S0 request.
    """

    def __init__(
        self,
        model: StageWorldModel,
        checkpoint: CheckpointContract,
        feature_contract: FeatureInputContract,
        *,
        cache: InferenceStateCache | None = None,
        survival_time_unit: str = "year",
        survival_parameterization: str = "piecewise_constant_hazard_rate",
        survival_open_tail_interval: bool = True,
    ) -> None:
        if checkpoint.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ArtifactError(
                code="CHECKPOINT_SCHEMA_MISMATCH",
                message="Checkpoint schema is not supported by this inference runtime.",
            )
        if checkpoint.model_version != model.config.model_version:
            raise ArtifactError(
                code="MODEL_VERSION_MISMATCH",
                message="Loaded model architecture/version does not match checkpoint metadata.",
                details={
                    "checkpoint_model_version": checkpoint.model_version,
                    "runtime_model_version": model.config.model_version,
                },
            )
        runtime_model_config = asdict(model.config)
        checkpoint_model_config = dict(checkpoint.model_config)
        if checkpoint_model_config != runtime_model_config:
            changed = sorted(
                key
                for key in set(checkpoint_model_config) | set(runtime_model_config)
                if checkpoint_model_config.get(key) != runtime_model_config.get(key)
            )
            raise ArtifactError(
                code="MODEL_CONFIG_MISMATCH",
                message="Loaded model configuration does not match the checkpoint contract.",
                details={"fields": changed},
            )
        if checkpoint.endpoint.lower() != "os":
            raise ArtifactError(
                code="CHECKPOINT_ENDPOINT_MISMATCH",
                message="The initial inference runtime accepts only an OS checkpoint.",
            )
        if checkpoint.phase != "joint_survival":
            raise ArtifactError(
                code="CHECKPOINT_PHASE_NOT_INFERABLE",
                message="Survival inference requires a joint_survival checkpoint.",
            )
        if survival_time_unit not in {"day", "year"}:
            raise ConfigurationError(
                code="INVALID_SURVIVAL_TIME_UNIT",
                message="Inference survival_time_unit must be day or year.",
            )
        expected_survival_contract = {
            "schema_version": "stageworld-survival-contract-v1",
            "endpoint": checkpoint.endpoint,
            "parameterization": survival_parameterization,
            "time_unit": survival_time_unit,
            "cutpoints": tuple(model.config.survival_cutpoints),
            "open_tail_interval": survival_open_tail_interval,
            "num_causes": model.config.survival_causes,
        }
        checkpoint_survival_contract = dict(checkpoint.survival_contract)
        if checkpoint_survival_contract != expected_survival_contract:
            changed = sorted(
                key
                for key in set(checkpoint_survival_contract) | set(expected_survival_contract)
                if checkpoint_survival_contract.get(key) != expected_survival_contract.get(key)
            )
            raise ArtifactError(
                code="SURVIVAL_CONTRACT_MISMATCH",
                message="Inference survival settings do not match the checkpoint contract.",
                details={"fields": changed},
            )
        configured_modalities = set(model.modality_names)
        contracted_modalities = {item.modality for item in feature_contract.modalities}
        if configured_modalities != contracted_modalities:
            raise ConfigurationError(
                code="FEATURE_MODEL_MODALITY_MISMATCH",
                message="Feature contract modalities must exactly match the loaded model.",
                details={
                    "model_modalities": sorted(configured_modalities),
                    "contract_modalities": sorted(contracted_modalities),
                },
            )
        for item in feature_contract.modalities:
            expected_dim = model.input_projections[item.modality].in_features
            if item.provenance.feature_dim != expected_dim:
                raise ConfigurationError(
                    code="FEATURE_MODEL_DIMENSION_MISMATCH",
                    message="Feature contract dimension does not match the loaded model.",
                    details={"modality": item.modality},
                )
        self.model = model
        self.model.eval()
        self.checkpoint = checkpoint
        self.feature_contract = feature_contract
        self.cache = cache if cache is not None else InferenceStateCache()
        self.survival_time_unit = survival_time_unit

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def _days_to_timeline(self, value: Tensor) -> Tensor:
        if self.model.config.timeline_time_unit == "day":
            return value
        return value / self.model.config.days_per_year

    def _timeline_scalar_to_days(self, value: float) -> float:
        if self.model.config.timeline_time_unit == "day":
            return value
        return value * self.model.config.days_per_year

    def _validate_mode(
        self, firewall: FeatureFirewall, manifest: FeatureManifest, patient_id: str
    ) -> None:
        if manifest.patient_id != patient_id:
            raise DataContractError(
                code="FEATURE_PATIENT_MISMATCH",
                message="Feature manifest patient does not match the requested anonymous patient.",
            )
        modes = {
            "checkpoint": self.checkpoint.mode.value,
            "feature_manifest": manifest.mode.value,
            "feature_firewall": firewall.policy.mode.value,
        }
        if len(set(modes.values())) != 1:
            raise ArtifactError(
                code="INFERENCE_MODE_MISMATCH",
                message=(
                    "Checkpoint, feature manifest, and temporal policy modes must match exactly."
                ),
                details=modes,
            )
        if manifest.schema_version != self.feature_contract.schema_version:
            raise ArtifactError(
                code="FEATURE_SCHEMA_MISMATCH",
                message="Feature manifest schema does not match the checkpoint input contract.",
                details={
                    "expected": self.feature_contract.schema_version,
                    "received": manifest.schema_version,
                },
            )

    def _prefix_identity(self, prefix: Prefix, manifest: FeatureManifest) -> PrefixIdentity:
        entries: list[PrefixEntry] = []
        for observation in prefix.observations:
            assert observation.available_at_days is not None
            entries.append(
                PrefixEntry(
                    record_kind="observation",
                    record_id=observation.observation_id,
                    acquired_or_event_time_days=observation.acquired_at_days,
                    available_at_days=observation.available_at_days,
                    role_or_status=observation.role.value,
                    feature_version=observation.feature_version or "unversioned",
                )
            )
        for measurement in prefix.clinical_measurements:
            assert measurement.available_at_days is not None
            entries.append(
                PrefixEntry(
                    record_kind="clinical",
                    record_id=measurement.measurement_id,
                    acquired_or_event_time_days=measurement.acquired_at_days,
                    available_at_days=measurement.available_at_days,
                    role_or_status=measurement.field_name,
                    feature_version=self.feature_contract.for_modality(
                        "clinical"
                    ).provenance.source_version,
                )
            )
        for treatment in prefix.treatments:
            assert treatment.available_at_days is not None
            entries.append(
                PrefixEntry(
                    record_kind="treatment",
                    record_id=treatment.event_id,
                    acquired_or_event_time_days=(
                        treatment.end_days
                        if treatment.end_days is not None
                        else treatment.start_days
                    ),
                    available_at_days=treatment.available_at_days,
                    role_or_status=treatment.planned_or_delivered.value,
                    feature_version=self.feature_contract.action_feature_version,
                )
            )
        entries.sort(
            key=lambda item: (
                item.available_at_days,
                item.record_kind,
                item.record_id,
                item.role_or_status,
            )
        )
        return PrefixIdentity(
            feature_schema_version=manifest.schema_version,
            feature_manifest_lineage_id=manifest.manifest_lineage_id,
            entries=tuple(entries),
        )

    def _cache_key(self, prefix: Prefix, manifest: FeatureManifest) -> StateCacheKey:
        return StateCacheKey(
            patient_id=prefix.patient_id,
            prefix=self._prefix_identity(prefix, manifest),
            stage=prefix.query.stage.value,
            query_time_days=float(prefix.query.query_time_days),
            checkpoint_id=self.checkpoint.checkpoint_id,
            weight_version=self.checkpoint.weight_version,
            model_version=self.checkpoint.model_version,
        )

    def _tokens_to_device(self, tokens: ObservationTokens) -> ObservationTokens:
        return ObservationTokens(
            values=tokens.values.detach().to(device=self.device, dtype=self.dtype),
            valid=tokens.valid.detach().to(device=self.device),
            modality=tokens.modality.detach().to(device=self.device),
            acquired_time=self._days_to_timeline(
                tokens.acquired_time.detach().to(device=self.device, dtype=self.dtype)
            ),
            available_time=self._days_to_timeline(
                tokens.available_time.detach().to(device=self.device, dtype=self.dtype)
            ),
            provenance=tokens.provenance,
            source_id=tokens.source_id,
            modality_name=tokens.modality_name,
            coords=(
                None
                if tokens.coords is None
                else tokens.coords.detach().to(device=self.device, dtype=self.dtype)
            ),
            coordinate_system=tokens.coordinate_system,
            quality_flags=tokens.quality_flags,
        )

    def _validate_tokens(
        self,
        tokens: ObservationTokens,
        *,
        record_id: str,
        modality: str,
        acquired_at_days: float,
        available_at_days: float,
        record_feature_version: str | None,
    ) -> ObservationTokens:
        if tokens.values.shape[0] != 1:
            raise DataContractError(
                code="INFERENCE_BATCH_MUST_BE_ONE",
                message="Each anonymous feature-history record must have batch size one.",
            )
        if tokens.values.requires_grad or (
            tokens.coords is not None and tokens.coords.requires_grad
        ):
            raise DataContractError(
                code="ONLINE_FEATURE_TENSOR_FORBIDDEN",
                message="Feature-history tensors must be detached before inference caching.",
            )
        if tokens.modality_name != modality:
            raise DataContractError(
                code="FEATURE_MODALITY_MISMATCH",
                message="Feature tensor modality does not match its filtered history record.",
                details={"record_id": record_id},
            )
        expected = self.feature_contract.for_modality(modality)
        if tokens.provenance != expected.provenance:
            raise ArtifactError(
                code="FEATURE_VERSION_MISMATCH",
                message=(
                    "Feature encoder/weight/preprocess provenance does not match the "
                    "checkpoint contract."
                ),
                details={"record_id": record_id, "modality": modality},
            )
        if (
            expected.record_feature_version is not None
            and record_feature_version != expected.record_feature_version
        ):
            raise ArtifactError(
                code="FEATURE_VERSION_MISMATCH",
                message="Record feature version does not match the checkpoint contract.",
                details={"record_id": record_id, "modality": modality},
            )
        valid = tokens.valid
        if not valid.any():
            raise DataContractError(
                code="EMPTY_FEATURE_RECORD",
                message="A present history record cannot be represented only by padding tokens.",
                details={"record_id": record_id},
            )
        acquired = tokens.acquired_time[valid].detach().cpu().float()
        available = tokens.available_time[valid].detach().cpu().float()
        if not torch.allclose(
            acquired, torch.full_like(acquired, acquired_at_days), rtol=0.0, atol=1e-5
        ) or not torch.allclose(
            available, torch.full_like(available, available_at_days), rtol=0.0, atol=1e-5
        ):
            raise DataContractError(
                code="FEATURE_TIME_MISMATCH",
                message="Feature tensor clocks do not match the filtered history metadata.",
                details={"record_id": record_id},
            )
        return self._tokens_to_device(tokens)

    def _collect_events(
        self, prefix: Prefix, manifest: FeatureManifest
    ) -> tuple[
        tuple[_FeatureEvent, ...],
        tuple[_ActionEvent, ...],
        tuple[_UnavailableObservationBoundary, ...],
        tuple[InputReference, ...],
        tuple[str, ...],
    ]:
        features: list[_FeatureEvent] = []
        actions: list[_ActionEvent] = []
        unavailable_boundaries: list[_UnavailableObservationBoundary] = []
        references: list[InputReference] = []
        flags: list[str] = []

        for observation in prefix.observations:
            if (
                observation.missing_reason is not None
                or observation.quality_status is QualityStatus.FAILED
                or observation.local_asset_id is None
            ):
                flags.append("observation_unavailable")
                if observation.quality_status is QualityStatus.FAILED:
                    flags.append("observation_failed_qc")
                explicit_unavailable_event = (
                    observation.quality_status is QualityStatus.FAILED
                    or observation.missing_reason
                    in {MissingCategory.MISSING, MissingCategory.FAILED_QC}
                )
                if explicit_unavailable_event and observation.available_at_days is not None:
                    unavailable_boundaries.append(
                        _UnavailableObservationBoundary(
                            record_id=observation.observation_id,
                            acquired_at_days=observation.acquired_at_days,
                            available_at_days=observation.available_at_days,
                        )
                    )
                continue
            if observation.source_type is SourceType.PREDICTED:
                raise DataContractError(
                    code="PREDICTED_AS_OBSERVED_FORBIDDEN",
                    message="Predicted observations cannot enter observed-history inference.",
                )
            try:
                raw_tokens = manifest.observation_features[observation.observation_id]
            except KeyError as exc:
                raise ArtifactError(
                    code="FEATURE_RECORD_MISSING",
                    message="A permitted observed record has no feature-manifest entry.",
                    details={"record_id": observation.observation_id},
                ) from exc
            assert observation.available_at_days is not None
            modality = observation.modality.value
            tokens = self._validate_tokens(
                raw_tokens,
                record_id=observation.observation_id,
                modality=modality,
                acquired_at_days=observation.acquired_at_days,
                available_at_days=observation.available_at_days,
                record_feature_version=observation.feature_version,
            )
            reference = InputReference(
                record_kind="observation",
                record_id=observation.observation_id,
                source_type=observation.source_type.value,
                feature_version=(observation.feature_version or tokens.provenance.source_version),
            )
            features.append(
                _FeatureEvent(
                    record_id=observation.observation_id,
                    acquired_at_days=observation.acquired_at_days,
                    available_at_days=observation.available_at_days,
                    tokens=tokens,
                    initializable=observation.role is ObservationRole.BASELINE_CT,
                    reference=reference,
                )
            )
            references.append(reference)
            if observation.available_at_days > observation.acquired_at_days:
                flags.append("delayed_observation_update")
                flags.append(f"delayed_{observation.role.value}_update")
            if observation.availability_basis is AvailabilityBasis.INFERRED_CONSERVATIVE:
                flags.append("inferred_availability")
            if observation.availability_basis is AvailabilityBasis.RESEARCH_ASSUMPTION:
                flags.append("research_availability_assumption")
            if observation.quality_status is QualityStatus.UNKNOWN:
                flags.append("observation_quality_unknown")
            for row in tokens.quality_flags:
                flags.extend(row)

        for measurement in prefix.clinical_measurements:
            if measurement.missing_reason is not None or measurement.typed_value is None:
                flags.append("clinical_measurement_unavailable")
                if (
                    measurement.missing_reason
                    in {MissingCategory.MISSING, MissingCategory.FAILED_QC}
                    and measurement.available_at_days is not None
                ):
                    unavailable_boundaries.append(
                        _UnavailableObservationBoundary(
                            record_id=measurement.measurement_id,
                            acquired_at_days=measurement.acquired_at_days,
                            available_at_days=measurement.available_at_days,
                        )
                    )
                continue
            try:
                raw_tokens = manifest.clinical_features[measurement.measurement_id]
            except KeyError as exc:
                raise ArtifactError(
                    code="FEATURE_RECORD_MISSING",
                    message="A permitted clinical record has no feature-manifest entry.",
                    details={"record_id": measurement.measurement_id},
                ) from exc
            assert measurement.available_at_days is not None
            tokens = self._validate_tokens(
                raw_tokens,
                record_id=measurement.measurement_id,
                modality="clinical",
                acquired_at_days=measurement.acquired_at_days,
                available_at_days=measurement.available_at_days,
                record_feature_version=None,
            )
            reference = InputReference(
                record_kind="clinical",
                record_id=measurement.measurement_id,
                source_type=measurement.source_type.value,
                feature_version=tokens.provenance.source_version,
            )
            features.append(
                _FeatureEvent(
                    record_id=measurement.measurement_id,
                    acquired_at_days=measurement.acquired_at_days,
                    available_at_days=measurement.available_at_days,
                    tokens=tokens,
                    initializable=True,
                    reference=reference,
                )
            )
            references.append(reference)
            if measurement.available_at_days > measurement.acquired_at_days:
                flags.append("delayed_clinical_update")

        for treatment in prefix.treatments:
            if treatment.planned_or_delivered is not TreatmentStatus.DELIVERED:
                flags.append("planned_treatment_not_applied_to_observed_risk")
                continue
            try:
                feature = manifest.action_features[treatment.event_id]
            except KeyError as exc:
                raise ArtifactError(
                    code="ACTION_FEATURE_MISSING",
                    message="A permitted delivered treatment has no action feature entry.",
                    details={"record_id": treatment.event_id},
                ) from exc
            if feature.feature_version != self.feature_contract.action_feature_version:
                raise ArtifactError(
                    code="ACTION_FEATURE_VERSION_MISMATCH",
                    message=(
                        "Treatment action feature version does not match the checkpoint contract."
                    ),
                    details={"record_id": treatment.event_id},
                )
            if feature.values.shape[0] != self.model.config.action_input_dim:
                raise DataContractError(
                    code="ACTION_FEATURE_DIMENSION_MISMATCH",
                    message="Treatment feature dimension does not match the loaded model.",
                )
            assert treatment.available_at_days is not None
            if treatment.end_days is None:
                raise DataContractError(
                    code="DELIVERED_ACTION_TIME_MISSING",
                    message="Delivered treatment inference requires an actual end/event time.",
                )
            reference = InputReference(
                record_kind="treatment",
                record_id=treatment.event_id,
                source_type="observed_delivered",
                feature_version=feature.feature_version,
            )
            actions.append(
                _ActionEvent(
                    record_id=treatment.event_id,
                    replay_at_days=max(treatment.available_at_days, treatment.end_days),
                    event_time_days=treatment.end_days,
                    available_at_days=treatment.available_at_days,
                    treatment=treatment,
                    feature=feature,
                    reference=reference,
                )
            )
            references.append(reference)

        features.sort(key=lambda item: (item.available_at_days, item.record_id))
        actions.sort(key=lambda item: (item.replay_at_days, item.record_id))
        unavailable_boundaries.sort(
            key=lambda item: (item.available_at_days, item.record_id)
        )
        references.sort(key=lambda item: (item.record_kind, item.record_id))
        return (
            tuple(features),
            tuple(actions),
            tuple(unavailable_boundaries),
            tuple(references),
            _unique(flags),
        )

    def _action_tokens(self, events: Sequence[_ActionEvent]) -> ActionTokens:
        count = max(1, len(events))
        values = torch.zeros(
            1,
            count,
            self.model.config.action_input_dim,
            dtype=self.dtype,
            device=self.device,
        )
        valid = torch.zeros(1, count, dtype=torch.bool, device=self.device)
        event_time = torch.zeros(1, count, dtype=self.dtype, device=self.device)
        available_time = torch.zeros(1, count, dtype=self.dtype, device=self.device)
        event_type = torch.zeros(1, count, dtype=torch.long, device=self.device)
        status = torch.zeros(1, count, dtype=torch.long, device=self.device)
        exposure = torch.zeros(1, count, dtype=self.dtype, device=self.device)
        provenance: list[str] = []
        for index, item in enumerate(events):
            values[0, index] = item.feature.values.detach().to(device=self.device, dtype=self.dtype)
            valid[0, index] = True
            event_time[0, index] = item.event_time_days
            available_time[0, index] = item.available_at_days
            event_type[0, index] = treatment_kind_id(item.treatment.treatment_kind)
            status[0, index] = treatment_status_id(item.treatment.planned_or_delivered)
            exposure[0, index] = item.feature.known_exposure
            provenance.append(
                f"treatment:{item.record_id}:{item.feature.feature_version}:observed_delivered"
            )
        return ActionTokens(
            values=values,
            valid=valid,
            event_time=self._days_to_timeline(event_time),
            available_time=self._days_to_timeline(available_time),
            event_type=event_type,
            planned_or_delivered=status,
            known_exposure=exposure,
            provenance=tuple(provenance),
        )

    def _time_tensor(self, value: float) -> Tensor:
        value_tensor = torch.tensor([value], dtype=self.dtype, device=self.device)
        return self._days_to_timeline(value_tensor)

    @staticmethod
    def _merge_feature_events(events: Sequence[_FeatureEvent]) -> list[ObservationTokens]:
        by_modality: dict[str, list[ObservationTokens]] = {}
        for event in events:
            by_modality.setdefault(event.tokens.modality_name, []).append(event.tokens)
        return [
            merge_observation_tokens(by_modality[modality])
            for modality in ("ct", "pathology", "clinical")
            if modality in by_modality
        ]

    def _transition(
        self, state: BeliefState, events: Sequence[_ActionEvent], target_time: float
    ) -> BeliefState:
        state_time_days = self._timeline_scalar_to_days(float(state.query_time.item()))
        if target_time < state_time_days - 1e-5:
            raise DataContractError(
                code="HISTORY_REPLAY_MOVES_BACKWARD",
                message="Feature history cannot be replayed backward in time.",
            )
        if not events and math.isclose(
            target_time, state_time_days, rel_tol=0.0, abs_tol=1e-5
        ):
            return state
        return self.model.predict_prior(
            state,
            self._action_tokens(events),
            self._time_tensor(target_time),
            deterministic=True,
        )

    def _compute_state(
        self,
        query: Query,
        feature_events: Sequence[_FeatureEvent],
        action_events: Sequence[_ActionEvent],
        unavailable_boundaries: Sequence[_UnavailableObservationBoundary],
        quality_flags: Sequence[str],
    ) -> BeliefState:
        initializable = [item for item in feature_events if item.initializable]
        if not initializable:
            raise DataContractError(
                code="INITIAL_STATE_INPUT_REQUIRED",
                message="No baseline CT or permitted clinical feature is available at this query.",
                remediation=(
                    "Wait for a legal baseline input; never initialize from future pathology."
                ),
            )
        initial_time = min(item.available_at_days for item in initializable)
        if any(
            not item.initializable and item.available_at_days < initial_time
            for item in feature_events
        ):
            raise DataContractError(
                code="NON_BASELINE_PRECEDES_INITIAL_STATE",
                message="A non-baseline observation precedes every initialization input.",
            )
        initial_indices = {
            index
            for index, item in enumerate(feature_events)
            if item.initializable
            and math.isclose(item.available_at_days, initial_time, rel_tol=0.0, abs_tol=1e-6)
        }
        initial = [item for index, item in enumerate(feature_events) if index in initial_indices]
        initial_tokens = self._merge_feature_events(initial)
        initial_clinical = next(
            (value for value in initial_tokens if value.modality_name == "clinical"), None
        )
        state = self.model.initialize(
            [value for value in initial_tokens if value.modality_name != "clinical"],
            initial_clinical,
            self._time_tensor(initial_time),
            deterministic=True,
        )

        remaining_features = [
            item for index, item in enumerate(feature_events) if index not in initial_indices
        ]
        early_actions = [item for item in action_events if item.replay_at_days <= initial_time]
        state = self._transition(state, early_actions, initial_time)
        remaining_actions = [item for item in action_events if item.replay_at_days > initial_time]
        acquisition_boundaries = {
            item.acquired_at_days
            for item in remaining_features
            if initial_time < item.acquired_at_days <= query.query_time_days
        }
        missing_observation_boundaries = {
            boundary
            for item in unavailable_boundaries
            for boundary in (item.acquired_at_days, item.available_at_days)
            if initial_time < boundary <= query.query_time_days
        }
        replay_times = sorted(
            {
                *acquisition_boundaries,
                *missing_observation_boundaries,
                *(item.available_at_days for item in remaining_features),
                *(item.replay_at_days for item in remaining_actions),
                query.query_time_days,
            }
        )
        for replay_time in replay_times:
            arriving_actions = [
                item
                for item in remaining_actions
                if item.replay_at_days <= replay_time + 1e-6
            ]
            state = self._transition(state, arriving_actions, replay_time)
            consumed_action_ids = {item.record_id for item in arriving_actions}
            remaining_actions = [
                item for item in remaining_actions if item.record_id not in consumed_action_ids
            ]
            arriving_features = [
                item
                for item in remaining_features
                if math.isclose(item.available_at_days, replay_time, rel_tol=0.0, abs_tol=1e-6)
            ]
            if arriving_features:
                state = self.model.update_posterior(
                    state,
                    self._merge_feature_events(arriving_features),
                    self._time_tensor(replay_time),
                    deterministic=True,
                )
        combined_flags = _unique((*state.quality_flags, *quality_flags))
        if self.checkpoint.mode is RunMode.SYNTHETIC:
            combined_flags = _unique((*combined_flags, "synthetic_input"))
        result = replace(state, quality_flags=combined_flags)
        result.validate()
        return result

    def replay_query(
        self,
        firewall: FeatureFirewall,
        manifest: FeatureManifest,
        query: Query,
    ) -> ReplayResult:
        """Build and replay one query independently from its model-external prefix."""

        self._validate_mode(firewall, manifest, query.patient_id)
        if query.eligibility is not QueryEligibility.ELIGIBLE:
            raise DataContractError(
                code="QUERY_NOT_ELIGIBLE",
                message="Inference is unavailable for a query outside its declared risk set.",
            )
        prefix = firewall.build_prefix(query.patient_id, query)
        (
            feature_events,
            action_events,
            unavailable_boundaries,
            references,
            flags,
        ) = self._collect_events(prefix, manifest)
        key = self._cache_key(prefix, manifest)
        cached = self.cache.get(key)
        if cached is not None:
            return ReplayResult(
                state=cached,
                cache_key=key,
                input_manifest=references,
                quality_flags=cached.quality_flags,
                cache_hit=True,
                population=prefix.patient.cohort_id,
                query=query,
            )
        with torch.inference_mode():
            state = self._compute_state(
                query,
                feature_events,
                action_events,
                unavailable_boundaries,
                flags,
            )
        self.cache.put(key, state)
        return ReplayResult(
            state=state,
            cache_key=key,
            input_manifest=references,
            quality_flags=state.quality_flags,
            cache_hit=False,
            population=prefix.patient.cohort_id,
            query=query,
        )

    @staticmethod
    def _horizon_grid(query: Query, horizons_days: Sequence[float] | None) -> tuple[float, ...]:
        raw = (
            (0.0, float(query.prediction_horizon_days))
            if horizons_days is None
            else tuple(float(value) for value in horizons_days)
        )
        if not raw:
            raise DataContractError(
                code="EMPTY_PREDICTION_HORIZONS",
                message="At least one prediction horizon is required.",
            )
        values = raw if raw[0] == 0.0 else (0.0, *raw)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise DataContractError(
                code="INVALID_PREDICTION_HORIZONS",
                message="Prediction horizons must be finite and nonnegative.",
            )
        if any(right <= left for left, right in zip(values, values[1:], strict=False)):
            raise DataContractError(
                code="INVALID_PREDICTION_HORIZONS",
                message="Prediction horizons must increase strictly without duplicates.",
            )
        if values[-1] > query.prediction_horizon_days + 1e-6:
            raise DataContractError(
                code="HORIZON_EXCEEDS_QUERY_WINDOW",
                message="Requested curve exceeds the query's declared prediction window.",
            )
        return values

    def _model_horizons(self, days: tuple[float, ...]) -> Tensor:
        divisor = 365.25 if self.survival_time_unit == "year" else 1.0
        return torch.tensor(
            [value / divisor for value in days], dtype=self.dtype, device=self.device
        )

    def predict_query(
        self,
        firewall: FeatureFirewall,
        manifest: FeatureManifest,
        query: Query,
        *,
        endpoint: str = "os",
        horizons_days: Sequence[float] | None = None,
    ) -> PredictionOutput:
        endpoint_normalized = endpoint.strip().lower()
        if endpoint_normalized != self.checkpoint.endpoint:
            raise ArtifactError(
                code="ENDPOINT_VERSION_MISMATCH",
                message="Requested endpoint does not match the loaded checkpoint.",
                details={
                    "requested": endpoint_normalized,
                    "checkpoint": self.checkpoint.endpoint,
                },
            )
        replay = self.replay_query(firewall, manifest, query)
        horizon_days = self._horizon_grid(query, horizons_days)
        with torch.inference_mode():
            prediction = self.model.predict_survival(
                replay.state,
                endpoint_normalized,
                self._model_horizons(horizon_days),
                stage=query.stage.value.upper(),
                simulated=False,
            )
        survival = tuple(float(value) for value in prediction.survival[0].detach().cpu())
        risk = tuple(float(value) for value in prediction.risk[0].detach().cpu())
        rates_tensor = prediction.rates[0].detach().cpu()
        if rates_tensor.ndim == 1:
            rates_tensor = rates_tensor.unsqueeze(-1)
        hazard_rates = tuple(tuple(float(value) for value in interval) for interval in rates_tensor)
        cif = None
        if prediction.cif is not None:
            cif = tuple(
                tuple(float(value) for value in row) for row in prediction.cif[0].detach().cpu()
            )
        return PredictionOutput(
            patient_id=query.patient_id,
            population=replay.population,
            query_id=query.query_id,
            stage=query.stage.value,
            query_time_days=query.query_time_days,
            endpoint=endpoint_normalized,
            horizon_days=horizon_days,
            survival=survival,
            risk=risk,
            hazard_rates=hazard_rates,
            cif=cif,
            checkpoint_id=self.checkpoint.checkpoint_id,
            weight_version=self.checkpoint.weight_version,
            model_version=self.checkpoint.model_version,
            feature_schema_version=manifest.schema_version,
            feature_manifest_lineage_id=manifest.manifest_lineage_id,
            input_manifest=replay.input_manifest,
            scenario=Scenario.observed_history(),
            quality_flags=replay.quality_flags,
            assumptions=(
                "conditional_os_given_event_free_at_query",
                "prognostic_association_not_causal_treatment_effect",
            ),
            uncertainty_summary={
                "method": "deterministic_posterior_mean",
                "clinical_confidence_interval": False,
            },
            simulated=False,
        )

    def predict_patient_history(
        self,
        firewall: FeatureFirewall,
        manifest: FeatureManifest,
        queries: Sequence[Query],
        *,
        endpoint: str = "os",
        horizons_days: Sequence[float] | None = None,
    ) -> tuple[PredictionOutput, ...]:
        """Predict queries in caller order; each query gets an independent legal replay."""

        if not queries:
            raise DataContractError(
                code="EMPTY_QUERY_HISTORY",
                message="Patient history inference requires at least one query.",
            )
        return tuple(
            self.predict_query(
                firewall,
                manifest,
                query,
                endpoint=endpoint,
                horizons_days=horizons_days,
            )
            for query in queries
        )

    def predict_future_observation(
        self,
        state: BeliefState,
        target_modality: str,
        scenario: Scenario,
        *,
        target_time_days: float | None = None,
        actions: ActionTokens | None = None,
    ) -> FutureObservationOutput:
        """Roll out and decode a labelled feature without changing the observed state."""

        if scenario.kind is ScenarioKind.OBSERVED_HISTORY:
            raise DataContractError(
                code="FUTURE_SCENARIO_LABEL_REQUIRED",
                message=(
                    "Future feature decoding requires predicted or hypothetical scenario semantics."
                ),
            )
        with torch.inference_mode():
            simulated_state = clone_inference_state(state)
            if target_time_days is not None or actions is not None:
                target_time = (
                    simulated_state.query_time.detach().clone()
                    if target_time_days is None
                    else self._days_to_timeline(
                        torch.full(
                            (state.memory.shape[0],),
                            float(target_time_days),
                            dtype=self.dtype,
                            device=self.device,
                        )
                    )
                )
                if actions is None:
                    batch_size = state.memory.shape[0]
                    actions = ActionTokens(
                        values=state.memory.new_zeros(
                            batch_size, 1, self.model.config.action_input_dim
                        ),
                        valid=torch.zeros(
                            batch_size, 1, dtype=torch.bool, device=state.memory.device
                        ),
                        event_time=state.query_time.new_zeros(batch_size, 1),
                        available_time=state.query_time.new_zeros(batch_size, 1),
                    )
                simulated_state = self.model.predict_prior(
                    simulated_state,
                    actions,
                    target_time,
                    deterministic=True,
                )
            distribution = self.model.predict_future_observation(
                simulated_state,
                target_modality,
                scenario=scenario.label,
                target_time=simulated_state.query_time,
            )
        return FutureObservationOutput(
            modality=distribution.modality,
            mean=distribution.mean.detach().clone(),
            log_std=distribution.log_std.detach().clone(),
            target_time_days=self._timeline_scalar_to_days(
                float(distribution.target_time[0].item())
            ),
            provenance=distribution.provenance,
            scenario=scenario,
        )
