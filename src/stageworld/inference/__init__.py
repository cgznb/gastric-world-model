"""Public feature-history inference and state-cache API."""

from .cache import (
    InferenceStateCache,
    PrefixEntry,
    PrefixIdentity,
    StateCacheKey,
    clone_inference_state,
)
from .contracts import (
    CHECKPOINT_SCHEMA_VERSION,
    FEATURE_INPUT_SCHEMA_VERSION,
    PREDICTION_SCHEMA_VERSION,
    ActionFeature,
    CheckpointContract,
    FeatureInputContract,
    FeatureManifest,
    FutureObservationOutput,
    IdentityStatus,
    InputReference,
    ModalityFeatureContract,
    PredictionOutput,
    Scenario,
    ScenarioKind,
)
from .engine import InferenceEngine, ReplayResult

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "FEATURE_INPUT_SCHEMA_VERSION",
    "PREDICTION_SCHEMA_VERSION",
    "ActionFeature",
    "CheckpointContract",
    "FeatureInputContract",
    "FeatureManifest",
    "FutureObservationOutput",
    "IdentityStatus",
    "InferenceEngine",
    "InferenceStateCache",
    "InputReference",
    "ModalityFeatureContract",
    "PredictionOutput",
    "PrefixEntry",
    "PrefixIdentity",
    "ReplayResult",
    "Scenario",
    "ScenarioKind",
    "StateCacheKey",
    "clone_inference_state",
]
