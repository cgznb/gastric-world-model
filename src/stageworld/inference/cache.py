"""Thread-safe inference-state cache with explicit temporal identity."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock

from stageworld.errors import ConfigurationError
from stageworld.model import BeliefState


@dataclass(frozen=True, slots=True)
class PrefixEntry:
    record_kind: str
    record_id: str
    acquired_or_event_time_days: float | None
    available_at_days: float
    role_or_status: str
    feature_version: str


@dataclass(frozen=True, slots=True)
class PrefixIdentity:
    """Hashable canonical prefix description; it is not a persisted checksum."""

    feature_schema_version: str
    feature_manifest_lineage_id: str
    entries: tuple[PrefixEntry, ...]


@dataclass(frozen=True, slots=True)
class StateCacheKey:
    patient_id: str
    prefix: PrefixIdentity
    stage: str
    query_time_days: float
    checkpoint_id: str
    weight_version: str
    model_version: str


def clone_inference_state(state: BeliefState) -> BeliefState:
    """Return a detached tensor copy so cache callers cannot mutate shared state."""

    return BeliefState(
        memory=state.memory.detach().clone(),
        query_time=state.query_time.detach().clone(),
        stochastic_mean=(
            None if state.stochastic_mean is None else state.stochastic_mean.detach().clone()
        ),
        stochastic_log_std=(
            None if state.stochastic_log_std is None else state.stochastic_log_std.detach().clone()
        ),
        sample=None if state.sample is None else state.sample.detach().clone(),
        state_kind=state.state_kind,
        provenance=tuple(state.provenance),
        quality_flags=tuple(state.quality_flags),
    )


class InferenceStateCache:
    """Bounded in-memory LRU cache; no patient state is written to disk."""

    def __init__(self, max_entries: int = 256) -> None:
        if max_entries <= 0:
            raise ConfigurationError(
                code="INVALID_STATE_CACHE_SIZE",
                message="Inference state cache size must be positive.",
            )
        self.max_entries = max_entries
        self._values: OrderedDict[StateCacheKey, BeliefState] = OrderedDict()
        self._lock = RLock()

    def get(self, key: StateCacheKey) -> BeliefState | None:
        with self._lock:
            state = self._values.get(key)
            if state is None:
                return None
            self._values.move_to_end(key)
            return clone_inference_state(state)

    def put(self, key: StateCacheKey, state: BeliefState) -> None:
        state.validate()
        if state.memory.shape[0] != 1:
            raise ConfigurationError(
                code="STATE_CACHE_BATCH_UNSUPPORTED",
                message="Inference state cache stores one anonymous patient per entry.",
            )
        with self._lock:
            self._values[key] = clone_inference_state(state)
            self._values.move_to_end(key)
            while len(self._values) > self.max_entries:
                self._values.popitem(last=False)

    def clear_patient(self, patient_id: str) -> int:
        with self._lock:
            targets = [key for key in self._values if key.patient_id == patient_id]
            for key in targets:
                del self._values[key]
            return len(targets)

    def keys(self) -> tuple[StateCacheKey, ...]:
        with self._lock:
            return tuple(self._values)

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)
