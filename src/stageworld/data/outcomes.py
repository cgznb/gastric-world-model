"""Outcome-only label construction and landmark eligibility."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from stageworld.errors import ConfigurationError, DataContractError

from .contracts import (
    AdjudicationStatus,
    DataMode,
    EventType,
    Landmark,
    LandmarkExclusion,
    LandmarkSet,
    Outcome,
    Query,
    QueryEligibility,
    SurvivalLabel,
)
from .firewall import FeatureFirewall


class ZeroTimePolicy(StrEnum):
    EXCLUDE = "exclude"
    ALLOW_EVENT = "allow_event"


@dataclass(frozen=True, slots=True)
class OutcomeDefinition:
    endpoint_name: str
    event_type: EventType
    event_code: str | int | bool | None
    origin_definition: str | None
    label_version: str
    mode: DataMode
    status_mapping_confirmed: bool = False
    origin_confirmed: bool = False
    timeline_confirmed: bool = False
    zero_time_policy: ZeroTimePolicy = ZeroTimePolicy.EXCLUDE

    def validate(self) -> None:
        missing: list[str] = []
        if self.event_code is None:
            missing.append("event_code")
        if not self.origin_definition:
            missing.append("origin_definition")
        if not self.status_mapping_confirmed:
            missing.append("status_mapping_confirmed")
        if not self.origin_confirmed:
            missing.append("origin_confirmed")
        if not self.timeline_confirmed:
            missing.append("timeline_confirmed")
        if missing:
            raise ConfigurationError(
                "endpoint_contract_unconfirmed",
                "Outcome-supervised label construction requires a confirmed endpoint contract",
                remediation="Obtain clinical sign-off for event coding, origin, and timeline",
                details={"missing": missing, "mode": self.mode.value},
            )


class OutcomeBuilder:
    """Build labels from the isolated outcomes table only."""

    def __init__(self, outcomes: Iterable[Outcome], definition: OutcomeDefinition) -> None:
        definition.validate()
        self.definition = definition
        selected = [
            outcome
            for outcome in outcomes
            if outcome.endpoint_name == definition.endpoint_name
        ]
        self._outcomes: dict[str, Outcome] = {}
        for outcome in selected:
            if outcome.patient_id in self._outcomes:
                raise DataContractError(
                    "duplicate_outcome",
                    "A patient has more than one record for the configured endpoint",
                    details={"patient_id": outcome.patient_id},
                )
            self._outcomes[outcome.patient_id] = outcome

    @staticmethod
    def _normalized_status(value: object) -> str:
        return str(value).strip().casefold()

    def _terminal(self, patient_id: str) -> tuple[Outcome, float, bool]:
        try:
            outcome = self._outcomes[patient_id]
        except KeyError as exc:
            raise DataContractError(
                "outcome_not_found", "No configured endpoint record exists for the patient"
            ) from exc
        if outcome.adjudication_status is not AdjudicationStatus.CONFIRMED:
            raise DataContractError(
                "outcome_not_adjudicated",
                "Outcome record is not confirmed",
                details={"patient_id": patient_id},
            )
        if outcome.event_type is not self.definition.event_type:
            raise DataContractError(
                "event_type_mismatch", "Outcome event type differs from endpoint definition"
            )
        if outcome.origin_definition != self.definition.origin_definition:
            raise DataContractError(
                "outcome_origin_mismatch", "Outcome and configured origin definitions differ"
            )
        is_event = self._normalized_status(outcome.source_status) == self._normalized_status(
            self.definition.event_code
        )
        if is_event:
            if outcome.event_date_days is None:
                raise DataContractError(
                    "event_date_missing", "An observed endpoint event requires an event date"
                )
            if (
                outcome.censor_date_days is not None
                and outcome.censor_date_days < outcome.event_date_days
            ):
                raise DataContractError(
                    "censor_before_event",
                    "Censor date precedes an event marked as observed",
                )
            return outcome, outcome.event_date_days, True
        if outcome.event_date_days is not None:
            raise DataContractError(
                "event_date_status_conflict",
                "A non-event status carries an event date and requires adjudication",
            )
        if outcome.censor_date_days is None:
            raise DataContractError(
                "censor_date_missing", "A non-event endpoint requires a censor date"
            )
        return outcome, outcome.censor_date_days, False

    def terminal_time(self, patient_id: str) -> tuple[float, bool]:
        _, terminal, event = self._terminal(patient_id)
        return terminal, event

    def build_label(self, patient_id: str, query_time_days: float) -> SurvivalLabel:
        outcome, terminal, event = self._terminal(patient_id)
        remaining = terminal - query_time_days
        if remaining < 0:
            raise DataContractError(
                "negative_remaining_time",
                "The endpoint terminal time precedes the query; it must not be clipped",
                details={"patient_id": patient_id},
            )
        if remaining == 0 and not (
            event and self.definition.zero_time_policy is ZeroTimePolicy.ALLOW_EVENT
        ):
            raise DataContractError(
                "zero_remaining_time",
                "A same-time event or censor requires an explicit endpoint policy",
                details={"patient_id": patient_id},
            )
        return SurvivalLabel(
            patient_id=patient_id,
            endpoint_name=self.definition.endpoint_name,
            query_time_days=query_time_days,
            remaining_time_days=remaining,
            event=event,
            label_version=outcome.label_version,
        )


class LandmarkBuilder:
    """Join legal prefixes to labels only after independent construction."""

    def __init__(self, firewall: FeatureFirewall, outcome_builder: OutcomeBuilder) -> None:
        self.firewall = firewall
        self.outcome_builder = outcome_builder

    def build(self, queries: Iterable[Query]) -> LandmarkSet:
        landmarks: list[Landmark] = []
        exclusions: list[LandmarkExclusion] = []
        for query in queries:
            if query.eligibility is QueryEligibility.UNKNOWN:
                raise DataContractError(
                    "query_eligibility_unknown",
                    "Landmark eligibility must be established without future completion data",
                    details={"query_id": query.query_id},
                )
            if query.eligibility is QueryEligibility.NOT_APPLICABLE:
                exclusions.append(
                    LandmarkExclusion(query.query_id, query.patient_id, "node_not_applicable")
                )
                continue
            terminal, event = self.outcome_builder.terminal_time(query.patient_id)
            remaining = terminal - query.query_time_days
            if remaining < 0:
                reason = "event_already_occurred" if event else "followup_ended_before_query"
                exclusions.append(LandmarkExclusion(query.query_id, query.patient_id, reason))
                continue
            if remaining == 0 and not (
                event
                and self.outcome_builder.definition.zero_time_policy
                is ZeroTimePolicy.ALLOW_EVENT
            ):
                exclusions.append(
                    LandmarkExclusion(query.query_id, query.patient_id, "zero_followup_at_query")
                )
                continue
            prefix = self.firewall.build_prefix(query.patient_id, query)
            label = self.outcome_builder.build_label(query.patient_id, query.query_time_days)
            landmarks.append(
                Landmark(
                    landmark_id=f"{query.query_id}:{self.outcome_builder.definition.endpoint_name}",
                    patient_id=query.patient_id,
                    stage=query.stage,
                    query_time_days=query.query_time_days,
                    prefix=prefix,
                    label=label,
                )
            )
        return LandmarkSet(tuple(landmarks), tuple(exclusions))
