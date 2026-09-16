"""Structured failures for configuration, data, and runtime gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(eq=False)
class StageWorldError(RuntimeError):
    """An expected hard failure with a stable machine-readable code."""

    code: str
    message: str
    remediation: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, self.message)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.remediation:
            payload["remediation"] = self.remediation
        if self.details:
            payload["details"] = self.details
        return payload


class ConfigurationError(StageWorldError):
    """Invalid or unsafe project configuration."""


class DataContractError(StageWorldError):
    """Input data violate a typed or temporal contract."""


class PermissionGateError(StageWorldError):
    """An explicitly gated operation was requested without approval."""


class ArtifactError(StageWorldError):
    """An artifact is incomplete, incompatible, or lacks provenance."""


class ResourceError(StageWorldError):
    """A bounded operation cannot continue with the available local resources."""
