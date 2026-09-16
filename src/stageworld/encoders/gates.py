"""Offline-only access gates for foundation-model adapters."""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from stageworld.config import RunMode
from stageworld.errors import ArtifactError, ConfigurationError, PermissionGateError

_DIGEST_ONLY = re.compile(r"^[0-9a-fA-F]{7,128}$")


def validate_named_version(value: str, *, field_name: str) -> None:
    """Require a readable release/revision label without persisting a digest."""

    if not value or _DIGEST_ONLY.fullmatch(value.strip()):
        raise ConfigurationError(
            code="UNNAMED_ENCODER_VERSION",
            message=f"{field_name} must be a non-digest, human-readable version.",
            remediation="Use an approved release name or a named local revision.",
            details={"field": field_name},
        )


@dataclass(frozen=True)
class EncoderAccess:
    """All authority needed before an adapter may execute real image code."""

    mode: RunMode
    source_version: str
    component_versions: Mapping[str, str]
    preprocess_version: str
    version_approved: bool = False
    license_name: str | None = None
    license_approved: bool = False
    weight_paths: Mapping[str, Path] = field(default_factory=dict)
    remote_code_path: Path | None = None
    remote_code_approved: bool = False
    network_enabled: bool = False

    def validate_versions(self, required_components: tuple[str, ...]) -> None:
        validate_named_version(self.source_version, field_name="source_version")
        validate_named_version(self.preprocess_version, field_name="preprocess_version")
        missing = [name for name in required_components if name not in self.component_versions]
        if missing:
            raise ConfigurationError(
                code="MISSING_COMPONENT_VERSION",
                message="Every required encoder component needs an explicit version.",
                details={"missing_components": missing},
            )
        for name in required_components:
            validate_named_version(self.component_versions[name], field_name=f"component:{name}")
        if not self.version_approved:
            raise PermissionGateError(
                code="ENCODER_VERSION_NOT_APPROVED",
                message="The named encoder source and component versions are not approved.",
            )

    def validate_feature_access(self, required_components: tuple[str, ...]) -> None:
        if self.mode is not RunMode.REAL_FEATURES:
            raise ConfigurationError(
                code="FEATURE_MODE_REQUIRED",
                message="Precomputed real features require mode=real_features.",
            )
        if self.network_enabled:
            raise PermissionGateError(
                code="ENCODER_NETWORK_FORBIDDEN",
                message="Encoder execution must remain offline.",
            )
        self.validate_versions(required_components)
        if not self.license_name or not self.license_approved:
            raise PermissionGateError(
                code="ENCODER_LICENSE_NOT_APPROVED",
                message="Use of cached foundation features requires approved source terms.",
            )

    def validate_image_access(
        self,
        *,
        required_components: tuple[str, ...],
        required_weights: tuple[str, ...],
        required_dependencies: tuple[str, ...] = (),
        requires_remote_code: bool = False,
    ) -> None:
        if self.mode is not RunMode.REAL_IMAGES:
            raise ConfigurationError(
                code="IMAGE_MODE_REQUIRED",
                message="Foundation-model image extraction requires mode=real_images.",
            )
        if self.network_enabled:
            raise PermissionGateError(
                code="ENCODER_NETWORK_FORBIDDEN",
                message="Encoder execution must remain offline; network loaders are prohibited.",
            )
        self.validate_versions(required_components)
        if not self.license_name or not self.license_approved:
            raise PermissionGateError(
                code="ENCODER_LICENSE_NOT_APPROVED",
                message="The encoder license/access terms require explicit approval.",
            )
        missing_weights = [
            name
            for name in required_weights
            if name not in self.weight_paths or not Path(self.weight_paths[name]).is_file()
        ]
        if missing_weights:
            raise ArtifactError(
                code="ENCODER_WEIGHTS_MISSING",
                message="Required local encoder weights are missing.",
                remediation="Provide approved local weight files; automatic download is disabled.",
                details={"missing_components": missing_weights},
            )
        missing_dependencies = [
            name for name in required_dependencies if importlib.util.find_spec(name) is None
        ]
        if missing_dependencies:
            raise ArtifactError(
                code="ENCODER_DEPENDENCY_MISSING",
                message="Required encoder dependencies are not installed.",
                details={"missing_dependencies": missing_dependencies},
            )
        if requires_remote_code:
            if not self.remote_code_approved:
                raise PermissionGateError(
                    code="REMOTE_CODE_NOT_APPROVED",
                    message="This adapter requires separately audited local remote code.",
                )
            if self.remote_code_path is None or not self.remote_code_path.is_dir():
                raise ArtifactError(
                    code="REMOTE_CODE_MISSING",
                    message="The approved local remote-code directory is missing.",
                )
