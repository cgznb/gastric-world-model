"""Typed configuration and safety gates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .errors import ConfigurationError


class RunMode(StrEnum):
    SYNTHETIC = "synthetic"
    REAL_FEATURES = "real_features"
    REAL_IMAGES = "real_images"


@dataclass(frozen=True)
class ProjectSettings:
    name: str = "stageworld_gc"
    mode: RunMode = RunMode.SYNTHETIC
    schema_version: str = "1.0"


@dataclass(frozen=True)
class PathSettings:
    clinical_excel: str | None = None
    field_mapping: str | None = None
    identity_hmac_key_file: str | None = None
    reference_pdf: str | None = None
    ct_root: str | None = None
    pathology_root: str | None = None
    feature_root: str | None = None
    data_artifact_root: str | None = None
    treatment_manifest: str | None = None
    approved_data_root: str | None = None
    output_root: str = "artifacts/synthetic"


@dataclass(frozen=True)
class PrivacySettings:
    allow_phi_in_agent_context: bool = False
    allow_external_tracking: bool = False
    allow_network_during_training: bool = False
    identity_map_outside_repo: bool = True


@dataclass(frozen=True)
class PermissionSettings:
    allow_weight_download: bool = False
    allow_remote_code: bool = False
    allow_long_training: bool = False
    approved_weight_licenses: tuple[str, ...] = ()


@dataclass(frozen=True)
class EncoderSettings:
    ct_weight_path: str | None = None
    ct_source_version: str = "unconfigured-ct-source-v1"
    ct_component_version: str = "unconfigured-ct-component-v1"
    ct_preprocess_version: str = "unconfigured-ct-preprocess-v1"
    ct_inference_precision: str = "bf16"
    segmentation_home: str | None = None


@dataclass(frozen=True)
class ClinicalSettings:
    cohort_definition: str | None = None
    primary_endpoint: str = "os"
    os_event_mapping: Mapping[str, int] | None = None
    baseline_origin_definition: str | None = None
    stage0_definition: str | None = None
    stage1_definition: str | None = None
    stage2_definition: str | None = None
    pathology_availability_basis: str | None = None
    recurrence_definition: str | None = None
    pcr_definition: str | None = None
    clinical_signoff_version: str | None = None
    endpoint_confirmed: bool = False
    time_contract_confirmed: bool = False
    development_stages: tuple[str, ...] = ("s0", "s1", "s2")
    ct_series_selection_policy_version: str = "paired-ct-phase-selection-v1"
    treatment_summary_cutoff: str | None = None


@dataclass(frozen=True)
class ModelSettings:
    survival_task: str = "s0_s1_pred"
    dropout: float = 0.1
    ct_encoder: str = "merlin"
    pathology_encoder: str = "titan_conch_v1_5"
    pathology_alternatives: tuple[str, ...] = ("uni2_h_mil", "prism2_base")
    freeze_foundation_encoders: bool = True
    hidden_dim: int = 256
    state_tokens: int = 24
    stochastic_dim_per_token: int = 16
    use_stochastic_state: bool = True
    attention_heads: int = 8
    transition_blocks: int = 4
    observation_blocks: int = 2
    resampler_blocks: int = 2
    ct_tokens: int = 16
    pathology_tokens: int = 8
    clinical_input_dim: int = 8
    action_input_dim: int = 8
    ct_input_dim: int = 32
    pathology_input_dim: int = 32
    use_llm_treatment_planner: bool = False
    use_pixel_generator: bool = False
    use_counterfactual_diversity: bool = False


@dataclass(frozen=True)
class SurvivalSettings:
    parameterization: str = "piecewise_constant_hazard_rate"
    time_unit: str = "year"
    finite_cutpoints: tuple[float, ...] = (0.0, 1.0, 3.0, 5.0)
    open_tail_interval: bool = True
    report_horizons_years: tuple[float, ...] = (1.0, 3.0)
    competing_risks_enabled: bool = False
    num_causes: int = 1


@dataclass(frozen=True)
class TrainingSettings:
    joint_backbone_lr: float = 2e-5
    lr_warmup_epochs: int = 5
    lr_min_ratio: float = 0.1
    development_min_delta: float = 1e-4
    phases: tuple[str, ...] = ("baseline", "world_pretrain", "joint_survival")
    seed: int = 17
    comparison_seeds: tuple[int, ...] = (17, 43, 97)
    patient_batch_size: int = 16
    optimizer: str = "adamw"
    lr: float = 2e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    mixed_precision: str = "auto"
    target_encoder_policy: str = "frozen_external"
    patient_normalized_losses: bool = True
    checkpoint_selection: str = "preregistered_stage_average_validation_ibs"
    smoke_max_steps: int = 20
    smoke_max_minutes: int = 5
    world_pretrain_steps: int = 12
    joint_survival_steps: int = 16
    development_protocol: str | None = None
    development_max_epochs: int = 60
    development_patience: int | None = 10
    development_max_minutes: int | None = 180


@dataclass(frozen=True)
class EvaluationSettings:
    split_strategy: str | None = None
    external_manifest: str | None = None
    patient_level_resampling: bool = True
    forbid_test_tuning: bool = True
    report_non_estimable_metrics: bool = True
    bootstrap_replicates: int = 100


@dataclass(frozen=True)
class StageWorldConfig:
    project: ProjectSettings = field(default_factory=ProjectSettings)
    paths: PathSettings = field(default_factory=PathSettings)
    privacy: PrivacySettings = field(default_factory=PrivacySettings)
    permissions: PermissionSettings = field(default_factory=PermissionSettings)
    encoders: EncoderSettings = field(default_factory=EncoderSettings)
    clinical: ClinicalSettings = field(default_factory=ClinicalSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    survival: SurvivalSettings = field(default_factory=SurvivalSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def mode(self) -> RunMode:
        return self.project.mode

    @property
    def output_root(self) -> Path:
        return Path(self.paths.output_root)

    def safety_findings(self) -> list[dict[str, str]]:
        findings: list[dict[str, str]] = []
        if self.privacy.allow_phi_in_agent_context:
            findings.append({"code": "PHI_CONTEXT_FORBIDDEN", "severity": "error"})
        if self.privacy.allow_external_tracking:
            findings.append({"code": "EXTERNAL_TRACKING_FORBIDDEN", "severity": "error"})
        if self.privacy.allow_network_during_training:
            findings.append({"code": "NETWORK_DURING_TRAINING_FORBIDDEN", "severity": "error"})
        if not self.privacy.identity_map_outside_repo:
            findings.append({"code": "IDENTITY_MAP_MUST_BE_EXTERNAL", "severity": "error"})
        binary_protocol = (
            self.clinical.primary_endpoint == "pcr_recurrence"
            and self.training.development_protocol == "ct6-pcr-recurrence-cv5-v1"
        )
        if self.clinical.primary_endpoint.lower() != "os" and not binary_protocol:
            findings.append({"code": "NON_OS_ENDPOINT_NOT_CONFIRMED", "severity": "error"})
        if self.model.use_llm_treatment_planner:
            findings.append({"code": "TREATMENT_PLANNER_OUT_OF_SCOPE", "severity": "error"})
        if self.model.use_counterfactual_diversity:
            findings.append({"code": "COUNTERFACTUAL_DIVERSITY_OUT_OF_SCOPE", "severity": "error"})
        if self.model.use_pixel_generator:
            findings.append({"code": "PIXEL_GENERATOR_NOT_MAIN_PATH", "severity": "error"})
        if self.encoders.ct_inference_precision not in {"off", "fp16", "bf16"}:
            findings.append({"code": "INVALID_CT_INFERENCE_PRECISION", "severity": "error"})
        return findings

    def clinical_blockers(self) -> list[str]:
        required: dict[str, object] = {
            "cohort_definition": self.clinical.cohort_definition,
            "os_event_mapping": self.clinical.os_event_mapping,
            "baseline_origin_definition": self.clinical.baseline_origin_definition,
            "clinical_signoff_version": self.clinical.clinical_signoff_version,
        }
        stage_requirements = {
            "s0": {"stage0_definition": self.clinical.stage0_definition},
            "s1": {"stage1_definition": self.clinical.stage1_definition},
            "s2": {
                "stage2_definition": self.clinical.stage2_definition,
                "pathology_availability_basis": self.clinical.pathology_availability_basis,
            },
        }
        for stage in self.clinical.development_stages:
            required.update(stage_requirements.get(stage, {}))
        missing = [name for name, value in required.items() if value in (None, "", {})]
        if not self.clinical.endpoint_confirmed:
            missing.append("endpoint_confirmed")
        if not self.clinical.time_contract_confirmed:
            missing.append("time_contract_confirmed")
        return missing

    def validate(self, *, command: str, supervised: bool = False) -> None:
        findings = self.safety_findings()
        if findings:
            raise ConfigurationError(
                code="UNSAFE_CONFIGURATION",
                message="Configuration violates StageWorld safety boundaries.",
                remediation="Disable prohibited data/network/model options.",
                details={"findings": findings},
            )
        if self.model.hidden_dim <= 0 or self.model.hidden_dim % self.model.attention_heads:
            raise ConfigurationError(
                code="INVALID_MODEL_DIMENSIONS",
                message="hidden_dim must be positive and divisible by attention_heads.",
            )
        stages = self.clinical.development_stages
        if (
            not stages
            or len(stages) != len(set(stages))
            or any(stage not in {"s0", "s1", "s2"} for stage in stages)
        ):
            raise ConfigurationError(
                code="INVALID_DEVELOPMENT_STAGES",
                message="clinical.development_stages must contain unique s0, s1, or s2 values.",
            )
        if self.clinical.ct_series_selection_policy_version not in {
            "paired-ct-phase-selection-v1",
            "paired-ct-first-acquisition-selection-v1",
        }:
            raise ConfigurationError(
                code="INVALID_CT_SERIES_SELECTION_POLICY",
                message="The configured CT series-selection policy is not supported.",
                details={
                    "policy": self.clinical.ct_series_selection_policy_version,
                },
            )
        cuts = self.survival.finite_cutpoints
        if (
            len(cuts) < 2
            or cuts[0] != 0.0
            or any(b <= a for a, b in zip(cuts, cuts[1:], strict=False))
        ):
            raise ConfigurationError(
                code="INVALID_SURVIVAL_CUTPOINTS",
                message="finite_cutpoints must start at zero and increase strictly.",
            )
        if self.survival.time_unit not in {"day", "year"}:
            raise ConfigurationError(
                code="INVALID_TIME_UNIT",
                message="survival.time_unit must be 'day' or 'year'.",
            )
        if supervised and self.mode is not RunMode.SYNTHETIC:
            blockers = self.clinical_blockers()
            if blockers:
                raise ConfigurationError(
                    code="CLINICAL_SIGNOFF_REQUIRED",
                    message=(
                        "Real outcome-supervised execution is blocked by unsigned clinical fields."
                    ),
                    remediation=(
                        "Resolve the consolidated checklist in BLOCKERS.md and version the signoff."
                    ),
                    details={"missing": blockers, "command": command},
                )
        if self.mode is not RunMode.SYNTHETIC:
            self._validate_real_paths(command)

    def _validate_real_paths(self, command: str) -> None:
        if not self.paths.approved_data_root:
            raise ConfigurationError(
                code="APPROVED_DATA_ROOT_REQUIRED",
                message="Real modes require paths.approved_data_root.",
            )
        approved = Path(self.paths.approved_data_root).expanduser().resolve()
        relevant = {
            "clinical_excel": self.paths.clinical_excel,
            "ct_root": self.paths.ct_root,
            "pathology_root": self.paths.pathology_root,
        }
        for name, raw_path in relevant.items():
            if raw_path is None:
                continue
            candidate = Path(raw_path).expanduser().resolve()
            if not candidate.is_relative_to(approved):
                raise ConfigurationError(
                    code="PATH_OUTSIDE_APPROVED_ROOT",
                    message=f"{name} is outside paths.approved_data_root.",
                    details={"field": name, "command": command},
                )


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigurationError(
            code="INVALID_CONFIG_SECTION",
            message=f"Configuration section '{name}' must be a mapping.",
        )
    return dict(value)


def load_config(path: str | Path) -> StageWorldConfig:
    source = Path(path)
    if not source.is_file():
        raise ConfigurationError(
            code="CONFIG_NOT_FOUND",
            message="Configuration file does not exist.",
            details={"path": str(source)},
        )
    loaded = OmegaConf.load(source)
    OmegaConf.resolve(loaded)
    raw = OmegaConf.to_container(loaded, resolve=True)
    if not isinstance(raw, Mapping):
        raise ConfigurationError(
            code="INVALID_CONFIG", message="Configuration root must be a mapping."
        )

    project_raw = _mapping(raw.get("project"), "project")
    try:
        project_raw["mode"] = RunMode(project_raw.get("mode", RunMode.SYNTHETIC))
    except ValueError as exc:
        raise ConfigurationError(
            code="INVALID_RUN_MODE",
            message="project.mode must be synthetic, real_features, or real_images.",
        ) from exc

    permissions_raw = _mapping(raw.get("permissions"), "permissions")
    permissions_raw["approved_weight_licenses"] = tuple(
        permissions_raw.get("approved_weight_licenses", ()) or ()
    )
    model_raw = _mapping(raw.get("model"), "model")
    model_raw["pathology_alternatives"] = tuple(
        model_raw.get("pathology_alternatives", ("uni2_h_mil", "prism2_base")) or ()
    )
    survival_raw = _mapping(raw.get("survival"), "survival")
    survival_raw["finite_cutpoints"] = tuple(
        float(x) for x in survival_raw.get("finite_cutpoints", (0.0, 1.0, 3.0, 5.0))
    )
    survival_raw["report_horizons_years"] = tuple(
        float(x) for x in survival_raw.get("report_horizons_years", (1.0, 3.0))
    )
    training_raw = _mapping(raw.get("training"), "training")
    training_raw["phases"] = tuple(
        training_raw.get("phases", ("baseline", "world_pretrain", "joint_survival"))
    )
    training_raw["comparison_seeds"] = tuple(training_raw.get("comparison_seeds", (17, 43, 97)))
    clinical_raw = _mapping(raw.get("clinical"), "clinical")
    clinical_raw["development_stages"] = tuple(
        clinical_raw.get("development_stages", ("s0", "s1", "s2"))
    )

    return StageWorldConfig(
        project=ProjectSettings(**project_raw),
        paths=PathSettings(**_mapping(raw.get("paths"), "paths")),
        privacy=PrivacySettings(**_mapping(raw.get("privacy"), "privacy")),
        permissions=PermissionSettings(**permissions_raw),
        encoders=EncoderSettings(**_mapping(raw.get("encoders"), "encoders")),
        clinical=ClinicalSettings(**clinical_raw),
        model=ModelSettings(**model_raw),
        survival=SurvivalSettings(**survival_raw),
        training=TrainingSettings(**training_raw),
        evaluation=EvaluationSettings(**_mapping(raw.get("evaluation"), "evaluation")),
        raw={str(key): value for key, value in raw.items()},
    )
