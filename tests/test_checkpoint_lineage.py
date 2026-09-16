from __future__ import annotations

import math
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from stageworld.config import StageWorldConfig, load_config
from stageworld.errors import ArtifactError
from stageworld.inference import CHECKPOINT_SCHEMA_VERSION, CheckpointContract
from stageworld.synthetic_workflow import (
    BUILD_SCHEMA,
    FEATURE_SCHEMA,
    SOURCE_SCHEMA,
    _config_lineage,
    _validate_checkpoint_for_run,
    build_model,
)
from stageworld.training import TrainingPhase

ROOT = Path(__file__).resolve().parents[1]


def _config() -> StageWorldConfig:
    return load_config(ROOT / "configs/project.synthetic.yaml")


def _contract(config: StageWorldConfig, **changes: object) -> CheckpointContract:
    model = build_model(config)
    values: dict[str, object] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_id": "checkpoint-lineage-test-v1",
        "weight_version": "weights-lineage-test-v1",
        "model_version": model.config.model_version,
        "model_config": asdict(model.config),
        "endpoint": "os",
        "survival_contract": {
            "schema_version": "stageworld-survival-contract-v1",
            "endpoint": "os",
            "parameterization": config.survival.parameterization,
            "time_unit": config.survival.time_unit,
            "cutpoints": model.config.survival_cutpoints,
            "open_tail_interval": config.survival.open_tail_interval,
            "num_causes": model.config.survival_causes,
        },
        "mode": config.mode.value,
        "config_lineage_id": _config_lineage(config),
        "data_lineage_id": "synthetic-data-lineage-v1",
        "cohort_artifact_id": "synthetic-cohort-artifact-v1",
        "split_version": "synthetic-split-lineage-v1",
        "ct_feature_artifact_id": "synthetic-ct-features-v1",
        "pathology_feature_artifact_id": "synthetic-pathology-features-v1",
        "timeline_contract_version": "synthetic-timeline-contract-v1",
        "outcome_contract_version": "synthetic-outcome-contract-v1",
        "training_seed": config.training.seed,
        "source_schema_version": SOURCE_SCHEMA,
        "cohort_schema_version": BUILD_SCHEMA,
        "feature_schema_version": FEATURE_SCHEMA,
        "phase": TrainingPhase.JOINT_SURVIVAL.value,
        "step": 3,
        "parent_checkpoint_id": "parent-checkpoint-lineage-test-v1",
        "parent_weight_version": "parent-weights-lineage-test-v1",
        "parent_phase": TrainingPhase.WORLD_PRETRAIN.value,
        "parent_config_lineage_id": _config_lineage(config),
        "parent_data_lineage_id": "synthetic-data-lineage-v1",
        "parent_cohort_artifact_id": "synthetic-cohort-artifact-v1",
        "parent_split_version": "synthetic-split-lineage-v1",
        "parent_ct_feature_artifact_id": "synthetic-ct-features-v1",
        "parent_pathology_feature_artifact_id": "synthetic-pathology-features-v1",
        "parent_timeline_contract_version": "synthetic-timeline-contract-v1",
        "parent_outcome_contract_version": "synthetic-outcome-contract-v1",
    }
    values.update(changes)
    return CheckpointContract.from_mapping(values)


def _validate(contract: CheckpointContract, config: StageWorldConfig) -> None:
    _validate_checkpoint_for_run(
        contract,
        config,
        build_model(config),
        data_lineage_id="synthetic-data-lineage-v1",
        cohort_artifact_id="synthetic-cohort-artifact-v1",
        split_version="synthetic-split-lineage-v1",
        ct_feature_artifact_id="synthetic-ct-features-v1",
        pathology_feature_artifact_id="synthetic-pathology-features-v1",
        timeline_contract_version="synthetic-timeline-contract-v1",
        outcome_contract_version="synthetic-outcome-contract-v1",
        phase=TrainingPhase.JOINT_SURVIVAL,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("training_seed", 18),
        ("source_schema_version", "legacy-source-v1"),
        ("cohort_schema_version", "legacy-cohort-v1"),
        ("feature_schema_version", "legacy-features-v1"),
        ("cohort_artifact_id", "other-cohort-v1"),
        ("split_version", "other-split-v1"),
        ("ct_feature_artifact_id", "other-ct-features-v1"),
        ("pathology_feature_artifact_id", "other-pathology-features-v1"),
        ("timeline_contract_version", "other-timeline-v1"),
        ("outcome_contract_version", "other-outcome-v1"),
    ],
)
def test_runtime_gate_rejects_seed_schema_and_split_lineage(
    field: str, value: object
) -> None:
    config = _config()
    with pytest.raises(ArtifactError) as error:
        _validate(_contract(config, **{field: value}), config)
    assert error.value.code == "CHECKPOINT_RUNTIME_CONTRACT_MISMATCH"


@pytest.mark.parametrize(
    "survival_changes",
    [
        {"finite_cutpoints": (0.0, 1.5, 3.0, 5.0)},
        {"time_unit": "day"},
        {"open_tail_interval": False},
        {"num_causes": 2, "competing_risks_enabled": True},
    ],
)
def test_runtime_gate_rejects_changed_survival_contract(
    survival_changes: dict[str, object],
) -> None:
    config = _config()
    changed = replace(config, survival=replace(config.survival, **survival_changes))
    with pytest.raises(ArtifactError) as error:
        _validate(_contract(config), changed)
    assert error.value.code == "CHECKPOINT_SURVIVAL_CONTRACT_MISMATCH"


def test_runtime_gate_rejects_same_shape_model_config_masquerade() -> None:
    config = _config()
    model_config = asdict(build_model(config).config)
    contract = _contract(config, model_config={**model_config, "max_rollout_days": 999.0})
    with pytest.raises(ArtifactError) as error:
        _validate(contract, config)
    assert error.value.code == "CHECKPOINT_MODEL_CONFIG_MISMATCH"


@pytest.mark.parametrize(
    "section,changes",
    [
        ("training", {"patient_batch_size": 4}),
        ("training", {"grad_clip_norm": 2.0}),
        ("training", {"world_pretrain_steps": 99}),
        ("model", {"ct_encoder": "same-shape-other-encoder"}),
        ("survival", {"report_horizons_years": (2.0, 4.0)}),
    ],
)
def test_runtime_gate_rejects_changed_explicit_config_lineage(
    section: str, changes: dict[str, object]
) -> None:
    config = _config()
    changed_section = replace(getattr(config, section), **changes)
    changed = replace(config, **{section: changed_section})
    with pytest.raises(ArtifactError) as error:
        _validate(_contract(config), changed)
    assert error.value.code == "CHECKPOINT_RUNTIME_CONTRACT_MISMATCH"
    assert "config_lineage_id" in error.value.details["fields"]


def test_runtime_gate_accepts_exact_checkpoint_contract() -> None:
    config = _config()
    _validate(_contract(config), config)


def test_joint_checkpoint_contract_requires_complete_parent_weight_lineage() -> None:
    config = _config()
    with pytest.raises(ArtifactError) as error:
        _contract(config, parent_weight_version=None)
    assert error.value.code == "INCOMPLETE_PARENT_CHECKPOINT_LINEAGE"


def test_config_lineage_distinguishes_adjacent_floating_point_values() -> None:
    config = _config()
    changed_lr = replace(
        config,
        training=replace(config.training, lr=math.nextafter(config.training.lr, math.inf)),
    )
    first_horizon = config.survival.report_horizons_years[0]
    changed_horizon = replace(
        config,
        survival=replace(
            config.survival,
            report_horizons_years=(
                math.nextafter(first_horizon, math.inf),
                *config.survival.report_horizons_years[1:],
            ),
        ),
    )

    assert _config_lineage(config) != _config_lineage(changed_lr)
    assert _config_lineage(config) != _config_lineage(changed_horizon)
