"""Outcome-isolated S0/S1 OS development on frozen, ROI-localized CT features."""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from stageworld.artifacts import (
    atomic_write_json,
    atomic_write_private_json,
    new_artifact_id,
    read_json,
)
from stageworld.config import StageWorldConfig
from stageworld.data.contracts import AdjudicationStatus, DataMode, EventType, Outcome
from stageworld.data.gastric_roi import GASTRIC_ROI_VERSION, offline_network
from stageworld.data.outcomes import OutcomeBuilder, OutcomeDefinition
from stageworld.data.paired_ct import (
    PAIRED_CT_COHORT_SCHEMA,
    PRIVATE_ASSET_SCHEMA,
    PRIVATE_OUTCOME_SCHEMA,
)
from stageworld.data.treatment_regimens import (
    COARSE_TUMOR_100EP_PROTOCOL,
    DESCRIPTOR_COUNT,
    REGIMEN_FULL_EPOCH_PROTOCOLS,
    REGIMEN_PROTOCOLS,
    REGIMEN_SCHEMA,
    TUMOR_REGIMEN_100EP_PROTOCOL,
)
from stageworld.data.treatment_summary import (
    TREATMENT_PROTOCOL,
    TREATMENT_SCHEMA,
    attach_treatments,
    require_treatment_cutoff,
)
from stageworld.data.tumor_roi import COARSE_TUMOR_ROI_VERSION, TUMOR_ROI_VERSION
from stageworld.errors import ArtifactError, ConfigurationError, DataContractError
from stageworld.evaluation import (
    CalibrationBinSpec,
    CensoringDistribution,
    EvaluationCohort,
    MetricResult,
    SplitRole,
    calibration_curve,
    cumulative_dynamic_auc,
    integrated_brier_score,
    ipcw_brier_score,
    ipcw_concordance_index,
    make_patient_bootstrap_plan,
)
from stageworld.model import StageWorldModel
from stageworld.real_workflow import (
    DISABLED_PATHOLOGY_ARTIFACT_ID,
    NO_OUTCOME_WORLD_PRETRAIN_CONTRACT,
    REAL_CT_FEATURE_SCHEMA,
    REAL_TIMELINE_CONTRACT_VERSION,
    RealFeatureBundle,
    _feature_root,
    _private_directory,
    _validate_lineage,
    load_real_world_pretrain_batches,
    real_data_root,
)
from stageworld.survival import piecewise_exponential_nll, risk_probability
from stageworld.synthetic_workflow import _config_lineage, build_model
from stageworld.training import (
    CheckpointMetadata,
    ExperimentRegistry,
    LocalEventLogger,
    LossWeights,
    StageWorldTrainer,
    TrainingPhase,
    WorldModelBatch,
    checkpoint_payload_mismatches,
    checkpoint_snapshot_path,
    new_checkpoint_metadata,
)

OS_PROTOCOL = "gastric-roi-os-development-v1"
TREATMENT_OS_PROTOCOLS = (TREATMENT_PROTOCOL, *REGIMEN_PROTOCOLS)
SUPPORTED_OS_PROTOCOLS = (OS_PROTOCOL, *TREATMENT_OS_PROTOCOLS)
OS_OUTCOME_CONTRACT = "os-v1-E-BS-BU-BV-landmark-remaining-years-v1"


def _authorize(config: StageWorldConfig) -> None:
    config.validate(command="real-os-development", supervised=True)
    if (
        config.training.development_protocol not in SUPPORTED_OS_PROTOCOLS
        or config.encoders.ct_preprocess_version
        != (
            COARSE_TUMOR_ROI_VERSION
            if config.training.development_protocol == COARSE_TUMOR_100EP_PROTOCOL
            else TUMOR_ROI_VERSION
            if config.training.development_protocol == TUMOR_REGIMEN_100EP_PROTOCOL
            else GASTRIC_ROI_VERSION
        )
        or tuple(config.clinical.development_stages) != ("s0", "s1")
        or config.survival.time_unit != "year"
        or not config.permissions.allow_long_training
        or config.training.checkpoint_selection != "stage_mean_validation_nll_v1"
    ):
        raise ConfigurationError(
            code="REAL_OS_PROTOCOL_MISMATCH", message="Select the locked ROI OS protocol."
        )
    if config.training.development_protocol in TREATMENT_OS_PROTOCOLS:
        require_treatment_cutoff(config)


def _load_development_bundle(config: StageWorldConfig) -> RealFeatureBundle:
    bundle = load_real_world_pretrain_batches(config)
    if config.training.development_protocol in TREATMENT_OS_PROTOCOLS:
        return attach_treatments(config, bundle)
    return bundle


def _timeline_contract(bundle: RealFeatureBundle) -> str:
    if bundle.treatment_snapshot is not None:
        return f"{REAL_TIMELINE_CONTRACT_VERSION}:{bundle.treatment_snapshot['artifact_id']}"
    return REAL_TIMELINE_CONTRACT_VERSION


def attach_os_labels(
    batches: Sequence[WorldModelBatch],
    outcomes: Sequence[Mapping[str, Any]],
    *,
    label_version: str,
    stages: tuple[int, ...] = (0, 1),
) -> tuple[WorldModelBatch, ...]:
    """Only approved development patients enter the independent outcome builder."""
    if stages not in ((0, 1), (1,)):
        raise DataContractError(code="OS_STAGES_INVALID", message="Use S0/S1 or S1 only.")
    allowed = {patient for batch in batches for patient in batch.patient_ids}
    records = []
    for row in outcomes:
        if row.get("patient_id") not in allowed:
            continue
        if (
            row.get("source_status") not in (0, 1, "0", "1")
            or row.get("label_version") != label_version
        ):
            raise DataContractError(
                code="OS_LABEL_CONTRACT_MISMATCH", message="Unexpected OS label coding."
            )
        records.append(
            Outcome(
                **{
                    **row,
                    "event_type": EventType(row["event_type"]),
                    "adjudication_status": AdjudicationStatus(row["adjudication_status"]),
                }
            )
        )
    definition = OutcomeDefinition(
        endpoint_name="os",
        event_type=EventType.DEATH,
        event_code=1,
        origin_definition="clinical_excel_column_E",
        label_version=label_version,
        mode=DataMode.REAL_IMAGES,
        status_mapping_confirmed=True,
        origin_confirmed=True,
        timeline_confirmed=True,
    )
    builder = OutcomeBuilder(records, definition)
    result = []
    for batch in batches:
        durations = torch.zeros(batch.batch_size, 3)
        events = torch.zeros(batch.batch_size, 3, dtype=torch.long)
        valid = torch.zeros(batch.batch_size, 3, dtype=torch.bool)
        for row_index, patient in enumerate(batch.patient_ids):
            for stage, times in enumerate((batch.s0_time, batch.s1_time)):
                if stage not in stages:
                    continue
                label = builder.build_label(patient, float(times[row_index]))
                durations[row_index, stage] = label.remaining_time_days / 365.25
                events[row_index, stage] = int(label.event)
                valid[row_index, stage] = True
        result.append(
            replace(
                batch, survival_durations=durations, survival_events=events, survival_valid=valid
            )
        )
    return tuple(result)


def _labelled_splits(
    config: StageWorldConfig, bundle: RealFeatureBundle, *, stages: tuple[int, ...] = (0, 1)
) -> dict[str, tuple[WorldModelBatch, ...]]:
    labels = read_json(real_data_root(config) / "outcomes.json")
    if labels.get("schema_version") != PRIVATE_OUTCOME_SCHEMA:
        raise ArtifactError(code="OS_SCHEMA_MISMATCH", message="Unexpected private outcome schema.")
    if _validate_lineage((labels,)) != (bundle.data_lineage_id, bundle.cohort_artifact_id):
        raise ArtifactError(
            code="OS_LINEAGE_MISMATCH", message="Outcomes belong to a different cohort."
        )
    return {
        split: attach_os_labels(
            batches,
            labels["outcomes"],
            label_version=labels["outcome_label_version"],
            stages=stages,
        )
        for split, batches in bundle.batches_by_split.items()
    }


def _forward(model: StageWorldModel, batch: WorldModelBatch) -> Any:
    # Explicit input allowlist: neither labels nor detached future targets are model inputs.
    return model(
        ct0=batch.ct0,
        clinical0=batch.clinical0,
        s0_time=batch.s0_time,
        treatment_actions=batch.treatment_actions,
        ct1_acquisition_time=batch.ct1_acquisition_time,
        ct1=batch.ct1,
        s1_time=batch.s1_time,
        surgery_actions=batch.surgery_actions,
        pathology_acquisition_time=batch.pathology_acquisition_time,
        pathology=batch.pathology,
        s2_time=batch.s2_time,
        horizons=batch.horizons,
        ct1_availability_time=batch.ct1_availability_time,
        ct1_unavailable_event_mask=batch.ct1_unavailable_event_mask,
        pathology_availability_time=batch.pathology_availability_time,
        pathology_unavailable_event_mask=batch.pathology_unavailable_event_mask,
        deterministic=True,
    )


@torch.inference_mode()
def predict_rates(model: StageWorldModel, batches: Sequence[WorldModelBatch]) -> Tensor:
    model.eval()
    device = next(model.parameters()).device
    rates = []
    for batch in batches:
        output = _forward(model, batch.to(device))
        rates.append(torch.stack((output.survival_s0.rates, output.survival_s1.rates), dim=1).cpu())
    result = torch.cat(rates).float()
    if result.ndim == 4 and result.shape[-1] == 1:
        result = result.squeeze(-1)
    if result.ndim != 3 or not torch.isfinite(result).all() or (result <= 0).any():
        raise DataContractError(
            code="INVALID_OS_RATES", message="Nonfinite or invalid survival output."
        )
    return result


def _labels(batches: Sequence[WorldModelBatch]) -> tuple[Tensor, Tensor]:
    return (
        torch.cat([b.survival_durations[:, :2] for b in batches]),
        torch.cat([b.survival_events[:, :2] for b in batches]),
    )


@torch.inference_mode()
def future_ct_diagnostics(
    model: StageWorldModel,
    train: Sequence[WorldModelBatch],
    validation: Sequence[WorldModelBatch],
) -> dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device
    predicted = torch.cat(
        [_forward(model, batch.to(device)).future_ct.mean.cpu() for batch in validation]
    )
    target = torch.cat([batch.future_ct_target for batch in validation])
    mean = torch.cat([batch.future_ct_target for batch in train]).mean(0, keepdim=True)
    persistence = torch.cat(
        [
            (batch.ct0.values * batch.ct0.valid[..., None]).sum(1, keepdim=True)
            / batch.ct0.valid.sum(1, keepdim=True).clamp_min(1)[..., None]
            for batch in validation
        ]
    )
    return {
        "world_model_mse": float((predicted - target).square().mean()),
        "persistence_mse": float((persistence - target).square().mean()),
        "training_mean_mse": float((mean - target).square().mean()),
        "target_between_patient_variance": float(target.var(0).mean()),
        "prediction_between_patient_variance": float(predicted.var(0).mean()),
        "validation_patients": len(target),
        "outcome_labels_used": False,
        "space": "frozen_stomach_roi_swinunetr_spatial_mean",
    }


def stage_mean_nll(
    rates: Tensor, batches: Sequence[WorldModelBatch], cuts: Sequence[float]
) -> float:
    durations, events = _labels(batches)
    boundaries = tuple(cuts)
    return float(
        torch.stack(
            [
                piecewise_exponential_nll(
                    rates[:, stage], durations[:, stage], events[:, stage], boundaries
                )
                for stage in range(2)
            ]
        ).mean()
    )


@torch.inference_mode()
def treatment_response_probe(
    model: StageWorldModel, batches: Sequence[WorldModelBatch], *, regimen: bool = False
) -> dict[str, Any]:
    """An explicit synthetic input perturbation, not an estimate of treatment benefit."""
    model.eval()
    device = next(model.parameters()).device
    maximum = {"s0_rate": 0.0, "s1_rate": 0.0, "prior_state": 0.0, "future_ct": 0.0}
    for batch in batches:
        is_regimen = REGIMEN_SCHEMA in batch.treatment_actions.provenance
        if TREATMENT_SCHEMA not in batch.treatment_actions.provenance and not is_regimen:
            raise ArtifactError(code="TREATMENT_PROBE_INPUT", message="Expected treatment inputs.")
        if regimen and not is_regimen:
            raise ArtifactError(code="TREATMENT_PROBE_INPUT", message="Expected regimen inputs.")
        original = _forward(model, batch.to(device))
        values = batch.treatment_actions.values.clone()
        value_column = DESCRIPTOR_COUNT if is_regimen else 5
        token = 5 if regimen else 1
        values[:, token, value_column] = 1 - values[:, token, value_column]
        values[:, token, value_column + 1] = 1
        changed = replace(
            batch,
            treatment_actions=replace(
                batch.treatment_actions,
                values=values,
                provenance=(
                    *batch.treatment_actions.provenance,
                    "hypothetical_toggle_not_observed",
                ),
            ),
        )
        perturbed = _forward(model, changed.to(device))
        for name, left, right in (
            ("s0_rate", original.survival_s0.rates, perturbed.survival_s0.rates),
            ("s1_rate", original.survival_s1.rates, perturbed.survival_s1.rates),
            ("prior_state", original.prior_ct1.memory, perturbed.prior_ct1.memory),
            ("future_ct", original.future_ct.mean, perturbed.future_ct.mean),
        ):
            maximum[name] = max(maximum[name], float((left - right).abs().max()))
    if maximum["s0_rate"] != 0:
        raise ArtifactError(code="TREATMENT_LEAKS_INTO_S0", message="Future treatment changed S0.")
    return {
        "probe": "hypothetical_sox_mention_toggle"
        if regimen
        else "hypothetical_immunotherapy_flag_toggle",
        "maximum_absolute_differences": maximum,
        "s0_invariant": True,
        "causal_effect_estimate": False,
        "clinical_benefit_estimate": False,
    }


def evaluate_rates(
    rates: Tensor,
    train_batches: Sequence[WorldModelBatch],
    validation_batches: Sequence[WorldModelBatch],
    cuts: Sequence[float],
    *,
    method: str,
    prediction_id: str,
    bootstrap_replicates: int = 100,
    stages: tuple[int, ...] = (0, 1),
) -> list[dict[str, Any]]:
    if stages not in ((0, 1), (1,)) or rates.ndim != 3 or rates.shape[1] != len(stages):
        raise DataContractError(code="OS_EVALUATION_STAGES", message="Bind rates to their stages.")
    boundaries = tuple(cuts)
    train_times, train_events = _labels(train_batches)
    val_times, val_events = _labels(validation_batches)
    train_ids = tuple(p for b in train_batches for p in b.patient_ids)
    val_ids = tuple(p for b in validation_batches for p in b.patient_ids)
    if set(train_ids) & set(val_ids):
        raise DataContractError(
            code="PATIENT_SPLIT_OVERLAP", message="Train and validation overlap."
        )
    grid = torch.linspace(0.25, 3.0, 12)
    results = []
    for rate_index, stage in enumerate(stages):
        reference = EvaluationCohort(
            f"{prediction_id}-train-s{stage}",
            SplitRole.TRAIN,
            "os",
            train_ids,
            tuple(train_times[:, stage].tolist()),
            tuple(train_events[:, stage].tolist()),
        )
        cohort = EvaluationCohort(
            f"{prediction_id}-validation-s{stage}",
            SplitRole.DEVELOPMENT,
            "os",
            val_ids,
            tuple(val_times[:, stage].tolist()),
            tuple(val_events[:, stage].tolist()),
        )
        censoring = CensoringDistribution.fit(reference)
        probabilities = risk_probability(rates[:, rate_index], grid, boundaries).tolist()
        for horizon in (1.0, 3.0):
            risk = risk_probability(rates[:, rate_index], torch.tensor([horizon]), boundaries)[
                :, 0
            ].tolist()
            for metric in (
                cumulative_dynamic_auc(cohort, risk, horizon, censoring),
                ipcw_concordance_index(cohort, risk, horizon, censoring),
                ipcw_brier_score(cohort, risk, horizon, censoring),
            ):
                results.append(
                    {
                        "stage": f"s{stage}",
                        "method": method,
                        "split": "validation",
                        "prediction_artifact_id": prediction_id,
                        **metric.as_dict(),
                    }
                )
        metric = integrated_brier_score(cohort, probabilities, grid.tolist(), censoring)
        results.append(
            {
                "stage": f"s{stage}",
                "method": method,
                "split": "validation",
                "prediction_artifact_id": prediction_id,
                "integration_window_years": [0.25, 3.0],
                **metric.as_dict(),
            }
        )
        stage_results = [row for row in results if row["stage"] == f"s{stage}"]
        _add_bootstrap_intervals(
            stage_results,
            cohort,
            np.asarray(probabilities),
            grid.tolist(),
            censoring,
            n_resamples=bootstrap_replicates,
        )
        for horizon in (1.0, 3.0):
            risks = risk_probability(rates[:, rate_index], torch.tensor([horizon]), boundaries)[
                :, 0
            ].tolist()
            bins = CalibrationBinSpec.fit(
                cohort, risks, bins=3, source_prediction_artifact_id=prediction_id
            )
            calibration = calibration_curve(cohort, risks, horizon, censoring, bins).as_dict()
            for row in stage_results:
                if row["metric"] == "ipcw_brier_score" and row["horizon"] == horizon:
                    row["calibration"] = calibration
    return results


def _add_bootstrap_intervals(
    results: list[dict[str, Any]],
    cohort: EvaluationCohort,
    probabilities: np.ndarray,
    grid: Sequence[float],
    censoring: CensoringDistribution,
    *,
    n_resamples: int,
) -> None:
    """Use the same seeded patient draws at both stages and for both methods."""
    if n_resamples <= 0:
        return
    plan = make_patient_bootstrap_plan(cohort.patient_ids, n_resamples=n_resamples, seed=17)
    times, events = cohort.arrays()
    estimates: dict[tuple[str, float | None], list[float]] = {
        (row["metric"], row["horizon"]): [] for row in results
    }
    functions: dict[str, Callable[..., MetricResult]] = {
        "cumulative_dynamic_auc": cumulative_dynamic_auc,
        "ipcw_concordance_index": ipcw_concordance_index,
        "ipcw_brier_score": ipcw_brier_score,
    }
    for replicate in range(n_resamples):
        indices = plan.expand_rows(cohort.patient_ids, replicate)
        sampled = EvaluationCohort(
            "patient-bootstrap",
            SplitRole.DEVELOPMENT,
            "os",
            tuple(f"draw-{i}" for i in range(len(indices))),
            tuple(times[indices].tolist()),
            tuple(events[indices].tolist()),
        )
        for name, horizon in estimates:
            if name == "integrated_brier_score":
                metric = integrated_brier_score(
                    sampled, probabilities[indices].tolist(), grid, censoring
                )
            else:
                assert horizon is not None
                column = list(grid).index(float(horizon))
                metric = functions[name](
                    sampled, probabilities[indices, column].tolist(), float(horizon), censoring
                )
            if metric.estimable and metric.estimate is not None:
                estimates[(name, horizon)].append(metric.estimate)
    for row in results:
        values = estimates[(row["metric"], row["horizon"])]
        row["bootstrap_valid_replicates"] = len(values)
        row["bootstrap_replicates"] = n_resamples
        row["confidence_interval_95"] = (
            np.quantile(values, [0.025, 0.975]).tolist()
            if row["status"] == "ok" and len(values) >= max(20, int(0.8 * n_resamples))
            else None
        )
        row["interval_scope"] = (
            "patient_sampling_conditional_on_selected_model_not_selection_adjusted"
        )


def _baseline_inputs(batches: Sequence[WorldModelBatch], stage: int) -> Tensor:
    rows = []
    for batch in batches:
        x0 = batch.ct0.values.mean(dim=1)
        if stage == 0:
            rows.append(torch.cat((x0, batch.s0_time[:, None] / 365.25), dim=1))
        else:
            features = [
                x0,
                batch.ct1.values.mean(dim=1),
                batch.s0_time[:, None] / 365.25,
                batch.s1_time[:, None] / 365.25,
            ]
            if any(
                s in batch.treatment_actions.provenance for s in (TREATMENT_SCHEMA, REGIMEN_SCHEMA)
            ):
                features.append(batch.treatment_actions.values.flatten(1))
            rows.append(torch.cat(features, dim=1))
    return torch.cat(rows).float()


def fit_ridge_baseline(
    train: Sequence[WorldModelBatch],
    validation: Sequence[WorldModelBatch],
    cuts: Sequence[float],
    *,
    input_builder: Callable[[Sequence[WorldModelBatch], int], Tensor] = _baseline_inputs,
) -> tuple[Tensor, list[dict[str, Any]]]:
    """Fixed L2=1 linear softplus-rate control with the same stage-visible CT features."""
    durations, events = _labels(train)
    boundaries = tuple(cuts)
    output, snapshots = [], []
    for stage in range(2):
        train_x, val_x = input_builder(train, stage), input_builder(validation, stage)
        mean, std = train_x.mean(0), train_x.std(0).clamp_min(1e-5)
        x = (train_x - mean) / std
        head = nn.Linear(x.shape[1], len(cuts) - 1)
        nn.init.zeros_(head.weight)
        nn.init.constant_(head.bias, -3.0)
        optimizer = torch.optim.LBFGS(
            head.parameters(), max_iter=100, line_search_fn="strong_wolfe"
        )

        def closure(
            active_optimizer: Any = optimizer,
            active_head: nn.Linear = head,
            inputs: Tensor = x,
            stage_index: int = stage,
        ) -> Tensor:
            active_optimizer.zero_grad()
            rate = torch.nn.functional.softplus(active_head(inputs)) + 1e-6
            loss = piecewise_exponential_nll(
                rate,
                durations[:, stage_index],
                events[:, stage_index],
                boundaries,
            )
            loss = loss + active_head.weight.square().sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            output.append(torch.nn.functional.softplus(head((val_x - mean) / std)) + 1e-6)
        snapshots.append(
            {
                "stage": stage,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "weight": head.weight.detach().tolist(),
                "bias": head.bias.detach().tolist(),
                "penalty": "sum_squared_weights",
                "lambda": 1.0,
            }
        )
    return torch.stack(output, dim=1), snapshots


def _metadata(
    config: StageWorldConfig,
    bundle: RealFeatureBundle,
    model: StageWorldModel,
    phase: TrainingPhase,
    parent: Mapping[str, Any] | None = None,
) -> CheckpointMetadata:
    kwargs: dict[str, Any] = {}
    if parent is not None:
        source = parent["metadata"]
        for name in (
            "checkpoint_id",
            "phase",
            "config_lineage_id",
            "data_lineage_id",
            "cohort_artifact_id",
            "split_version",
            "ct_feature_artifact_id",
            "pathology_feature_artifact_id",
            "timeline_contract_version",
            "outcome_contract_version",
        ):
            kwargs[f"parent_{name}"] = source[name]
        kwargs["parent_weight_version"] = parent["weight_version"]
    return new_checkpoint_metadata(
        model=model,
        mode=config.mode.value,
        config_lineage_id=_config_lineage(config),
        data_lineage_id=bundle.data_lineage_id,
        cohort_artifact_id=bundle.cohort_artifact_id,
        split_version=bundle.split_version,
        ct_feature_artifact_id=bundle.feature_artifact_id,
        pathology_feature_artifact_id=DISABLED_PATHOLOGY_ARTIFACT_ID,
        timeline_contract_version=_timeline_contract(bundle),
        outcome_contract_version=(
            OS_OUTCOME_CONTRACT if parent else NO_OUTCOME_WORLD_PRETRAIN_CONTRACT
        ),
        training_seed=config.training.seed,
        source_schema_version=PRIVATE_ASSET_SCHEMA,
        cohort_schema_version=PAIRED_CT_COHORT_SCHEMA,
        feature_schema_version=REAL_CT_FEATURE_SCHEMA,
        phase=phase,
        selection_rule=(
            config.training.checkpoint_selection
            if parent
            else f"fixed_{config.training.world_pretrain_steps}_steps_v1"
        ),
        **kwargs,
    )


def _new_trainer(
    config: StageWorldConfig, model: StageWorldModel, event_path: Path
) -> StageWorldTrainer:
    event_path.touch(mode=0o600, exist_ok=False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay
    )
    return StageWorldTrainer(
        model,
        optimizer,
        device="cuda" if torch.cuda.is_available() else "cpu",
        mixed_precision=config.training.mixed_precision,
        grad_clip_norm=config.training.grad_clip_norm,
        survival_time_unit="year",
        event_logger=LocalEventLogger(event_path),
    )


def _validate_training_schedule(config: StageWorldConfig, steps_per_epoch: int) -> None:
    settings = config.training
    if (
        steps_per_epoch < 1
        or settings.world_pretrain_steps < 1
        or settings.joint_survival_steps < 1
        or settings.development_max_epochs < 1
        or (settings.development_max_minutes is not None and settings.development_max_minutes < 1)
        or (settings.development_patience is not None and settings.development_patience < 1)
    ):
        raise ConfigurationError(code="OS_TRAINING_BUDGET_INVALID", message="Invalid OS budget.")
    if settings.development_protocol in REGIMEN_FULL_EPOCH_PROTOCOLS and (
        settings.world_pretrain_steps % steps_per_epoch != 0
        or settings.world_pretrain_steps < 100 * steps_per_epoch
        or settings.development_max_epochs < 100
        or settings.joint_survival_steps < settings.development_max_epochs * steps_per_epoch
        or settings.development_patience is not None
    ):
        raise ConfigurationError(
            code="OS_100_EPOCH_BUDGET_REQUIRED",
            message="Require at least 100 full epochs per phase and disabled early stopping.",
        )


def run_real_os_development(config: StageWorldConfig) -> dict[str, Any]:
    """Complete real pretraining, joint survival, selected-model prediction and evaluation."""
    _authorize(config)
    manifest = read_json(_feature_root(config) / "ct_manifest.json")
    if manifest.get("limit_per_split") is not None:
        raise ConfigurationError(
            code="PILOT_FEATURES_NOT_OS_COHORT", message="Complete development extraction first."
        )
    bundle = _load_development_bundle(config)
    steps_per_epoch = len(bundle.batches_by_split["train"])
    _validate_training_schedule(config, steps_per_epoch)
    if bundle.training_patient_count < 100 or bundle.validation_patient_count < 30:
        raise DataContractError(
            code="OS_COHORT_TOO_SMALL", message="Insufficient complete development ROI pairs."
        )
    random.seed(config.training.seed)
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    model = build_model(config)
    run_id = new_artifact_id("roi-os-run")
    run_root = config.output_root / "runs" / run_id
    _private_directory(config.output_root)
    _private_directory(run_root.parent)
    _private_directory(run_root)
    _private_directory(run_root / "checkpoint_versions")
    if bundle.treatment_snapshot is not None:
        atomic_write_private_json(run_root / "treatment_snapshot.json", bundle.treatment_snapshot)
    registry = ExperimentRegistry(config.output_root / "experiment_registry.csv")
    started = time.time()
    deadline = (
        float("inf")
        if config.training.development_max_minutes is None
        else time.monotonic() + config.training.development_max_minutes * 60
    )
    registry_row = {
        "run_id": run_id,
        "mode": config.mode.value,
        "phase": "joint_survival",
        "seed": config.training.seed,
        "config_lineage_id": _config_lineage(config),
        "data_lineage_id": bundle.data_lineage_id,
        "started_at_unix": started,
    }
    registry.update({**registry_row, "status": "running"})
    schedule = {
        "steps_per_epoch": steps_per_epoch,
        "world_target_steps": config.training.world_pretrain_steps,
        "world_target_epochs": config.training.world_pretrain_steps / steps_per_epoch,
        "joint_target_epochs": config.training.development_max_epochs,
        "joint_step_limit": config.training.joint_survival_steps,
        "early_stopping_enabled": config.training.development_patience is not None,
        "time_guard_minutes": config.training.development_max_minutes,
    }
    atomic_write_json(run_root / "training_schedule.json", schedule)
    try:
        with offline_network():
            world_metadata = _metadata(config, bundle, model, TrainingPhase.WORLD_PRETRAIN)
            trainer = _new_trainer(config, model, run_root / "world_events.jsonl")
            world_weights = LossWeights(survival=0, future_ct=1, future_pathology=0, kl=0.01)
            world_history = []
            train_inputs = bundle.batches_by_split["train"]
            order = list(range(len(train_inputs)))
            atomic_write_json(
                config.output_root / "training_progress.json",
                {"run_id": run_id, "phase": "world_pretrain", "epoch": 0, **schedule},
            )
            for step in range(config.training.world_pretrain_steps):
                if step % len(order) == 0:
                    random.shuffle(order)
                if time.monotonic() >= deadline:
                    raise ArtifactError(
                        code="OS_TIME_BUDGET_REACHED", message="Development time guard reached."
                    )
                record = trainer.optimizer_step(
                    [train_inputs[order[step % len(order)]]],
                    phase=TrainingPhase.WORLD_PRETRAIN,
                    weights=world_weights,
                    kl_beta=min(1.0, (step + 1) / 40),
                )
                world_history.append(record)
                if (step + 1) % len(order) == 0:
                    atomic_write_json(
                        config.output_root / "training_progress.json",
                        {
                            "run_id": run_id,
                            "phase": "world_pretrain",
                            "optimizer_steps": step + 1,
                            "epoch": (step + 1) // steps_per_epoch,
                            **schedule,
                            "outcome_labels_used": False,
                            "elapsed_seconds": time.time() - started,
                        },
                    )
            parent_path = run_root / "world_pretrain.pt"
            trainer.save_checkpoint(
                parent_path,
                world_metadata,
                sampler_state={"completed_epochs": len(world_history) // steps_per_epoch},
            )
            parent = torch.load(parent_path, map_location="cpu", weights_only=True)
            pretraining_future_metrics = future_ct_diagnostics(
                model,
                train_inputs,
                bundle.batches_by_split["validation"],
            )
            model.load_state_dict(parent["model_state"], strict=True)
            labelled = _labelled_splits(config, bundle)
            train, validation = labelled["train"], labelled["validation"]
            train_times, train_events = _labels(train)
            val_times, val_events = _labels(validation)
            if int(train_events[:, 0].sum()) < 5 or int(val_events[:, 0].sum()) < 2:
                raise DataContractError(
                    code="OS_EVENT_SUPPORT_TOO_SMALL",
                    message="Insufficient development death events.",
                )
            metadata = _metadata(config, bundle, model, TrainingPhase.JOINT_SURVIVAL, parent)
            trainer = _new_trainer(config, model, run_root / "joint_events.jsonl")
            weights = LossWeights(survival=1, future_ct=0.1, future_pathology=0, kl=0.001)
            best, stale, best_epoch = math.inf, 0, 0
            best_path = run_root / "selected.pt"
            history = []
            stop_reason = "epoch_limit"
            for epoch in range(1, config.training.development_max_epochs + 1):
                order = list(range(len(train)))
                random.shuffle(order)
                records = []
                for index in order:
                    if time.monotonic() >= deadline:
                        raise ArtifactError(
                            code="OS_TIME_BUDGET_REACHED", message="Development time guard reached."
                        )
                    records.append(
                        trainer.optimizer_step(
                            [train[index]],
                            phase=TrainingPhase.JOINT_SURVIVAL,
                            weights=weights,
                            kl_beta=min(1.0, epoch / 5),
                        )
                    )
                rates = predict_rates(model, validation)
                score = stage_mean_nll(rates, validation, config.survival.finite_cutpoints)
                if not math.isfinite(score):
                    raise DataContractError(
                        code="OS_VALIDATION_NONFINITE", message="Nonfinite OS validation NLL."
                    )
                improved = score < best - 1e-5
                if improved:
                    best, stale, best_epoch = score, 0, epoch
                    trainer.save_checkpoint(
                        best_path,
                        metadata,
                        sampler_state={"epoch": epoch},
                        transfer_parent_model_state=parent["model_state"],
                    )
                else:
                    stale += 1
                history.append(
                    {
                        "epoch": epoch,
                        "validation_stage_mean_nll": score,
                        "selected": improved,
                        "optimizer_steps": trainer.state.optimizer_step,
                    }
                )
                atomic_write_json(
                    config.output_root / "training_progress.json",
                    {
                        "run_id": run_id,
                        "phase": "joint_survival",
                        **history[-1],
                        **schedule,
                        "best_epoch": best_epoch,
                        "elapsed_seconds": time.time() - started,
                    },
                )
                if (
                    config.training.development_patience is not None
                    and stale >= config.training.development_patience
                ):
                    stop_reason = "validation_early_stopping"
                    break
                if trainer.state.optimizer_step >= config.training.joint_survival_steps:
                    stop_reason = "step_limit"
                    break
            completed_steps = trainer.state.optimizer_step
            final_path = run_root / "final.pt"
            trainer.save_checkpoint(
                final_path,
                metadata,
                sampler_state={"epoch": len(history)},
                transfer_parent_model_state=parent["model_state"],
            )
            final_payload = torch.load(final_path, map_location="cpu", weights_only=True)
            final_snapshot = checkpoint_snapshot_path(final_path, final_payload["weight_version"])
            trainer.load_checkpoint(best_path, expected=metadata, restore_rng=False)
            selected = torch.load(best_path, map_location="cpu", weights_only=True)
            snapshot = checkpoint_snapshot_path(best_path, selected["weight_version"])
            if checkpoint_payload_mismatches(
                selected, torch.load(snapshot, map_location="cpu", weights_only=True)
            ):
                raise ArtifactError(
                    code="OS_SNAPSHOT_MISMATCH", message="Selected snapshot does not match."
                )
            rates = predict_rates(model, validation)
            future_metrics = future_ct_diagnostics(model, train, validation)
            prediction_id = new_artifact_id("real-os-predictions")
            prediction_path = run_root / "validation_predictions.json"
            patient_ids = [patient for batch in validation for patient in batch.patient_ids]
            atomic_write_private_json(
                prediction_path,
                {
                    "schema_version": "stageworld-real-os-rates-v1",
                    "prediction_artifact_id": prediction_id,
                    "checkpoint_id": metadata.checkpoint_id,
                    "weight_version": selected["weight_version"],
                    "ct_feature_artifact_id": bundle.feature_artifact_id,
                    "treatment_artifact_id": (
                        None
                        if bundle.treatment_snapshot is None
                        else bundle.treatment_snapshot["artifact_id"]
                    ),
                    "split_version": bundle.split_version,
                    "split": "validation",
                    "patient_ids": patient_ids,
                    "rates": rates.tolist(),
                    "stages": ["s0", "s1"],
                    "time_unit": "year",
                    "deterministic": True,
                },
            )
            metrics = evaluate_rates(
                rates,
                train,
                validation,
                config.survival.finite_cutpoints,
                method="stageworld",
                prediction_id=prediction_id,
            )
            baseline, baseline_state = fit_ridge_baseline(
                train, validation, config.survival.finite_cutpoints
            )
            baseline_id = new_artifact_id("real-os-ridge")
            atomic_write_private_json(
                run_root / "ridge_baseline.json",
                {
                    "artifact_id": baseline_id,
                    "patient_ids": patient_ids,
                    "rates": baseline.tolist(),
                    "training_feature_artifact_id": bundle.feature_artifact_id,
                    "models": baseline_state,
                },
            )
            metrics.extend(
                evaluate_rates(
                    baseline,
                    train,
                    validation,
                    config.survival.finite_cutpoints,
                    method=(
                        "ridge_same_ct_features"
                        if bundle.treatment_snapshot is None
                        else "ridge_same_ct_and_treatment_features"
                    ),
                    prediction_id=baseline_id,
                )
            )
            original_splits = read_json(real_data_root(config) / "split_assignments.json")
            original_counts = Counter(row["split"] for row in original_splits["assignments"])
            summary = {
                "schema_version": "stageworld-real-os-development-v1",
                "status": "completed",
                "protocol": config.training.development_protocol,
                "run_id": run_id,
                "development_only": True,
                "clinical_validation": False,
                "expert_segmentation_review_completed": False,
                "ct_preprocess_version": config.encoders.ct_preprocess_version,
                "roi_target": (
                    "tumor_guided_context_with_explicit_stomach_fallback"
                    if config.encoders.ct_preprocess_version == COARSE_TUMOR_ROI_VERSION
                    else "unreviewed_gastric_associated_pan_cancer_candidates"
                    if config.encoders.ct_preprocess_version == TUMOR_ROI_VERSION
                    else "whole_stomach"
                ),
                "explicit_tumor_growth_prediction": False,
                "training_patients": bundle.training_patient_count,
                "validation_patients": bundle.validation_patient_count,
                "training_deaths": int(train_events[:, 0].sum()),
                "validation_deaths": int(val_events[:, 0].sum()),
                "reserved_test_patients": bundle.held_out_test_patient_count,
                "source_cohort_patients": sum(original_counts.values()),
                "excluded_roi_pair_counts": {
                    "train": original_counts["train"] - bundle.training_patient_count,
                    "validation": original_counts["validation"] - bundle.validation_patient_count,
                },
                "test_used": False,
                "pretraining_steps": len(world_history),
                "pretraining_completed_epochs": len(world_history) // steps_per_epoch,
                "pretraining_partial_epoch_steps": len(world_history) % steps_per_epoch,
                "joint_completed_epochs": len(history),
                "training_schedule": schedule,
                "pretraining_initial_loss": world_history[0]["loss"],
                "pretraining_final_loss": world_history[-1]["loss"],
                "future_ct_validation": future_metrics,
                "future_ct_validation_phase": "selected_joint_survival",
                "future_ct_pretraining": pretraining_future_metrics,
                "treatment_artifact_id": (
                    None
                    if bundle.treatment_snapshot is None
                    else bundle.treatment_snapshot["artifact_id"]
                ),
                "treatment_inputs": (
                    None
                    if bundle.treatment_snapshot is None
                    else bundle.treatment_snapshot["aggregate"]
                ),
                "treatment_response_probe": (
                    None
                    if bundle.treatment_snapshot is None
                    else treatment_response_probe(model, validation)
                ),
                "regimen_response_probe": (
                    treatment_response_probe(model, validation, regimen=True)
                    if config.training.development_protocol in REGIMEN_PROTOCOLS
                    else None
                ),
                "joint_optimizer_steps": completed_steps,
                "selected_epoch": best_epoch,
                "selection_rule": config.training.checkpoint_selection,
                "selected_validation_nll": best,
                "final_validation_nll": history[-1]["validation_stage_mean_nll"],
                "final_checkpoint": str(final_snapshot),
                "final_weight_version": final_payload["weight_version"],
                "stop_reason": stop_reason,
                "training_history": history,
                "metrics": metrics,
                "checkpoint": str(snapshot),
                "checkpoint_id": metadata.checkpoint_id,
                "weight_version": selected["weight_version"],
                "parent_checkpoint_id": world_metadata.checkpoint_id,
                "prediction_artifact_id": prediction_id,
                "predictions": str(prediction_path),
                "ct_feature_artifact_id": bundle.feature_artifact_id,
                "split_version": bundle.split_version,
                "elapsed_seconds": time.time() - started,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "confidence_intervals": "100_patient_bootstrap_conditional_not_selection_adjusted",
                "external_validation": False,
                "architecture": asdict(model.config),
            }
            atomic_write_json(run_root / "summary.json", summary)
            atomic_write_json(config.output_root / "os_summary.json", summary)
            from stageworld.real_report import write_real_os_report

            summary["report"] = write_real_os_report(config, summary)
            atomic_write_json(run_root / "summary.json", summary)
            atomic_write_json(config.output_root / "os_summary.json", summary)
            registry.update(
                {**registry_row, "status": "completed", "finished_at_unix": time.time()}
            )
            atomic_write_json(
                config.output_root / "training_progress.json",
                {
                    "run_id": run_id,
                    "phase": "completed",
                    "optimizer_steps": completed_steps,
                    "pretraining_completed_epochs": len(world_history) // steps_per_epoch,
                    "joint_completed_epochs": len(history),
                    "best_epoch": best_epoch,
                    "selected_validation_nll": best,
                    "test_used": False,
                    "summary": str(config.output_root / "os_summary.json"),
                },
            )
            return {
                key: summary[key]
                for key in (
                    "status",
                    "run_id",
                    "training_patients",
                    "validation_patients",
                    "training_deaths",
                    "validation_deaths",
                    "selected_epoch",
                    "selected_validation_nll",
                    "pretraining_steps",
                    "pretraining_completed_epochs",
                    "joint_completed_epochs",
                    "joint_optimizer_steps",
                    "metrics",
                    "elapsed_seconds",
                )
            }
    except BaseException as error:
        code = str(getattr(error, "code", type(error).__name__.upper()))
        registry.update(
            {
                **registry_row,
                "status": "failed",
                "failure_code": code,
                "finished_at_unix": time.time(),
            }
        )
        atomic_write_json(
            run_root / "failure.json",
            {
                "status": "failed",
                "code": code,
                "elapsed_seconds": time.time() - started,
                "run_id": run_id,
            },
        )
        raise


def _load_selected_development_model(
    config: StageWorldConfig,
) -> tuple[dict[str, Any], RealFeatureBundle, StageWorldModel, dict[str, Any]]:
    _authorize(config)
    summary = read_json(config.output_root / "os_summary.json")
    if (
        summary.get("status") != "completed"
        or summary.get("protocol") != config.training.development_protocol
    ):
        raise ArtifactError(
            code="OS_RUN_NOT_COMPLETE", message="Complete the real OS development run first."
        )
    bundle = _load_development_bundle(config)
    path = Path(summary["checkpoint"]).resolve()
    if not path.is_relative_to(config.output_root.resolve()):
        raise ArtifactError(
            code="OS_CHECKPOINT_LOCATION_INVALID", message="Unexpected checkpoint location."
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = CheckpointMetadata(**payload["metadata"])
    expected = {
        "config_lineage_id": _config_lineage(config),
        "data_lineage_id": bundle.data_lineage_id,
        "cohort_artifact_id": bundle.cohort_artifact_id,
        "split_version": bundle.split_version,
        "ct_feature_artifact_id": bundle.feature_artifact_id,
        "outcome_contract_version": OS_OUTCOME_CONTRACT,
        "phase": "joint_survival",
        "checkpoint_id": summary["checkpoint_id"],
        "timeline_contract_version": _timeline_contract(bundle),
    }
    if any(getattr(metadata, key) != value for key, value in expected.items()):
        raise ArtifactError(
            code="OS_CHECKPOINT_LINEAGE_MISMATCH", message="Selected OS lineage has changed."
        )
    if (
        payload["weight_version"] != summary["weight_version"]
        or summary["ct_feature_artifact_id"] != bundle.feature_artifact_id
    ):
        raise ArtifactError(
            code="OS_CHECKPOINT_LINEAGE_MISMATCH", message="Selected weight or feature changed."
        )
    if path.parent.name != "checkpoint_versions" or path.stem != payload["weight_version"]:
        raise ArtifactError(
            code="OS_IMMUTABLE_SNAPSHOT_REQUIRED", message="Use the named immutable OS snapshot."
        )
    if bundle.treatment_snapshot is not None:
        if (
            summary.get("treatment_artifact_id") != bundle.treatment_snapshot["artifact_id"]
            or read_json(path.parent.parent / "treatment_snapshot.json")
            != bundle.treatment_snapshot
        ):
            raise ArtifactError(
                code="OS_TREATMENT_SNAPSHOT_MISMATCH", message="Treatment inputs changed."
            )
    parent_path = path.parent / f"{metadata.parent_weight_version}.pt"
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    if (
        parent.get("weight_version") != metadata.parent_weight_version
        or parent.get("metadata", {}).get("checkpoint_id") != metadata.parent_checkpoint_id
        or checkpoint_payload_mismatches(
            parent.get("model_state"), payload.get("transfer_parent_model_state")
        )
    ):
        raise ArtifactError(
            code="OS_TRANSFER_PARENT_MISMATCH", message="Transferred parent no longer matches."
        )
    model = build_model(config)
    if payload["model_config"] != asdict(model.config):
        raise ArtifactError(
            code="OS_MODEL_CONFIG_MISMATCH", message="Selected architecture changed."
        )
    model.load_state_dict(payload["model_state"], strict=True)
    return summary, bundle, model, payload


def predict_real_os_development(config: StageWorldConfig) -> dict[str, Any]:
    """Recompute validation predictions from the selected immutable checkpoint, without outcomes."""
    with offline_network():
        summary, bundle, model, _ = _load_selected_development_model(config)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        batches = bundle.batches_by_split["validation"]
        rates = predict_rates(model, batches)
        prediction_id = new_artifact_id("real-os-predictions")
        path = config.output_root / "runs" / summary["run_id"] / f"{prediction_id}.json"
        atomic_write_private_json(
            path,
            {
                "schema_version": "stageworld-real-os-rates-v1",
                "prediction_artifact_id": prediction_id,
                "checkpoint_id": summary["checkpoint_id"],
                "weight_version": summary["weight_version"],
                "ct_feature_artifact_id": bundle.feature_artifact_id,
                "treatment_artifact_id": summary.get("treatment_artifact_id"),
                "split_version": bundle.split_version,
                "split": "validation",
                "patient_ids": [p for b in batches for p in b.patient_ids],
                "rates": rates.tolist(),
                "stages": ["s0", "s1"],
                "time_unit": "year",
                "deterministic": True,
            },
        )
        return {
            "status": "ok",
            "predictions": str(path),
            "prediction_artifact_id": prediction_id,
            "patient_count": bundle.validation_patient_count,
            "outcome_labels_used": False,
            "test_used": False,
            "development_only": True,
        }


def evaluate_real_os_development(
    config: StageWorldConfig,
    *,
    predictions: Path | None = None,
) -> dict[str, Any]:
    """Evaluate only the fixed validation population and exact selected checkpoint."""
    with offline_network():
        summary, bundle, _, _ = _load_selected_development_model(config)
        path = (predictions or Path(summary["predictions"])).resolve()
        if not path.is_relative_to(config.output_root.resolve()):
            raise ArtifactError(
                code="OS_PREDICTION_LOCATION_INVALID", message="Unexpected prediction location."
            )
        prediction = read_json(path)
        for field in (
            "checkpoint_id",
            "weight_version",
            "ct_feature_artifact_id",
            "split_version",
            "treatment_artifact_id",
        ):
            if prediction.get(field) != summary.get(field):
                raise ArtifactError(
                    code="OS_PREDICTION_LINEAGE_MISMATCH", message="Prediction lineage differs."
                )
        patient_ids = [p for b in bundle.batches_by_split["validation"] for p in b.patient_ids]
        if (
            prediction.get("patient_ids") != patient_ids
            or prediction.get("split") != "validation"
            or prediction.get("time_unit") != "year"
            or prediction.get("stages") != ["s0", "s1"]
        ):
            raise ArtifactError(
                code="OS_PREDICTION_POPULATION_MISMATCH", message="Prediction population differs."
            )
        labelled = _labelled_splits(config, bundle)
        metrics = evaluate_rates(
            torch.tensor(prediction["rates"]),
            labelled["train"],
            labelled["validation"],
            config.survival.finite_cutpoints,
            method="stageworld",
            prediction_id=prediction["prediction_artifact_id"],
            bootstrap_replicates=config.evaluation.bootstrap_replicates,
        )
        result = {
            "status": "ok",
            "development_only": True,
            "clinical_validation": False,
            "test_used": False,
            "prediction_artifact_id": prediction["prediction_artifact_id"],
            "metrics": metrics,
        }
        output = path.with_name(f"{prediction['prediction_artifact_id']}-evaluation.json")
        atomic_write_json(output, result)
        return {**result, "evaluation": str(output)}


def verify_real_os_development(config: StageWorldConfig) -> dict[str, Any]:
    """Audit a completed run without selecting new weights or modifying patient membership."""
    from stageworld.real_report import write_real_os_report

    with offline_network():
        summary, bundle, model, payload = _load_selected_development_model(config)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        labelled = _labelled_splits(config, bundle)
        validation = labelled["validation"]
        rates = predict_rates(model, validation)
        saved = read_json(summary["predictions"])
        expected_patients = [p for b in validation for p in b.patient_ids]
        if saved.get("patient_ids") != expected_patients:
            raise ArtifactError(
                code="OS_AUDIT_PATIENT_MISMATCH", message="Prediction order changed."
            )
        old_rates = torch.tensor(saved["rates"], dtype=rates.dtype)
        if not torch.allclose(rates, old_rates, rtol=1e-5, atol=1e-7):
            raise ArtifactError(
                code="OS_AUDIT_PREDICTION_MISMATCH", message="Saved OS risks did not replay."
            )
        replay_nll = stage_mean_nll(rates, validation, config.survival.finite_cutpoints)
        if not math.isclose(
            replay_nll, summary["selected_validation_nll"], rel_tol=1e-5, abs_tol=1e-7
        ):
            raise ArtifactError(
                code="OS_AUDIT_SELECTION_MISMATCH", message="Selection NLL did not replay."
            )
        split = read_json(real_data_root(config) / "split_assignments.json")
        test_ids = {r["patient_id"] for r in split["assignments"] if r["split"] == "test"}
        train_ids = {p for b in labelled["train"] for p in b.patient_ids}
        if test_ids & (train_ids | set(expected_patients)) or train_ids & set(expected_patients):
            raise ArtifactError(code="OS_AUDIT_SPLIT_OVERLAP", message="Patient split overlap.")
        run_root = Path(summary["predictions"]).parent
        all_events: list[dict[str, Any]] = []
        for filename in ("world_events.jsonl", "joint_events.jsonl"):
            with (run_root / filename).open() as stream:
                import json

                all_events.extend(json.loads(line) for line in stream if line.strip())

        def finite_tree(value: Any) -> bool:
            if isinstance(value, float):
                return math.isfinite(value)
            if isinstance(value, Mapping):
                return all(finite_tree(item) for item in value.values())
            if isinstance(value, list):
                return all(finite_tree(item) for item in value)
            return True

        if not finite_tree(all_events):
            raise ArtifactError(
                code="OS_AUDIT_NONFINITE_LOG", message="A training scalar is nonfinite."
            )
        duration_audit = {}
        if (
            config.training.development_protocol in REGIMEN_FULL_EPOCH_PROTOCOLS
            and "final_checkpoint" not in summary
        ):
            raise ArtifactError(
                code="OS_AUDIT_DURATION_MISMATCH", message="Final-epoch checkpoint is required."
            )
        if "final_checkpoint" in summary:
            steps_per_epoch = len(labelled["train"])
            _validate_training_schedule(config, steps_per_epoch)
            world_steps = summary["pretraining_steps"]
            joint_steps = summary["joint_optimizer_steps"]
            counts_match = all(
                [r["optimizer_step"] for r in all_events if r["phase"] == phase]
                == list(range(1, steps + 1))
                for phase, steps in (
                    ("world_pretrain", world_steps),
                    ("joint_survival", joint_steps),
                )
            )
            final_path = run_root / "final.pt"
            final = torch.load(final_path, map_location="cpu", weights_only=True)
            snapshot_path = checkpoint_snapshot_path(final_path, final["weight_version"])
            snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=True)
            if (
                not counts_match
                or summary["pretraining_completed_epochs"] != world_steps // steps_per_epoch
                or summary["pretraining_partial_epoch_steps"] != world_steps % steps_per_epoch
                or summary["joint_completed_epochs"] * steps_per_epoch != joint_steps
                or summary["joint_completed_epochs"] != len(summary["training_history"])
                or final["trainer_state"]["optimizer_step"] != joint_steps
                or final["sampler_state"].get("epoch") != summary["joint_completed_epochs"]
                or final["metadata"] != payload["metadata"]
                or final["weight_version"] != summary["final_weight_version"]
                or str(snapshot_path) != summary["final_checkpoint"]
                or checkpoint_payload_mismatches(final, snapshot)
                or checkpoint_payload_mismatches(
                    final["transfer_parent_model_state"], payload["transfer_parent_model_state"]
                )
            ):
                raise ArtifactError(
                    code="OS_AUDIT_DURATION_MISMATCH",
                    message="Epoch counts, event logs or final checkpoint did not match.",
                )
            if config.training.development_protocol in REGIMEN_FULL_EPOCH_PROTOCOLS and (
                world_steps != config.training.world_pretrain_steps
                or summary["joint_completed_epochs"] != config.training.development_max_epochs
            ):
                raise ArtifactError(
                    code="OS_AUDIT_DURATION_MISMATCH", message="Full epoch budget not completed."
                )
            try:
                model.load_state_dict(final["model_state"], strict=True)
                final_nll = stage_mean_nll(
                    predict_rates(model, validation), validation, config.survival.finite_cutpoints
                )
                if not math.isclose(
                    final_nll, summary["final_validation_nll"], rel_tol=1e-5, abs_tol=1e-7
                ):
                    raise ArtifactError(
                        code="OS_AUDIT_FINAL_NLL_MISMATCH", message="Final OS NLL did not replay."
                    )
            finally:
                model.load_state_dict(payload["model_state"], strict=True)
            duration_audit = {
                "training_duration_verified": True,
                "pretraining_completed_epochs": summary["pretraining_completed_epochs"],
                "joint_completed_epochs": summary["joint_completed_epochs"],
                "final_checkpoint_verified": True,
                "final_validation_nll_replayed": final_nll,
            }
        private_roots = [run_root, _feature_root(config)]
        if bundle.treatment_snapshot is not None and config.paths.treatment_manifest:
            private_roots.append(Path(config.paths.treatment_manifest).parent)
        audited_paths = list(private_roots) + [
            p for root in private_roots for p in root.rglob("*") if p.is_file() or p.is_dir()
        ]
        unsafe = sum(bool(p.stat().st_mode & 0o077) for p in audited_paths)
        if unsafe:
            raise ArtifactError(
                code="OS_AUDIT_PERMISSIONS", message="A private artifact has unsafe permissions."
            )
        recomputed = evaluate_real_os_development(config)
        prior_metrics = {
            (r["stage"], r["horizon"], r["metric"]): r
            for r in summary["metrics"]
            if r["method"] == "stageworld"
        }
        for row in recomputed["metrics"]:
            old = prior_metrics[(row["stage"], row["horizon"], row["metric"])]
            if row["status"] != old["status"] or row["estimate"] != old["estimate"]:
                raise ArtifactError(
                    code="OS_AUDIT_METRIC_MISMATCH", message="Saved OS metric did not replay."
                )
        report_path = write_real_os_report(config, summary)
        treatment_probe = (
            None
            if bundle.treatment_snapshot is None
            else treatment_response_probe(model, validation)
        )
        audit = {
            "status": "ok",
            **duration_audit,
            "run_id": summary["run_id"],
            "checkpoint_id": summary["checkpoint_id"],
            "weight_version": payload["weight_version"],
            "immutable_parent_exact_match": True,
            "prediction_replayed": True,
            "maximum_rate_absolute_difference": float((rates - old_rates).abs().max()),
            "selected_validation_nll_replayed": replay_nll,
            "metrics_replayed": True,
            "training_log_records": len(all_events),
            "all_training_scalars_finite": True,
            "audited_private_paths": len(audited_paths),
            "unsafe_permissions": unsafe,
            "test_patient_overlap": 0,
            "training_validation_overlap": 0,
            "report": report_path,
            "clinical_validation": False,
            "treatment_response_probe": treatment_probe,
        }
        atomic_write_json(run_root / "verification.json", audit)
        atomic_write_json(config.output_root / "verification.json", audit)
        atomic_write_json(
            config.output_root / "training_progress.json",
            {
                "run_id": summary["run_id"],
                "phase": "completed",
                "optimizer_steps": summary["joint_optimizer_steps"],
                **duration_audit,
                "best_epoch": summary["selected_epoch"],
                "selected_validation_nll": replay_nll,
                "test_used": False,
                "summary": str(config.output_root / "os_summary.json"),
            },
        )
        return audit
