"""Command-line entry point for audited StageWorld workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

import typer

from .artifacts import atomic_write_json
from .config import RunMode, StageWorldConfig, load_config
from .data import (
    DataMode,
    ExcelAuditConfig,
    HMACPseudonymizer,
    audit_excel,
    build_paired_ct_cohort,
    load_paired_ct_mapping,
    run_selected_ct_qc,
    write_paired_ct_artifacts,
    write_redacted_excel_audit,
)
from .errors import ConfigurationError, DataContractError, PermissionGateError, StageWorldError
from .pipeline import (
    generate_synthetic_report,
    run_synthetic_evaluation,
    run_synthetic_prediction,
)
from .real_survival import SUPPORTED_OS_PROTOCOLS
from .real_workflow import extract_real_ct_features, run_real_world_pretraining_smoke
from .release_scan import scan_release_tree
from .runtime import doctor_report
from .synthetic_workflow import (
    build_synthetic_cohort,
    extract_synthetic_features,
    make_synthetic_artifacts,
    run_synthetic_training,
)
from .training import TrainingPhase

app = typer.Typer(
    name="stageworld",
    help="StageWorld-GC research commands with explicit data and permission modes.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """StageWorld-GC command group."""


def _checked_config(path: Path, *, command: str, supervised: bool = False) -> StageWorldConfig:
    config = load_config(path)
    config.validate(command=command, supervised=supervised)
    return config


def _emit(payload: dict[str, object]) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


def _fail(error: StageWorldError) -> None:
    _emit({"status": "error", "error": error.as_dict()})
    raise typer.Exit(code=2)


@app.command("doctor")
def doctor(
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    write_report: Annotated[bool, typer.Option("--write-report/--no-write-report")] = True,
) -> None:
    """Inspect local capabilities without loading clinical rows or model weights."""

    try:
        config = _checked_config(config_path, command="doctor")
        report = doctor_report(config)
        report["status"] = "ok" if not report["safety_findings"] else "blocked"
        if write_report:
            destination = config.output_root / "audit" / "doctor.json"
            atomic_write_json(destination, report)
            report["report"] = str(destination)
        _emit(report)
    except StageWorldError as error:
        _fail(error)


@app.command("audit")
def audit(
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
) -> None:
    """Run a redacted configuration/data-structure audit without building labels."""

    try:
        config = _checked_config(config_path, command="audit")
        destination = config.output_root / "audit" / "data_audit.json"
        if config.mode is RunMode.SYNTHETIC:
            payload: dict[str, object] = {
                "status": "ok",
                "mode": "synthetic",
                "schema_version": config.project.schema_version,
                "contains_real_clinical_data": False,
                "endpoint_contract_confirmed": config.clinical.endpoint_confirmed,
                "time_contract_confirmed": config.clinical.time_contract_confirmed,
                "source_state": "generated"
                if (config.output_root / "data" / "cohort.json").is_file()
                else "not_yet_generated",
            }
            atomic_write_json(destination, payload)
        else:
            if not config.paths.clinical_excel:
                raise ConfigurationError(
                    code="CLINICAL_WORKBOOK_REQUIRED",
                    message="Real-data audit requires paths.clinical_excel.",
                )
            source = Path(config.paths.clinical_excel)
            if not source.is_file():
                raise ConfigurationError(
                    code="CLINICAL_WORKBOOK_NOT_FOUND",
                    message="Configured clinical workbook does not exist.",
                )
            audit_config = ExcelAuditConfig(header_rows=(1,))
            if config.paths.field_mapping:
                mapping = load_paired_ct_mapping(config.paths.field_mapping)
                audit_config = ExcelAuditConfig(
                    header_rows=(mapping.field_header_row,),
                    date_columns=frozenset(
                        {
                            mapping.columns["baseline_origin"],
                            mapping.columns["death_date"],
                            mapping.columns["censor_date"],
                        }
                    ),
                    field_encodings={
                        mapping.columns["os_status"]: frozenset(
                            {str(mapping.event_code), str(mapping.censor_code)}
                        )
                    },
                )
            report = audit_excel(source, audit_config)
            write_redacted_excel_audit(report, destination)
            payload = {
                "status": "ok",
                "mode": config.mode.value,
                "record_count": report.record_count,
                "column_count": report.column_count,
                "duplicate_header_count": report.duplicate_header_count,
                "error_cell_count": report.error_cell_count,
                "report_contains_cell_values": False,
            }
        payload["report"] = str(destination)
        _emit(payload)
    except StageWorldError as error:
        _fail(error)


@app.command("make-synthetic")
def make_synthetic(
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    out: Annotated[Path | None, typer.Option("--out", file_okay=False)] = None,
    patient_count: Annotated[int, typer.Option("--patient-count", min=8)] = 12,
) -> None:
    """Generate a completely fictional cohort and numeric source tensors."""

    try:
        config = _checked_config(config_path, command="make-synthetic")
        _emit(make_synthetic_artifacts(config, out=out, patient_count=patient_count))
    except StageWorldError as error:
        _fail(error)


@app.command("build-cohort")
def build_cohort(
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    dicom_workers: Annotated[
        int,
        typer.Option("--dicom-workers", min=1, max=64),
    ] = 8,
) -> None:
    """Build legal prefixes and outcome landmarks after endpoint validation."""

    try:
        config = _checked_config(config_path, command="build-cohort", supervised=True)
        if config.mode is RunMode.SYNTHETIC:
            _emit(build_synthetic_cohort(config))
            return
        required_paths = {
            "clinical_excel": config.paths.clinical_excel,
            "ct_root": config.paths.ct_root,
            "field_mapping": config.paths.field_mapping,
            "identity_hmac_key_file": config.paths.identity_hmac_key_file,
        }
        missing_paths = [name for name, value in required_paths.items() if not value]
        if missing_paths:
            raise ConfigurationError(
                code="REAL_COHORT_PATHS_REQUIRED",
                message="The paired-CT adapter requires workbook, CT, mapping, and HMAC paths.",
                details={"missing": missing_paths},
            )
        if config.clinical.development_stages != ("s0", "s1"):
            raise ConfigurationError(
                code="PAIRED_COHORT_STAGE_SCOPE_MISMATCH",
                message="The current real adapter is approved only for S0/S1 development.",
            )
        mapping = load_paired_ct_mapping(str(required_paths["field_mapping"]))
        configured_status = dict(config.clinical.os_event_mapping or {})
        if configured_status != {"death": 1, "alive_or_censored": 0}:
            raise ConfigurationError(
                code="OS_V1_STATUS_CONTRACT_MISMATCH",
                message="The project configuration must encode death=1 and alive/censored=0.",
            )
        if config.clinical.baseline_origin_definition != "clinical_excel_column_E":
            raise ConfigurationError(
                code="OS_V1_ORIGIN_CONTRACT_MISMATCH",
                message="The project configuration must use Excel column E as the OS origin.",
            )
        project_root = Path(__file__).resolve().parents[2]
        pseudonymizer = HMACPseudonymizer.from_file(
            str(required_paths["identity_hmac_key_file"]),
            project_root=project_root,
        )
        horizon_days = max(config.survival.report_horizons_years) * 365.25
        result = build_paired_ct_cohort(
            workbook_path=str(required_paths["clinical_excel"]),
            ct_root=str(required_paths["ct_root"]),
            mapping=mapping,
            pseudonymizer=pseudonymizer,
            mode=DataMode(config.mode.value),
            metadata_cache_root=(config.output_root / "data" / "restricted" / "dicom_metadata"),
            split_seed=config.training.seed,
            prediction_horizon_days=horizon_days,
            selection_policy_version=(config.clinical.ct_series_selection_policy_version),
            workers=dicom_workers,
        )
        _emit(
            write_paired_ct_artifacts(
                result,
                output_root=config.output_root,
                split_seed=config.training.seed,
            )
        )
    except StageWorldError as error:
        _fail(error)


@app.command("qc-ct")
def qc_ct(
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    sample_per_role: Annotated[
        int,
        typer.Option("--sample-per-role", min=1, max=350),
    ] = 20,
    pixel_slices: Annotated[
        int,
        typer.Option("--pixel-slices", min=1, max=9),
    ] = 3,
) -> None:
    """Run bounded, outcome-blind automated QC on selected CT volumes."""

    try:
        config = _checked_config(config_path, command="qc-ct")
        if config.mode is not RunMode.REAL_IMAGES:
            raise ConfigurationError(
                code="REAL_IMAGE_MODE_REQUIRED",
                message="Selected-volume CT QC requires real_images mode.",
            )
        if not config.paths.approved_data_root or not config.paths.identity_hmac_key_file:
            raise ConfigurationError(
                code="CT_QC_PATHS_REQUIRED",
                message="CT QC requires an approved data root and identity HMAC key.",
            )
        project_root = Path(__file__).resolve().parents[2]
        pseudonymizer = HMACPseudonymizer.from_file(
            config.paths.identity_hmac_key_file,
            project_root=project_root,
        )
        _emit(
            run_selected_ct_qc(
                asset_manifest_path=(
                    config.output_root / "data" / "restricted" / "asset_bindings.json"
                ),
                approved_root=config.paths.approved_data_root,
                output_root=config.output_root,
                pseudonymizer=pseudonymizer,
                sample_per_role=sample_per_role,
                pixel_slices=pixel_slices,
                sample_seed=config.training.seed,
            )
        )
    except (OSError, ValueError):
        _fail(
            DataContractError(
                code="CT_QC_ARTIFACT_UNREADABLE",
                message="The selected CT manifest could not be read for QC.",
            )
        )
    except StageWorldError as error:
        _fail(error)


@app.command("extract-features")
def extract_features(
    modality: Annotated[str, typer.Option("--modality")],
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    limit_per_split: Annotated[
        int | None,
        typer.Option("--limit-per-split", min=1),
    ] = None,
    include_test: Annotated[
        bool,
        typer.Option("--include-test/--exclude-test"),
    ] = False,
    device: Annotated[str, typer.Option("--device")] = "cuda",
) -> None:
    """Extract explicitly selected CT or pathology features with no model fallback."""

    try:
        if modality not in {"ct", "pathology"}:
            raise ConfigurationError(
                code="UNSUPPORTED_MODALITY",
                message="--modality must be ct or pathology.",
            )
        selected_modality: Literal["ct", "pathology"] = "ct" if modality == "ct" else "pathology"
        config = _checked_config(config_path, command="extract-features")
        if config.mode is RunMode.SYNTHETIC:
            _emit(extract_synthetic_features(config, selected_modality))
            return
        if selected_modality == "ct" and config.mode is RunMode.REAL_IMAGES:
            if not config.paths.identity_hmac_key_file:
                raise ConfigurationError(
                    code="HMAC_KEY_REQUIRED",
                    message="Real CT extraction requires the existing private identity key.",
                )
            project_root = Path(__file__).resolve().parents[2]
            pseudonymizer = HMACPseudonymizer.from_file(
                config.paths.identity_hmac_key_file,
                project_root=project_root,
            )
            _emit(
                extract_real_ct_features(
                    config,
                    pseudonymizer=pseudonymizer,
                    limit_per_split=limit_per_split,
                    include_test=include_test,
                    device=device,
                )
            )
            return
        if config.mode is not RunMode.SYNTHETIC:
            raise PermissionGateError(
                code="AUTHORIZED_ENCODER_RUNTIME_REQUIRED",
                message=(
                    "Real feature extraction needs approved local weights and parity validation."
                ),
                remediation=(
                    "Provide approved weight paths/revisions; no weight is downloaded "
                    "automatically."
                ),
                details={
                    "encoder": config.model.ct_encoder
                    if modality == "ct"
                    else config.model.pathology_encoder
                },
            )
    except StageWorldError as error:
        _fail(error)


@app.command("train")
def train(
    phase: Annotated[TrainingPhase, typer.Option("--phase")],
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = False,
) -> None:
    """Run a bounded world-pretraining or joint-survival phase."""

    try:
        config = _checked_config(
            config_path,
            command="train",
            supervised=phase is TrainingPhase.JOINT_SURVIVAL,
        )
        if config.mode is RunMode.SYNTHETIC:
            _emit(run_synthetic_training(config, phase=phase, resume=resume))
            return
        if (
            phase is TrainingPhase.JOINT_SURVIVAL
            and config.training.development_protocol in SUPPORTED_OS_PROTOCOLS
        ):
            from stageworld.real_survival import run_real_os_development

            if resume:
                raise PermissionGateError(
                    code="REAL_OS_RESUME_REQUIRES_RUN_SELECTION",
                    message="Automatic resume is not enabled; prior run artifacts are preserved.",
                )
            _emit(run_real_os_development(config))
            return
        if config.mode is RunMode.REAL_IMAGES and phase is TrainingPhase.WORLD_PRETRAIN:
            _emit(run_real_world_pretraining_smoke(config, resume=resume))
            return
        if config.mode is not RunMode.SYNTHETIC:
            raise PermissionGateError(
                code="REAL_TRAINING_NOT_APPROVED",
                message="Only bounded outcome-blind real world pretraining is currently enabled.",
                remediation=(
                    "Audit the smoke checkpoint before enabling outcome-supervised training."
                ),
            )
    except StageWorldError as error:
        _fail(error)


@app.command("predict")
def predict(
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    query_file: Annotated[
        Path | None, typer.Option("--query-file", exists=True, dir_okay=False)
    ] = None,
) -> None:
    """Replay legal anonymous history and predict conditional OS at requested nodes."""

    try:
        config = _checked_config(config_path, command="predict")
        if config.training.development_protocol in SUPPORTED_OS_PROTOCOLS:
            from stageworld.real_survival import predict_real_os_development

            if query_file is not None:
                raise PermissionGateError(
                    code="REAL_OS_FIXED_POPULATION_ONLY",
                    message="This protocol predicts the fixed validation population only.",
                )
            _emit(predict_real_os_development(config))
            return
        if config.mode is not RunMode.SYNTHETIC:
            raise PermissionGateError(
                code="REAL_INFERENCE_NOT_VALIDATED",
                message=(
                    "Real feature/image inference awaits approved weights and schema validation."
                ),
            )
        _emit(run_synthetic_prediction(config, query_file=query_file))
    except StageWorldError as error:
        _fail(error)


@app.command("predict-generated-s1")
def predict_generated_s1(
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    query_file: Annotated[Path, typer.Option("--query-file", exists=True, dir_okay=False)],
    output_file: Annotated[Path, typer.Option("--output-file")],
    device: Annotated[str, typer.Option("--device")] = "cpu",
) -> None:
    """Predict generated S1 from a portable model and baseline-only feature packet."""
    from stageworld.generated_inference import predict_generated_query

    try:
        _emit(predict_generated_query(config_path, query_file, output_file, device=device))
    except StageWorldError as error:
        _fail(error)


@app.command("evaluate")
def evaluate(
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    predictions: Annotated[
        Path | None, typer.Option("--predictions", exists=True, dir_okay=False)
    ] = None,
    protocol: Annotated[
        Path | None, typer.Option("--protocol", exists=True, dir_okay=False)
    ] = None,
) -> None:
    """Evaluate versioned predictions under a predeclared censoring protocol."""

    try:
        config = _checked_config(config_path, command="evaluate", supervised=True)
        if config.training.development_protocol in SUPPORTED_OS_PROTOCOLS:
            from stageworld.real_survival import evaluate_real_os_development

            if protocol is not None:
                raise PermissionGateError(
                    code="REAL_OS_FIXED_PROTOCOL_ONLY",
                    message="Use the locked ROI OS protocol.",
                )
            _emit(evaluate_real_os_development(config, predictions=predictions))
            return
        if config.mode is not RunMode.SYNTHETIC:
            raise PermissionGateError(
                code="REAL_EVALUATION_NOT_APPROVED",
                message="Real evaluation remains blocked by the unsigned clinical protocol.",
            )
        _emit(run_synthetic_evaluation(config, predictions=predictions, protocol_path=protocol))
    except StageWorldError as error:
        _fail(error)


@app.command("report")
def report(
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    run_dir: Annotated[Path | None, typer.Option("--run-dir", file_okay=False)] = None,
) -> None:
    """Generate a report only from persisted predictions, metrics, and lineage."""

    try:
        config = _checked_config(config_path, command="report")
        if config.training.development_protocol in SUPPORTED_OS_PROTOCOLS:
            from stageworld.real_report import generate_real_os_report

            if run_dir is not None:
                raise PermissionGateError(
                    code="REAL_OS_SELECTED_RUN_ONLY",
                    message="Report the selected development run.",
                )
            _emit(generate_real_os_report(config))
            return
        if config.mode is not RunMode.SYNTHETIC:
            raise PermissionGateError(
                code="REAL_REPORT_NOT_APPROVED",
                message="A real report requires a completed, locked clinical evaluation.",
            )
        _emit(generate_synthetic_report(config, run_dir=run_dir))
    except StageWorldError as error:
        _fail(error)


@app.command("release-scan")
def release_scan(
    root: Annotated[
        Path,
        typer.Option("--root", exists=True, file_okay=False),
    ] = Path("."),
) -> None:
    """Run a count-only credential, sensitive-path, and raw-asset release gate."""

    try:
        payload = scan_release_tree(root).as_dict()
        _emit(payload)
        if payload["status"] != "ok":
            raise typer.Exit(code=2)
    except StageWorldError as error:
        _fail(error)


if __name__ == "__main__":
    app()
