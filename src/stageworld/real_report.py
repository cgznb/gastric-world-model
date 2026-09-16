"""Aggregate-only reporting from executed real OS development artifacts."""

from __future__ import annotations

from typing import Any

from stageworld.artifacts import read_json
from stageworld.config import StageWorldConfig
from stageworld.errors import ArtifactError


def write_real_os_report(config: StageWorldConfig, summary: dict[str, Any]) -> str:
    if summary.get("status") != "completed" or summary.get("clinical_validation") is not False:
        raise ArtifactError(
            code="OS_REPORT_REQUIRES_COMPLETION", message="No completed development run."
        )
    root = config.output_root / "runs" / summary["run_id"]
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    conditioning = (
        "CT and five confirmed pre-CT treatment-modality summaries plus elapsed time; "
        "no individual drug regimens, doses or cycle dates."
        if summary.get("treatment_artifact_id")
        else "CT and elapsed time only."
    )
    if (summary.get("treatment_inputs") or {}).get("drug_regimens_used"):
        conditioning = (
            "CT, five confirmed pre-CT modalities, allowlisted named-regimen/drug mentions, "
            "and train-standardized log cycle totals. No raw text, doses or per-cycle dates."
        )
    tumor_candidates = (
        summary.get("roi_target") == "unreviewed_gastric_associated_pan_cancer_candidates"
    )
    coarse_tumor = summary.get("roi_target") == (
        "tumor_guided_context_with_explicit_stomach_fallback"
    )
    roi_description = (
        "Tumor-guided coarse CT crops: FLARE23 candidates within 40 mm of stomach, "
        "30-mm margin; stomach plus 40 mm when no candidate is found. "
        "Intact CT context is retained; fallback is not a tumor detection or complete response."
        if coarse_tumor else
        "FLARE23 pan-cancer tumor candidates associated with stomach by a 10-mm proximity rule. "
        "Primary-tumor identity and segmentation accuracy have not been expert-validated."
        if tumor_candidates
        else "Stomach-organ ROI, not tumor segmentation."
    )
    cohort_intro = (
        "Pairs with two usable contextual crops enter, including stomach fallbacks. The original "
        if coarse_tumor else
        "Only pairs with two nonempty tumor-candidate ROIs enter this model. The original "
        if tumor_candidates
        else "Only pairs with two passing gastric ROIs enter this model. The original "
    )
    lines = [
        "# Tumor-Guided Coarse ROI OS: Development Results"
        if coarse_tumor else "# Gastric Tumor Candidate ROI OS: Development Results"
        if tumor_candidates else "# Gastric ROI OS: Development Results",
        "",
        "Research development only. Not held-out clinical validation or a clinical decision tool.",
        "",
        f"Run: `{summary['run_id']}`. Seed17. {roi_description}",
        "",
        f"Conditioning: {conditioning}",
        "",
        f"Training: {summary['training_patients']} patients, {summary['training_deaths']} deaths. "
        f"Validation: {summary['validation_patients']} patients, "
        f"{summary['validation_deaths']} deaths. "
        f"Reserved test: {summary['reserved_test_patients']} patients, not used.",
        "",
        f"World pretraining: {summary['pretraining_steps']} steps. "
        f"Joint survival: {summary['joint_optimizer_steps']} steps. "
        f"One shared checkpoint selected at epoch{summary['selected_epoch']} by validation NLL.",
        "",
        *(
            [
                f"Completed epochs: world {summary['pretraining_completed_epochs']} "
                f"(+{summary['pretraining_partial_epoch_steps']} partial-epoch steps), "
                f"joint {summary['joint_completed_epochs']}. "
                "The final-epoch weights are saved separately from the validation-selected "
                f"weights: `{summary['final_checkpoint']}`. "
                f"Final-epoch validation NLL: {summary['final_validation_nll']:.8f}.",
                "",
            ]
            if "final_checkpoint" in summary
            else []
        ),
        "OS labels reuse the original flattened-source artifact: BS1=death/BS0=censored; "
        "E origin, BU death, BV censoring. The grouped treatment workbook does not replace labels. "
        "Horizons are remaining years after each S0/S1 query.",
        "",
        "| Method | Stage | Horizon (years) | Metric | Estimate | "
        "95% Patient-Bootstrap Interval | Events |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in summary["metrics"]:
        estimate = "not estimable" if row["estimate"] is None else f"{row['estimate']:.4f}"
        interval = row.get("confidence_interval_95")
        interval_text = (
            "not estimable" if interval is None else f"{interval[0]:.4f} to {interval[1]:.4f}"
        )
        horizon = "0.25-3.0" if row["metric"] == "integrated_brier_score" else str(row["horizon"])
        lines.append(
            f"| {row['method']} | {row['stage']} | {horizon} | {row['metric']} | "
            f"{estimate} | {interval_text} | {row['n_events']} |"
        )
    lines += [
        "",
        "Censoring weights are estimated from the corresponding training-stage cohort. "
        "Bootstrap intervals condition on the selected model "
        "and do not adjust for model selection. "
        "The validation set selected the model, so these estimates can be optimistic.",
        "",
        "A lower Brier/IBS is better; it is not a discrimination metric. "
        "Without deaths by a horizon, a small Brier score does not show that "
        "the model can distinguish future deaths.",
        "",
        "Non-estimable metric reasons: "
        + "; ".join(
            sorted(
                {
                    f"{row['stage']} {row['horizon']}y {row['metric']}: {row['reason']}"
                    for row in summary["metrics"]
                    if row["status"] == "not_estimable"
                }
            )
        ),
        "",
        "## CT Quality Control",
        "",
    ]
    extraction_path = config.output_root / "features/ct_summary.json"
    if extraction_path.exists():
        extraction = read_json(extraction_path)
        lines.append(
            f"Requested studies: {extraction['requested_study_count']}. "
            f"Studies in complete passing pairs: {extraction['complete_study_count']}. "
            f"Failures by automatic QC/error code: {extraction['failure_counts']}."
        )
        if coarse_tumor:
            lines.append(
                "ROI sources before pair filtering: "
                f"{extraction.get('roi_source_counts_before_pair_filter', {})}. "
                "ROI sources in retained pairs: "
                f"{extraction.get('roi_source_counts_in_complete_pairs', {})}."
            )
    else:
        lines.append(
            "The fixed CT cache is reused; no new segmentation or extraction was performed."
            if summary.get("treatment_artifact_id")
            else "An extraction aggregate is not available for this run."
        )
    lines += [
        "",
        cohort_intro + f"{summary['source_cohort_patients']}-patient "
        "cohort and patient split are preserved. First acquisition is used independently at each "
        "stage; no alternate phase or center-crop fallback is substituted.",
        "",
        f"Excluded pairs by original split: {summary['excluded_roi_pair_counts']}.",
        "",
        "## Future CT Prediction",
        "",
        "Scoring these fixed-feature comparisons does not use outcome labels. "
        "MSE is lower-is-better; it does not measure generated-image quality. "
        "Evaluated checkpoint phase: "
        f"{summary.get('future_ct_validation_phase', 'world_pretrain')}. "
        "Historical v1 summaries lacking a phase field were evaluated after world pretraining, "
        "not after OS checkpoint selection.",
        "",
        "| Predictor | Validation MSE |",
        "|---|---|",
        f"| World model ({summary.get('future_ct_validation_phase', 'world_pretrain')}) | "
        f"{summary['future_ct_validation']['world_model_mse']:.6f} |",
        *(
            [
                "| World model (world_pretrain) | "
                f"{summary['future_ct_pretraining']['world_model_mse']:.6f} |"
            ]
            if summary.get("future_ct_pretraining")
            else []
        ),
        f"| Baseline CT persistence | {summary['future_ct_validation']['persistence_mse']:.6f} |",
        f"| Training-set mean | {summary['future_ct_validation']['training_mean_mse']:.6f} |",
        "",
        "The world-model future-prediction hypothesis is not supported when these "
        "simple controls have lower error. No causal or clinical benefit is inferred.",
        "",
        "## Limitations",
        "",
        (
            "Coarse localization is unreviewed and no cohort segmentation labels were used. "
            "A stomach fallback is an anatomical crop, not a confirmed tumor mask. "
            "Candidate and fallback crops can differ in location and scale; both are recorded. "
            "Future CT feature prediction does not predict tumor masks, volume or growth. "
            "Few deaths; one seed; unknown CT phase; no clinical covariates or pathology."
            if coarse_tumor else
            "Automatic pan-cancer predictions and heuristic stomach association are unreviewed. "
            "Positive-pair selection may remove complete responders and segmentation failures. "
            "This is an exploratory, detection-conditioned cohort, not full-cohort evidence. "
            "Future CT feature prediction does not predict tumor masks, volume or growth. "
            "Few deaths; one seed; unknown CT phase; no clinical covariates or pathology."
            if tumor_candidates else
            "Few deaths; single seed; retrospective complete-pair selection; "
            "automatic stomach masks without expert validation; unknown CT phase; "
            "no baseline clinical covariates, tumor masks, pathology, external validation "
            "or causal treatment-effect claim."
        ),
        " Complete-pair selection and gastric-QC exclusions restrict applicability; "
        "S0 results must not be generalized to all baseline patients. S0 and S1 are "
        "different query times: their metric difference is not a matched-time estimate "
        "of the benefit of acquiring another CT.",
        "",
        "## Lineage",
        "",
        f"Feature artifact: `{summary['ct_feature_artifact_id']}`.",
        "",
        f"Selected checkpoint: `{summary['checkpoint_id']}`; "
        f"weights: `{summary['weight_version']}`.",
        "",
        f"Parent: `{summary['parent_checkpoint_id']}`; "
        f"predictions: `{summary['prediction_artifact_id']}`.",
    ]
    if summary.get("treatment_artifact_id"):
        lines += [
            "",
            "## Treatment Input Audit",
            "",
            f"Treatment artifact: `{summary['treatment_artifact_id']}`.",
            "",
            "Summaries are assigned to the target CT boundary as interval descriptors, "
            "not invented dosing timestamps. They do not enter observed baseline S0 risks. "
            "Future scenarios require explicit assumptions and are not causal treatment effects.",
            "",
            f"Structural input-perturbation diagnostic: {summary.get('treatment_response_probe')}.",
        ]
        if summary.get("regimen_response_probe"):
            audit = summary["treatment_inputs"]
            lines += [
                "",
                f"Named-code coverage: {audit['patients_with_named_regimen_code']}/"
                f"{audit['patients']}; any named descriptor: "
                f"{audit['patients_with_any_named_mention']}/{audit['patients']}.",
                "",
                "Literal mentions are not an adjudicated drug dictionary or a reconstructed "
                "dosing sequence. Unmentioned names stay unknown, not confirmed absent. "
                "Cycle counts outside 0-60, invalid counts and modality conflicts stay unknown; "
                "the cycle transform is fitted on training patients only.",
                "",
                f"Regimen input diagnostic: {summary['regimen_response_probe']}.",
            ]
    report = root / "os_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report.chmod(0o600)
    if summary.get("training_history"):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(7, 4))
        history = summary["training_history"]
        axis.plot(
            [x["epoch"] for x in history],
            [x["validation_stage_mean_nll"] for x in history],
            color="#176b58",
            marker=".",
        )
        axis.axvline(summary["selected_epoch"], color="#b94b4b", linestyle="--")
        axis.set(
            xlabel="Epoch",
            ylabel="Mean S0/S1 validation OS NLL",
            title="Real CT development: shared-checkpoint selection",
        )
        axis.grid(alpha=0.2)
        figure.tight_layout()
        image_path = root / "validation_nll.png"
        figure.savefig(image_path, dpi=140)
        plt.close(figure)
        image_path.chmod(0o600)
    return str(report)


def generate_real_os_report(config: StageWorldConfig) -> dict[str, Any]:
    from stageworld.real_survival import _load_selected_development_model

    summary, _, _, _ = _load_selected_development_model(config)
    path = write_real_os_report(config, summary)
    return {"status": "ok", "report": path, "clinical_validation": False, "development_only": True}
