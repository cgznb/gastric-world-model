"""Locked internal-holdout experiment; test results never select models or seeds."""

from stageworld.generated651_spec import SEEDS

TASK = "event_multistage_v2_seed_holdout"
FAMILIES = (
    "event_v1",
    "event_v2",
    "event_v2_no_adapter",
    "event_v2_no_mask",
    "event_v2_single_member",
)


def specification() -> dict:
    return {
        "task": TASK,
        "patients": 651,
        "seeds": list(SEEDS),
        "families": ["event_v1", "event_v2"],
        "split": "per_seed_joint_pcr_recurrence_stratified_80_10_10",
        "partition_sizes": {"train": 521, "validation": 65, "test": 65},
        "test_gate": "all_seeds_and_families_selected_before_any_test_evaluation",
        "test_use": "final_reporting_only_no_selection_calibration_or_threshold_tuning",
        "independent_external_validation": False,
        "previously_used_development_cohort": True,
        "aggregation": "individual_seed_test_metrics_no_best_seed_selection_or_patient_pooling",
        "endpoint": "recorded_recurrence_metastasis_status_not_incident_risk",
        "time_inputs": False,
        "observed_followup_inputs": False,
        "hidden": 128,
        "v2_layers": 3,
        "v1_layers": 4,
        "adapter_rank": 8,
        "members": 4,
        "mask_probability": 0.25,
        "mask_loss_weight": 0.05,
        "pretrain_loss": "CT_set + 0.5*pCR_member_BCE + masked_CT0_auxiliary",
        "joint_loss": (
            "balanced_recurrence_member_BCE + 0.5*pCR_member_BCE "
            "+ 0.1*CT_set + masked_CT0_auxiliary"
        ),
        "pretrain_selection": "minimum_validation_CT_set_plus_0.5_pCR_BCE",
        "joint_selection": "maximum_validation_recurrence_AUPRC",
        "max_epochs": 100,
        "patience": 15,
        "min_delta": 0.0001,
        "batch_size": 32,
        "learning_rate": 0.0002,
        "joint_world_learning_rate": 0.00002,
        "weight_decay": 0.01,
        "clip_norm": 1.0,
        "threshold": 0.5,
        "calibrated": False,
        "precision": "BF16_CUDA_training_FP32_evaluation",
    }


def model_dimensions(family: str, image_dim: int, hidden: int = 128) -> dict:
    if family not in FAMILIES:
        raise ValueError("Unknown event holdout model family")
    if family == "event_v1":
        return {"image_dim": image_dim, "hidden": hidden, "layers": 4}
    return {
        "image_dim": image_dim,
        "hidden": hidden,
        "layers": 3,
        "rank": min(8, hidden),
        "members": 1 if family == "event_v2_single_member" else 4,
        "variant": "no_stage_adapter" if family == "event_v2_no_adapter" else "full",
    }
