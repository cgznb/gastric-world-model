"""Prespecified event-only terminal classification experiment."""

from stageworld.generated651_spec import SEEDS

TASK = "event_multistage_terminal_binary_v1"
ABSENT, PRESENT, UNKNOWN, CONFLICT = range(4)
STAGES = ("baseline", "neoadjuvant", "surgery", "postoperative")


def specification() -> dict:
    return {
        "task": TASK,
        "patients": 651,
        "folds": 5,
        "seeds": list(SEEDS),
        "stages": list(STAGES),
        "event_status": ["absent", "present", "unknown", "conflict"],
        "time_inputs": False,
        "cycles_used": False,
        "observed_followup_inputs": False,
        "terminal_only_recurrence": True,
        "logistic_anchor_in_model": False,
        "postoperative_policy": "use_recorded_yes_only_after_confirmed_surgery",
        "hidden": 128,
        "layers": 4,
        "heads": 4,
        "state_tokens": 27,
        "pretrain_loss": "CT_set_loss + 0.5 * ordinary_pCR_BCE",
        "joint_loss": "balanced_recurrence_BCE + 0.5 * ordinary_pCR_BCE + 0.1 * CT_set_loss",
        "pretrain_selection": "minimum_validation_pretrain_loss",
        "joint_selection": "maximum_validation_recurrence_AUPRC",
        "max_epochs": 100,
        "patience": 15,
        "min_delta": 0.0001,
        "learning_rate": 0.0002,
        "joint_world_learning_rate": 0.00002,
        "weight_decay": 0.01,
        "batch_size": 32,
        "clip_norm": 1.0,
        "runtime_limit": None,
        "threshold": 0.5,
        "calibrated": False,
        "aggregation": "each_seed_separately_fivefold_mean_and_sample_SD",
        "selection_and_reporting": "same_validation_fold_development_results",
        "endpoint": "recorded_recurrence_metastasis_status_not_new_event_after_treatment",
        "real_small_scale_trial": "omitted_at_user_request",
    }
