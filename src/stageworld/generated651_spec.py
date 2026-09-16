"""Complete-case validation selection with separate fivefold results for each seed."""

from dataclasses import asdict

from stageworld.binary700_spec import Candidate
from stageworld.generated700_spec import specification as architecture_specification

SEEDS = (17, 43, 97, 131, 173, 211, 257, 307, 359, 419)
SPLIT_SEED = 17
NEURAL = (
    Candidate("generated_v2_bce", "generated"),
    Candidate("generated_v2_balanced", "generated", weight="balanced"),
    Candidate("generated_v2_focal", "generated", "focal", "balanced"),
)
ANCHOR_C = 1.0
RESIDUAL_SCALE = 1.0
THRESHOLD = 0.5


def specification() -> dict:
    architecture = architecture_specification()
    return {
        "schema": "generated651-fivefold-per-seed-v1",
        "patients": 651,
        "inclusion": "complete_CT0_CT1_and_both_recorded_binary_labels",
        "folds": 5,
        "split_seed": SPLIT_SEED,
        "seeds": list(SEEDS),
        "candidates": [asdict(candidate) for candidate in NEURAL],
        "inner_split": False,
        "fresh_refit": False,
        "selection_and_reporting_partition": "same_validation_fold",
        "external_validation": False,
        "aggregation": "each_seed_separately_mean_and_sample_SD_across_five_folds",
        "across_seed_aggregation": False,
        "anchor_C": ANCHOR_C,
        "anchor_fit": "ordinary_logistic_training_fold_only",
        "residual_scale": RESIDUAL_SCALE,
        "classification_threshold": THRESHOLD,
        "calibration": None,
        "validation_hyperparameter_search": False,
        "model_and_optimizer": {
            key: architecture[key]
            for key in (
                "hidden",
                "transition_layers",
                "attention_heads",
                "attention_fastpath",
                "spatial_tokens",
                "condition_tokens",
                "prediction_head",
                "world_loss",
                "joint_CT_weight",
                "max_epochs",
                "patience",
                "min_delta",
                "batch_size",
                "learning_rate",
                "joint_world_learning_rate",
                "weight_decay",
                "clip_norm",
                "focal_gamma",
                "runtime_limit",
                "CT1_target",
                "spatial_registration_assumed",
            )
        },
        "world_selection": "minimum_validation_CT_set_loss",
        "classifier_selection": "maximum_mean_validation_endpoint_AUPRC",
    }
