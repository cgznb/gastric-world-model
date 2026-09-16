"""Locked architecture comparisons on the existing original700 folds."""

from dataclasses import asdict

from stageworld.binary700_spec import Candidate

SEEDS = (17, 43, 97)
STATISTICAL = (
    Candidate("logistic", "logistic"),
    Candidate("logistic_balanced", "logistic", weight="balanced"),
)
NEURAL = (
    Candidate("generated_v2_frozen_bce", "generated_frozen"),
    Candidate("generated_v2_bce", "generated"),
    Candidate("generated_v2_balanced", "generated", weight="balanced"),
    Candidate("generated_v2_focal", "generated", "focal", "balanced"),
)


def specification() -> dict:
    return {
        "schema": "gastric-generated700-v2",
        "patients": 700,
        "outer_folds": 5,
        "outer_train_evaluation": [560, 140],
        "inner_train_validation": [448, 112],
        "fresh_refit": True,
        "seeds": list(SEEDS),
        "candidates": [asdict(c) for c in (*STATISTICAL, *NEURAL)],
        "hidden": 128,
        "transition_layers": 4,
        "attention_heads": 4,
        "attention_fastpath": False,
        "spatial_tokens": 27,
        "condition_tokens": 6,
        "prediction_head": "independent_two_way_attention_gated_pool",
        "anchor": "ordinary_unweighted_logistic",
        "inner_residual_scales": [0.0, 0.25, 0.5, 1.0],
        "logistic_C": [0.01, 0.1, 1.0, 10.0],
        "world_loss": "global_SmoothL1+cosine+0.25_sliced_Wasserstein+0.1_std_SmoothL1",
        "joint_CT_weight": 0.1,
        "neural_selection": "mean_inner_endpoint_AUPRC",
        "world_selection": "inner_world_loss",
        "max_epochs": 100,
        "patience": 15,
        "min_delta": 0.0001,
        "batch_size": 32,
        "learning_rate": 0.0002,
        "joint_world_learning_rate": 0.00002,
        "weight_decay": 0.01,
        "clip_norm": 1.0,
        "focal_gamma": 2.0,
        "runtime_limit": None,
        "spatial_registration_assumed": False,
        "CT1_target": "unordered_27x768_frozen_feature_set",
        "external_validation": False,
        "reference_study": "gastric_world_model_binary700_20260916",
    }
