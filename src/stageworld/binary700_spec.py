"""Finite, prespecified comparisons on the complete original 700-patient cohort."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    loss: str = "bce"
    weight: str = "none"


STATISTICAL = (
    Candidate("logistic", "logistic"),
    Candidate("logistic_sqrt", "logistic", weight="sqrt"),
    Candidate("logistic_balanced", "logistic", weight="balanced"),
    Candidate("histgb", "histgb"),
    Candidate("histgb_balanced", "histgb", weight="balanced"),
)
NEURAL = (
    Candidate("tabular_bce", "tabular"),
    Candidate("tabular_balanced", "tabular", weight="balanced"),
    Candidate("tabular_focal", "tabular", "focal", "balanced"),
    Candidate("ct_bce", "ct"),
    Candidate("ct_balanced", "ct", weight="balanced"),
    Candidate("generated_bce", "generated"),
    Candidate("generated_balanced", "generated", weight="balanced"),
    Candidate("generated_focal", "generated", "focal", "balanced"),
)
SEEDS = (17, 43, 97)


def specification() -> dict:
    return {
        "schema": "gastric-binary700-v1",
        "patients": 700,
        "outer_folds": 5,
        "outer_training_patients": 560,
        "outer_evaluation_patients": 140,
        "inner_train_validation": [448, 112],
        "refit_all_outer_training": True,
        "seeds": list(SEEDS),
        "candidates": [asdict(c) for c in (*STATISTICAL, *NEURAL)],
        "logistic_C": [0.01, 0.1, 1.0, 10.0],
        "neural_selection": "mean_inner_endpoint_AUPRC",
        "world_selection": "inner_CT_SmoothL1_plus_cosine",
        "max_epochs": 100,
        "patience": 15,
        "min_delta": 0.0001,
        "batch_size": 32,
        "learning_rate": 0.0002,
        "weight_decay": 0.01,
        "clip_norm": 1.0,
        "focal_gamma": 2.0,
        "operating_points": ["raw_0.5", "inner_balanced", "inner_sensitivity_0.8"],
        "calibration": "inner_only_monotone_Platt_C1",
        "holdout_role": "former_106_merged_into_development_by_user",
        "external_validation": False,
        "spatial_target_loss": False,
        "runtime_limit": None,
    }
