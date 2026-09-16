"""Locked, finite comparisons; outcome definitions remain those of the completed study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Arm:
    name: str
    architecture: str
    policy: str = "joint"
    positive_weight: str = "none"
    endpoint: int | None = None
    outer_evaluation: bool = True


ARMS = (
    Arm("direct", "direct"),
    Arm("residual_joint", "residual"),
    Arm("legacy_deterministic", "legacy_deterministic"),
    Arm("residual_frozen", "residual", "frozen"),
    Arm("residual_warm", "residual", "warm"),
    Arm("residual_weighted", "residual", "warm", "sqrt"),
    Arm("single_pcr_diagnostic", "residual", "warm", endpoint=0, outer_evaluation=False),
    Arm("single_recurrence_diagnostic", "residual", "warm", endpoint=1, outer_evaluation=False),
)
PROTOCOL = "ct6-binary-improvement-v1"


def study_spec() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "arms": [asdict(arm) for arm in ARMS],
        "seeds": [17, 43, 97],
        "folds": 5,
        "epochs_per_phase": 100,
        "patience": 15,
        "min_delta": 0.0001,
        "head_lr": 0.0002,
        "warm_backbone_lr": 0.00002,
        "frozen_head_epochs_before_warm": 5,
        "weight_decay": 0.01,
        "batch_size": 32,
        "ct_loss": "global_mean_frozen_768_SmoothL1_plus_cosine",
        "joint_ct_weight": 0.1,
        "kl_weight": 0,
        "selection": "own_inner_validation_mean_unweighted_endpoint_BCE_no_refit",
        "logistic_C_candidates": [0.01, 0.1, 1.0],
        "logistic_ct_pca_components": 32,
        "threshold_selection": "own_inner_validation_balanced_accuracy_tie_nearest_0.5",
        "weighted_probability_correction": "logit_minus_log_training_positive_weight",
        "calibration": "analytic_weight_correction_only_no_fitted_calibrator",
        "test_used": False,
        "spatial_loss_enabled": False,
        "spatial_loss_gate": "requires_verified_CT0_defined_paired_geometry",
        "stop_after_finite_candidates": True,
    }
