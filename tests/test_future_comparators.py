from __future__ import annotations

import pytest
import torch

from stageworld.data.split import SplitName
from stageworld.errors import DataContractError
from stageworld.future_comparators import (
    PersistenceFuturePredictor,
    TimeTreatmentFuturePredictor,
    TrainingMeanFuturePredictor,
)

TARGET_SPACE = "frozen-ct-global-v1"
CONDITION_SCHEMA = "elapsed-days-plus-known-treatment-v1"


def test_persistence_copies_only_valid_same_space_features() -> None:
    predictor = PersistenceFuturePredictor(target_space_id=TARGET_SPACE)
    source = torch.tensor(
        [
            [[1.0, 2.0], [float("nan"), float("nan")]],
            [[3.0, 4.0], [5.0, 6.0]],
        ]
    )
    valid = torch.tensor([[True, False], [True, True]])

    prediction = predictor(source, valid, source_space_id=TARGET_SPACE)

    torch.testing.assert_close(
        prediction.mean,
        torch.tensor([[[1.0, 2.0], [0.0, 0.0]], [[3.0, 4.0], [5.0, 6.0]]]),
    )
    assert torch.equal(prediction.valid, valid)
    assert prediction.metadata.task == "future_observation_representation"
    assert prediction.metadata.fit_split is None
    assert sum(parameter.numel() for parameter in predictor.parameters()) == 0

    with pytest.raises(DataContractError) as error:
        predictor(source, valid, source_space_id="different-target-space")
    assert error.value.code == "COMPARATOR_FEATURE_SPACE_MISMATCH"


def test_training_mean_uses_masked_training_targets_and_marks_unsupported_tokens() -> None:
    target = torch.tensor(
        [
            [[1.0, 3.0], [10.0, 20.0], [float("nan"), float("nan")]],
            [[3.0, 5.0], [999.0, 999.0], [float("nan"), float("nan")]],
            [[5.0, 7.0], [999.0, 999.0], [float("nan"), float("nan")]],
        ]
    )
    valid = torch.tensor([[True, True, False], [True, False, False], [True, False, False]])
    predictor = TrainingMeanFuturePredictor.fit(
        target,
        valid,
        target_space_id=TARGET_SPACE,
        split=SplitName.TRAIN,
    )

    prediction = predictor(2, target_space_id=TARGET_SPACE)

    expected = torch.tensor([[[3.0, 5.0], [10.0, 20.0], [0.0, 0.0]]] * 2)
    torch.testing.assert_close(prediction.mean, expected)
    assert torch.equal(
        prediction.valid,
        torch.tensor([[True, True, False], [True, True, False]]),
    )
    assert torch.equal(predictor.support_counts, torch.tensor([3, 1, 0]))
    assert prediction.metadata.fit_split is SplitName.TRAIN
    assert sum(parameter.numel() for parameter in predictor.parameters()) == 0


@pytest.mark.parametrize("split", [SplitName.VALIDATION, SplitName.TEST])
def test_fitted_future_comparators_reject_nontraining_splits(split: SplitName) -> None:
    target = torch.ones(3, 1, 2)
    valid = torch.ones(3, 1, dtype=torch.bool)
    with pytest.raises(DataContractError) as error:
        TrainingMeanFuturePredictor.fit(
            target,
            valid,
            target_space_id=TARGET_SPACE,
            split=split,
        )
    assert error.value.code == "COMPARATOR_FIT_SPLIT_LEAKAGE"

    with pytest.raises(DataContractError) as error:
        TimeTreatmentFuturePredictor.fit(
            target,
            valid,
            torch.arange(3, dtype=torch.float32),
            torch.ones(3, 1),
            target_space_id=TARGET_SPACE,
            condition_schema_id=CONDITION_SCHEMA,
            split=split,
        )
    assert error.value.code == "COMPARATOR_FIT_SPLIT_LEAKAGE"


def test_fitted_future_comparators_require_detached_frozen_targets() -> None:
    target = torch.ones(3, 1, 2, requires_grad=True)
    valid = torch.ones(3, 1, dtype=torch.bool)
    with pytest.raises(DataContractError) as error:
        TrainingMeanFuturePredictor.fit(
            target,
            valid,
            target_space_id=TARGET_SPACE,
            split=SplitName.TRAIN,
        )
    assert error.value.code == "COMPARATOR_TARGET_REQUIRES_GRAD"


def _linear_change(elapsed: torch.Tensor, treatment: torch.Tensor) -> torch.Tensor:
    first = 0.4 + 0.03 * elapsed + 0.8 * treatment[:, 0] - 0.2 * treatment[:, 1]
    second = -0.7 + 0.01 * elapsed - 0.5 * treatment[:, 0] + 0.6 * treatment[:, 1]
    return torch.stack((first, second), dim=-1).unsqueeze(1)


def test_time_treatment_predictor_learns_held_out_change_with_source_offset() -> None:
    elapsed = torch.linspace(1.0, 24.0, 24)
    treatment = torch.stack(
        (
            torch.sin(torch.arange(24, dtype=torch.float32)),
            torch.cos(torch.arange(24, dtype=torch.float32) * 0.7),
        ),
        dim=-1,
    )
    source = torch.stack((elapsed * 0.1, -elapsed * 0.2), dim=-1).unsqueeze(1)
    target = source + _linear_change(elapsed, treatment)
    valid = torch.ones(24, 1, dtype=torch.bool)
    predictor = TimeTreatmentFuturePredictor.fit(
        target,
        valid,
        elapsed,
        treatment,
        source=source,
        source_valid=valid,
        source_space_id=TARGET_SPACE,
        target_space_id=TARGET_SPACE,
        condition_schema_id=CONDITION_SCHEMA,
        split=SplitName.TRAIN,
        ridge=1e-8,
    )

    held_out_time = torch.tensor([4.5, 19.5])
    held_out_treatment = torch.tensor([[0.25, -0.4], [-0.6, 0.8]])
    held_out_source = torch.tensor([[[2.0, -1.0]], [[-3.0, 4.0]]])
    prediction = predictor(
        held_out_time,
        held_out_treatment,
        source=held_out_source,
        source_valid=torch.ones(2, 1, dtype=torch.bool),
        source_space_id=TARGET_SPACE,
        target_space_id=TARGET_SPACE,
        condition_schema_id=CONDITION_SCHEMA,
    )

    expected = held_out_source + _linear_change(held_out_time, held_out_treatment)
    torch.testing.assert_close(prediction.mean, expected, atol=2e-5, rtol=2e-5)
    assert prediction.valid.all()
    assert prediction.metadata.uses_source_offset
    assert prediction.metadata.task == "future_observation_representation"
    assert sum(parameter.numel() for parameter in predictor.parameters()) == 0


def test_direct_conditional_predictor_supports_cross_modality_pathology_target() -> None:
    elapsed = torch.linspace(0.0, 11.0, 12)
    treatment = torch.stack((elapsed.remainder(3.0), elapsed.square() / 100.0), dim=-1)
    target = _linear_change(elapsed, treatment)
    valid = torch.ones(12, 1, dtype=torch.bool)
    predictor = TimeTreatmentFuturePredictor.fit(
        target,
        valid,
        elapsed,
        treatment,
        target_space_id="frozen-pathology-global-v1",
        condition_schema_id=CONDITION_SCHEMA,
        split=SplitName.TRAIN,
        ridge=1e-8,
    )

    prediction = predictor(
        elapsed[:3],
        treatment[:3],
        target_space_id="frozen-pathology-global-v1",
        condition_schema_id=CONDITION_SCHEMA,
    )

    torch.testing.assert_close(prediction.mean, target[:3], atol=2e-5, rtol=2e-5)
    assert not prediction.metadata.uses_source_offset
    with pytest.raises(ValueError, match="does not accept a source offset"):
        predictor(
            elapsed[:3],
            treatment[:3],
            source=target[:3],
            source_valid=valid[:3],
            source_space_id="frozen-pathology-global-v1",
            target_space_id="frozen-pathology-global-v1",
            condition_schema_id=CONDITION_SCHEMA,
        )


def test_conditional_predictor_masks_missing_treatment_and_gates_schemas() -> None:
    elapsed = torch.arange(8, dtype=torch.float32)
    treatment = torch.stack((elapsed, elapsed.square()), dim=-1)
    treatment_valid = torch.ones_like(treatment, dtype=torch.bool)
    treatment_valid[:, 1] = False
    treatment[:, 1] = float("nan")
    target = (1.0 + elapsed).reshape(8, 1, 1)
    target_valid = torch.ones(8, 1, dtype=torch.bool)
    predictor = TimeTreatmentFuturePredictor.fit(
        target,
        target_valid,
        elapsed,
        treatment,
        treatment_valid=treatment_valid,
        target_space_id=TARGET_SPACE,
        condition_schema_id=CONDITION_SCHEMA,
        split=SplitName.TRAIN,
    )

    alternate = treatment.clone()
    alternate[:, 1] = 100_000.0
    first = predictor(
        elapsed,
        treatment,
        treatment_valid=treatment_valid,
        target_space_id=TARGET_SPACE,
        condition_schema_id=CONDITION_SCHEMA,
    )
    second = predictor(
        elapsed,
        alternate,
        treatment_valid=treatment_valid,
        target_space_id=TARGET_SPACE,
        condition_schema_id=CONDITION_SCHEMA,
    )
    torch.testing.assert_close(first.mean, second.mean)

    with pytest.raises(DataContractError) as error:
        predictor(
            elapsed,
            treatment,
            treatment_valid=treatment_valid,
            target_space_id=TARGET_SPACE,
            condition_schema_id="different-condition-schema",
        )
    assert error.value.code == "COMPARATOR_FEATURE_SPACE_MISMATCH"
