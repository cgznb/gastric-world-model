from __future__ import annotations

import pytest
import torch

from stageworld.encoders import (
    CTGeometry,
    EncoderProvenance,
    ObservationTokens,
    PathologyPatchBatch,
    WSIGeometry,
)
from stageworld.errors import DataContractError


def provenance(dim: int = 4) -> EncoderProvenance:
    return EncoderProvenance(
        encoder_name="test_encoder",
        source_version="release-v1",
        component_versions=(("backbone", "weights-v1"),),
        preprocess_version="preprocess-v1",
        feature_dim=dim,
    )


def tokens(**changes: object) -> ObservationTokens:
    values: dict[str, object] = {
        "values": torch.ones(1, 2, 4),
        "valid": torch.tensor([[True, False]]),
        "modality": torch.tensor([[0, 0]]),
        "acquired_time": torch.tensor([[1.0, 0.0]]),
        "available_time": torch.tensor([[2.0, 0.0]]),
        "provenance": provenance(),
        "source_id": (("ct-a", "padding"),),
        "modality_name": "ct",
    }
    values.update(changes)
    return ObservationTokens(**values)  # type: ignore[arg-type]


def test_observation_contract_and_valid_provenance_ids() -> None:
    item = tokens()
    item.validate()
    assert item.modality_name == "ct"
    assert item.provenance_ids() == (("ct-a",),)
    assert item.modality_id is item.modality


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"values": torch.ones(1, 3, 4)}, "INVALID_TENSOR_SHAPE"),
        ({"valid": torch.ones(1, 2)}, "INVALID_VALID_MASK"),
        ({"modality_name": "future"}, "INVALID_MODALITY_NAME"),
        (
            {
                "available_time": torch.tensor([[0.5, 0.0]]),
                "acquired_time": torch.tensor([[1.0, 0.0]]),
            },
            "AVAILABILITY_PRECEDES_ACQUISITION",
        ),
        ({"values": torch.tensor([[[float("nan")] * 4, [1.0] * 4]])}, "NONFINITE_OBSERVATION"),
    ],
)
def test_observation_contract_rejects_invalid_values(changes: dict[str, object], code: str) -> None:
    with pytest.raises(DataContractError) as exc:
        tokens(**changes)
    assert exc.value.code == code


def test_ct_grid_centers_preserve_physical_geometry() -> None:
    geometry = CTGeometry(
        spacing_mm=torch.tensor([[2.0, 3.0, 4.0]]),
        origin_mm=torch.tensor([[10.0, 20.0, 30.0]]),
        direction=torch.eye(3)[None],
        spatial_shape=torch.tensor([[4, 4, 4]]),
    )
    coords = geometry.feature_grid_centers(
        (2, 2, 2), device=torch.device("cpu"), dtype=torch.float32
    )
    assert coords.shape == (1, 8, 3)
    assert torch.allclose(coords[0, 0], torch.tensor([11.0, 21.5, 32.0]))
    assert torch.allclose(coords[0, -1], torch.tensor([15.0, 27.5, 40.0]))


def test_wsi_level0_coordinates_convert_to_physical_centers() -> None:
    geometry = WSIGeometry(
        coords_level0=torch.tensor([[[0.0, 0.0], [100.0, 200.0]]]),
        valid=torch.tensor([[True, True]]),
        mpp=torch.tensor([[0.5, 0.25]]),
        patch_size_level0=torch.tensor([200.0]),
        level0_size=torch.tensor([[1000.0, 1000.0]]),
    )
    expected = torch.tensor([[[0.05, 0.025], [0.10, 0.075]]])
    assert torch.allclose(geometry.physical_centers_mm(), expected)


def test_pathology_feature_coordinate_count_must_match() -> None:
    geometry = WSIGeometry(
        coords_level0=torch.zeros(1, 2, 2),
        valid=torch.ones(1, 2, dtype=torch.bool),
        mpp=torch.ones(1, 2),
        patch_size_level0=torch.tensor([224.0]),
    )
    with pytest.raises(DataContractError) as exc:
        PathologyPatchBatch(
            features=torch.zeros(1, 3, 8),
            geometry=geometry,
            patch_encoder_name="unit",
            patch_encoder_version="unit-v1",
            input_patch_size_px=224,
            magnification=20.0,
            slide_ids=("slide",),
            tissue_sources=("unknown",),
            acquired_time=torch.tensor([1.0]),
            available_time=torch.tensor([2.0]),
        )
    assert exc.value.code == "FEATURE_COORDINATE_COUNT_MISMATCH"
