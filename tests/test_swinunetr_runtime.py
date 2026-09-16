from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from torch import nn

from stageworld.encoders import (
    LegacySwinUNETRPatchMerging,
    load_swinunetr_ssl_backbone,
    map_swinunetr_ssl_state_dict,
)
from stageworld.errors import ArtifactError

_TASK_HEAD_KEYS = (
    "contrastive_head.bias",
    "contrastive_head.weight",
    "convTrans3d.bias",
    "convTrans3d.weight",
    "norm.bias",
    "norm.weight",
    "rotation_head.bias",
    "rotation_head.weight",
)


def _release_payload() -> dict[str, object]:
    state = {f"module.{name}": torch.zeros(1) for name in _TASK_HEAD_KEYS}
    state.update(
        {
            "module.layers1.0.blocks.0.mlp.fc1.weight": torch.ones(2, 1),
            "module.layers1.0.blocks.0.mlp.fc2.weight": torch.ones(1, 2),
        }
    )
    return {"state_dict": state}


def test_swinunetr_release_key_mapping_is_explicit_and_exact() -> None:
    mapped, excluded = map_swinunetr_ssl_state_dict(_release_payload())

    assert excluded == _TASK_HEAD_KEYS
    assert set(mapped) == {
        "layers1.0.blocks.0.mlp.linear1.weight",
        "layers1.0.blocks.0.mlp.linear2.weight",
    }

    payload = _release_payload()
    state = payload["state_dict"]
    assert isinstance(state, dict)
    state.pop("module.rotation_head.bias")
    with pytest.raises(ArtifactError) as error:
        map_swinunetr_ssl_state_dict(payload)
    assert error.value.code == "SWINUNETR_TASK_HEAD_CONTRACT_MISMATCH"


def test_legacy_patch_merging_preserves_released_three_dimensional_order() -> None:
    merging = LegacySwinUNETRPatchMerging(dim=1, spatial_dims=3)
    merging.norm = nn.Identity()
    merging.reduction = nn.Identity()
    volume = torch.arange(8.0).reshape(1, 2, 2, 2, 1)

    merged = merging(volume)

    assert merged.shape == (1, 1, 1, 1, 8)
    assert torch.equal(merged.flatten(), torch.tensor([0.0, 4.0, 2.0, 1.0, 5.0, 2.0, 1.0, 7.0]))


@pytest.mark.requires_weights
def test_local_swinunetr_release_loads_all_backbone_tensors() -> None:
    configured = os.environ.get("STAGEWORLD_SWINUNETR_WEIGHT")
    if not configured or not Path(configured).is_file():
        pytest.skip("STAGEWORLD_SWINUNETR_WEIGHT does not name an available local checkpoint")

    result = load_swinunetr_ssl_backbone(configured, device="cpu")

    assert len(result.state_dict_report.loaded_keys) == 126
    assert result.state_dict_report.loaded_parameter_fraction == 1.0
    assert result.state_dict_report.missing_keys == ()
    assert result.state_dict_report.unexpected_keys == ()
    assert result.state_dict_report.shape_mismatches == ()
    assert len(result.excluded_task_head_keys) == 8
    assert not any(parameter.requires_grad for parameter in result.backend.parameters())
    assert not result.backend.training
