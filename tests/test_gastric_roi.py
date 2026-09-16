import socket

import numpy as np
import pytest

from stageworld.data.gastric_roi import offline_network, stomach_field_of_view
from stageworld.errors import ArtifactError, DataContractError


def test_stomach_cube_keeps_full_organ_and_margin_without_anisotropic_distortion():
    mask = np.zeros((100, 100, 120), dtype=np.uint8)
    mask[20:40, 30:50, 15:95] = 1
    affine = np.diag([3.0, -3.0, 3.0, 1.0])
    result, qc = stomach_field_of_view(mask, affine)
    assert qc["field_of_view_mm"] == 280
    assert np.allclose(np.diag(result)[:3], 280 / 96)
    corners = np.array([[19.5, 29.5, 14.5, 1], [39.5, 49.5, 94.5, 1]]) @ affine.T
    lo = result[:3, 3] - 0.5 * np.diag(result)[:3]
    hi = lo + 280
    assert np.all(corners[:, :3].min(0) - lo >= 20 - 1e-5)
    assert np.all(hi - corners[:, :3].max(0) >= 20 - 1e-5)


def test_empty_truncated_and_fragmented_roi_fail_closed():
    mask = np.zeros((64, 64, 64), dtype=np.uint8)
    with pytest.raises(DataContractError) as exc:
        stomach_field_of_view(mask, np.diag([3.0, 3.0, 3.0, 1.0]))
    assert exc.value.code == "ROI_EMPTY"
    mask[:10, 10:30, 10:30] = 1
    with pytest.raises(DataContractError) as exc:
        stomach_field_of_view(mask, np.diag([3.0, 3.0, 3.0, 1.0]))
    assert exc.value.code == "ROI_TRUNCATED"
    mask[:] = 0
    mask[5:15, 5:15, 5:15] = 1
    mask[35:45, 35:45, 35:45] = 1
    with pytest.raises(DataContractError) as exc:
        stomach_field_of_view(mask, np.diag([3.0, 3.0, 3.0, 1.0]))
    assert exc.value.code == "ROI_FRAGMENTED"


def test_offline_inference_rejects_internet_connects():
    with offline_network(), socket.socket() as sock:
        with pytest.raises(ArtifactError) as exc:
            sock.connect(("127.0.0.1", 9))
        assert exc.value.code == "NETWORK_FORBIDDEN"
