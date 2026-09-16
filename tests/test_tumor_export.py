from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from stageworld.data.tumor_export import (
    ParallelProbabilityConfiguration,
    probability_resampling_workers,
    resample_probability_channels,
)
from stageworld.errors import ArtifactError


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("as_tensor", [False, True])
@pytest.mark.parametrize("spacing,target_spacing,target,extra", [
    ([5, 1, 1], [3, 1, 1], [9, 17, 19], {}),
    ([1, 5, 1], [1, 3, 1], [11, 7, 19], {}),
    ([1, 1, 5], [1, 1, 3], [11, 17, 7], {}),
    ([2.5, 1.5, 1.5], [5, 0.7, 0.7], [3, 17, 19], {}),
    ([1, 1, 1], [1.5, 1.5, 1.5], [9, 17, 19], {}),
    ([5, 1, 1], [5, 1, 1], [5, 7, 9], {}),
    ([5, 1, 1], [2.5, 1, 1], [10, 7, 9], {}),
    ([5, 1, 1], [2.5, 1, 1], [10, 11, 13], {"force_separate_z": False}),
    ([5, 1, 1], [2.5, 1, 1], [10, 11, 13], {"order_z": 1}),
    ([1, 5, 5], [1, 2.5, 2.5], [7, 11, 13], {}),
])
def test_probability_export_matches_upstream_exactly(dtype, as_tensor, spacing,
                                                     target_spacing, target, extra):
    upstream = pytest.importorskip("nnunetv2.preprocessing.resampling.default_resampling")
    kwargs = {"is_seg": False, "order": 1, "order_z": 0, "force_separate_z": None, **extra}
    original = partial(upstream.resample_data_or_seg_to_shape, **kwargs)
    configuration = SimpleNamespace(
        configuration={"resampling_fn_probabilities": "resample_data_or_seg_to_shape",
                       "resampling_fn_probabilities_kwargs": kwargs},
        resampling_fn_probabilities=original, spacing=spacing,
    )
    values = np.random.default_rng(18).normal(size=(5, 5, 7, 9)).astype(dtype)
    saved = values.copy()
    data = torch.from_numpy(values) if as_tensor else values
    expected = original(data, target, spacing, target_spacing)
    for workers in (1, 4):
        adapter = ParallelProbabilityConfiguration(configuration, max_workers=workers)
        actual = adapter.resampling_fn_probabilities(data, target, spacing, target_spacing)
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected, strict=True)
        assert adapter.spacing == spacing
    np.testing.assert_array_equal(values, saved, strict=True)


def test_probability_workers_reduce_for_large_coordinate_grids():
    assert probability_resampling_workers((15, 10, 10, 10), (20, 20, 20)) == 4
    assert probability_resampling_workers((15, 50, 256, 256), (500, 512, 512)) == 1
    assert probability_resampling_workers((1, 10, 10, 10), (20, 20, 20)) == 1
    with pytest.raises(ValueError):
        probability_resampling_workers((15, 10, 10, 10), (20, 20, 20), max_workers=0)


def test_probability_adapter_rejects_unverified_resampler():
    with pytest.raises(ArtifactError) as error:
        ParallelProbabilityConfiguration(SimpleNamespace(
            configuration={"resampling_fn_probabilities": "unknown"},
        ))
    assert error.value.code == "TUMOR_RESAMPLER_UNSUPPORTED"


@pytest.mark.parametrize("max_workers,budget", [(1, 6 * 1024**3), (4, 1)])
def test_single_worker_keeps_resampler_temporary_arrays_channel_bounded(max_workers, budget):
    values = np.random.default_rng(23).normal(size=(15, 5, 7, 9)).astype(np.float32)
    calls = []

    def resample(data, new_shape, current_spacing, new_spacing):
        assert data.shape[0] == 1
        calls.append(data.shape[0])
        return data.copy()

    actual = resample_probability_channels(
        values, values.shape[1:], [1, 1, 1], [1, 1, 1], resample=resample,
        max_workers=max_workers, temporary_budget_bytes=budget,
    )
    np.testing.assert_array_equal(actual, values, strict=True)
    assert len(calls) == values.shape[0]
