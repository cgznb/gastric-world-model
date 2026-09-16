"""Memory-bounded, equivalent evaluation of nnU-Net probability resampling."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from functools import partial
from typing import Any

import numpy as np
import torch

from stageworld.errors import ArtifactError


def resample_separate_nearest_axis(
    data: np.ndarray | torch.Tensor, new_shape: Sequence[int],
    current_spacing: Sequence[float], new_spacing: Sequence[float], *,
    resample: Callable[..., np.ndarray], resample_kwargs: dict[str, Any],
) -> np.ndarray:
    """Replace the separable nearest-axis 3D coordinate grid with identical 1D indices."""
    from nnunetv2.configuration import ANISO_THRESHOLD  # type: ignore[import-untyped]
    from nnunetv2.preprocessing.resampling.default_resampling import (  # type: ignore[import-untyped]
        determine_do_sep_z_and_axis,
    )
    from scipy.ndimage import map_coordinates  # type: ignore[import-untyped]

    separate, axis = determine_do_sep_z_and_axis(
        resample_kwargs.get("force_separate_z", False), current_spacing, new_spacing,
        resample_kwargs.get("separate_z_anisotropy_threshold", ANISO_THRESHOLD),
    )
    if not separate or axis is None or resample_kwargs.get("order_z", 0) != 0:
        return resample(data, new_shape, current_spacing, new_spacing)
    intermediate_shape = list(new_shape)
    source_size = int(data.shape[axis + 1])
    target_size = int(new_shape[axis])
    intermediate_shape[axis] = source_size
    intermediate = resample(data, intermediate_shape, current_spacing, new_spacing)
    if source_size == target_size:
        return intermediate
    # The other two axes have unit scale in upstream's final order-0 mapping.
    coordinates = (source_size / target_size) * (np.arange(target_size) + 0.5) - 0.5
    indices = map_coordinates(
        np.arange(source_size, dtype=np.int64), coordinates[None], order=0, mode="nearest",
    )
    return np.take(intermediate, indices, axis=axis + 1)


def probability_resampling_workers(
    source_shape: Sequence[int], target_shape: Sequence[int], *,
    max_workers: int = 4, temporary_budget_bytes: int = 6 * 1024**3,
) -> int:
    """Account conservatively for nnU-Net's float64 interpolation coordinate grids."""
    if max_workers < 1 or temporary_budget_bytes < 1:
        raise ValueError("Resampling workers and temporary budget must be positive")
    if len(source_shape) != 4 or len(target_shape) != 3:
        raise ValueError("Require CXYZ input and XYZ target shapes")
    if min(*source_shape, *target_shape) < 1:
        raise ValueError("Resampling dimensions must be positive")
    worker_bytes = 80 * math.prod(target_shape) + 8 * math.prod(source_shape[1:])
    return max(1, min(int(source_shape[0]), max_workers, temporary_budget_bytes // worker_bytes))


def resample_probability_channels(
    data: np.ndarray | torch.Tensor, new_shape: Sequence[int],
    current_spacing: Sequence[float], new_spacing: Sequence[float], *,
    resample: Callable[..., np.ndarray], max_workers: int = 4,
    temporary_budget_bytes: int = 6 * 1024**3,
) -> np.ndarray:
    """Preserve per-channel interpolation, dtype and ordering with bounded pending work."""
    workers = probability_resampling_workers(
        data.shape, new_shape, max_workers=max_workers,
        temporary_budget_bytes=temporary_budget_bytes,
    )
    result: np.ndarray | None = None
    channels = int(data.shape[0])

    def process(index: int) -> np.ndarray:
        return resample(data[index:index + 1], new_shape, current_spacing, new_spacing)

    # Limit both executing and completed-but-unconsumed channel arrays.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(process, index): index for index in range(workers)}
        next_index = workers
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                index = pending.pop(future)
                channel = future.result()
                if result is None:
                    result = np.empty((channels, *channel.shape[1:]), dtype=channel.dtype)
                result[index:index + 1] = channel
                if next_index < channels:
                    pending[pool.submit(process, next_index)] = next_index
                    next_index += 1
    if result is None:
        raise RuntimeError("No resampling channels completed")
    return result


class ParallelProbabilityConfiguration:
    """Delegate nnU-Net settings, replacing only its independent probability channels."""

    def __init__(self, base: Any, *, max_workers: int = 4,
                 temporary_budget_bytes: int = 6 * 1024**3) -> None:
        if base.configuration["resampling_fn_probabilities"] != "resample_data_or_seg_to_shape":
            raise ArtifactError(
                code="TUMOR_RESAMPLER_UNSUPPORTED",
                message="Channel parallelism requires the verified nnU-Net resampler.",
            )
        self._base = base
        resample = partial(
            resample_separate_nearest_axis, resample=base.resampling_fn_probabilities,
            resample_kwargs=base.configuration["resampling_fn_probabilities_kwargs"],
        )
        self._resample = partial(
            resample_probability_channels, resample=resample,
            max_workers=max_workers, temporary_budget_bytes=temporary_budget_bytes,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    @property
    def resampling_fn_probabilities(self) -> Callable[..., np.ndarray]:
        return self._resample
