"""Video variants for probing temporal sensitivity."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch import Tensor

from .io import _linear_sample, _resample_to_length


def reverse(video: Tensor) -> Tensor:
    """Return the time-reversed video."""

    return video.flip(0)


def shuffle_window(video: Tensor, win: int, seed: int) -> Tensor:
    """Shuffle frames within non-overlapping windows of size ``win``."""

    if win <= 1:
        return video.clone()

    rng = np.random.default_rng(seed)
    frames = []
    for start in range(0, video.shape[0], win):
        chunk = video[start : start + win]
        order = rng.permutation(chunk.shape[0])
        frames.append(chunk[order])
    return torch.cat(frames, dim=0)


def time_warp(video: Tensor, rate: float, seed: int | None = None) -> Tensor:
    """Uniformly warp video speed by ``rate`` and resample back to original length."""

    if rate <= 0:
        raise ValueError("rate must be positive")

    t = video.shape[0]
    warped_len = max(int(round(t / rate)), 1)
    base_indices = torch.linspace(0, max(t - 1, 0), steps=warped_len, device=video.device)
    warped = _linear_sample(video, base_indices)
    return _resample_to_length(warped, t)


VARIANT_BUILDERS: Dict[str, callable] = {
    "reverse": reverse,
}
