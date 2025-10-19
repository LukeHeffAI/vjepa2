"""Video loading helpers for the V-JEPA 2 probes."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

try:  # pragma: no cover - optional dependency
    from decord import VideoReader, cpu
except ImportError:  # pragma: no cover - optional dependency
    VideoReader = None  # type: ignore[assignment]
    cpu = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from torchvision.io import read_video
except ImportError:  # pragma: no cover - optional dependency
    read_video = None  # type: ignore[assignment]


class VideoIOError(RuntimeError):
    """Raised when the video cannot be loaded."""


def _linear_sample(video: Tensor, indices: Tensor) -> Tensor:
    """Sample ``video`` at floating point frame ``indices`` using linear interpolation."""

    t = video.shape[0]
    if t == 1:
        return video.repeat(indices.shape[0], 1, 1, 1)

    indices = indices.to(video.device)
    lower = torch.clamp(indices.floor().long(), 0, t - 1)
    upper = torch.clamp(lower + 1, 0, t - 1)
    weight = (indices - lower.float()).view(-1, 1, 1, 1)
    lower_frames = video.index_select(0, lower)
    upper_frames = video.index_select(0, upper)
    return lower_frames * (1 - weight) + upper_frames * weight


def _center_crop(video: Tensor, crop_size: int) -> Tensor:
    """Apply a center crop to ``video`` with square ``crop_size``."""

    _, _, h, w = video.shape
    if h == crop_size and w == crop_size:
        return video
    top = max((h - crop_size) // 2, 0)
    left = max((w - crop_size) // 2, 0)
    return video[:, :, top : top + crop_size, left : left + crop_size]


def _read_with_decord(path: Path) -> Tuple[Tensor, float]:
    if VideoReader is None:  # pragma: no cover - optional dependency
        raise VideoIOError("Decord is not available")
    reader = VideoReader(str(path), ctx=cpu(0))
    frames = reader.get_batch(range(len(reader)))  # type: ignore[call-arg]
    fps = float(reader.get_avg_fps())
    video = torch.from_numpy(frames.asnumpy())  # T, H, W, C
    return video, fps


def _read_with_torchvision(path: Path) -> Tuple[Tensor, float]:
    if read_video is None:  # pragma: no cover - optional dependency
        raise VideoIOError("torchvision.io.read_video is not available")
    video, _, info = read_video(str(path), pts_unit="sec")
    fps = float(info["video_fps"])
    return video, fps


def _resample_to_length(video: Tensor, length: int) -> Tensor:
    if video.shape[0] == length:
        return video
    indices = torch.linspace(0, max(video.shape[0] - 1, 0), steps=length, device=video.device)
    return _linear_sample(video, indices)


def load_video(
    path: str | Path,
    target_fps: int = 16,
    min_short_side: int = 224,
    crop_size: Optional[int] = 224,
    max_frames: int = 256,
    min_frames: int = 64,
) -> Tuple[Tensor, int]:
    """Load a video, resample temporally, and resize spatially.

    Args:
        path: Path to a video file.
        target_fps: Target frames per second for temporal resampling.
        min_short_side: Shortest spatial side after resize. ``None`` to skip.
        crop_size: Center crop size. ``None`` to skip.
        max_frames: If the resampled clip exceeds this length, take a centered
            temporal crop.
        min_frames: Minimum number of frames. If the clip is shorter, loop the
            video until the threshold is met.

    Returns:
        A tuple ``(video, fps)`` where ``video`` has shape ``T×C×H×W`` with
        floating point values in ``[0, 1]`` and ``fps`` is the effective frame
        rate after resampling.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    loaders = []
    if VideoReader is not None:  # pragma: no cover - optional dependency
        loaders.append(_read_with_decord)
    loaders.append(_read_with_torchvision)

    last_error: Optional[Exception] = None
    raw_video: Optional[Tensor] = None
    fps: Optional[float] = None
    for loader in loaders:
        try:
            raw_video, fps = loader(path)
            break
        except Exception as exc:  # pragma: no cover - intentionally broad
            last_error = exc
            continue

    if raw_video is None or fps is None:
        raise VideoIOError(f"Failed to load video {path}: {last_error}")

    if raw_video.dim() == 4 and raw_video.shape[-1] == 3:
        raw_video = raw_video.permute(0, 3, 1, 2)
    raw_video = raw_video.to(torch.float32) / 255.0

    # Temporal resampling to target FPS.
    duration = raw_video.shape[0] / max(fps, 1e-6)
    target_frames = max(int(round(duration * target_fps)), 1)
    video = _resample_to_length(raw_video, target_frames)

    if video.shape[0] < min_frames:
        repeat = math.ceil(min_frames / video.shape[0])
        video = video.repeat(repeat, 1, 1, 1)
        video = video[:min_frames]

    if video.shape[0] > max_frames:
        start = max((video.shape[0] - max_frames) // 2, 0)
        video = video[start : start + max_frames]

    # Spatial resize.
    if min_short_side is not None:
        _, _, h, w = video.shape
        short_side = min(h, w)
        if short_side != min_short_side:
            scale = min_short_side / float(short_side)
            new_h = int(round(h * scale))
            new_w = int(round(w * scale))
            video = F.interpolate(video, size=(new_h, new_w), mode="bilinear", align_corners=False)

    if crop_size is not None:
        video = _center_crop(video, crop_size)

    return video, target_fps
