"""Test configuration for environments without optional multimedia deps."""

from __future__ import annotations

import sys
import types
from typing import Tuple

import numpy as np
import torch
from torchvision.transforms import functional as tvf


def _resize_numpy(img: np.ndarray, size: Tuple[int, int], interpolation: int) -> np.ndarray:
    tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    mode = tvf.InterpolationMode.BILINEAR if interpolation == 1 else tvf.InterpolationMode.NEAREST
    resized = tvf.resize(tensor, size[::-1], interpolation=mode)
    resized = resized.permute(1, 2, 0).round().clamp(0, 255)
    return resized.to(dtype=torch.uint8).numpy()


if "cv2" not in sys.modules:
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.INTER_LINEAR = 1
    cv2_stub.INTER_NEAREST = 0
    cv2_stub.resize = lambda img, size, interpolation=1: _resize_numpy(np.asarray(img), size, interpolation)
    sys.modules["cv2"] = cv2_stub
