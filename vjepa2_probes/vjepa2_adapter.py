"""Adapter that exposes a lightweight V-JEPA 2 inference API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor


def _linear_sample_latents(latents: Tensor, indices: Tensor) -> Tensor:
    length = latents.shape[0]
    if length == 1:
        return latents.repeat(indices.shape[0], 1)
    lower = torch.clamp(indices.floor().long(), 0, length - 1)
    upper = torch.clamp(lower + 1, 0, length - 1)
    weight = (indices - lower.float()).view(-1, 1)
    lower_vals = latents.index_select(0, lower)
    upper_vals = latents.index_select(0, upper)
    return lower_vals * (1 - weight) + upper_vals * weight


@dataclass
class VJEPA2Adapter:
    """Thin wrapper around the V-JEPA 2 encoder and predictor."""

    device: Optional[str] = None
    backbone: Optional[str] = None
    pretrained: bool = True

    def __post_init__(self) -> None:
        self._device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.backbone = self.backbone or os.environ.get("VJEPA2_PROBES_BACKBONE", "vit_large")
        self._encoder, self._predictor = self._load_backbone(self.backbone, pretrained=self.pretrained)
        self._encoder.to(self._device)
        self._encoder.eval()
        if self._predictor is not None:
            self._predictor.to(self._device)
            self._predictor.eval()

    @staticmethod
    def _load_backbone(name: str, *, pretrained: bool) -> tuple[torch.nn.Module, Optional[torch.nn.Module]]:
        try:
            from src.hub import backbones as hub_backbones
        except ImportError as exc:  # pragma: no cover - requires project install
            raise RuntimeError("Unable to import V-JEPA 2 backbones. Ensure the project is installed.") from exc

        attr = f"vjepa2_{name}"
        if not hasattr(hub_backbones, attr):  # pragma: no cover - defensive branch
            raise ValueError(f"Unknown V-JEPA 2 backbone '{name}'.")

        encoder, predictor = getattr(hub_backbones, attr)(pretrained=pretrained)
        return encoder, predictor

    def encode(self, video_tensor: Tensor, fps: int | None = None) -> Dict[str, Tensor]:
        """Return per-frame pooled latents for ``video_tensor``."""

        if video_tensor.dim() != 4:
            raise ValueError("video_tensor must have shape T×C×H×W")

        tubelet = getattr(self._encoder.patch_embed, "tubelet_size", 1)
        usable_frames = (video_tensor.shape[0] // tubelet) * tubelet
        if usable_frames <= 0:
            raise ValueError("Video has fewer frames than the encoder tubelet size.")
        if usable_frames != video_tensor.shape[0]:
            video_tensor = video_tensor[:usable_frames]

        tensor = video_tensor.to(self._device)
        tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0)  # B,C,T,H,W

        with torch.no_grad():
            tokens = self._encoder(tensor)

        tokens = tokens.squeeze(0)  # N×D
        patch_embed = getattr(self._encoder, "patch_embed")
        patch_size = getattr(patch_embed, "patch_size", 16)
        if isinstance(patch_size, tuple):
            patch_h = patch_size[0]
            patch_w = patch_size[1]
        else:
            patch_h = patch_w = int(patch_size)

        spatial_h = video_tensor.shape[2] // patch_h
        spatial_w = video_tensor.shape[3] // patch_w
        patches_per_frame = max(spatial_h * spatial_w, 1)
        frames = tokens.shape[0] // patches_per_frame
        if frames <= 0:
            raise RuntimeError("Failed to infer frame count from encoder tokens.")

        frame_tokens = tokens.reshape(frames, patches_per_frame, -1)
        pooled = frame_tokens.mean(dim=1)
        if tubelet > 1:
            pooled = pooled.repeat_interleave(tubelet, dim=0)
        if pooled.shape[0] != video_tensor.shape[0]:
            indices = torch.linspace(0, max(pooled.shape[0] - 1, 0), steps=video_tensor.shape[0], device=pooled.device)
            pooled = _linear_sample_latents(pooled, indices)

        return {
            "pooled": pooled.detach().cpu(),
            "frame_tokens": frame_tokens.detach().cpu(),
        }

    # The predictive APIs are optional. They intentionally raise informative errors
    # so that callers can handle the missing functionality gracefully.
    def predict(self, *args, **kwargs):  # pragma: no cover - optional
        raise NotImplementedError("Predictive decoding is not implemented in the default adapter.")

    def masked_prediction_loss(self, *args, **kwargs):  # pragma: no cover - optional
        raise NotImplementedError("Masked prediction loss is not exposed by the default adapter.")


_DEFAULT_ADAPTER: Optional[VJEPA2Adapter] = None


def get_adapter() -> VJEPA2Adapter:
    """Return a module-level singleton adapter."""

    global _DEFAULT_ADAPTER
    if _DEFAULT_ADAPTER is None:
        _DEFAULT_ADAPTER = VJEPA2Adapter()
    return _DEFAULT_ADAPTER


def register_adapter(adapter: VJEPA2Adapter) -> None:
    """Override the module-level adapter instance."""

    global _DEFAULT_ADAPTER
    _DEFAULT_ADAPTER = adapter


def encode(video_tensor: Tensor, fps: int | None = None) -> Dict[str, Tensor]:
    """Proxy to :meth:`VJEPA2Adapter.encode`. Useful for monkeypatching in tests."""

    return get_adapter().encode(video_tensor, fps=fps)


def predict(*args, **kwargs):  # pragma: no cover - optional
    return get_adapter().predict(*args, **kwargs)


def masked_prediction_loss(*args, **kwargs):  # pragma: no cover - optional
    return get_adapter().masked_prediction_loss(*args, **kwargs)
