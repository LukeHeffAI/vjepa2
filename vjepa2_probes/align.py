"""Alignment utilities for latent representations."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import Tensor


def _to_tensor(latents: Tensor | np.ndarray) -> Tensor:
    if isinstance(latents, np.ndarray):
        return torch.from_numpy(latents)
    return latents


def _resample_to_length(latents: Tensor, length: int) -> Tensor:
    if latents.shape[0] == length:
        return latents
    indices = torch.linspace(0, max(latents.shape[0] - 1, 0), steps=length, device=latents.device)
    lower = torch.clamp(indices.floor().long(), 0, latents.shape[0] - 1)
    upper = torch.clamp(lower + 1, 0, latents.shape[0] - 1)
    weight = (indices - lower.float()).view(-1, 1)
    lower_vals = latents.index_select(0, lower)
    upper_vals = latents.index_select(0, upper)
    return lower_vals * (1 - weight) + upper_vals * weight


def align_latents(a: Tensor | np.ndarray, b: Tensor | np.ndarray) -> Tuple[Tensor, Tensor]:
    """Align ``b`` to the length of ``a`` via linear interpolation."""

    a_t = _to_tensor(a).to(torch.float32)
    b_t = _to_tensor(b).to(torch.float32)
    b_aligned = _resample_to_length(b_t, a_t.shape[0])
    return a_t, b_aligned


def _normalize(latents: Tensor) -> Tensor:
    return latents / latents.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def cosine_sim_mean_per_timestep(a: Tensor | np.ndarray, b: Tensor | np.ndarray) -> np.ndarray:
    """Return per-timestep cosine similarity between two latent sequences."""

    a_t, b_t = align_latents(a, b)
    a_norm = _normalize(a_t)
    b_norm = _normalize(b_t)
    cosine = (a_norm * b_norm).sum(dim=-1)
    return cosine.cpu().numpy()


def cosine_sim_global(a: Tensor | np.ndarray, b: Tensor | np.ndarray) -> float:
    """Cosine similarity between mean-pooled latent vectors."""

    a_t, b_t = align_latents(a, b)
    a_mean = _normalize(a_t).mean(dim=0)
    b_mean = _normalize(b_t).mean(dim=0)
    cosine = torch.nn.functional.cosine_similarity(a_mean.unsqueeze(0), b_mean.unsqueeze(0))
    return float(cosine.item())
