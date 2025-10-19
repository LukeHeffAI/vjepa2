"""Metrics helpers for V-JEPA 2 probes."""

from __future__ import annotations

from typing import Dict

from .align import cosine_sim_global, cosine_sim_mean_per_timestep


def compute_representation_metrics(clean_lat, variant_lat) -> Dict[str, object]:
    """Compute cosine similarity based metrics between clean and variant latents."""

    series = cosine_sim_mean_per_timestep(clean_lat, variant_lat)
    metrics = {
        "cosine_t_mean": float(series.mean()),
        "cosine_t_series": series.astype(float).tolist(),
        "cosine_global": float(cosine_sim_global(clean_lat, variant_lat)),
    }
    return metrics


def compute_loss_delta(clean_loss: float, variant_loss: float) -> Dict[str, float]:
    """Return the delta between clean and variant masked prediction losses."""

    return {
        "loss_delta": float(variant_loss - clean_loss),
        "clean_loss": float(clean_loss),
        "variant_loss": float(variant_loss),
    }
