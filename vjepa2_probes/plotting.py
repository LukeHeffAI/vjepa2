"""Plotting helpers for V-JEPA 2 probes."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]
    _MATPLOTLIB_ERROR = RuntimeError("matplotlib is required for plotting; install it to enable figure exports.")
else:
    _MATPLOTLIB_ERROR = None


def _prepare_outpath(outpath: Path | str) -> Path:
    path = Path(outpath)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_cosine_series(series_by_variant: Dict[str, Iterable[float]], outpath: Path | str) -> None:
    """Plot cosine similarity series per variant."""

    if plt is None:
        raise _MATPLOTLIB_ERROR
    outpath = _prepare_outpath(outpath)
    plt.figure(figsize=(10, 4))
    for name, series in series_by_variant.items():
        arr = np.asarray(list(series), dtype=float)
        plt.plot(np.arange(len(arr)), arr, label=name)
    plt.xlabel("Frame index")
    plt.ylabel("Cosine similarity")
    plt.title("Per-timestep cosine similarity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_bars(values_by_variant: Dict[str, float], title: str, ylabel: str, outpath: Path | str) -> None:
    """Plot a bar chart for scalar metrics."""

    if plt is None:
        raise _MATPLOTLIB_ERROR
    outpath = _prepare_outpath(outpath)
    labels = list(values_by_variant.keys())
    values = [float(values_by_variant[label]) for label in labels]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
