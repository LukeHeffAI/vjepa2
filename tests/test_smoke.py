"""Smoke tests for the V-JEPA 2 probes."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch

from vjepa2_probes.arrow_of_time import main as cli_main
from vjepa2_probes.variants import reverse, shuffle_window, time_warp


def _synthetic_video(num_frames: int = 32, size: int = 64) -> torch.Tensor:
    video = torch.zeros((num_frames, 3, size, size), dtype=torch.float32)
    for i in range(num_frames):
        top = i % (size // 2)
        video[i, :, top : top + 10, top : top + 10] = 1.0
    return video


def _patch_cli(monkeypatch, video: torch.Tensor, fps: int = 16) -> None:
    def fake_load_video(*args, **kwargs):
        return video, fps

    def fake_encode(tensor: torch.Tensor, fps: int):
        rng = np.random.default_rng(tensor.shape[0])
        latents = torch.from_numpy(rng.standard_normal((tensor.shape[0], 8)).astype(np.float32))
        return {"pooled": latents}

    def fake_get_adapter():
        return SimpleNamespace(backbone="test-backbone")

    def fake_masked_prediction_loss(*args, **kwargs):
        raise NotImplementedError

    monkeypatch.setattr("vjepa2_probes.arrow_of_time.load_video", fake_load_video)
    monkeypatch.setattr("vjepa2_probes.arrow_of_time.encode", fake_encode)
    monkeypatch.setattr("vjepa2_probes.arrow_of_time.get_adapter", fake_get_adapter)
    monkeypatch.setattr("vjepa2_probes.arrow_of_time.masked_prediction_loss", fake_masked_prediction_loss)


def test_variants_preserve_shape():
    video = _synthetic_video()
    reversed_video = reverse(video)
    assert reversed_video.shape == video.shape
    shuffled_video = shuffle_window(video, win=3, seed=0)
    assert shuffled_video.shape == video.shape
    warped_video = time_warp(video, rate=1.5)
    assert warped_video.shape == video.shape


def test_cli_smoke(tmp_path, monkeypatch):
    video = _synthetic_video()
    _patch_cli(monkeypatch, video)

    exit_code = cli_main(["--video", "synthetic.mp4", "--output", str(tmp_path), "--fps", "16"])
    assert exit_code == 0

    metrics_path = tmp_path / "metrics.json"
    assert metrics_path.exists()

    with open(metrics_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["adapter_backbone"] == "test-backbone"
    assert "reverse" in payload["variants"]


def test_cli_from_config(tmp_path, monkeypatch):
    video = _synthetic_video()
    _patch_cli(monkeypatch, video)

    config_path = tmp_path / "config.json"
    run_dir = tmp_path / "configured"
    save_config_path = tmp_path / "resolved.json"
    config_payload = {
        "video": "synthetic.mp4",
        "output": str(run_dir),
        "fps": 12,
        "shuffle_windows": [4],
        "warps": [0.5, 2.0],
        "seed": 123,
        "compute_loss": False,
    }
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config_payload, handle)

    exit_code = cli_main(["--config", str(config_path), "--save_config", str(save_config_path)])
    assert exit_code == 0

    with open(save_config_path, "r", encoding="utf-8") as handle:
        saved_config = json.load(handle)
    assert saved_config["video"] == "synthetic.mp4"
    assert saved_config["fps"] == 12
    assert saved_config["shuffle_windows"] == [4]
    assert saved_config["warps"] == [0.5, 2.0]

    with open(run_dir / "metrics.json", "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert set(payload["variants"]).issuperset({"reverse", "shuffle-4", "warp-0.5", "warp-2.0"})
