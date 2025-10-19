"""CLI entry point for probing V-JEPA 2's temporal sensitivity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from . import metrics
from .align import align_latents
from .io import load_video
from .plotting import plot_bars, plot_cosine_series
from .variants import reverse, shuffle_window, time_warp
from .vjepa2_adapter import encode, get_adapter, masked_prediction_loss


def _load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a JSON object")
    return data


def _ensure_types(args: argparse.Namespace) -> None:
    if isinstance(args.video, str):
        args.video = Path(args.video)
    if isinstance(args.output, str):
        args.output = Path(args.output)
    if isinstance(args.config, str):
        args.config = Path(args.config)
    if isinstance(args.save_config, str):
        args.save_config = Path(args.save_config)
    if isinstance(args.shuffle_windows, tuple):
        args.shuffle_windows = list(args.shuffle_windows)
    if isinstance(args.warps, tuple):
        args.warps = list(args.warps)


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON configuration file overriding CLI defaults")
    parser.add_argument("--save_config", type=Path, help="Write the resolved configuration to this JSON file")
    parser.add_argument("--video", type=Path, help="Path to the input video")
    parser.add_argument("--output", type=Path, default=Path("./vjepa2_probe_outputs"), help="Output directory")
    parser.add_argument("--fps", type=int, default=16, help="Target FPS")
    parser.add_argument("--shuffle_windows", type=int, nargs="+", default=[2, 3, 5], help="Shuffle window sizes")
    parser.add_argument("--warps", type=float, nargs="+", default=[0.5, 0.75, 1.5, 2.0], help="Time warp rates")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--compute_loss", action="store_true", help="Attempt to compute masked prediction loss deltas")

    preliminary_args, _ = parser.parse_known_args(argv)
    if preliminary_args.config:
        try:
            config_data = _load_config(preliminary_args.config)
        except ValueError as exc:
            parser.error(str(exc))
        known_options = {action.dest for action in parser._actions}
        unknown_keys = set(config_data) - known_options
        if unknown_keys:
            parser.error(f"Unknown configuration fields: {', '.join(sorted(unknown_keys))}")
        parser.set_defaults(**config_data)

    args = parser.parse_args(argv)
    if args.video is None:
        parser.error("--video must be specified either via CLI or configuration file")

    _ensure_types(args)
    return args


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on runtime
        torch.cuda.manual_seed_all(seed)


def _save_latents(out_dir: Path, name: str, latents: torch.Tensor) -> None:
    path = out_dir / f"latents_{name}.npy"
    np.save(path, latents.cpu().numpy())


def _load_latents(path: Path) -> torch.Tensor:
    return torch.from_numpy(np.load(path))


def build_variants(video: torch.Tensor, shuffle_windows: List[int], warps: List[float], seed: int) -> Dict[str, torch.Tensor]:
    variants: Dict[str, torch.Tensor] = {
        "clean": video,
        "reverse": reverse(video),
    }
    for win in shuffle_windows:
        variants[f"shuffle-{win}"] = shuffle_window(video, win=win, seed=seed)
    for rate in warps:
        variants[f"warp-{rate}"] = time_warp(video, rate=rate)
    return variants


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    _set_seed(args.seed)

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    video, fps = load_video(args.video, target_fps=args.fps)
    variants = build_variants(video, args.shuffle_windows, args.warps, args.seed)

    adapter = get_adapter()

    latents: Dict[str, torch.Tensor] = {}
    for name, clip in tqdm(variants.items(), desc="Encoding variants"):
        latent_path = output_dir / f"latents_{name}.npy"
        if latent_path.exists():
            latents[name] = _load_latents(latent_path)
            continue
        outputs = encode(clip, fps=fps)
        latents[name] = outputs["pooled"]
        _save_latents(output_dir, name, latents[name])

    clean_latents = latents["clean"]
    variant_metrics: Dict[str, Dict[str, object]] = {}
    cosine_series: Dict[str, List[float]] = {}
    cosine_global: Dict[str, float] = {}
    for name, lat in latents.items():
        if name == "clean":
            continue
        aligned_clean, aligned_variant = align_latents(clean_latents, lat)
        rep_metrics = metrics.compute_representation_metrics(aligned_clean, aligned_variant)
        variant_metrics[name] = rep_metrics
        cosine_series[name] = rep_metrics["cosine_t_series"]  # type: ignore[index]
        cosine_global[name] = rep_metrics["cosine_global"]  # type: ignore[index]

    metrics_payload: Dict[str, object] = {
        "video": str(args.video),
        "fps": fps,
        "adapter_backbone": getattr(adapter, "backbone", None),
        "variants": variant_metrics,
    }

    loss_values: Dict[str, float] = {}
    if args.compute_loss:
        try:
            clean_loss = masked_prediction_loss(video, mask_schedule=None)
            loss_values["clean"] = float(clean_loss)
            for name, clip in variants.items():
                if name == "clean":
                    continue
                loss = masked_prediction_loss(clip, mask_schedule=None)
                loss_values[name] = float(loss)
                loss_metrics = metrics.compute_loss_delta(clean_loss, loss)
                metrics_payload["variants"][name].update(loss_metrics)  # type: ignore[index]
        except NotImplementedError:
            metrics_payload["loss_available"] = False
        else:
            metrics_payload["loss_available"] = True

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    resolved_config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(resolved_config, f, indent=2)

    if args.save_config:
        args.save_config.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_config, "w", encoding="utf-8") as handle:
            json.dump(resolved_config, handle, indent=2)

    try:
        if cosine_series:
            plot_cosine_series(cosine_series, output_dir / "cosine_series.png")
        if cosine_global:
            plot_bars(
                cosine_global,
                title="Global cosine similarity",
                ylabel="Cosine",
                outpath=output_dir / "cosine_global_bars.png",
            )
        if args.compute_loss and metrics_payload.get("loss_available"):
            deltas = {name: metrics_payload["variants"][name]["loss_delta"] for name in loss_values if name != "clean"}
            plot_bars(deltas, title="Masked prediction loss delta", ylabel="Δ loss", outpath=output_dir / "loss_delta_bars.png")
    except RuntimeError as exc:
        print(f"Plotting skipped: {exc}")

    readme_lines = [
        "V-JEPA 2 arrow-of-time probe",
        f"Video: {args.video}",
        f"Target FPS: {fps}",
        f"Variants: {', '.join(name for name in variants if name != 'clean')}",
        f"Loss computed: {metrics_payload.get('loss_available', False)}",
    ]
    with open(output_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(readme_lines))

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
