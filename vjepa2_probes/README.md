# V-JEPA 2 Temporal Probes

This package contains utilities to measure how sensitive a V-JEPA 2 checkpoint is to the arrow of time.

## Quickstart

```bash
pip install -e .
python -m vjepa2_probes.arrow_of_time \
  --video /path/to/video.mp4 \
  --output ./vjepa2_probe_outputs/sample_run \
  --fps 16 \
  --shuffle_windows 2 3 5 \
  --warps 0.5 0.75 1.5 2.0
```

The command produces JSON metrics, cached latent numpy arrays, and Matplotlib figures summarising the cosine similarities between the clean clip and its temporal variants. Use `--compute_loss` to attempt masked prediction loss deltas (if the adapter exposes the API).

### Configuration files

Instead of passing long argument lists each time, place them in a JSON file and load it via `--config`:

```json
{
  "video": "./samples/sample.mp4",
  "output": "./vjepa2_probe_outputs/sample_run",
  "fps": 16,
  "shuffle_windows": [2, 3, 5],
  "warps": [0.5, 0.75, 1.5, 2.0],
  "seed": 42,
  "compute_loss": true
}
```

```bash
python -m vjepa2_probes.arrow_of_time --config sample_config.json
```

Use `--save_config resolved.json` to persist the resolved arguments after overriding options on the command line.

## Adapter expectations

The default adapter loads the pretrained V-JEPA 2 backbones exposed in `src.hub.backbones`. If you maintain a customised model, override the adapter by calling `vjepa2_probes.vjepa2_adapter.register_adapter` with your own implementation. The CLI only requires the `encode(video_tensor, fps)` method to return a dictionary containing a `"pooled"` tensor of shape `T × D`.

## Outputs

Each run creates an output directory containing:

- `metrics.json`: cosine metrics for every variant.
- `run_config.json`: the CLI arguments used for the run.
- `latents_*.npy`: cached per-frame latents.
- `cosine_series.png` and `cosine_global_bars.png`: Matplotlib summaries.
- `loss_delta_bars.png`: only when `--compute_loss` succeeds.
- `README.txt`: a short log of the run.

The plots are ready for reports; for example:

![Cosine similarity plot](cosine_series.png)
