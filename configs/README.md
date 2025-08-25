# OppaiOracle – Configuration Guide

This folder contains all runtime knobs for training, inference, data prep, and exports. Files are plain YAML/JSON so you can commit changes and review diffs.

> **Tip:** If a setting sounds unfamiliar, search this repo for the key name to see where it’s used.

## Files at a glance

- **`training_config.yaml`** – Optimizer, scheduler, checkpoint cadence, distillation, curriculum learning.
- **`inference_config.yaml`** – How a trained model runs: device, preprocessing, thresholds, output format, API server.
- **`augmentation.yaml`** – Train‑time augmentations and orientation/flip policy.
- **`dataset_prep.yaml`** – How we slice and stage the raw dataset for training.
- **`export_config.yaml`** – ONNX/TensorRT export options and output paths.
- **`runtime.yaml`** – Determinism and CUDA/cuDNN runtime toggles.
- **`logging.yaml`** – Log level, file logging, rotation.
- **`paths.yaml`** – Canonical paths used by scripts (vocab, logs, outputs).
- **`vocabulary.yaml`** – Token/vocab hygiene: special tokens and frequency cutoffs.
- **`orientation_map.json`** – Left↔Right mappings and patterns used by flips at train/infer time.

## Common gotchas & invariants

- **Image size vs patch size** – `data.image_size` must be divisible by `data.patch_size`; the model also requires this. If not, training will error or waste pixels.
- **Effective batch size** – Roughly `batch_size * gradient_accumulation_steps * world_size`. Too large can OOM; see training notes in the file.
- **Determinism vs speed** – `runtime.deterministic: true` disables some benchmarks; use only when reproducibility is critical.
- **Normalization & pad color** – Keep `normalize_mean/std` and `pad_color` in inference aligned with training to avoid drops.

## How to override safely

Create a copy of any YAML here and pass its path to the relevant script, or set a small override with an env var or CLI flag if supported. Keep diffs small and focused so they’re easy to review.

## Orientation map docs

See `orientation_map.README.md` in this folder for how flips and symmetric/asymmetric tags are handled.

---

### Changelog

-
- Added human-readable comments across configs
- Added this README and `orientation_map.README.md`
- Migrated `inference_config.yaml` to nested schema (preprocessing/runtime/postprocessing/io/input).
- Fixed `vocabulary.ignore_tags_file` to point at `./Tags_ignore.txt` (repo root).
- Made `export_config.yaml` reference `./artifacts/thresholds.json` to match the calibration tool output.
