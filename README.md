# OppaiOracle (MAID)

OppaiOracle, also known as MAID, is a PyTorch system to train, evaluate, and deploy image‑tagging models. It supports configuration‑driven experiments, ONNX export, robust vocabulary handling, and orientation‑aware augmentation.

## Features

- Training: AMP/bfloat16, channels_last, gradient checkpointing, safe checkpointing.
- Dataset loading: Manifest or per‑image JSON sidecars; letterbox resize; optional LMDB L2 cache.
- Orientation‑aware aug: Flip with tag swaps via `orientation_map.json` and safety modes.
- Evaluation: Macro/micro F1 and mAP via TorchMetrics; standalone validation loop.
- ONNX pipeline: Export with preprocessing wrapper, metadata embed/extract, and integrity checks.
- Config system: Single unified YAML for training/validation/inference/export.

## Project Structure

```
.
├── configs/                  # Unified config + orientation map docs
├── logs/                     # Logs and checkpoints
├── scripts/                  # Utilities (e.g., deterministic run)
├── tools/                    # Linting, audits, helpers
├── TEst and review/          # Evaluation & visualization helpers
├── utils/                    # Logging, metrics, ingestion, path utils
├── Configuration_System.py   # Load/validate unified config
├── train_direct.py           # Training entrypoint (AMP, checkpoints, monitoring)
├── validation_loop.py        # Standalone validation/eval pipeline
├── model_architecture.py     # ViT‑based tagger and stability guards
├── dataset_loader.py         # Sidecar/manifest loaders, LMDB L2 cache, orientation
├── Inference_Engine.py       # PyTorch inference with embedded vocab support
├── ONNX_Export.py            # Export with preprocessing + integrity checks
├── onnx_infer.py             # ONNX inference
└── requirements.txt          # Dependencies
```

## Installation

1) Create a Python 3.12 env and install deps
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Notes:
- The requirements pin PyTorch (>=2.8) and torchvision. If you need a specific CUDA build, install the matching torch/torchvision first, then `pip install -r requirements.txt`.
- ONNX GPU runtime is expected (`onnxruntime-gpu`).

## Quick Commands

```bash
# Compile all (sanity)
git ls-files '*.py' | xargs -I {} python -m py_compile "{}"

# Validate config (after installing deps)
python Configuration_System.py validate configs/unified_config.yaml

# Import sanity (catches top-level runtime errors)
python - << 'PY'
import importlib
for m in ['train_direct','validation_loop','Inference_Engine','ONNX_Export','dataset_loader']:
  print('import', m); importlib.import_module(m)
print('OK')
PY

# Dataset preflight (optional)
python Dataset_Analysis.py --help
```

## Configuration Basics

- Storage path: Under `data.storage_locations`, set exactly one `enabled: true` pointing to real data (root containing images + per‑image JSONs, or `train.json`/`val.json` manifests and `images/`).
- Normalization: Set `data.normalize_mean/std`. Validation inherits from the unified config.
- Vocabulary: Use a real `vocabulary.json` (no `tag_###` placeholders). Export fails fast if placeholders are detected.
- Orientation flips: If `data.random_flip_prob > 0`, provide `configs/orientation_map.json` or set `data.strict_orientation_validation: false`. Safety modes: conservative | balanced | permissive.
- LR warmup semantics: `CosineAnnealingWarmupRestarts` is stepped once per EPOCH. Therefore `training.warmup_steps` is in EPOCHS. Recommended: 3–10. The default `10000` keeps LR near min for the whole run — lower it before training.

Edit config at `configs/unified_config.yaml`.

## Training

```bash
python train_direct.py --config configs/unified_config.yaml
```

Tips:
- AMP chooses bfloat16 when supported; model and DataLoader respect `channels_last` when configured.
- The training script prompts to build a vocabulary if missing. Keep it consistent across train/eval/export.
- Determinism: Use `scripts/run_train_deterministic.sh` or set `training.deterministic: true` and a seed; expect some perf trade‑offs.

## Evaluation

- Standalone validation: `validation_loop.py` reads settings from the unified config.
- Batch evaluation and visualization:
```bash
python "TEst and review/batch_evaluate.py" --help
python "TEst and review/live_viewer.py" /path/to/results.jsonl
python "TEst and review/visualize_results.py" --help
```

## ONNX Export & Inference

- Export
```bash
python ONNX_Export.py --config configs/unified_config.yaml
```
- ONNX inference
```bash
python onnx_infer.py --config configs/unified_config.yaml --image /path/to/image.jpg
```
- PyTorch inference
```bash
python Inference_Engine.py --config configs/unified_config.yaml --image /path/to/image.jpg
```

Notes:
- Export embeds preprocessing and validates the vocabulary; it also tries to extract training normalization and tag metadata from the checkpoint.
- For GPU, ensure `onnxruntime-gpu` is installed and matches your CUDA.

## Known Issues & Fixes

- Validation header glitch: If you encounter a stray literal prefix before the shebang in `validation_loop.py`, remove it so the first line is a shebang or docstring. Import‑sanity catches this.
- LR warmup units: Warmup is epoch‑based. Set `training.warmup_steps` to a small integer (e.g., 3–10).
- Background validator lifecycle: `dataset_loader.BackgroundValidator` is daemonized. Training does not explicitly stop it; this is benign for normal runs. If embedding the loader elsewhere, call `validator.stop()` during teardown.
- Validation memory: `validation_loop.py` aggregates predictions/targets to compute metrics. For very large sets, consider chunked/streaming metrics if you extend it.

## Troubleshooting

- No data found: Ensure one `data.storage_locations[*].enabled` is true and the path exists. Sidecar mode scans `*.json` recursively.
- Orientation warnings: Provide `configs/orientation_map.json` or disable flips. Conservative mode blocks flips on unmapped tags.
- Vocabulary errors: Placeholder `tag_###` indicates a corrupted/placeholder vocab; regenerate via `vocabulary.py` helpers or allow the training prompt to build it.
- Import/validation fails due to missing `torch`: Install requirements first (`pip install -r requirements.txt`), or install the appropriate PyTorch wheel for your CUDA.

## Security & Safety

- Checkpoint safety: Code uses `safe_checkpoint.safe_load_checkpoint`. Avoid introducing raw `torch.load` of pickled objects.
- Logging hygiene: Training/inference use a listener queue and stop listeners on exit.
