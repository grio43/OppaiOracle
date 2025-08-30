# Agent Instructions

This document guides AI agents contributing to OppaiOracle.

## Project Overview

OppaiOracle (MAID) is a PyTorch system to train, evaluate, and deploy image‑tagging models. It supports configuration‑driven experiments, ONNX export, robust vocabulary handling, and orientation‑aware augmentation.

## Key Files and Directories

- `configs/unified_config.yaml`: Single source of truth for training/validation/inference/export.
- `train_direct.py`: Training entrypoint with AMP, checkpointing, and monitoring.
- `validation_loop.py`: Standalone validation/evaluation pipeline.
- `evaluation_metrics.py`: Macro/micro F1 and mAP via TorchMetrics.
- `model_architecture.py`: ViT‑based tagger; masking, grad checkpointing, stability guards.
- `dataset_loader.py`: Sidecar/manifest loaders, letterbox, LMDB L2 cache, orientation plumbing.
- `Inference_Engine.py`, `onnx_infer.py`: PyTorch/ONNX inference with embedded vocab support.
- `ONNX_Export.py`: Export with preprocessing wrapper and integrity checks.
- `vocabulary.py`, `model_metadata.py`, `safe_checkpoint.py`: Vocabulary integrity, metadata embed/extract, safe loads.
- `utils/`, `tools/`, `scripts/`, `TEst and review/`: Logging, metrics, calibration, evaluation & visualization.

## Agent Runbook

- Environment: `pip install -r requirements.txt`
- Config validate: `python Configuration_System.py validate configs/unified_config.yaml`
- Compile check: `git ls-files '*.py' | xargs -I {} python -m py_compile "{}"`
- Import sanity (catches top‑level runtime errors):
  - `python - << 'PY'\nimport importlib;\nfor m in [\n 'train_direct','validation_loop','Inference_Engine','ONNX_Export','dataset_loader'\n]:\n  print('import',m); importlib.import_module(m)\nprint('OK')\nPY`
- Dataset preflight (optional): `python Dataset_Analysis.py --help`
- Train: `python train_direct.py --config configs/unified_config.yaml`
- Evaluate: use `validation_loop.py` or `TEst and review/` scripts.
- Export ONNX: `python ONNX_Export.py --config configs/unified_config.yaml`
- ONNX infer: `python onnx_infer.py --config configs/unified_config.yaml --image <path>`

## Quality Gates

- Config basics:
  - **Storage path**: Ensure one `data.storage_locations[*].enabled: true` points to real data.
  - **Normalization**: `data.normalize_mean/std` present; validation inherits from unified config.
  - **Vocabulary**: Use a real `vocabulary.json` (no `tag_###` placeholders). The code fails fast via `verify_vocabulary_integrity`.
  - **Orientation flips**: If `data.random_flip_prob>0`, ensure `data.orientation_map_path` exists or set `data.strict_orientation_validation=false`.
  - **AMP precision**: Training uses `bfloat16` when supported; inference adopts checkpoint precision.

- LR scheduler semantics:
  - Step-based: the trainer now calls `scheduler.step()` after each optimizer update (respects gradient accumulation). `training.warmup_steps` is in optimizer steps.
  - Default: `warmup_steps: 10000`. Tune per dataset if convergence is slow to start.

- Checkpoint safety:
  - Always use `safe_checkpoint.safe_load_checkpoint` (already wired). Do not introduce raw `torch.load` with pickled objects.

- Logging hygiene:
  - Training/inference use a listener queue; ensure listeners are stopped on exit (the code handles this in `train_direct.main()` and ONNX infer).

## Known Issues & Fixes

- Validation header glitch: `validation_loop.py` begins with a stray literal prefix before the shebang when viewed in some repos. Import‑sanity catches this. If present, remove any leading non‑comment text so the first line is a shebang or a docstring.
- LR warmup units: Warmup is step‑based (optimizer updates). Set `training.warmup_steps` to match your desired warmup duration (e.g., ~3–10 epochs worth of updates).
- Background validator lifecycle: `dataset_loader.BackgroundValidator` is daemonized but not explicitly stopped by training. This is benign for normal runs; if you embed loaders elsewhere, call `validator.stop()` during teardown.
  - The trainer now attempts to stop any loader `validator` on exit.
  - Memory in validation: `validation_loop.py` aggregates all predictions/targets to compute metrics. For very large sets consider chunking or streaming metrics if you extend it.

## Coding Standards (for agents)

- Side effects: Keep heavy work behind `if __name__ == '__main__':` and avoid side effects at import time (import sanity depends on this).
- Stability: Prefer stateless functional metrics; clamp logits where needed to avoid non‑finite values (already implemented in `model_architecture.py`).
- Types & errors: Use clear exceptions with actionable messages; avoid silent fallbacks unless logged.
- Security: Keep using `safe_checkpoint`; don’t add generic pickle loads. Keep secrets in `sensitive_config.py`.
- Performance: Use `channels_last` where configured, AMP/bf16 when available, and respect DataLoader knobs (the wrapper auto‑guards `prefetch_factor` with 0 workers).

## Troubleshooting

- No data found: Ensure `data.storage_locations[*].enabled` is set and paths exist. Sidecar mode scans `*.json` recursively.
- Orientation warnings: Provide `configs/orientation_map.json` or disable flips. Conservative mode will block flips on unmapped tags.
- Vocabulary errors: Placeholder `tag_###` indicates a corrupted vocab; regenerate via `vocabulary.py` helpers.
- Determinism: Use `scripts/run_train_deterministic.sh` or set `training.deterministic=true` and seed; expect some perf trade‑offs.

## Quick Commands

- Compile all: `git ls-files '*.py' | xargs -I {} python -m py_compile "{}"`
- Validate config: `python Configuration_System.py validate configs/unified_config.yaml`
- Train: `python train_direct.py --config configs/unified_config.yaml`
- Export ONNX: `python ONNX_Export.py --config configs/unified_config.yaml`
- Infer (PyTorch): `python Inference_Engine.py --config configs/unified_config.yaml --image <img>`
- Infer (ONNX): `python onnx_infer.py --config configs/unified_config.yaml --image <img>`
