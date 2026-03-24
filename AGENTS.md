# OppaiOracle Agent Guide

## Project overview
OppaiOracle is a PyTorch-based multi-label image tagging system (anime). This repo
contains training, inference, evaluation, and export tooling.

## Repo map (start here)
- `train_direct.py`: main training entrypoint.
- `Start_AI_Training.ps1`: PowerShell wrapper that loads `payton_env.ps1` and launches training.
- `Configuration_System.py`: config schemas, validation, and env/CLI overrides.
- `configs/unified_config.yaml`: single source of truth for model, data, training, inference, export.
- `dataset_loader.py`: data pipeline for images + sidecar JSON.
- `vocabulary.py` / `vocabulary.json`: tag vocabulary (generate via code, avoid manual edits).
- `orientation_handler.py` + `configs/orientation_map.json`: left/right tag mapping for flips.
- `model_architecture.py`: ViT/Swin model definitions.
- `training_utils.py`, `loss_functions.py`, `evaluation_metrics.py`: training logic and metrics.
- `Monitor_log.py`: logging and TensorBoard integration.
- `Inference_Engine.py`: PyTorch inference workflow.
- `ONNX_Export.py` / `onnx_infer.py`: ONNX export and inference.
- `validation_loop.py`: standalone evaluation utilities.
- `TRAINING_PIPELINE_MAP.md`: training pipeline overview.

## Common commands
- Validate config:
  `python Configuration_System.py validate configs/unified_config.yaml`
- Train (preferred on Windows):
  `.\Start_AI_Training.ps1`
- Train (direct):
  `python train_direct.py --config configs/unified_config.yaml`
- Inference:
  `python Inference_Engine.py --model ./checkpoints/best_model.pt --config ./checkpoints/model_config.json --vocab vocabulary.json`
- Export ONNX:
  `python ONNX_Export.py <checkpoint> --output ./artifacts/model.onnx`
- TensorBoard:
  `.\Start_TensorBoard.ps1 -LogDir .\tensorboard -Port 6006`

## Config rules and invariants
- `configs/unified_config.yaml` is canonical; keep it in sync with `Configuration_System.py`.
- Env overrides use the `ANIME_TAGGER_` prefix with nested keys (see `Configuration_System.py`).
- `data.image_size` must be divisible by `data.patch_size`.
- Keep training and inference normalization (mean/std) and pad color aligned.
- Effective batch size is `batch_size * gradient_accumulation_steps * world_size`.
- Orientation flips are governed by `configs/orientation_map.json` and `data.orientation_safety_mode`.

## Data and vocabulary
- Datasets use sidecar JSON; `dataset_loader.py` expects that layout.
- `vocabulary.json` is generated; update via `vocabulary.py:create_vocabulary_from_datasets`.
- When adding directional tags, update `configs/orientation_map.json` and `configs/orientation_map.README.md`.

## Environment and secrets
- Python 3.12+ (see `pyproject.toml`).
- Dependencies live in `requirements.txt`.
- Copy `sensitive_config.py.example` to `sensitive_config.py` for local secrets; do not commit secrets.

## Generated artifacts (do not edit/commit)
- `experiments/`, `tensorboard/`, `logs/`, `analysis_sample_cache/`, `analysis_sample_output/`
- `l2_cache/`, `image_review/`, `__pycache__/`, `.ruff_cache/`
- Large analysis outputs like `analysis_sample_paths.txt` and `orphan_files_list.txt`

## Deprecations
- Legacy Lightning entrypoints were removed; use `train_direct.py`.
- Legacy safe-checkpoint wrapper was removed; use `training_utils.CheckpointManager`.

## Testing and validation
- No formal test suite; rely on config validation and small smoke runs for training/inference changes.
