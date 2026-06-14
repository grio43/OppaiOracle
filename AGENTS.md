# AGENTS.md — OppaiOracle

Guide for AI coding agents working in this repo. Keep it accurate; if you change a subsystem, update the relevant section here.

## Project overview

OppaiOracle is a **PyTorch multi-label anime image tagger** (Vision Transformer, trained from scratch — not ImageNet-pretrained). It predicts ~19K tags (target 18–24K) plus rating tags (rating tags now live inside the tag vocabulary, there is no separate rating head).

The model is trained with a **two-stage progressive-resolution plan**:

- **Phase 1** — 320×320, ~40-epoch from-scratch run.
- **Phase 2 (active)** — 448×448 fine-tune, target 15 epochs. Position embeddings are bicubically interpolated at the resolution switch; optimizer/scheduler state is reset.

Architecture: ViT-L/16, 18 transformer layers, hidden_size 1024, 16 heads, mlp_dim 4096 (~228M backbone, ~250M total). Training is procedural (no Trainer/Lightning class) via [train_direct.py](train_direct.py).

Design rationale and live status live in [todos/progressive-training-plan.md](todos/progressive-training-plan.md), [TRAINING_PIPELINE_MAP.md](TRAINING_PIPELINE_MAP.md), and [TRAINING_HEALTH_TRACKER.md](TRAINING_HEALTH_TRACKER.md).

## Repo map

### Config

- [configs/unified_config.yaml](configs/unified_config.yaml) — **canonical** config (model/data/training/inference/threshold_calibration/export/validation/monitor/debug). Values here override dataclass defaults.
- [Configuration_System.py](Configuration_System.py) — dataclass config schema + loader. `FullConfig` aggregates `ModelConfig`, `DataConfig`, `GradientClippingConfig`, `LossConfig`, `TrainingConfig`, `InferenceConfig`, `ExportConfig`, `ValidationConfig`, `ThresholdCalibrationConfig`, `MonitorConfig`, `DebugConfig`, etc. Provides env-override (`update_from_env`), a minimal CLI parser (`create_config_parser`: only `--config` and `--validate-only` — **configuration is YAML-only; there are no per-field CLI overrides**), `validate`/`generate` subcommands.
- [training_config.py](training_config.py) — single live helper: `scale_learning_rate` (batch-size LR scaling used by train_direct). The old dataset-aware "auto-config" helpers (weight-decay scaling, warmup/beta2 computation, batch-size/scheduler recommendation) were removed — weight decay is fixed at `training.weight_decay` and the scheduler is exactly `training.scheduler`.
- [schemas.py](schemas.py) — prediction output schemas (`TagPrediction`, `ImagePrediction`, `RunMetadata`, `PredictionOutput`) plus `canonical_vocab_bytes` / `compute_vocab_sha256`. Not a config module.

### Training

- [train_direct.py](train_direct.py) — **main training entrypoint** (procedural, ~2.5K LOC). `main()` parses args, loads `FullConfig`, dispatches validation-only or training. Core loop is `train_with_orientation_tracking(config)` (name retained for continuity; flip logic is inlined in the dataset). Handles vocab auto-build, dataloaders, ViT setup, AMP (bfloat16), gradient accumulation, checkpointing, metrics (F1-macro, mAP), early stopping, soft-stop, `torch.compile`, NaN/Inf detection.
- [Start_AI_Training.ps1](Start_AI_Training.ps1) — Windows launcher. Sources `payton_env.ps1`, sets VS Build Tools + Windows SDK paths for `torch.compile`, then runs `train_direct.py`. Args: `-ConfigPath`, `-TrainingArgs`, `-KeepOpenOnError`, `-KeepOpen`.
- [training_utils.py](training_utils.py) — training core: `TrainingState`, `CosineAnnealingWarmupRestarts`, `EarlyStopping`, `AsyncCheckpointWriter`, `CheckpointManager` (canonical checkpoint lifecycle), `TrainingUtils` (seed/optimizer/param-group/scheduler helpers), `validate_config_compatibility`, `detect_architecture_from_state_dict`.
- [schedulers.py](schedulers.py) — `LinearWarmupCosineLR` (warmup then cosine anneal).
- [adan_optimizer.py](adan_optimizer.py) — Adan optimizer (arXiv:2208.06677); single-/multi-tensor/CUDA-fused variants.
- [custom_drop_path.py](custom_drop_path.py) — `SafeDropPath` stochastic-depth layer used in the ViT.

### Data & vocabulary

- [dataset_loader.py](dataset_loader.py) — primary data pipeline. `SidecarJsonDataset` (on-the-fly JSON+image loading, Arrow IPC metadata cache, 95/5 auto split, augmentations, letterbox/padding-mask, RGB/BGR ordering) and legacy `DatasetLoader` (manifest mode). Owns inlined flip logic: `_deterministic_coin()` (CRC32 epoch-aware), `_decide_flip_mode()`, `set_epoch()`; `create_dataloaders()` factory.
- [vocabulary.py](vocabulary.py) — `TagVocabulary` + `create_vocabulary_from_datasets(...)`: scans sidecar JSONs, counts tag frequencies (excluding `Tags_ignore.txt`), writes [vocabulary.json](vocabulary.json). CLI: `python vocabulary.py <dataset_root>+`.
- [shared_vocabulary.py](shared_vocabulary.py) — `SharedVocabularyManager`: shares vocab across DataLoader workers via `shared_memory` to cut spawn overhead.
- [mask_utils.py](mask_utils.py) — padding-mask helpers: `ensure_pixel_padding_mask` (B,1,H,W bool, True=PAD), `pixel_to_token_ignore`.
- [vocabulary_utils/vocab_utils.py](vocabulary_utils/vocab_utils.py) — `load_vocab`/`save_vocab`/`compute_vocab_hash`/`diff_vocab` (accept list or dict).
- [vocabulary_utils/vocab_append.py](vocabulary_utils/vocab_append.py) — append-only vocab updater; preserves existing indices, appends new tags, keeps `<PAD>=0`/`<UNK>=1`.
- [vocabulary.json](vocabulary.json) — generated vocabulary. A JSON object with three sections: `tag_to_index`, `index_to_tag`, and `tag_frequencies` (the `tag_to_index` sub-map has ~19K entries, `<PAD>=0`, `<UNK>=1`). It is **not** a flat dict. **Do not hand-edit.**
- [Tags_ignore.txt](Tags_ignore.txt) — exclusion list for vocab generation (one tag per line; 78 entries).
- [selected_tags.csv](selected_tags.csv) — `(tag_id, name, category)` export for tagger-UI compatibility; vocabulary.json is canonical.

### Model / loss / metrics

- [model_architecture.py](model_architecture.py) — `BaseTagger` (abstract), `SimplifiedTagger` (**ViT**, PyTorch 2.5+ Flex Attention with `scaled_dot_product_attention` ONNX fallback), `VisionTransformerConfig` (its config dataclass), `create_model(config, architecture_type='vit')`. Returns `{'tag_logits': ..., 'logits': ...}`.
- [loss_functions.py](loss_functions.py) — `AsymmetricFocalLoss` (multi-label ASL; `gamma_pos`, `gamma_neg`, `alpha`, `clip`, `label_smoothing`, `ignore_indices=[0]`, per-class weights; log-space math for stability) and `MultiTaskLoss` wrapper (tag loss only).
- [evaluation_metrics.py](evaluation_metrics.py) — `MetricComputer` (F1 macro/micro, mAP; default threshold **0.2653**; `skip_indices=[0]`), `FrequencyBucketMetrics` (LVIS-style per-frequency buckets), `ThresholdCalibrator` (per-tag/per-bucket threshold search).

### Inference & export

- [Inference_Engine.py](Inference_Engine.py) — PyTorch inference CLI/engine. `InferenceConfig`, `ImagePreprocessor` (letterbox, normalize, transparency), `ModelWrapper`, `InferenceEngine` (single/batch, caching, monitoring, TTA flips, padding masks). Reads embedded-or-external vocab and preprocessing from the checkpoint. CLI: `--model`, `--config`, `--vocab`.
- [ONNX_Export.py](ONNX_Export.py) — `ONNXExporter`: exports the ViT model to ONNX with variants (`full`/`fp16`/`quantized`), graph optimization, dynamic batch, embedded metadata, `selected_tags.csv` export. Preprocessing is **external** (not baked into the graph); output includes sigmoid.
- [onnx_infer.py](onnx_infer.py) — ONNXRuntime inference CLI. Handles new (external preprocessing) and legacy (baked) models; EXIF transpose, transparency, channel order.
- [model_metadata.py](model_metadata.py) — `ModelMetadata`: embed/extract vocabulary (gzip+base64, SHA256-verified) and preprocessing params (`normalize_mean`/`std`, `image_size`, `patch_size`, `color_order`) in checkpoints/ONNX, with legacy fallback.

### Tooling (`tools/`, scripts)

- [validation_loop.py](validation_loop.py) — standalone eval CLI (no compile/scheduler/train loader). Args: `--checkpoint`/`--model`, `--data-dir`, `--json-dir`, `--vocab-path`, `--mode` (full/fast/tags/hierarchical), `--output-dir`, `--save-predictions`, `--create-plots`, `--no-amp`, `--device`. Batch size/workers, `max_samples`, and the prediction threshold come from unified_config.yaml (`validation.dataloader.*`, `validation.max_samples`, `inference.prediction_threshold`) — not CLI flags.
- [Monitor_log.py](Monitor_log.py) — `MetricMonitor`, TensorBoard integration, webhook alerts (via optional `sensitive_config.py`), psutil system monitoring; used by training and inference.
- [test_flip_pipeline.py](test_flip_pipeline.py) — integration test for horizontal-flip augmentation in DataLoader workers (distribution/determinism, pickle round-trip, end-to-end pixel correctness).
- [tools/prepare_phase2.py](tools/prepare_phase2.py) — Phase 1→2 checkpoint conversion (`--image-size` required; resets optimizer/scheduler/scaler/epoch/step; updates image_size across config; pos-embed interpolation happens on load).
- [tools/find_pr_threshold.py](tools/find_pr_threshold.py) — find Precision=Recall thresholds (micro/macro + per-tag), CSV/JSON output.
- [tools/run_validation_for_epoch.py](tools/run_validation_for_epoch.py) — replicate exact train-loop validation for a checkpoint; emits a `TRAINING_HEALTH_TRACKER` row.
- [tools/diagnose_f1.py](tools/diagnose_f1.py) — F1 pipeline diagnosis (logit/sigmoid/threshold, sklearn cross-check, optimal threshold).
- [tools/bench_precision.py](tools/bench_precision.py) — FP32/FP16/FP8 ONNX latency + throughput benchmark on CUDA.
- [tools/export_fp8.py](tools/export_fp8.py) / [tools/export_fp8_weightonly.py](tools/export_fp8_weightonly.py) — FP8 (E4M3) static / weight-only quantization; re-attach vocab+metadata afterward.
- [tools/validate_fp8.py](tools/validate_fp8.py) / [tools/eval_fp8_map.py](tools/eval_fp8_map.py) — FP8 structure validation / FP8-vs-FP32 mAP drift.
- [tools/release_v1_1.py](tools/release_v1_1.py) — build V1.1 safetensors (bfloat16) + `selected_tags.csv` into `huggingface_release/`; strips `_orig_mod`/DDP/compile prefixes.
- [tools/restamp_vocab_sha.py](tools/restamp_vocab_sha.py) — re-stamp `vocab_sha256` in checkpoints from embedded `vocab_b64_gzip`.
- [tools/corrections_report.py](tools/corrections_report.py) / [tools/corrections_report_md.py](tools/corrections_report_md.py) — analyze `image_review/corrections.json` (hardcoded paths).
- [utils/](utils/) — internal support package imported across the codebase: `exclusion_manager.py` (bad/corrupted-image tracking), `metadata_cache.py` / `metadata_ingestion.py` (Arrow metadata cache), `logging_setup.py` / `logging_sanitize.py`, `memory_monitor.py`, `file_handlers.py`, `path_utils.py`.
- PowerShell ops: [Start_TensorBoard.ps1](Start_TensorBoard.ps1), [Start_ImageReview.ps1](Start_ImageReview.ps1), [Start_ImageReview_Remote.ps1](Start_ImageReview_Remote.ps1) (Cloudflare tunnel; needs `cloudflared`).

### Docs & state

- [TRAINING_PIPELINE_MAP.md](TRAINING_PIPELINE_MAP.md) — current, accurate map of the training pipeline.
- [TRAINING_HEALTH_TRACKER.md](TRAINING_HEALTH_TRACKER.md) — live Phase 2 runbook: per-epoch validation logs and canary checks.
- [todos/progressive-training-plan.md](todos/progressive-training-plan.md) — research-grounded two-phase design.
- [deprecated_candiates.md](deprecated_candiates.md) — review list of possibly-unused helpers (not a removal manifest).

## Common commands

This project uses the venv at `L:\Dab\payton_env` (set up via [payton_env.ps1](payton_env.ps1)). PowerShell scripts source it automatically. When calling Python directly, prefer the venv interpreter (`L:\Dab\payton_env\Scripts\python.exe`).

```powershell
# Validate / generate config
python Configuration_System.py validate configs/unified_config.yaml
python Configuration_System.py generate ./config_examples

# Train (PowerShell wrapper — sets MSVC/SDK paths for torch.compile)
.\Start_AI_Training.ps1
.\Start_AI_Training.ps1 -ConfigPath configs/unified_config.yaml

# Train (direct). Configuration is YAML-only: edit configs/unified_config.yaml
# (or use ANIME_TAGGER_* env overrides) — there are NO per-field CLI overrides.
python train_direct.py --config configs/unified_config.yaml
python train_direct.py --config configs/unified_config.yaml --validate-only

# Generate vocabulary (do not hand-edit vocabulary.json)
python vocabulary.py <dataset_root1> [<dataset_root2> ...]
python vocabulary_utils/vocab_append.py

# PyTorch inference
python Inference_Engine.py --model ./checkpoints/best_model.pt --config ./checkpoints/model_config.json --vocab vocabulary.json

# ONNX export (incl. fp16) and ONNXRuntime inference
python ONNX_Export.py <checkpoint> --output ./artifacts/model.onnx
python ONNX_Export.py <checkpoint> -o model.onnx --variants full fp16 --image-size 448 --opset 19
python onnx_infer.py model.onnx image1.jpg image2.jpg --output predictions.json --threshold 0.3 --top_k 20

# Standalone validation (batch size / max_samples / threshold come from unified_config.yaml)
python validation_loop.py --checkpoint <path> --mode full --output-dir ./validation_results

# Flip-augmentation test
python test_flip_pipeline.py

# Ops launchers
.\Start_TensorBoard.ps1 -LogDir .\tensorboard -Port 6006
.\Start_ImageReview.ps1 -TensorBoardDir .\tensorboard -Port 8080
```

Using the venv interpreter explicitly (recommended for `tools/` scripts):

```powershell
L:\Dab\payton_env\Scripts\python.exe tools/find_pr_threshold.py --config configs/unified_config.yaml --checkpoint experiments/run1_vit/checkpoints/last.pt
L:\Dab\payton_env\Scripts\python.exe tools/prepare_phase2.py --checkpoint experiments/run1_vit/checkpoints/best_model.pt --output phase2_checkpoint.pt --image-size 448
```

Environment-variable override (prefix `ANIME_TAGGER_`, nested with `__`, case-insensitive):

```powershell
$env:ANIME_TAGGER_TRAINING__LEARNING_RATE = "1e-4"; python train_direct.py --config configs/unified_config.yaml
$env:ANIME_TAGGER_DATA__IMAGE_SIZE = "448"; python train_direct.py --config configs/unified_config.yaml
```

## Config rules & invariants

- **Canonical source:** [configs/unified_config.yaml](configs/unified_config.yaml). Keep the [Configuration_System.py](Configuration_System.py) dataclass hierarchy in sync with it. `--config` has **no default and is not required**; if omitted, the config is built from env overrides + dataclass defaults (no error). The PowerShell wrappers always pass it.
- **Resolution:** `data.image_size` is the single source of truth and is synced to `model.image_size` and `validation.preprocessing.image_size` at startup (`FullConfig.validate`). **Phase 2 target = 448** (Phase 1 was 320).
- **Patch divisibility:** `image_size % patch_size == 0` (enforced in `ModelConfig`). ViT `patch_size=16` → 28×28 = 784 tokens at 448px.
- **Normalization:** mean = std = `[0.5, 0.5, 0.5]` for anime-optimized from-scratch training. Train and inference normalization, pad color, and `color_order` **must stay aligned** across all code paths.
- **Color order:** `data.color_order` is `RGB` (default) or `BGR`. All per-channel values (`normalize_mean`, `normalize_std`, `pad_color`) are interpreted in this order; a single channel flip is applied after PIL→numpy materialization for BGR.
- **Effective batch size:** `batch_size (48) * gradient_accumulation_steps (9) [* world_size]` = **432** samples/optimizer step on a single GPU. Note: `FullConfig.compute_effective_batch_size()` multiplies by `self.training.world_size`, which is **not currently a field on `TrainingConfig`** — that method will raise on a single-GPU run; compute 432 directly, or add a `world_size` field (default 1) before calling it.
- **LR scaling:** base `learning_rate=1.0e-5`, `lr_scaling_mode='sqrt'` → `sqrt(effective_batch/256)` ≈ 1.3× → ~1.4e-5 peak (Phase 2). Modes: `sqrt`/`linear`/`none`.
- **Weight decay:** fixed at `0.05` (`training.weight_decay`) — no dataset-size scaling, by design.
- **Loss (Phase 2):** `AsymmetricFocalLoss` with `gamma_pos=0.0`, `gamma_neg=7.0` (hard-negative focus), `clip=0.2`, `label_smoothing=0.0`, `ignore_indices=[0]`. Phase 1 used `gamma_neg=4.0`, `clip=0.05`. Static class weights are removed as redundant with ASL asymmetry. `MetricComputer` default threshold is `0.2653`.
- **AMP:** `amp_dtype = bfloat16` (required on CUDA; float16 not supported). `GradScaler` is disabled for bfloat16.
- **Optimizers:** `adam`, `adamw`, `adamw8bit`, `sgd`, `rmsprop`, `adan`.
- **Gradient clipping:** enabled, `max_norm=1.0`. **NaN/Inf checks** run periodically (`NAN_CHECK_INTERVAL_STEPS`, default 50).
- **Resume:** `training.resume_from` ∈ `none`/`false`/`off` / `latest` / `best` / `<path>`; defaults to `latest` if checkpoints exist. Mid-epoch resume tracks batch/sample-in-epoch. Architecture is `vit` (the only supported architecture), taken from `model.architecture_type` or inferred from state-dict keys (`patch_embed`, `blocks`).
- **Checkpoints** embed config + preprocessing params (normalize mean/std, image_size, patch_size, color_order) and vocab for reproducible inference. `torch.compile` is deferred until after checkpoint load (preserves tensor strides; requires Triton).
- **Early stopping** watches `val/f1_macro` (patience 4, burn-in 2). Note: F1 is calibration-floored by the fixed 0.2653 threshold under Phase 2's logit shift; the health tracker recommends moving auto-stop to `val/mAP`.
- **Soft-stop:** SIGINT/SIGTERM are queued to the next optimizer-step boundary; a `STOP_TRAINING` sentinel file also triggers a clean stop.

## Data & vocabulary

**Sidecar JSON layout (primary).** Each image has a sibling JSON (e.g. `12345.json` next to `12345.jpg`):

```json
{"filename": "12345.jpg", "tags": "tag1 tag2", "rating": "general"}
```

`tags` accepts a space-separated string or a list; `rating` ∈ general/sensitive/questionable/explicit and is mapped into the multi-hot vector (rating tags are part of the vocabulary). `SidecarJsonDataset` scans the dataset root, auto-splits 95/5 train/val, and caches metadata as **Arrow IPC** under `logs/metadata_cache/` (memory-mapped, shared across workers, version/hash/count-validated). Split caches live in `logs/splits/`. A file-based exclusion manager (`cache_exclusions.txt`) tracks corrupted images.

Manifest mode (`DatasetLoader`, requires `train.json`/`val.json`/`images/`) is **legacy** and does not support flip augmentation; use sidecar mode for new work.

**Vocabulary.** `vocabulary.json` is a JSON object with three sections — `tag_to_index`, `index_to_tag`, and `tag_frequencies` (it is *not* a flat map). The `tag_to_index` sub-map has ~19K entries (`<PAD>=0`, `<UNK>=1`). It is **generated**, never hand-edited:

```powershell
python vocabulary.py <dataset_root>+           # full rebuild via create_vocabulary_from_datasets
python vocabulary_utils/vocab_append.py         # append new tags, preserve existing indices
```

Tags listed in [Tags_ignore.txt](Tags_ignore.txt) are excluded during generation. Vocabulary size sets the output shape: labels are `(num_classes,)` multi-hot vectors. `OO_AUTO_REBUILD_VOCAB=1` forces an auto-rebuild; `train_direct.py` prompts to rebuild if the vocab is missing (falls back to non-interactive when not a TTY).

**Horizontal flip / directional tags (current mechanism).** `orientation_handler.py` no longer exists; there is **no directional-tag swapping** because the vocabulary contains no orientation-sensitive tags. Flip logic is **inlined in `SidecarJsonDataset`**:

- Per-image deterministic-but-epoch-varying decision via `_deterministic_coin()` (CRC32 of `image_id + epoch`), gated by `random_flip_prob`.
- `_decide_flip_mode()` honors an optional `flip_overrides_path` JSON: `{"force_flip": [...], "never_flip": [...]}`, `{"flip": [...]}`, or a bare list.
- `set_epoch()` re-rolls flips across epochs; flip state survives worker pickling via `__getstate__`/`__setstate__`.
- Behavior is covered by [test_flip_pipeline.py](test_flip_pipeline.py).

`configs/orientation_map.json` and its README still exist on disk but are **vestigial** — leftovers of the removed orientation system. `Inference_Engine.py` declares an `ORIENTATION_MAP_PATH` constant but never actually loads the file (flip TTA averages predictions elementwise, with no index remapping); training ignores it entirely. Treat it as a deletion candidate, not a live input.

**Padding masks.** `True = PAD` (letterbox fill). Pixel masks are produced during letterboxing and pooled to token-level ignore masks for attention (see [mask_utils.py](mask_utils.py)).

## Environment & secrets

- **Python:** [pyproject.toml](pyproject.toml) requires `>=3.12`. [payton_env.ps1](payton_env.ps1) creates/activates the venv at `L:\Dab\payton_env` (its `-PythonVersion` default is `3.11`; pass `-PythonVersion 3.12` to match pyproject when creating a fresh venv). It sets `OPPAI_ORACLE_ROOT`, `PYTHONPATH`, `VIRTUAL_ENV`, `PATH`.
- **Dependencies:** [requirements.txt](requirements.txt) (torch ≥ 2.9.1, torchvision ≥ 0.24.1, onnx, onnxruntime-gpu ≥ 1.22, tensorboard, scikit-learn, fastapi, safetensors, …). The pinned floor is **torch ≥ 2.9.1** (Flex Attention itself needs ≥ 2.5); `torch.compile` additionally requires Triton.
- **Setup:** `.\payton_env.ps1 -VenvPath L:\Dab\payton_env -PythonVersion 3.12 -InstallDeps`.
- **Secrets:** copy `sensitive_config.py.example` → `sensitive_config.py` (git-ignored). Used by [Monitor_log.py](Monitor_log.py) for optional webhook URLs; absence is handled gracefully.
- **torch.compile on Windows:** needs Visual Studio Build Tools + Windows SDK; `Start_AI_Training.ps1` configures these paths.

## Generated artifacts — do not edit or commit

These are runtime/generated and excluded by [.gitignore](.gitignore) (no Git LFS):

- `experiments/` — training run outputs (checkpoints, logs).
- `tensorboard/` — TensorBoard event files.
- `logs/` — training logs, `metadata_cache/` (Arrow), `splits/`, dedup hashes (only `.gitkeep` tracked).
- `exported_model/`, `huggingface_release/` — ONNX/safetensors export artifacts (per-variant subdirs: `V1_onnx`, `V1.1_onnx`, `V1.1_fp8_onnx`, …).
- `l2_cache/`, `analysis_sample_cache/`, `analysis_sample_output/` — caches.
- `__pycache__/`, `.ruff_cache/` — bytecode/lint caches.
- `image_review/` — tracked web UI, but produces runtime output (`corrections.json`).
- IDE/config dirs: `.claude/`, `.gemini/`, `.code-review/`, `.vscode/`, `.research/`.

**Stray junk files** (untracked, safe to ignore or clean — do not treat as project inputs): `=3.9.0`, `nul`, `Untitled-1.json`, `tunnel_log*.txt`, `Start_*.lnk`, `test_roll.onnx` / `test_roll.onnx.data`, `exported_model_fp16.onnx`, `layer_sweep.csv` at repo root.

## Deprecations / removed

- **`orientation_handler.py` — REMOVED** (commit 518ac59). Do not reference it. Flip logic is inlined in `SidecarJsonDataset`; directional-tag swapping no longer happens.
- **`configs/orientation_map.json` / `.README.md` — VESTIGIAL.** Not loaded anywhere: `Inference_Engine.py` declares an `ORIENTATION_MAP_PATH` constant but never reads it, and training ignores it. Deletion candidate. There is no `data.orientation_safety_mode` field.
- **PyTorch Lightning entrypoints — REMOVED.** `train_direct.py` is the sole training entrypoint.
- **Legacy safe-checkpoint wrapper — REMOVED.** Use `training_utils.CheckpointManager` exclusively.
- **Offline HDF5 preprocessing — GONE.** There is no `dataset_preprocessor.py` in the tree; the production pipeline is on-the-fly JSON loading with no offline preprocessing step.
- **Rating head — removed.** Ratings are tags in the vocabulary (`MultiTaskLoss` has tag loss only).

## Testing & validation

There is no full pytest suite. To sanity-check changes:

- **Config:** `python Configuration_System.py validate configs/unified_config.yaml` after any schema/YAML change.
- **Flip pipeline:** `python test_flip_pipeline.py` (determinism, epoch variation, worker serialization, pixel correctness) after touching dataset/flip code.
- **Quick model/loss/metric smoke checks:**

  ```powershell
  python -c "from model_architecture import create_model, VisionTransformerConfig; print(create_model(config=VisionTransformerConfig(image_size=448)))"
  python -c "from loss_functions import AsymmetricFocalLoss; print(AsymmetricFocalLoss(gamma_pos=0.0, gamma_neg=7.0, alpha=1.0, clip=0.2))"
  python -c "from evaluation_metrics import MetricComputer; print(MetricComputer(num_labels=100, threshold=0.2653))"
  ```

- **Eval without training:** `python validation_loop.py --checkpoint <path> --mode full ...`, or `tools/run_validation_for_epoch.py` (replicates the in-train validation: metrics, skip indices, 30K subsample seed) and check the row it emits to [TRAINING_HEALTH_TRACKER.md](TRAINING_HEALTH_TRACKER.md).
- **F1/threshold debugging:** `tools/diagnose_f1.py` and `tools/find_pr_threshold.py`.
- **FP8 export integrity:** `tools/validate_fp8.py` / `tools/eval_fp8_map.py` (structure + FP8-vs-FP32 mAP drift) after quantization.
