# AGENTS.md — OppaiOracle

Guide for AI coding agents working in this repo. Keep it accurate; if you change a subsystem, update the relevant section here.

## Project overview

OppaiOracle is a **PyTorch multi-label anime image tagger**, trained from scratch.
The active setup is **V2 Phase 1**: patch16, width 896, 18 layers, 14 heads,
MLP 2320, QK normalization and LayerScale 0.1 (~151M parameters at the old 19K
vocabulary size). The final vocabulary size will be measured after all new data arrives.

- **Phase 1 (configured, not launched):** 320×320, WSD, at most 60 epochs including cooldown.
- **Phase 2/3 (future):** 448 then 512; only Phase 1 is implemented by this setup.
- Training remains procedural through [train_direct.py](train_direct.py).
- Dataset: `E:\Dataset`, including `Danbooru` and another top-level subset the user will add.
  **Do not build the production vocabulary until that subset is present.**
- Operational instructions and old/new rating rules: [todos/v2-phase1-setup.md](todos/v2-phase1-setup.md).

Design rationale and live status live in [todos/v2-plan.md](todos/v2-plan.md) (the authoritative V2 plan; see the Docs section below for the full doc set) and [TRAINING_HEALTH_TRACKER.md](TRAINING_HEALTH_TRACKER.md) (V1 Phase-2 run archive).

## Repo map

### Config

- [configs/unified_config.yaml](configs/unified_config.yaml) — **canonical** config (model/data/training/inference/threshold_calibration/export/validation/monitor/debug). Values here override dataclass defaults.
- [Configuration_System.py](Configuration_System.py) — dataclass config schema + loader. `FullConfig` aggregates `ModelConfig`, `DataConfig`, `GradientClippingConfig`, `LossConfig`, `TrainingConfig`, `InferenceConfig`, `ExportConfig`, `ValidationConfig`, `ThresholdCalibrationConfig`, `MonitorConfig`, `DebugConfig`, etc. Provides env-override (`update_from_env`), a minimal CLI parser (`create_config_parser`: only `--config` and `--validate-only` — **configuration is YAML-only; there are no per-field CLI overrides**), `validate`/`generate` subcommands.
- [training_config.py](training_config.py) — single live helper: `scale_learning_rate` (batch-size LR scaling used by train_direct). The old dataset-aware "auto-config" helpers (weight-decay scaling, warmup/beta2 computation, batch-size/scheduler recommendation) were removed — weight decay is fixed at `training.weight_decay` and the scheduler is exactly `training.scheduler`.
- [schemas.py](schemas.py) — prediction output schemas (`TagPrediction`, `ImagePrediction`, `RunMetadata`, `PredictionOutput`) plus `canonical_vocab_bytes` / `compute_vocab_sha256`. Not a config module.

### Training

- [train_direct.py](train_direct.py) — **main training entrypoint** (procedural, ~2.5K LOC). `main()` parses args, loads `FullConfig`, dispatches validation-only or training. Core loop is `train_with_orientation_tracking(config)` (name retained for continuity; flip logic is inlined in the dataset). Handles vocab auto-build, dataloaders, ViT setup, AMP (bfloat16), gradient accumulation, checkpointing, metrics (F1-macro, mAP), early stopping, soft-stop, `torch.compile`, NaN/Inf detection.
- [Start_AI_Training.ps1](Start_AI_Training.ps1) — Windows launcher. Sources `payton_env.ps1`, sets VS Build Tools + Windows SDK paths for `torch.compile`, then runs `train_direct.py`. Args: `-ConfigPath`, `-TrainingArgs`, `-KeepOpenOnError`, `-KeepOpen`. `-CheckTrainingStack` runs the bounded synthetic CUDA probe instead of production training.
- [training_utils.py](training_utils.py) — training core: `TrainingState`, `CosineAnnealingWarmupRestarts`, `AsyncCheckpointWriter`, `CheckpointManager` (canonical checkpoint lifecycle), `TrainingUtils` (seed/optimizer/param-group/scheduler helpers), `validate_config_compatibility`, `detect_architecture_from_state_dict`.
- [schedulers.py](schedulers.py) — `WarmupStableDecayLR` (V2 update-based warmup → stable → 1-sqrt cooldown with persisted state) and legacy `LinearWarmupCosineLR`.
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
- [Tags_ignore.txt](Tags_ignore.txt) — exclusion list for vocab generation (77 unique category-prefixed exclusions; legacy bare spellings are also recognized).
- [selected_tags.csv](selected_tags.csv) — `(tag_id, name, category)` export for tagger-UI compatibility; vocabulary.json is canonical.

### Model / loss / metrics

- [model_architecture.py](model_architecture.py) — `BaseTagger` (abstract), `SimplifiedTagger` (**ViT**, PyTorch 2.5+ Flex Attention with `scaled_dot_product_attention` ONNX fallback), `VisionTransformerConfig` (its config dataclass), `create_model(config, architecture_type='vit')`. Returns `{'tag_logits': ..., 'logits': ...}`.
- [loss_functions.py](loss_functions.py) — `AsymmetricFocalLoss` (multi-label ASL; `gamma_pos`, `gamma_neg`, `alpha`, `clip`, `label_smoothing`, `ignore_indices=[0]`, per-class weights; fp32 probability/log/focal math under bf16 autocast, detached focal weights by default) and `MultiTaskLoss` wrapper (tag loss only).
- [evaluation_metrics.py](evaluation_metrics.py) — `MetricComputer` (F1 macro/micro, mAP; default threshold **0.7927**, the measured micro-F1-optimal point that replaced 0.2653; `skip_indices=[0]`), `FrequencyBucketMetrics` (LVIS-style per-frequency buckets), `ThresholdCalibrator` (per-tag/per-bucket threshold search).

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
- [tools/check_training_stack.py](tools/check_training_stack.py) — two-layer synthetic CUDA probe using configured V2 width/grid and compile mode; compares bf16 Flex Attention outputs/loss/gradients with SDPA, checks AdamW8bit/fp32-head states and partial-batch validation. Run via `Start_AI_Training.ps1 -CheckTrainingStack` for MSVC/SDK setup. Timing is not a full-model benchmark.
- [tools/export_fp8.py](tools/export_fp8.py) / [tools/export_fp8_weightonly.py](tools/export_fp8_weightonly.py) — FP8 (E4M3) static / weight-only quantization; re-attach vocab+metadata afterward.
- [tools/validate_fp8.py](tools/validate_fp8.py) / [tools/eval_fp8_map.py](tools/eval_fp8_map.py) — FP8 structure validation / FP8-vs-FP32 mAP drift.
- [tools/release_v1_1.py](tools/release_v1_1.py) — build V1.1 safetensors (bfloat16) + `selected_tags.csv` into `huggingface_release/`; strips `_orig_mod`/DDP/compile prefixes.
- [tools/restamp_vocab_sha.py](tools/restamp_vocab_sha.py) — re-stamp `vocab_sha256` in checkpoints from embedded `vocab_b64_gzip`.
- [tools/corrections_report.py](tools/corrections_report.py) / [tools/corrections_report_md.py](tools/corrections_report_md.py) — analyze `image_review/corrections.json` (hardcoded paths).
- [utils/](utils/) — internal support package imported across the codebase: `exclusion_manager.py` (bad/corrupted-image tracking), `metadata_cache.py` / `metadata_ingestion.py` (Arrow metadata cache), `logging_setup.py` / `logging_sanitize.py`, `memory_monitor.py`, `file_handlers.py`, `path_utils.py`.
- PowerShell ops: [Start_TensorBoard.ps1](Start_TensorBoard.ps1), [Start_ImageReview.ps1](Start_ImageReview.ps1), [Start_ImageReview_Remote.ps1](Start_ImageReview_Remote.ps1) (Cloudflare tunnel; needs `cloudflared`).

### Docs & state

- [todos/README.md](todos/README.md) — entry point for the V2 roadmap, document ownership, dependency order, and evidence index. Active plans stay at `todos/`; supporting reviews live in `todos/reviews/`; superseded snapshots live in `todos/archive/`. Detailed work queues stay in their owning plans rather than being duplicated in the index.
- [todos/v2-augmentation.md](todos/v2-augmentation.md) — consolidated status of both augmentation reviews, pending D1–D4 decisions, and downstream routing. Adopted settings stay in `v2-plan.md` §7; review recommendations are not approvals.
- [todos/reviews/vit-major-issues-audit-2026-09-05.md](todos/reviews/vit-major-issues-audit-2026-09-05.md) — implementation audit and bounded reproductions; closure checks are integrated into `v2-plan.md` §12.
- [todos/v2-plan.md](todos/v2-plan.md) — the authoritative V2 plan (decisions + config). Amend it in place; never add a dated correction layer. September verification retains detached ASL 7/0/.05, distinguishes the paper’s 4/0/.05 baseline, and corrects SoViT shape, missingness, calibration, and gate assumptions. Freeze the gold sampling protocol before P1; the realized V2-dependent pool follows fixed-model inference. Frozen evidence appendix: [todos/reviews/v2-plan-review-2026-07-28.md](todos/reviews/v2-plan-review-2026-07-28.md) (the plan wins on any disagreement).
- [todos/v2-monitoring.md](todos/v2-monitoring.md) — canary checks, decision rules, and the scalar-reduction procedure for a live run. Bands are V1-calibrated and need re-deriving against V2's val set and threshold.
- [todos/janitor-cleaning-model-plan.md](todos/janitor-cleaning-model-plan.md) — throwaway cleaning-model campaign (one pass over the ~5.92M corpus, then discard).
- ordinal-fusion track — spec split out in `2c1cfa1` (2026-07-29), retired from the tree 2026-07-30; recover via `git show 2c1cfa1:todos/ordinal-fusion-track.md`. Living summary in janitor plan §7.1. Deferred DINOv2 external-anchor fusion (ordinal / M2 cleaning); a separate program from the janitor, which is structurally blind to the diagonal.
- [todos/v2.1.1-gold-training-pipeline.md](todos/v2.1.1-gold-training-pipeline.md) — gold-training pipeline (renamed 2026-07-30 from `v2.1.1-suggestion-engine-buildout.md`; the old name described a scope its own IS-NOT list disavows); frozen evidence base in [.research/_wf5_v2.1.1_verification.md](.research/_wf5_v2.1.1_verification.md).
- [.research/golden_set_plan.md](.research/golden_set_plan.md) (taxonomy, rubrics, method) + [.research/golden_set_collection_targets.md](.research/golden_set_collection_targets.md) (all per-bucket numbers + Anima provenance, §8). The seam is strict: no target tables in the plan.
- [TRAINING_HEALTH_TRACKER.md](TRAINING_HEALTH_TRACKER.md) — **V1 Phase-2 run archive** (soft-stopped at E5; per-epoch metrics only — methodology moved to `todos/v2-monitoring.md`). Row format still emitted by `tools/run_validation_for_epoch.py`, but a freshly emitted row is *not* numerically comparable to E0–E5 (different threshold).

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
- **Resolution:** `data.image_size` is the single source of truth and is synced to `model.image_size` and `validation.preprocessing.image_size` at startup (`FullConfig.validate`). **Active Phase 1 = 320**; later targets are 448/512.
- **Patch divisibility:** `image_size % patch_size == 0` (enforced in `ModelConfig`). ViT `patch_size=16` → 28×28 = 784 tokens at 448px.
- **Normalization:** mean = std = `[0.5, 0.5, 0.5]` for anime-optimized from-scratch training. Train and inference normalization, pad color, and `color_order` **must stay aligned** across all code paths.
- **Color order:** `data.color_order` is `RGB` (default) or `BGR`. All per-channel values (`normalize_mean`, `normalize_std`, `pad_color`) are interpreted in this order; a single channel flip is applied after PIL→numpy materialization for BGR.
- **Effective batch size:** provisional `64 × 16 × world_size(1) = 1024`. Profile the final vocabulary on the target GPU before the long run. `TrainingConfig.world_size` now exists (default 1).
- **LR scaling:** base `2.7e-4`, sqrt scaling against 256 → `5.4e-4` stable LR at batch 1024.
- **V2 schedule:** `training.scheduler=wsd`, 10,000-update warmup, 15% elapsed-update cooldown, terminal LR zero, maximum 60 epochs including cooldown. Plateau dispatch starts disabled until stable-phase jitter review; budget cooldown is automatic. WSD rejects changed resume geometry. Soft stop pauses; it does not certify phase completion. NaN/Inf loss/gradients abort. Attention-logit checks use a two-image diagnostic sample, not an exhaustive maximum.
- **Prepared-data guard:** `tools/prepare_v2_dataset.py --config configs/unified_config.yaml` builds the new vocabulary and exact 30K split after all subsets arrive. Startup verifies manifest/hash/count/recipe conformance. Existing V2 checkpoints block preparation/reindexing. No production vocabulary has been built during setup.
- **Weight decay:** fixed at `0.05` (`training.weight_decay`) — no dataset-size scaling, by design.
- **Loss (fixed training contract):** `AsymmetricFocalLoss` with `gamma_pos=0.0`, `gamma_neg=7.0`, `clip=0.05`, `alpha=1.0`, `label_smoothing=0.0`, `detach_focal_weight=true`, `ignore_indices=[0, 1]`. `ASLDriveManager` verifies config and live criterion after resume; conflicting checkpoint gamma aborts, gamma overrides and schedules are refused. Telemetry has no loss-mutation authority. Phase 1 used `gamma_neg=4.0`, `clip=0.05`. Static class weights are removed as redundant with ASL asymmetry. `MetricComputer` default threshold is `0.7927` (measured micro-F1 optimum; replaced 0.2653).
- **AMP:** `amp_dtype = bfloat16` (required on CUDA; float16 not supported). `GradScaler` is disabled for bfloat16.
- **Optimizers:** `adam`, `adamw`, `adamw8bit`, `sgd`, `rmsprop`, `adan`. bitsandbytes 0.50 uses block-wise 8-bit states unconditionally; do not pass the removed `block_wise` constructor argument. The V2 fp32 tag-head/pos-embed overrides remain enabled.
- **Gradient clipping:** enabled, `max_norm=1.0`. **NaN/Inf checks** run periodically (`NAN_CHECK_INTERVAL_STEPS`, default 50).
- **Resume:** `training.resume_from` ∈ `none`/`false`/`off` / `latest` / `best` / `<path>`; defaults to `latest` if checkpoints exist. Mid-epoch resume tracks batch/sample-in-epoch. Architecture is `vit` (the only supported architecture), taken from `model.architecture_type` or inferred from state-dict keys (`patch_embed`, `blocks`).
- **Checkpoints** embed config + preprocessing params (normalize mean/std, image_size, patch_size, color_order) and vocab for reproducible inference. `torch.compile` is deferred until after checkpoint load (preserves tensor strides; requires Triton).
- **Stop system** lives in [stop_conditions.py](stop_conditions.py) and watches the configured `selection_metric` — currently **`val_mAP`** (dispatched via `_SELECTION_METRICS` in `train_direct.py`). Every rule is a **pure function of `TrainingState.eval_history`** (one record per *validated* epoch, keyed by epoch), so a verdict is identical before and after a soft stop and a replayed epoch cannot double-count. Rules: `plateau` (best of last *k* vs best of previous *k* < `min_delta`, confirmed), `peak_decline`, `overfit_loss` (canary #6), plus informational `budget` and `clean_margin`. Geometry is per-phase via `training.early_stopping_phases`. `clean_margin` carries `asl_val/sibling_gap_macro` — **veto-only, never a green light** (v2-plan.md §9.2): a *falling* margin corroborates a tripped `peak_decline`; a holding/rising margin is uninformative — a model that has memorized a wrong positive *maximizes* the gap — and must never demote a stop signal. Holding/rising margin leaves the decline trigger intact; falling margin may corroborate it. Default `early_stopping_mode: advise` — a trip is logged and written to `logs/stop_decisions.jsonl`, and only the epoch budget ends the run; set `halt` to let it stop training. `early_stopping_patience`/`_threshold` now govern **best-checkpoint selection only**. A cross-metric `best_metric` reset guard handles resuming across a `selection_metric` change. Tests: [test_stop_conditions.py](test_stop_conditions.py).
- **Checkpoint integrity:** a found incompatible resume or missing explicit checkpoint path aborts. Final completion and policy halt save `last.pt` even when not best. A full async queue applies backpressure before synchronous fallback; retention runs after pointer updates and shutdown surfaces write failures. Corrected validation uses fp32 logits before sigmoid; `TrainingState.measurement_contract` resets incompatible best/burn-in/stop history, frozen macro support, cooldown comparisons and telemetry before resume; optimizer/scheduler progress is preserved.
- **Validation memory:** a single fp32/int8 host buffer (preserving unknown targets as -1) replaces concatenation. Frequency buckets use 32-column scratch chunks and exact per-label AP; headline mAP remains binned. `validation.ap_update_chunk_size: 8` bounds GPU AP updates independently of inference batch size, preserving the 200-bin confusion counts. ASL validation telemetry runs independently of TensorBoard.
- **Bad images (prepared V2):** failed images are skipped and reported once per full path in `<log_dir>/bad_images.txt` (path, image ID, error; tab-separated). `utils/bad_image_report.py` owns one low-priority background writer, batches appends every 30 seconds, and flushes on clean exit. No image hashes, preparation decode scan, per-worker file writes, or blacklist are added. Existing `cache_exclusions.txt` does not remove or blacklist prepared rows; sample positions stay fixed for resume and repaired files are retried normally. Validation uses readable images and logs `val/samples_evaluated` / `val/samples_skipped`; an entirely failed validation pass saves progress without inventing a score or selecting a best checkpoint. Tests: [test_training_readiness.py](test_training_readiness.py).
- **Soft-stop:** SIGINT/SIGTERM are queued to the next optimizer-step boundary; a `STOP_TRAINING` sentinel file also triggers a clean stop.

## Data & vocabulary

**Sidecar JSON layout (primary).** New data lives below `E:\Dataset`; each top-level subset may contain nested shard directories:

```json
{"filename": "12345.jpg", "tags": "gen:1girl char:alice meta:highres", "rating": "s"}
```

`utils/metadata_ingestion.py` preserves category prefixes, punctuation and case, parses V2 whitespace-delimited strings plus legacy comma-delimited strings/lists, and centralizes rating mapping. `g/s/q/e` → `rating:general/sensitive/questionable/explicit`; `s` means **sensitive**, while spelled-out `safe` still means general. Missing/unknown ratings retain the image for content-tag training: all four rating targets are -1 (unobserved), excluded from ASL, validation metrics and calibration. Known ratings supervise one positive and three negatives; the separate field overrides inline rating tokens. ASL mean reduction averages observed labels per image, then images. Ratings remain mandatory vocabulary entries and share ASL with other tags; no separate rating head. Rating-field frequencies are counted once per rated image; V2 preparation records rated/unrated counts, and rating bias priors use only rated images. Blank/null/malformed/unrecognized rating fields are unobserved. Optional ASL rating diagnostics also mask unknown targets. Sparse or entirely unrated validation is supported; buckets with no positive calibration evidence retain the default threshold.

`utils/sidecar_discovery.py` discovers image/JSON pairs and skips environments/updater JSONs. V2 image IDs include a directory hash so numeric IDs from different subsets do not collide. Split and Arrow caches are version 3.0; Arrow selections remain mmap-backed across worker spawn. Top-level subset changes invalidate preparation and vocabulary file-list caches; the bare loader preserves cached validation membership when appending new subsets to training. After preparation, the dataset is frozen: rerun preparation before training if files or labels change inside an existing subset.

User decision for V2: **fresh deterministic 30K validation set**, drawn only after both subsets are ready; all other images train. No extra CALIB/TEST reservation. Preparation records TRAIN/EVAL membership and label hashes plus source-sidecar SHA256 snapshots under `logs/v2_phase1`; startup verifies artifacts, not every raw sidecar. Old V1 split-cache lists were removed; source images were retained. Full-corpus counts await preparation.

Manifest mode (`DatasetLoader`, requires `train.json`/`val.json`/`images/`) is **legacy** and does not support flip augmentation; use sidecar mode for new work.

**Vocabulary.** `vocabulary.json` is a JSON object with three sections — `tag_to_index`, `index_to_tag`, and `tag_frequencies` (it is *not* a flat map). The `tag_to_index` sub-map has ~19K entries (`<PAD>=0`, `<UNK>=1`). It is **generated**, never hand-edited:

```powershell
python vocabulary.py <dataset_root>+           # full rebuild via create_vocabulary_from_datasets
python vocabulary_utils/vocab_append.py         # append new tags, preserve existing indices
```

Tags listed in [Tags_ignore.txt](Tags_ignore.txt) are excluded during generation. V2 writes `vocabulary/v2/vocabulary.json`; the root `vocabulary.json` remains a legacy artifact. Preparation is required before V2 startup and bypasses the old interactive rebuild prompt. Vocabulary size sets the output shape: labels are `(num_classes,)` multi-hot vectors. `OO_AUTO_REBUILD_VOCAB=1` forces an auto-rebuild; `train_direct.py` prompts to rebuild if the vocab is missing (falls back to non-interactive when not a TTY).

**Horizontal flip / directional tags (current mechanism).** `orientation_handler.py` no longer exists; there is **no directional-tag swapping**. The augmentation review identified four chirality exceptions (`left-handed`, `left-to-right_manga`, `right-over-left_kimono`, `right-to-left_comic`); their proposed name-based exclusion is pending D1 in [todos/v2-augmentation.md](todos/v2-augmentation.md). Flip logic is **inlined in `SidecarJsonDataset`**:

- Per-image deterministic-but-epoch-varying decision via `_deterministic_coin()` (CRC32 of `image_id + epoch`), gated by `random_flip_prob`.
- `_decide_flip_mode()` honors an optional `flip_overrides_path` JSON: `{"force_flip": [...], "never_flip": [...]}`, `{"flip": [...]}`, or a bare list.
- `set_epoch()` re-rolls flips across epochs; flip state survives worker pickling via `__getstate__`/`__setstate__`.
- Behavior is covered by [test_flip_pipeline.py](test_flip_pipeline.py).

`configs/orientation_map.json` and its README still exist on disk but are **vestigial** — leftovers of the removed orientation system. `Inference_Engine.py` declares an `ORIENTATION_MAP_PATH` constant but never actually loads the file (flip TTA averages predictions elementwise, with no index remapping); training ignores it entirely. Treat it as a deletion candidate, not a live input.

**Padding masks.** `True = PAD` (letterbox fill). Pixel masks are produced during letterboxing and pooled to token-level ignore masks for attention (see [mask_utils.py](mask_utils.py)). Block-mask creation reuses `torch.compile(create_block_mask)` when Triton is available; do not use PyTorch's deprecated private `_compile` argument. Compilation remains lazy. `_create_block_mask` uses `torch.compiler.disable(recursive=False)` to keep host dispatch outside the dynamic model graph (avoiding PyTorch 2.14's reproduced Inductor `CantSplit` failure); the inner mask kernels remain compiled. Retain `compile_fullgraph: false`.

## Environment & secrets

- **Python:** [pyproject.toml](pyproject.toml) requires `>=3.12`. [payton_env.ps1](payton_env.ps1) creates new environments with Python 3.12 by default and preserves the interpreter of existing environments, including the tested legacy Python 3.11.9 venv at `L:\Dab\payton_env`. An explicit `-PythonVersion` enforces that version. It sets `OPPAI_ORACLE_ROOT`, `PYTHONPATH`, `VIRTUAL_ENV`, `PATH`; `-PythonExe` selects the interpreter for creation.
- **Dependencies:** [requirements.txt](requirements.txt) pins the matched training stack: **torch 2.14.0, torchvision 0.29.0, triton-windows 3.8.0.post28, bitsandbytes 0.50.2**. Windows Triton is platform-gated; Linux PyTorch supplies upstream Triton. [requirements-training-cu130.txt](requirements-training-cu130.txt) selects explicit CUDA 13.0 wheels before the rest of the requirements. Update both manifests together. Details and upstream sources: [docs/training-dependencies.md](docs/training-dependencies.md).
- **Setup:** `.\payton_env.ps1 -VenvPath L:\Dab\payton_env -InstallDeps` updates the existing venv, stops on pip failures, and runs `pip check`. New environments default to Python 3.12. Legacy torchaudio 2.11 is unused and must be removed before upgrading torch. Verify kernels with `.\Start_AI_Training.ps1 -CheckTrainingStack`.
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

- **Rating edge cases:** `python -B test_rating_system.py` (43 field variants through uncached/cold/warm Arrow workers, complete unrated preparation, CPU/CUDA bf16 gradient isolation, sparse metrics, telemetry and resume reset).
- **V2 regressions:** `python -B test_v2_pipeline.py` (new tag/rating formats, unrated-image retention and masked loss/metrics, exact holdout preparation, subset IDs, Arrow workers, QK/LayerScale, WSD completion/resume, CUDA fp32-head/8-bit-backbone states).
- **Training audit:** `python -B test_training_audit.py` (CUDA loss gradients, metric precision, holdout isolation, worker allocations, queue saturation, completion/halt checkpoints, fixed-loss enforcement). Uses temporary data and checkpoints.
- **Training readiness fixes:** `python -B test_training_readiness.py` (report deduplication across restarts, background-only file I/O, Arrow/fallback workers, repaired-image retry, real WSD resume with bad TRAIN/EVAL images, all-failed EVAL, identical chunked AP counts on CPU/CUDA).
- **Config:** `python Configuration_System.py validate configs/unified_config.yaml` after any schema/YAML change.
- **Flip pipeline:** `python test_flip_pipeline.py` (determinism, epoch variation, worker serialization, pixel correctness) after touching dataset/flip code.
- **Quick model/loss/metric smoke checks:**

  ```powershell
  python -c "from model_architecture import create_model, VisionTransformerConfig; print(create_model(config=VisionTransformerConfig(image_size=448)))"
  python -c "from loss_functions import AsymmetricFocalLoss; print(AsymmetricFocalLoss(gamma_pos=0.0, gamma_neg=7.0, alpha=1.0, clip=0.2))"
  python -c "from evaluation_metrics import MetricComputer; print(MetricComputer(num_labels=100, threshold=0.7927))"
  ```

- **Eval without training:** `python validation_loop.py --checkpoint <path> --mode full ...`, or `tools/run_validation_for_epoch.py` (replicates the in-train validation: metrics, skip indices, 30K subsample seed) and check the row it emits to [TRAINING_HEALTH_TRACKER.md](TRAINING_HEALTH_TRACKER.md).
- **F1/threshold debugging:** `tools/diagnose_f1.py` and `tools/find_pr_threshold.py`.
- **FP8 export integrity:** `tools/validate_fp8.py` / `tools/eval_fp8_map.py` (structure + FP8-vs-FP32 mAP drift) after quantization.
