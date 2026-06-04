# Training Pipeline Map

This document outlines the core components of the training pipeline for the OppaiOracle project. It focuses strictly on files involved in the training process, excluding supporting tools and analysis scripts.

> **Last refreshed:** 2026-04-27 against the working tree at `l:\Dab\OppaiOracle`. Verified by reading current sources; reflects the post-`64a87d9` refactor (vocabulary, ONNX export, training pipeline) and the local removal of `orientation_handler.py`.

## 1. Entry Points
*   **`train_direct.py`**: The main entry point for the training process. Procedural orchestrator (no `Trainer` class) that loads config, builds the model and dataloaders, runs the training loop, manages checkpoints, and handles soft-stop / sentinel-file shutdowns.
    *   **`main()`** (~L2495): Parses CLI args, loads `FullConfig`, dispatches to validation-only mode or the training loop.
    *   **`train_with_orientation_tracking(config)`** (~L272): The training loop. Despite its name (kept for continuity with the prior orientation refactor), it now drives the inlined flip logic in `SidecarJsonDataset` rather than a separate handler — see §3.
    *   Helpers: `assert_finite()`, `_compute_class_weights()`, `_shutdown_dataloader_workers()`.
*   **`Start_AI_Training.ps1`** (+ `Start_AI_Training.lnk`): PowerShell launcher that sources `payton_env.ps1`, configures Visual Studio Build Tools paths (for `torch.compile`), and invokes `train_direct.py`.
*   **`payton_env.ps1`**: Activates the `L:\Dab\payton_env` virtualenv (Python 3.11), sets `PYTHONPATH`, `VIRTUAL_ENV`, and MSVC paths.
*   **Sibling launchers** (not part of the training loop, but co-located): `Start_TensorBoard.ps1`, `Start_ImageReview.ps1`, `Start_ImageReview_Remote.ps1` (Cloudflared tunnel).

## 2. Configuration System
*   **`Configuration_System.py`**: Central config schema. Defines the dataclass hierarchy, validation, YAML/JSON I/O, and CLI override parsing.
    *   **`FullConfig`**: Aggregates every section below into a single root object loaded by `train_direct.main()`.
    *   **`ConfigManager`**: Loader/validator backing `load_config()`.
    *   **Section dataclasses**: `ModelConfig`, `DataConfig`, `StorageLocation`, `GradientClippingConfig`, `LossConfig`, `TrainingConfig`, `InferenceConfig`, `ExportConfig`, `ValidationConfig` (composes `ValidationDataloaderConfig` + `ValidationPreprocessingConfig`), `ThresholdCalibrationConfig`, `TBImageLoggingConfig`, `MonitorConfig`, `DebugConfig`, `AdamW8bitConfig`, `SchedulerConfig` (+ `SchedulerType` enum).
    *   **Helpers**: `load_config()`, `create_config_parser()`, `deep_update()`, `generate_example_configs()`.
*   **`training_config.py`**: Hyperparameter-scaling utilities (separate from the dataclass schema). Provides its own `AdamW8bitConfig` / `SchedulerConfig` plus dataset-aware scaling helpers — `compute_effective_batch_size`, `scale_learning_rate`, `compute_warmup_steps`, `scale_weight_decay`, `adjust_beta2_for_long_training`, `get_adamw8bit_config`, `recommend_scheduler`, `compute_cycle_steps`, `get_scheduler_config`, `create_scheduler_from_config`, plus batch-size guards (`detect_gpu_memory`, `get_recommended_batch_size`).
*   **`configs/unified_config.yaml`**: Single source of truth consumed by `FullConfig`. Top-level sections: `model`, `data`, `training`, `inference`, `threshold_calibration`, `export`, `validation`, `monitor`, `debug`.
*   **`configs/orientation_map.json`** (+ `orientation_map.README.md`): Left/right tag swap mapping. **No longer consumed by any code** — the orientation handler was removed and `Inference_Engine.py` no longer references it; this file is now vestigial (deletion candidate).
*   **`configs/README.md`**: General config guidance.

## 3. Data Pipeline
*   **`dataset_loader.py`**: The core data-loading module. Now also owns flip-augmentation logic.
    *   **`SidecarJsonDataset`** (~L1811): Primary dataset. Per-image JSON sidecars; **horizontal-flip decisions are inlined here** via `_deterministic_coin()` (CRC32 epoch-aware hash), `_decide_flip_mode()` (force/never/random based on `flip_overrides_path` + `random_flip_prob`), and `set_epoch()` for cross-epoch variation. Flip state is preserved across worker pickling via `__getstate__`/`__setstate__`.
    *   **`DatasetLoader`** (~L954): Manifest-mode PyTorch Dataset (legacy path).
    *   **`DataLoader`** (~L413): Wrapper around `torch.utils.data.DataLoader` for custom worker / threading edge cases.
    *   **`create_dataloaders`** (~L2632): Factory for train/val loaders + vocab creation.
    *   **`AugmentationStats`** / **`validate_dataset`**: Augmentation counters and dataset sanity checks.
    *   **`ArrowMetadataAccessor`**: Zero-copy Arrow IPC metadata cache.
    *   **`IndependentColorJitter`**: Color augmentation respecting per-channel dtype.
    *   **`ResumableSampler`**: `DistributedSampler` subclass with epoch tracking for resume.
    *   **`WorkerInitializer`**: Per-worker init hook (RNG, vocab, etc.).
    *   **`BackgroundValidator`** (Thread): Async background validation runner.
    *   Manifest/split-cache helpers and rating mapping (`_map_rating_to_tag`, `_compute_exclusion_hash`, split-cache I/O).
*   **`vocabulary.py`**: Tag vocabulary management.
    *   **`TagVocabulary`**: Bidirectional tag↔index mapping, frequency counts, rating tags, JSON I/O.
    *   **`load_vocabulary_for_training()`**: Cached load from file or directory.
    *   **`create_vocabulary_from_datasets()`**: Parallel scan of JSON sidecars; emits frequency-sorted vocab.
    *   **`verify_vocabulary_integrity()`**: Hash-based integrity checks.
*   **`shared_vocabulary.py`**: Cross-worker shared-memory vocabulary (Python 3.8+ `multiprocessing.shared_memory`) to avoid per-worker duplication.
    *   **`SharedVocabularyManager`**, **`populate_vocab_from_shared()`**, **`is_shared_memory_available()`**.
*   **`vocabulary_utils/`**: Standalone helpers (`vocab_append.py`, `vocab_utils.py`) for offline vocab maintenance — not imported by the training loop.

> **Removed:** `orientation_handler.py` was deleted in this working tree (still present in older commits). All flip logic now lives in `SidecarJsonDataset` (above), and `test_flip_pipeline.py` at the repo root validates the inlined behavior (deterministic per-image flips, epoch variation, worker-pickle safety, end-to-end pixel correctness). `configs/orientation_map.json` is no longer read by the training pipeline.

## 4. Model Architecture
*   **`model_architecture.py`**: Defines the neural network structure. Uses PyTorch 2.5+ Flex Attention (Triton-backed when available).
    *   **`BaseTagger`** (ABC): Common interface for all tagger variants.
    *   **`SimplifiedTagger`**: Vision Transformer (ViT) tagger.
    *   **`VisionTransformerConfig`**: Dataclass for ViT dimensions / hyperparameters.
    *   **`TransformerBlock`**: Single ViT block (Flex Attention + MLP).
    *   **`LayerNormFp32`**: FP32 layer norm for stability under bf16/fp16.
    *   **`create_model()`**: Factory; constructs the ViT model (`SimplifiedTagger`).
    *   **`initialize_tag_head_bias()`**: Per-class bias init from priors.
    *   **`_check_triton_available()`**: Runtime guard for the Triton-backed Flex Attention path.
*   **`model_metadata.py`**: `ModelMetadata` dataclass — checkpoint provenance (vocab hash, arch fingerprint, etc.) stamped into saved checkpoints.
*   **`custom_drop_path.py`**: `SafeDropPath` for stochastic depth.
*   **`mask_utils.py`**: Attention-mask utilities for Flex Attention block masks.

## 5. Training Logic & Utilities
*   **`loss_functions.py`**: Loss functions for multi-label tagging.
    *   **`AsymmetricFocalLoss`**: Primary loss; asymmetric focal with class weights, label smoothing, ignore indices, and an LRU-cached keep-mask.
    *   **`MultiTaskLoss`**: Combines tag loss and rating loss into a single scalar.
*   **`training_utils.py`**: Training-loop infrastructure.
    *   **`CheckpointManager`** (~L1109): Save/load best/latest/periodic checkpoints.
    *   **`AsyncCheckpointWriter`** (~L933): Background-thread checkpoint writer.
    *   **`TrainingState`** (~L704): Persistent epoch/step/best-loss/metrics state for resume.
    *   **`MixedPrecisionTrainer`** (~L2368): bf16/fp16 AMP context.
    *   **`CosineAnnealingWarmupRestarts`** (~L770): Custom LR scheduler.
    *   **`EarlyStopping`** (~L866): Patience-based stop with best-checkpoint hook.
    *   **`LearningRateSchedulerFactory`** (~L2267): Scheduler construction.
    *   **`TrainingMetricsTracker`** (~L2465): Per-step/per-epoch metric aggregation.
    *   **`TrainingUtils`** (~L2530): Optimizer/scheduler selection and parameter-group setup.
    *   **`InvalidCheckpointError`** (exception type).
    *   **Functions**: `setup_seed()`, `log_sample_order_hash()`, `log_index_order_hash()`, `detect_architecture_from_state_dict()`, `validate_config_compatibility()`, plus RNG-state pack/unpack helpers.
*   **`adan_optimizer.py`**: Adan optimizer implementation. `Adan` class with optional fused CUDA kernels (falls back to pure PyTorch when the extension is unavailable); `MultiTensorApply` helper.
*   **`schedulers.py`**: `LinearWarmupCosineLR` — linear warmup → cosine annealing, PyTorch 2.x compatible (epoch-stepped).
*   **`Monitor_log.py`**: Training monitoring, logging, and alerting.
    *   **`TrainingMonitor`** (~L833): Top-level monitor — TensorBoard writers, metric aggregation, system stats.
    *   **`ThreadSafeMetricsTracker`** (~L365): Thread-safe metric logging.
    *   **`SystemMonitor`** (~L545): CPU/GPU/RAM/disk telemetry.
    *   **`ImageLogger`** (~L156): TensorBoard image-grid logging.
    *   **`AlertSystem`** (~L177): Webhook alerts (Discord/Slack) with domain whitelisting (`_validate_webhook_url`, `_resolve_webhook_url`).

## 6. Evaluation & Metrics
*   **`evaluation_metrics.py`**: Multi-label metric computation.
    *   **`MetricComputer`**: Macro/Micro F1, mAP (micro/macro/weighted) with configurable threshold and skip indices; device-aware mask cache.
    *   **`FrequencyBucketMetrics`**: Per-frequency-bucket (head/mid/tail) metric breakdown.
    *   **`ThresholdCalibrator`**: Optimal-threshold search.
*   **`validation_loop.py`**: Standalone validation harness with its own `main()`.
    *   **`ValidationConfig`** (module-local dataclass, distinct from `Configuration_System.ValidationConfig`).
    *   **`ValidationRunner`**: Comprehensive validation entry point (hierarchical metrics, threshold sweeps, per-tag analysis).
    *   **Note**: Not imported by `train_direct.py` — `train_direct.py` has its own inline validation step. Run this module directly for richer evaluation.
*   **`schemas.py`**: Pydantic schemas used by validation/inference paths.

## 7. Operational / Observability (adjacent, not training code)
*   **`TRAINING_HEALTH_TRACKER.md`**: Live-run agent runbook — per-epoch mAP/F1 logs, canary decision rules, stop/continue verdicts. References specific TensorBoard run dirs and patience settings (`val/f1_macro` patience=8, burn_in=8). External observability doc, not part of the pipeline code.
*   **`tools/`** (standalone utilities, not imported by the training loop):
    *   **`prepare_phase2.py`**: Convert a Phase-1 checkpoint to Phase-2 (resets optimizer state, updates `image_size`, interpolates positional embeddings).
    *   **`restamp_vocab_sha.py`**: Re-stamp vocab SHA256 in older checkpoints to match canonical-JSON hashing.
*   **`test_flip_pipeline.py`**: End-to-end test for the inlined flip logic in `SidecarJsonDataset` (distribution, determinism, epoch variation, worker-serialization safety).

## 8. Legacy / Deprecated
*   Legacy Lightning entrypoints and pre-unified config helpers were removed during housekeeping (commits `0b913a6`, `d056daa`, `64a87d9`).
*   The standalone `orientation_handler.py` module is gone; flip logic is now inside `SidecarJsonDataset`.
*   Use `train_direct.py`, `training_utils.CheckpointManager`, and `training_config.py` (for hyperparameter scaling) / `Configuration_System.py` (for the dataclass schema).
