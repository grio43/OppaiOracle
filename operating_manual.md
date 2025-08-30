OppaiOracle (MAID) — Operator’s Manual

Purpose: This guide is for day‑to‑day operation of OppaiOracle: running training, monitoring, evaluation, exporting, and inference at scale. It assumes the environment is already installed and configured. For install/setup, see README and configs.

Scope: Daily workflows, knobs to turn safely, health checks, how to react to common situations, and operational guardrails. Not a developer internals doc.

Key Abbreviations
- UC: Unified config at `configs/unified_config.yaml`
- CKPT: Checkpoint under `<output_root>/<experiment_name>/checkpoints/`
- TB: TensorBoard

1) Configuration Hygiene
- Single Data Root: Ensure exactly one `data.storage_locations[*].enabled: true` points to a real dataset root containing either:
  - Manifest mode: `images/`, `train.json`, `val.json`, or
  - Sidecar mode: per‑image `*.json` next to images across shard folders.
- Normalization: Keep `data.normalize_mean/std` set and consistent across training/validation/inference/export. Validation inherits from UC unless overridden.
- Orientation Flips: If `data.random_flip_prob > 0`:
  - Provide `configs/orientation_map.json`; or
  - Set `data.strict_orientation_validation: false` to allow flips without a map (not recommended beyond experiments).
  - Safety modes: `conservative | balanced | permissive`. Conservative is safest in production.
- LR Warmup Units: The scheduler steps once per optimizer update (after gradient accumulation), so `training.warmup_steps` is in optimizer steps. Typical guidance: use ~3–10 epochs worth of updates (compute as `ceil(len(train_loader)/gradient_accumulation_steps) × desired_epochs`). The UC default `10000` is intentionally conservative and may keep LR near minimum too long—tune per dataset.
- Vocabulary: Always use a real `vocabulary.json` (no `tag_###` placeholders). The system fails fast when placeholders are detected.
- Logging: Set `log_dir` and ensure disk space; TB dir defaults to `<output_root>/<experiment_name>`.

2) Daily Training Operations
- Start/Resume Training
  - Command: `python train_direct.py --config configs/unified_config.yaml`
  - Resume policy: `training.resume_from` supports `none | latest | best | /path/to/ckpt.pt`.
  - Deterministic run: `scripts/run_train_deterministic.sh --config configs/unified_config.yaml`
  - AMP: Enabled by default; picks bf16 when available, falls back to fp16 and disables GradScaler for bf16.
  - Memory layout: Set `training.memory_format: channels_last` to unlock Tensor Core throughput on Ampere+.
- Effective Batch Size
  - Use `data.batch_size` × `training.gradient_accumulation_steps`. Increase accumulation if VRAM is tight.
  - Prefer bf16 + channels_last + gradient checkpointing (`model.gradient_checkpointing: true`) to fit larger models.
- DataLoader Knobs
  - `num_workers`: 0 for debug; 4–16 typical. Wrapper auto‑guards `prefetch_factor` and `persistent_workers` when 0.
  - `pin_memory`: true on CUDA systems.
  - L2 cache (LMDB): Enable with `data.l2_cache_enabled: true`; set `l2_cache_path` and `l2_max_size_gb`. Writer is auto‑spawned.
- Orientation‑Aware Augmentation
  - Tune `data.random_flip_prob`; ensure `orientation_map.json` if `strict_orientation_validation: true`.
  - Override exceptional IDs via `data.flip_overrides_path` (JSON with `force_flip`/`never_flip`).
- Vocabulary Lifecycle
  - On startup, if missing, the trainer can rebuild a vocabulary by scanning JSON sidecars; accept only when you intend to refresh vocab. Keep `vocabulary.json` consistent across train/val/export.
- Monitoring
  - TB logging: model graph, training/validation metrics, sample predictions (first val batch). Find under `<output_root>/<experiment_name>/tensorboard/` or `monitor.tensorboard_dir`.
  - System alerts (optional): `monitor.enable_alerts: true` with webhook in `sensitive_config.py`.
  - Logs: JSON console + rotating file logs under `log_dir`.
- Checkpoints
  - Frequency: `training.save_steps` or whenever new best `val_f1_macro` occurs.
  - Best vs latest: `save_best_only` controls retention; limit total via `save_total_limit`.
  - Embedded metadata: Checkpoints embed vocabulary and preprocessing for reproducible export/inference.
- Early Stopping
  - Controlled via `training.early_stopping_patience` and `early_stopping_threshold` on `val_f1_macro`.

3) Health Checks & Quick Verifications
- Compile all: `git ls-files '*.py' | xargs -I {} python -m py_compile "{}"`
- Config validate: `python Configuration_System.py validate configs/unified_config.yaml`
- Import sanity: see README quick command (imports core modules to catch top‑level errors).
- Dataset preflight: `python Dataset_Analysis.py --help` (lightweight scans/stats).
- Sanity batch: Start training and watch first few steps for finite loss and TB graph emit.

4) Evaluation & QA
- Standalone Validation
  - Command: `python validation_loop.py --config configs/unified_config.yaml`
  - Modes: `full | fast | tags | hierarchical`. Use `fast` for time‑boxed checks; `tags` to focus on specific labels.
  - Batch settings: `validation.dataloader` section in UC; validation inherits normalization from UC.
  - Metrics: Macro/micro F1 and mAP via TorchMetrics.
  - Memory note: Validation aggregates predictions/targets in RAM to compute metrics—use `fast` or smaller `batch_size` for very large sets.
- Threshold Calibration (optional)
  - Use `tools/calibrate_thresholds.py` with saved logits/labels to produce per‑tag thresholds JSON; pass to inference via `thresholds_path`.
- Review Tools (TEst and review/)
  - `batch_evaluate.py`, `visualize_results.py`, `live_viewer.py` for exploratory QA and visualization.

5) Export to ONNX
- Command (from UC defaults):
  - `python ONNX_Export.py --config configs/unified_config.yaml --checkpoint <ckpt.pt> -o exported/model.onnx`
- Behavior
  - Wraps model with preprocessing; ONNX input is raw `uint8` `BHWC` (batch, height, width, 3).
  - Embeds vocabulary and preprocessing metadata when possible; export fails fast on placeholder vocab.
  - Opset: defaults to 19; exporter clamps to a supported range when needed.
  - Variants: `--variants full quantized` (dynamic quantization for CPU available).
  - Validation: Structural ONNX check runs by default; numerical comparison is available via a legacy method if needed.
- Operational Tips
  - Ensure `onnxruntime-gpu` fits the CUDA driver on the host.
  - If export complains about vocabulary: confirm training wrote a checkpoint with embedded vocab, or pass a valid `vocab_dir`.
  - For reproducibility downstream, prefer checkpoints with embedded vocab and preprocessing.

6) Inference Ops (PyTorch & ONNX)
- PyTorch Inference
  - Command (single image): `python Inference_Engine.py --config configs/unified_config.yaml --image <img>`
  - Batch/dir: Use the directory processing entrypoints in `Inference_Engine` or custom scripts.
  - Precision: Inherits from checkpoint metadata; will use `bf16` on supported GPUs.
  - Caching: Enable `enable_cache` with `cache_size` and `cache_ttl_seconds` for repeated queries.
- ONNX Inference
  - Command: `python onnx_infer.py <model.onnx> <images...> [--top_k N] [--threshold T] [--output results.json] [--vocab vocabulary.json]`
  - Behavior: Prefers embedded vocabulary/params; falls back to external vocab when missing.
  - Providers: Default `['CUDAExecutionProvider','CPUExecutionProvider']`; override with `--providers` if needed.
  - Transparency handling: PNGs with alpha are composited on grey to match training; results drop the `gray_background` tag if compositing was applied.

7) Data Pipeline Details & Knobs
- Modes
  - Manifest mode: `images/`, `train.json`, `val.json` at the active data root.
  - Sidecar mode: Scans `*.json` next to images across shards; deterministic 95/5 split.
- JSON Schema (sidecar)
  - Required: `{ "filename": "12345.jpg", "tags": ["tag1","tag2",...], "rating": "general|sensitive|questionable|explicit|unknown" }`
  - Tags may be a space‑delimited string or list.
- Padding & Letterbox
  - Images letterboxed to square `data.image_size` using pad_color `[114,114,114]`; a padding mask is propagated to the model’s attention to ignore padded tokens.
- LMDB L2 Cache
  - Enable `data.l2_cache_enabled`; set `l2_cache_path`, and `l2_max_size_gb`. Read is lazy per worker; writer is non‑blocking.
  - Cache contains normalized tensors keyed by `image_id`.
- Background Validator
  - A daemonized `BackgroundValidator` checks sample integrity; benign if not explicitly stopped. If embedding datasets elsewhere, call `validator.stop()` on teardown.

8) Vocabulary Management
- Source of Truth: `vocabulary.json` at repo root (or `vocab_path` in UC).
- Ignore List: `Tags_ignore.txt` omits tags entirely from training/inference.
- Integrity Checks
  - Placeholders like `tag_####` are rejected across training, eval, and export.
  - Operators should never override this—fix the source vocab.
- Rebuilds
  - The trainer can rebuild automatically by scanning the dataset. This is intensive; prefer explicit offline rebuilds when iterating on vocab.

9) Checkpointing & Recovery
- Safe Loading: Only `safe_checkpoint.safe_load_checkpoint` is used—no generic pickle loads.
- Resume Recipes
  - Fault tolerance: Set `training.resume_from: latest` for long runs.
  - Best model: Use `best` for evaluation/export.
- Embedded Metadata
  - Checkpoints embed normalization and patch size; ONNX export and inference read these for consistent preprocessing.

10) Monitoring, Alerts, and Housekeeping
- TB & Logs
  - TB: Inspect training curves, sample predictions; prune old runs periodically.
  - Logs: Rotating logs under `log_dir`; compress backups as needed.
- System Metrics
  - Enable GPU/CPU/disk monitoring under `monitor.*`. Alerts dispatched via Discord webhook in `sensitive_config.py`.
- Cleanup
  - Caches: Rotate/clear `l2_cache` if disk pressure occurs.
  - Checkpoints: Keep `save_total_limit` small; archive best CKPTs off the box.

11) Performance Playbook
- Throughput
  - Prefer `bf16` on Ampere+; set `training.memory_format: channels_last`.
  - Increase `data.num_workers`, `prefetch_factor`; keep `pin_memory: true` on CUDA.
  - Use gradient checkpointing to trade compute for memory.
- Optimization Schedule
  - Warmup in optimizer steps. Aim for ~3–10 epochs worth of updates (compute `ceil(len(train_loader)/gradient_accumulation_steps) × desired_epochs`). Adjust `lr_end` and `num_cycles` thoughtfully.
- Stability
  - Non‑finite guardrails exist in the model; if tripped, enable `training.enable_anomaly_detection: true` and set `debug.*` toggles to dump failing tensors.

12) Troubleshooting Quick Map
- “No enabled storage location found” on train start
  - Set one `data.storage_locations[*].enabled: true` to a real root.
- Orientation flip warnings/errors
  - Provide `configs/orientation_map.json` or set `data.strict_orientation_validation: false`. Keep `conservative` safety in production.
- Vocabulary placeholder/Integrity errors
  - Replace broken `vocabulary.json` with the correct file; do not ignore.
- Dataloader ValueError with `prefetch_factor`
  - Occurs only when `num_workers: 0`; wrapper auto‑guards. Set workers > 0 or drop `prefetch_factor`.
- Validation OOM / long runtimes
  - Use `mode: fast`, reduce batch size, or subset `max_samples`.
- ONNX export failure
  - Check that checkpoint exists and has embedded vocab or pass a valid `vocab_dir`. Ensure `onnxruntime-gpu` matches the host’s CUDA.
- ONNX inference missing vocabulary
  - Pass `--vocab vocabulary.json` for legacy models without embedded vocab.

Appendix: High‑Value Defaults to Review per Run
- `training.warmup_steps`: 3–10 (epoch‑based).
- `data.random_flip_prob`: 0.0–0.5, ensure orientation map when > 0.
- `data.num_workers`: fit to host; 8–16 typical on modern CPUs.
- `data.l2_cache_enabled`: true when storage is slow and disk allows.
- `training.memory_format`: `channels_last` on Ampere+.
- `training.resume_from`: `latest` for iterative work; `best` for polishing/finetune.

File Pointers (for Operators)
- Unified config: `configs/unified_config.yaml`
- Trainer: `train_direct.py:main`
- Validation: `validation_loop.py:ValidationRunner`
- Inference (PyTorch): `Inference_Engine.py:InferenceEngine`
- ONNX export: `ONNX_Export.py:ONNXExporter`
- ONNX inference: `onnx_infer.py:main`
- Vocabulary tools: `vocabulary.py` and `tools/`
