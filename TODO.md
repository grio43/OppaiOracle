TODO — Orphans, Critical and Major Fixes

Scope: Only orphan code, critical errors, and major errors that block or meaningfully degrade training, validation, export, or inference.

Critical Errors
- ONNX export missing import: `torch.nn.functional as F`
  - File: `ONNX_Export.py:1`
  - Symptom: `NameError: F` in `InferenceWrapper.forward()` when calling `F.pad`/`F.interpolate`.
  - Fix: `import torch.nn.functional as F` near other imports.
  - 2025 context: Newer PyTorch ONNX export paths still rely on torchvision-free pre/post transforms; keeping preprocessing in-graph (pad/resize/normalize) is supported with opset 18/19. Ensure `F.interpolate` uses `align_corners=False` (already set) to avoid export warnings.
  - References: ONNX opset 18–19 support; PyTorch ONNX exporter notes for SDPA and image ops.

- ONNX inference alpha compositing uses incorrect mask
  - File: `onnx_infer.py:102`
  - Symptom: `PIL.Image.paste` called with `mask=img` (RGBA image) instead of alpha channel, causing bad transparency handling or runtime errors.
  - Fix: `background.paste(img, mask=img.split()[-1])`.
  - 2025 context: Pillow 10/11 keep the same API; mask must be a single-channel image (usually alpha). Using the full RGBA as mask is incorrect and may error on strict builds.
  - References: Pillow Image.paste docs (mask argument requires single-band image).

- Fragile type annotation when monitoring is unavailable
  - File: `Inference_Engine.py:107`
  - Symptom: `InferenceConfig` annotates `monitor_config: Optional[MonitorConfig]` but `Monitor_log` import is optional. On Python 3.12, this can raise `NameError` at class creation if the import fails.
  - Fix: use string annotations: `monitor_config: Optional['MonitorConfig'] = None` and/or guard with `if TYPE_CHECKING:` for the import.
  - 2025 context: Postponed evaluation of annotations is still opt‑in; forward references should be strings for optional/conditional imports. Alternative: add `from __future__ import annotations` at file top.
  - References: Python typing `TYPE_CHECKING`; dataclasses + forward refs best practices.

- Hard-coded absolute path fallback for vocabulary
  - File: `ONNX_Export.py:155`
  - Symptom: Tries `/media/andrewk/qnap-public/workspace/OppaiOracle/vocabulary.json` if normal paths fail; breaks portability.
  - Fix: remove hard-coded path; surface a clear error instructing to use embedded vocab or provide `export.vocab_dir`.
  - 2025 context: Best practice is embedding vocab in checkpoints/ONNX metadata (already supported) or taking it from unified config. Hard-coded absolute fallbacks cause CI/container failures.
  - References: ONNX custom metadata map usage; ModelMetadata helpers in this repo.

- Packaging constraints need platform‑specific install guidance
  - File: `requirements.txt:29–30`
  - Symptom: `torch>=2.8.0` and `torchvision>=0.23.0` are current as of 2025 but wheel availability depends on OS/Python/CUDA. Blind `pip install -r` may fail without the correct extra index (or CPU fallback) and matching CUDA.
  - Fix: document installation via the PyTorch selector (pip command with proper `--index-url` for CUDA version) and pin a tested pair (e.g., Py3.12 + CUDA 12.x: torch==2.8.0, torchvision==0.23.0). Offer CPU-only fallback (`pip install torch==…+cpu` route) for machines without NVIDIA drivers.
  - References: PyTorch “Get Started” installer matrix (2.8), torchvision compat matrix.

- Unused helper with incorrect signature (would error if called)
  - File: `onnx_infer.py:115`
  - Symptom: `_preprocess_simple(image_path, image_size, mean, std)` calls `_preprocess(...)` with extra args that don’t exist.
  - Fix: remove dead function or implement correct signature.
  - 2025 context: Keep a single preprocessing path; the ONNX graph includes preprocessing. A second codepath invites divergence and export/infer mismatches.

Major Issues
- Lightning training path is incomplete/broken (treat as orphan unless finished)
  - Files: `train_lightning.py`, `lightning_module.py:28,43,62`
  - Problems:
    - DataModule returns empty loaders (non-functional placeholder).
    - `LightningModule` expects outputs `outputs['tag']`/`['rating']` but the model returns `{'tag_logits','rating_logits'}` causing KeyError.
  - Options: mark Lightning path experimental and exclude from docs; or fix to use `tag_logits`/`rating_logits` and implement a real DataModule wired to `dataset_loader`.
  - 2025 fix guidance:
    - Dataloaders: wrap `dataset_loader.create_dataloaders()` and feed `LitOppai` batches as `(images, targets_dict)`.
    - Outputs: rename to `outputs['tag_logits']` / `outputs['rating_logits']` when computing loss/metrics; pass `targets['tag_labels']` and `targets['rating_labels']`.
    - TorchMetrics: in PL 2.5, prefer `task="multilabel"` metrics or `Multilabel*` with `num_labels=vocab_size` and thresholding on probabilities.
  - References: Lightning 2.4–2.5 data flow; torchmetrics multilabel APIs.

- Epoch-based warmup misconfigured to 10000
  - File: `configs/unified_config.yaml: training.warmup_steps`
  - Context: `CosineAnnealingWarmupRestarts` is stepped once per epoch (see `train_direct.py:754`), so `warmup_steps` is in epochs. Current value `10000` will hold LR near minimum for the entire run.
  - Fix: set to a small integer (e.g., `3–10`).
  - 2025 context: PL 2.x and our scheduler both advance once/epoch unless explicitly hooked per‑step; validate `interval="epoch"`. Keep warmup short to avoid freezing learning rate.
  - References: Cosine warmup restarts papers and PL scheduler interval docs.

- Non-portable default data path enabled
  - File: `configs/unified_config.yaml: data.storage_locations[0].path`
  - Symptom: Points to a machine-specific path and `enabled: true` by default; new users will hit “No data” errors.
  - Fix: disable by default or switch to an obvious placeholder; document how to set one `enabled: true` path.
  - 2025 context: Many users run in containers; prefer environment variables or relative workspace paths. Keep exactly one `enabled: true` entry.

- Duplicated path utilities increase confusion
  - Files: `utils/metadata_ingestion.py:47` and `utils/path_utils.py:26`
  - Symptom: Two different `validate_image_path`/`safe_join` implementations; only `utils/path_utils.py` is used in the loaders.
  - Fix: remove duplicates from `metadata_ingestion.py` and import from `utils/path_utils` where needed.
  - 2025 context: Centralize path traversal protections in one module; DataLoader already relies on `utils/path_utils`.

- Logging couples to dataset module
  - File: `utils/logging_setup.py:15`
  - Symptom: Imports `dataset_loader.CompressingRotatingFileHandler` just to get a rotating file handler; this pulls in heavy dataset imports for generic logging.
  - Fix: move `CompressingRotatingFileHandler` into `utils/` (e.g., `utils/logging_handlers.py`) and import it there to avoid side effects.
  - 2025 context: Avoid heavy/side‑effect imports in logging (import‑sanity). Use stdlib handlers or a tiny, isolated helper module.

Orphan/Dead Code
- `_preprocess_simple` in `onnx_infer.py` (unused and wrong signature). Remove or fix.
- `vocab_utils.py` is only used by `train_lightning.py`. If Lightning is deferred, remove or migrate needed helpers.
- `dataset_loader.py:408, 415` placeholders (`AugmentationStats`, `validate_dataset`) are unused; either implement or remove from public surface.
 - 2025 context: Keep public surface minimal; remove placeholders or hide behind private helpers to avoid import‑sanity failures and user confusion.

Preflight/Config Gating (non-code but critical to avoid runtime failures)
- Provide a real `vocabulary.json` (no `tag_###` placeholders) for training/export
  - Files: `vocabulary.py`, `ONNX_Export.py`
  - Action: Generate via `vocabulary.create_vocabulary_from_datasets(...)` or embed vocabulary in checkpoints; code fails fast by design.
  - 2025 context: Prefer embedding vocab in checkpoints/ONNX (ModelMetadata) for reproducibility; external files can drift across runs.

- Orientation flips require map when enabled
  - Files: `configs/unified_config.yaml: data.random_flip_prob`, `configs/orientation_map.json`
  - Action: Ensure `orientation_map.json` exists when `random_flip_prob>0`, or set `data.strict_orientation_validation=false`.
  - 2025 context: Safety modes (“conservative/balanced/permissive”) should be explicit; default to conservative to avoid mislabeled flips.

Notes
- Validation header glitch mentioned in the runbook is not present; `validation_loop.py` starts with a proper shebang/docstring.

Quick References (2025)
- Pillow paste mask: https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.paste
- Python typing forward refs/TYPE_CHECKING: https://docs.python.org/3/library/typing.html#typing.TYPE_CHECKING
- PyTorch get-started installer (2.8): https://pytorch.org/get-started/locally
- ONNX Runtime GPU providers and versions: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- TorchMetrics multilabel: https://torchmetrics.readthedocs.io/en/stable/classification/multilabel_f1_score.html
