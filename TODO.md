# TODOs (Critical → Minor)

- Critical: Resolve ONNX Runtime package conflict in requirements
  - Action: Choose either `onnxruntime` (CPU) or `onnxruntime-gpu` (GPU) and remove the other; otherwise imports are undefined due to duplicate modules.
  - Why: Having both wheels installed causes conflicts at runtime/import time.
  - Files: requirements.txt:9, requirements.txt:11

- Critical: Add missing dependency for evaluation metrics
  - Action: Add `torchmetrics>=1.0` to requirements.
  - Why: `evaluation_metrics.py` imports TorchMetrics; without it, validation/training will fail on import.
  - Files: evaluation_metrics.py:4, requirements.txt

- Critical: Guard DataLoader settings when `num_workers=0`
  - Action: Document/enforce that `prefetch_factor` must be unset and `persistent_workers` must be False when `num_workers=0` (train/val/infer).
  - Why: PyTorch raises ValueError if these are set with zero workers; easy config foot‑gun.
  - Files: dataset_loader.py:867, 883; validation_loop.py:416, 433; Inference_Engine.py:841; configs/unified_config.yaml (dataloader section)

- Critical: Fix ONNX inference helper API mismatch
  - Action: Align `_preprocess_simple` with `_preprocess` or remove it if unused.
  - Why: Current signature/reference mismatch can break callers expecting the simple path.
  - Files: onnx_infer.py:115

- Major: Plan migration to `torch.onnx.dynamo_export`
  - Action: Track/plan replacing `torch.onnx.export` with `torch.onnx.dynamo_export` while preserving dynamic shapes and validation.
  - Why: Keeps exporter future‑proof on PyTorch 2.8+; avoids deprecation churn.
  - Files: ONNX_Export.py:572

- Major: Loosen Protobuf upper bound
  - Action: Allow protobuf 6.x (e.g., `protobuf>=4.25.3`) instead of `<6`.
  - Why: ONNX 1.19 works with protobuf 6; current cap can cause ecosystem conflicts in 2025.
  - Files: requirements.txt:21

- Major: Document optional ORT transformer optimizer dependency
  - Action: Note in README/export docs that `onnxruntime.transformers` optimizer is optional and how to install it when desired.
  - Why: Avoids confusion when optimizer is unavailable and fallback kicks in.
  - Files: ONNX_Export.py:665, README.md

- Major: Orientation mapping coverage audit
  - Action: Run `OrientationHandler.validate_mappings()` and dataset‑level checks to identify unmapped left/right tags; extend `configs/orientation_map.json` where safe.
  - Why: Reduces label noise or augmentation loss; improves flip safety/coverage.
  - Files: orientation_handler.py; configs/orientation_map.json; configs/orientation_map.README.md

- Minor: Modernize Pillow resampling constants
  - Action: Use `Image.Resampling.BILINEAR` instead of `Image.BILINEAR`.
  - Why: Future‑proof against deprecations; functionally identical today.
  - Files: dataset_loader.py:202, 604

- Minor: Update config docs referencing `export_config.yaml`
  - Action: Ensure `configs/README.md` matches reality (unified export section vs separate file) or add the referenced file.
  - Why: Avoids onboarding/doc confusion.
  - Files: configs/README.md, configs/unified_config.yaml

- Minor: Verify LR scheduler semantics
  - Action: Confirm `CosineAnnealingWarmupRestarts` stepping (per‑epoch) and `first_cycle_steps=num_epochs` match intended schedule/warmup.
  - Why: Prevents accidental LR shape mismatches across long runs.
  - Files: train_direct.py:609, training_utils.py (scheduler)

- Minor: Clarify default ONNXRuntime providers in docs
  - Action: Document GPU/CPU provider selection and expected fallbacks when GPU build is not present.
  - Why: Reduces user confusion during ONNX inference environment setup.
  - Files: onnx_infer.py:155, README.md

