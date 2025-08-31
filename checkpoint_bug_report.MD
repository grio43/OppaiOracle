Checkpointing Bugs To Fix

- validation_loop uses num_tags before init: In `validation_loop.py:343`, `_load_model()` sets `model_config['num_tags'] = self.num_tags` before `self.num_tags` is defined (set later in `__init__` after vocabulary load at `validation_loop.py:256`). This can raise an AttributeError and/or build a model with the wrong head size.
  - Impact: Validation can crash or load an incorrectly sized head when only a checkpoint is provided.
  - Evidence: `validation_loop.py:343`, `validation_loop.py:256`.
  - Fix direction (for code owners): Load vocabulary (or infer tags from checkpoint `state_dict`) before creating the model, or read `num_tags/num_classes` from checkpoint metadata/state.

- Config key mismatch for model config in checkpoints: Consumers expect `checkpoint['config']['model_config']`, but training saves `config.to_dict()` with a `model` section, not `model_config`.
  - Impact: `validation_loop.py` falls back to defaults (`VisionTransformerConfig`) when exporting/validating with a checkpoint that lacks `config.model_config`, risking shape mismatches or subtle behavior drift.
  - Evidence: reader paths at `validation_loop.py:330-336`, `ONNX_Export.py:471-477`; writer at `training_utils.py:516-520` sets `checkpoint['config'] = config` (which contains `model`, not `model_config`).
  - Fix direction: Either embed a `model_config` block alongside `config`, or update consumers to read from `config['model']` when `model_config` is absent.

- Lightning checkpoints are incompatible with safe loader: `train_lightning.py` uses PyTorch Lightning `.ckpt` files (pickled objects), while the rest of the stack expects plain `.pt` state_dict checkpoints via `safe_checkpoint.safe_load_checkpoint`.
  - Impact: `.ckpt` files are not consumable by `validation_loop.py` and other tools that intentionally reject pickled models.
  - Evidence: Lightning save/resume at `train_lightning.py:103-131`; strict rejection in `validation_loop.py:384-388`.
  - Fix direction: Document format differences clearly and/or add a converter path to extract a plain state_dict from Lightning checkpoints.

- GradScaler state not persisted: Training saves optimizer/scheduler states but not AMP GradScaler state when using fp16 (`train_direct.py`).
  - Impact: On resume with fp16, scaler restarts from defaults; usually minor but can transiently affect stability.
  - Evidence: Save code in `training_utils.py:557-562` lacks scaler; GradScaler used in `train_direct.py:506-519`, but no scaler state is attached to the checkpoint.
  - Fix direction: Persist and restore scaler state in checkpoints when fp16 is active.

- Minor security gap on older PyTorch: `safe_checkpoint.safe_load_checkpoint` falls back to `torch.load(..., weights_only=False)` when `weights_only` is unavailable.
  - Impact: On legacy PyTorch, unpickling occurs before structural validation, which is less safe than weights-only loads.
  - Evidence: `safe_checkpoint.py:33-38`.
  - Fix direction: Consider guarding with a restricted pickle module or requiring newer PyTorch for critical paths.

Notes
- The following prior concerns appear resolved: atomic writes for `last.pt` and numbered checkpoints (`training_utils.py:613-639`), metadata embedding for vocabulary and preprocessing (`training_utils.py:540-606`, `model_metadata.py:24-170`), and resume policy in `train_direct.py:537-584`.
