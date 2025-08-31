Checkpointing Bugs To Fix

- Resume overrides best metric incorrectly: `train_direct.py:575`
  - Bug: After loading `training_state` from the checkpoint, code overwrites `training_state.best_metric` with the current checkpoint's `metrics['val_f1_macro']`. This loses the historical best stored in `training_state` and can break early stopping and best-model detection after resume.
  - Needs fix: Do not overwrite `training_state.best_metric`; rely on the value inside loaded `training_state` or only update if the loaded metrics are explicitly marked as best.

- `save_best_only` semantics are wrong/non-functional for retention: `training_utils.py:622`, `train_direct.py:832`
  - Bug: Config field `training.save_best_only` is wired to `keep_best` (copying a `best_model.pt`) but does not suppress periodic checkpoint saves. Despite the name, non-best checkpoints are still written at cadence, so "best only" is not enforced.
  - Needs fix: Either (a) implement true best-only retention (skip cadence saves unless `is_best`), or (b) rename config to reflect actual behavior (e.g., `keep_best_copy`) and add a separate switch for best-only saving.

- Unused parameters/fields in `CheckpointManager`: `training_utils.py:439`, `training_utils.py:455`
  - Bug: `save_frequency` and `best_metric` are defined but never used. This confuses behavior and maintenance.
  - Needs fix: Remove unused fields or wire them into logic (e.g., enforce `save_frequency`).

- Missing error handling for `InvalidCheckpointError` in convenience loaders: `training_utils.py:757`, `training_utils.py:767`
  - Bug: `load_latest_checkpoint` only catches `FileNotFoundError` and `ValueError`. `safe_load_checkpoint` can raise `InvalidCheckpointError`, which will bubble up and crash unexpectedly here.
  - Needs fix: Catch `InvalidCheckpointError` consistently in both `load_latest_checkpoint` and `load_best_checkpoint` (and log a clear warning).

- Vocabulary embedding error policy contradicts fail-fast guidance: `model_metadata.py:58-66`, `training_utils.py:540-564`
  - Bug: When placeholders are detected, `embed_vocabulary` raises then catches its own exception and returns the checkpoint without an embedded vocabulary. This silently produces non self-contained checkpoints, diverging from the documented "fail fast" behavior on corrupted vocabularies.
  - Needs fix: Propagate the error (do not swallow it) or make `save_checkpoint` enforce a hard failure when vocab embedding fails, in line with project guidance.

- Unreachable code after raise: `model_metadata.py:65`
  - Bug: `return checkpoint` sits immediately after a `raise ValueError`, making it dead code.
  - Needs fix: Remove the unreachable return.

- Latest checkpoint selection inconsistency: `training_utils.py:739-741` vs. `training_utils.py:753-756`
  - Bug: `get_latest_checkpoint` returns the last in-memory path, which may be stale if files change on disk. The loader path computes latest by mtime. This inconsistency can cause resume to pick an older file.
  - Needs fix: Either refresh/sort by mtime in `get_latest_checkpoint` or have callers use the mtime-based loader consistently.

