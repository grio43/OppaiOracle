# Deprecated Candidates

This list is for in-depth review of pipeline-related code that appears unused.
Evidence is based on repo-wide `rg` searches and a static scan for module-level
definitions with no call sites. Dynamic imports, external scripts, and manual
CLI usage may not show up here.

## Standalone utilities or entry points with no importers
- `training_utils.py` __main__ self-test block near end of file (only runs when executed directly)

## Unused helpers inside pipeline modules (no call sites)
- `dataset_loader.py:1737` _cleanup_all_validators (no call sites; no atexit hook)
- `training_utils.py:621` log_index_order_hash; `training_utils.py:950` EarlyStopping; `training_utils.py:2240` LearningRateSchedulerFactory; `training_utils.py:2341` MixedPrecisionTrainer; `training_utils.py:2438` TrainingMetricsTracker (only referenced in the self-test block)

Suggested review approach:
1) Confirm whether any of the above are called by external scripts or manual workflows.
2) If not, remove or move to a `tools/` or `examples/` area with explicit docs.
3) If intended for production, wire into the pipeline and add minimal validation.
