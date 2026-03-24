# Deprecated Candidates

This list is for in-depth review of pipeline-related code that appears unused.
Evidence is based on repo-wide `rg` searches and a static scan for module-level
definitions with no call sites. Dynamic imports, external scripts, and manual
CLI usage may not show up here.

## Standalone utilities or entry points with no importers
- `training_config.py` (only imported by `example_optimizer_configs.py`, which has no importers)
- `example_optimizer_configs.py` (no references in repo; appears to be a sample only)
- `orientation_handler.py:1185` validate_dataset_orientation_tags (no call sites; only useful if invoked directly)
- `vocabulary.py:1120` create_dataset_config and `vocabulary.py:1211` clear_vocabulary_build_cache (no call sites)
- `training_utils.py` __main__ self-test block near end of file (only runs when executed directly)

## Unused helpers inside pipeline modules (no call sites)
- `Configuration_System.py:1780` resolve_opset (not used by ONNX export flow)
- `Monitor_log.py:162` trim (no call sites)
- `model_architecture.py:31` _check_flex_attention_available (no call sites)
- `dataset_loader.py:1737` _cleanup_all_validators (no call sites; no atexit hook)
- `cache_codec.py:137` set_sidecar_cache_size; `cache_codec.py:152` get_sidecar_cache_stats; `cache_codec.py:157` clear_sidecar_cache; `cache_codec.py:210` encode_tensor; `cache_codec.py:244` decode_tensor (no call sites)
- `training_utils.py:621` log_index_order_hash; `training_utils.py:950` EarlyStopping; `training_utils.py:2240` LearningRateSchedulerFactory; `training_utils.py:2341` MixedPrecisionTrainer; `training_utils.py:2438` TrainingMetricsTracker (only referenced in the self-test block)

Suggested review approach:
1) Confirm whether any of the above are called by external scripts or manual workflows.
2) If not, remove or move to a `tools/` or `examples/` area with explicit docs.
3) If intended for production, wire into the pipeline and add minimal validation.
