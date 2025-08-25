# Configuration Migration Guide

## Consolidated Configuration Structure

The configuration system has been unified from multiple YAML files into a single structure that leverages the `Configuration_System.py` dataclass framework.

### File Mapping

Old files → New sections in `unified_config.yaml`:

- `augmentation.yaml` → `data:` section (augmentation settings)
- `dataloader.yaml` → `data:` section (batch_size, num_workers, etc.)
- `dataset_prep.yaml` → Removed (use scripts with config overrides)
- `export_config.yaml` → `export:` section
- `inference_config.yaml` → `inference:` section
- `logging.yaml` → Top-level logging settings
- `paths.yaml` → Top-level path settings
- `runtime.yaml` → `training:` section (deterministic, benchmark)
- `train_config.yaml` → `data:` section (image_size, pad_color)
- `training_config.yaml` → `training:` section
- `validation_config.yaml` → Handled via split='val' in dataset
- `vocabulary.yaml` → `data:` section (vocab_dir)

### Migration Steps

1. **Backup existing configs**: `cp -r configs configs.backup`
2. **Use unified config**: Replace references to individual YAML files with `configs/unified_config.yaml`
3. **Update scripts**: Change config loading from multiple files to single file
4. **Validation**: Run `python Configuration_System.py validate configs/unified_config.yaml`

### Breaking Changes

- `focal_alpha_pos` and `focal_alpha_neg` are deprecated; use unified `focal_alpha`
- Dataset prep settings moved to script arguments
- Validation config merged into data config with split handling

### Benefits

- Single source of truth for all configuration
- Built-in validation via Configuration_System.py
- Type checking and auto-completion in IDEs
- Easy config diffing and versioning
- Environment variable overrides: `ANIME_TAGGER_TRAINING__LEARNING_RATE=0.001`

### Backward Compatibility

Individual YAML files are preserved for reference. The system can still load them separately if needed.
