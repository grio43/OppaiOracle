# Configuration Migration Guide

## Action checklist (phase 1)
- [ ] Switch all scripts to load `configs/unified_config.yaml`.
- [ ] Add CI step: `python Configuration_System.py validate configs/unified_config.yaml`.
- [ ] Add a warning at runtime if a legacy YAML is passed directly.
- [ ] Confirm normalization values are consistent with training data choices.
- [ ] Document any env var overrides used in prod runs.

## Notes
Legacy YAMLs remain read-only references during migration. Prefer edits to `unified_config.yaml`.
