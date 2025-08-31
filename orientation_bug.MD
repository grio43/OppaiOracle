Orientation Mapping: Bugs, Gaps, and Worklist

Summary
- Orientation-aware flipping is implemented via `OrientationHandler` and wired into sidecar JSON dataloading. Several items have been fixed; remaining issues and new findings are tracked here with actionable tasks.

Critical Bugs
- Telemetry payload shape mismatch
  - `train_direct.py` expects `(stat_type, stats_dict)` from `stats_queue` and unpacks `(stat_type, stats_data)`, but `SidecarJsonDataset` enqueues a bare dict. This triggers queue processing warnings and drops augmentation stats.
  - Action: Enqueue as `('aug_stats', payload)` instead of just `payload`.

- Mislabelled `flip_skipped_unmapped`
  - Dataloader maps `flip_skipped_unmapped` to `OrientationHandler.stats['skipped_flips']`, which includes both text- and safety-based skips. This double counts with `flip_skipped_text` and misrepresents unmapped-only skips.
  - Action: Map `flip_skipped_unmapped` to `blocked_by_safety` and keep `flip_skipped_text` from `blocked_by_text`.

- Forced flips not counted in `flip_total`
  - In `__getitem__` force path, tags are swapped via per-tag `swap_tag(...)` and `flip_bit=True`, but handler `total_flips` is never incremented.
  - Action: Either call `swap_tags(tags, skip_safety_check=True)` in force mode or explicitly increment `total_flips` in the force branch.

- Unmapped-tag false positives incomplete filter
  - `swap_tag` now filters `bright/upright/straight/light/fight`, but still flags `copyright`-containing tags (e.g., "copyright_notice") as unmapped and does not consider `single_`-style descriptors.
  - Action: Mirror `can_safely_flip` filters fully in `swap_tag` unmapped tracking (add `copyright`, consider `single_`).

Behavioral Gaps
- Mapping validation not integrated into config validation
  - `train_direct.py` enforces `validate_mappings()` when `strict_orientation_validation=true`, but `Configuration_System.py validate` does not run mapping checks.
  - Action: Hook `OrientationHandler.validate_mappings()` into `Configuration_System.py validate` when `data.strict_orientation_validation=true` and `random_flip_prob>0`.

Orphan / Unused Code
- `OrientationMonitor` unused
  - Defined at `orientation_handler.py:688`, writes `unmapped_orientation_tags.txt`, but never instantiated.
  - Action: Either wire it into training (periodic checks + file output), or remove to reduce maintenance.

- `handle_complex_tags` unused
  - `orientation_handler.py:408` handles heterochromia and other asymmetric cases but is never called; `SidecarJsonDataset` uses only `swap_tags`.
  - Action: Integrate into flip decision (call before `swap_tags`) or document as future work and remove from handler proper (move to a helpers module).

- CI validator wiring
  - `orientation_handler.py` exposes `validate_dataset_tags` / `validate_dataset_orientation_tags`, but config validation and training do not use dataset-tag validation.
  - Action: Provide an optional preflight that scans dataset tag vocab and runs `validate_dataset_tags` with a threshold before training.

Data/Mapping Hygiene
- Duplicate explicit mapping key
  - `configs/orientation_map.json:~60`: `bandage_over_right_eye` appears twice. JSON keeps the last occurrence, but duplicates hinder review.
  - Action: Deduplicate keys and add a lint step that flags duplicates before training.

- Mapping validation coverage
  - `validate_mappings()` is enforced at train startup when `strict_orientation_validation=true`, but duplicate‑key linting is not implemented and config validation doesn’t run mapping checks.
  - Action: Add duplicate‑key detection and integrate mapping checks into `Configuration_System.py validate`.

Performance / Quality
- Regex cost per-tag per-sample
  - `swap_tag` applies regex patterns tag-by-tag per sample. For large vocabularies this adds overhead.
  - Action: Use `precompute_all_mappings(known_vocabulary)` (`orientation_handler.py:374`) once from the vocabulary to build an index map; rely on O(1) lookups during training.

- Flip-rate semantics
  - `safe_flips` is incremented on safety decisions (even when a flip is not performed), while `total_flips` reflects performed flips. `flip_rate = flip_safe/flip_total` in `Monitor_log` can exceed 1.
  - Action: Either report a decision rate separately or change `flip_safe` to count only performed safe flips.

Concrete Work Items (prioritized)
1) Fix stats_queue payload and mappings
   - Enqueue `('aug_stats', payload)` and map fields as: `flip_total=total_flips`, `flip_safe` (performed safe flips or rename), `flip_skipped_text=blocked_by_text`, `flip_skipped_unmapped=blocked_by_safety`, `flip_blocked_safety=blocked_by_safety` (or drop one to avoid duplication).

2) Count forced flips in stats
   - In force path, call `swap_tags(..., skip_safety_check=True)` or increment `total_flips` to keep stats consistent.

3) Complete unmapped filter in `swap_tag`
   - Add `copyright` and consider `single_` to parity with `can_safely_flip`.

4) Deduplicate orientation map and add lint
   - Remove duplicate keys in `configs/orientation_map.json` and add a preflight that flags duplicates before training.

5) Integrate mapping checks into config validation
   - When `strict_orientation_validation=true` and flips are enabled, run `validate_mappings()` in `Configuration_System.py validate`.

6) Optional: precompute mapping for training
   - Build a tag→tag swap dict from the training vocabulary once and use O(1) lookups during tag encoding.

7) Decide fate of `OrientationMonitor` and `handle_complex_tags`
   - Wire in or remove/move to docs to reduce dead code.

8) Tests and docs
   - Add unit tests for telemetry shape/mapping, forced-flip counting, unmapped filter parity, and TTA remap. Update docs to reflect flip-rate semantics.

Nice-to-Have
- Add a CLI to generate a left/right index map from a given vocab and mapping, saved to disk for inference.
- Expose top blocking tags (from `unmapped_blocking_frequency`) in TensorBoard as text scalars.

References (one-line anchors)
- orientation_handler.py:175
- orientation_handler.py:245
- orientation_handler.py:343
- orientation_handler.py:358
- orientation_handler.py:380
- orientation_handler.py:688
- dataset_loader.py:860
- dataset_loader.py:896
- dataset_loader.py:1069
- Inference_Engine.py:336
- Inference_Engine.py:523
- Monitor_log.py:1276
- configs/unified_config.yaml:~170 data.orientation_map_path
