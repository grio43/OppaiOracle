Orientation Mapping: Bugs, Gaps, and Worklist

Summary
- Orientation-aware flipping is implemented via `OrientationHandler` and wired into sidecar JSON dataloading. Several correctness bugs, unused knobs, and integration gaps exist. This document lists concrete issues and actionable tasks to address them.

Critical Bugs
- Double safety check inflates stats and adds overhead
  - `dataset_loader.py:855` calls `orientation_handler.should_skip_flip(tags)` inside `_decide_flip_mode`, then `__getitem__` calls `orientation_handler.swap_tags(...)` which calls `should_skip_flip` again (`orientation_handler.py:349`). This double-counts `images_analyzed`/`safe_flips` and can skip flips twice.
  - Action: Ensure flip-safety is checked once per sample. Options: pass a flag to `swap_tags` to skip internal check, or have `__getitem__` call `swap_tag` directly when a prior decision is made.

- Unmapped-tag false positives in `swap_tag`
  - `orientation_handler.py:297`: Tracks `unmapped_tags` for any tag containing "left"/"right", but unlike `can_safely_flip` (`orientation_handler.py:175`) it does not filter substrings like "bright", "upright", "copyright", etc. This overstates unmapped tags and blocks flips unnecessarily under conservative modes.
  - Action: Mirror `can_safely_flip` filters in `swap_tag` unmapped tracking.

- `skip_unmapped` config is ignored
  - `orientation_handler.py:36`/`orientation_handler.py:51`: The constructor stores `skip_unmapped`, but flip decision paths use only `safety_mode` and mapping checks. Behavior does not change with `skip_unmapped=true`.
  - Action: Define and implement semantics (e.g., in balanced mode, if `skip_unmapped` true, force skip when a swap would be no-op) or remove the knob to avoid confusion.

- TTA horizontal flip does not swap logits
  - `Inference_Engine.py:492`: With `tta_flip=true`, outputs are averaged between original and flipped inputs without swapping left/right indices for the flipped pass, corrupting directional tags.
  - Action: Precompute index mapping from vocab using `OrientationHandler.precompute_all_mappings` (`orientation_handler.py:374`) and apply it to the flipped logits before averaging.

Behavioral Gaps
- Manifest mode silently disables orientation flipping
  - `DatasetLoader` (manifest mode) explicitly sets flip bit false and has no orientation handler; only `SidecarJsonDataset` (sidecar mode) performs flips.
  - References: `dataset_loader.py:232` (manifest dataset comment), `dataset_loader.py:988` (flip only in sidecar path).
  - Action: Either (a) add flipping/mapping support to manifest mode, or (b) document clearly that manifest mode is non-orientation-aware and enforce with a warning when `random_flip_prob>0`.

- No telemetry wiring from handler to monitor
  - `Monitor_log.py:1276` expects keys like `flip_total`, `flip_safe`, `flip_skipped_text`, `flip_skipped_unmapped`, `flip_blocked_safety`. `OrientationHandler` tracks `total_flips`, `skipped_flips`, `blocked_by_text`, `blocked_by_safety`, etc., but nothing publishes these to the `stats_queue` (`train_direct.py:339`, `train_direct.py:731`).
  - Action: Add an adapter in the dataloader loop (or a small sidecar callback) to publish flip stats at intervals to `stats_queue` with Monitor-compatible keys.

- Redundant `setup_orientation_aware_training`
  - `train_direct.py:146` returns `dataloader_overrides` that are unused by `create_dataloaders` (which reads from `config.data` directly). This can drift and confuse.
  - Action: Remove or repurpose to perform early validation only; avoid returning unused overrides.

Orphan / Unused Code
- `OrientationMonitor` unused
  - Defined at `orientation_handler.py:688`, writes `unmapped_orientation_tags.txt`, but never instantiated.
  - Action: Either wire it into training (periodic checks + file output), or remove to reduce maintenance.

- `handle_complex_tags` unused
  - `orientation_handler.py:408` handles heterochromia and other asymmetric cases but is never called; `SidecarJsonDataset` uses only `swap_tags`.
  - Action: Integrate into flip decision (call before `swap_tags`) or document as future work and remove from handler proper (move to a helpers module).

- CI validator stubs
  - `train_direct.py:876` `validate_orientation_mappings()` is a no-op; `orientation_handler.py:866` has a functional dataset-tags validator not wired in.
  - Action: Hook validation into a preflight command or `Configuration_System.py validate` path.

Data/Mapping Hygiene
- Duplicate explicit mapping key
  - `configs/orientation_map.json:~60`: `bandage_over_right_eye` appears twice. JSON keeps the last occurrence, but duplicates hinder review.
  - Action: Deduplicate keys and add a lint step that flags duplicates before training.

- Mapping validation not enforced
  - `orientation_handler.validate_mappings()` can detect bidirectional/circular/regex conflicts but isn’t invoked during config validation.
  - Action: Run `validate_mappings()` at startup when `strict_orientation_validation=true` and fail fast on critical issues.

Performance / Quality
- Regex cost per-tag per-sample
  - `swap_tag` applies regex patterns tag-by-tag per sample. For large vocabularies this adds overhead.
  - Action: Use `precompute_all_mappings(known_vocabulary)` (`orientation_handler.py:374`) once from the vocabulary to build an index map; rely on O(1) lookups during training.

- Better skip reason logging
  - `OrientationHandler.can_safely_flip` returns reasons; dataset discards them.
  - References: `orientation_handler.py:175`, `dataset_loader.py:855` and `dataset_loader.py:889`.
  - Action: Capture top reasons and include in monitor metrics (e.g., count of "text/watermark" vs "unmapped" blocks).

Concrete Work Items (prioritized)
1) Single-check flip decision
   - Update `SidecarJsonDataset.__getitem__` (`dataset_loader.py:875`) to avoid a second `should_skip_flip` call after `_decide_flip_mode`.
   - Option: add `swap_tags(tags, already_decided=True)` or a `skip_safety_check` flag.

2) Fix unmapped tracking false positives
   - Apply the same filters from `can_safely_flip` to `swap_tag` before adding to `unmapped_tags` (`orientation_handler.py:297`).

3) Implement `skip_unmapped` semantics or remove
   - If kept: when `skip_unmapped=true` and `safety_mode` is `balanced`, skip flipping if any orientation tag maps to itself (no-op).
   - Wire into `can_safely_flip` and/or `_decide_flip_mode`.

4) Correct TTA flip averaging
   - Precompute index mapping from vocabulary using `OrientationHandler.precompute_all_mappings` (`orientation_handler.py:374`).
   - In `Inference_Engine.ModelWrapper.predict` (`Inference_Engine.py:480`), when `tta_flip`, apply index remap to `outputs_flipped` before averaging.

5) Publish flip stats to monitor
   - Add a small publisher (e.g., in `SidecarJsonDataset` every N items) to emit a dict with keys expected by `Monitor_log` (`Monitor_log.py:1276`).
   - Keys: `flip_total`, `flip_safe`, `flip_skipped_text`, `flip_skipped_unmapped`, `flip_blocked_safety`.
   - Source them from `OrientationHandler.get_statistics()` (`orientation_handler.py:547`) with a mapping.

6) Enforce mapping validation on startup
   - When `strict_orientation_validation=true`, run `validate_mappings()` after loading `orientation_map.json` and fail on critical issues (`train_direct.py:202`).
   - Also add a config-time lint that flags duplicate keys in JSON.

7) Manifest mode decision
   - Either implement orientation flipping for `DatasetLoader` (manifest mode) or issue a warning and force `random_flip_prob=0` in that mode.

8) Remove/repurpose `setup_orientation_aware_training`
   - `train_direct.py:146` is currently redundant; keep only validation and logging, remove unused `dataloader_overrides`.

9) Decide fate of `OrientationMonitor` and `handle_complex_tags`
   - Wire them in or remove/move to docs/tests to reduce dead code.

10) Tests and docs
   - Add unit tests for: single-check flow, unmapped filter, TTA left/right remap, manifest-vs-sidecar behavior, and monitor stat publishing.
   - Update README and `configs/orientation_map.README.md` to reflect final semantics.

Nice-to-Have
- Add a CLI to generate a left/right index map from a given vocab and mapping, saved to disk for inference.
- Expose top blocking tags (from `unmapped_blocking_frequency`) in TensorBoard as text scalars.

References (one-line anchors)
- orientation_handler.py:175
- orientation_handler.py:245
- orientation_handler.py:297
- orientation_handler.py:349
- orientation_handler.py:374
- orientation_handler.py:688
- dataset_loader.py:855
- dataset_loader.py:875
- dataset_loader.py:889
- dataset_loader.py:988
- Inference_Engine.py:480
- Inference_Engine.py:492
- Monitor_log.py:1276
- configs/unified_config.yaml:112
- configs/unified_config.yaml:130

