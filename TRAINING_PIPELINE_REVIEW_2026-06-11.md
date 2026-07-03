# Training Pipeline Review — 2026-06-11

Scope: data pipeline (`dataset_loader.py`, `utils/`, `vocabulary.py`), training loop (`train_direct.py`, `training_utils.py`, `schedulers.py`), model/loss/metrics (`model_architecture.py`, `loss_functions.py`, `evaluation_metrics.py`, `adan_optimizer.py`), and config/validation (`Configuration_System.py`, `configs/unified_config.yaml`, `validation_loop.py`, `training_config.py`). Four independent review passes, findings verified against code; the Critical finding was re-verified by hand.

Known/triaged items were **not** re-reported: mp.Value flip-epoch under spawn (correct), Arrow cache mid-run staleness (WONTFIX by design), fixed WD=0.05 (intentional), `compute_effective_batch_size`/world_size (documented in AGENTS.md), early-stop on f1_macro vs mAP (known recommendation).

---

## TL;DR

1. **CRITICAL: validation leaks into training.** With sidecar mode + the combined Arrow cache + `num_workers > 0` (the live config), DataLoader workers silently reload the *unfiltered* train+val Arrow table, so the val set served is a subset of the rows the train loader serves. Val metrics are measured on trained-on images; early stopping and the health tracker have been reading inflated numbers. Fix before the next run. (Finding 1)
2. **HIGH: a real (if rare) GPU data-corruption race** — the H2D side-stream prefetch never calls `record_stream()`. (Finding 2)
3. **HIGH: every documented CLI override (`--training.learning_rate`, …) is silently ignored** — `load_config` is called without `args`. (Finding 3)
4. The biggest compute wins: stop pickling ~5–6M `Path` objects into every worker (Finding 4), fix the per-microbatch `global_step` gating (9× duplicate logging/syncs, Finding 10), stop double-deep-copying + synchronously writing `last.pt` (Finding 12), make the torch.compile warmup compile the *training* graph instead of an inference graph (Finding 13), and stop accumulating ~7 GB of validation tensors (int64 targets, unconditional, sometimes never read) (Finding 11).

---

## Critical

### 1. Train/val split aliasing: workers load the unfiltered combined Arrow table (train/val leakage)
**Category:** Bug · **Files:** `dataset_loader.py:825-845` (`ArrowMetadataAccessor.__getstate__`/`__setstate__`), `:783-801` (`_ensure_table`), `:2027-2068` (split/exclusion filtering), `:2833-2850` (combined cache build)

`create_dataloaders` builds **one** Arrow cache from `train_list + val_list` and each `SidecarJsonDataset` filters the table to its split in the main process (`arrow_table = arrow_table.filter(mask)`, line 2040). But the accessor pickles only the cache path and the (filtered) length:

```python
def __getstate__(self):
    return {"_cache_path": self._cache_path, "_len": self._len}
```

In each worker, `_ensure_table()` reloads the **full combined cache from disk** and `__getitem__` does `self._table.slice(idx, 1)` — so index `idx` now addresses the unfiltered train+val table. Consequences with the live config (`num_workers: 6`, `persistent_workers: true`):

- The **train** dataset serves rows `0..N_train-1` of the combined table — an arbitrary mix of intended train and val files (row order is additionally nondeterministic: `_build_arrow_cache` appends in `as_completed()` order, `utils/metadata_cache.py:321-324`).
- The **val** dataset serves rows `0..N_val-1` of the *same* table — a strict subset of what the train loader serves. **Validation evaluates on trained-on images**, and ~95% of the intended held-out files are trained on.
- The exclusion filter (lines 2043-2056) is likewise lost in workers; the last M legitimate rows of the table are never served because `_len` was shrunk by filtering.

Each row is internally self-consistent, so nothing crashes or logs — training "looks" healthy. With `num_workers=0` the in-process filtered table is used and behavior is correct, which is why tests wouldn't catch it.

**Fix:** make the filter survive pickling — store selected row indices (`pc.indices_nonzero(mask)`, composed with the exclusion mask) in `__getstate__` and apply `table.take(indices)` in `_ensure_table`; or write a per-split Arrow file and point the accessor at it.

**Follow-up once fixed:** past val/f1 and val/mAP rows in `TRAINING_HEALTH_TRACKER.md` are optimistically biased and not comparable to post-fix numbers; re-baseline.

---

## High

### 2. Missing `record_stream()` on side-stream H2D transfers — latent data-corruption race
**Category:** Bug · **Files:** `train_direct.py:1458-1469` (train), `:2123-2133` (val)

Inputs are copied inside `with h2d_ctx:` on a side stream, then consumed on the default stream after `wait_stream(h2d_stream)`. No `record_stream()` exists anywhere in the repo, so the caching allocator considers those blocks owned by the side stream only; when `images`/`tag_labels` are rebound next iteration, the allocator can reuse the memory for the next `.to()` on the side stream **while backward on the default stream still reads the activations**. The accumulation-boundary sync (`torch.isfinite(grad_norm)`) covers only 1 of every 9 microbatches. This is the classic prefetcher bug (Apex prefetcher fixes it with `record_stream`).

**Fix:** after `wait_stream`, call `t.record_stream(torch.cuda.current_stream())` for `images`, `tag_labels`, `pmask` (train and val) — or drop the side stream entirely: the code issues the copy and immediately waits on it, so it gains almost no overlap anyway.

### 3. All CLI config overrides are silently ignored
**Category:** Bug · **Files:** `train_direct.py:2461-2466` vs `Configuration_System.py:2132-2179, 2182-2229`

`main()` builds the full parser (defining `--training.learning_rate`, `--data.batch_size`, etc.) but calls `config = load_config(args.config)` without passing `args`, so `update_from_args` never runs. Every override documented in AGENTS.md parses successfully and is discarded; the run proceeds on YAML values with no warning.

**Fix:** `load_config(args.config, args=args)`.

### 4. ~5–6M `Path` objects pickled into every DataLoader worker
**Category:** Compute · **Files:** `dataset_loader.py:1855`, `:2114-2134`

`self.json_files = list(json_files)` holds the full split file list, and `SidecarJsonDataset.__getstate__` doesn't drop it — so it is pickled to each of the 6 spawn workers: on the order of a GB of redundant RAM per worker plus tens of seconds of spawn-time pickling, defeating the path-only pickling the Arrow accessor was built for ("~15 GB per worker" savings per its docstring). `json_files` is only used in `__init__`, never in `__getitem__`.

**Fix:** `state['json_files'] = []` in `__getstate__`.

### 5. `ThresholdCalibrator` runs ~800K Python loop iterations per validation epoch
**Category:** Compute · **Files:** `evaluation_metrics.py:575-585, 625-641`

`_calibrate_per_tag` / `_calibrate_per_bucket` nest a Python loop over ~41 thresholds × ~19K tags, each iteration doing several O(N) numpy ops; `_calibrate_per_bucket` also recomputes `support` inside the threshold loop although it's invariant in `t`. When calibration is enabled this runs inside the epoch loop (`train_direct.py:2246`).

**Fix:** vectorize per threshold over the full matrix (`pred_bin = preds > t` once → `tp/fp/fn` via column sums → `argmax` over the (41, C) F1 grid); hoist support out of the loop.

---

## Medium — bugs

### 6. Periodic-save resume position off by one (one batch retrained on resume)
`train_direct.py:1673-1674`. The `save_steps` checkpoint records `batch_in_epoch = step` *after* batch `step` was consumed; the soft-stop and one-shot save paths both correctly use `step + 1`. Resuming from a periodic checkpoint reprocesses one batch and shifts all subsequent accumulation-window boundaries by one. **Fix:** use `step + 1`.

### 7. Threshold calibration silently disabled when TensorBoard is off
`train_direct.py:2208, 2237-2261`. The `ThresholdCalibrator` block is nested inside `if all_val_probs and config.training.use_tensorboard:` — an unrelated logging flag gates a training artifact (`calibrated_thresholds`) used at inference. **Fix:** hoist calibration out of the TB gate.

### 8. ASL clip margin not applied to the negative log term (deviates from the ASL paper)
`loss_functions.py:278, 288-291`. The clip shifts only the focal *weight* (`probs_neg = (probs - clip).clamp(min=0)`), while the BCE term uses unshifted probabilities. Official ASL uses `log(1 − p + m)` for negatives; at p=0.99 this implementation yields ≈ −4.6 vs the paper's ≈ −1.56 — substantially harsher on hard negatives. The YAML sets `clip: 0.2` "per ASL paper", so the calibration assumption doesn't match the code; combined with γ_neg=7 this compounds the known missing-positive risk. **Fix:** shift the negative log term (`log((1 − p + m).clamp(...))`), or document the deviation and recalibrate `m`.

### 9. Hard logit clamp at ±15 zeroes gradients for confidently-wrong predictions
`model_architecture.py:676-678`. `torch.clamp(tag_logits, -15, 15)` in the training forward has zero gradient outside the bounds, so a target=1 sample with raw logit −20 gets **no head gradient** — exactly the samples that most need correction (and missing-positive recovery). BCE-with-logits is stable at any magnitude, so the clamp buys nothing during training. Tag-bias init bottoms out near −11.5, close to the edge. **Fix:** clamp only for inference/export, or use a soft clamp (`15 * tanh(logits / 15)`).

### 10. `global_step`-gated blocks fire once per *microbatch* — 9× duplicate logging, GPU syncs, and a corrupted `step_time` metric
`train_direct.py:1515, 1547-1557, 1836-1856` + `Monitor_log.py:1093-1107`. `global_step` only increments on optimizer updates, but `global_step % logging_steps == 0` (and the mem-monitor / NaN-check equivalents) are evaluated per microbatch — when the condition is true, all ~9 microbatches of the window satisfy it: 9 `loss.item()` syncs, 9 duplicate TB writes at the same step, 9 psutil scans. The duplicates also make `Monitor_log`'s `steps_since_last_log` = 0 (so `train/step_time` records a single-microbatch time, not the per-update average) and advance `steps_without_improvement` 9× per event, skewing the "Training Stuck" alert. The image-logging block already shows the correct pattern (`_last_image_log_step`). **Fix:** gate these on the update boundary or apply the `_last_*_step` guard. (Also note the cadence: `logging_steps: 10000` updates ≈ 0.8 epoch → only ~19 train-loss points across a 15-epoch run.)

### 11. Frequency-bucket macro metrics include zero-support tags
`evaluation_metrics.py:459-469`. `compute_bucketed_metrics` runs macro F1/mAP over all bucket columns without dropping classes with zero positives in the val draw — rare buckets are dragged toward 0 and measure draw sparsity, not model quality, inconsistent with the headline metrics (which filter via `_drop_zero_positive_classes` / `keep_classes`). **Fix:** drop zero-support columns per bucket before macro calls; report `num_supported_tags`.

### 12. Env-var overrides: scientific notation becomes a string; env path bypasses type coercion
`Configuration_System.py:1945-1962, 2013-2034`. `_parse_env_value` only tries `float()` when `'.' in value`, so `ANIME_TAGGER_TRAINING__LEARNING_RATE=1e-5` lands as the **string** `"1e-5"`, applied via raw `setattr` with no dataclass coercion. For `learning_rate` it crashes later with a bare `TypeError`; for fields `validate()` never compares (`lr_end`, `adam_epsilon`) the string silently reaches optimizer/scheduler construction. **Fix:** try `int` then `float` unconditionally; route env updates through the same coercion as `from_dict`.

### 13. `validation_loop.py` reads raw YAML and inverts the image-size priority
`validation_loop.py:190-298` vs `Configuration_System.py:1714-1719`. The standalone runner uses `yaml.safe_load` (no `FullConfig.validate()`, so no image-size sync) and gives `validation.preprocessing.image_size` priority over `data.image_size`. Editing only `data.image_size` (as documented) trains at the new size while standalone eval silently runs at the stale 448 — rescued only when the checkpoint embeds preprocessing params. Related: it clobbers an explicit `--batch-size` CLI arg with the YAML value (`validation_loop.py:225-228`, CLI > file precedence inverted), omits `data.pad_color` when building its data config (falls back to (114,114,114)), and its `max_samples` rebuild drops `prefetch_factor`/`persistent_workers`/`worker_init_fn`. **Fix:** load via `load_config()`/`FullConfig`; apply YAML dataloader values only when CLI args weren't explicitly provided; pass `pad_color` through.

### 14. Three uncoordinated sources of the prediction threshold
`train_direct.py:1321-1331, 2213-2225`; `validation_loop.py:108`; `unified_config.yaml:336, 354`. In-train F1 uses `threshold_calibration.default_threshold`; bucketed metrics use `inference.prediction_threshold`; standalone validation hard-codes `0.2653` with no CLI flag or YAML read. All equal 0.2653 today, but changing one silently desynchronizes the others. **Fix:** single source (`inference.prediction_threshold`), validated for equality in `FullConfig.validate()`; add a CLI flag to validation_loop.

---

## Medium — compute

### 15. Validation accumulates full (N×19K) matrices twice — GPU and CPU — with int64 targets, sometimes never read
`train_direct.py:1332, 2143-2154, 2208`. Two compounding issues per validation pass at 30K × ~19.3K labels:
- `MultilabelAveragePrecision(average=None)` with default `thresholds=None` retains every update's preds/targets **on GPU** (~2.3 GB fp32 + ~4.6 GB int64) and concatenates at `compute()`.
- The loop *also* appends full CPU copies (`all_val_probs` / `all_val_targs`), with `targs = tag_labels.long()` (int64, 8 bytes for a {0,1} value) — another ~7 GB host RAM — **unconditionally**, while the consumer is gated on `use_tensorboard`. With TB off it's transferred, held, and discarded.

**Fix:** pass `thresholds=<int>` to the AP metric (binned mode, constant memory); store CPU targets as `bool`/`uint8` (probs fp16); skip CPU accumulation when neither TB nor calibration needs it.

### 16. `last.pt` saved synchronously with a second full deep copy — async writer largely defeated
`training_utils.py:1507, 1548-1562`. Every `save_checkpoint()` runs `_deep_to_cpu(checkpoint)` **twice** (once for the async numbered save, once for `last.pt`) and writes `last.pt` synchronously on the training thread: two ~3 GB D2H copies plus a ~3 GB blocking `torch.save` per periodic save — the 30–90 s stall the `AsyncCheckpointWriter` exists to eliminate. **Fix:** deep-copy once and reuse; write the numbered file, then produce `last.pt` via `os.replace`/hardlink of the just-written file (keeps crash safety).

### 17. torch.compile warmup compiles the wrong graph
`train_direct.py:1261-1288`. The warmup forward runs under `torch.no_grad()` with no autocast, so Dynamo compiles an *inference* graph (it guards on grad mode and autocast state); the first real batch triggers a full recompile anyway. The 2–5 min "overlapped" compile is additive startup cost. **Fix:** run warmup forward+backward under `amp_autocast()` with grad enabled (then `zero_grad(set_to_none=True)`), or delete the block.

### 18. Full-dataset `rglob` walk on every startup inside the split-cache "fast path"
`dataset_loader.py:530-547`. `_try_load_cached_split` validates the cache by re-counting the filesystem (`sum(1 for jp in root.rglob("*.json") ...)`) — minutes of cold-cache NTFS I/O per launch for ~5.6M files, costing nearly as much as the scan the cache avoids. **Fix:** sampled existence/new-file probe (the Arrow cache's mtime-sampled staleness check is the model), or age-gate the full count.

### 19. Every sample materializes its Arrow metadata row twice
`dataset_loader.py:2329-2334, 2358`. `__getitem__` fetches `self.items[idx]` once for the exclusion check and again for the annotation — each fetch is a `slice` + full-column `as_py()` including the tags list, doubling per-sample metadata decode every epoch. **Fix:** fetch once, reuse `ann["image_id"]`.

---

## Low

Bugs (latent / edge / cosmetic):

- **L1.** Manifest-mode `DatasetLoader` is unpicklable (lambda `defaultdict` + `ExclusionManager` lock) → crashes with `num_workers > 0` under spawn (`dataset_loader.py:1000, 1009-1012, 1078-1083`). Legacy path only.
- **L2.** Manifest-mode error samples add `flip_applied`/`flip_mode` keys success samples lack → `default_collate` KeyError when an error sample is first in a batch (`dataset_loader.py:1180-1188` vs `:1573-1583`).
- **L3.** Sidecar torchvision-v1 fallback yields float32 images but bf16 error samples → mixed-dtype `torch.stack` crash in that environment (`dataset_loader.py:1893, 1937-1939, 2572-2577`).
- **L4.** Soft stop requested mid-accumulation in the **final** epoch exits without flushing or saving (`train_direct.py:1945, 2004-2012`).
- **L5.** `CosineAnnealingWarmupRestarts` warmup off-by-one: constructor's implicit `step()` makes the first served LR warmup-step 1, and cycle 0 differs from post-restart cycles by one step (`training_utils.py:754, 808-810`).
- **L6.** GradScaler state saved but never passed to `load_checkpoint` on resume — silent scale reset if fp16 is ever enabled (`train_direct.py:1098-1109` vs `training_utils.py:2082-2088`).
- **L7.** Pre-backward NaN path calls `scaler.update()` with no recorded inf-checks — would assert on the only configuration it was written for (enabled fp16 scaler) (`train_direct.py:1530`).
- **L8.** `training_state.val_f1_macro`/`val_mAP` set as non-field attrs on the dataclass → dropped by `asdict`, resume's validation-skip branch reports 0.0; `from_dict(cls(**data))` will `TypeError` on legacy checkpoints with removed keys (`train_direct.py:2290-2291` vs `training_utils.py:707-711`).
- **L9.** `sampler_state` saved into every checkpoint but `ResumableSampler.load_state()` never called — its dataset-size-change guard is dead; resume after a dataset size change applies a stale offset silently (`training_utils.py:1372-1383` vs `train_direct.py:1400-1414`).
- **L10.** Non-finite-grad-norm `continue` skips the boundary microbatch's loss accounting and `del loss_detached` (`train_direct.py:1627-1636, 1699-1705`). Reporting only.
- **L11.** Unused `EarlyStopping` class uses `np.Inf` — `AttributeError` under NumPy ≥ 2.0 at construction; `MixedPrecisionTrainer.train_step` returns `loss.item()` per step. Both dead code — delete rather than adopt (`training_utils.py:853, 904`).
- **L12.** No 2D pos-embed interpolation in forward: smaller-than-trained inputs silently add the raster-order *prefix* of the 28×28 grid (`model_architecture.py:557`). Trap for multi-resolution eval.
- **L13.** `SafeDropPath` draws its uniform in bf16 — keep-prob quantization of a few tenths of a percent (`custom_drop_path.py:18`).
- **L14.** `MetricComputer` docstring claims default threshold 0.5; field is 0.2653 (`evaluation_metrics.py:41-43` vs `:72`).
- **L15.** `TrainingConfig.num_cycles: float = 0.5` (HF fraction semantics) vs trainer's `int(...)` cycle count — `int(0.5) == 0` if the YAML key is removed (`Configuration_System.py:1203` vs `train_direct.py:856-861`).
- **L16.** Debug sub-flags (`log_gradient_norm`, `log_activation_stats`, `log_input_stats`) run even with `debug.enabled: false`, contradicting `validate()`'s own warning (`Configuration_System.py:1631-1638` vs `train_direct.py:1475, 1489, 1603`).
- **L17.** `FullConfig.validate()` skips the `validation` and `threshold_calibration` sections entirely (`Configuration_System.py:1687`).
- **L18.** `config.model.num_labels = num_tags` at runtime makes any later `config.validate()` raise (grouped-head invariant 20×10000) (`train_direct.py:598-599` vs `Configuration_System.py:804-809`).
- **L19.** `validation.frequency_bins` is consumed via `getattr` but doesn't exist on `ValidationConfig` — the YAML knob is silently dropped (`train_direct.py:2216`).

Compute (small wins / dead weight):

- **L20.** ASL computes the positive focal log/exp over the full (B×19K) tensor although `gamma_pos=0` makes it identically 1 — short-circuit it (`loss_functions.py:305-314`).
- **L21.** `attn_kpm.any()` forces a GPU→CPU sync every forward; with letterboxed data it's almost always True anyway (`model_architecture.py:630`).
- **L22.** `compute_per_tag_metrics` makes 4 `.item()` calls per tag (~77K syncs); use `.cpu().tolist()` per vector (`evaluation_metrics.py:357-364`).
- **L23.** Dead `metric_computer` constructed in train_direct and never used (`train_direct.py:610-613`).
- **L24.** Shared-memory vocab machinery does double work: the vocab is pickled into workers anyway, then overwritten from shm (`dataset_loader.py:2114-2134, 913-936`).
- **L25.** `BackgroundValidator` thread started but nothing ever enqueues work — dead machinery spinning on a 1 s timeout (`dataset_loader.py:1045-1047, 1635-1655`).
- **L26.** `ResumableSampler._cached_indices` keeps a never-read ~5.6M-element list alive all run (`dataset_loader.py:378-385`).
- **L27.** Known-bad samples (empty tags / unknown rating) re-discovered every epoch and emitted as 1.2 MB zero tensors that are collated, pinned, then filtered out — filter them once at Arrow-table init (`dataset_loader.py:2371-2376, 2576-2586`).
- **L28.** Image extension re-discovered by filesystem probing per sample per epoch (up to 4 `exists()` + `resolve()`; whole-directory `iterdir()` on miss) although the sidecar JSON has the exact filename — store it in the Arrow cache (`utils/metadata_cache.py:197-209`, `utils/path_utils.py:96-114`).
- **L29.** Model-graph logging (`use_tensorboard and not use_compile`) consumes a throwaway `next(iter(train_loader))` — spins up all workers, discards prefetched batches, perturbs restored RNG on resume. Inactive with `use_compile: true` (`train_direct.py:1292-1303`).
- **L30.** Dead "auto-config" helpers in `training_config.py` (only `scale_learning_rate` is live) while the YAML comment claims the auto-config system adjusts WD and selects the scheduler — none of which happens. Fix the comment; prune the helpers (`unified_config.yaml:240-245`).
- **L31.** `validation.max_samples: 30000` subsampler can never fire because `data.max_val_samples: 30000` already caps the split (`30000 < 30000` is false); two knobs for one behavior, and the rebuild path would lose `worker_init_fn` if it ever fired (`train_direct.py:537-578`).
- **L32.** Stale arithmetic in tuning comments: prefetch math assumes 8 workers × batch 128 (actual 6 × 48); `checkpoint_every_n_layers: 4` comment claims "every layer/block" (`unified_config.yaml:50, 99, 391`).

---

## Verified clean (spot-summary)

- **Padding-mask polarity end-to-end:** letterbox True=PAD → `pixel_to_token_ignore` → CLS always attendable → single inversion in both Flex (`~kpm`, q&kv) and SDPA (bool True=ATTEND) paths. No double inversion; fully-padded rows can't reach the loss.
- **Gradient accumulation math:** loss ÷ accum before backward; clip on unscaled grads at boundaries only; partial-window flush at epoch end rescales by `accum/accum_count` correctly; `updates_per_epoch = ceil(steps/accum)` matches actual scheduler steps.
- **bf16 AMP:** scaler correctly disabled; loss accumulated in an fp32 GPU scalar with one `.item()` per epoch; loss/log/exp math runs fp32 under autocast.
- **Resume coverage:** RNG (Python/NumPy/torch/CUDA) fail-closed; vocab SHA256 fail-closed; scheduler `step_in_cycle`/cycle restored; optimizer state device-migrated; pos-embed interpolation across the 320→448 switch.
- **Async checkpoint writer:** no tensor-mutation race (`_deep_to_cpu` clones on the training thread before queueing).
- **Flip pipeline:** image/mask flipped on matching axes; val flip prob 0.0; CRC32 coin deterministic per (image_id, epoch); `set_epoch` driven by the loop.
- **Label encoding & PAD/UNK consistency:** unknown tags skipped (no UNK pollution); loss `ignore_indices=[0,1]`, metrics `skip_indices=[0,1]`, streaming slice `skip_metric_cols=2` — consistent everywhere.
- **ASL numerics:** focal gating captured pre-smoothing; log-space weights with 1e-6 floor and ±88 exp clamp; `(p − clip).clamp(min=0)` direction correct.
- **Adan:** moment updates, bias corrections, proximal WD, step-1 handling, and state round-trip all match the official implementation (dormant — live optimizer is adamw8bit).
- **Soft-stop machinery:** async-signal-safe, boundary-aligned, stale sentinel cleared (except L4 above).
- **YAML↔dataclass key mapping:** every key in the shipped `unified_config.yaml` maps to a real field; env bool parsing handles `"false"` correctly.
- Split determinism, mid-epoch resume skip-by-index, exclusion-manager locking, and the error-sample filter contract in both loops all check out.

---

## Suggested fix order

| Order | Findings | Why |
|---|---|---|
| 1 | **1** (split aliasing) | Invalidates every val metric; ~95% of the holdout is being trained on. Fix + re-baseline before any further Phase 2 epochs. |
| 2 | **2** (record_stream), **6** (resume off-by-one) | Correctness of training itself; both are small diffs. |
| 3 | **3** (CLI overrides), **7** (calibration behind TB gate), **12** (env coercion) | Silent config/artifact loss. |
| 4 | **4**, **15**, **16**, **17**, **10**, **18**, **19** | The compute wins: worker spawn cost, validation memory, checkpoint stalls, compile warmup, duplicate logging, startup walk, double row fetch. |
| 5 | **8**, **9**, **11** | Loss/metric quality — coordinate with the planned label-noise work (γ_neg/SPLC decisions) since 8 and 9 both bear on missing-positive handling. |
| 6 | Low items | Opportunistic; L4/L5/L8/L9 next time the checkpoint code is open. |
