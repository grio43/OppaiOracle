# OppaiOracle Training-Code Review Checklist

This checklist covers the **training path only** (train_direct.py and everything it touches: config, data, vocabulary, model, loss/metrics, checkpointing, optimizer/scheduler, monitoring, the standalone validation harness, and shared utils). Inference/export tooling is out of scope except where it shares a contract with training. Use it during a focused review pass: tick each box as you confirm the behavior in code, paying special attention to the triage section first. Each item carries a severity and, where a concrete defect is suspected, a `[BUG?]` tag.

_Generated: 2026-06-04_

## How to read this

- **Severity:** 🔴 high (can silently corrupt training/metrics, lose checkpoints, or crash) · 🟠 medium (drift, comparability, or latent footgun) · 🟢 low (minor, defensive, or dormant code path).
- **`[BUG?]`** marks items flagged as a *suspected concrete defect* (status `suspected-bug`) — these have a specific code-level hypothesis and should be confirmed/fixed, not just spot-checked. Items without the tag are verification questions: confirm the current behavior is intended.
- File references (`file.py:line`) point at the exact code to inspect.

---

## ✅ Verification & fix log — 2026-06-04 (high-severity triage pass)

All 35 `🔴 High [BUG?]` items were verified against the current code (35-agent verification pass + independent cross-checks, including empirical Windows-spawn and config/vocab load tests). Fixes were applied for the confirmed, low-risk ones; behavior-changing/intended-behavior items are noted. **Edits take effect on the next process launch/resume — the in-flight Phase 2 process keeps its loaded code until then.** All modified files byte-compile and the live `unified_config.yaml` + `vocabulary.json` still load and validate.

Verdict tally: **24 CONFIRMED, 6 PARTIAL, 5 REFUTED** — corrected to **23 real / 7 not-a-bug** after the H10 finding below.

### ⚠️ H10 was a FALSE POSITIVE — do NOT "fix" it
The verification agent marked H10 (`mp.Value('i',0,lock=False)` not nulled in `__getstate__`) CONFIRMED-high, reasoning that a bare `pickle.dumps` of the value fails under spawn. **Empirically refuted**: a real `DataLoader(num_workers=2, persistent_workers=True)` on Windows spawn *does* transfer the value (the spawn machinery duplicates the shared-memory handle) **and workers observe live `set_epoch()` updates** (`set_epoch(3)→[3]`, `set_epoch(7)→[7]`). The proposed fix (rehydrate a fresh `mp.Value` per worker) would have **disconnected the shared memory and frozen the flip epoch** — i.e. it would have *introduced* the H9 bug. H9 (frozen flips) is correctly **REFUTED**; the flip-epoch mechanism works as designed.

### Fixed (verified, applied)
| ID | Bug | File(s) | Note |
|---|---|---|---|
| H1 | Partial-window accum flush under-weights grads | train_direct.py | rescale grads ×`accum/accum_count` before unscale_ |
| H2 | Phase1→2 resume silently restarts (image_size critical) | training_utils.py | `data.image_size` moved critical→important (pos_embed interpolates); `patch_size` stays critical |
| H3 | Early-stop patience advances on validation-skipped epochs | train_direct.py | best/patience updates gated on `should_validate` |
| H6 | Patch divisibility checked pre-sync | Configuration_System.py | re-validate `image_size%patch_size` on synced sizes |
| H7 | Env override `'1'`/`'0'`→bool | Configuration_System.py | parse numbers before bool keywords |
| H8 | `amp_dtype: float16` passes config validation | Configuration_System.py | bf16-only enforced in `TrainingConfig.validate` |
| H13 | Vocab integrity not contiguity-aware | vocabulary.py | gapless `0..N-1` assertion (live vocab verified contiguous) |
| H14 | MLP dropout un-configurable | train_direct.py | bridge `hidden_dropout_prob`→`dropout` (current value 0.10 unchanged) |
| H15 | `num_hidden_layers` default 17≠18 | model_architecture.py, Configuration_System.py | defaults → 18 |
| H16 | zero-positive drop applied to micro-F1/mAP | evaluation_metrics.py | drop scoped to macro only (standalone harness + threshold calibration) |
| H18 | `_check_numerical_stability` hardcodes clamp=15 | model_architecture.py | reads `config.logit_clamp_value`; wording fixed |
| H19 | Critical RNG-restore RuntimeError swallowed | training_utils.py | re-raise so resume aborts on RNG-restore failure |
| H20 | Cosine `get_lr` div-by-zero for multi-cycle | training_utils.py | clamp initial `cur_cycle_steps ≥ warmup_steps+1` |
| H21 | `get_timer_stats` slices a `deque` (TypeError) | Monitor_log.py | `list(...)[-100:]` |
| H22 | Webhook SSRF via `netloc` | Monitor_log.py | use `parsed.hostname` + reject embedded creds |
| H23 | `validate_specific_tags` mispairs tag↔column | validation_loop.py | iterate found-tag names in index order |
| H30 | `STOP_TRAINING` sentinel never cleared | train_direct.py | clear stale sentinel at startup |
| H31 | prepare_phase2 leaves stale `preprocessing_params` | tools/prepare_phase2.py | update top-level `preprocessing_params`/legacy `image_size` |
| H34 | `shutdown(timeout=60)` too short for big ckpt | train_direct.py | → 300s (the writer default) |
| H4 | `world_size` AttributeError (dead scaling API) | Configuration_System.py | added `TrainingConfig.world_size: int = 1` |

### Open items & decisions — status carried into the NEXT context window (2026-06-04)

**▶ Recommended next actions — ✅ APPLIED 2026-06-04 (medium/low pass):**
- **H12** ✅ FIXED — vocab-SHA resume guard now fails CLOSED. `load_checkpoint` gained `enforce_vocab_check` (set True only on the resume path) + `allow_unverified_vocab_resume` escape hatch (new `training.allow_unverified_vocab_resume: false` config + YAML). Both fail-open paths (current SHA None/"unknown"; checkpoint embeds no SHA) now raise `InvalidCheckpointError` unless the escape hatch is set. A concrete SHA mismatch is ALWAYS fatal, even for the non-enforcing inference/export/validation callers (which otherwise keep their legacy load-without-SHA behavior). Verified with an 8-scenario integration test. Files: `training_utils.py`, `train_direct.py`, `Configuration_System.py`, `configs/unified_config.yaml`.
- **H5 + H29 cleanup** ✅ DONE (scoped) — removed the misleading dead `FullConfig.optimizer`/`FullConfig.scheduler` sub-configs (`AdamW8bitConfig`/`SchedulerType`/`SchedulerConfig`) and the entire dead FullConfig scaling API (`compute_effective_batch_size`/`scale_learning_rate`/`scale_weight_decay`/`get_optimizer_kwargs`/`get_scheduler_kwargs`/…) — the "silent second source of truth" that looked authoritative but was never consumed (live path reads `config.training.*` and scales LR via `training_config.scale_learning_rate`). `to_dict`/`from_dict` round-trip verified; legacy checkpoint configs that still embed `optimizer:`/`scheduler:` keys are tolerated (skipped with a warning). LEFT the separate `training_config.py` helper library intact (it's a documented standalone module; its `scale_weight_decay` is explicit-mode, not silently applied).

**■ Decided / closed (no action needed):**
- **H29 — KEEP `weight_decay` FIXED at 0.05.** Verified deep-research conclusion (peer-reviewed/arXiv, 3-vote adversarial): optimal AdamW λ scales ~**1/N** (inverse-*linear* in dataset size) and *linearly* in batch — NOT 1/√N; the `inverse_sqrt`-by-dataset-size helper also misreads Loshchilov–Hutter "normalized WD" = λ_norm·√(b/(B·T)) (a joint term over batch+points+epochs, not N alone). The invariant to hold is the EMA timescale **τ_epoch = 1/(η·λ·N/B)**; scaling LR by √batch while holding WD fixed is the correct standard recipe (DeiT uses fixed WD=0.05 for 300 ep). At this run's settings **τ_epoch ≈ 105–137 epochs ≫ 15-epoch budget**, so WD is barely doing EMA work — its exact value is low-stakes (a WD=0 vs 0.05 ablation would confirm). For missing-positive label noise the lever is the **LOSS** (asymmetric/ASL γ_neg; Park 2023 robust ASL), **not** weight decay (longer training → *smaller* optimal WD). Papers: AdamW [1711.05101], Wang & Aitchison [2405.13698], Power Lines [2505.13738], NeurIPS24 [2310.04415], DeiT [2012.12877]. Full note in memory `project-weight-decay-fixed`.
- **H32 — DROPPED.** Operator does not use `best_model.pt`; `last.pt` (the resume file) is written *synchronously*, so crash-resume is unaffected. Nothing to protect.
- **H24 — SKIP.** `pad_color` is the default `(114,114,114)` everywhere; revisit only if pad color changes or fully self-describing checkpoints are needed for deployment.
- **H26 / H27 — WONTFIX.** Build-once cache + no mid-run sidecar edits → staleness conditions can't occur (memory `project-arrow-cache-build-once`). Dead `file_list_hash` is a harmless no-op.
- **H28** — *build-time* drop (distinct from staleness), dormant for all-numeric-ID data. Optional cheap guard for a build-once workflow: a dropped-count reconciliation log at end of cache build, if a future dataset has filenames violating `DEFAULT_ID_RE`.

### Refuted (no code change; docs only where noted)
- **H9** — flips NOT frozen (verified live). **H10** — false positive (see above). **H11** — sidecars are comma-delimited (verified against real `*.json`); parsing is correct, only AGENTS.md / docstrings are wrong. **H17** — the watched `val/f1_macro` is computed by train_direct's own per-class path, not `MetricComputer`; the `effective==size` short-circuit is a safe optimization (dead `metric_computer` at train_direct.py:610 is the only cleanup). **H25** — pageable D2H blocks the host, so no torch.cat race (a defensive end-of-loop sync is optional). **H35** — Adan fused single-tensor write-back is correct (kernel writes `out_p=param.data`) AND the path is dead (`fused=False`).

### New issues found during this pass (added to the list)
- 🟠 **[BUG?]** `train_direct.py` builds `tag_names = [vocab.index_to_tag[i] for i in range(len(vocab.index_to_tag))]` (image/prediction logging) — a vocab index gap would `KeyError`-crash logging while silently mis-labeling encodes. Now *prevented* by the H13 contiguity check, but the indexing pattern remains fragile if that check is ever bypassed (`skip_validation=True`).
- 🟢 **[BUG?]** `utils/metadata_cache.py` `_compute_file_list_hash` result is stored in cache meta but **never read back** anywhere — a dead "integrity" field giving false assurance (relates to H26/H27).
- 🟢 **[BUG?]** `utils/metadata_cache.py` `_stratified_sample` uses **unseeded `random.sample`**, so the staleness verdict is non-deterministic run-to-run (a stale cache can pass one launch and fail the next).
- 🟢 **[BUG?]** `adan_optimizer.py:432` (fused multi-tensor, currently dead) uses deprecated `torch.cuda.IntTensor([0])` on the default device — would break a multi-GPU param group if `fused=True` were ever enabled.

---

## ✅ Verification & fix log — 2026-06-04 (medium/low pass)

Verified each item against current code (4 parallel verification agents + independent cross-checks) before fixing. All edited files byte-compile; the live config validates with consistent image/patch sizes; the H12 guard passed an 8-scenario integration test.

### Fixed (verified, applied)
| ID | Bug | File(s) | Note |
|---|---|---|---|
| M1 | Webhook POST follows redirects (SSRF pivot) | Monitor_log.py | `allow_redirects=False` |
| M2 | Token-redaction regex misses `-`/`_` (JWT/Slack/base64url) | Monitor_log.py | prefix-anchored xox/JWT rules + `[A-Za-z0-9_\-]{32,}` |
| M3 | `TrainingMonitor._setup_logging` stacks duplicate handlers | Monitor_log.py | close+remove prior handlers on re-init |
| M4 | Double-logging (module handlers + root QueueHandler) | Monitor_log.py | `propagate=False` on the module logger |
| M5 | `log_validation` "best" tracked by LOWER loss vs higher-is-better f1 | Monitor_log.py | track `f1_macro` higher-is-better (logging/alert only) |
| M6 | `add_metric` stores NaN/Inf when `_to_safe_float`→None | Monitor_log.py | drop non-finite/non-scalar sample |
| M7 | `psutil.disk_usage('/')` monitors wrong volume on Windows | Monitor_log.py | derive from `config.log_dir`, walk to existing ancestor |
| M8 | `get_timer_stats`/`get_metric_stats` emit `np.float64`→stringified JSON | Monitor_log.py | cast to native `float`/`int` |
| E1 | `ThresholdCalibrator.calibrate` missing `.cpu()` before `.numpy()` | evaluation_metrics.py | add `.cpu()` (defensive; CPU no-op today) |
| E2 | `compute_bucketed_metrics` not fp32-cast (bf16 precision near threshold) | evaluation_metrics.py | `.float()` to match `compute_all_metrics` |
| E4 | Freq buckets silently drop tags below `bins[0]`/above last edge | evaluation_metrics.py | `_ensure_full_frequency_coverage` prepends 0 / appends inf (both bucketing paths) |
| A1 | Adan clones ALL grads even when no clip applies (per-step overhead) | adan_optimizer.py | pass `p.grad` directly in the no-clip branch (verified read-only downstream) |
| A3 | `CosineAnnealingWarmupRestarts.get_lr` ignores per-group `base_lrs` | training_utils.py | scale by `base_lr/base_max_lr` (exact no-op for uniform LRs; verified numerically) |
| A5 | `config.training.cycle_decay` read but undefined → fixed at 0.9 | Configuration_System.py, unified_config.yaml | added `cycle_decay: 0.9` field+key |
| X2 | Windows `msvcrt.locking` ignores `exclusive` (serializes readers) | utils/exclusion_manager.py | documented Windows-only limitation (no shared primitive exists) |
| X4 | `validate_fast` abandons dataloader iterator → orphan workers | validation_loop.py | explicit `_shutdown_workers()` before building fast_loader |
| X5 | Exclusion write dedup raw vs reader-normalized (dup growth) | utils/exclusion_manager.py | normalize existing+new ids on write (canonical on-disk form) |
| — | `validate_dataset` no-op gives false assurance | dataset_loader.py, train_direct.py | warn loudly it does nothing; remove misleading "complete" log |
| — | `data.patch_size` not synced from `model.patch_size` | Configuration_System.py | sync in `FullConfig.validate` (mask grid can't diverge) |
| — | `_log_to_backends` doubled `epoch/…/epoch` TB hierarchy | Monitor_log.py | drop redundant `use_epoch=True` (keys already `epoch/`-namespaced) |
| — | `ValidationRunner._setup_logging` stacks duplicate console handlers | validation_loop.py | guard against adding a second StreamHandler |

### Refuted / not-a-bug (no code change)
- **M9** — `MemoryMonitor` thresholds are genuinely absolute GB (default = 90/95% of total *converted to GB*); doc matches code.
- **E3** — torchmetrics 1.8.2 binarizes with strict `>` (verified in source); the per-tag/calibrator `>` is already aligned. No `>=` discrepancy.
- **A2** — `group['step']` is a standard param-group key; PyTorch's default `state_dict`/`load_state_dict` serialize+restore it, so bias correction resumes correctly.
- **A3-div** — the cosine div-by-zero was already fixed (H20 clamp `cur_cycle_steps >= warmup_steps+1` present at `training_utils.py:752`/`809`).

### Verified-but-not-fixed (decision noted)
- **carry_accum** (🟠) — CONFIRMED that a soft-stop carry epoch mixes the prior epoch's loss accounting (running_loss/total_train_samples/processed_batches not reset to preserve the gradient window). Impact is limited to `avg_train_loss` *reporting* on the single carry epoch during a soft-stop shutdown; **left as-is** — fixing risks the gradient-carry/accumulation logic for a cosmetic gain on a shutdown path.
- **A4** (🟢) — `LinearWarmupCosineLR` warmup/anneal off-by-ones CONFIRMED, but it is NOT the live scheduler (dead factory-only path); left documented.
- **X1** (🟠) — non-immediate exclusion loss is latent only; the live call sites pass `immediate=True` (`dataset_loader.py`), so production is unaffected.
- **X3** (🟠) — Windows lock-region-from-EOF desync is mostly theoretical under build-once + immediate handle close; the proposed file-attribute hack is risky for negligible benefit. Left documented via X2.
- **E5 / mAP_average='macro'** (🟠) — draw-dependent macro instability is an intended trade-off (WONTFIX per existing design note); project note already recommends moving auto-stop to `val/mAP`.
- **vocab_append** tag-frequency reset / missing-min_frequency (🟠) — real, but `vocab_append.py` is a manual utility and these change vocab-building semantics the user should choose deliberately (esp. the v2 vocab plan). Deferred to operator decision.

---

## ⚠️ Suspected issues to triage first

Every `suspected-bug` item, flattened and ordered by severity. Confirm or refute each before trusting a Phase-2 run.

### 🔴 High

- [ ] 🔴 **[BUG?]** Epoch-boundary accumulation flush does not rescale gradients for a partial window: each micro-batch divides loss by the FULL `accum`, but the flush takes a full `optimizer.step()` on `accum_count < accum` micro-batches without multiplying grads by `accum/accum_count`, under-weighting that update by `accum_count/accum` — biasing training and the effective LR of every non-divisible epoch and every soft-stop carry (`train_direct.py:1557-1563`, `train_direct.py:1921-1953`).
- [ ] 🔴 **[BUG?]** Phase1(320)→Phase2(448) resume silently restarts from scratch: `validate_config_compatibility` treats `data.image_size` as CRITICAL (strict=True → ValueError), and for `resume_from` in {latest,best} that ValueError is caught and the checkpoint is SKIPPED, so pos_embed interpolation in `load_checkpoint` never runs. Only works if `tools/prepare_phase2.py` rewrites the embedded config to 448 first (`train_direct.py:1043-1066`, `training_utils.py:433-462`, `training_utils.py:553-563`, `training_utils.py:1777-1834`).
- [ ] 🔴 **[BUG?]** Early-stopping patience advances on validation-SKIPPED epochs: when `should_validate` is False, `val_f1_macro`/`val_mAP` are read from cached state but the early-stopping block still executes; `val_f1_macro==best` (no improvement) so if `lr_ratio<0.5` patience increments on an epoch that produced no new metric, prematurely advancing toward the limit of 4 (`train_direct.py:2034-2059`, `train_direct.py:2300-2347`).
- [ ] 🔴 **[BUG?]** `FullConfig.compute_effective_batch_size()` references `self.training.world_size`, which is not a field on `TrainingConfig` → AttributeError before the `<=0` guard; this poisons the entire scaling API (scale_learning_rate/compute_total_steps/compute_warmup_steps/scale_weight_decay/adjust_beta2_for_long_training/get_optimizer_kwargs/get_scheduler_kwargs) (`Configuration_System.py:1795-1800`, `Configuration_System.py:1173-1317`).
- [ ] 🔴 **[BUG?]** `config.optimizer` (AdamW8bitConfig: base_lr=1e-4, weight_decay=0.01, wd_scaling_mode) and `config.scheduler` (SchedulerConfig) are loaded/validated/serialized but NEVER used by training — train_direct recomputes LR/effective batch inline and reads only `config.training.*`. These are a silent second source of truth that looks authoritative but does nothing (`Configuration_System.py:1851-1899`, `Configuration_System.py:1619-1694`, `train_direct.py:775-797`).
- [ ] 🔴 **[BUG?]** Resolution-sync invariant runs AFTER each sub-config's `validate()`: `FullConfig.validate` mutates `model.image_size`/`validation.preprocessing.image_size` from `data.image_size`, but `ModelConfig.validate` checked `image_size % patch_size` on the OLD model value, so a non-divisible `data.image_size` is propagated with no re-validation (`Configuration_System.py:1733-1776`, `Configuration_System.py:811-819`).
- [ ] 🔴 **[BUG?]** Env override coercion writes wrong types: `_parse_env_value` returns bool for `'1'`/`'0'` BEFORE the numeric branch, and `_apply_nested_updates` setattr's the raw value with no per-field coercion — so `MODEL__PATCH_SIZE=1`→bool True, `DATA__BATCH_SIZE=0`→bool False, silently writing a bool into an int field (`Configuration_System.py:2066-2117`, `Configuration_System.py:2132-2153`, `Configuration_System.py:2060-2064`).
- [ ] 🔴 **[BUG?]** `amp_dtype: float16` passes config validation (TrainingConfig does not restrict it) but `train_direct.py` hard-rejects anything but bfloat16 — the bf16-only invariant is enforced far downstream instead of in the config layer (`Configuration_System.py:1202-1205`, `Configuration_System.py:1259-1317`, `train_direct.py:872-879`).
- [ ] 🔴 **[BUG?]** Flip-augmentation epoch counter `_current_epoch = mp.Value('i', 0, lock=False)` is created at construction then pickled to each worker on Windows spawn; the parent's `set_epoch()` likely never reaches the worker copy, freezing horizontal flips at epoch 0 forever (and the `set_epoch never called` warning lives in the worker so it never fires) (`dataset_loader.py:1908`, `dataset_loader.py:2164`, `dataset_loader.py:2179`, `dataset_loader.py:2215`, `dataset_loader.py:2287`).
- [ ] 🔴 **[BUG?]** `mp.Value('i',0,lock=False)` may not pickle on Windows spawn (or yields a disconnected copy): `__getstate__` nulls `_stats_queue`/`_exclusion_manager` but NOT `_current_epoch`, so either worker creation fails or workers get a frozen-epoch copy — both must be confirmed (`dataset_loader.py:2094`, `dataset_loader.py:2104`, `dataset_loader.py:1908`).
- [ ] 🔴 **[BUG?]** Tag delimiter mismatch (the single most consequential silent-data surface): `parse_tags_field` and `vocabulary` both split string tag fields on `,`, but the documented sidecar format is space-delimited (`"t1 t2"`). If real data is space-delimited, every multi-tag string collapses to one UNK token at BOTH build and encode time — vocab and labels are wrong but mutually consistent, so no integrity check fires (`utils/metadata_ingestion.py:13/22-23`, `vocabulary.py:139-144`, `vocabulary.py:717-720`, `dataset_loader.py:2079`, `utils/metadata_cache.py:194`).
- [ ] 🔴 **[BUG?]** Resume vocab-SHA guard fails OPEN: `compute_vocab_sha256` returns `'unknown'` (not an exception) on a missing/unreadable vocab, and train_direct sets `current_vocab_sha=None` on exception yet still calls `load_checkpoint`, so a checkpoint trained on a different vocab can load against a mismatched head/label mapping with no error (`train_direct.py:1070-1085`, `schemas.py:147-161`, `vocabulary.py:1011`, `training_utils.py:1379-1386`, `training_utils.py:1726-1739`).
- [ ] 🔴 **[BUG?]** `from_json`/`load_vocabulary` integrity check is not contiguity-aware: `_verify_vocabulary_integrity` confirms bidirectional round-trips but does NOT verify indices form a gapless `0..N-1` range. A vocab with a gap (max index > len) makes the highest-index tags fail `encode_tags`' `idx < vocab_size` bounds check and silently drop from every label (`vocabulary.py:1098-1166`, `vocabulary.py:644-658`, `vocabulary_utils/vocab_append.py:382`).
- [ ] 🔴 **[BUG?]** MLP/projection dropout is un-configurable: the YAML's `model.hidden_dropout_prob` is stripped by `_unused_config_keys` and `VisionTransformerConfig.dropout` (the field that actually drives the MLP `nn.Dropout`) has NO YAML key, so dropout always uses the dataclass default 0.1; setting 0.0/0.3 is silently overridden (`model_architecture.py:132`, `model_architecture.py:243-249`, `train_direct.py:626-633`, `configs/unified_config.yaml:35`, `Configuration_System.py:762`).
- [ ] 🔴 **[BUG?]** `num_hidden_layers` default (17) disagrees with the YAML/spec (18): any model-construction path that omits the YAML value builds a 17-layer model whose state_dict will not match an 18-layer checkpoint (`model_architecture.py:129`, `configs/unified_config.yaml:29`).
- [ ] 🔴 **[BUG?]** `_drop_zero_positive_classes` filters out zero-positive classes BEFORE micro-F1 AND mAP (not just macro-F1), inflating micro-F1 (removes guaranteed-FP columns) and changing the mAP averaging set, making `val/f1_micro` and `val/mAP` non-comparable across draws and against fixed-threshold competitors (`evaluation_metrics.py:116`, `evaluation_metrics.py:121-148`).
- [ ] 🔴 **[BUG?]** `_drop_zero_positive_classes` silently disables filtering when `effective == preds.size(1)` (line 146 early-return), so on a dense draw NO classes are dropped and macro-F1 averages over all classes, while a sparse draw drops thousands — the macro-F1/mAP definition flips draw-to-draw, producing a noisy early-stopping signal (watched metric is `val/f1_macro`) (`evaluation_metrics.py:144-148`).
- [ ] 🔴 **[BUG?]** `_check_numerical_stability` hardcodes `clamp_threshold = 15.0` instead of `config.logit_clamp_value`, and the warning says "Clamping" though this method does not clamp — misleading diagnostics decoupled from the real clamp value (`model_architecture.py:474-481`, `model_architecture.py:673-675`).
- [ ] 🔴 **[BUG?]** Critical RNG-restore RuntimeError is swallowed: the `RuntimeError` raised at `training_utils.py:2090` for critical RNG-restore failure is inside the same `try` whose broad `except Exception` (line 2099) catches and downgrades it to a logged error, so resume silently proceeds with the wrong data order — the entire guard is defeated (`training_utils.py:2009-2102`).
- [ ] 🔴 **[BUG?]** `CosineAnnealingWarmupRestarts.get_lr` divides by `(cur_cycle_steps - warmup_steps)`; for multi-cycle configs where `first_cycle_steps = total_updates // num_cycles <= warmup_steps`, this is 0/0 or negative → NaN/inf LR, and scheduler.step errors are only logged (`training_utils.py:758-768`, `train_direct.py:846-864`).
- [ ] 🔴 **[BUG?]** `get_timer_stats` slices a `deque` (`self.timers[name][-100:]`) — `deque` does not support slice indexing → TypeError on every call, which aborts `save_metrics` inside its try/except and silently drops the metrics checkpoint (`Monitor_log.py:468-482`, `Monitor_log.py:351`, `Monitor_log.py:491`).
- [ ] 🔴 **[BUG?]** Webhook SSRF allowlist matches `parsed.netloc` (which includes `user:pass@host:port`) instead of `parsed.hostname`, mishandling embedded credentials and IPv6/port forms — switch to `parsed.hostname` for a robust host-confusion-resistant allowlist (`Monitor_log.py:85-98`).
- [ ] 🔴 **[BUG?]** `validate_specific_tags` mispairs tag names with prediction columns: results are built with `zip(specific_tags, tag_indices_cpu)`, but `tag_indices_cpu` drops UNK/not-found tags, so the lists differ in length — zip truncates AND mislabels every subsequent tag's metrics (`validation_loop.py:1092-1106`, `validation_loop.py:1161-1163`).
- [ ] 🔴 **[BUG?]** `pad_color` is never persisted in the checkpoint and never set in the validation `CSDataConfig`, so validation always letterboxes with the default `(114,114,114)`; if training used a different pad color, val metrics are non-comparable to train and to deployed inference (`validation_loop.py:640-650`, `model_metadata.py:188-194`, `dataset_loader.py:2622`).
- [ ] 🔴 **[BUG?]** Async-transfer correctness hazard in `validate_full`: final D2H copies use `non_blocking=True` and the GPU sync only happens every 50 batches and once at end — with `measure_inference_time=False` and a sample count not divisible by 50, the last <50 batches may be `torch.cat`'d before their async copies finish, silently corrupting predictions (`validation_loop.py:900-903`, `validation_loop.py:923-924`, `validation_loop.py:940-953`, `validation_loop.py:966-968`).
- [ ] 🔴 **[BUG?]** Arrow staleness check never detects in-place JSON edits: `_is_arrow_cache_stale` only compares sampled mtime + file-count drift, and `_compute_file_list_hash` hashes path STRINGS only (no mtime/content) — an mtime-preserving edit or restore silently serves stale tags/rating for the whole run (`utils/metadata_cache.py:442`, `utils/metadata_cache.py:60`, `utils/metadata_cache.py:490`).
- [ ] 🔴 **[BUG?]** Net-zero/small file churn does not invalidate the cache: file-count drift tolerance is `max(100, 0.1%)`, so deleting N JSONs and adding N different ones (drift 0) — or any change within tolerance — is invisible; added images are never trained, deleted images become FileNotFound error samples (`utils/metadata_cache.py:479`, `utils/metadata_cache.py:482`).
- [ ] 🔴 **[BUG?]** `sanitize_identifier` ValueErrors in `_build_arrow_cache` are caught with a per-file warning + `continue`; a systematic naming convention violating `DEFAULT_ID_RE` (spaces/unicode/parentheses common in anime filenames) silently drops every such image from the cache and thus from training (`utils/metadata_cache.py:192`, `utils/metadata_cache.py:204`, `utils/path_utils.py:24`).
- [ ] 🔴 **[BUG?]** Weight decay is NOT inverse-sqrt scaled by dataset size as documented: train_direct passes raw `weight_decay=0.05` to the optimizer while it DOES scale LR; `scale_weight_decay()` exists but is only consumed by `create_optimizer_config`, which the live path never calls (`train_direct.py:794`, `train_direct.py:779`, `training_config.py:161`, `training_config.py:303`).
- [ ] 🔴 **[BUG?]** STOP_TRAINING sentinel is never cleared: `stop_sentinel` is checked at loop entry/epoch boundary but there is NO `stop_sentinel.unlink()` anywhere (unlike save/image sentinels), so a leftover `logs/STOP_TRAINING` immediately stops the very next training invocation until manually deleted — a resume footgun (`train_direct.py:967`, `train_direct.py:1690`, `train_direct.py:1962`, `train_direct.py:1807`).
- [ ] 🔴 **[BUG?]** Phase1→Phase2 image_size may not be updated where inference/validation reads it: `prepare_phase2.py` rewrites `cfg['data'/'model'/'validation'].image_size` but does NOT touch any embedded `preprocessing_params` block, while inference/validation PREFER `preprocessing_params` — a stale 320 there yields a silent train/inference resolution mismatch (`tools/prepare_phase2.py:74`, `Inference_Engine.py:565`, `model_metadata.py`).
- [ ] 🔴 **[BUG?]** `best_model.pt` copy runs inside the async worker-thread callback after the numbered file lands; an async-save failure, a too-short shutdown timeout, or process exit before drain can silently lose the best checkpoint even when `is_best=True` (`training_utils.py:1457-1499`, `train_direct.py:2320-2343`).
- [ ] 🔴 **[BUG?]** `AsyncCheckpointWriter` swallows worker-loop exceptions and `_last_error` is never inspected by train_direct, so a run of failed async saves ends "cleanly" with no usable checkpoint; an exception between `queue.get()` and the inner try can also leave `pending_count` stuck, hanging `wait_pending` to timeout (`training_utils.py:920-973`, `training_utils.py:1059-1062`, `train_direct.py:2367-2368`).
- [ ] 🔴 **[BUG?]** `shutdown(timeout=60.0)` may be too short: the writer documents 30-90s saves and its own default timeout is 300s, so a large final best checkpoint queued near end-of-run can exceed 60s, `wait_pending` returns False, the worker stops, and the `best_model.pt` callback never runs (`training_utils.py:889-893`, `training_utils.py:1012-1051`, `train_direct.py:2367-2368`).
- [ ] 🔴 **[BUG?]** Verify the fused single-tensor Adan path writes results back: `_fused_adan_single_tensor` builds `p_data_fp32 = param.data.float()` and passes it as the read buffer plus `out_p = param.data`, but `p_data_fp32` is never copied back — if the C++ kernel updates the fp32 master copy and expects the caller to copy it to `param`, weights silently freeze when `fused=True` (`adan_optimizer.py:463-464`, `adan_optimizer.py:473`).

### 🟠 Medium

- [ ] 🟠 **[BUG?]** Replace/remove `validate_dataset`: it is a no-op placeholder returning `{}` in dataset_loader, yet `config.debug.validate_input_data` triggers it, giving false assurance that inputs/labels are validated before training (`train_direct.py:587-592`, `dataset_loader.py:1765-1767`).
- [ ] 🟠 **[BUG?]** Soft-stop `carry_accum` across epochs keeps the previous epoch's `running_loss`/`total_train_samples`/`processed_batches` while a new epoch begins, so `avg_train_loss` for the carry epoch mixes two epochs' samples; also `set_epoch(epoch)` changes flip decisions mid-accumulation-window, mixing augmentation states within one optimizer step (`train_direct.py:1356-1369`, `train_direct.py:1960-1976`).
- [ ] 🟠 **[BUG?]** `config.training.cycle_decay` (read at `train_direct.py:850`) is not a `TrainingConfig` field and not in the YAML, so SGDR restart decay (gamma) is permanently fixed at the 0.9 fallback and cannot be set from config — inert at `num_cycles=1`, a footgun the moment `num_cycles>1` (`train_direct.py:849-870`, `Configuration_System.py:1196-1201`).
- [ ] 🟠 **[BUG?]** `DataConfig.patch_size` (default 16) is independent of `ModelConfig.patch_size` and is NOT synced in `FullConfig.validate`; the token-level padding-mask pooling uses `data.patch_size`, so changing one but not the other silently mismatches the attention mask grid against the model's patch grid with no validation error (`Configuration_System.py:1005`, `Configuration_System.py:1758-1776`).
- [ ] 🟠 **[BUG?]** `_cleanup_old_checkpoints()` runs immediately after queueing an async save while the numbered file may not yet exist; `_refresh_checkpoint_list` filters by `p.exists()` and drops the just-queued path, so retention is computed without it — can over-prune one extra old checkpoint or leave `max_checkpoints+1` files (`training_utils.py:1465-1475`, `training_utils.py:1528-1588`).
- [ ] 🟠 **[BUG?]** Per-worker DataLoader RNG is not captured/restored across resume: `_save/_restore_rng_states` only handle main-process RNG, and `worker_init_fn` reseeds from `torch.initial_seed()+worker_id` fresh each epoch, so mid-epoch resume restarts worker augmentation/sampling streams instead of continuing — breaks bit-reproducible resume (`training_utils.py:132-165`, `training_utils.py:2735-2750`).
- [ ] 🟠 **[BUG?]** `batch_in_epoch` off-by-one between save paths: periodic save stores `batch_in_epoch = step` and `sample_in_epoch = step*batch_size`, while the soft-stop path uses `step+1`, so resume can re-process or skip exactly one batch depending on which save produced last.pt (`train_direct.py:1648-1650`, `train_direct.py:1716-1717`, `train_direct.py:1777-1778`).
- [ ] 🟠 **[BUG?]** Adan optimizer-state restore can break bias correction: `group['step']` (drives bias_correction1/2/3) is a param-group key, not per-param `self.state`; if checkpoint save/load drops it, resume restarts bias correction from 0/1, inflating the first resumed update by ~1/bc and destabilizing the run (`adan_optimizer.py:204/208`, `train_direct.py:1656`).
- [ ] 🟠 **[BUG?]** Per-step grad cloning in `Adan.step` clones ALL grads even in the else-branch where no clip multiply is applied — pure overhead (~full param-sized fp32 alloc per step) and a possible OOM contributor at 448px Phase 2; condition the clone on clipping (`adan_optimizer.py:216`, `adan_optimizer.py:221`).
- [ ] 🟠 **[BUG?]** `CosineAnnealingWarmupRestarts.get_lr` ignores `base_lrs` (`for _ in self.base_lrs` returns the same `max_lr` for every group), collapsing any intended per-group LR (layer decay, separate head LR); contrast `LinearWarmupCosineLR` which preserves per-group `base_lrs` (`schedulers.py:70`, `training_utils.py:761`, `training_utils.py:2628`).
- [ ] 🟠 **[BUG?]** SSRF: webhook POST follows redirects by default — an allowlisted host can 3xx-redirect the alert POST to an internal/arbitrary endpoint; set `allow_redirects=False` (`Monitor_log.py:313-314`).
- [ ] 🟠 **[BUG?]** Token-redaction regex `[A-Za-z0-9]{32,}` ignores `-`/`_`, so JWTs/Slack `xoxb-...`/base64url tokens leak partially while benign long alphanumerics (SHA256/git rev/run_id) get clobbered — both over- and under-inclusive (`Monitor_log.py:254-255`).
- [ ] 🟠 **[BUG?]** `TrainingMonitor._setup_logging()` unconditionally adds a fresh `StreamHandler` (and `FileHandler`) to the module logger on every construction with no dedup, stacking duplicate handlers and leaking file handles on Phase1→Phase2 / resume / re-init (`Monitor_log.py:926-940`).
- [ ] 🟠 **[BUG?]** Double-logging: `setup_logging` attaches a QueueHandler to ROOT while `TrainingMonitor._setup_logging` adds StreamHandler+FileHandler to the `Monitor_log` module logger (which propagates to root), so each monitor line is emitted twice and to two physical log files (`Monitor_log.py:926-940`, `utils/logging_setup.py:136-141`, `train_direct.py:2426-2431`).
- [ ] 🟠 **[BUG?]** `log_validation` tracks "best" by LOWER val loss (`best_val_metric=float('inf')`, `<` comparison) while early stopping/best-selection uses higher-is-better `val/f1_macro` (or recommended `val/mAP`) — the monitor's notion of "best" is inverted vs the checkpoint/early-stopping notion (`Monitor_log.py:836`, `Monitor_log.py:1110-1117`).
- [ ] 🟠 **[BUG?]** `add_metric` can store a non-finite value: when `_to_safe_float` returns None it keeps the original (possibly NaN/Inf) python float and appends it to history and pushes it to a Prometheus gauge unguarded, poisoning `get_metric_stats`'s `np.mean` (`Monitor_log.py:384-412`, `utils/logging_sanitize.py:7-48`).
- [ ] 🟠 **[BUG?]** `SystemMonitor` calls `psutil.disk_usage('/')`, which on Windows reports the current working drive root — not the L: volume holding data/checkpoints — so the "Low Disk Space" critical alert monitors the wrong volume (`Monitor_log.py:730`, `Monitor_log.py:1577-1588`).
- [ ] 🟠 **[BUG?]** `save_metrics` even if the deque-slice bug is fixed: `get_timer_stats` returns `np.float64` and `json.dump(default=str)` stringifies them, turning numeric metrics into strings in the saved JSON and corrupting downstream parsing (`Monitor_log.py:484-501`, `Monitor_log.py:459-466`, `Monitor_log.py:475-482`).
- [ ] 🟠 **[BUG?]** Orphaned-worker leak in `validate_fast`: it iterates the full multi-worker dataloader for 50 batches and `break`s without draining/shutting down workers before building `fast_loader`, accumulating orphan worker processes on Windows spawn across repeated runs (`validation_loop.py:1050-1077`).
- [ ] 🟠 **[BUG?]** `ExclusionManager.add_exclusion(immediate=False)` (the DEFAULT) only appends to `_pending_writes`, flushing at >=10; `flush_pending`/`_flush_pending_internal` have NO external callers and there is no atexit/finalizer flush, so any non-immediate use loses up to 9 exclusions per worker on exit (`utils/exclusion_manager.py:180/204/232`, `dataset_loader.py:2527`).
- [ ] 🟠 **[BUG?]** Windows `msvcrt.locking` always takes a MANDATORY EXCLUSIVE byte-range lock — the `exclusive` param is ignored — so the "shared read fast path" actually serializes readers against the writer, spuriously firing the cached-snapshot fallback and missing new exclusions (`utils/exclusion_manager.py:386/402/372`).
- [ ] 🟠 **[BUG?]** Windows exclusion lock region is sized from current EOF (`max(1, f.seek(0,2))`); if process B appends after A locks, the byte ranges differ so the whole-file exclusive guarantee is not held, and release may `LK_UNLCK` a region that was not locked — interleaved/lost exclusion lines (`utils/exclusion_manager.py:404/435/283`).
- [ ] 🟠 **[BUG?]** torch.compile warmup compiles a DIFFERENT graph than training: the warmup runs `model(dummy_images)` with NO padding_mask and BEFORE `model.train()`, so real batches (mask present, training=True, gradient-checkpointing branch) recompile on the first real step, defeating the "overlap compile with worker warmup" optimization (`train_direct.py:1251`, `train_direct.py:1356`, `model_architecture.py:588`, `model_architecture.py:634`).
- [ ] 🟠 **[BUG?]** `RunMetadata.top_k` is annotated `int` (non-Optional) but validation_loop passes None, and `asdict` serializes `top_k: null` into the standardized predictions JSON — a schema-contract violation for consumers expecting an int (`validation_loop.py:1466-1467`, `schemas.py:56-70`).
- [ ] 🟠 **[BUG?]** Per-tag/calibrator threshold uses strict `>` (`compute_per_tag_metrics`, `ThresholdCalibrator`) while torchmetrics `multilabel_f1_score` binarizes with `>=`, so calibrated per-tag thresholds won't reproduce the macro-F1 they were selected to maximize (`evaluation_metrics.py:299`, `evaluation_metrics.py:533/588`, `evaluation_metrics.py:118-123`).
- [ ] 🟠 **[BUG?]** `ThresholdCalibrator.calibrate` calls `.detach().float().numpy()` / `.detach().numpy()` WITHOUT `.cpu()`; any caller passing CUDA tensors crashes with "can't convert cuda tensor to numpy" (`evaluation_metrics.py:493-494`, `train_direct.py:2212`).
- [ ] 🟠 **[BUG?]** `FrequencyBucketMetrics`/`ThresholdCalibrator` buckets silently drop tags whose frequency is below `bins[0]` or above the last finite edge (no else branch); if `frequency_bins` doesn't start at 0 / end at inf, rare/very-common tags vanish from the breakdown the LVIS-style diagnostic exists to expose (`evaluation_metrics.py:376-379`, `evaluation_metrics.py:568-571`, `evaluation_metrics.py:356-368`).
- [ ] 🟠 **[BUG?]** `compute_bucketed_metrics` does NOT cast `bucket_preds` to fp32 (unlike `compute_all_metrics` line 97), so with bf16 validation preds the bucket F1/mAP lose precision near the 0.2653 threshold — inconsistent with the headline metrics (`evaluation_metrics.py:402`, `evaluation_metrics.py:97`).
- [ ] 🟠 **[BUG?]** `vocab_append.main` rebuilds `tag_frequencies` from the CURRENT scan only (`freqs[tag] = counts.get(tag, 0)`), resetting any previously-frequent tag missing from the latest snapshot to 0; downstream min_frequency/class-weighting consumers then treat a legitimately-trained tag as dead (`vocabulary_utils/vocab_append.py:399-405`, `train_direct.py:191-254`).
- [ ] 🟠 **[BUG?]** `vocab_append` appends EVERY new tag with no `min_frequency` filter, unlike `create_vocabulary_from_datasets` (min_frequency=125 + top_k), so running it after the canonical builder injects rare freq-1 tags and changes `num_tags` — the two tools build inconsistent vocabularies and heads (`vocabulary_utils/vocab_append.py:385-393`, `vocabulary.py:745-751`, `vocabulary.py:1171-1232`).

### 🟢 Low

- [ ] 🟢 **[BUG?]** `from_dict` lossy float→int guard only fires inside Union branches; for a plain (non-Optional) int field the final else assigns verbatim, so a YAML `patch_size: 16.0` stays a float 16.0 and flows into patch-count math (`Configuration_System.py:460-489`).
- [ ] 🟢 **[BUG?]** `LinearWarmupCosineLR` warmup never reaches `base_lr`: `scale = e/max(1,warmup_epochs)`, so the last warmup step yields `(W-1)/W < 1` and `base_lr` is only hit at the first cosine step — a small upward jump at the boundary, non-trivial for short warmups (`schedulers.py:63/67/83`).
- [ ] 🟢 **[BUG?]** `LinearWarmupCosineLR` reaches `eta_min` only at `e >= max_epochs` (t==1); at the final intended epoch `max_epochs-1` the LR is still above `eta_min` — confirm the intended terminal LR (`schedulers.py:82/84/86`).
- [ ] 🟢 **[BUG?]** `_log_to_backends` builds `tag = f"{name}{'/epoch' if use_epoch else ''}"`, so an already-`epoch/`-prefixed key becomes `epoch/train_loss/epoch` — doubled hierarchy clutters the TensorBoard scalar tree (`Monitor_log.py:1122-1140`, `Monitor_log.py:1621-1627`).
- [ ] 🟢 **[BUG?]** `_normalize_exclusion_line` truncates an image_id whose stem literally ends in a known extension (e.g. id `foo.png`) to `foo`, but `add_exclusion` stores the raw id, so the reloaded set never matches the in-memory id and the image is never skipped (`utils/exclusion_manager.py:64-65`, `dataset_loader.py:2529`).
- [ ] 🟢 **[BUG?]** Exclusion-file dedup mismatch: `_write_exclusions_internal` dedups against raw `line.strip()` while the reader normalizes to stems, so a stem-form write is not deduped against an existing path-form line → duplicate logical exclusions and unbounded file growth (`utils/exclusion_manager.py:276`, `utils/exclusion_manager.py:58/148`).
- [ ] 🟢 **[BUG?]** `ValidationRunner._setup_logging()` unconditionally `addHandler`s on the module logger every construction, duplicating console output and accumulating handlers across a sweep loop (`validation_loop.py:487-517`).

---

## Config system

#### Correctness bug

- [ ] 🔴 `FullConfig.compute_effective_batch_size()` references non-existent `self.training.world_size` → AttributeError, poisoning the whole scaling API; the active path bypasses it but it is dead-on-arrival for single-GPU [BUG?] (`Configuration_System.py:1795-1800`, `Configuration_System.py:1173-1317`).
- [ ] 🟠 `LossConfig` dataclass defaults (alpha=0.5, gamma_neg=3.0, gamma_pos=1.0, clip=0.05) are stale Phase-1-ish values that drift from the active YAML Phase-2 (alpha=1.0, gamma_neg=7.0, gamma_pos=0.0, clip=0.2); confirm no path constructs `LossConfig()` without the YAML and trains with gamma_neg=3.0 (`Configuration_System.py:1135-1170`, `configs/unified_config.yaml:286-295`).

#### Config & invariants

- [ ] 🔴 Resolution-sync invariant runs AFTER each sub-config `validate()`, so patch-divisibility is checked on the stale `model.image_size`, not the synced `data.image_size`; a non-divisible data size is propagated with no re-validation [BUG?] (`Configuration_System.py:1733-1776`, `Configuration_System.py:811-819`).
- [ ] 🔴 `amp_dtype: float16` passes config validation but train_direct hard-rejects it — the bf16-only invariant belongs in the config layer, not far downstream [BUG?] (`Configuration_System.py:1202-1205`, `Configuration_System.py:1259-1317`, `train_direct.py:872-879`).
- [ ] 🟠 `DataConfig.patch_size` is independent of `ModelConfig.patch_size` and not synced, so the token-mask grid can silently diverge from the patch grid [BUG?] (`Configuration_System.py:1005`, `Configuration_System.py:1758-1776`).
- [ ] 🟠 `config.training.cycle_decay` is read but is not a field and not in the YAML, hardcoding SGDR restart decay to 0.9 [BUG?] (`train_direct.py:849-870`, `Configuration_System.py:1196-1201`).
- [ ] 🟠 Confirm `model.image_size` used by `create_model` matches `data.image_size` at the moment `to_dict()` is read; if config is mutated after load without a `validate()`, the model could be built at the stale default 448 while data feeds another resolution → pos-embed/token mismatch at the first forward (`train_direct.py:619-633`, `Configuration_System.py:1758-1776`, `Configuration_System.py:2340-2348`).
- [ ] 🟠 Reconcile the `num_labels` invariant: `num_labels=0` (auto) means `num_groups=20 * tags_per_group=10000 = 200000` is meaningless dead config; setting `num_labels` explicitly to the real vocab size would be REJECTED because it won't equal 200000 (`Configuration_System.py:783-809`, `configs/unified_config.yaml:46-48`).
- [ ] 🟠 Confirm `load_from_file` / `from_dict` do NOT run `validate()` (and thus skip the image_size sync); any caller that loads without the final `validate()` leaves `model.image_size`/`validation.preprocessing.image_size` disagreeing with `data.image_size` (`Configuration_System.py:1957-1963`, `Configuration_System.py:2301-2348`, `Configuration_System.py:757/915`).
- [ ] 🟠 The default-constructed `FullConfig` is internally inconsistent (Data=512, ValidationPreprocessing=512, Model=448) until `validate()` syncs; `generate_example_configs` never calls validate(), so emitted examples ship with contradictory image sizes (`Configuration_System.py:915/1469/757/2351-2366`).
- [ ] 🟠 Threshold source split: streaming val F1 pulls `threshold_calibration.default_threshold` while bucketed metrics use `inference.prediction_threshold` (both 0.2653 today); if they diverge, F1 is silently reported at two thresholds (`train_direct.py:1297-1311`, `train_direct.py:2177-2189`).
- [ ] 🟠 `ThresholdCalibrationConfig` defines `search_min/max/step` the YAML omits (silent defaults), and `from_dict` skips unknown keys at WARNING — a misspelled `defualt_threshold` is dropped silently, reverting to 0.2653 (`Configuration_System.py:1472-1481`, `Configuration_System.py:393-394`, `configs/unified_config.yaml:344-349`).
- [ ] 🟠 Deprecated `TrainingConfig.max_grad_norm` (1.0) coexists with `gradient_clipping.max_norm`; confirm the consumer reads the nested field so editing the YAML `gradient_clipping.max_norm` actually takes effect (`Configuration_System.py:1207-1209`, `Configuration_System.py:1128-1132`, `configs/unified_config.yaml:265-268`).
- [ ] 🟠 `DataConfig.validate` requires every enabled storage path to exist/`is_dir` at config-load time, so loading on any box without `L:/Dab/Dab` mounted fails validation outright — blocking even `--validate-only` (`Configuration_System.py:1044-1064`, `configs/unified_config.yaml:61-65`).
- [ ] 🟠 Monitor YAML section sets only a subset of `MonitorConfig` fields; notably `normalize_mean/std` default 0.5 and are NOT synced from `data.normalize_mean` in `FullConfig.validate`, so a future normalization change would garble TensorBoard image denormalization (`Configuration_System.py:1501-1566`, `Configuration_System.py:1554-1556`, `configs/unified_config.yaml:396-430`).
- [ ] 🟢 `from_dict` plain int fields are not coerced; a YAML float for a plain int is stored verbatim [BUG?] (`Configuration_System.py:460-489`).
- [ ] 🟢 `ValidationConfig`/`ValidationDataloaderConfig`/`ValidationPreprocessingConfig` have no `validate()` and `FullConfig.validate`'s loop omits `validation`/`threshold_calibration`, so `validation.dataloader.batch_size<=0` or a negative `max_samples` is never caught (`Configuration_System.py:1733-1746`, `Configuration_System.py:1456-1489`).
- [ ] 🟢 `experiment_name` `default_factory` mints a fresh timestamped name on every load if the YAML omits the key, so resume/checkpoint-path derivation could point at a different directory across processes (active YAML overrides it) (`Configuration_System.py:1713-1714`, `configs/unified_config.yaml:7`).

#### API & interface drift

- [ ] 🔴 `config.optimizer`/`config.scheduler` are loaded/validated/serialized but never consumed by training — a silent second source of truth (base_lr=1e-4, weight_decay=0.01) that looks authoritative but does nothing [BUG?] (`Configuration_System.py:1851-1899`, `Configuration_System.py:1619-1694`, `train_direct.py:775-797`).
- [ ] 🟠 TWO independent `AdamW8bitConfig` and `SchedulerConfig`/`SchedulerType` definitions exist (training_config.py vs Configuration_System.py) with differing fields/defaults; `create_scheduler_from_config` expects the training_config variant while `FullConfig.scheduler` is the other — passing one where the other is expected silently uses different defaults (`training_config.py:28-65`, `training_config.py:475-509`, `Configuration_System.py:1619-1694`).
- [ ] 🟠 Confirm the active path never calls any `FullConfig` scaling method and reads only `config.training.*` (`train_direct.py:775-797`).
- [ ] 🟢 `SchedulerConfig.restart_decay`/`warmup_strategy` are defined but never emitted by `get_scheduler_kwargs` or consumed in training — false configurability (`Configuration_System.py:1669-1694`, `Configuration_System.py:1860-1899`).

#### Error handling & robustness

- [ ] 🔴 Env override type coercion writes bool/raw values into typed fields with no per-field coercion (`'1'`→True, `'0'`→False) and `_apply_nested_updates` bypasses `BaseConfig.update`/`from_dict` validation [BUG?] (`Configuration_System.py:2066-2117`, `Configuration_System.py:2132-2153`).
- [ ] 🟠 CLI `--section.field` overrides only work for the tiny hardcoded `type=` allowlist in `create_config_parser`; arbitrary nested keys (e.g. `training.gradient_accumulation_steps`, `data.color_order`) cannot be overridden at all (`Configuration_System.py:2251-2298`, `Configuration_System.py:2119-2153`).
- [ ] 🟠 `_apply_nested_updates` `setattr`s raw values, never invoking the Union/dataclass coercion in `BaseConfig.update`/`from_dict`; many fields (memory_format, compile_mode) have no type assertion to catch a wrong type (`Configuration_System.py:2132-2153`, `Configuration_System.py:2020-2064`).

#### Reproducibility & determinism

- [ ] 🟢 Confirm `seed=42 + deterministic=false + benchmark=true` (the active YAML) is the intended non-reproducible default; validation only errors when BOTH deterministic and benchmark are True, so cuDNN autotune makes runs non-bit-reproducible despite a fixed seed (`Configuration_System.py:1259-1270`, `configs/unified_config.yaml:311-314`).

#### Numerical stability

- [ ] 🟢 `get_recommended_batch_size` is unguarded against `target_steps_per_epoch=0` (ZeroDivisionError) and tiny datasets yielding 0 steps/grad_accum (`training_config.py:384-455`).
- [ ] 🟢 `compute_cycle_steps` divides by `(cycle_mult-1)`; confirm the `cycle_mult==1.0` branch is taken first and `num_cycles>=1` always (else `total_steps//0`) (`training_config.py:593-637`).

#### Security

- [ ] 🟢 `validate_webhook_url` is only applied to the sensitive_config import, not to a `monitor.alert_webhook_url` supplied via YAML/env; the `_is_allowed_domain` TLD heuristic (only `.com/.org/.net/.io/.co`) is brittle (`Configuration_System.py:106-138`, `Configuration_System.py:140-147`, `Configuration_System.py:1529-1530`).

#### Platform (Windows)

- [ ] 🟢 Confirm `to_yaml`/`to_json` atomic write (`with_suffix(suffix + '.tmp')` + fsync + `replace`) is correct on Windows for both extensions and leaves no orphaned `.tmp` under AV/file-lock races (`Configuration_System.py:258-330`, `Configuration_System.py:332-379`).

---

## Dataset loader + masks

#### Data integrity & labeling

- [ ] 🔴 Tag delimiter mismatch (`parse_tags_field` comma-split vs documented space-delimited sidecars) collapses every multi-tag string into one UNK token, near-empty labels routed to error samples [BUG?] (`utils/metadata_ingestion.py:13/22`, `dataset_loader.py:2079`, `utils/metadata_cache.py:194`).
- [ ] 🟠 Confirm train/val have zero image overlap: the split is by JSON path after `rglob`+shuffle (disjoint FILE paths), but two shards holding sidecars for the same stem can leak one copy into train and one into val; dedup should be by image identity (`sanitize_identifier(stem)`), not path (`dataset_loader.py:2764/2796/2071`).
- [ ] 🟠 Confirm `_map_rating_to_tag` int 0→`rating:general` does not collide with PAD index 0 and that all four rating tags exist in `vocabulary.json`; a missing `rating:general` returns None and the empty-rating guard drops EVERY general-rated image (`dataset_loader.py:2569/2589/2261/2351`).
- [ ] 🟢 Verify PAD index 0 is never set as a positive label: `encode_tags` accepts `0 <= idx < vocab_size`; confirm no real tag maps to index 0 and rating tags never resolve to 0/1 (`vocabulary.py:652`, `dataset_loader.py:2264/1160`).

#### Correctness bug

- [ ] 🔴 Verify every consumer filters `sample['error']==True` before loss/metrics: error samples carry all-zero `tag_labels` and a fully-PAD mask; with Phase-2 gamma_neg=7 an unfiltered error sample strongly biases toward all-negative, and a fully-PAD mask can NaN attention/pooling (`dataset_loader.py:2546/1548/2351/1361`).
- [ ] 🟢 Letterbox padding-mask is intentionally NOT rotated after random rotation, so rotated-corner pad fill is treated as valid; confirm pad_color corners match the letterbox fill distribution and don't bias the model at 448px (`dataset_loader.py:2400/299/320`).
- [ ] 🟢 `process_image_cpu` pastes `pad_color` onto an RGB canvas BEFORE the BGR `.flip(0)`; confirm `pad_color` must be symmetric (or specified pre-flip) so padded pixels stay the intended gray under BGR (`dataset_loader.py:288/2424/2479`).

#### Reproducibility & determinism

- [ ] 🔴 Flip-augmentation epoch `mp.Value` likely frozen at epoch 0 in workers under Windows spawn + persistent_workers; `set_epoch()` updates only the parent [BUG?] (`dataset_loader.py:1908/2164/2179/2215/2287`).
- [ ] 🟠 `WorkerInitializer` seeds stdlib `random` and numpy with `torch.initial_seed() % 2**32`; confirm this value actually differs per worker (it should include the worker_id offset) — if not, all workers share one augmentation RNG stream, correlating ColorJitter/rotation/blur/erasing (`dataset_loader.py:857/869`).
- [ ] 🟢 `_deterministic_coin` maps CRC32→[0,1] and flips on `v < prob`; confirm empirical flip rate matches `random_flip_prob` and is uniform across structured (sequential numeric) image_ids — CRC32 has weak uniformity (`dataset_loader.py:2183/2215`).
- [ ] 🟢 Rotation/blur/ColorJitter use stdlib `random.*` seeded once per worker at startup, not per-epoch; on mid-epoch resume the augmentation stream restarts rather than continuing, so resumed augmentations differ from the original run (`dataset_loader.py:317/2394/870`).

#### Numerical stability

- [ ] 🔴 A fully-padded image (mask all True, e.g. an error sample setting `padding_mask = ones`) makes every token IGNORE via `pixel_to_token_ignore` (threshold 0.9); if not filtered, masked softmax / mean-over-valid-tokens produces NaN/inf poisoning the batch (`mask_utils.py:94-95`, `dataset_loader.py:2558`).
- [ ] 🔴 Confirm the training path passes `mask_semantics='pad'` to `ensure_pixel_padding_mask`, NOT the default `'auto'`: 'auto' inverts masks whose pad-fraction > 0.5, so heavily-letterboxed tall/wide art at 448px would get its mask INVERTED, marking real content as PAD (`mask_utils.py:48/54`, `dataset_loader.py:293`).
- [ ] 🟢 Confirm `IndependentColorJitter` operates on PIL/uint8 (pre-tensor) and that `RandomErasing(value=0.5)` on the bf16 scaled tensor normalizes to 0.0 as commented and supports bf16 (`dataset_loader.py:2390/2420/1942`).

#### Config & invariants

- [ ] 🟠 Split-cache vs `max_val_samples`: the cache stores the raw 95/5 split, but `max_val_samples` moves "excess" val into train AFTER loading the cache; if `max_val_samples` changes between runs the train/val boundary shifts while the cache header still validates — contaminating the val baseline (`dataset_loader.py:2798/2802/2807`).
- [ ] 🟠 Confirm `image_size % patch_size == 0` is enforced for the dataset path BEFORE training: `process_image_cpu` outputs `image_size`² tensors and `pixel_to_token_ignore` asserts divisibility only when not tracing; a non-multiple size asserts mid-training or mispools the mask silently under compile/trace (`dataset_loader.py:288`, `mask_utils.py:88`, `dataset_loader.py:1861`).

#### Checkpoint & resume

- [ ] 🟠 `ResumableSampler.__len__` returns `max(0, base_len - start_index)`; if iteration is interrupted at a step boundary, the sampler is re-pickled with a non-zero `start_index` that no longer matches the new epoch, shrinking `len(loader)` and throwing off step/scheduler math (`dataset_loader.py:368/378/380`).
- [ ] 🟠 `load_state` zeroes `_start_index` on dataset size change, but the train loop independently computes `sample_offset` from `resume_sample_idx` and calls `set_start_index` AFTER, re-applying an offset that `load_state` just rejected as unsafe (`dataset_loader.py:353`, `train_direct.py:1381/1389`).

#### Distributed / DDP

- [ ] 🟠 `ResumableSampler` is hardcoded `num_replicas=1, rank=0`; if DDP is ever enabled, `set_start_index` applies a single global offset per rank while each rank gets a different subset — resume would skip the wrong samples per rank (`dataset_loader.py:2903/339/368`).

#### Concurrency & thread-safety

- [ ] 🟠 Per-worker `ExclusionManager` writes `cache_exclusions.txt` with `add_exclusion(immediate=True)`; confirm cross-process appends are atomic on Windows or a lost/partial write retries a corrupt image every run or corrupts the file (`dataset_loader.py:2125/2527/1477`).

#### Performance & memory

- [ ] 🟠 When `prebuilt_arrow_table` is provided, confirm the FILTERED `ArrowMetadataAccessor` (not the full combined table) is what gets pickled to workers; otherwise each worker re-mmaps the full table and the filter is lost, inflating per-worker memory (`dataset_loader.py:2007/2020/2040`).
- [ ] 🟠 Confirm `ArrowMetadataAccessor.__getitem__` returns `tags` as a list of strings and that the Arrow `dir` column (the JSON parent) round-trips to a usable image directory where the image is a sibling of the sidecar (`dataset_loader.py:804/2159/2366`, `utils/metadata_cache.py:349`).

#### Error handling & robustness

- [ ] 🟠 `DatasetLoader.__getitem__` calls `_track_error_distribution` to detect tag-correlated corruption bias, but the ACTIVE `SidecarJsonDataset.__getitem__` has no equivalent — confirm this monitoring gap is intentional or port it (`dataset_loader.py:1464/1498/2502`).
- [ ] 🟢 When `len(val_ds)==0`, `val_loader` is None; confirm early stopping and best-checkpoint selection degrade gracefully (they watch `val/f1_macro`) rather than crashing or selecting on stale metrics (`dataset_loader.py:2995/3003`, `train_direct.py:1350`).

#### Platform (Windows)

- [ ] 🔴 `mp.Value('i',0,lock=False)` not nulled in `__getstate__` — may fail to pickle on spawn or yield a disconnected copy (root cause of the frozen-epoch bug) [BUG?] (`dataset_loader.py:2094/2104/1908`).

#### Logging & monitoring

- [ ] 🟢 The "set_epoch never called" warning and `_epoch_was_set` flag live on the worker copy with a stale `mp.Value`, so the warning is misleading; `_stats_queue` force-nulled in workers means flip/orientation telemetry is silently never emitted (`dataset_loader.py:2204/1899/2104`).

---

## Vocabulary + shared mem

#### Data integrity & labeling

- [ ] 🔴 Tag delimiter used to BUILD the vocab (`vocabulary.py` comma-split) vs to ENCODE labels must match the real sidecar format; if space-delimited, the entire vocabulary becomes whole-string multi-tag tokens — internally consistent but degenerate [BUG?] (`vocabulary.py:139-144/717-720`, `utils/metadata_ingestion.py:13-26`, `dataset_loader.py:2079`).
- [ ] 🔴 `_ensure_rating_tags()` is called in build/load/from_json/from_file but NOT in `populate_vocab_from_shared`; if the shared blob's source vocab lacked rating tags, the main process appends them (growing vocab_size) while workers do not — diverging the label-encode vocab from the head-sizing vocab (`shared_vocabulary.py:241`, `vocabulary.py:622/877/943`).
- [ ] 🔴 Confirm rating tags never get a frequency-ranked interior index: `_ensure_rating_tags` appends at the end, but if rating tags ever appear in `tag_counts` they'd be frequency-ranked in `build_from_tag_counts`; mixed-provenance vocab files would then map `rating:explicit` to different indices, mislabeling ratings (`vocabulary.py:622-629/769-770/738-760`).

#### Concurrency & thread-safety

- [ ] 🔴 `populate_vocab_from_shared` overwrites `tag_to_index`/`index_to_tag`/`pad_token`/`unk_token`/`unk_index`/`ignored_tag_indices` from the blob but never runs `_verify_vocabulary_integrity` nor recomputes invariants — a worker trusts a possibly-stale/truncated blob verbatim, unlike every disk-load path (`shared_vocabulary.py:241-283`, `vocabulary.py:1098`).

#### Reproducibility & determinism

- [ ] 🔴 Frequency-cache staleness (`_get_cached_frequencies`) only validates a `_file_count` within `max(10, 0.1%)` tolerance — never mtime/content; an edited sidecar within tolerance reuses stale counts, which change the `(-count, tag)` index ranking and silently mislabel against a model head trained on the old indices (`vocabulary.py:337/372-382/745-748`).
- [ ] 🟠 Three distinct vocab-hash algorithms (schemas full-canonical-JSON, shared_vocabulary sorted-items sha256, vocab_utils newline-joined list) are not interchangeable; a "match" in one says nothing about the others, misleading operators verifying vocab identity across tools (`schemas.py:123`, `shared_vocabulary.py:126/258`, `vocabulary_utils/vocab_utils.py:78`, `vocabulary_utils/vocab_append.py:416`, `train_direct.py:1072`).
- [ ] 🟠 Confirm index assignment is byte-for-byte reproducible: `_count_tags_parallel` uses ThreadPool + `as_completed` + `Counter.update` (order-independent) and `sorted(key=(-count,tag))` (total order); verify no locale/float dependence and that orjson/json parse identical tag strings so `_iter_chunks` ordering can't affect counts (`vocabulary.py:454/497-501/745-748/274-323`).
- [ ] 🟢 `populate_vocab_from_shared` verifies only when both `_vocab_hash` and `_vocab_size` are present, and the 16-char-prefix fallback accepts a truncated legacy hash — weakening tamper detection; confirm no current writer relies on the truncated form (`shared_vocabulary.py:126-128/193-195/258-263`).

#### Correctness bug

- [ ] 🔴 `_verify_vocabulary_integrity` does not check contiguous `0..N-1` indices; a gap makes high-index tags fail `encode_tags` bounds and drop from every label [BUG?] (`vocabulary.py:1098-1166/644-658`, `vocabulary_utils/vocab_append.py:382`).
- [ ] 🟠 `vocab_append.main` rebuilds `tag_frequencies` from the current scan only, zeroing previously-frequent missing tags [BUG?] (`vocabulary_utils/vocab_append.py:399-405`, `train_direct.py:191-254`).
- [ ] 🟠 `vocab_append` appends every new tag with no `min_frequency` filter, inconsistent with the canonical builder [BUG?] (`vocabulary_utils/vocab_append.py:385-393`, `vocabulary.py:745-751/1171-1232`).
- [ ] 🟢 `get_tag_from_index` raises ValueError for any tag matching `tag_<digits>`; a legitimate tag literally named `tag_1` would crash decode during metrics/inference (`vocabulary.py:672-681/1110-1129`).
- [ ] 🟢 SharedMemory size handling is byte-exact ONLY because the original `data_bytes` length is propagated via `shared_vocab_info`; confirm `dataset_loader.py:2677` always passes `shared_vocab_manager.vocab_size` (data length), never `shm.size` (possibly page-padded) — else `json.loads` sees trailing NUL and raises (`shared_vocabulary.py:137-145/169-191`, `dataset_loader.py:2676-2677`).

#### Config & invariants

- [ ] 🔴 `load_vocabulary` special-token backfill: if a loaded JSON lacks `<PAD>`/`<UNK>` it appends them at `len(tag_to_index)`, leaving real tags at index 0/1 and violating the PAD=0/UNK=1 invariant assumed by `ignore_indices=[0]`/`skip_indices=[0]` everywhere — silently never training/scoring those classes (`vocabulary.py:856-864/926-936`).
- [ ] 🟢 Confirm the file-list and frequency caches key on dataset-path identity (`sha1(resolved path)[:16]`) correctly; if the same path is reused for different data, the file-list cache passes the existence sample and the frequency cache passes the count tolerance, feeding outdated counts into index assignment (`vocabulary.py:159-161/164-199/337-397`).

#### Error handling & robustness

- [ ] 🔴 Shared-vocab worker fallback: a failure in `load_from_shared`/`populate_vocab_from_shared` only warns, leaving `_shared_vocab_loaded=False` and the worker proceeding with whatever vocab was pickled; confirm the pickled fallback is ALWAYS a complete, identical vocab — else workers encode labels against different vocab instances (`dataset_loader.py:903-926`).
- [ ] 🟠 `build_from_tag_counts` does not error when `min_frequency` is too high; a stale-zeroed/small dataset can produce a ~6-tag vocab (PAD/UNK/4 ratings), sizing the head to `num_tags=6` with no hard failure — add a sanity floor (`vocabulary.py:738-800/1171-1241`).

#### API & interface drift

- [ ] 🟠 Vocab cache returns `_VOCAB_CACHE[key].copy()` on hit but stores the original on miss; confirm `copy()` includes EVERY field a consumer reads (`_tag_vector_dtype`, `unk_index`, `RATING_TAGS`, `ignored_tag_indices`) — drift between `__init__` and `copy()` is a recurring bug source, and the first caller could mutate the live cached instance before a copy is taken (`vocabulary.py:588-613/1082-1085`).
- [ ] 🟢 `vocab_utils.compute_vocab_hash` builds its ordered list via `index_to_tag.get(i, unk_token)`, inserting `<UNK>` for gaps, so the printed SHA reflects phantom rows and can mismatch a hash from `tag_to_index` — misleading operators (`vocabulary_utils/vocab_append.py:290-293/415-416`, `vocabulary_utils/vocab_utils.py:78-89`).

#### Data integrity & labeling (rating-tag append)

- [ ] 🔴 Confirm rating-tag append indices match what sized the head across ALL provenance (load-time append vs `build_from_tag_counts`); see contiguity/append items above (`vocabulary.py:622-629/769-770/738-760`).

#### Numerical stability

- [ ] 🟢 `encode_tags` returns `bfloat16` multi-hot; bf16 represents 0/1 exactly, but confirm collate/`pin_memory`/`torch.stack` accept bf16 targets and downstream metric code doesn't assume fp32 (`vocabulary.py:546-559/635-659`, `dataset_loader.py:2483`).

#### Platform (Windows)

- [ ] 🟠 Shared-memory cleanup is POSIX-oriented: `_cleanup_shared_memory` re-opens each segment by name to close+unlink, which on Windows (ref-counted, `unlink` ~no-op) can fail; on a crash, spawn workers get their own empty `_SHARED_MEMORY_REGISTRY`, so parent segments may never be cleaned — confirm no leak/noisy resource_tracker on the Windows+spawn setup (`shared_vocabulary.py:36-71/209-238`, `dataset_loader.py:2680`).

#### Security

- [ ] 🟠 `load_vocabulary` accepts `skip_validation=True` "for trusted/cached sources"; confirm no caller passes it on an untrusted/checkpoint-embedded vocab, bypassing placeholder/bidirectional checks and letting a corrupted index map size the head (`vocabulary.py:819-884/54-104`).

#### Checkpoint & resume

- [ ] 🔴 Resume vocab-SHA guard fails open when the SHA returns `'unknown'`/None [BUG?] (`train_direct.py:1070-1085`, `schemas.py:147-161`, `vocabulary.py:1011`).
- [ ] 🟠 `_load_ignore_tags` resolves `Tags_ignore.txt` relative to `vocabulary.py`; if missing it returns an empty set with only a debug log, so ignored tags silently leak into the vocab as real classes — confirm vocab_append uses the SAME ignore source as the canonical builder (`vocabulary.py:54-104/541`, `vocabulary_utils/vocab_append.py:325-333`).

---

## Model architecture

#### Config & invariants

- [ ] 🔴 MLP/projection dropout is un-configurable: YAML `hidden_dropout_prob` is stripped, `VisionTransformerConfig.dropout` has no YAML key, always defaults to 0.1 [BUG?] (`model_architecture.py:132/243-249`, `train_direct.py:626-633`, `configs/unified_config.yaml:35`, `Configuration_System.py:762`).
- [ ] 🟠 `checkpoint_every_n_layers=4` with `gradient_checkpointing=True` checkpoints blocks 0,4,8,12,16, but the YAML comment says "every layer — max memory savings"; confirm 4 is intended (else a 448px run may OOM under the wrong assumption) and that intervening blocks still receive the shared `block_mask` (`model_architecture.py:632-660`, `configs/unified_config.yaml:50`).

#### API & interface drift

- [ ] 🔴 `num_hidden_layers` default 17 disagrees with YAML/spec 18; a defaults-only build produces a 17-layer model that won't load an 18-layer checkpoint [BUG?] (`model_architecture.py:129`, `configs/unified_config.yaml:29`).
- [ ] 🟢 Confirm output dict keys `tag_logits` and `logits` both reference the same already-clamped tensor and no consumer expects a pre-clamp `logits` or a separate `rating_logits` head (ratings are in-vocab) (`model_architecture.py:677-680`).

#### Correctness bug

- [ ] 🔴 Confirm the flex_attention path and the SDPA/ONNX path apply IDENTICAL masking: flex builds `block_mask` only when `attn_kpm.any()`, while SDPA unconditionally builds `sdpa_attn_mask = ~attn_kpm` — divergence would make export/ONNX inference mismatch training silently (`model_architecture.py:610-628/337-355/499-518`).
- [ ] 🔴 Verify pos_embed bicubic interpolation at 320→448: the special-token count is derived circularly (`saved_grid = isqrt(saved_len-1)` assumes exactly 1 special token, then `num_special = saved_len - saved_grid²`); confirm `num_special==1` and correct grids for 401→785 (`training_utils.py:1805-1828`).
- [ ] 🟠 Confirm per-sample `BlockMask` handles mixed batches (some samples padded, some not): `_create_block_mask` is called with the full `(B,L)` `attn_kpm` whenever `.any()` is true batch-wide; verify `mask_mod` indexes `attend_mask[b, ...]` per-sample so non-padded samples aren't over-masked (`model_architecture.py:496-518/627-628`).

#### Numerical stability

- [ ] 🟠 `SafeDropPath` generates its Bernoulli mask in `x.dtype` (bf16): `keep_prob + torch.rand(..., dtype=bf16)` then `floor_()` — the 8-bit mantissa can't represent values near the keep-prob threshold, biasing the realized stochastic-depth rate (drop_path up to 0.20 across 18 layers) and the `x.div(keep_prob)` scaling (`custom_drop_path.py:17-20`, `model_architecture.py:241`).
- [ ] 🟠 Verify `LayerNormFp32` under the active config (`use_fp32_layernorm=False` default): all LayerNorms and the conv patch_embed run in bf16, so the fp32 safeguard is disabled despite the comment claiming "better stability" — for a 250M from-scratch ViT this risks LayerNorm variance error and slow divergence (`model_architecture.py:110-118/153/527-548`).
- [ ] 🟢 Confirm pos_embed interpolation needs no re-normalization and that `x + self.pos_embed[:, :x.size(1), :]` matches the interpolated length; bicubic overshoot at edges (no LayerNorm before the residual add) could destabilize the first Phase-2 blocks (`model_architecture.py:553-555`, `training_utils.py:1821-1827`).
- [ ] 🟢 Confirm `logit_clamp_value` (default 15.0, not in YAML) clamps symmetrically and does not double-clip with ASL's own log-space clipping in a way that flattens gradients for the hard negatives gamma_neg=7 targets (`model_architecture.py:672-675/151`).
- [ ] 🟢 Verify patch_embed re-init (`trunc_normal std=0.02`) runs AFTER `self.apply(_init_weights)` (which sets Conv2d std≈0.0028) and zeroes bias; this ordering is load-bearing and undocumented in tests — a reorder would make patch-embed activations ~7x too small and stall from-scratch convergence (`model_architecture.py:407-415/425-430`).

#### Logging & monitoring

- [ ] 🔴 `_check_numerical_stability` hardcodes `clamp_threshold = 15.0` and the message says "Clamping" though it doesn't clamp [BUG?] (`model_architecture.py:474-481/673-675`).

#### Data integrity & labeling

- [ ] 🟢 `initialize_tag_head_bias` iterates index 0 (PAD) and 1 (UNK), setting bias from `tag_frequencies` (freq 0 → bias ≈ -11.5); confirm PAD/UNK are excluded everywhere the head output is consumed and that PAD freq is never accidentally nonzero (`model_architecture.py:82-95`, `train_direct.py:637-642`).

#### API & interface drift (attention dropout)

- [ ] 🟠 The flex path applies `F.dropout` to V tokens (not attention weights) while the SDPA path passes `dropout_p` to SDPA (true attention-weight dropout); confirm this V-dropout approximation is acceptable and doesn't diverge train vs export, or change regularization vs older checkpoints (`model_architecture.py:286-294/350-355`).

#### Checkpoint & resume

- [ ] 🔴 `ModelMetadata` vocab SHA round-trip: `embed` stores a compact JSON blob but computes SHA over `canonical_vocab_bytes`; `extract` recomputes over the parsed blob. Confirm canonicalization is stable so a re-embedded→extracted vocab never spuriously fails SHA and returns None (which would make the pipeline load a fallback vocab and shift all tag indices) (`model_metadata.py:84-100/144-164`, `schemas.py:100`).
- [ ] 🟢 Gradient-checkpointing closure `create_block_forward(block, block_mask)` binds both as arguments per iteration (correct); confirm no variant captures the loop variable across iterations (classic late-binding bug) (`model_architecture.py:648-658`).

#### Data integrity & labeling (preprocessing fallbacks)

- [ ] 🟠 `extract_preprocessing_params` legacy path defaults `image_size=512` (model is 448) and `color_order='RGB'`; confirm any consumer re-validates against the embedded model config rather than trusting these defaults (512%16==0 passes but pos_embed/token count mismatches the 448 model) (`model_metadata.py:201-224`).
- [ ] 🟢 Confirm `color_order` embedded matches the order used to compute `normalize_mean/std` at train time so inference reproduces preprocessing exactly (symmetric 0.5 today, but a BGR run with asymmetric stats would mispreprocess) (`model_metadata.py:170-199`).

#### Reproducibility & determinism

- [ ] 🟠 `create_model()` two-layer kwarg filtering (`_unused_config_keys` + create_model field filter) can silently drop a renamed/new config field to defaults with at most a warning — the exact failure mode behind the `hidden_dropout_prob` no-op; confirm every field that SHOULD come from YAML survives (`train_direct.py:619-633`, `model_architecture.py:741-758`).
- [ ] 🟢 `create_block_mask` is called with `_compile=_TRITON_AVAILABLE` (cached at import); confirm Triton-compiled vs eager mask paths produce identical masks so the same checkpoint reproduces across machines with/without Triton (`model_architecture.py:28-38/505-518`).

#### Error handling & robustness

- [ ] 🟢 The all-keys-masked guard only runs when `check_numerical_stability` is True and not onnx_mode; confirm `token_ignore_threshold=0.9` plus a fully-padded image cannot collapse attention to CLS-only (degenerate garbage logits) silently when the debug flag is off (`model_architecture.py:594-603`).

---

## Loss + metrics

#### Evaluation & metrics

- [ ] 🔴 `_drop_zero_positive_classes` filters zero-positive classes before micro-F1 AND mAP, inflating them and breaking cross-draw comparability [BUG?] (`evaluation_metrics.py:116/129-148/121-126`).
- [ ] 🔴 `_drop_zero_positive_classes` early-returns full tensors when `effective == preds.size(1)`, flipping the macro-F1/mAP definition draw-to-draw and producing a noisy early-stopping signal [BUG?] (`evaluation_metrics.py:144-148`).
- [ ] 🟢 `find_optimal_threshold` optimizes f1 computed AFTER `_drop_zero_positive_classes`, so the "optimal" threshold is selected on a filtered class set that differs from inference — biased for rare/zero-support classes at deployment (`evaluation_metrics.py:185/239-246`).
- [ ] 🟠 Confirm `mAP_average='macro'` default is intended given the plan to move auto-stop to `val/mAP`: macro mAP over the draw-dependent filtered class set is as unstable as macro-F1 (`evaluation_metrics.py:55`, `train_direct.py:610-613`, `validation_loop.py:417-420`).

#### Numerical stability

- [ ] 🟠 Confirm the loss forward's log-space math (`torch.log((1-probs).clamp(min=1e-6))`, `exp(gamma*log(...))` clamped at MAX_EXP=88) is acceptable in bf16: the `_LOG_FLOOR=1e-6` reasoning assumes fp32, and gamma_neg=7 amplifies error on a 7-bit mantissa — force-cast to fp32 if needed (`loss_functions.py:303-315/281`).
- [ ] 🟠 With clip=0.2, gamma_neg=7: confident negatives `p<=0.2` → neg_exp ≈ `7*log(1e-6) = -96.6` clamped to -88 → `exp(-88)≈0` (denormal/underflow → exactly 0 in bf16); confirm zeroing easy-negative gradient is intended and the active negative band (`p in (0.2, ~0.5)`) doesn't starve borderline negatives under the Phase-2 logit shift (`loss_functions.py:288-315`).
- [ ] 🟢 Confirm Adan denom order matches the paper (single line 322, multi 382): `sqrt(n_t)/sqrt(bc3)` THEN `.add_(eps)` so eps is not scaled by `sqrt(bc3)` — listed here for cross-path equivalence (`loss_functions`/`adan_optimizer.py:322/382`).

#### Correctness bug

- [ ] 🟠 `reduction='mean'` divides `focal_loss` by `(B * kept_classes)`, so loss magnitude depends on `num_classes` and on how many indices are ignored — coupling the effective LR to vocab size and ignore-set size; confirm the Phase1→Phase2 / vocab-regen scale shift is intended (`loss_functions.py:360-366`).
- [ ] 🟠 `class_weights` multiply `focal_loss` but the mean denominator is the unweighted element count (`mean(loss*weight)`, not a weighted mean); rare-tag up-weighting is diluted by the fixed denominator — confirm this matches the WSML rare-positive intent (`loss_functions.py:358/361-366`, `train_direct.py:220-225`).
- [ ] 🟢 Confirm focal gating uses pre-smoothing `targets_for_focal` (line 269) for pos/neg masks while smoothed targets feed BCE; moot at `label_smoothing=0.0` (Phase 2) but verify the separation is correct if smoothing is ever re-enabled (`loss_functions.py:265-278/314-315`).
- [ ] 🟢 Confirm all-negative rows (zero positives after PAD/UNK removal) contributing only `neg_loss` with the full element count in the denominator is intended given alpha/gamma in the missing-positive-bias regime (`loss_functions.py:314-323/366`).

#### Concurrency & thread-safety

- [ ] 🟢 The class-level `_keep_mask_cache` (ClassVar dict, LRU evict via `next(iter(...))`/`del`) is shared across all instances/threads; eviction is not thread-safe (concurrent miss → KeyError/lost entry), though the key includes the ignore set so correctness holds (`loss_functions.py:35-36/246-255`).

#### Performance & memory

- [ ] 🟢 Confirm the keep-mask LRU (cap 100, key `(num_classes, device, ignore)`) never thrashes; verify the dimension-mismatch truncate path in `validation_loop` never reaches the loss with varying `num_classes` (`loss_functions.py:246-259`, `validation_loop.py:862-870`).

#### Data integrity & labeling

- [ ] 🟠 Index alignment across FOUR masks: loss `ignore_indices=[0,1]`, metric `skip_indices=[0,1]`, class_weights zero at `idx<=1`, PAD=0/UNK=1 — all independent hardcoded constants with no single source of truth; a vocab change breaks the invariant silently (`train_direct.py:760/612/197-199`, `validation_loop.py:420`).
- [ ] 🟢 All metric methods binarize targets via `(targs>0.5).long()`; safe for multi-hot {0,1} today, but any future soft-label/mixup would be silently binarized, decoupling metric ground truth from the training target (`evaluation_metrics.py:100-103/271-274`).

#### Evaluation & metrics (threshold/precision)

- [ ] 🟠 Threshold-boundary semantics: torchmetrics uses `>=`, this code uses `>` in per-tag/calibrator paths [BUG?] (`evaluation_metrics.py:299/533/588/118-123`).
- [ ] 🟠 `compute_bucketed_metrics` lacks the fp32 cast that `compute_all_metrics` has [BUG?] (`evaluation_metrics.py:402/97`).
- [ ] 🟢 `compute_per_tag_metrics` does not cast preds to fp32 and reports `support` as a float `.item()`; confirm the bf16/threshold edge and float-count are acceptable (`evaluation_metrics.py:269/299-317/326`).
- [ ] 🟠 `FrequencyBucketMetrics`/`ThresholdCalibrator` silently drop out-of-range-frequency tags [BUG?] (`evaluation_metrics.py:376-379/568-571/356-368`).
- [ ] 🟢 `find_optimal_threshold` grid `[0.1..0.9 step 0.05]` and calibrator `np.arange` never land on 0.2653, so "optimal" is grid-quantized and may report worse-than-default; `best_threshold` defaults to 0.5 on degenerate ties (`evaluation_metrics.py:233-246/499`).

#### Error handling & robustness

- [ ] 🟠 `ThresholdCalibrator.calibrate` omits `.cpu()` before `.numpy()` — crashes on CUDA tensors [BUG?] (`evaluation_metrics.py:493-494`, `train_direct.py:2212`).

#### API & interface drift

- [ ] 🟠 `MultiTaskLoss.forward` returns `(tag_loss, losses_dict)` with `losses_dict` having `total` and `tag_loss`; confirm the train loop uses the returned scalar (first element), reads the right key, and that gradient accumulation divides exactly once (a prior "loss double division" bug is on record) (`loss_functions.py:387-417`).

#### Logging & monitoring

- [ ] 🟢 `.item()` on every metric forces a CPU sync; confirm `multilabel_average_precision` returning a non-exception NaN (degenerate column) is surfaced rather than silently propagating NaN mAP to logging/early-stopping (`evaluation_metrics.py:124-127/193-196`).

---

## Train loop core (train_direct.py)

#### Correctness bug

- [ ] 🔴 Epoch-boundary accumulation flush under-weights the partial window (no `accum/accum_count` rescale) [BUG?] (`train_direct.py:1557-1563/1921-1953`).
- [ ] 🟠 Soft-stop `carry_accum` across epochs mixes two epochs' loss accounting and changes flip decisions mid-window [BUG?] (`train_direct.py:1356-1369/1960-1976`).

#### Logging & monitoring

- [ ] 🟠 `monitor.log_step` (line 1812, guarded by `global_step % logging_steps == 0`) runs for ALL micro-batches in the accumulation window using the unincremented `global_step` while `loss_item` is only populated on the final micro-batch via `anticipated_step` — producing duplicate/empty-loss (`loss_item=None`) log points; confirm `loss_item` is never None at line 1812 (`train_direct.py:1523-1533/1811-1820`).

#### Numerical stability

- [ ] 🟠 The periodic pre-backward NaN check and the post-extraction NaN check discard the ENTIRE accumulation window (`optimizer.zero_grad` + `accum_count=0`) on a single bad micro-batch, wasting already-good accumulated gradients and, under frequent NaNs, stalling progress — decide if wiping the window is intended (`train_direct.py:1491-1511/1536-1551`).
- [ ] 🟢 `scaler.update()` (in-loop NaN skip, line 1506) and `scaler.unscale_()` (epoch flush, line 1927) are unconditional, unlike the in-loop step path guarded by `if use_scaler`; no-ops for bf16 today, but `update()` without a preceding `step()` would corrupt GradScaler scale tracking if fp16 is re-enabled (`train_direct.py:1505-1507/1576-1577/1927`).

#### Data integrity & labeling

- [ ] 🟠 `validate_dataset` is a no-op placeholder while `validate_input_data` implies real validation [BUG?] (`train_direct.py:587-592`, `dataset_loader.py:1765-1767`).
- [ ] 🟢 Confirm `tag_labels` dtype into the loss: labels transfer as-is and ASL casts 2D targets to `logits.dtype`; under bf16, confirm clean {0,1} multi-hot are not rounded in a way that mis-gates `targets_for_focal` (0/1 are exact in bf16, but fractional/smoothed values would corrupt gating) (`train_direct.py:1439`, `loss_functions.py:208-269`).
- [ ] 🟢 Error-sample filtering only indexes tensors with `size(0)==len(error_flags)`; list-type per-sample fields like `image_id` are NOT filtered, so logged predictions can be attributed to the wrong image IDs — confirm no downstream consumer indexes labels by image_id position (`train_direct.py:1415-1429/1859-1876`).

#### Reproducibility & determinism

- [ ] 🟠 TF32 enablement (lines 396-399) runs whenever CUDA capability>=8 regardless of `config.training.deterministic`, while `use_deterministic_algorithms(True)` is also set — TF32 matmuls are non-deterministic at the bit level, undermining the requested deterministic mode (`train_direct.py:387-399`).
- [ ] 🟢 Mid-epoch resume is not bit-reproducible: persistent-worker RNG can't be serialized, and the fallback batch-skip path loads+discards batches, advancing worker RNG differently than the original run (`train_direct.py:1119-1125/1397-1412`).

#### Config & invariants

- [ ] 🟢 Loss `reduction='mean'` over `(B, C_kept)` interacts with accumulation when error-sample filtering shrinks B: dividing every micro-batch by the same `accum` over-weights smaller (filtered) batches vs a true global mean (`train_direct.py:1417-1429/1559`, `loss_functions.py:360-366`).
- [ ] 🟢 SGDR step budget: `first_cycle_steps = total_updates // num_cycles` truncates and `updates_per_epoch` uses ceil-division; confirm the cosine cycle boundaries don't desync from the actual update count given the per-non-divisible-epoch extra flush step (`train_direct.py:837-864/1940-1944`).

#### Checkpoint & resume

- [ ] 🟠 Mid-epoch resume sample offset: `start_step = sample_offset // train_loader.batch_size` and `set_start_index` expects a SAMPLE index; confirm `sample_in_epoch` was stored as a sample count with the CURRENT batch_size, not a legacy one (`train_direct.py:1374-1390/1649-1650`, `dataset_loader.py:339-382`).
- [ ] 🟠 Best-metric early stopping watches `val_f1_macro`, computed only over positively-supported classes and floored by the fixed 0.2653 threshold under Phase 2's logit shift; confirm `is_best`/patience still select a meaningful best (project note recommends `val/mAP`) (`train_direct.py:2138-2147/2300-2347`).
- [ ] 🟢 Confirm scaler state save/restore is meaningful for bf16's disabled GradScaler and that an fp16 resume (if ever enabled) restores scale correctly (`train_direct.py:925-940/1653-1666`).

#### Concurrency & thread-safety

- [ ] 🟢 The keyboard-listener daemon thread reuses `soft_stop_event` as its stop_event, which is never set on normal completion — on Unix the terminal may be left in cbreak mode and the thread runs until process exit; a soft stop also kills the hotkey thread (conflated concerns) (`train_direct.py:291-375`).
- [ ] 🟢 On Windows, `signal.signal(SIGTERM, ...)` delivery is limited; the documented soft-stop-via-signal may not fire, hard-killing training mid-step and leaving a partial checkpoint (`train_direct.py:293-309/1686-1747`).
- [ ] 🟠 `config.data.stats_queue = stats_queue` (an `mp.Queue`) is assigned to the config and shipped to spawn workers; confirm it pickles/transfers correctly on Windows and that `create_dataloaders` does not deep-copy/serialize the config in a way that breaks the queue handle (`train_direct.py:516-532`).

#### Performance & memory

- [ ] 🟢 `_saved_tag_logits = outputs['tag_logits'].detach()` (line 1555) is created EVERY step even when not logging images; `images`/`tag_labels`/`batch` are never `del`'d and persist until the next iteration — a detached `(B, ~19K)` tensor plus full image batch held longer than needed (`train_direct.py:1555/1565-1571/1856-1877`).
- [ ] 🟠 Validation accumulates full `(N, num_tags)` `all_val_probs`/`all_val_targs` on CPU for bucketed metrics/calibration; ~30k × ~19K is multi-GB of host RAM held until epoch end, risking OOM during validation against the 115/125 GB budget (`train_direct.py:2117-2118/2171-2237`).
- [ ] 🟢 Confirm the H2D stream pipelining actually overlaps: each step issues the transfer on `h2d_stream` then `current_stream().wait_stream(h2d_stream)` before the forward, serializing transfer+compute within the step unless the transfer was issued on the previous iteration (`train_direct.py:1431-1445/2085-2097`).

#### Distributed / DDP

- [ ] 🟠 Single-GPU vs DDP: the loop divides loss by `accum` with no gradient all-reduce or world_size handling, and `set_epoch` only fires for a `DistributedSampler`, yet there is no DDP model wrap — running under torchrun would train replicas without gradient averaging, silently diverging; confirm this file is single-process only (`train_direct.py:1340-1342/1575-1622`).

#### API & interface drift

- [ ] 🟠 Confirm `model(images, padding_mask=pmask)` matches the current `VisionTransformer.forward` signature and that True=PAD token-ignore is honored (cast to bool, pooled to token level); a dropped/renamed kwarg would be silently ignored and attention would attend to letterbox-pad tokens (`train_direct.py:1433-1463/2086-2101`).

#### Evaluation & metrics

- [ ] 🟠 Verify metric column alignment: streaming metrics slice `probs[:, 2:]`/`targs[:, 2:]` (`skip_metric_cols=2`) while bucketed metrics use full-width tensors with `skip_indices=[0,1]` and loss uses `ignore_indices=[0,1]`; confirm sliced index `i` maps to vocab `i+2` everywhere and no off-by-two corrupts per-class F1/mAP attribution (`train_direct.py:2110-2147/2178-2214`).

---

## Train orchestration (train_direct.py setup)

#### Checkpoint & resume

- [ ] 🔴 Phase1(320)→Phase2(448) `latest`/`best` resume silently restarts from scratch (strict image_size mismatch caught and skipped, pos_embed interpolation never runs) unless `prepare_phase2.py` rewrote the embedded config [BUG?] (`train_direct.py:1043-1066`, `training_utils.py:433-462/553-563/1777-1834`).
- [ ] 🟠 Explicit-path resume (`resume_from=<path>`) with an image_size mismatch RE-RAISES the strict ValueError (hard crash) instead of skipping — inconsistent with latest/best (skip) for the same incompatibility (`train_direct.py:1050-1059`, `training_utils.py:558-563`).
- [ ] 🟠 Mid-epoch resume sample offset uses the CURRENT `train_loader.batch_size` while `sample_in_epoch` was saved as `step*batch_size`; if batch_size changed between runs, resume skips the wrong number of samples despite the "batch-size-agnostic" comment (`train_direct.py:1374-1390/1649-1650/1716-1717`).
- [ ] 🔴 `best_metric` is reconciled from checkpoint metrics ONLY when `ckpt['is_best']` is True; resuming a periodic/last checkpoint (is_best False) relies on `TrainingState.from_dict` carrying `best_metric` — if it resets to `-inf`, the next epoch is unconditionally "best", overwriting `best_model.pt` with a worse model and resetting patience (`train_direct.py:1088-1096`, `training_utils.py:660/701-712`).
- [ ] 🟠 Confirm a guaranteed per-epoch "last" checkpoint exists: with `save_best_only=false` and `save_steps=10000`, a non-improving epoch not crossing a save_steps boundary writes no checkpoint, so a crash loses up to ~10000 steps (`train_direct.py:1636-1674/2318-2343`).
- [ ] 🟠 `start_epoch` arithmetic: mid-epoch uses `ckpt['epoch']-1`, boundary uses `ckpt['epoch']` directly (both clamp negative to 0); a 0-based or missing `epoch` causes an off-by-one that silently re-runs or skips a full epoch (`train_direct.py:1102-1127/1645/2003`).

#### Evaluation & metrics

- [ ] 🔴 Early-stopping patience advances on validation-SKIPPED epochs (cached stale val metrics drive the patience block) [BUG?] (`train_direct.py:2034-2059/2300-2347`).
- [ ] 🟠 Macro-F1/mAP support filter (`keep_classes = val_pos_counts>0`) is what early stopping watches; the subsample is fixed so support should be stable, but confirm stability (instability → noisy early stopping); project notes recommend `val/mAP` (`train_direct.py:2138-2148/2307-2314`).

#### Numerical stability

- [ ] 🟢 `scaler.update()`/`scaler.unscale_()` called on the disabled bf16 scaler — no-op today, but the epoch-boundary flush ignores `use_scaler` entirely unlike the main step (`train_direct.py:1506/1576-1618/1927-1937`).
- [ ] 🟢 `_compute_class_weights` indexes `active_pre_clip[n//100]`, `[9*n//10]`, `[99*n//100]` without bounds checks; currently dead (`class_weight_strategy=null`) but wired — confirm indices never reach `len(active_pre_clip)` for small vocab (`train_direct.py:227-236/742-751`).

#### Correctness bug

- [ ] 🟠 Epoch-boundary partial-accumulation flush uses under-scaled gradients and still increments `global_step`/steps the scheduler [BUG?] (`train_direct.py:1557-1563/1921-1945`).
- [ ] 🟠 `carry_accum` skips the per-epoch reset of `running_loss`/`processed_batches`/`total_train_samples`, mixing two epochs in the carry-epoch avg [BUG?] (`train_direct.py:1356-1369/1960-1976`).

#### Reproducibility & determinism

- [ ] 🟠 Confirm RNG state (torch/numpy/python + per-worker dataloader RNG) is restored on resume: `setup_seed` runs once at startup from `config.training.seed`; nothing reseeds to the checkpoint's RNG state, and persistent-worker RNG can't be serialized (`train_direct.py:377-378/1120-1125`).
- [ ] 🟢 The val subset uses `np.random.RandomState(config.training.seed)` independent of resume (fixed across runs); confirm the fixed ~30k draw is representative for early-stopping on `val/f1_macro` given the long-tailed vocab (most classes have zero positives in any draw) (`train_direct.py:537-578/2138-2153`).

#### Config & invariants

- [ ] 🟠 Confirm `model.image_size` matches `data.image_size` when `to_dict()` is read (relies on `validate()` having synced); a stale 448 model vs differing data feed → token/pos-embed mismatch at first forward (`train_direct.py:619-633`, `Configuration_System.py:1758-1776/2340-2348`).
- [ ] 🟠 `MultilabelF1Score` threshold is pulled from `threshold_calibration.default_threshold` (fallback 0.5), NOT `inference.prediction_threshold`; if they diverge, streaming val F1 and bucketed metrics report at two thresholds (`train_direct.py:1297-1311/2177-2189`).
- [ ] 🟢 `AsymmetricFocalLoss ignore_indices=[0,1]` is hardcoded (PAD+UNK), contradicting the documented `[0]` (PAD only) and absent from the YAML; confirm excluding UNK from loss is desired and vocab has no real tag at index 1 (`train_direct.py:753-763/610-613/1299`).
- [ ] 🟠 Confirm `experiment_name` normalization (`_normalize_experiment_name` appends `_vit`) doesn't split the resume checkpoint_dir from where checkpoints are written; legacy `latest`/`best` fallback only triggers when `had_arch_token` was False, so a run that previously had the arch token loses its legacy fallback and starts fresh (`train_direct.py:266-286/944-950/1021-1031`).

#### Concurrency & thread-safety

- [ ] 🟢 On Windows, `msvcrt.getch()` in a daemon thread races with SIGINT trapped to `soft_stop_event`; confirm Ctrl+C reliably sets the event and the daemon thread doesn't swallow the interrupt or block process exit (`train_direct.py:293-374/305-309`).

#### Distributed / DDP

- [ ] 🟢 `compute_effective_batch_size()` (which references the missing `world_size`) would AttributeError; train_direct computes effective batch inline (line 778) — confirm no setup codepath invokes `compute_effective_batch_size()` (`train_direct.py:778-784/1340-1342`).

#### Error handling & robustness

- [ ] 🟠 Broad `except Exception` around `scheduler.step()` only logs a warning while `global_step`/`optimizer_updates` already incremented; a persistent failure trains at a frozen/wrong LR, decoupling the schedule from updates with a warning easy to miss at 10000-step cadence (`train_direct.py:1623-1627/1941-1944`).
- [ ] 🟢 Error-sample filtering leaves list-type per-sample fields (`image_id`) unfiltered while tensors are filtered — cosmetic mislabel in image logging; confirm no downstream consumer indexes labels by image_id position (`train_direct.py:1416-1429/1860-1876`).

#### Logging & monitoring

- [ ] 🟠 Confirm `monitor.log_step` receives a valid loss at a logging boundary not predicted by `should_log` (which uses `anticipated_step = global_step+1` only at the accumulation boundary); a missed prediction leaves `loss_item=None` and logs a None loss (`train_direct.py:1523-1533/1811-1820`).

#### Platform (Windows)

- [ ] 🟢 `_shutdown_dataloader_workers` reaches into `loader._iterator._shutdown_workers` (private API) and `stats_queue.join_thread()`; fragile across torch versions on Windows spawn — confirm the original train_loader workers are cleaned on all exit paths, not just the rebuilt val_loader (`train_direct.py:35-50/558-573/2400-2414`).

---

## training_utils: checkpoint / state / scheduler

#### Checkpoint & resume

- [ ] 🔴 `best_model.pt` copy happens in the async worker-thread callback after the numbered file lands; lost on async-save failure / short shutdown timeout / pre-drain exit [BUG?] (`training_utils.py:1457-1499`, `train_direct.py:2320-2343`).
- [ ] 🔴 On is_best with the async writer, the numbered+best save is async but last.pt is synchronous; `wait_pending(timeout=60)` is shorter than the writer's documented 30-90s save / 300s default, so a large final best can time out and the best copy never runs [BUG?] (`training_utils.py:889-893/1012-1051`, `train_direct.py:2367-2368`).
- [ ] 🟠 `_cleanup_old_checkpoints()` runs before the queued async file exists, dropping the just-appended path from tracking and over-pruning [BUG?] (`training_utils.py:1465-1475/1528-1588`).
- [ ] 🟠 `batch_in_epoch` off-by-one between periodic (`step`) and soft-stop (`step+1`) saves breaks exact resume [BUG?] (`train_direct.py:1648-1650/1716-1717/1777-1778`).
- [ ] 🟠 Scheduler resume `load_state_dict` restores ALL attributes from the checkpoint, overriding the freshly-constructed Phase-2 scheduler's `first_cycle_steps`/`warmup_steps`/`max_lr`; confirm a Phase-2 resume is NOT given a Phase-1 scheduler whose state clobbers the intended schedule (`training_utils.py:1949-1998`).
- [ ] 🟠 `load_state_dict(strict=True)` after pos_embed interpolation and rating_head removal; confirm no benign key delta (added buffer, renamed norm) triggers the catch-all "Architecture mismatch" RuntimeError and falsely aborts an otherwise compatible resume of a 250M model (`training_utils.py:1836-1860`).
- [ ] 🟠 `_safe_load_checkpoint` falls back to `weights_only=False` on `pickle.UnpicklingError`; confirm the checkpoints this project writes (config stored via `to_dict`) actually load with `weights_only=True` so the unsafe fallback is never routinely hit (`training_utils.py:1313-1318/1618-1641`).
- [ ] 🟢 `load_checkpoint` silently deletes any state_dict key containing `rating_head`; confirm no current parameter name contains that substring (deletion happens before strict load, so it can't be flagged) (`training_utils.py:1770-1775`).
- [ ] 🔴 Vocab-SHA guard only fires when `expected_vocab_sha256` is provided AND the checkpoint embeds `vocab_sha256`; a missing embedded SHA only warns and proceeds, letting the tag head address wrong indices after vocab regen — confirm checkpoints always embed it [BUG?] (`training_utils.py:1379-1386/1726-1739`, `train_direct.py:1078-1085`).
- [ ] 🟠 `best_metric` persistence: resume reads `ckpt['val_f1_macro']` but periodic/soft-stop saves write only `{'train_loss':...}`; confirm the fallback to `training_state.best_metric` works so a worse epoch doesn't overwrite the true best — the dual source of truth (metrics dict vs training_state) is fragile (`train_direct.py:1091-1094/2337`, `training_utils.py:1288`).

#### Logging & monitoring

- [ ] 🔴 The critical RNG-restore `RuntimeError` (line 2090) is caught and downgraded by the broad `except` (line 2099), so resume silently continues with wrong data order — the guard is defeated [BUG?] (`training_utils.py:2009-2102`).

#### Concurrency & thread-safety

- [ ] 🔴 `AsyncCheckpointWriter` swallows worker-loop exceptions, `_last_error` is never inspected by train_direct, and an exception between `queue.get()` and the inner try can leave `pending_count` stuck → hung `wait_pending` and no usable checkpoint [BUG?] (`training_utils.py:920-973/1059-1062`, `train_direct.py:2367-2368`).
- [ ] 🔴 Confirm model/optimizer state is snapshotted (cloned) on the main thread before the writer runs: `state_dict()` returns live GPU views and `_deep_to_cpu` does `detach().cpu().clone()`; verify `save_async` is NEVER reached with un-cloned tensors and that `_deep_to_cpu` completes before the next `optimizer.step()` mutates weights (`training_utils.py:1159-1174/1283-1291/1471-1472`).

#### Reproducibility & determinism

- [ ] 🟠 Per-worker DataLoader RNG is not captured/restored; worker streams restart each epoch from their epoch-start seed, breaking bit-reproducible mid-epoch resume [BUG?] (`training_utils.py:132-165/2735-2750`).
- [ ] 🟠 RNG restore happens inside `load_checkpoint`, but train_direct then calls `dataset.set_epoch()`/`sampler.set_epoch()` AFTER load and may reseed via `worker_init_fn`; confirm the restored global RNG is not immediately overwritten by post-load seeding (`training_utils.py:2009-2102`, `train_direct.py:1341-1354`).
- [ ] 🟢 `_restore_rng_states` treats CUDA RNG restore failure as non-critical (only warns) while python/numpy/torch_cpu failures raise; confirm GPU RNG not being restored is acceptable for bf16-on-CUDA resume (any cuda dropout would make resume non-reproducible) (`training_utils.py:2080-2102/219-265`).

#### Numerical stability

- [ ] 🔴 `CosineAnnealingWarmupRestarts.get_lr` divides by `(cur_cycle_steps - warmup_steps)` → division by zero/negative for multi-cycle configs where a cycle is shorter than warmup [BUG?] (`training_utils.py:758-768`, `train_direct.py:846-864`).
- [ ] 🟢 pos_embed bicubic interpolation casts `.float()` then back to the saved bf16 dtype; confirm the round-trip doesn't perturb Phase-2 init and that `align_corners=False` matches `prepare_phase2.py` exactly (else the Phase-2 spatial prior differs by which path produced the checkpoint) (`training_utils.py:1799-1828`).

#### Correctness bug

- [ ] 🟠 `CosineAnnealingWarmupRestarts.__init__` increments `step_in_cycle` to 1 during construction (PyTorch's `_LRScheduler.__init__` calls `step()` once), so warmup starts one step in and the initial LR applied is for step 1 — a subtle shift that compounds resume-step accounting (`training_utils.py:742-756/790-815`).

#### Error handling & robustness

- [ ] 🟠 `scheduler.step()` exceptions are caught and only logged; combined with the cosine division-by-zero risk, the LR can silently go NaN/constant for a whole run (`train_direct.py:1623-1627`, `training_utils.py:758-768`).

#### API & interface drift

- [ ] 🟠 `get_optimizer` passes `weight_decay` into bnb `AdamW8bit` AND uses `get_parameter_groups` (which already sets per-group `weight_decay`); confirm the top-level value doesn't override the `0.0` no-decay group (decaying pos_embed/cls_token/bias) — asymmetric vs plain `AdamW` which omits the top-level value (`training_utils.py:2542-2564/2620-2639`).
- [ ] 🟢 `get_parameter_groups` no_decay matches bare substrings `norm`/`_token`/`patch_embed`; confirm these don't accidentally match weights that SHOULD be decayed and that the patch_embed.proj.weight exclusion matches actual module names (`training_utils.py:2620-2644`).

#### Checkpoint & resume (early stopping class)

- [ ] 🟢 The unused `EarlyStopping` class defaults `mode='min'` but the live metric `val_f1_macro` is MAX; confirm no other entrypoint (validation_loop imports from training_utils) instantiates it with defaults — it would "improve" on decreasing F1 (`training_utils.py:817-862`, `train_direct.py:2265-2316`, `validation_loop.py:65`).

#### Config & invariants

- [ ] 🟢 `validate_config_compatibility` skips any critical param when either side is None; confirm `num_labels`/`image_size`/`patch_size`/`architecture_type` are always present in both, else a genuine mismatch is skipped and surfaces only as a cryptic load_state_dict size error (`training_utils.py:446-462/531-547`).

#### Distributed / DDP

- [ ] 🟢 `save_checkpoint` early-returns for non-primary ranks and only rank-0 RNG/sampler state is checkpointed; confirm single-GPU nullcontext (filelock optional) is safe and document that multi-GPU resume would diverge per-rank RNG (`training_utils.py:1100-1104/1336-1347/2702-2709`).

#### Platform (Windows)

- [ ] 🟠 Atomic saves use `tempfile.mkstemp` + `os.replace`; on Windows `os.replace` raises `PermissionError` if the destination is open (AV / TensorBoard / inference reader), so the sync last.pt write can intermittently fail — confirm no orphaned `.tmp` accumulates and repeated failures don't leave no crash-resume pointer (`training_utils.py:936-965/1132-1151/1512-1526`).

#### Performance & memory

- [ ] 🟠 On is_best, the full checkpoint (state_dict + optimizer state for 250M params, AdamW/Adan ≈ 2-3x params) is deep-cloned to CPU at least twice (async/numbered + last.pt) while the async copy is still owned by the worker — confirm host RAM holds two full CPU copies without an OOM/swap stall at checkpoint time (`training_utils.py:1471-1472/1500-1520`).

---

## Optimizer (Adan) + schedulers

#### Correctness bug

- [ ] 🔴 Fused single-tensor Adan computes `p_data_fp32` but may never write it back to `param` — silent weight freeze if `fused=True` [BUG?] (`adan_optimizer.py:463-464/473`).
- [ ] 🔴 Verify the three Adan paths (single/multi/fused) produce numerically identical updates; spot-check the `neg_pre_grad` reset (single line 335 vs multi 403-404 vs fused 439-440/493) — if the fused kernel already overwrites `neg_pre_grad`, the Python reset is wrong/redundant and the optimizer trajectory diverges by foreach/fused flags (`adan_optimizer.py:335/403/439/493`).
- [ ] 🟠 Per-group `step` counting (`group['step'] += 1`) advances bias_correction1/2/3 for params skipped on a step (`p.grad is None`), mis-scaling their next update — benign if all params get grads every step, confirm none are intermittently grad-None (`adan_optimizer.py:203/208/213`).
- [ ] 🟠 First-step `neg_pre_grad` init: confirm grads and `neg_pre_grad` are scaled by the SAME clip so the step-1 diff is exactly 0; flag if Adan `max_grad_norm` is ever set >0 alongside the external `train_direct` clip (double clip) (`adan_optimizer.py:218/229`, `train_direct.py:1601`).
- [ ] 🟢 Confirm grads passed to Adan are cloned to avoid mutating model `.grad` (inner functions reuse `neg_pre_grad`/moment buffers as scratch, read grads only); a mutation would corrupt the post-step grad-norm logging (`adan_optimizer.py:219/311/365`).

#### Numerical stability

- [ ] 🟠 Confirm Adan state (`exp_avg`, `exp_avg_sq`, `exp_avg_diff`, `neg_pre_grad`) is fp32: `torch.zeros_like(p)` ties state dtype to param dtype, and master weights are fp32 today; if weights ever become bf16, moments underflow on a 7-bit mantissa (`adan_optimizer.py:225/137`, `train_direct.py:881`).
- [ ] 🟢 Confirm denom order (`sqrt(n_t)/sqrt(bc3)` then `+eps`) is identical in single (322) and multi (382) paths so eps flooring matches across paths (`adan_optimizer.py:322/382`).

#### Config & invariants

- [ ] 🟠 No double gradient clipping: `train_direct` clips at max_norm=1.0 before `optimizer.step()`, and Adan's `max_grad_norm` defaults to 0.0 and is NOT passed by `get_optimizer`; confirm this fragile invariant holds (both active → effective clip is min, halving effective LR near the threshold) (`adan_optimizer.py:89/154`, `training_utils.py:2584`, `train_direct.py:1601`).
- [ ] 🟠 Decoupled weight decay: confirm the `no_prox=False` proximal form (`param.div_(1 + lr*wd)` after the update) matches the Adan paper and that base wd=0.05 applies via this div form consistently across single/multi/fused, with the no-decay group at 0.0 (`adan_optimizer.py:330/396`, `training_utils.py:2624`).
- [ ] 🟢 Adan beta validation indexes `betas[2]`, so a 2-tuple raises IndexError not a clean ValueError; confirm no path constructs Adan with a 2-element betas (the live path appends `adan_beta3`) (`adan_optimizer.py:99/103`, `train_direct.py:768`).

#### Checkpoint & resume

- [ ] 🟠 Adan `state_dict` must round-trip `group['step']` (drives bias correction); confirm PyTorch's default `state_dict` serializes the custom param-group key, else resume restarts bias correction and spikes the first resumed update [BUG?] (`adan_optimizer.py:204/208`, `train_direct.py:1656`).
- [ ] 🟠 `restart_opt()` sets `group['step']=0` and zeroes moments only for `requires_grad` params, leaving stale moments on frozen params; confirm the Phase1→Phase2 switch fully reconstructs the optimizer or that no params change `requires_grad` across the switch (`adan_optimizer.py:128/130/132`).

#### API & interface drift

- [ ] 🔴 Confirm which scheduler the live path uses: `train_direct.py:856` builds `CosineAnnealingWarmupRestarts` (step-based), NOT `schedulers.LinearWarmupCosineLR` (epoch-stepped by contract); if `create_scheduler('cosine')` is ever used with per-step `.step()`, a 5-epoch warmup finishes in 5 STEPS and LR collapses to eta_min for the rest of the run (`schedulers.py:11/58`, `training_utils.py:2247`, `train_direct.py:856/1624`).
- [ ] 🟢 `LinearWarmupCosineLR` has no `state_dict`/`load_state_dict` override (stateless beyond `last_epoch`, so resume is correct); confirm checkpoint resume logic does not hard-assume the restarts scheduler's `step_in_cycle`/`cycle`/`cur_cycle_steps` keys (`schedulers.py:90`, `training_utils.py:770`, `train_direct.py:1657`).

#### Correctness bug (schedulers)

- [ ] 🟢 `LinearWarmupCosineLR` warmup never reaches `base_lr` (off-by-one at the boundary) [BUG?] (`schedulers.py:63/67/83`).
- [ ] 🟢 `LinearWarmupCosineLR` reaches `eta_min` only at `e >= max_epochs`; confirm intended terminal LR [BUG?] (`schedulers.py:82/84/86`).

#### Numerical stability (live scheduler)

- [ ] 🔴 Confirm the LIVE `CosineAnnealingWarmupRestarts` cannot divide by zero: verify `warmup_steps = warmup_epochs*updates_per_epoch` is always strictly less than `first_cycle_steps = total_updates` (a `warmup_epochs >= num_epochs` config error is not validated and yields NaN/negative LR into Adan) (`training_utils.py:767`, `train_direct.py:847/854`).

#### Distributed / DDP

- [ ] 🟢 Adan's global grad-norm clip computes the norm over LOCAL params only (no all-reduce); inert today (`max_grad_norm=0`) but would break DDP gradient sync if ever enabled per-rank (`adan_optimizer.py:177/184`).

#### Error handling & robustness

- [ ] 🟠 Broad try/except around `scheduler.step()` can mask a real bug for these pure-arithmetic schedulers; a caught exception silently freezes LR — monitor the warning or convert to fail-fast (`train_direct.py:1623`, `training_utils.py:766`, `schedulers.py:74`).

#### Config & invariants (get_last_lr)

- [ ] 🟢 `get_last_lr()` after `optimizer.step()` then `scheduler.step()` reports the LR for the NEXT step, not the one just applied — confirm this one-step lead in logged LR is understood (else LR plots are shifted by one update) (`train_direct.py:1618/2245`, `schedulers.py:58`).

---

## Monitoring (Monitor_log + logging utils)

#### Security

- [ ] 🔴 Webhook SSRF allowlist matches `parsed.netloc` instead of `parsed.hostname` — host-confusion bypass [BUG?] (`Monitor_log.py:85-98`).
- [ ] 🟠 Webhook POST follows redirects by default (redirect/DNS-rebinding SSRF); set `allow_redirects=False` [BUG?] (`Monitor_log.py:313-314`).
- [ ] 🟠 Token-redaction regex `[A-Za-z0-9]{32,}` is both over- and under-inclusive (leaks `-`/`_` tokens, clobbers benign hashes) [BUG?] (`Monitor_log.py:254-255`).
- [ ] 🟠 Confirm the webhook URL (which embeds the Discord/Slack secret token) is never logged: `_resolve_webhook_url`/`_execute_webhook_in_thread` log only the title, but a future `response.raise_for_status()` text can include the URL (`Monitor_log.py:107-113/315-316`).

#### Performance & memory

- [ ] 🔴 `get_timer_stats` slices a `deque` → TypeError on every call, silently aborting `save_metrics` [BUG?] (`Monitor_log.py:468-482/351/491`).
- [ ] 🟢 Confirm `MemoryMonitor` env-var thresholds are documented as absolute GB, not percentages: `MEMORY_WARN_THRESHOLD_GB=90` warns at 90GB regardless of total, silently mis-tuning OOM alerts (`utils/memory_monitor.py:29-49/110-127`).

#### Logging & monitoring

- [ ] 🟠 `TrainingMonitor._setup_logging()` stacks duplicate handlers / leaks file handles on every construction [BUG?] (`Monitor_log.py:926-940`).
- [ ] 🟠 Double-logging between `setup_logging` (root QueueHandler) and `TrainingMonitor._setup_logging` (module-logger handlers that propagate to root) [BUG?] (`Monitor_log.py:926-940`, `utils/logging_setup.py:136-141`, `train_direct.py:2426-2431`).
- [ ] 🟠 `save_metrics` stringifies `np.float64` via `json.dump(default=str)`, corrupting numeric metrics into strings [BUG?] (`Monitor_log.py:484-501/459-466/475-482`).
- [ ] 🟢 `_log_to_backends` produces `epoch/train_loss/epoch` doubled hierarchy [BUG?] (`Monitor_log.py:1122-1140/1621-1627`).
- [ ] 🟢 GPU `memory_used_gb`/`memory_percent` come from `torch.cuda.memory_reserved()` (caching-allocator high-water mark), not true usage, and utilization/temperature are hardcoded 0 — alerts trigger on reserved cache or are dead panels (`Monitor_log.py:762-781/1599-1606`).
- [ ] 🟢 `log_step` step-time uses `(step - last_logged_step)`; on resume the fresh monitor measures wall-clock-since-construction over the step delta, inflating the first step_time and tripping "Slow Training" (`Monitor_log.py:1049-1056/1545-1551`).
- [ ] 🟢 `log_predictions` may pass a non-HWC array to `add_image(dataformats='HWC')` for an unexpected channel count (`img.squeeze()`), and assumes probs/targets are full vocab-aligned vectors for TP/FN/FP labeling (`Monitor_log.py:1255-1269/1271-1301`).

#### Evaluation & metrics

- [ ] 🟠 `log_validation` tracks "best" by LOWER loss while early-stopping uses higher-is-better f1/mAP — inverted semantics [BUG?] (`Monitor_log.py:836/1110-1117`).

#### Data integrity & labeling

- [ ] 🟠 `add_metric` can store/append a NaN/Inf when `_to_safe_float` returns None [BUG?] (`Monitor_log.py:384-412`, `utils/logging_sanitize.py:7-48`).
- [ ] 🟠 `_denormalize_img` reads `config.normalize_mean/std/color_order` with defaults; if MonitorConfig lacks these or they drift from `dataset_loader`, denorm + BGR flip garble TensorBoard preview images (`Monitor_log.py:822-827/1198-1213`).
- [ ] 🟢 `ImageLogger.log_images` uses `make_grid(normalize=True)` (per-batch min-max rescale), NOT the configured denormalization, so preview grids have inconsistent brightness vs `log_predictions` (`Monitor_log.py:142-146/1494-1507`).

#### Concurrency & thread-safety

- [ ] 🟠 SummaryWriter `add_*` is called from multiple call sites with no lock; concurrent validation/image logging overlapping a flush can interleave/corrupt the event file (`Monitor_log.py:905/1168-1175/1621-1630`).
- [ ] 🟠 Confirm the `QueueListener` from `setup_logging` is stopped on ALL exit paths and that calling `setup_logging` more than once doesn't leak listener/queue threads (validation_loop already warns `QueueListener.stop()` can hang with a backlog) (`utils/logging_setup.py:129-141`, `train_direct.py:2442-2446`, `validation_loop.py:746-748`).
- [ ] 🟢 `AlertSystem` has no lock around `last_alert_time`/`alert_counts`; concurrent `send_alert` (training loop + system metrics) during `_prune_old_alerts` can raise "dict changed size" or lose suppression state → alert storms (`Monitor_log.py:162-201/169-184`).
- [ ] 🟢 Confirm `SystemMonitor.stop()` reliably terminates the ThreadPoolExecutor worker mid-flight and cannot emit "cannot schedule new futures" during atexit (`Monitor_log.py:574-603/605-668`).

#### Platform (Windows)

- [ ] 🟠 `psutil.disk_usage('/')` monitors the wrong volume on Windows (not L:) — "Low Disk Space" alert can never fire for the data volume [BUG?] (`Monitor_log.py:730/1577-1588`).

#### Distributed / DDP

- [ ] 🟠 `TrainingMonitor` falls back to `is_primary=True` when dist is not initialized; a monitor built BEFORE `dist.init_process_group` makes every rank primary, each opening a TB writer + file handler (event-file/FD contention) — confirm dist is initialized before construction (`Monitor_log.py:840-871/887-906`, `utils/logging_setup.py:113-127`).

#### Checkpoint & resume

- [ ] 🟢 Confirm atexit cleanup and `close()` are idempotent/ordered: `close()` sets `_closed=True` but the atexit path does not, relying on SummaryWriter/SystemMonitor tolerating repeated stop/close — confirm no exception path leaves the TB event file unflushed (`Monitor_log.py:964-986/1697-1751`).

#### Reproducibility & determinism

- [ ] 🟢 `_get_git_info()` shells out to `git` at module import (two subprocesses), re-running in every spawn worker on Windows — hidden latency and a provenance footgun if the working tree changes mid-run (`utils/logging_setup.py:22-31`).

---

## Validation harness (validation_loop + schemas)

#### Data integrity & labeling

- [ ] 🔴 `validate_specific_tags` mispairs tag names with prediction columns via `zip(specific_tags, tag_indices_cpu)` when any requested tag is missing [BUG?] (`validation_loop.py:1092-1106/1161-1163`).
- [ ] 🟢 Confirm `_save_predictions_standardized` image-id resolution: `metadata[i].get('path', metadata[i].get('image_id', ...))` — `path` is generally absent (falls back to image_id, fine), but a dict-of-lists branch carrying `paths` (plural) yields `image_{i}` placeholders (`validation_loop.py:907-918/1504`).

#### Config & invariants

- [ ] 🔴 `pad_color` never persisted in the checkpoint and never set in the validation `CSDataConfig` — validation always uses the default `(114,114,114)` [BUG?] (`validation_loop.py:640-650`, `model_metadata.py:188-194`, `dataset_loader.py:2622`).
- [ ] 🟠 Confirm `normalize_mean/std` at validation equal training: `_load_model` prefers checkpoint `preprocessing_params`, else `_val_mean/_val_std` from YAML; the FileNotFound/missing-key example text shows ImageNet stats `[0.485,...]` contradicting the project standard `[0.5,0.5,0.5]` — a user copying the example would mispreprocess (`validation_loop.py:176-179/257-269/596-614`).
- [ ] 🟠 Confirm validation `image_size` matches the ACTIVE phase (448): `_DEFAULT_VAL_IMAGE_SIZE=512` and `CSDataConfig` default 512; a legacy checkpoint without `preprocessing_params` validates at 512 while the model was trained at 448, changing token count (1024 vs 784) — a shape/quality error (`validation_loop.py:84/289-298/600-608/643`).
- [ ] 🟢 Confirm the module-local `ValidationConfig` and `Configuration_System.ValidationConfig` (CSValConfig) are not conflated; they carry distinct fields (local: prediction_threshold/seed/max_samples; CS: dataloader/preprocessing) and a refactor could lose settings (`validation_loop.py:60-64/87-141/651-656`).

#### Evaluation & metrics

- [ ] 🔴 Confirm the standalone split matches the in-train split: `create_dataloaders` keys the 95/5 split on `sha1(active_data_path)` with seed=42 by DEFAULT; if `json_dir` or the training seed differs from the run that produced the checkpoint, the harness validates a DIFFERENT (possibly train-overlapping) set (`validation_loop.py:657-666`, `dataset_loader.py:2602-2607/2784-2798`, `train_direct.py:540`).
- [ ] 🟠 Confirm the prediction threshold is consistent: the harness reads dataloader+preprocessing from the YAML but IGNORES `inference.prediction_threshold`, defaulting `ValidationConfig.prediction_threshold` to 0.2653; if that threshold is tuned for training, standalone F1 diverges (`validation_loop.py:108/417-421`, `train_direct.py:610-612`, `configs/unified_config.yaml:330`).
- [ ] 🟠 `f1_macro`/`mAP` from `validate_full()` are NOT comparable across different `--max-samples` because `_drop_zero_positive_classes` changes the effective label set with the draw (`evaluation_metrics.py:116/129-148`, `validation_loop.py:976-979`).
- [ ] 🟠 Confirm `skip_indices=[0,1]` matches train (`train_direct.py` uses `[0,1]`); the orientation doc mentions `[0]` in one place — if train ever used `[0]`, macro/micro F1 denominators differ by the UNK column (`validation_loop.py:417-421`, `train_direct.py:610-612/2186`).
- [ ] 🟠 Per-image CSV: confirm `all_metadata` ordering matches `all_predictions` after any `mismatch_strategy='skip_batch'` subsetting (metadata must be appended only for non-skipped batches), and that pred_binary uses the same 0.2653 threshold (`validation_loop.py:872-918/1537-1555`).

#### Reproducibility & determinism

- [ ] 🟠 Confirm the `--max-samples` subsample seed (`np.random.RandomState(config.seed)`, default 42) reproduces the SAME indices as the in-train subsample (`np.random.RandomState(config.training.seed)`); independent code paths/seeds (`validation_loop.py:140/669-672`, `train_direct.py:538-543`).

#### Concurrency & thread-safety

- [ ] 🟠 Orphaned-worker leak in `validate_fast` (breaks a multi-worker loop without draining) [BUG?] (`validation_loop.py:1050-1077`).
- [ ] 🟠 Confirm the `QueueListener`/`_log_queue` lifecycle: the queue is created unconditionally, the listener starts only for the primary, `__getstate__` nulls the queue for workers, and `_cleanup_logging` drains with `get_nowait()` then closes — confirm no dropped in-flight worker logs or deadlock on early-exception paths (`validation_loop.py:158-159/430-442/737-783/662`).

#### Correctness bug

- [ ] 🔴 Async-transfer hazard: final `non_blocking=True` D2H copies may be `torch.cat`'d before a CUDA sync when `measure_inference_time=False` and batch count isn't a multiple of 50 [BUG?] (`validation_loop.py:900-903/923-924/940-953/966-968`).
- [ ] 🟠 Confirm hierarchical-output flattening order matches between predictions and targets: `validate_full` reshapes `(B,G,T)→(B,-1)` and `validate_hierarchical` reshapes flat `(B, G*T)→(B,G,T)`; if the model's group/tag ordering differs from the flat vocab index order, column `k` pairs incorrectly (only guard is total width) (`validation_loop.py:846-848/1259-1272`).

#### Numerical stability

- [ ] 🟠 Confirm bf16 autocast logits are upcast to fp32 BEFORE sigmoid/threshold: `validate_full` does `torch.sigmoid(logits)` on the bf16 autocast output; sigmoid in bf16 loses precision around 0.2653, flipping boundary predictions vs an fp32 path — verify train and harness apply sigmoid at the same precision (`validation_loop.py:834-884`, `evaluation_metrics.py:96-103`).
- [ ] 🟢 `average_precision_score` in `validate_specific_tags` is called per tag after a >0-positives guard but not a >=1-negative guard; a tag present in every sample yields degenerate AP=1.0 / sklearn warning (`validation_loop.py:1166-1185`).

#### Checkpoint & resume

- [ ] 🟠 Confirm `compute_vocab_sha256(vocab_data=extract_vocabulary())` uses the same canonical bytes as the checkpoint's stored hash; `tag_frequencies` must be present in both (and unreordered) to avoid spurious mismatch warnings / strict-mode failures (`validation_loop.py:347-365`, `schemas.py:100-120`, `model_metadata.py:89-93/148-153`).
- [ ] 🟠 External-vocab hash mismatch only warns (raises only if `strict_vocab_validation`, default False), and the model HEAD width is not verified to equal the external vocab size before running — a wrong vocab of the same length but different ordering silently maps predictions to wrong tag names (`validation_loop.py:373-408/137`).

#### API & interface drift

- [ ] 🟠 `RunMetadata.top_k` is non-Optional `int` but set to None → `top_k: null` in the predictions JSON [BUG?] (`validation_loop.py:1466-1467`, `schemas.py:56-70`).
- [ ] 🟢 `ValidationConfig.create_visualizations` defaults True (dataclass) but CLI `create_plots` defaults False — programmatic vs CLI behavior disagree (surprising plot-dir writes for library callers) (`validation_loop.py:117/1644-1647/1669`).

#### Logging & monitoring

- [ ] 🟢 `ValidationRunner._setup_logging()` stacks duplicate handlers across constructions [BUG?] (`validation_loop.py:487-517`).

#### Error handling & robustness

- [ ] 🟠 Confirm the validation metrics-failure fallback (NaN for f1_macro/f1_micro/mAP) is handled NaN-safely by early stopping: `best < current` is False for NaN (good, no new best), but confirm the no-improvement counter increments correctly and a NaN never triggers a premature stop (`validation_loop.py:980-993`).
- [ ] 🟢 Confirm cleanup runs on the error path: `validate()` calls `_cleanup_logging()` only on success; an exception inside `runner.validate()` leaves dataloader workers (spawned in `create_dataloader`) un-shut-down on Windows (`validation_loop.py:687-735/1690-1699`).

#### Platform (Windows)

- [ ] 🟢 Confirm the H2D stream overlap (`torch.cuda.Stream` + non_blocking + `current_stream().wait_stream`) is correct on Windows/CUDA and that `padding_mask` on `h2d_stream` is fully synced before `model()` reads it (`validation_loop.py:807-835`).

---

## utils: cache / exclusion / path

#### Data integrity & labeling

- [ ] 🔴 Arrow staleness never detects in-place JSON edits (mtime-only sample + path-string-only hash) [BUG?] (`utils/metadata_cache.py:442/60/490`).
- [ ] 🔴 Net-zero/within-tolerance file churn does not invalidate the cache (`max(100, 0.1%)` tolerance) [BUG?] (`utils/metadata_cache.py:479/482`).
- [ ] 🔴 `sanitize_identifier` failures in `_build_arrow_cache` silently drop swaths of the dataset (per-file warn + continue) [BUG?] (`utils/metadata_cache.py:192/204`, `utils/path_utils.py:24`).
- [ ] 🔴 `parse_tags_field` comma-split vs space-delimited sidecars collapses every multi-tag string into one UNK token [BUG?] (`utils/metadata_ingestion.py:22`, `utils/metadata_cache.py:194`, `dataset_loader.py:2079`).
- [ ] 🟠 Confirm exclusions are applied identically in train AND val: each `SidecarJsonDataset` builds its own `ExclusionManager` at `self.root/'cache_exclusions.txt'`; with a shared Arrow table across splits, cross-dataset exclusions propagate only via the shared file + 30s reload, so a sample failing in train can still be fed to val (`dataset_loader.py:1848/2023`, `utils/exclusion_manager.py:301`).

#### Concurrency & thread-safety

- [ ] 🟠 `ExclusionManager.add_exclusion(immediate=False)` (the default) has no flush caller and no atexit flush — non-immediate exclusions of <10 items are lost per worker [BUG?] (`utils/exclusion_manager.py:180/204/232`, `dataset_loader.py:2527`).
- [ ] 🟢 Confirm `reload_if_stale` only GROWS the in-memory exclusion set and the snapshot-based worker init cannot drop exclusions; coarse Windows mtime granularity (~1-2s) plus the mtime-equality short-circuit can cause workers to miss exclusions added within the same mtime tick (`dataset_loader.py:2133`, `utils/exclusion_manager.py:327/313`).
- [ ] 🟢 Confirm the `.arrow.lock` build does not race the `unlink` in `try_load_arrow_cache` (the pre-unlink and meta write are unprotected), creating a window where a concurrent reader sees no cache and falls back to slow parsing or redundant rebuild (`utils/metadata_cache.py:577/387/419`).

#### Platform (Windows)

- [ ] 🟠 `msvcrt.locking` always takes a mandatory EXCLUSIVE lock (ignores the `exclusive` flag); the "shared read fast path" actually serializes/contends [BUG?] (`utils/exclusion_manager.py:386/402/372`).
- [ ] 🟠 Windows exclusion lock region sized from current EOF does not cover bytes appended after acquisition; interleaved/lost lines under concurrent appends [BUG?] (`utils/exclusion_manager.py:404/435/283`).
- [ ] 🟠 Confirm `os.replace(temp, cache_path)` and `unlink` succeed while a prior Arrow cache is memory-mapped open by a worker; on Windows a sharing violation raises `PermissionError`, aborting a force_rebuild / staleness-triggered rebuild (`utils/metadata_cache.py:378/577/247`).
- [ ] 🟠 Confirm `sanitize_identifier` + NTFS case-insensitivity cannot collide two image_ids onto one file or let an excluded id ('IMG1') fail to exclude a differently-cased row ('img1'); exclusion sets and id keys are case-sensitive while the filesystem is not (`utils/path_utils.py:11/103`, `dataset_loader.py:2315`).

#### Reproducibility & determinism

- [ ] 🟠 Staleness sampling is non-deterministic: `_stratified_sample` uses unseeded `random.sample`, so two runs sample disjoint files and a stale cache may pass on one launch and fail on the next — while `_compute_file_list_hash` samples deterministically by sorted path (`utils/metadata_cache.py:103/119/493`).
- [ ] 🟠 Arrow build row order is non-deterministic (`all_items` extended via `as_completed`), so a rebuild reshuffles the dataset index→image mapping, desyncing mid-epoch resume (sample-in-epoch) and cached splits (`utils/metadata_cache.py:315/317/339`).

#### Correctness bug

- [ ] 🟢 `_normalize_exclusion_line` mangles an image_id whose stem literally ends in a known extension [BUG?] (`utils/exclusion_manager.py:64-65`, `dataset_loader.py:2529`).
- [ ] 🟢 Exclusion-file dedup mismatch: writer dedups raw `line.strip()`, reader normalizes to stems [BUG?] (`utils/exclusion_manager.py:276/58/148`).
- [ ] 🟢 `_compute_file_list_hash` sample slices can overlap/duplicate for `total` just above 3000, weakening the 64-bit-truncated fingerprint (`utils/metadata_cache.py:72/84`).

#### Config & invariants

- [ ] 🟠 Cache key isolates datasets (`sha1(root)`) but not config: `parse_tags_field` semantics and `sanitize_identifier` rules are NOT part of `_ARROW_CACHE_VERSION` or the hash, so changing tag-parsing/id-sanitization logic without bumping the version leaves stale, wrongly-parsed metadata that passes all staleness checks (`utils/metadata_cache.py:57/146`, `utils/metadata_ingestion.py:13`).

#### Error handling & robustness

- [ ] 🟠 Confirm a truncated `.arrow` (crash mid-write) whose `.meta` still has the correct count is detected: verify `pa_ipc.open_file` raises (rather than returning fewer rows) and a count mismatch triggers rebuild rather than training on a short table (`utils/metadata_cache.py:243/566/366`).
- [ ] 🟢 If `_do_arrow_write` succeeds but the meta write fails, a new `.arrow` is paired with a missing/old `.meta`; a leftover OLD meta with a matching count could pass validation against the NEW table (`utils/metadata_cache.py:419/556/577`).

#### Logging & monitoring

- [ ] 🟠 `_build_arrow_cache` does not surface a high drop ratio: it logs only "Parsed N items" and `meta['count']` reflects the reduced set, so a large parse/sanitize drop produces a smaller-than-expected cache that validates cleanly — add a drop-ratio warning (`utils/metadata_cache.py:328/330/422`).

#### Concurrency & thread-safety (filelock)

- [ ] 🟠 Confirm `filelock` is a HARD dependency: when absent, `_build_arrow_cache` runs `_do_arrow_write` with NO lock, so concurrent rebuilds (train+val launched together) can interleave temp writes / `os.replace` and corrupt the cache (`utils/metadata_cache.py:42/411/387`).

#### Performance & memory

- [ ] 🟢 Confirm `reload_if_stale` does not stat the file on every batch boundary unnecessarily and that `_load_internal` does not hold the threading lock across the full file open+read+filelock (which would serialize that worker's `__getitem__` and stall the input pipeline) (`dataset_loader.py:1310`, `utils/exclusion_manager.py:316/121`).

#### Security

- [ ] 🟢 Confirm `safe_join`/`validate_image_path` confinement holds for symlinks/junctions and UNC paths on Windows: `_within_any` uses `.resolve()` (follows junctions), and `allowed_external_roots` widens the trusted set to the whole dataset_root — a junction under root could escape (`utils/path_utils.py:42/73/88`).

#### Error handling & robustness (log rotation)

- [ ] 🟢 `CompressingRotatingFileHandler.doRollover` wraps every rename/remove/gzip in bare `try/except pass`; on Windows a rename can fail if another process holds the log open, silently disabling further rotation and letting logs grow unbounded (`utils/file_handlers.py:42/53/67`).

---

## Cross-cutting

#### Config & invariants

- [ ] 🔴 Weight decay is NOT inverse-sqrt scaled by dataset size as documented (LR is scaled, WD is not) [BUG?] (`train_direct.py:794/779`, `training_config.py:161/303`).
- [ ] 🟠 Confirm `total_updates`/`warmup_steps` are computed from the FULL `train_loader` length, not a shrunk `ResumableSampler` length: `set_start_index` runs inside the epoch loop AFTER scheduler construction (so first-epoch is correct), but verify no resumed path makes `len(train_loader)` reflect the shrunk count at line 837 (`train_direct.py:837/856/1389`, `dataset_loader.py:378`).
- [ ] 🟠 Verify PAD=0/UNK=1 ignore consistency across ALL four sites end-to-end (label encode, loss `ignore_indices=[0,1]`, metric `skip_indices=[0,1]`, head-bias init), and that `[0,1]` (not the documented `[0]`) is authoritative including in inference/export (`train_direct.py:760/612`, `model_architecture.py`, `vocabulary.py`).
- [ ] 🟠 Confirm `image_size % patch_size == 0` is enforced AFTER the `FullConfig.validate()` sync and for the ACTIVE phase via a single early, loud check covering data+model+validation together (defaults 512/512/448 rely on the sync) (`Configuration_System.py`, `model_architecture.py`, `mask_utils.py`).
- [ ] 🟠 Confirm the metadata/Arrow cache and split cache are invalidated when image-affecting config changes: the Arrow key omits image_size/patch_size/color_order/normalize/pad_color/tag-delimiter — the cache holds only tags (resolution-irrelevant) but a `parse_tags_field`/`sanitize_identifier` change serves labels under old semantics with no version bump (`utils/metadata_cache.py:57`, `utils/metadata_ingestion.py:23`).

#### Reproducibility & determinism

- [ ] 🟠 torch.compile warmup compiles a different graph (mask=None, training=False) than real steps (mask present, training=True), forcing recompilation on the first real batch [BUG?] (`train_direct.py:1251/1356`, `model_architecture.py:588/634`).
- [ ] 🟢 Acknowledge the full reproducibility chain on resume: flip is reproducible (CRC32), but rotation/blur/jitter use per-worker `random.*` seeded at startup only, and the discard-batches fallback advances them differently — a resumed run is not bit-reproducible vs an uninterrupted one (`training_utils.py:2743`, `dataset_loader.py`, `train_direct.py:1397`).
- [ ] 🟠 Verify graceful, deterministic degradation when Triton/flex_attention is unavailable: the eager flex/SDPA fallback (and skipped torch.compile) must yield identical masks/outputs to the Triton path — divergence between `sdpa_attn_mask=~attn_kpm` (unconditional) and the flex `block_mask` (only when `.any()`) produces different gradients across machines (`model_architecture.py:621/627`, `train_direct.py:724`).

#### Performance & memory

- [ ] 🟠 The flex_attention path's data-dependent `attn_kpm.any()` Python branch and the `.item()` host sync (when `check_numerical_stability` is on) force torch.compile graph breaks / guard-driven recompilation whenever the padding pattern toggles between batches, eroding the speedup and causing latency spikes (`model_architecture.py:627/601`).
- [ ] 🟠 Verify host-RAM growth from the UNION of validation full `(N×~19K)` CPU tensors held to epoch end + the async writer's two CPU clones of model+optimizer state on is_best epochs + SummaryWriter buffers — all coinciding at the same epoch boundary, a host-OOM no single-subsystem reviewer would predict (`train_direct.py:2069`, `training_utils.py:1471/1518`).

#### Numerical stability

- [ ] 🟠 The dataset normalizes and STORES images in bf16 (`(x-0.5)/0.5` at bf16) BEFORE the model's `pixel_values.float()`, so the fp32-patch-embed safeguard cannot recover precision already lost — confirm whether images should be normalized/stored in fp32 and cast to bf16 only at the autocast boundary (`dataset_loader.py:1403/225`, `model_architecture.py:536`).
- [ ] 🟢 Verify `amp_autocast(dtype=bf16)` is not a redundant no-op given the dataset already emits bf16 images/labels (autocast then governs intermediate matmuls, not the input cast) and that this doesn't mask a real fp32→bf16 boundary bug if autocast/amp_dtype changes (`train_direct.py:1462`, `dataset_loader.py:1054`).
- [ ] 🟢 Verify bf16 multi-hot LABELS survive `pin_memory` + non_blocking H2D + the loss target cast without losing exact 0/1 gating; confirm pin_memory of bf16 works on the torch/Windows build and that `label_smoothing=0` keeps `targets_for_focal` exact (`dataset_loader.py:1121`, `train_direct.py:1439`, `loss_functions.py:269`).

#### Error handling & robustness

- [ ] 🔴 STOP_TRAINING sentinel is never unlinked, so a leftover file immediately stops the next run [BUG?] (`train_direct.py:967/1690/1962/1807`).
- [ ] 🟢 Verify an OOM during a forward/backward micro-batch is handled (no try/except around `model()`/backward for `OutOfMemoryError`): a transient OOM at 448px propagates uncaught, abandoning the partial gradient window and any in-flight async checkpoint, possibly leaving `h2d_stream` desynchronized — decide hard-crash-and-resume vs catch+skip (`train_direct.py:1434/1485/1561`).

#### Checkpoint & resume

- [ ] 🔴 Phase1→Phase2 image_size may not be updated in the embedded `preprocessing_params` block that inference/validation prefer [BUG?] (`tools/prepare_phase2.py:74`, `Inference_Engine.py:565`, `model_metadata.py`).
- [ ] 🟠 Verify torch.compile `_orig_mod.` prefix handling is symmetric across save (compiled) AND load (uncompiled — compile applied AFTER load): confirm the prefix add/strip never produces a strict=True key-mismatch RuntimeError that aborts a valid resume (`training_utils.py:1286/1757`, `train_direct.py:1206`).
- [ ] 🟠 Verify behavior when a checkpoint write is interrupted (second SIGINT / disk full during async save or last.pt `os.replace`): confirm atomic-rename + queued-soft-stop guarantee a valid checkpoint exists after any interruption so the next resume doesn't pick a truncated/stale file (`training_utils.py:1515/1137`, `train_direct.py:2368`).

#### Distributed / DDP

- [ ] 🟠 Confirm DDP is either fully wired or fully absent: no DDP model wrap, loss divided by accum with no all-reduce, Adan grad-norm clip local-only, `compute_effective_batch_size` multiplies by the missing `world_size` — half-wired scaffolding invites a torchrun launch with no gradient sync and a crash; hard-assert single-process or correctly all-reduce, and document it (`train_direct.py:1340`, `adan_optimizer.py:154`, `Configuration_System.py`).

#### Platform (Windows)

- [ ] 🟠 Verify the `mp.Queue(maxsize=1000)` assigned to `config.data.stats_queue` survives Windows spawn pickling of the config into DataLoader workers and that no worker dereferences it after it's nulled (the dataset's `__getstate__` nulls `_stats_queue`, but `config.data.stats_queue` is a separate reference path) (`train_direct.py:516/521`, `dataset_loader.py`).

#### Logging & monitoring

- [ ] 🟢 Verify Monitor_log + setup_logging do not stack duplicate handlers or leak writer threads across a single process constructing the monitor more than once (resume / Phase1→Phase2) — records propagate to root and are emitted twice (`Monitor_log.py`, `utils/logging_setup.py`).

#### Evaluation & metrics

- [ ] 🟠 Verify train-time and standalone-validation metrics are comparable across the Phase1→Phase2 switch: both use the fixed 0.2653 threshold + `_drop_zero_positive_classes`, and under Phase 2's logit shift this floors F1 — the auto-stop metric (`val/f1_macro`, patience 4) may trigger/select "best" on a metric that doesn't track real quality (project note: move to `val/mAP`) (`train_direct.py:610/2271`, `evaluation_metrics.py`).

#### Data integrity & labeling

- [ ] 🔴 Tag delimiter consistent end-to-end (vocab build + label encode both comma-split) must be validated against a REAL sidecar — both paths share the same assumption, so no integrity check fires if it's wrong [BUG?] (`vocabulary.py:140`, `utils/metadata_ingestion.py:23`, `dataset_loader.py`).

---

## Pre-run smoke checks

Run these from the project venv (`L:\Dab\payton_env`) before trusting a training launch. Tick each.

- [ ] Config validates and the resolution sync produces a patch-divisible, internally-consistent config:
  `python Configuration_System.py --config configs/unified_config.yaml --validate-only` (or the project's `--validate` entrypoint).
- [ ] Tag-delimiter / flip-pipeline sanity (the highest-risk silent-data surface):
  `python test_flip_pipeline.py` — and manually inspect a couple of real sidecar JSONs to confirm `tags` delimiter matches `parse_tags_field`.
- [ ] Vocabulary integrity: load `vocabulary.json`, confirm PAD=0/UNK=1, all four rating tags present, contiguous `0..N-1` indices, and that the resume-time `compute_vocab_sha256` does NOT return `'unknown'`.
- [ ] Model + loss + metric import/forward smoke: build the model at the ACTIVE phase image_size, run one dummy `model(images, padding_mask=pmask)` in `model.train()` mode (mask present), and one `AsymmetricFocalLoss`/`MetricComputer` forward to confirm shapes and that dropout/num_layers came from the YAML.
- [ ] Standalone validation against the checkpoint, using the SAME `json_dir`/seed as the training run:
  `python validation_loop.py --checkpoint <path> --config configs/unified_config.yaml --max-samples <N>` — confirm it uses the same split, threshold, image_size, normalize, and pad_color as training.
- [ ] Confirm `logs/STOP_TRAINING` is absent and `experiment_name`/checkpoint_dir resolve to the intended resume directory.
- [ ] (If resuming Phase1→Phase2) Confirm `tools/prepare_phase2.py` rewrote the embedded config AND any embedded `preprocessing_params` to image_size=448, and that the resume actually loaded weights (not silently restarted from scratch).

---

## Sign-off

Reviewer: ______________________   Date: ______________   Commit/branch reviewed: ______________   Triage section cleared: ☐   All high-severity items resolved or accepted: ☐
