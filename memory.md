# OppaiOracle — Verified Training-Pipeline Issues

_Date: 2026-08-21 · Full code review + per-issue deep verification. Duplicate-image handling excluded (handled upstream before training)._
_Second pass, same day: 6-agent fan-out over the full training pipeline (train_direct both halves, training_utils, dataset_loader, loss/metrics/telemetry, stop-conditions/schedulers/config) plus manual review of model_architecture/mask_utils/custom_drop_path/shared_vocabulary. ~~Every item ... REPORTED-UNVALIDATED~~ → **VALIDATED same day (third pass, independent session): all claims re-checked against source; 2 retracted (S8, L20), 3 amended (L1, L14, L15); everything else stands. See "Third pass" at bottom.**_

## Confirmed issues (ranked)

### 1. Scheduler warmup off-by-one — CONFIRMED, FIX APPLIED IN CODE
- `training_utils.py:858` (`step_in_cycle = 0`) + `:974-976` (`step()` increments first).
- `_LRScheduler.__init__` calls `step()` once at construction → fresh runs train update *k* at `get_lr(k)` instead of `get_lr(k-1)`. Warmup one step short; cosine peak one step early.
- Live because `configs/unified_config.yaml` uses `warmup_epochs: 2` → `warmup_steps > 0`.
- Resume is SELF-CONSISTENT (checkpoint stores post-increment value) — no fresh-vs-resume divergence, just a uniform one-step shift vs intent.
- **Fix:** `self.step_in_cycle = -1` at init. Checkpoint-compatible; only fresh-run geometry changes.
- ✅ 2026-08-21 second pass: fix IS APPLIED — `step_in_cycle = -1` now at `training_utils.py:865` (independently re-derived by 2 agents). Remaining TODO: regression test + fresh-run LR-curve sanity check in a validation session.

### 2. Stale vocabulary caches — CONFIRMED
- `logs/vocab_cache/*.frequencies.json` and `.filelist.txt` validate only by file count / existence.
- In-place tag edits never invalidate → stale vocab/min-frequency cut silently reused.
- No `--rebuild-vocab` CLI flag; deleting cache dir is the only escape hatch.
- **Fix:** sampled `(size, mtime_ns)` fingerprint (~512 files, deterministic sample) in both cache keys + expose `use_cache=False` via CLI.

### 3. Swallowed `scheduler.step()` failures — CONFIRMED
- `train_direct.py:2373-2377` and `~2784-2787`: try/except → warning only.
- Transient failures implausible (pure arithmetic); corrupt resume state (`cur_cycle_steps=0`) → raises forever → LR frozen silently. No stall detection anywhere.
- Second pass re-derived the adjacent half: `load_state_dict` accepts `cur_cycle_steps=0` verbatim → per-step restart churn, LR ramps min→peak within ~3 steps (silent wrong schedule, not NaN).
- **Fix:** fail-fast after 5 consecutive failures + webhook escalation + validate restored scheduler state at resume time.

### 4. Clean-margin demotion on stale evidence — CONFIRMED, FIXED (residual hole found)
- Old bug: `clean_margin()` filtered None records before taking tail → demotion could use margin data older than the decline window.
- **Fix applied** (agent): aligned-window version at `stop_conditions.py:591-640`; holes → `pending` → decline stays armed (fail-safe). Regression test added; 39/39 pass. Review with `git diff stop_conditions.py`.
- ⚠️ Second-pass residual (UNVALIDATED): the aligned window anchors to the record **tail**, not to the decline onset. Two traced failure shapes still demote a tripped `peak_decline` to advisory: (a) stale window — corroborating margin fall slides OUT of the window before demotion is evaluated; (b) endpoint fragility — a ±0.01 wobble at window end reverses a net fall. Fix direction: measure margin change from the record at/after `best_epoch` (decline onset) to latest, require margins present across the whole span (else pending), use a two-half slope instead of endpoint delta. See Second-pass #2.

### 5. Config type coercion gap — CONFIRMED, latent
- Plain (non-Union) fields get no cast (`Configuration_System.py:488`); PyYAML parses unquoted `1e-4` as STRING (YAML 1.1 needs a dot).
- Current config safe (uses `1.0e-5` style) but one future edit away from opaque failure; bools like `log_to_file: "false"` silently stay truthy.
- Unknown YAML keys are warn-only at `from_dict` (:393-395).
- **Fix:** `_coerce_scalar` helper for plain fields + `strict_unknown=True` at training entrypoint only.



## Downgraded / defensive-only

### 7. All-unknown-tag images — hole real, unreachable today
- Guards check list non-emptiness, not encoded-vector sum; BUT rating bit force-set (sum ≥ 1 always) and scan of 60k sidecars found 0 images with zero in-vocab content tags at `min_frequency: 500`.
- Tag-level unknown rate: 12.2% (sub-cut rare tags train as negatives under high gamma_neg — inherent to frequency cutoffs, by design).
- **Fix (defense-in-depth):** all-zero check after `encode_tags` in both dataset paths → error-sample.

---

# Second pass — 2026-08-21 (all UNVALIDATED — re-verify in an independent session)

## HIGH

### S1. `eval_steps` unit mismatch guts validation / early-stop / checkpoint density
- `configs/unified_config.yaml:339` sets `eval_steps: 11538` ("~0.92 epoch" — authored in MICROBATCH units). Sole consumer `train_direct.py:2935` compares against OPTIMIZER UPDATES (~1,389/epoch at accum=9) → real cadence ≈ 8.3 epochs.
- On the 15-epoch config: ~2 validated epochs total → early stopping inert (rules stuck `pending`), best-model sampled twice, crash can strand ~7 epochs. Startup guard at `train_direct.py:1888-1896` warns but proceeds.
- **Fix:** set `eval_steps ≈ 1280` (update units), document the unit, make the guard fail-fast when `eval_steps >= updates_per_epoch` without opt-out.
- ✅ Third pass CONFIRMED: config:339 `eval_steps: 11538`; train_direct.py:829 `updates_per_epoch = (steps_per_epoch + accum - 1) // accum`; :2372 `global_step += 1` per optimizer update; :2935 `global_step - last_validation_step >= eval_steps`; guard :1888-1896 warn-only. Cadence ≈ 8.3 epochs as claimed.

## MEDIUM

### S2. selection_metric switch on resume leaves mixed-scale eval_history
- `train_direct.py:1180-1237`: `best_metric` is re-seeded on metric switch, but `_eval_history` keeps old-metric records; `make_record` doesn't store which metric `selection_value` is. f1-scale (~0.27) + mAP-scale (~0.65) records → nonsense plateau/peak_decline math.
- **Fix:** clear eval_history when `_sel_ckpt != _sel_now` (condition already detected there), or tag records with metric name and filter in `phase_history`.
- ✅ Third pass CONFIRMED: re-seed block (train_direct.py:1198-1237) touches only best_metric/patience/selection_metric — never `_eval_history` (only full phase-transition reset does, :1243); `make_record` (stop_conditions.py:322-357) stores no metric tag; `evaluate()` consumes `selection_value` blindly (:753).

### S3. Env overrides accept inf/NaN; validate() passes them
- `Configuration_System.py:2530-2537` → `float('inf'/'nan')` stored; `learning_rate <= 0` check misses it; `FullConfig.validate()` PASSED with LR=inf (verified against real YAML).
- **Fix:** reject non-finite floats in `_parse_env_value` + `isfinite` checks in `TrainingConfig.validate`. Related: env typos produce raw tracebacks instead of `ConfigValidationError` (:2513, :1661, :1723).
- ✅ Third pass CONFIRMED LIVE: probe against real YAML — `ANIME_TAGGER_TRAINING__LEARNING_RATE=inf` → stored as inf, `FullConfig.validate()` PASSED. `float(value)` at Configuration_System.py:2535; only `<= 0` check at :1612.

### S4. Transient __getitem__ failures become PERMANENT dataset exclusions
- `dataset_loader.py:2501-2534` (SidecarJsonDataset; same pattern `:1561-1580`): any exception incl. FileNotFoundError persists a permanent exclusion to `cache_exclusions.txt` after 2 encounters; applied globally by bare stem; fixing the cause never resurrects the images. Shard-folder rename between runs can evict an entire shard.
- **Fix:** persist only decode/corrupt-class errors (UnidentifiedImageError/truncated); NotFound-class requires cross-run confirmation or per-run promotion.
- ✅ Third pass CONFIRMED: catch-all except at dataset_loader.py:2501 → `add_exclusion(..., immediate=True)` at :2526; `max_retries=2` (:1037/:1729); exclusion ids normalized to bare stems (utils/exclusion_manager.py:58-67) and reloaded cross-worker/run (:2285-2289). Same pattern at :1561-1580.

### S5. Worker-side Arrow-cache failure → silent all-error-sample flood
- `dataset_loader.py:875-906` raises in `_ensure_table`, but `__getitem__` swallows at `:2297-2302` → whole dataset becomes error samples → trainer skips all-error batches and "runs" while learning nothing. Verified: pyarrow `ArrowIndexError` (row-count drift) hits the same swallow path.
- **Fix:** re-raise accessor/cache-load exceptions out of ann materialization; keep per-sample handling for image I/O only.
- ✅ Third pass CONFIRMED + WORSE: `items[idx]` failure → ann=None (:2297-2302) → synthetic IndexError (:2332) → catch-all (:2501) → PERMANENT exclusion persisted per sample (:2526). `_ensure_table` raises RuntimeError (:888-893); `ArrowIndexError` from stale row-count hits the same swallow.

### S6. Async saves violate `save_total_limit`
- `training_utils.py:1666/1734/1786`: cleanup counts only on-disk files; in-flight async writes escape the count. Verified: 4 saves with limit=2 left 4 files, persisting across shutdown.
- **Fix:** count queued paths in cleanup, or prune after `wait_pending`/on shutdown.
- ✅ Third pass CONFIRMED: `_refresh_checkpoint_list` (training_utils.py:1776-1792) counts only on-disk `checkpoint_*.pt` + `.exists()`; in-flight async write escapes; no shutdown prune.

### S7. `save_checkpoint` reports success for async saves that later fail
- `training_utils.py:1664-1670, :1079, :1738`: path returned + appended to `checkpoints_saved` BEFORE the write; on async failure the phantom path is embedded in every subsequent checkpoint. Transient disk-full at step N → later resume hits FileNotFoundError on a "saved" checkpoint.
- **Fix:** failure callback set consulted by callers; surface `writer.last_error` after each `wait_pending`.
- ✅ Third pass CONFIRMED: path appended at queue time (:1666/:1675/:1691/:1701/:1738); failure callback only logs (:1650-1651); `last_error` consulted once, in `shutdown()` (:1289-1295) — and see L10: one transient failure makes that alarm fire forever.

### S8. ~~ASL negative-term deviates from Ridnik 2021~~ → RETRACTED (third pass, NOT-A-BUG)
- The OFFICIAL Alibaba-MIIL/ASL `AsymmetricLoss` computes exactly `los_neg = (1-y) * log(clamp(xs_neg + clip, max=1))` with `xs_neg = 1 - sigmoid(x)` — i.e. `−log(1−p+m)` capped at −log m — weighted by `(p−m)^γ⁻` focal factors. That is the SAME form as `loss_functions.py:301-303` (`xs_neg = (1.0 - probs + self.clip).clamp(max=1)` → `bce_neg`) composed with `:327-335` (`neg_weights` from `(probs - clip)+^γ⁻`). The review's baseline `−log(p−m)` matches no ASL variant in the paper or reference code; the measured ratios (1.8×…58×) were computed against a strawman. Repo implementation is faithful ASL.

### S9. Focal weights computed from bf16 sigmoid under autocast — DOWNGRADED to MINOR recommendation (2026-08-21); FURTHER STUDY warranted
- `loss_functions.py:281` via `train_direct.py:2161-2184`: `sigmoid` stays bf16 under autocast (bce_with_logits upcasts, sigmoid doesn't) → claimed ~3.1% mean gradient rel-err, max 100% near the clip margin (bf16 rounding flips `(p−clip)` sign). asl_telemetry already upcasts; the loss doesn't.
- **Fix (if/when adopted):** `probs = torch.sigmoid(logits.float())`. Estimated cost: +25–40 MB peak VRAM (<0.5% of job), <1% step time.
- ✅ Mechanism CONFIRMED third pass — bare sigmoid under `amp_autocast()` (train_direct.py:2161), bf16 config, `bce_with_logits` upcasts while sigmoid doesn't, telemetry upcasts at asl_telemetry.py:448. Magnitudes NOT re-measured (no torch in validation env).
- ⬇️ Why downgraded: unbiased rounding noise, NOT a correctness bug — no train/inference mismatch, no metric distortion, gradients internally consistent; expected final-model effect statistically indistinguishable (below seed variance; an A/B can't isolate it). Value is confound removal — chiefly worth doing BEFORE the planned clip 0.2→0.05 move (gate-flip exposure ∝ γ⁻/(p−m) grows sharply at m=0.05).
- 🔎 FURTHER STUDY: quantify empirically before acting — run a torch A/B in payton_env measuring peak VRAM and gradient rel-err (bf16 vs fp32-sigmoid path) to replace the unverified 3.1%/100% figures with measured ones; bundle with L12 (val-side fp16 double-quantization) as a single V2 "numerics pass" at a phase boundary so there is one attribution break, not two.

## LOW (condensed — full traces in session transcript)

- **L1** `train_direct.py:263-283` + gate `:1079-1082` — `_normalize_experiment_name` rewrites any name containing "vit" anywhere but last (`vit-l16`→`l16-vit`) but sets `legacy_experiment` only when NO vit token existed → pre-existing run dirs orphaned, `resume_from='latest'` silently starts FROM SCRATCH in a new dir. Fix: drop the `had_arch_token` gate (legacy lookup no-ops safely). ⚠ Third pass PARTIAL: effect overstated — resume logs a WARNING ('no checkpoint found … Starting fresh'), not fully silent; only dirs created under an older spelling/manual setup are orphaned.
- **L2** `train_direct.py:1927-3644` — no try/finally around epoch loop: any exception (OOM, assert_finite under debug, second-Ctrl+C escalation) skips checkpoint-writer drain + monitor close; queued saves lost. The :309-313 handler comment claiming "finally blocks still run" is false.
- **L3** `train_direct.py:3000-3101` — all-batches-failed validation fabricates val_loss=0.0 + zero metrics into TB/eval_history as real measurements (train side has the guard; val doesn't). Mirror the train-side carry-forward.
- **L4** `train_direct.py` — no terminal checkpoint after natural completion (5 save sites, none post-loop); `last.pt` stale by up to save_steps. Add unconditional boundary save.
- **L5** `train_direct.py:1947-1960, :2751` — carried soft-stop accumulation window blends two epochs into `avg_train_loss` / stop-record train_loss (one record per stop request misrepresents its epoch).
- **L6** `train_direct.py:1640` — recovery message names nonexistent key `training.compile` (actual: `training.use_compile`).
- **L7** `train_direct.py:384-396` — Unix keyboard listener leaks cbreak termios state on normal exit (daemon thread frozen mid-select; Windows host unaffected).
- **L8** `train_direct.py:296-338` — SIGTERM soft-stop inert on Windows (TerminateProcess; no SIGBREAK handler). Sentinel file + Ctrl+C are the only working mechanisms; document or add SIGBREAK handler.
- **L9** `training_utils.py:88` — `setup_seed` returns PRE-override determinism flag; with runtime.deterministic override, `use_deterministic_algorithms` fights cudnn flags. Fix: `return user_seed, det`. Latent (no runtime.yaml today).
- **L10** `training_utils.py:1037/1079/1186` — `_last_error` never cleared on success → one transient failure poisons shutdown alarm for process lifetime.
- **L11** `evaluation_metrics.py:426-430` — rating tags (no frequency entry) land in the rarest bucket → LVIS rare-bucket diagnostic inflated. Exclude or own-bucket them.
- **L12** `train_direct.py:3039, :3084` + `evaluation_metrics.py:310` — val probabilities double-quantized (bf16 sigmoid → fp16 CPU accumulation) before bucketed metrics/calibration; flips per-tag F1. Fix: sigmoid on `.float()`.
- **L13** `dataset_loader.py:734-738` — split cache writes two files non-atomically; crash between them = new-train + old-val (leakage until next successful rewrite). Single file or SPLIT_ID cross-check.
- **L14** `dataset_loader.py:2024-2028` + `metadata_cache.py:233/465/586-588` — int ratings break Arrow build → sequential fallback EVERY run (✅ third pass REAL; nuance: not literally silent — ERROR+warning logged once per run; ints are a documented input shape per `_map_rating_to_tag`). `str()`-coerce in `_parse_json_batch`.
- **L15** `dataset_loader.py:2031` — zero-row Arrow filter enters fallback with `.append` on accessor. ⚠ Third pass PARTIAL: the AttributeError is swallowed by per-file except (:2068) → parse_failures; actual outcomes are ArrowCacheTruncatedError (>~100k files) or silently empty corpus. Fix unchanged: add `not self._using_arrow` to the condition.
- **L16** `dataset_loader.py:610` + `metadata_cache.py:631` — FILE_COUNT tolerance floor `max(100, 0.1%)` masks small corruption (verified: 4/210 deleted lines pass). Scale floor with size.
- **L17** `Configuration_System.py:517-519` — `update()` with partial nested dict resets siblings to class defaults (verified: alpha-only update reset gamma_neg 6.0→3.0). Latent.
- **L18** `Configuration_System.py` — no `warmup_epochs < num_epochs` check anywhere live (only guard is in dead schedulers.py); `warmup_epochs: 50` typo = all-warmup run, no error.
- **L19** `asl_telemetry.py:201-244` — with `asl_schedule.enabled: false`, gamma override applies with NO dwell/clamp guards (all inside the enabled gate). Latent.
- **L20 → RETRACTED (third pass)** `asl_telemetry.py:115-125` — mechanism inverted: absent `phase` key makes the reset branch False, so bookkeeping is RETAINED, not dropped; after a real flip+reset that yields negative elapsed dwell which REFUSES steps (over-conservative), never lands early. train_direct.py:1129 documents legacy checkpoints resume normally.
- **L21** `evaluation_metrics.py:277-287` — `find_optimal_threshold` all-zero fallback returns grid edge 0.1 instead of init 0.5 (benign).
- **L22** `stop_conditions.py` minor: demoted decline prints as `TRIP` with no advisory marker (:300-310); bare `NaN` serialized into stop_decisions.jsonl (invalid strict JSON, :436); fingerprint_advisories treats gap-separated epochs as consecutive (:679-692); budget reports negative "remaining" on overrun.
- **L23** Config observation (not a bug): `training.phase: 2` window caps γ at 6.0 while YAML `gamma_neg: 7.0` → permanent OUTSIDE-WINDOW warning each start. Behavior correct and logged.

## Dead code (maintenance traps — delete or wire deliberately)

- `shared_vocabulary.py` — ENTIRE FILE dead (nothing imports it; shared-vocab path removed; WorkerInitializer docstring confirms).
- `training_utils.py:993-1004` — explicit-epoch `step(epoch)` branch: broken math (stale cycle/max_lr; log() raises for cycle_mult<1).
- `training_utils.py:2913` — worker_init_fn seed can reach ≥2³²−1 → np.random.seed ValueError (currently unwired).
- `training_utils.py:91-129` — `log_sample_order_hash` consumes live loader RNG if ever wired (hash describes an order never trained on; shifts real order). `log_index_order_hash` does it correctly.
- `LearningRateSchedulerFactory` + `schedulers.py` — live path constructs `CosineAnnealingWarmupRestarts` directly (train_direct.py:847); factory's 'cosine' branch ignores warmup_steps; schedulers.py has latent math issues (epoch 0 at warmup_start_lr≈0, ramp never reaches base_lr, wrong warning text, float warmup truncated).
- `stats_queue` per-epoch drain `train_direct.py:2700` — producer died with the orientation handler (dataset_loader.py:1826-1830).
- fp16/GradScaler block `train_direct.py:897-929` — unreachable (bf16-only validation), benign future-proofing.
- `training_utils.py:1763` — `oldest == self.best_checkpoint` guard dead (best_model.pt never matches the checkpoint_* glob).

## Retracted during verification (NOT bugs)

- Clip-path BCE gradient spike — focal weight `(p−clip)₊^γ` zeroes confident negatives; math sound.
- Micro-F1 includes PAD/UNK — keep-mask applied before micro computation.
- mp.Queue breaks checkpoints — `to_dict()` serializes declared dataclass fields only.
- Fresh-vs-resume LR divergence — both share the same one-step shift.
- (2nd pass) Burn-in duplicate-append into eval_history — not reachable through any save/resume ordering.
- (2nd pass) Rotation-mask correctness — full suite passes on current code (independent ground truth, sign-discriminating, RNG pinned to 2 draws).

## Verified clean

ASL focal weights/clip/margin semantics (log-term form is the open S8); GradScaler/bf16 handling; accumulation semantics + epoch-boundary flush rescale (mean over actual microbatches, magnitude preserved); step ordering (unscale→clip→step→zero→sched); resume math (epoch conversions, sampler offsets, bit-exact RNG save/restore round-trip); checkpoint save↔restore field symmetry (all TrainingState in-loop fields are declared dataclass fields); weight-decay exclusion groups; Adan matches arXiv:2208.06677; EXIF/RGBA/augmentation alignment; split determinism + seed-mismatch/count-drift/version cache gates; flip determinism (per (id,epoch), val pinned p=0); multi-label head (logits+sigmoid); no in-loop threshold tuning; soft-stop replay dedup; ThresholdCalibrator grid (no off-by-one, tie-break benign); keep-mask before micro-F1; model_architecture mask semantics (Flex True=IGNORE inverted internally / SDPA True=ATTEND — consistent both paths); eval-only logit clamp (zero-grad outside bounds correctly skipped in training); pos-embed bicubic interpolation (runtime + checkpoint-resume paths agree); mask_utils pooling; SafeDropPath float32 rand draw; vocabulary.json load converts index keys back to int (bias-init/class-weight lookups hit); Monitor_log best-model tracking follows selection_metric (was hardcoded f1_macro, fixed); stop_conditions plateau geometry ≡ first_confirmable_epoch; overfit_loss straddle guard; halt/advise/off wiring (halt really breaks).

## Open investigation — RESOLVED (2026-08-21 warmup fan-out, 3 agents)

Epoch↔step scheduling math audited end-to-end. Results:
- steps_per_epoch/warmup_steps units CORRECT (accumulation-divided, val split pre-removed)
- No double-stepping between in-loop (:2373) and epoch-boundary flush (:2784) — mutually exclusive via accum_count gate
- Mid-epoch soft-stop resume bit-identical (saves always post-scheduler.step)
- Retarget non-accumulating, gamma decay clean across cycle-boundary resumes
- max_lr/min_lr wiring correct (peak 1.299e-5 real sqrt-scaling; TB logging truthful, pre-step capture)

Off-by-one severity ESCALATED: cycle restart fires after N−1 updates → the FINAL
update of a fresh run jumps from ≈min_lr onto the fresh warmup ramp (~0.9×peak).
Fix `step_in_cycle = -1` — NOW APPLIED at training_utils.py:865 (see issue #1).

Advisory (no action yet):
- num_cycles>1 floor-division undershoot (config uses 1 — unaffected)
- Legacy ckpt w/o scheduler state silently restarts LR schedule (guard disarmed, train_direct.py:1341)
- Changing num_epochs alone hard-errors on resume (deliberate default policy)
- Latent: _get_layer_wise_params emits lr_scale keys nothing reads (layer_decay would no-op)

## Validation-session checklist (for the follow-up pass)

1. Re-verify S1 first (highest impact, one config number): count actual validations in a short dry run or trace `last_validation_step` updates.
2. S2/S3: reproduce via resume with switched selection_metric; env `LEARNING_RATE=inf` against `--validate-only`.
3. S4/S5: delete/rename a shard folder + corrupt the Arrow cache; confirm exclusion persistence / error-sample flood.
4. S6/S7: 3 rapid soft-stop saves with save_total_limit=2; kill the writer thread mid-save and inspect `checkpoints_saved` for phantom paths.
5. S8/S9: numeric A-B of loss forms + gradient rel-err under autocast (scripts existed in the review session; recreate).
6. L1: create a run dir named e.g. `vit-l16`, resume with `resume_from=latest`, confirm from-scratch start.
7. After validation, promote items to the "Confirmed issues" section above with CONFIRMED/line-numbers re-checked against HEAD.

---

# Third pass — 2026-08-21 VALIDATION RESULTS (independent session: 3 verification subagents + primary-source re-checks + live probes)

Method: every claim re-opened at its cited location and judged against quoted code; S3 verified by live probe against the real YAML; C2's test suite executed for real. Line refs below are current HEAD.

## Verdict summary
- **S1–S7, S9: CONFIRMED REAL** (annotations inline above). S9 subsequently DOWNGRADED to minor recommendation — see its section.
- **S8: RETRACTED — NOT-A-BUG.** Official Alibaba-MIIL/ASL uses the identical `−log(1−p+m)` negative term capped at −log m, weighted by `(p−m)^γ⁻`. The review compared against a nonexistent baseline.
- **L20: RETRACTED — NOT-A-BUG** (mechanism inverted; code retains bookkeeping and refuses steps — over-conservative, never early).
- **L1 PARTIAL** (orphaning real but not silent — warning is logged), **L14 REAL w/ nuance** (fallback not literally silent), **L15 PARTIAL** (AttributeError swallowed; real failure = truncation error or silent empty corpus).
- **All other L-items (L2-L13, L16-L19, L21-L23): REAL**, line numbers re-pinned (e.g. L3 zeros persist via train_direct.py:3001-3101/3186/3432 vs guarded train side :2887-2890; L4 five save sites :2419/2533/2601/2852/3516, none post-loop :3564+; L11 confirmed in the real vocabulary.json — rating tags indices 19290-19293 absent from tag_frequencies; L12 double quantization chain :3039→:3084→bucketed metrics/calibrator).
- **D1-D8 dead code: ALL CONFIRMED** (D2's branch additionally unreachable — live call sites pass no epoch arg; D3 boundary off-by-one: ValueError needs ≥2³², odds ~2⁻³² per spawn; D6 corroborated by Monitor_log.py:1468; D7 unreachable via unconditional bf16 validation at train_direct.py:877).
- **Fix statuses:** C1 ✅ applied (`training_utils.py:865` `step_in_cycle = -1`); C2 ✅ applied + **test suite green: 39 passed in 9.88s** (pytest 9.0.2; note: default Hermes venv python lacks pytest — use system Python); C3 ✅ still disabled (`unified_config.yaml:585 enabled: false`).

## Remaining actions (unchanged priorities)
1. S1 config fix (`eval_steps ≈ 1280`) — highest impact, one number.
2. S2/S3/S4/S5/S6/S7 fixes as described inline.
3. Delete dead code D1-D8 or wire deliberately.
4. Update issue #4 residual: endpoint-delta fragility confirmed mechanically (stop_conditions.py:605/:623) — two-half slope fix direction stands.
5. S9 (minor): leave as-is for now; before any clip 0.2→0.05 experiment, run the torch A/B measurement probe (payton_env) and consider bundling S9+L12 as one V2 numerics pass.
