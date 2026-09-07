# V2 Training Monitoring — canaries, decision rules, scalar reduction

Active V2 Phase 1 uses WSD with a 60-epoch ceiling including 15% elapsed-update
cooldown. Plateau dispatch starts disabled until stable-phase jitter is reviewed.
The fresh 30K holdout and final vocabulary await the second dataset subset; V1
bands are historical. See [Phase 1 setup](v2-phase1-setup.md). Two-image sampled
attention maxima and binned oracle/decile diagnostics are implemented; their scope
is diagnostic, and oracle F1 does not select checkpoints.


[Roadmap](README.md) · [Production plan](v2-plan.md) · [Augmentation decisions](v2-augmentation.md)

> **Provenance.** This is the monitoring methodology developed during the **V1 Phase-2** (448×448
> fine-tune) ViT run, relocated out of `TRAINING_HEALTH_TRACKER.md` on **2026-07-29** when that file
> was reduced to a pure V1-P2 run archive. The archive keeps the measurement; this file keeps the
> method.
>
> **⚠️ Every band below is V1-calibrated.** The mAP regime thresholds, the rare-bucket floor, the
> head/mid ratio, the `mean_active` and logit-center numbers, and the LR-context values all derive
> from V1's val set, V1's ~19K vocabulary, V1's peak LR of 1.4e-5, and a frozen F1 threshold of
> **0.2653**. V2 changes the vocabulary, the val set, and the operating threshold (V1 reference **0.7927**, to be refitted for V2). **Re-derive
> every numeric band against V2's own baseline epoch before flagging anything.** The *shapes* of the
> checks (direction, ratio, consecutive-pair escalation) are what transfers; the numbers are not.
>
> Canary numbering is preserved from the V1 tracker so the archive's `🟡 canary #N` citations still
> resolve. **Numbers 4 and 5 are intentionally vacant** — they were F1-ratio canaries, self-demoted
> during V1-P2 as calibration-floored, never cited by a data row, and deleted on relocation.

## Startup-window suppression

A phase transition (optimizer reset, resolution jump, changed loss shape, changed regularization)
moves the loss landscape on several axes at once, so the first epochs are not steady state.

- **Suppress all canary flags during the warmup window** (V1-P2: E0-E1, matching
  `warmup_epochs=2`). Under a linear LR ramp, growth shape is dominated by LR, not learning — do not
  anchor canary #1's bands to warmup epochs. Record observations in Notes for later context instead.
- **Apply canaries normally from the first post-warmup epoch onward.**
- **What is a real problem regardless:** `val/mAP` still below the carry-in baseline several epochs
  past warmup, `val/loss` still rising past warmup, gradient blowup, NaN, or a step-function plateau.
- **A prediction worth not repeating:** the V1-P2 expectation was 1-2 epochs of instability with
  `val/mAP` *dipping below* the Phase 1 baseline before recovering. It did not happen — P2 E0 came in
  **above** baseline immediately (0.659055 vs 0.651841). Treat the suppression window as a
  no-flag policy, not as a prediction that the metric will get worse.

## Procedure (run after every validation event)

1. **Pull the epoch-level scalars** for the run directory. The validation step `S` is whatever the
   actual logged step is — recompute steps-per-epoch from the first validation event of the phase
   rather than carrying a previous phase's value (batch size and image size both change it).

   ```bash
   l:/Dab/payton_env/Scripts/python.exe -c "
   from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
   ea = EventAccumulator('tensorboard/<RUN-DIR>', size_guidance={'scalars': 0})
   ea.Reload()
   # Epoch-level scalars (one value per validation event)
   want = ('val/', 'val_bucketed/', 'train/loss_epoch', 'train/learning_rate', 'train/skipped_batches')
   for t in sorted(ea.Tags().get('scalars', [])):
       if t.startswith(want):
           for e in ea.Scalars(t):
               print(f'{t}: step={e.step} value={e.value:.6f}')
   "
   ```

   Then the **stability scalars** (per-step series — reduce to per-epoch summaries):

   ```bash
   l:/Dab/payton_env/Scripts/python.exe -c "
   from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
   ea = EventAccumulator('tensorboard/<RUN-DIR>', size_guidance={'scalars': 0})
   ea.Reload()
   stability = (
       'train/grad_norm',                # per-step; report epoch max + mean
       'train/tag_logits_min',           # per-step; report epoch min
       'train/tag_logits_max',           # per-step; report epoch max
       'train/tag_logits_mean',          # per-step; report epoch mean
       'train/nan_inf_loss_detected',    # only logged when triggered (presence = event)
       'train/nan_grad_skipped',         # only logged when triggered (presence = event)
   )
   for t in stability:
       events = ea.Scalars(t) if t in ea.Tags().get('scalars', []) else []
       if not events:
           print(f'{t}: ABSENT (= 0 events for nan flags, or scalar never written)')
           continue
       vals = [e.value for e in events]
       print(f'{t}: count={len(vals)} min={min(vals):.4g} max={max(vals):.4g} mean={sum(vals)/len(vals):.4g}')
       # For per-epoch breakdown, filter events by step range — recompute steps_per_epoch from val events
   "
   ```

   **Gating (corrected 2026-07-29).** These were previously documented as gated by
   `debug.log_gradient_norm` / `debug.log_activation_stats` alone. They were not: the real condition
   was `debug.enabled AND <sub-flag>`, and `debug.enabled` has been **`false` in every committed
   config since 2025-08-30**. So the documented pre-flight check passed while the scalars were dead —
   the V1-P2 run that did produce them was on an uncommitted local edit. As of the fix:

   - `train/grad_norm` now rides the norm `clip_grad_norm_` already returns on **every** optimizer
     update, so it is free, is no longer behind `debug.enabled`, and is written once per
     `logging_steps` updates. It is on unless `debug.log_gradient_norm` is explicitly `false` —
     `DebugConfig.log_gradient_norm` now defaults to `True` and `validate()` warns when it is off.
     (Until 2026-07-29 the dataclass default was `False`, so a config omitting the key lost the
     canary silently; the trainer's `getattr(..., True)` fallback was dead code, because a dataclass
     field always exists.) Only **finite** norms are written: a non-finite one is reported through
     `train/nan_grad_skipped` instead of putting `inf` into a series whose documented reduction is
     the epoch max.
   - `train/tag_logits_*` and `train/image_*` are gated on `debug.log_activation_stats` /
     `debug.log_input_stats` **alone**. Confirm the one you want is `true`; `debug.enabled` is
     irrelevant to them now (it still controls the per-microbatch `assert_finite` calls, which cost
     `2 × accum` host syncs per update — that coupling is why the master switch is off).
   - **One event per logged step.** These scalars used to be written once per *microbatch*, because
     `global_step` is constant across an accumulation window and the periodic test keyed on it alone.
     At `accum=9` that meant 9 events all stamped with the same step, and V1-P2's "n=9 samples per
     epoch" was really one accumulation window sampled 9 times at a single instant — not epoch-wide
     drift. They are now gated on the update boundary and stamped with the same step as `train/loss`.

2. **Fill the per-epoch row.** Column → scalar mapping. Write `—` for absent scalars (this is the
   convention `tools/run_validation_for_epoch.py` emits and the archive uses).

   | Column | Source | Notes |
   |---|---|---|
   | Step | derived | actual logged validation step |
   | train_loss | `train/loss_epoch` | per-epoch mean train loss; co-logged with `val/loss` |
   | val/loss | `val/loss` | |
   | val/mAP | `val/mAP` | **primary progress signal** — the frozen-list macro-mAP (`validation.freeze_macro_tag_list`); this is the series the plateau rule, `min_delta` 3e-3, and the fingerprint gate all read. **Single series — weight EMA was cut 2026-08-01 (v2-plan §6), so there is no second curve to log and the gate reads these raw values.** |
   | growth | derived | `val/mAP[E] / val/mAP[E-1]`; `—` for the first epoch of the phase |
   | 500-999 | `val_bucketed/500-999/mAP` | rare-tag canary (#2) |
   | 1K-5K | `val_bucketed/1000-4999/mAP` | |
   | 5K-10K | `val_bucketed/5000-9999/mAP` | |
   | 10K+ | `val_bucketed/10000+/mAP` | head bucket; canary #3 uses 5K-10K / 10K+ ratio |
   | f1_micro | `val/f1_micro` | **DRILL-DOWN ONLY** — calibration-floored, do not flag |
   | f1_macro | `val/f1_macro` | **DRILL-DOWN ONLY** — calibration-floored |
   | mean_active | `val/mean_active` | mean tags above the frozen validation threshold (`threshold_calibration.default_threshold`) per val image; record which threshold value was in force |
   | lr_end | `train/learning_rate` | value at the largest step ≤ `S`. Stale by construction when the LR log cadence is coarser than an epoch — note the interpolated actual |
   | skips | `train/skipped_batches` | scalar absent ⇒ 0 (logged only when >0) |
   | discarded | `train/discarded_batches` | measured-but-gradient-discarded microbatches; absent ⇒ 0. Companion rate: `train/discard_rate`. See canary #9 |
   | Notes | derived | flag tags from canary checks + interpretation |

   Neither `skips` nor `discarded` is written at all on an epoch that ends in a soft stop (the run
   breaks out before the epoch-end scalar writes) — read `—`, not `0`, for such an epoch.

3. **Fill the stability sub-log row.** Each cell summarizes per-step scalars within the current
   epoch's step range.

   | Column | Source | Reduction | Notes |
   |---|---|---|---|
   | grad_norm_max | `train/grad_norm` | max over epoch | Canary #11. Compare to previous-epoch max |
   | grad_norm_mean | `train/grad_norm` | mean over epoch | Context for max — distinguishes spike from drift |
   | logits_min | `train/tag_logits_min` | min over epoch | |
   | logits_max | `train/tag_logits_max` | max over epoch | |
   | logits_mean | `train/tag_logits_mean` | mean over epoch | Canary #13 — sustained drift = calibration slipping |
   | nan_inf_loss_count | `train/nan_inf_loss_detected` | count of events in epoch step range | Canary #12 red on any value > 0 |
   | nan_grad_skip_count | `train/nan_grad_skipped` | count of events in epoch step range | Canary #12 red on any value > 0 |
   | Stability notes | derived | — | Anything anomalous; cite logit drift here when justifying mAP-vs-F1 divergence |

   **Record the sample count per cell.** V1-P2's `train/grad_norm` logged on a ~10K-step cadence
   against ~13.6K-step epochs, so most epochs had n=1 and `max == mean` — canary #11's "epoch-max
   grows >2×" band is unreliable at that sampling. `train/grad_norm` is now free to compute (it reuses
   `clip_grad_norm_`'s return value), so **lower `training.logging_steps` for V2** rather than
   accepting n=1; state n in the notes either way. The same cadence governs `train/tag_logits_*`,
   which now writes one event per logged step instead of `accum` events sharing a step.

4. **Run the canary checks.** Append each triggered flag to Notes as `🟡 <name>` or `🔴 <name>`.
   Canaries #1-3 and #6-9 read from the per-epoch row; #11-13 read from the stability sub-log row;
   #10 is a TensorBoard drill-down.

5. **Apply the decision rules.** Stop or continue accordingly.

## Canary checks

Bands assume post-warmup steady state. Suppress firing during the startup window (above).
**All numeric bands are V1-calibrated — re-derive for V2.**

### 1. Overall mAP growth (`val/mAP` ratio vs. previous epoch)

mAP-regime indexed. For 2-epoch validation spans, apply the band to the per-epoch geometric mean.

- **Healthy:**
  - mAP < 0.50 → ≥1.05× per epoch (re-acquiring/surpassing the carry-in baseline).
  - mAP 0.50-0.65 → ≥1.02× per epoch.
  - mAP > 0.65 → ≥1.01× per epoch **and** smooth deceleration shape.
- **Yellow:** per-epoch growth below the regime band for two consecutive validation pairs.
- **Red:** mAP *decreases* at any point after the startup window, **OR** per-epoch growth <1.005×
  for two consecutive pairs in late training, **OR** step-function plateau (Δ jumps rather than
  smoothly approaching zero).
- **⛔ RETIRED AS A STOP RULE (2026-07-29). Keep it as a reading aid only.** This canary is what
  actually ended both V1 phases, and it was wrong both times. V1-P2 posted growth 1.0062, 1.0068,
  1.0022, 1.0046, 1.0032 — *every* steady-state epoch below the >0.65 band — while `val/mAP`,
  `val/loss` and the rare-tag bucket all moved monotonically in the healthy direction. A rule that
  flags a converging model from its second steady-state epoch onward guarantees a premature stop, and
  1.01×/epoch compounded from 0.659 over 15 epochs implies mAP 0.765, which was never plausible.
  The stop decision now belongs to [stop_conditions.py](../stop_conditions.py)'s `plateau` rule:
  **best of the last k validated epochs vs best of the previous k, against a `min_delta` of 3e-3
  (the measured instrument floor)**, confirmed over consecutive evaluations. Per-epoch growth ratios
  are a difference of two noisy quantities and are not usable at mAP > 0.65.
- **Known failure mode, for the record.** In V1-P2 this canary fired a mechanical red at E4 that the
  operator correctly overrode: two consecutive pairs sat below the band while growth was
  *accelerating* (1.0022 → 1.0046). A run just past warmup with cosine decay engaging naturally posts
  sub-band growth that nonetheless trends up (V1 ran cosine; under WSD's constant stable LR this
  confound is absent — see canary #8). If you keep reading the ratio at all, require the
  sequence to be non-increasing before treating two yellows as meaningful.

### 2. Rarest populated bucket mAP — the rare-tag canary
- Movement is the signal; absolute value is small.
- **Healthy:** moving each pair, even by tiny amounts.
- **Yellow:** stuck at the same value for 2+ consecutive pairs after the startup window.
- **Red:** falls below the carry-in baseline late in the phase — the phase is destroying rare-tag
  learning rather than improving it. (V1's `500-999` baseline was **0.624979**, from the Phase 1 E32
  row in the archive. An earlier version of this band said "~0.59", which was wrong.)
- Check `num_tags` per bucket first: V1's `300-499` bucket was empty, so `val_bucketed/300-499/mAP`
  read 0.000 forever and was not a signal.

### 3. Head/mid bucket ratio: `5K-10K mAP / 10K+ mAP`
- **Healthy:** ≥0.90, stable or rising.
- **Yellow:** <0.85, **OR** drops by >0.05 across 2 consecutive pairs.
- **Red:** <0.75, **OR** monotonically falling across 3+ pairs (head-collapse signature).

### 4-5. *(vacant — F1-ratio canaries, demoted and deleted)*

`f1_micro/mAP` and `f1_macro/f1_micro` were tracked in V1-P2 and then demoted to drill-down-only:
the validation F1 threshold is frozen at metric-construction time and never re-derived, so a
loss-shape change (e.g. `gamma_neg` / `clip`) shifts the logit distribution and makes both absolute
level *and* direction uninterpretable — LR decay tightens calibration against a fixed cutoff
regardless of model state. Use canaries #2 and #3 for the rare-vs-common health signal; both are
threshold-independent. Numbering is left vacant so archive citations stay stable.

### 6. Train vs val loss
- Trust mAP for *progress*; loss for *instability*.
- **Yellow:** `train_loss / val_loss` ratio drops >15% across 3 consecutive epochs in late training —
  early overfitting signal.
- **Red:** `val/loss` *increases* for 2 consecutive epochs while `train/loss_epoch` decreases (after
  the startup window).
- **Do not** read `val/loss < train/loss_epoch` as anti-overfitting: drop_path and dropout are active
  in `model.train()` and disabled in `model.eval()`, so train_loss > val_loss is expected.

### 7. Mean activations per image (calibration-floored — direction only)
- **Healthy:** monotonic decline as logits tighten with LR decay (or recovery from a startup spike).
  **WSD carve-out (v2-plan §6):** the decay-driven reading applies only inside a cooldown. During
  the constant-LR stable phase there is no decay — a sustained decline there is unexplained by
  schedule and must be investigated (conversely, the fingerprints below are *more* informative
  during the stable phase).
- **Yellow:** rises by >2× without a corresponding mAP jump (calibration regression).
- **Red:** rises monotonically across 3+ pairs while mAP stalls or falls. Or: pinned at a small
  constant <10 with bucket spread widening (head collapse).
- **Do not rationalize a level jump at a phase/config boundary as "expected calibration shift"** —
  V1's 3.4× jump (1322 → 4493) at the P2 loss change *was* the shipped defect, and it slipped
  through inside the startup suppression window. Cross-version firing is judged by Gate 2 at
  re-derived thresholds (v2-plan §9.3), not by this direction-only canary.

### 8. End-of-epoch LR
- Not a flag on its own — context for plateaus. If mAP plateaus **and** `lr_end` is <5% of peak LR,
  the plateau is schedule-induced. If mAP plateaus **and** `lr_end` is still >50% of peak, the
  plateau is real.
- **Under WSD (v2-plan §6, decided 2026-07-30) this canary simplifies:** during the stable phase
  `lr_end` sits at the constant stable LR, so a plateau there is real by construction; the
  schedule-induced reading applies only inside a cooldown. This de-confounding is one of WSD's
  operational benefits.

### 9. Skipped batches (`train/skipped_batches`, `train/discarded_batches`)
- **Healthy:** both scalars absent (= 0) most epochs.
- **Yellow:** >50 skips in an epoch.
- **Red:** >10× increase vs. previous epoch — data-loader regression or numerical instability.
- **Read both.** `train/skipped_batches` counts microbatches that never produced a loss measurement
  (failed load, NaN loss). `train/discarded_batches` counts microbatches that *were* measured — so
  they are in `train/loss_epoch` — but whose gradients died with their accumulation window. A single
  mid-window NaN destroys up to `accum` microbatches of work while incrementing `skipped_batches` by
  1, so `skip_rate` alone understates the damage by up to `accum`-fold. `discarded_batches ≫
  skipped_batches` means the loss curve is being computed over work that never reached the weights.

### 10. Train loss curve (per-step `train/loss`, drill-down only)
- Use TensorBoard directly when canary #6 fires. Healthy: smooth downward trend with small spikes
  recovering within ~50 steps.
- **Red:** spikes that don't recover, or step-function jumps.

### 11. Gradient norm (`train/grad_norm`, per-step)
- Read the per-epoch max via the TensorBoard "max" reducer. On unless
  `debug.log_gradient_norm` is explicitly `false` (dataclass default is now `True`, and `validate()`
  warns at startup when it is off) — **not** gated by `debug.enabled` any more, and it is the pre-clip
  norm `clip_grad_norm_` computes on every update, so the value is exact rather than a second
  approximation. Non-finite norms are deliberately **excluded** from this series; they surface as
  canary #12 events, so an `inf` can never contaminate the epoch max.
- **Healthy:** stable across epochs, occasional spikes that recover within ~10 steps.
- **Yellow:** epoch-max grows >2× vs. the previous epoch's max.
- **Red:** epoch-max grows >5× vs. previous, or sustained drift upward across 3+ epochs (precedes
  loss blowup).
- **Sampling caveat:** unreliable at n=1 sample/epoch (see step 3 above). Lower `logging_steps` —
  the norm is no longer expensive to log.
- **Post-detach note (v2-plan §4):** V1-P2's logged norm *levels* are not comparable context once
  the focal weight is detached — derive V2's reference epoch-max/mean from its own first
  post-warmup epochs. The ratio bands above (>2×/>5× within-run) are scale-invariant and transfer.

### 12. NaN / Inf flags (`train/nan_inf_loss_detected`, `train/nan_grad_skipped`)
- Binary flags, written only when triggered — absence of the scalar means zero events. That now holds
  for the non-finite-norm cause on both paths: the epoch-boundary accumulation flush used to discard a
  window on a non-finite norm without writing `nan_grad_skipped`, so the canary was blind to every
  epoch boundary. **One gap remains by design:** if the flush itself raises, the partial window is
  discarded with no scalar (it is not a NaN-gradient event). That path logs
  `"Epoch-boundary accumulation flush failed"` and increments `train/discarded_batches`, so check the
  log and canary #9 before reading an absent flag as "nothing happened".
- `nan_grad_skipped` from the in-loop path is stamped with the *anticipated* step of the failed
  update, so it falls inside the epoch whose update failed. It previously carried the pre-increment
  `global_step`, which put a first-update-of-epoch failure in the previous epoch's step range.
- **Red:** any non-zero value at any step. Stop training and investigate (most likely AMP overflow or
  data-loader corruption).

### 13. Logit-distribution drift (`train/tag_logits_min/max/mean`)
- Direction-only diagnostic. Requires `debug.log_activation_stats: true` (that flag alone — see the
  gating note in step 1).
- **Use:** quantifies the phase's logit-distribution shift. After the startup window, expect
  `tag_logits_mean` to settle into a new band. If it keeps drifting late in the phase, calibration
  will keep slipping — flag for a mid-run threshold recompute.
- Not a yellow/red signal on its own; cite it when justifying mAP-vs-F1 divergence in Notes.
- Watch for pinned values: V1-P2's `tag_logits_min` sat at exactly −11.50 across all 45 logged
  events, an output-side clamp rather than a learned floor.

## Decision rules

- **Startup window:** suppress all yellow/red flags. Record observations in Notes for later context.
- **Two consecutive yellow flags on the same metric (post-warmup):** treat as red — *but* see canary
  #1's known failure mode; require the underlying sequence to be non-improving before promoting.
- **Any red flag early in the phase:** stop training, diagnose. Most likely cause is config wiring
  (augmentation missing, pos_embed not interpolated, optimizer not reset) — verify before assuming
  model failure.
- **Any red flag late in the phase:** finish the current epoch, then decide.
- **A stop trigger fires while `val/mAP` is still rising:** read
  `logs/stop_decisions.jsonl` — the record states which rule tripped and on what numbers. With the
  default `early_stopping_mode: advise` nothing has ended; the run continues to the epoch budget and
  the call is yours. If the rule was wrong, that is a band to re-derive, not a run to salvage.
- **Under WSD (v2-plan §6, decided 2026-07-30), the plateau gate's consequence is not "stop":**
  plateau confirmed on the `val/mAP` curve → launch the **1-sqrt cooldown**
  (~10–20% of the phase's elapsed steps); the cooldown, not the gate, ends the phase. The epoch
  budget bounds the stable phase's wall-clock, and a hand stop before the gate costs only the
  cooldown, not a forfeited anneal.
- **Operator stop rule (v2-plan §12):** planned phase completion includes cooldown.
  A soft-stop may pause and later resume the same schedule. Numerical/divergence aborts
  halt immediately; never continue invalid training to finish cooldown. Diagnose and
  recover a finite checkpoint before resuming. Budget exhaustion without convergence is
  reported as budget-limited, with cooldown time reserved in advance.
- **Stable-phase readouts:** annealing can lower loss and improve mAP; neither level
  shift is guaranteed, and a lower loss is better. Use the branched cooldown for an
  annealed readout. Validate the provisional .003/window=3 gate using detrended temporal
  fluctuations; distinguish these from paired sampling uncertainty and binning error.
- **Epoch budget:** state it up front and hold to it. It is now the *only* thing that ends an
  advise-mode run, and every stop-check log line reports position against it (`budget` trigger).
  V1-P2 was budgeted 15 epochs and manually soft-stopped at 6 — see [v2-plan.md](v2-plan.md) for
  what that measurement implies.

### Resolved (2026-07-29) — the stop system was built

[stop_conditions.py](../stop_conditions.py) + wiring in `train_direct.py`, tests in
[test_stop_conditions.py](../test_stop_conditions.py). What changed:

- **Stop decisions are pure functions of `TrainingState.eval_history`**, one record per *validated*
  epoch, keyed by epoch and replaced on replay. No counters, so a verdict is identical across a soft
  stop and a re-validated epoch cannot double-count. The old `patience_counter` is derived
  (epochs-since-best) for reporting only.
- **`early_stopping_threshold` (5e-7) is no longer the stop threshold** — it now governs
  best-checkpoint selection only, where near-zero is correct. The stop threshold is the new
  `early_stopping_min_delta: 0.003`, a provisional practical-significance setting per
  [v2-plan.md](v2-plan.md) §8.2. The earlier "~1e-3" suggestion in this section was superseded: 1e-3
  sits within a factor of 1.5 of V1-P2's *smallest real* per-epoch gain (+0.00148).
- **The plateau test is window-over-window, not per-epoch**, precisely because 3e-3 straddles a single
  epoch's real gain. V1-P2's E0–E5 replay is a test case: it must not stop.
- **The `lr < 0.5 × scheduler.max_lr` patience gate is gone.** On `num_cycles: 1` it was false until
  the midpoint of the anneal, so the auto-stop could not fire before ~epoch 8.5 of P2's 15 — with
  patience 4, the earliest possible stop was ~epoch 13. The floor is now the explicit
  `early_stopping_min_epochs`.
- **Burn-in `strategy` is honored.** The old code did `max(baseline, max(burn_in_values))`, always the
  max, so `median`/`mean`/`last` were dead and the post-burn-in bar was the burn-in high-water mark.
- **The macro-average tag list is frozen per phase** (`validation.freeze_macro_tag_list`) and
  persisted in the checkpoint, so the metric denominator cannot drift between the epochs a stop rule
  compares. Drift against the frozen list is logged as a warning rather than absorbed.

### Scope limits of the stop rules — read before trusting one

Verified 2026-07-29. Each rule is sound for what it measures; none of them measures the project's
central worry, and one of them cannot in principle.

- **`plateau` measures window maxima, not a noise floor.** Fixed EVAL removes repeated
  resampling but does not make finite-sample error a common additive offset. Binning error
  can also change with calibration. A larger `min_delta` makes plateau easier to trigger;
  it does not guarantee longer training. Recheck the provisional .003/window=3 on V2's
  first ~10 stable-phase validations and freeze before arming (v2-plan §6/§8.2).
- **`peak_decline` cannot identify the cause.** Missing labels can distort AP in either
  direction; a falling noisy-label score does not prove overtraining or better true-label
  recognition. A holding/rising sibling gap cannot excuse a decline, since wrong-positive
  memorization can maximize it. The code and regressions now enforce this veto-only rule;
  the old demotion is fixed. Reviewed real labels adjudicate the underlying quality.
- **The boundary observable: `asl_val/sibling_gap_macro` — veto-only, never a green light (sign
  corrected 2026-07-30, v2-plan §9.2).** For val rows where exactly one member of a confusable
  group (`hair_color`, `eye_color`, `hair_length`, `breast_size` —
  `configs/confusable_groups.json`) is labelled positive, the mean of `p(labelled) − max
  p(unlabelled siblings)`. The old claim that a sibling-positive label "is reliable evidence of
  negativity" is **withdrawn**: the filter only requires one sibling to be *labelled*, which does
  not exclude an unlabelled co-present sibling, and a memorized wrong positive *maximizes* this
  margin — the observable is *increasing* in exactly the noise mode no loss knob can touch (ρ is
  not identified from the mined campaign, v2-plan §10). So: a **falling** margin corroborates a decline; a holding/rising margin
  proves nothing and must not silence any stop signal. Computed by `asl_telemetry.compute_val`,
  rides in `eval_history` as `clean_margin`; it never halts on its own. A *missing* value reads as
  `pending`, never as agreement — check `training.asl_telemetry.enabled` and that
  `sibling_groups_path` resolves.
- **`overfit_loss` is structurally blind to noise memorization.** It detects classic overfitting —
  fitting train idiosyncrasy the val set does not share. Missing-positive memorization is not that:
  train and val are drawn from the *same* noisy annotation process, so memorizing the shared noise
  *lowers* `val/loss` (Zhao & Gomes, arXiv:2102.08427). It is also not comparable across a loss
  change; the rule carries a `loss_id` per record and returns `pending` on a window that straddles
  one (defensive only — no γ steps are planned in V2, v2-plan §4). Keep it — it is cheap and it caught nothing in
  V1 for the good reason that there was nothing to catch (zero train/val gap) — but do not read it as
  the anti-memorization detector.
- **`advise` remains the default even with the arbiter wired.** The arbiter covers four confusable
  groups; it can **corroborate** a `peak_decline` (falling margin) but can neither demote it nor
  certify a `plateau` (v2-plan §9.2, veto-only), and the gold slice — the only broad clean-label
  instrument — is still not in the training validation loop. Switch to `halt` per phase once the
  bands have been re-derived on V2's own val curve.
- **Plateau coverage is per-phase now** (`training.early_stopping_phases`, keyed by phase).
  The rule cannot confirm before `burn_in + 2·window + confirm − 1` validated epochs, so one geometry
  cannot serve a ~55-epoch from-scratch phase and a 3–4 epoch detail phase: at `window=3/confirm=2`
  the latter can never confirm. Shipped defaults, per §7's phase table — phase 1 `window=3/confirm=2`
  (first confirmable E9), phase 2 `window=2/confirm=1` (E6), phase 3 `mode: off` ("run it, don't gate
  it"). The effective geometry is logged at startup with its coverage, config validation errors if the
  phase's budget cannot reach it, and a coverage under 3 epochs warns.

## Missing-positive bias diagnostics (per-epoch scalar fingerprints)

`val/mAP` alone is partly blind to suppression of unlabeled-but-correct tags, because the val set is
also missing-positive-noisy (Zhao & Gomes, *Evaluating Multi-label Classifiers with Noisy Labels*,
[arXiv:2102.08427](https://arxiv.org/abs/2102.08427)). The weakly-supervised multi-label literature
(Cole et al., *Multi-Label Learning from Single Positive Labels*,
[CVPR 2021](https://arxiv.org/abs/2106.09708); Liu et al., *Early-Learning Regularization*,
[NeurIPS 2020](https://arxiv.org/abs/2007.00151); Kim et al., *Large Loss Matters*,
[CVPR 2022](https://arxiv.org/abs/2206.03740)) treats memorization of noisy negatives as a per-epoch
phase transition with detectable scalar signatures, not an end-of-training mystery.

**None of these were logged for V1-P2.** Three of the five now exist (2026-07-29), computed
**streaming per validation batch** — scalar accumulators only, no full-matrix retention, so they stay
bounded at the current 30K EVAL cap (v2-plan §8.1); a larger holdout is deferred. Enabled by
`validation.log_memorization_fingerprints`; emitted as `val_fingerprint/*` in TensorBoard and carried
in each `eval_history` record. They are flagged after two consecutive moves in the problem direction
(`stop_conditions.fingerprint_advisories`) and are marked **advisory** in the verdict, so they can
never contribute to a stop action — per the "How to use" note below. Status per row:

| Diagnostic | Status |
|---|---|
| `pred_pos_ratio` | **live** — streaming, at `inference.prediction_threshold` |
| `mean_sigmoid_topK_unlabeled` | **live** — streaming, `validation.fingerprint_topk` (default 10) |
| `logit_std_rank_11_50` | **live** — streaming, `validation.fingerprint_rank_window` (default 11–50), computed in logit space via the inverse sigmoid |
| `cooccur_jaccard@K` | not built — needs training-set tag-pair frequencies loaded into the val loop |
| `rare_bucket_ECE` | not built — needs per-bucket calibration bins in the val loop |

| Diagnostic | Computation | Direction = problem | Citation |
|---|---|---|---|
| `pred_pos_ratio` | `mean_active / mean_labeled_positives_per_image` | Falls monotonically below ~1.0 = suppression collapse (Cole's Expected Positive Regularizer formalizes the same ratio) | Cole CVPR 2021 |
| `mean_sigmoid_topK_unlabeled` | mean σ on top-K logits *not in the GT label set*, K≈10, per image, averaged across val | Falls monotonically = model is learning to suppress co-occurring/unlabeled-correct tags | Liu NeurIPS 2020 (ELR) |
| `cooccur_jaccard@K` | Jaccard between pred-pair frequencies and training-set tag-pair frequencies, top-K pairs | Falls = predicted co-occurrence structure diverging from real co-occurrence (fitting the labeled subset, not the world) | Zhao & Gomes 2021 |
| `rare_bucket_ECE` | Expected Calibration Error restricted to low-frequency-bucket tags | Rises while rare-bucket mAP plateaus = calibration drift before metric drift | Wei et al., *To Smooth or Not?*, [ICML 2022](https://proceedings.mlr.press/v162/wei22b.html) |
| `logit_std_rank_11_50` | Std of logits at ranks 11-50 per image, averaged | Compresses (decreasing std) = second-tier predictions collapsing onto suppression mode | Kim CVPR 2022 |

**How to use:** direction-only signals, like canary #7. Two consecutive monotonic moves in the
"problem" direction is the trigger to (a) spot-check specific predictions early rather than waiting
for the end of the run, and (b) escalate to the plan — **do not change loss hyperparameters
mid-run; γ_neg is fixed at 7 and the loss is a single configuration (v2-plan §4, 2026-07-30)**.
None are sufficient cause to stop training on their own — they are confirmatory diagnostics
for a `val/mAP` pattern that already looks suspect.

**⚠️ That last clause is a gate, and the implementation enforces it.** All three live fingerprints are
calibration-coupled, and LR decay moves all three in the "problem" direction during entirely healthy
training — canary #7 says so outright about the closely-related `mean_active` ("Healthy: monotonic
decline as logits tighten with LR decay"; under WSD that excuse exists only inside a cooldown —
see #7's carve-out). A bare two-consecutive-moves test therefore fires on every
good run, which is canary #1's failure mode rebuilt one layer down, and the way an operator learns to
ignore a flag. `stop_conditions.fingerprint_advisories` flags a fingerprint only when **the selection
metric also failed to improve by more than `min_delta` across the same span**. Separately,
`val/pred_pos_ratio` (the fingerprint stored as `pred_pos_ratio`) counts predictions
**above a threshold** divided by observed positives. Its advisory <1 threshold is an
engineering heuristic, not a conclusion of Cole's paper; its meaning depends on the
operating threshold and annotation coverage. Separately, `asl[_val]/epr_decile_*` sums
**raw scores** over observed positives, without thresholding. Neither has a universal
healthy level. In particular, raw ASL score mass need not approach 1/(1−ρ), even with
complete labels; use trends alongside ranking progress and reviewed-real recall.
The formation watch and diagnosed γ4 restart contingency are owned by v2-plan §4;
this diagnostic alone never changes the loss or orders a restart.

**Implementation note:** the three live rows needed no logit-retention hook after all — a `topk` plus
scalar accumulators per batch is enough, and that is what shipped. Retaining logits for the whole val
pass (the originally-assumed design) would not have survived the val-set expansion in
[v2-plan.md](v2-plan.md) §8.1. The two remaining rows need extra per-tag structures in the val loop,
not retained logits.

## Weight decay — keep FIXED at 0.05 (research-settled 2026-06-04)

A verified literature pass concluded the dormant `inverse_sqrt`-by-dataset-size WD helper is *wrong*:
optimal AdamW λ scales ~1/N and linearly in batch, not 1/√N, and the helper also misreads
Loshchilov–Hutter normalized WD. Scaling LR by √batch while holding WD fixed is the correct recipe
(DeiT: fixed 0.05). At V1-P2's settings the WD EMA timescale τ_epoch ≈ 105–137 epochs ≫ the 15-epoch
budget, so WD was doing very little — do **not** raise it to fight missing-positive label noise; that
lever is the **loss** (fixed ASL, γ_neg=7 — v2-plan §4), not the regularizer.

## Per-tag-group AP slices

**Adopted scope:** V2 §7's 2026-08-01 monitoring riders cover luminance-coded pairs
(`white_hair`/`grey_hair`, `dark_skin`/`pale_skin`) and the letterboxed/border tag group.
These are selected per-tag AP diagnostics, not per-tag F1 or a new selection metric.
Resolve group members by vocabulary name and record the list with the run artifacts.

- Accumulate AP over the validation event using the same score precision and binned-AP
  convention as the main instrument; do not average independently computed batch APs.
  V2 §12 audit #2 requires the fp32 cast **before** sigmoid.
- Report per-tag AP and positive support, grouped for comparison against the fixed V2
  baseline epoch on the same EVAL/TAGSET. Mark unsupported tags explicitly.
- Treat changes as direction-only diagnostics for augmentation exposure. Do not copy V1
  numeric bands or invent a stop threshold from these small slices.
- Logging remains implementation work; this section supplies the operational home for the
  already-adopted riders, not evidence that the logger emits them today.

**Pending expansion:** round 2 proposes hue-control pairs, script-text tags, `dutch_angle`,
and the blur family, plus a phase-boundary brightness/contrast fallback. These remain
proposals in [the augmentation decision record](v2-augmentation.md), not active canaries or
an automatic permission to alter the recipe mid-run.

## Things explicitly *not* to track

- **Per-tag F1.** Too noisy, too many tags. Investigate via TensorBoard if a bucket-level flag fires. The selected per-tag **AP** slices above are the explicit exception for augmentation diagnostics.
- **Train mAP.** Not logged at the epoch level. Trust train loss curves instead.
- **Wall-clock timing or throughput.** Not a model-health signal — but v2-plan §12 requires a
  throughput/wall-clock budget, so track it in the run log; it is only excluded as a *canary* here.
- **EMA divergence.** No EMA model exists in the codebase, and v2-plan §6 cut it (2026-08-01), so
  there is no EMA-vs-online gap to watch. Excluded permanently, not pending.
- **AUROC.** Not computed in the in-training validation loop.
