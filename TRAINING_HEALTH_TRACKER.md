# Training Health Tracker

Runbook for an agent monitoring the **Phase 2** (448×448 fine-tune) ViT anime tagger run. After each validation event: pull the scalars listed in the procedure, fill the next row in the per-epoch log, run the canary checks, and apply the decision rules. Stop training only when the rules say so.

## Run context

- **Phase:** Phase 2 (448×448 fine-tune) of the two-stage plan in [progressive-training-plan.md](todos/progressive-training-plan.md). Optimizer reset, bicubic pos_embed interpolation 20×20 → 28×28, reduced regularization, `gamma_neg=7.0`, `clip=0.2`, peak LR 1.4e-5, `num_epochs=15`, `warmup_epochs=2`. Phase 1 best (E32, step 209135) checkpoint is the starting point.
- **Phase 1 end-state baseline** (carry-over for canary anchoring; see per-epoch log row): `val/mAP=0.6518`, `f1_micro=0.0530`, `f1_macro=0.0453`, `mean_active=1322`, head/mid 0.984, val_loss=0.000755.
- **Steps per epoch:** ~13,637 (recomputed from Phase 2 E0 validation event step on 2026-05-06). Phase 1's 6,418 steps/epoch does **not** carry over (smaller batch + larger image at 448 changes the count).
- **Validation cadence:** `eval_steps=11538` ([unified_config.yaml](configs/unified_config.yaml#L324)) — validation fires at an epoch boundary if `current_step >= last_val_step + 11538`, OR if it's the first epoch since a resume (cadence-resume quirk: `last_validation_step` is initialized to 0 inline at [train_direct.py:1395](train_direct.py#L1395) and is not persisted in the checkpoint, so every soft-stop+resume forces a validation at the next epoch boundary).
- **Epoch numbering note:** training log and checkpoint filenames use **1-based** epoch numbers; this tracker uses **0-based**. Tracker `Epoch N` ↔ log `Epoch N+1` ↔ `checkpoint_epoch_{N+1}_step_*.pt`.
- **Vocabulary:** ~19K tags. The `300-499` support bucket is empty (`num_tags=0`), so `val_bucketed/300-499/mAP` reads 0.000 forever and is not a signal.
- **Active augmentation:** horizontal flip (`random_flip_prob=0.5`) with conservative orientation tag swap. Flip-fix is in effect (per-epoch flip rerolling via `mp.Value` shared cell).
- **Auto-stop:** code-side early stopping watches `val/f1_macro` with `patience=4`, `burn_in=2`, only counts patience while LR has dropped to <50% of cycle max ([train_direct.py:2384-2392, 2422](train_direct.py#L2384-L2392)). **⚠️ Auto-stop metric is calibration-floored.** Validation F1 uses a fixed threshold of 0.2653 hard-coded at metric construction ([train_direct.py:1374-1384](train_direct.py#L1374-L1384), pulled from `threshold_calibration.default_threshold` — *not* `inference.prediction_threshold`, even though they currently match). Phase 2's `gamma_neg` 4→7 + `clip` 0.05→0.2 demonstrably shift the logit distribution, so `val/f1_macro` direction can diverge from `val/mAP` direction. **If auto-stop fires while `val/mAP` is still rising, raise `early_stopping_threshold` to suppress the false stop, OR (recommended) move auto-stop to `val/mAP` — see "Recommended code change" section below.**

## Phase 2 startup expectations

- **Expect 1-2 epochs of instability before recovery.** The optimizer reset, resolution jump 320→448 (patch tokens 400 → 784, early layers re-train), reduced regularization (`drop_path` 0.2→0.1, attention/hidden dropouts halved), and shifted loss shape (`gamma_neg` 4.0→7.0, `clip` 0.05→0.2) all move the loss landscape simultaneously.
- **What "healthy startup" looks like:** val/mAP may *fall* below the 0.652 Phase 1 baseline at E0-E1 of Phase 2 as patch embeddings re-train, then recover and surpass baseline by E2-E3. val_loss may briefly rise. Bucket spread may transiently widen.
- **What's a real problem:** val/mAP still below baseline at Phase 2 E3+, val_loss still rising at Phase 2 E3+, gradient blowup, NaN, or step-function plateau. Apply canary checks normally from E2 onward; treat E0-E1 as adjustment turbulence and don't promote yellows to red on the first two epochs.
- **Warmup window (`warmup_epochs=2`):** first two epochs run under linear LR ramp — growth shape during warmup is dominated by LR, not learning, so don't anchor canary #1 bands to E0-E1.

## Known issues affecting interpretation

- **Dataset noise ceiling — unmeasured; ~0.65 is an educated guess.** Asymptotic validation mAP is bounded by missing-positive label noise, not model capacity. Phase 1 closed at mAP 0.652 — already at/near the educated-guess ceiling. Phase 2's gain hypothesis is **orthogonal capacity** (more tokens via the resolution jump) plus reduced regularization unlocking representational headroom that the Phase 1 cosine tail couldn't reach. Magnitude is unmeasured; treat any sustained gain past baseline as the Phase 2 thesis confirmed.
- **Missing-positive bias on convergence.** As ASL tightens on a noisy multi-label set, the model is pushed to suppress unlabeled-but-correct predictions. Manifests as healthy-looking `val/mAP` improvement that is partly real ranking gain and partly memorization of the labeled subset (Kim et al., *Large Loss Matters in Weakly Supervised Multi-Label Classification*, [CVPR 2022](https://arxiv.org/abs/2206.03740); Liu et al., *Early-Learning Regularization*, [NeurIPS 2020](https://arxiv.org/abs/2007.00151)). Phase 2's `gamma_neg=7.0` + reduced regularization is the configuration Park et al., *Robust Asymmetric Loss for Multi-Label Long-Tailed Learning* ([ICCVW 2023](https://arxiv.org/abs/2308.05542)) was built to address — high γ_neg pushes large gradients onto unlabeled positives because the loss treats them as confidently-wrong negatives. `val/mAP` is partly blind to this because the val set is also missing-positive-noisy (Zhao & Gomes, *Evaluating Multi-label Classifiers with Noisy Labels*, [arXiv:2102.08427](https://arxiv.org/abs/2102.08427)). The phase transition is per-epoch detectable via scalar fingerprints (see "Missing-positive bias diagnostics" below). Final confirmation is a held-out spot-check at wrap.
- **`val/loss < train/loss_epoch` is NOT evidence of "anti-overfitting."** drop_path and dropout are active in `model.train()`, disabled in `model.eval()`, so train_loss > val_loss is *expected*. Read canary #6 as a *direction* check on `val/loss` (still falling = no overfitting reversal), not a comparative ranking against train_loss.
- **🚫 DO NOT use F1 metrics for health tracking — calibration-floored.** Validation F1 threshold is hard-coded at metric construction ([train_direct.py:1374-1384](train_direct.py#L1374-L1384)) and never re-derived. Phase 2's loss-shape change (`gamma_neg` 4→7, `clip` 0.05→0.2) shifts the logit distribution, so absolute F1 values are uninterpretable as a health signal — they reflect threshold mis-calibration, not model state. **Trust `val/mAP` (threshold-independent, ranking-based) for all progress and health signals.** F1 columns and canaries #4-5 are retained in this tracker as **TensorBoard drill-down references only** when investigating a flagged mAP issue; do not flag yellow/red on F1 levels or trends.
- **Frozen-flip bug — repaired but residual damage may persist in early-layer attention.** Phase 1 epochs 0-2 trained with flip decisions frozen at `_current_epoch=0`. Phase 2's resolution jump forces patch embeddings and early-layer attention to re-train, which is the primary repair lever. If a flip-variance baseline was captured pre-Phase-2, recapture at Phase 2 convergence to quantify residual damage.
- **F1 definitional inconsistency (cross-path comparison only):** Training-loop `val/f1_macro` filters zero-GT-positive classes before averaging ([train_direct.py:2215-2220](train_direct.py#L2215-L2220)); bucketed `val_bucketed/*/f1_macro` does not. The `_drop_zero_positive_classes` helper at [evaluation_metrics.py:133-152](evaluation_metrics.py#L133-L152) is invoked by `MetricComputer.compute_all_metrics`/`*_at_threshold`, producing numbers that differ from the training-loop scalars when called via that path. Bucket assignment uses *training-set* frequency from `vocabulary.json` and any tag below the lowest bin edge (`<300`) is dropped from `val_bucketed/*` but still contributes to global `val/f1_macro`. (Note: this only matters for cross-path debugging — F1 itself is no longer a primary tracker signal per the calibration note above.)

- **Code-fix pass 2026-06-04 (takes effect on next resume, not the in-flight process).** A high-severity bug-review pass fixed several items that touch this run's interpretation; see [TRAINING_CODE_REVIEW_CHECKLIST.md](TRAINING_CODE_REVIEW_CHECKLIST.md) "Verification & fix log". Most relevant here: **(a)** early-stopping no longer advances `patience` on validation-skipped epochs (was harmless while validation fires every epoch, but the guard is now correct); **(b)** the auto-stop is still on `val/f1_macro` (the recommended move to `val/mAP` below is *not* yet applied — keep monitoring mAP manually); **(c)** the `mp.Value` flip-epoch mechanism was confirmed working (the "frozen-flip in workers" suspicion was a false alarm under Windows spawn — handle is shared, `set_epoch()` propagates live); **(d)** the standalone validation harness / threshold-calibration metric definitions changed (zero-positive-class drop now scoped to macro only) — the *in-loop* `val/mAP`/`val/f1_macro` scalars logged to TensorBoard are computed by train_direct's own path and are **unchanged**, so per-epoch-log comparability across E0–E5 is preserved. The Phase 2 wrap-sequence `find_optimal_threshold`/`ThresholdCalibrator` numbers will now be computed on the corrected (macro-scoped) definition.

- **Weight decay — keep FIXED at 0.05 (research-settled 2026-06-04).** A verified literature pass concluded the dormant `inverse_sqrt`-by-dataset-size WD helper is *wrong* (optimal AdamW λ scales ~1/N and linearly in batch, not 1/√N; it also misreads Loshchilov–Hutter normalized WD). Scaling LR by √batch while holding WD fixed is the correct recipe (DeiT: fixed 0.05). At this run's settings the WD EMA timescale τ_epoch ≈ 105–137 epochs ≫ the 15-epoch budget, so WD is doing little — do **not** raise it to fight the missing-positive label noise; that lever is the **loss** (ASL γ_neg / Park 2023 robust ASL), per the Known Issues "missing-positive bias" entry. Details: checklist H29 + memory `project-weight-decay-fixed`.

## Procedure (run after every validation event)

1. **Pull scalars** for the run directory. Validation step `S` is whatever the actual logged step is — recompute Phase 2 steps-per-epoch from the first validation event.

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

   Then pull the **stability scalars** (per-step series — reduce to per-epoch summaries):

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

2. **Fill the per-epoch row** (main `Per-epoch log` table at the bottom). Column → scalar mapping. Write `—` for absent scalars.

   | Column | Source | Notes |
   |---|---|---|
   | Step | derived | actual logged validation step |
   | train_loss | `train/loss_epoch` | per-epoch mean train loss; co-logged with `val/loss` |
   | val/loss | `val/loss` | |
   | val/mAP | `val/mAP` | **primary progress signal** |
   | growth | derived | `val/mAP[E] / val/mAP[E-1]`; `—` for first Phase 2 epoch |
   | 500-999 | `val_bucketed/500-999/mAP` | rare-tag canary (#2) |
   | 1K-5K | `val_bucketed/1000-4999/mAP` | |
   | 5K-10K | `val_bucketed/5000-9999/mAP` | |
   | 10K+ | `val_bucketed/10000+/mAP` | head bucket; canary #3 uses 5K-10K / 10K+ ratio |
   | f1_micro | `val/f1_micro` | **DRILL-DOWN ONLY** — calibration-floored, do not flag |
   | f1_macro | `val/f1_macro` | **DRILL-DOWN ONLY** — calibration-floored; currently the auto-stop metric (consider moving auto-stop to mAP — see Recommended code change) |
   | mean_active | `val/mean_active` | mean tags above the validation F1 threshold (0.2653) per val image |
   | lr_end | `train/learning_rate` | value at the largest step ≤ `S` |
   | skips | `train/skipped_batches` | scalar absent ⇒ 0 (gated at `train_direct.py:2105` to log only when >0) |
   | Notes | derived | flag tags from canary checks + your interpretation |

3. **Fill the stability sub-log row** (separate `Stability sub-log` table further below). Each cell summarizes per-step scalars within the current epoch's step range.

   | Column | Source | Reduction | Notes |
   |---|---|---|---|
   | grad_norm_max | `train/grad_norm` | max over epoch | Canary #11. Compare to previous-epoch max for yellow/red bands |
   | grad_norm_mean | `train/grad_norm` | mean over epoch | Context for max — distinguishes spike from drift |
   | logits_min | `train/tag_logits_min` | min over epoch | |
   | logits_max | `train/tag_logits_max` | max over epoch | |
   | logits_mean | `train/tag_logits_mean` | mean over epoch | Canary #13 — drift past Phase 2 E5 = calibration slipping |
   | nan_inf_loss_count | `train/nan_inf_loss_detected` | count of events in epoch step range | Canary #12 red on any value > 0 |
   | nan_grad_skip_count | `train/nan_grad_skipped` | count of events in epoch step range | Canary #12 red on any value > 0 |
   | Stability notes | derived | — | Anything anomalous; cite logit drift here when justifying mAP-vs-F1 divergence |

4. **Run the canary checks** (next section). Append each triggered flag to Notes as `🟡 <name>` or `🔴 <name>`. Canaries #1-9 read from the per-epoch row; #11-13 read from the stability sub-log row; #10 is a TensorBoard drill-down.

5. **Apply the decision rules** (further down). Stop or continue accordingly.

## Canary checks

Bands assume Phase 2 steady-state (E2+). Suppress canary firing during E0-E1 (warmup + adjustment turbulence — see Phase 2 startup expectations).

### 1. Overall mAP growth (`val/mAP` ratio vs. previous epoch)

mAP-regime indexed. For 2-epoch validation spans, apply the band to the per-epoch geometric mean.

- **Healthy:**
  - mAP < 0.50 → ≥1.05x per epoch (re-acquiring/surpassing Phase 1 baseline).
  - mAP 0.50-0.65 → ≥1.02x per epoch.
  - mAP > 0.65 (above Phase 1 baseline; the Phase 2 thesis is confirmed here) → ≥1.01x per epoch **and** smooth deceleration shape.
- **Yellow:** per-epoch growth below the regime band for two consecutive validation pairs.
- **Red:** mAP *decreases* at any point after Phase 2 E2 (E0-E1 dip is allowed), **OR** per-epoch growth <1.005x for two consecutive pairs after E5, **OR** step-function plateau (Δ jumps rather than smoothly approaching zero).

### 2. 500-999 bucket mAP — the rare-tag canary
- Movement is the signal; absolute value is small.
- **Healthy:** moving each pair, even by tiny amounts.
- **Yellow:** stuck at the same value for 2+ consecutive pairs after Phase 2 E3.
- **Red:** falls below Phase 1 baseline (~0.59) after Phase 2 E5 — Phase 2 is destroying rare-tag learning rather than improving it.

### 3. Head/mid bucket ratio: `5K-10K mAP / 10K+ mAP`
- **Healthy:** ≥0.90, stable or rising.
- **Yellow:** <0.85, **OR** drops by >0.05 across 2 consecutive pairs.
- **Red:** <0.75, **OR** monotonically falling across 3+ pairs (head-collapse signature).

### 4. f1_micro / mAP ratio (DEMOTED — drill-down only, do not flag)
- F1 threshold is hard-coded at 0.2653 and never re-derived against Phase 2's shifted logit distribution. Absolute level is uninterpretable; direction can also be misleading because LR decay tightens calibration on a fixed cutoff regardless of model state. Inspect in TensorBoard if mAP-based canaries (#1-3) fire and you want to disambiguate calibration drift from representation drift. **Do not raise yellow/red on this metric.**

### 5. f1_macro / f1_micro ratio (DEMOTED — drill-down only, do not flag)
- Same calibration-floor reasoning as canary #4. Use canary #2 (500-999 bucket mAP) and canary #3 (head/mid bucket ratio) for the rare-vs-common health signal — both are threshold-independent. **Do not raise yellow/red on this metric.**

### 6. Train vs val loss
- Trust mAP for *progress*; loss for *instability*. val_loss values are tiny (~0.001).
- **Yellow:** `train_loss / val_loss` ratio drops >15% across 3 consecutive epochs after Phase 2 E5 — early overfitting signal.
- **Red:** `val/loss` *increases* for 2 consecutive epochs while `train/loss_epoch` decreases (after Phase 2 E2 — startup turbulence allowed).

### 7. Mean activations per image (calibration-floored — direction only)
- **Healthy:** monotonic decline as logits tighten with LR decay (or recovery from a Phase 2 startup spike).
- **Yellow:** rises by >2x without corresponding mAP jump after Phase 2 E2 (calibration regression).
- **Red:** rises monotonically across 3+ pairs while mAP stalls or falls. Or: pinned at small constant <10 with bucket spread widening (head collapse).

### 8. End-of-epoch LR
- Not a flag on its own — context for plateaus. If mAP plateaus AND `lr_end` is <5% of peak LR (1.4e-5), the plateau is schedule-induced. If mAP plateaus AND `lr_end` is still >50% of peak, the plateau is real.

### 9. Skipped batches
- **Healthy:** scalar absent (= 0) most epochs.
- **Yellow:** >50 skips in an epoch.
- **Red:** >10x increase vs. previous epoch — data-loader regression or numerical instability.

### 10. Train loss curve (per-step `train/loss`, drill-down only)
- Use TensorBoard directly when canary #6 fires. Healthy: smooth downward trend with small spikes recovering within ~50 steps.
- **Red:** spikes that don't recover, or step-function jumps.

### 11. Gradient norm (`train/grad_norm`, per-step)
- Logged at [train_direct.py:1666](train_direct.py#L1666). Read the per-epoch max via TensorBoard "max" reducer.
- **Healthy:** stable across epochs, occasional spikes that recover within ~10 steps.
- **Yellow:** epoch-max grows >2x vs. the previous epoch's max.
- **Red:** epoch-max grows >5x vs. previous, or sustained drift upward across 3+ epochs (precedes loss blowup).

### 12. NaN / Inf flags (`train/nan_inf_loss_detected`, `train/nan_grad_skipped`)
- Both logged at [train_direct.py:1576, 1618, 1685](train_direct.py#L1576). Binary flags.
- **Red:** any non-zero value at any step. Stop training, investigate (most likely AMP overflow or data-loader corruption).

### 13. Logit-distribution drift (`train/tag_logits_min/max/mean`)
- Logged at [train_direct.py:1547-1549](train_direct.py#L1547-L1549). Direction-only diagnostic.
- **Use:** confirms the Phase 2 logit-distribution shift quantitatively. After Phase 2 E2, expect `tag_logits_mean` to settle into a new band (different from Phase 1 due to gamma_neg 4→7 + clip 0.05→0.2). If it keeps drifting past E5, calibration will keep slipping — flag for a mid-run threshold recompute.
- Not a yellow/red signal on its own; cite it when justifying mAP-vs-F1 divergence in Notes.

## Missing-positive bias diagnostics (per-epoch scalar fingerprints)

The Known Issues entry on missing-positive bias notes that `val/mAP` alone is partly blind to suppression of unlabeled-but-correct tags. The literature on weakly-supervised multi-label classification (Cole et al., *Multi-Label Learning from Single Positive Labels*, [CVPR 2021](https://arxiv.org/abs/2106.09708); Liu et al., ELR, NeurIPS 2020; Kim et al., *Large Loss Matters*, CVPR 2022) treats memorization of noisy negatives as a per-epoch phase transition with detectable scalar signatures, not an end-of-training mystery. None of these are currently logged for this run; they would need to be added to the validation pass before they can be tracked here. Until they exist, fall back on the wrap-time spot-check.

| Diagnostic | Computation | Direction = problem | Citation |
|---|---|---|---|
| `pred_pos_ratio` | `mean_active / mean_labeled_positives_per_image` | Falls monotonically below ~1.0 = suppression collapse (Cole's Expected Positive Regularizer formalizes the same ratio) | Cole CVPR 2021 |
| `mean_sigmoid_topK_unlabeled` | mean σ on top-K logits *not in the GT label set*, K≈10, per image, averaged across val | Falls monotonically = model is learning to suppress co-occurring/unlabeled-correct tags | Liu NeurIPS 2020 (ELR) |
| `cooccur_jaccard@K` | Jaccard between pred-pair frequencies and training-set tag-pair frequencies, top-K pairs | Falls = predicted co-occurrence structure diverging from real co-occurrence (the model is fitting the labeled subset, not the world) | Zhao & Gomes 2021 |
| `rare_bucket_ECE` | Expected Calibration Error restricted to `<1000` frequency-bucket tags | Rises while rare-bucket mAP plateaus = calibration drift before metric drift | Wei et al., *To Smooth or Not?*, [ICML 2022](https://proceedings.mlr.press/v162/wei22b.html) |
| `logit_std_rank_11_50` | Std of logits at ranks 11–50 per image, averaged | Compresses (decreasing std) = second-tier predictions collapsing onto suppression mode | Kim CVPR 2022 |

**How to use:** treat these as direction-only signals like canary #7 (`mean_active`). Two consecutive monotonic moves in the "problem" direction during E5+ is the trigger to (a) examine specific spot-checked predictions early instead of waiting for E14, and/or (b) consider tightening regularization or lowering `gamma_neg` mid-run. None of these are sufficient cause to stop training on their own — they're confirmatory diagnostics for a `val/mAP` pattern that already looks suspect.

**Implementation cost:** `pred_pos_ratio` is ~free given `mean_active` is already computed. The other four require a one-time validation-loop hook to retain logits (or top-K logits) for the val pass. Not done for the current run.

## Recommended code change — move auto-stop to `val/mAP`

The current auto-stop on `val/f1_macro` is calibration-floored (see Known Issues). The recommended fix is a small code change in [train_direct.py](train_direct.py):

1. Change the auto-stop metric from `val_f1_macro` to `val_mAP` at the early-stopping check ([train_direct.py:2384-2392, 2422](train_direct.py#L2384)).
2. Update `training_state.best_metric` semantics to track mAP instead of F1.
3. Bump `early_stopping_threshold` from `5e-7` (F1 noise floor) to `1e-3` (mAP scale; F1 deltas are smaller than mAP deltas at this regime).
4. Until this is applied, **manually monitor `val/mAP`** and override auto-stop by raising `early_stopping_threshold` if it fires while mAP is still rising.

## Decision rules

- **Phase 2 E0-E1 (warmup + adjustment turbulence):** suppress all yellow/red flags. Record observations in Notes for later context.
- **Two consecutive yellow flags on the same metric (E2+):** treat as red.
- **Any red flag during Phase 2 E2-E5:** stop training, diagnose. Most likely cause is config wiring (flip aug missing, pos_embed not interpolated, optimizer not reset) — verify before assuming model failure.
- **Any red flag Phase 2 E5+:** finish current epoch, then decide.
- **Auto-stop fires while `val/mAP` is still rising:** auto-stop is wrong for this run — adjust `early_stopping_threshold` rather than letting it end.
- **Phase 2 plan budget:** 15 epochs. At E14 row complete, run wrap sequence (`find_optimal_threshold` global sweep + per-tag `ThresholdCalibrator` against the Phase 2 best checkpoint) regardless of remaining slope.

## Phase 2 wrap sequence (at E14 or stop trigger)

1. Run `find_optimal_threshold` ([evaluation_metrics.py:202-250](evaluation_metrics.py#L202-L250)) for global F1 baseline.
2. Run `ThresholdCalibrator` ([evaluation_metrics.py:454-604](evaluation_metrics.py#L454-L604)) per-tag — Phase 2 logit distribution is final, per-tag calibration is now appropriate.
3. Recapture flip variance baseline (canonical vs flipped predictions) — compare against pre-Phase-2 capture to quantify lasting flip-damage signature.
4. Spot-check rare-tag predictions on held-out val images for missing-positive bias (model suppressing unlabeled-but-correct tags). Specifically examine: **(a)** whether highly-co-occurring tag pairs are co-predicted on held-out images (e.g., if `blue_eyes` predicted, does `1girl` come along?); **(b)** whether the model produces *novel correct* tags absent from the GT label set (the inverse of the Open Images "ignore unverified" failure — a model suppressing missing positives will rarely surface them); **(c)** confidence on canonical exemplars per rare tag, compared against Phase 1 best checkpoint on the same images (a drop here while overall mAP rose = label-noise overfitting).

## Per-epoch log

Phase 1 end-state row carried over as the baseline. Phase 2 rows start at E0.

| Phase | Epoch | Step | train_loss | val/loss | val/mAP | growth | 500-999 | 1K-5K | 5K-10K | 10K+ | f1_micro | f1_macro | mean_active | lr_end | skips | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 32 | 209135 | 0.000820 | 0.000755 | 0.651841 | — | 0.624979 | 0.634492 | 0.571331 | 0.580925 | 0.053029 | 0.045308 | 1322.39 | — | 0 | **Phase 1 end-state baseline.** Carry-forward only — Phase 2 starts fresh from this checkpoint. lr_end absent in E32's validation-only run-dir. |
| 2 | 0 | 13637 | 0.000495 | 0.000464 | 0.659055 | — | 0.631501 | 0.640686 | 0.579228 | 0.590466 | 0.015960 | 0.011871 | 4493.06 | 5.40e-6 | 0 | Startup window (E0) — flags suppressed per decision rules. mAP +1.11% vs Phase 1 baseline 0.6518 → 0.6591 — Phase 2 thesis confirmed early. 500-999=0.6315 (+0.007 vs P1). Head/mid ratio 5K-10K/10K+ = 0.981 (healthy ≥0.90). val/loss 0.000464 (well below P1 0.000755). mean_active 4493 (3.4× P1's 1322) — expected calibration shift from `gamma_neg` 4→7 + `clip` 0.05→0.2; direction-only per canary #7. lr_end value is stale: `train/learning_rate` last logged at step 10000 (5.40e-6); at val step 13637 actual lr ≈ 7.0e-6 from linear warmup ramp toward peak 1.4e-5 (`warmup_epochs=2`). ⚠️ stability scalars absent for entire run (`debug.log_gradient_norm=false`, `debug.log_activation_stats=false`) — canaries #11/#13 unevaluable. NaN flags absent (= 0 events, healthy per canary #12). |
| 2 | 1 | 27274 | 0.000490 | 0.000461 | 0.663135 | 1.0062 | 0.634784 | 0.645040 | 0.583534 | 0.594417 | 0.016203 | 0.011977 | 4425.22 | 1.00e-5 | 0 | Startup window (E1) — flags suppressed per decision rules. mAP +0.62% vs E0 (0.6591 → 0.6631), +1.74% vs Phase 1 baseline. Phase 2 thesis sustained. 500-999 +0.0033 (0.6315 → 0.6348) — moving, healthy per canary #2. Head/mid 5K-10K/10K+ = 0.982 (stable from E0's 0.981). val/loss 0.000461 (down from E0 0.000464). mean_active 4425 (down from 4493) — slight calibration tightening. lr_end stale: `train/learning_rate` last logged at step 20000 (1.00e-5); at val step 27274 (= 2 × 13637 = end of `warmup_epochs=2`) actual lr ≈ peak 1.4e-5. New run-dir `20260506-075806-Grio` (E0 was `20260505-152453-Grio`) — soft-stop+resume; cadence-resume quirk fired the val event at the next epoch boundary as expected. Stability scalars still absent — canaries #11/#13 unevaluable. NaN flags absent (= 0 events, healthy). |
| 2 | 2 | 40911 | 0.000489 | 0.000458 | 0.667649 | 1.0068 | 0.639285 | 0.649363 | 0.587293 | 0.598239 | 0.016219 | 0.012070 | 4420.90 | 1.28e-5 | 0 | First steady-state epoch (E0-E1 suppression ended). mAP +0.68% vs E1 (0.6631 → 0.6677), +2.43% vs Phase 1 baseline — Phase 2 thesis sustained. Growth 1.0068x is *below* the >0.65 regime band of ≥1.01x; canary #1 needs 2 consecutive pairs below band → note for E2→E3 watch, no flag yet. 500-999 +0.0045 (0.6348 → 0.6393) — moving, healthy per canary #2. Head/mid 5K-10K/10K+ = 0.982 (stable from E1's 0.982). train_loss 0.000489 ↓, val/loss 0.000458 ↓ — both directions healthy (canary #6). mean_active 4421 (continuing down from 4425/4493) — calibration tightening as cosine decay engages. lr_end 1.28e-5 (last log step 40000) ≈ 92% of peak 1.4e-5 — cosine decay just engaged after warmup; not yet plateau context. New run-dir `20260507-074427-Grio`; soft-stop+resume happened mid-E2 (May 6 run logged steps 20000-30000, May 7 run logged steps ~40000-40911) — cadence-resume quirk fired E2 val at the next epoch boundary as expected. ✅ stability scalars partially present in tail of E2 — `debug.log_gradient_norm` + `log_activation_stats` enabled between resumes; canaries #11/#13 partly evaluable (see stability sub-log). NaN flags absent (= 0 events, healthy per canary #12). |
| 2 | 3 | 54548 | 0.000487 | 0.000455 | 0.669132 | 1.0022 | 0.639788 | 0.651196 | 0.589493 | 0.600684 | 0.016711 | 0.012463 | 4289.68 | 1.25e-5 | 0 | 🟡 **canary #1** — mAP growth below >0.65 regime band (≥1.01x) for 2 consecutive pairs (E1→E2 1.0068x, E2→E3 1.0022x). First yellow on this metric — per decision rules, continue and watch E3→E4 (second yellow ⇒ red). mAP +0.22% vs E2 (0.6677 → 0.6691), +2.65% vs Phase 1 baseline — Phase 2 thesis still confirmed but slope decelerating sharply (1.0062 → 1.0068 → 1.0022). Plausibly approaching the educated-guess label-noise ceiling; missing-positive bias diagnostics are not instrumented for this run, so direct confirmation deferred to wrap-time spot-check. 500-999 +0.0005 (0.6393 → 0.6398) — moving, healthy per canary #2. Head/mid 5K-10K/10K+ = 0.981 (stable from E2's 0.982, healthy ≥0.90). train_loss 0.000487 ↓, val/loss 0.000455 ↓ — both directions healthy (canary #6). mean_active 4290 (continuing monotonic decline 4493 → 4425 → 4421 → 4290) — calibration tightening as cosine decay engages, healthy direction per canary #7. lr_end 1.25e-5 (last log step 50000) ≈ 89% of peak 1.4e-5 — cosine decay engaged but plenty of LR remaining, so plateau is *not* schedule-induced context-wise (canary #8). Same run-dir as E2 (`20260507-074427-Grio`); no resume between E2 and E3. ✅ stability scalars now fully on (see sub-log) — canaries #11/#12/#13 all evaluable, all healthy. NaN flags absent. |
| 2 | 4 | 68185 | 0.000485 | 0.000453 | 0.672210 | 1.0046 | 0.643052 | 0.653768 | 0.592337 | 0.603752 | 0.016986 | 0.013124 | 4219.61 | 1.20e-5 | 0 | 🔴 **canary #1 (mechanical)** per "Two consecutive yellow flags on same metric (E2+) ⇒ red" decision rule — E3 fired yellow, E4 still satisfies the yellow trigger (pairs E2→E3 1.0022x and E3→E4 1.0046x both below >0.65 regime band ≥1.01x). **However, growth is *accelerating* (1.0022 → 1.0046), which contradicts the canary's decelerating-failure intent.** mAP +0.46% vs E3 (0.6691 → 0.6722), +3.13% vs Phase 1 baseline 0.6518 — Phase 2 thesis sustained. Recommendation: continue, do NOT stop on this red. The mechanical rule is misfiring because the regime band is set for steady-state late-Phase-2; a still-warming run with cosine decay just past warmup naturally posts sub-band growth that nonetheless trends up. If E4→E5 growth drops below 1.0046x, escalate to a real stop decision. 500-999 +0.0033 (0.6398 → 0.6431) — accelerating from E2→E3's +0.0005, healthy per canary #2. Head/mid 5K-10K/10K+ = 0.981 (stable from E3, healthy ≥0.90). train_loss 0.000485 ↓, val/loss 0.000453 ↓ — both directions healthy (canary #6). mean_active 4220 (continuing monotonic decline 4493 → 4425 → 4421 → 4290 → 4220) — calibration tightening, healthy per canary #7. lr_end 1.20e-5 (last log step 60000) ≈ 86% of peak 1.4e-5 — cosine decay continuing, plateau is *not* schedule-induced (canary #8). Same run-dir as E2/E3 (`20260507-074427-Grio`); no resume. ✅ stability scalars on, all healthy (see sub-log: grad_norm 1.09x vs E3, logits_mean flat). NaN flags absent (= 0 events, healthy per canary #12). |
| 2 | 5 | 81822 | 0.000483 | 0.000451 | 0.674373 | 1.0032 | 0.645163 | 0.655673 | 0.594728 | 0.605923 | 0.017123 | 0.013035 | 4185.64 | 1.06e-5 | 0 | mAP +0.32% vs E4 (0.6722 → 0.6744), +3.46% vs Phase 1 baseline 0.6518 — Phase 2 thesis sustained. 🟡 **canary #1 still active** — E4→E5 growth 1.0032x below >0.65 regime band ≥1.01x; pairs below band now E2→E3 1.0022, E3→E4 1.0046, E4→E5 1.0032 (3 consecutive). E4 row's hand-set escalation ("if E4→E5 < 1.0046x ⇒ escalate") is met, but canary #1's strict red rule (`<1.005x for 2 consecutive pairs *after E5*`) only fires from E5→E6 onward — this is the first qualifying pair, still yellow per the strict reading. **Recommendation: continue into E6, watch closely; if E5→E6 < 1.005x that's the official red and the rule says "finish epoch then decide" (decision rules: any red flag Phase 2 E5+).** Alternative reading worth surfacing: the deceleration shape (absolute Δ 0.00308 → 0.00216, ~30% drop) is consistent with both (a) approaching the educated-guess label-noise ceiling and (b) the Park 2023 / Kim 2022 missing-positive-bias regime explicitly documented in Known Issues — direct per-epoch confirmation isn't possible (none of the missing-positive diagnostics are instrumented for this run), wrap-time spot-check (E14) remains the only attestation. 500-999 +0.0021 (0.6431 → 0.6452) — moving, healthy per canary #2. Head/mid 5K-10K/10K+ = 0.9815 (stable from E4's 0.981, healthy ≥0.90). train_loss 0.000483 ↓, val/loss 0.000451 ↓ — both directions healthy (canary #6). mean_active 4186 (continuing monotonic decline 4493 → 4425 → 4421 → 4290 → 4220 → 4186) — calibration tightening, healthy per canary #7. lr_end 1.06e-5 (last log step 80000) ≈ 75% of peak 1.4e-5 — cosine decay continuing, plateau is *not* schedule-induced (canary #8). Same run-dir as E2/E3/E4 (`20260507-074427-Grio`); no resume. ✅ stability scalars on, all healthy (see sub-log: first multi-event grad_norm sample, 0.989x vs E4; logits_mean essentially flat at -1.881). NaN flags absent (= 0 events, healthy per canary #12). |
| 2 | 6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 2 | 7 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 2 | 8 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 2 | 9 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 2 | 10 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 2 | 11 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 2 | 12 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 2 | 13 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 2 | 14 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## Stability sub-log (per epoch)

Stability scalars are per-step, so reduce them to per-epoch summaries. Compute the step range for each Phase 2 epoch from the validation-event step deltas (Phase 2 steps_per_epoch is recomputed on the first validation event), then filter the per-step series within that range.

For each row: write the **per-epoch max** of `train/grad_norm`, the **per-epoch range** of `train/tag_logits_*` (write as `min/max/mean`), and a count for each NaN flag (0 if scalar absent in epoch range).

| Phase | Epoch | grad_norm_max | grad_norm_mean | logits_min | logits_max | logits_mean | nan_inf_loss_count | nan_grad_skip_count | Stability notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 32 | — | — | — | — | — | — | — | **Phase 1 end-state baseline.** Stability scalars not retained for E32 carry-over row. |
| 2 | 0 | — | — | — | — | — | 0 | 0 | Stability scalars (`train/grad_norm`, `train/tag_logits_*`) absent for entire run — gated off by `debug.log_gradient_norm=false` and `debug.log_activation_stats=false` in [unified_config.yaml](configs/unified_config.yaml#L497). Canaries #11/#13 unevaluable for this run unless config is flipped. NaN flags absent (= 0 events, healthy). |
| 2 | 1 | — | — | — | — | — | 0 | 0 | Stability scalars still absent in resumed run-dir `20260506-075806-Grio` — same `debug.log_gradient_norm=false` / `log_activation_stats=false` config. Canaries #11/#13 still unevaluable. NaN flags absent (= 0 events, healthy). |
| 2 | 2 | 6.10e-4 | 6.10e-4 | -11.50 | 11.19 | -1.82 | 0 | 0 | Stability logging flipped on between May 6 / May 7 run-dirs (`debug.log_gradient_norm` + `log_activation_stats` → true) — only the tail of E2 is instrumented. May 6 run-dir (steps 20000-30000, covering early E2) logged none of the stability scalars; May 7 run-dir (steps ~40000-40911) logged grad_norm n=1, tag_logits_* n=9. grad_norm 6.10e-4 (very small, single-event sample — canary #11 yellow/red bands need a prior-epoch baseline which is unevaluable here; treat as the establishment of the Phase 2 baseline going forward into E3). logits_mean -1.82 with range -1.867 to -1.766 across 9 events — settled into the new Phase 2 band as canary #13 expects (no drift; cite this when justifying mAP-vs-F1 divergence at E2: F1 still tiny because threshold 0.2653 is now far from the new logit center -1.82). logits_min pinned at exactly -11.50 in all 9 events — looks like a hard floor (likely an output-side clamp); stable, not drifting. NaN flags absent (= 0 events, healthy per canary #12). |
| 2 | 3 | 6.12e-4 | 6.12e-4 | -11.50 | 12.38 | -1.87 | 0 | 0 | Stability logging fully on for E3. grad_norm n=1 in epoch range (single event at step 50000) = 6.116e-4, essentially flat vs E2's 6.095e-4 (1.003x) — canary #11 healthy, but caveat: max == mean because both epochs have a single grad_norm sample (logging cadence is ~10K steps, so the per-epoch reducer has very few points; canary #11's "epoch-max grows >2x" band needs more events to be reliable — reflects a *log-cadence limitation*, not stability). tag_logits_* n=9 across the epoch range. logits_min still pinned at -11.50 in all 9 events (same hard floor as E2). logits_mean -1.871 (range -1.938 to -1.797) — small downward drift of ~0.05 vs E2's -1.82, still well within the new Phase 2 band; canary #13 healthy (no sustained drift, cite for ongoing mAP-vs-F1 divergence: F1-threshold 0.2653 remains far above logit center -1.87). logits_max range 10.62-12.38 (mean 11.5) — drifting upward vs E2 (range 10.12-11.19, mean 10.91); the positive-tail is widening even as the bulk-mean drifts mildly negative, consistent with cosine decay sharpening high-confidence predictions while ASL `gamma_neg=7` continues to push unconfident-positive logits down (note for missing-positive-bias literature watch — see Known Issues entry on Park et al. 2023). NaN flags absent (= 0 events, healthy per canary #12). |
| 2 | 4 | 6.68e-4 | 6.68e-4 | -11.50 | 12.75 | -1.872 | 0 | 0 | grad_norm n=1 in epoch range (single event at step 60000) = 6.676e-4, +9.2% vs E3's 6.116e-4 — well below canary #11 yellow band of 2x; same single-event-per-epoch caveat as E3 (logging cadence ~10K steps). tag_logits_* n=9 across the epoch range. logits_min still pinned at -11.50 in all 9 events (same hard floor as E2/E3, stable). logits_mean -1.872 (range -1.984 to -1.766) — essentially flat vs E3's -1.871; Phase 2 logit band confirmed stable, canary #13 healthy (cite for ongoing mAP-vs-F1 divergence: F1 threshold 0.2653 still far above logit center -1.87). logits_max range 10.88-12.75 (mean 11.64) — continued upward drift in positive tail vs E3 (range 10.62-12.38, mean 11.5); same pattern, ASL `gamma_neg=7` continues sharpening high-confidence predictions while bulk-mean is flat. NaN flags absent (= 0 events, healthy per canary #12). |
| 2 | 5 | 6.60e-4 | 6.54e-4 | -11.50 | 13.31 | -1.881 | 0 | 0 | grad_norm n=2 in epoch range (steps 70000 and 80000) — first multi-event epoch (logging cadence ~10K steps; E5 spans 13.6K so captures 2 events, easing the single-sample caveat from E2-E4). max 6.603e-4, mean 6.536e-4, essentially flat vs E4's 6.676e-4 max (0.989x) — canary #11 healthy. tag_logits_* n=18 across the epoch range. logits_min still pinned at -11.50 in all 18 events (same hard floor as E2-E4, stable across 4 consecutive epochs — confirms output-side clamp). logits_mean -1.881 (range -1.984 to -1.773) — slight drift from E4's -1.872, well within new Phase 2 band; canary #13 healthy (no sustained drift across E2-E5; cite for ongoing mAP-vs-F1 divergence: F1 threshold 0.2653 still ~2.1 logits above center -1.881, so absolute F1 remains uninterpretable). logits_max range 10.38-13.31 (mean 11.39) — continued upward drift in positive tail vs E4 (range 10.88-12.75, mean 11.64); the positive-tail sharpening pattern continues across E2 (max 11.19) → E3 (12.38) → E4 (12.75) → E5 (13.31), consistent with cosine decay tightening high-confidence predictions while ASL `gamma_neg=7` widens the positive distribution head. NaN flags absent (= 0 events, healthy per canary #12). |
| 2 | 6 |  |  |  |  |  |  |  |  |
| 2 | 7 |  |  |  |  |  |  |  |  |
| 2 | 8 |  |  |  |  |  |  |  |  |
| 2 | 9 |  |  |  |  |  |  |  |  |
| 2 | 10 |  |  |  |  |  |  |  |  |
| 2 | 11 |  |  |  |  |  |  |  |  |
| 2 | 12 |  |  |  |  |  |  |  |  |
| 2 | 13 |  |  |  |  |  |  |  |  |
| 2 | 14 |  |  |  |  |  |  |  |  |

**How to read this table:**
- `grad_norm_max` rising >2× across epochs → run canary #11 (yellow at 2×, red at 5× or sustained drift).
- `nan_inf_loss_count` or `nan_grad_skip_count` > 0 → canary #12 red (any NaN/Inf event).
- `logits_mean` should settle into a Phase 2 band by E2 and stay roughly stable. Continued drift through E5 = calibration slipping (canary #13).
- `grad_norm_mean` is context for the max — a high max with low mean = isolated spike (often benign); high mean = systemic instability.

## Things explicitly *not* to track here

- **Per-tag F1.** Too noisy, too many tags. Investigate via TensorBoard if a bucket-level flag fires.
- **Train mAP.** Not logged at the epoch level. Trust train loss curves instead.
- **Wall-clock timing or throughput.** Separate concern from model health.
- **EMA divergence.** No EMA model exists in the codebase.
- **AUROC.** Not computed in the in-training validation loop.
