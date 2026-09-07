# V1 Phase-2 Run Archive

> **STATUS (2026-07-29): closed archive of the V1 Phase-2 (448×448 fine-tune) ViT anime-tagger run —
> not a live runbook.** The run was manually soft-stopped at E5 (0-based) of a planned 15; rows E6+
> will never be filled and have been removed. Kept because it is the only per-epoch record of V1-P2
> metrics (cited by the v2 evidence base) and because `tools/run_validation_for_epoch.py` still emits
> rows in this file's table format. The canary bands, decision rules and scalar-reduction procedure
> developed here have moved to **[todos/v2-monitoring.md](todos/v2-monitoring.md)**. Operational
> guidance is superseded by [todos/v2-plan.md](todos/v2-plan.md).

## Run context (all past tense — properties of the run, not of current code)

- **Phase:** Phase 2 (448×448 fine-tune) of the retired two-stage progressive plan
  (`todos/progressive-training-plan.md`, deleted 2026-07-29 — full text in git history; superseded by
  [todos/v2-plan.md](todos/v2-plan.md)). **This bullet is the only surviving record of the P2
  configuration:** optimizer reset, bicubic pos_embed interpolation 20×20 → 28×28 (patch tokens
  400 → 784), reduced regularization (`drop_path` 0.2→0.1, attention/hidden dropouts halved),
  `gamma_neg=7.0`, `clip=0.2`, peak LR 1.4e-5, `num_epochs=15`, `warmup_epochs=2`. Started from the
  Phase 1 best checkpoint (**E32, step 209135**).
- **Steps per epoch:** ~**13,637** in P2 (recomputed from the P2 E0 validation-event step). Phase 1's
  **6,418** does not carry over (smaller batch + 448 resolution). Required to convert Step ↔ epoch.
- **Epoch numbering:** the training log and checkpoint filenames use **1-based** epoch numbers; this
  archive uses **0-based**. Archive `Epoch N` ↔ log `Epoch N+1` ↔
  `checkpoint_epoch_{N+1}_step_*.pt`. This is also the convention behind
  `tools/run_validation_for_epoch.py`'s `--epoch-label` default (`ckpt.epoch - 1`).
- **Vocabulary:** ~19K tags. The `300-499` support bucket was empty (`num_tags=0`), so
  `val_bucketed/300-499/mAP` reads 0.000 forever and is not a signal.
- **Augmentation:** horizontal flip (`random_flip_prob=0.5`) with conservative orientation tag swap;
  the flip-fix (per-epoch flip rerolling via the `mp.Value` shared cell) was in effect.
- **Validation cadence:** `eval_steps=11538`; validation fired at an epoch boundary once
  `current_step >= last_val_step + 11538`, **or** on the first epoch after a resume. *During this run*
  `last_validation_step` was process-local — initialized to 0 inline and **not** persisted in the
  checkpoint — so every soft-stop+resume forced a validation at the next epoch boundary. That is why
  the E1 and E2 rows land where they do. It has since been persisted (read from and written back to
  `training_state` in `train_direct.py`), so the quirk does not apply to new runs.
- **Historical F1 threshold — 0.2653.** Every `f1_micro`, `f1_macro` and `mean_active` cell below was
  computed at a frozen validation threshold of **0.2653**, taken from
  `threshold_calibration.default_threshold` at metric-construction time. Without this the three
  columns are uninterpretable. **This is a historical value only:** the config now carries **0.7927**
  at both `inference.prediction_threshold` and `threshold_calibration.default_threshold` (the measured
  micro P=R break-even for this model), and `FullConfig.validate` enforces that the two match.
- **Auto-stop, during the run:** early stopping watched `val/f1_macro` (`patience=4`, `burn_in=2`,
  patience accruing only once LR had fallen below 50% of cycle max) — a calibration-floored metric, so
  its direction could diverge from `val/mAP`, and mAP was monitored manually. Since applied:
  `selection_metric: val_mAP` in `configs/unified_config.yaml`, dispatched through
  `_SELECTION_METRICS` in `train_direct.py`.
- **Startup-window suppression:** E0 and E1 fell inside the planned warmup/adjustment window
  (`warmup_epochs=2`), so all canary flags were suppressed on those two rows — this is why rows E0/E1
  carry no flags. (The pre-run prediction of a 1-2 epoch dip below baseline was wrong: E0 came in
  immediately *above* it.)

## Interpretation caveats

- **`val/loss < train/loss_epoch` is NOT evidence of "anti-overfitting."** drop_path and dropout are
  active in `model.train()` and disabled in `model.eval()`, so train_loss > val_loss is *expected*.
  Read the `val/loss` column as a *direction* check (still falling = no overfitting reversal), never as
  a comparative ranking against `train_loss`.
- **🚫 Do not read the F1 columns as health signals.** The threshold was frozen at 0.2653 and never
  re-derived, while P2's loss-shape change (`gamma_neg` 4→7, `clip` 0.05→0.2) demonstrably shifted the
  logit distribution. Absolute F1 reflects threshold mis-calibration, not model state. `val/mAP`
  (threshold-independent, ranking-based) is the progress signal.
- **F1 definitional inconsistency (cross-path comparison only):** the training-loop `val/f1_macro`
  filters zero-GT-positive classes before averaging (`keep_classes = (val_pos_counts > 0)` in
  `train_direct.py`'s validation block); bucketed `val_bucketed/*/f1_macro` does not.
  `MetricComputer._drop_zero_positive_classes` in `evaluation_metrics.py` is applied on the
  `compute_all_metrics` / `*_at_threshold` path, producing numbers that differ from the in-loop
  scalars. Bucket assignment uses *training-set* frequency from `vocabulary.json`, and any tag below
  the lowest bin edge (`<300`) is dropped from `val_bucketed/*` but still contributes to global
  `val/f1_macro`. (A 2026-06-04 fix scoped the zero-positive-class drop to macro only on the
  standalone/calibration path; the **in-loop** `val/mAP` and `val/f1_macro` scalars are computed by
  train_direct's own path and were unchanged, so E0–E5 comparability across the rows below is intact.)
- **⚠️ A freshly emitted `tools/run_validation_for_epoch.py` row is NOT numerically comparable to rows
  E0–E5.** That tool reads `threshold_calibration.default_threshold` and
  `config.inference.prediction_threshold`, both now **0.7927**, whereas the rows below were computed at
  **0.2653** — so its `f1_micro` / `f1_macro` / `mean_active` cells are on a different operating point.
  Paste any new row below the E5 row and label it, never in-line as a continuation.
  (The *second* half of this caveat is fixed as of 2026-07-29: the tool built
  `MultilabelAveragePrecision` with no `thresholds=` — exact AP — while the training loop used
  `thresholds=ap_thresholds` (200-bin, which biases mAP down). The tool now reads
  `validation.ap_thresholds` too, so the **`val/mAP` cell** is back on the training loop's estimator.
  Rows emitted before that date carry an unquantified upward mAP bias in that cell. The bucket cells
  were never affected — both paths compute them through `FrequencyBucketMetrics`, which is exact in
  the tool and in the training loop alike; see the next bullet.)
- **In-loop `val/mAP` and `val_bucketed/*/mAP` use different AP estimators.** The headline is binned
  (`thresholds=ap_thresholds`, 200) because the streaming metric is fed per batch and the unbinned
  transient is ~7 GB of VRAM; the bucketed path runs on the fully accumulated CPU matrices via
  `multilabel_average_precision` with no `thresholds=`, i.e. exact. Canary #1 reads the first, #2/#3
  read the second — safe, because each compares a scalar only against its own history. Never compare
  a bucket mAP *level* against the headline mAP level. Aligning them is not cheap: binned mode on the
  full 30K × 19.3K matrix in one call would materialize ~10^11 elements, which is why the bucketed
  path stays exact.
- **Column order is the tool's paste contract.** `tools/run_validation_for_epoch.py` prints 16 cells
  to stdout for a human to copy; the per-epoch table has 17 columns (the tool omits the leading
  `Phase` cell). Nothing reads this file programmatically, so headings are free to change — but
  reordering or inserting a table column silently desynchronizes future pastes and nothing will error.
  Absent scalars are written `—`.

## Column schema (compressed — full mapping in [todos/v2-monitoring.md](todos/v2-monitoring.md))

**Per-epoch table:** `Step` = actual logged validation step. `train_loss` = `train/loss_epoch`.
`val/loss`, `val/mAP`, `f1_micro`, `f1_macro`, `mean_active` = the identically-named `val/*` scalars.
`growth` = `val/mAP[E] / val/mAP[E-1]`. `500-999` / `1K-5K` / `5K-10K` / `10K+` =
`val_bucketed/{500-999, 1000-4999, 5000-9999, 10000+}/mAP`. `lr_end` = `train/learning_rate` at the
largest step ≤ the validation step. `skips` = `train/skipped_batches` (absent ⇒ 0; logged only when
>0). `mean_active` = mean tags above the 0.2653 threshold per val image.

**Stability table:** `grad_norm_max` / `grad_norm_mean` = per-epoch max/mean of `train/grad_norm`;
`logits_min` / `logits_max` / `logits_mean` = per-epoch min/max/mean of
`train/tag_logits_{min,max,mean}`; the two NaN counts = number of `train/nan_inf_loss_detected` /
`train/nan_grad_skipped` events in the epoch's step range (absent ⇒ 0 events).

## Flag legend

Only these canaries are cited by the rows below; bands and decision rules are in
[todos/v2-monitoring.md](todos/v2-monitoring.md), where the numbering is preserved.

| Flag | Canary |
|---|---|
| #1 | `val/mAP` per-epoch growth vs the mAP-regime band (>0.65 regime → ≥1.01×) |
| #2 | `500-999` bucket mAP movement (rare-tag canary) |
| #3 | head/mid ratio `5K-10K mAP / 10K+ mAP` (healthy ≥0.90) |
| #6 | train-vs-val loss direction |
| #7 | `mean_active` direction (calibration-floored, direction only) |
| #8 | `lr_end` as plateau context (peak LR 1.4e-5) |
| #11 | `train/grad_norm` epoch-max growth (yellow 2×, red 5×) |
| #12 | NaN/Inf flags — red on any non-zero |
| #13 | `train/tag_logits_*` drift |

🟡 = yellow, 🔴 = red. #4/#5 were F1-ratio canaries, demoted and deleted on relocation; #9 (skips) and
#10 (per-step loss curve) never fired.

## Per-epoch log

Phase 1 end-state row carried over as the baseline. Phase 2 rows start at E0. **E5 is the last row of
the run.**

| Phase | Epoch | Step | train_loss | val/loss | val/mAP | growth | 500-999 | 1K-5K | 5K-10K | 10K+ | f1_micro | f1_macro | mean_active | lr_end | skips | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 32 | 209135 | 0.000820 | 0.000755 | 0.651841 | — | 0.624979 | 0.634492 | 0.571331 | 0.580925 | 0.053029 | 0.045308 | 1322.39 | — | 0 | **Phase 1 end-state baseline** — the anchor for every P2 delta. Carry-forward only; P2 started fresh from this checkpoint. Head/mid 0.984. `lr_end` absent: E32's run-dir was validation-only. |
| 2 | 0 | 13637 | 0.000495 | 0.000464 | 0.659055 | — | 0.631501 | 0.640686 | 0.579228 | 0.590466 | 0.015960 | 0.011871 | 4493.06 | 5.40e-6 | 0 | Startup window — flags suppressed. mAP **+1.11% vs P1 baseline**, i.e. above it immediately. Head/mid 0.981. `mean_active` 4493 = 3.4× P1's 1322, the expected calibration shift from `gamma_neg` 4→7 + `clip` 0.05→0.2. Run-dir `20260505-152453-Grio`. `lr_end` stale: `train/learning_rate` last logged at step 10000 (5.40e-6); actual at step 13637 ≈ 7.0e-6 on the warmup ramp. |
| 2 | 1 | 27274 | 0.000490 | 0.000461 | 0.663135 | 1.0062 | 0.634784 | 0.645040 | 0.583534 | 0.594417 | 0.016203 | 0.011977 | 4425.22 | 1.00e-5 | 0 | Startup window — flags suppressed. mAP +0.62% vs E0, **+1.74% vs P1**. Head/mid 0.982. New run-dir `20260506-075806-Grio` — soft-stop+resume before E1; the cadence-resume quirk fired the val event at the next epoch boundary as expected. `lr_end` stale: last logged at step 20000 (1.00e-5); actual at step 27274 (= 2 × 13637, end of `warmup_epochs=2`) ≈ peak 1.4e-5. |
| 2 | 2 | 40911 | 0.000489 | 0.000458 | 0.667649 | 1.0068 | 0.639285 | 0.649363 | 0.587293 | 0.598239 | 0.016219 | 0.012070 | 4420.90 | 1.28e-5 | 0 | First steady-state epoch (suppression ended). mAP +0.68% vs E1, **+2.43% vs P1**. Growth 1.0068 is *below* the >0.65 regime band (≥1.01×) — first pair below band, no flag yet (canary #1 needs two). Head/mid 0.982. Canaries #6/#7 healthy; `lr_end` ≈ 92% of peak, so no plateau context (#8). New run-dir `20260507-074427-Grio`; soft-stop+resume happened **mid-E2** (May 6 dir logged steps 20000-30000, May 7 dir steps ~40000-40911). Stability logging was flipped ON between those two run-dirs, so only E2's tail is instrumented. |
| 2 | 3 | 54548 | 0.000487 | 0.000455 | 0.669132 | 1.0022 | 0.639788 | 0.651196 | 0.589493 | 0.600684 | 0.016711 | 0.012463 | 4289.68 | 1.25e-5 | 0 | 🟡 **canary #1** — growth below the >0.65 band for 2 consecutive pairs (E1→E2 1.0068, E2→E3 1.0022). First yellow on this metric; per the decision rules, continue and watch E3→E4 (second yellow ⇒ red). mAP +0.22% vs E2, **+2.65% vs P1**; slope decelerating sharply (1.0062 → 1.0068 → 1.0022). Head/mid 0.981. Canaries #6/#7 healthy; `lr_end` ≈ 89% of peak so the deceleration is not schedule-induced (#8). Same run-dir as E2; no resume. Stability scalars now fully on — #11/#12/#13 all evaluable and healthy. |
| 2 | 4 | 68185 | 0.000485 | 0.000453 | 0.672210 | 1.0046 | 0.643052 | 0.653768 | 0.592337 | 0.603752 | 0.016986 | 0.013124 | 4219.61 | 1.20e-5 | 0 | 🔴 **canary #1 (mechanical) — operator overrode, training continued.** The "two consecutive yellows on the same metric ⇒ red" rule fired (pairs E2→E3 1.0022 and E3→E4 1.0046 both below the ≥1.01× band), **but growth was *accelerating* (1.0022 → 1.0046)**, which contradicts the canary's decelerating-failure intent — the band is calibrated for late-P2 steady state, not a run just past warmup with cosine decay engaging. mAP +0.46% vs E3, **+3.13% vs P1**. 500-999 +0.0033, accelerating from E2→E3's +0.0005. Head/mid 0.981. Hand-set escalation condition recorded at the time: *if E4→E5 growth < 1.0046×, escalate to a real stop decision*. `lr_end` ≈ 86% of peak. Same run-dir; no resume. Stability all healthy. |
| 2 | 5 | 81822 | 0.000483 | 0.000451 | 0.674373 | 1.0032 | 0.645163 | 0.655673 | 0.594728 | 0.605923 | 0.017123 | 0.013035 | 4185.64 | 1.06e-5 | 0 | **Final row of the run** — manually soft-stopped here, 6 of 15 planned epochs. 🟡 **canary #1 still active**: E4→E5 growth 1.0032 below band; pairs below band now 1.0022 / 1.0046 / 1.0032 (3 consecutive). E4's hand-set escalation was met, but #1's strict red rule (<1.005× for 2 consecutive pairs *after* E5) had only its first qualifying pair, so it stayed yellow. mAP +0.32% vs E4, **+3.46% vs P1**. Absolute Δ decelerating: 0.00308 → 0.00216 (~30% drop). Head/mid 0.9815. Canaries #6/#7 healthy; `lr_end` ≈ 75% of peak, so the deceleration was **not** schedule-induced (#8). Same run-dir; no resume. Stability all healthy (first multi-event grad_norm sample, 0.989× vs E4). |

## Stability sub-log (per epoch)

Reduced from per-step scalars over each epoch's step range. **Three facts apply to every instrumented
row and are therefore not repeated per row:**

1. `logits_min` sat at exactly **−11.50** in all 45 `tag_logits_*` events across E2–E5 — an
   output-side clamp; stable, not drifting.
   **Sampling artifact (identified 2026-07-29):** those 45 events are 5 distinct steps × 9 microbatches,
   not 45 independent samples. `train/tag_logits_*` was written once per *microbatch* — `global_step` is
   constant across an accumulation window — so every "n=9" cell below is one accumulation window measured
   9 times at a single instant, i.e. within-window batch spread, **not** epoch-wide drift. Read the n=9
   ranges in the rows below accordingly. Fixed in `train_direct.py`: one event per logged step, stamped
   with the same step as `train/loss`.
2. `train/grad_norm` logged on a ~10K-step cadence against ~13.6K-step epochs, so E2–E4 have **n=1**
   sample (hence `max == mean`) and E5 has n=2. Canary #11's "epoch-max grows >2×" band is weakly
   sampled here — a log-cadence limitation, not a stability finding.
3. The frozen F1 threshold **0.2653** sat ~2.1 logits above the logit center (≈−1.87) throughout,
   which is the quantitative reason absolute F1 stayed tiny while `val/mAP` rose (canary #13's
   mAP-vs-F1 divergence citation).

| Phase | Epoch | grad_norm_max | grad_norm_mean | logits_min | logits_max | logits_mean | nan_inf_loss_count | nan_grad_skip_count | Stability notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 32 | — | — | — | — | — | — | — | **Phase 1 end-state baseline.** Stability scalars were not retained for the E32 carry-over row — the absence is itself the fact. |
| 2 | 0 | — | — | — | — | — | 0 | 0 | Stability scalars absent for the whole run-dir. The gate at the time was **compound** — `debug.enabled AND <sub-flag>` — and `debug.enabled` is `false` in every committed config, so E2–E5 having data proves the run carried an uncommitted local edit enabling it; within that, the sub-flags were flipped `false`→`true` between the May 6 and May 7 run-dirs. Either term explains E0–E1's absence and the logs do not distinguish them. **Canaries #11/#13 unevaluable for E0 and E1.** NaN flags absent (= 0 events). (Both sub-flags read `true` in the config today, and as of 2026-07-29 they are the *only* term — `debug.enabled` no longer gates logging.) |
| 2 | 1 | — | — | — | — | — | 0 | 0 | As E0 — still absent in resumed run-dir `20260506-075806-Grio`; #11/#13 unevaluable. NaN flags absent (= 0 events). |
| 2 | 2 | 6.10e-4 | 6.10e-4 | -11.50 | 11.19 | -1.82 | 0 | 0 | Tail of E2 only — logging enabled between the May 6 and May 7 run-dirs; the May 6 dir (steps 20000-30000, early E2) logged none of these. grad_norm n=1; `tag_logits_*` n=9, logits_mean range −1.867…−1.766, logits_max range 10.12–11.19 (mean 10.91). Establishes the P2 stability baseline; no prior-epoch max exists to compare against. |
| 2 | 3 | 6.12e-4 | 6.12e-4 | -11.50 | 12.38 | -1.87 | 0 | 0 | grad_norm n=1 (step 50000, 6.116e-4) = 1.003× E2's 6.095e-4 — #11 healthy. n=9 logits; logits_mean range −1.938…−1.797 (≈0.05 downward drift vs E2, well inside the new P2 band, #13 healthy); logits_max 10.62–12.38 (mean 11.5) — positive tail widening vs E2 while the bulk mean drifts mildly negative. |
| 2 | 4 | 6.68e-4 | 6.68e-4 | -11.50 | 12.75 | -1.872 | 0 | 0 | grad_norm n=1 (step 60000, 6.676e-4) = **+9.2% vs E3**, well below #11's 2× yellow. n=9 logits; logits_mean range −1.984…−1.766, essentially flat vs E3's −1.871 (#13 healthy); logits_max 10.88–12.75 (mean 11.64) — tail drift continuing. |
| 2 | 5 | 6.60e-4 | 6.54e-4 | -11.50 | 13.31 | -1.881 | 0 | 0 | First multi-event epoch: grad_norm **n=2** (steps 70000 and 80000), max 6.603e-4, mean 6.536e-4 = 0.989× E4's max — #11 healthy. n=18 logits; logits_mean range −1.984…−1.773, slight drift from E4's −1.872, inside the P2 band (#13 healthy, no sustained drift across E2–E5); logits_max 10.38–13.31 (mean 11.39). Positive-tail sharpening across the instrumented span: E2 11.19 → E3 12.38 → E4 12.75 → E5 13.31. |
