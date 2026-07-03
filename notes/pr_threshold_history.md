# P=R threshold history

Per-checkpoint Precision=Recall break-even and F1-optimal points produced by [tools/find_pr_threshold.py](../tools/find_pr_threshold.py). PAD/UNK excluded from all numbers (`SKIP_INDICES = (0, 1)`).

**Comparability caveats**

- `val_samples` differs across rows (full ~296K vs subsampled 30K). Macro on a small sample inflates structural-zero coverage; trust micro for cross-row comparisons unless `val_samples` matches.
- Macro `support_ge_0` includes zero-positive tags (recall undefined → F1 sinks to zero), pulling the mean down. `support_ge_5` is the WD14-style cut.
- F1-optimal recall ≠ F1-optimal precision; `pr_breakeven` is the operating point where both are equal.
- Per-run JSON: `experiments/run1_vit/checkpoints/pr_threshold_<ckpt_stem>.json` — stable snapshots for non-`last` checkpoints are committed alongside.

## Runs

### Phase 2 E6 — `checkpoint_epoch_7_step_85517.pt` (= `last.pt` snapshot, 2026-05-09)

- Source: [pr_threshold_phase2_e6_step85517.json](../experiments/run1_vit/checkpoints/pr_threshold_phase2_e6_step85517.json)
- val_samples: **30,000** (default subsample; not full-val)
- num_tags_evaluated: 19,292
- val/mAP at the same epoch (per training-health tracker, E5 row): 0.6744 — E6 row not yet filled.

| Metric | Threshold | P | R | F1 |
|---|---|---|---|---|
| Micro **P=R** | **0.7927** | **0.6990** | **0.6990** | **0.6990** |
| Micro F1-opt | 0.8050 | 0.7305 | 0.6736 | 0.7009 |
| Macro support_ge_0 P=R | 0.7596 | 0.5970 | 0.5970 | 0.5970 |
| Macro support_ge_0 F1-opt | 0.7570 | 0.5936 | 0.6007 | 0.5716 |
| Macro support_ge_1 P=R (18,334/19,292 tags) | 0.7596 | 0.6281 | 0.6281 | 0.6281 |
| Macro support_ge_1 F1-opt | 0.7570 | 0.6246 | 0.6321 | 0.6015 |
| **Macro support_ge_5 P=R** (10,406/19,292 tags) | **0.7533** | **0.5864** | **0.5864** | **0.5864** |
| Macro support_ge_5 F1-opt | 0.7540 | 0.5887 | 0.5845 | 0.5601 |

**Per-tag operating points (each tag at its own threshold, support>=5, n=10,406):**
- P=R mean t=0.7504 (IQR [0.7140, 0.7884]); mean P=R = 0.5908
- F1-opt mean t=0.7558; mean F1 = 0.6410

### Phase 1 E27 — old `last.pt` (2026-05-03)

- Source: snapshot at [experiments/run1_vit/checkpoints/New folder/pr_threshold_last.json](../experiments/run1_vit/checkpoints/New folder/pr_threshold_last.json) (the top-level `pr_threshold_last.json` was overwritten by the Phase 2 E6 run).
- val_samples: **296,056** (full val split, `--full`)
- num_tags_evaluated: 19,292
- Phase 1 end-state baseline tracker row (E32) was a different/later checkpoint; this is E27 mid-phase.

| Metric | Threshold | P | R | F1 |
|---|---|---|---|---|
| Micro P=R | 0.6704 | 0.6591 | 0.6591 | 0.6591 |
| Micro F1-opt | 0.6890 | 0.6946 | 0.6313 | 0.6614 |
| Macro support_ge_0 P=R | 0.6138 | 0.5880 | 0.5880 | 0.5880 |
| Macro support_ge_5 P=R | 0.6138 | 0.5880 | 0.5880 | 0.5880 |

**Per-tag operating points (support>=5, n=19,290):**
- P=R mean t=0.6220 (IQR [0.5760, 0.6680]); mean P=R = 0.5903
- F1-opt mean t=0.6416; mean F1 = 0.6218

## Phase 2 vs Phase 1 deltas (read with caveats)

- **Threshold shifted right by ~0.13–0.18 across all operating points.** Consistent with the Phase 2 logit-distribution shift documented in [TRAINING_HEALTH_TRACKER.md](../TRAINING_HEALTH_TRACKER.md) Known Issues + canary #13 — `gamma_neg` 4→7 and `clip` 0.05→0.2 widen the positive tail (logits_max E5 = 13.31 vs Phase 1) and push the bulk-mean down (~-1.88), so the optimal cut moves up to compensate.
- **Micro P=R F1 +0.0399** (0.6591 → 0.6990), **Micro F1-opt F1 +0.0395** (0.6614 → 0.7009). Phase 2 thesis confirmed at the operating-point level too, not just val/mAP.
- **Macro support_ge_5 P=R F1 −0.0016** (0.5880 → 0.5864). Effectively flat. Caveat: 30K-sample macro on 10K supported tags is noisier than 296K-sample macro on 19K supported tags, so the "drop" is well within sampling noise. Re-run with `--full` at Phase 2 wrap (E14) for an apples-to-apples comparison.
- **Per-tag mean F1 +0.0192** (0.6218 → 0.6410) — small but consistent gain when each tag is allowed its own threshold.
