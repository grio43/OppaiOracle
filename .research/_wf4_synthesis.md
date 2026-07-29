This is a synthesis-and-judgment task. The research and adversarial critique are already done and the code facts verified. I will write the answer directly without further tool calls, since everything load-bearing is confirmed in the evidence and I should not produce report files.

# v1-on-gold fine-tune for the M4 hair-length down-bias: the concrete picture and the verdict

## 1. SHORT ANSWER

The right version is **DFR (last-layer re-training, Kirichenko ICLR'23)**: freeze the entire 18-layer backbone, refit ONLY the single `nn.Linear(1024, num_tags)` head (`model_architecture.py:405`) — ideally only its 6 hair-length output rows — on a group-balanced gold set, with `gamma_neg` dropped from 7→~1-2 and the head bias re-derived from the *balanced gold* prior, not the dirty 6M log-prior. **The one test that decides whether it's even worth attempting** is a frozen-feature ordinal probe: extract v1's post-`norm` CLS vector (`model_architecture.py:665-669`, literally the head's input) and fit a CORN/CORAL probe on it vs the same probe on a frozen DINOv2 backbone — if v1's features separate long↔very_long and lift medium to within a small margin of DINOv2 on a held-out gold split, DFR is viable; if v1 collapses while DINOv2 separates, v1-FT is dead on arrival at every freeze level. What it achieves: a cheap, safe, reversible recalibration of the deployed *suggestion engine* (better medium/very_long predictions in the labeling UI). What it does NOT achieve: it does not rewrite a single on-disk GT label, it cannot exceed v1's feature ceiling, and it must never be used as the bad-data detector (circularity — v1 memorized the exact convention you want surfaced).

## 2. THE SPECTRUM, RANKED

Ordered worst → best **for this defect** (directional-down ordinal bias + must preserve 19K tags):

| Rank | Method | What it touches | Verdict for M4 |
|---|---|---|---|
| WORST | **Full FT** (current default: all 192M + head, `requires_grad` everywhere) | Everything | **The trap.** Maximal feature distortion toward the in-distribution convention (Kumar ICLR'22); the down-bias persists anyway because it's simplicity-biased + the clean set is tiny (Wang ICCV'23 — all three persistence conditions fire); AND gamma_neg=7 hammers the ~19K unlabeled tags as confident negatives → severe catastrophic forgetting. Pays the most, buys the least. **Reject in every branch.** |
| 4th | **Partial unfreeze** (top N blocks + head) | Top blocks + head | Middle of the spectrum, weakest justification. More distortion + forgetting than DFR, still can't reach feature-resident instance-dependent noise. The dormant `_get_layer_wise_params` path (`training_utils.py:2733`) could drive it, but **do not** — see §4 scheduler trap. |
| 3rd | **LP-FT** (Kumar ICLR'22: DFR first, then unfreeze backbone at LR ≪ 1e-5) | Head first, then all | Only worth its feature cost if gold is large enough to early-stop on a clean boundary-val split. For 1.5-3K it adds all of full-FT's risk and buys little over DFR for the hair distinctions. Disciplined fallback, not the recommendation. |
| 2nd | **BitFit / LoRA** | Biases only / low-rank adapters | A forgetting/compute story with **no special debiasing power** over a directional convention. LoRA isn't even in the model (needs A/B modules added to `TransformerBlock`). Strictly dominated by DFR, which is cheaper and changes exactly the right tensor. |
| **BEST** | **DFR / last-layer retrain** (Kirichenko ICLR'23; Kang cRT ICLR'20) | The single `nn.Linear` head (or just its 6 hair rows) | **Recommended.** Zero feature distortion (backbone frozen), zero forgetting if column-restricted, and it's the *only* family that cleanly moves the two head-local components of the bias — the log-prior resting-threshold tilt and the CLS→tag decision boundary — at O(minutes) cost. |

**Recommendation: DFR, restricted to the hair-length head columns.** The bias decomposes into three layers: (i) the head resting-threshold tilt (log-prior bias from dirty frequencies, `model_architecture.py:88`), (ii) the CLS→tag linear boundary (dirty co-occurrence, 76% long-only), and (iii) a feature-resident, instance-dependent residue. DFR overwrites (i) and (ii) — the only parts that are head-local — maximally for its cost. It **cannot** touch (iii); that's the honest ceiling and the entire reason the DINOv2 detector dominates for *detection*.

## 3. THE FEATURE-CEILING GATE — RUN THIS FIRST

This is the single experiment that converts the whole decision from a judgment call into a measurement. It costs minutes because v1's 1024-d post-`norm` CLS vector IS the head input (zero abstraction between feature and head).

**Extraction.** Run v1 in `eval()` + `torch.no_grad()`. The exact tensor you want is `cls_output` at `model_architecture.py:667` (`x = self.norm(x); cls_output = x[:, 0]`). Add a one-line forward hook on `self.norm`'s output taking index 0, or temporarily `return cls_output`. Do the identical extraction on a **frozen DINOv2 ViT** (CLS or mean-pooled patch tokens) over the *same* gold images — this is your non-Danbooru control. Without the control you cannot distinguish "feature absent in v1" from "task is hard / gold too small," and the result is uninterpretable.

**Protocol.**
1. Freeze both backbones; cache 1024-d (v1) and DINOv2 CLS vectors for every gold image.
2. Split gold with **image-level** (not tag-level) stratification; **oversample medium and the long/very_long boundary** so each ordinal bin is balanced in the train split (DFR's group-balanced reweighting requirement — imbalance here silently re-injects the down-bias).
3. Fit a **rank-consistent ordinal probe (CORN/CORAL, Cao et al. 2020)** on the frozen features — not nominal BCE — so it respects `very_short<short<medium<long<very_long<absurdly_long` and shares statistical strength across adjacent bins (far more sample-efficient on 1.5-3K). Use strong L2 + a regularization sweep; average several balanced subsamples to denoise (Kirichenko's variance-reduction trick).
4. **Decision metrics, all on a held-out GOLD split** (never on dirty val): (a) long-vs-very_long balanced accuracy / AUROC; (b) **medium recall** specifically (your collapsed class); (c) ordinal rank metrics (Spearman, mean absolute ordinal error, off-by-one rate). Report all three for v1-frozen AND DINOv2-frozen.

**Decision rule.**
- **v1-frozen ≈ DINOv2-frozen on the boundary** (separates long↔very_long, lifts medium within a small margin) → **the feature EXISTS → greenlight DFR on v1.** Cheap win.
- **v1-frozen collapses long/very_long or can't lift medium WHILE DINOv2 separates them** → **the feature is ABSENT in v1 → DFR is capped at this ceiling, LP-FT/full-FT to re-learn it is the trap, go to the DINOv2 specialist.** The gap between the two frozen probes IS the quantified ceiling.

Likelihood note: the ceiling biting is **HIGH for the two hard cells** (medium-vs-long, long-vs-very_long), because v1's supervision for exactly those distinctions was the degraded signal (medium collapsed ~5.8%, 76% long-only). The easy bins (very_short/short/absurdly_long) probably survive in features. So the realistic expectation even in the green branch is DFR recovers the easy bins and stalls on the two boundary cells — and that stall is itself diagnostic of feature absence (LaBonte NeurIPS'23: last-layer retraining only works when the core feature was learned).

## 4. THE CONCRETE RECIPE (ordered checklist, wired to the code)

The codebase has a **single optimizer param_group** and no freeze machinery, but `get_parameter_groups` already filters `p.requires_grad` (`training_utils.py:2722/2727`) and `_get_layer_wise_params` does too (`:2760`) — so DFR needs only flag flips, zero structural change.

**☐ 0. (Conditional) Fix `loss_functions.py:269` BEFORE anything, IFF you use soft/ordinal labels.** `targets_for_focal = targets` is aliased *before* smoothing, so fractional ordinal targets leak into the focal pos/neg gating at `:314-315` (a 0.3 target becomes 0.3 positive weight, not a clamped gate) — physically wrong gradients, not merely smoothed. Replace with `targets_for_focal = (targets > 0.5).to(targets.dtype)`. If you keep hard {0,1} gold labels, the existing line is already correct — skip this. (Given the ordinal scale, soft labels are the right encoding, so you will most likely need this.)

**☐ 1. Freeze, then unfreeze only the head.** Right after model creation / checkpoint load and BEFORE the optimizer is built (`train_direct.py:797`):
```python
for p in model.parameters(): p.requires_grad_(False)
for p in model.tag_head.parameters(): p.requires_grad_(True)   # the single nn.Linear, model_architecture.py:405
# optional, cheap (2*1024 affine), often helps recalibrate the CLS read-out:
for p in model.norm.parameters(): p.requires_grad_(True)
```
The existing param-group builder yields a head-only optimizer automatically. No new plumbing for pure DFR.

**☐ 2. Column-restrict to the 6 hair rows (the forgetting guard — structural, not regularizer).** The `nn.Linear` is shared across all ~19K tags; refitting the whole head drags every column. Register a backward hook on `tag_head.weight`/`tag_head.bias` that multiplies grad by a {0,1} row-mask selecting only the 6 hair-length indices. **Frozen backbone + column-mask ⇒ the other 19K tags are mathematically frozen, end to end.** This is the literal "retrain 6 of ~19K output neurons" move.

**☐ 3. Mask the loss to the 6 hair columns.** Use the existing `ignore_indices` keep-mask path in `AsymmetricFocalLoss` (`loss_functions.py:46`, machinery at `:252-263`) — pass all tag indices EXCEPT the 6 hair tags (plus PAD=0/UNK). This neutralizes the **dominant catastrophic failure**: gold labels only the 6 length tags, so without masking, every co-occurring true tag on a gold image becomes a confident negative under gamma_neg=7 (`:315`) and gets actively suppressed. Verify the ~19K-long ignore list is built once and cached, not per-step.

**☐ 4. Lower gamma_neg.** Instantiate the gold loss with `gamma_neg≈1.0` (or 0), `gamma_pos=0`. gamma_neg=7 is *literally a learned reproduction of the down-bias* — "absent labels are usually correct, suppress the gradient for predicting them" = the annotator's missing-positive convention. On a clean balanced set you want near-symmetric gradients so the model is *allowed* to raise medium/very_long. (Constructor warns >10 but accepts any non-negative, `:111-119`.) SPLC/Hill (Zhang 2021) self-correct missing positives and are the right tool for the **noisy 6M corpus** path (v2), NOT for clean gold — on gold you trust the labels, so plain balanced low-gamma ASL/BCE is correct.

**☐ 5. Balanced sampler over the 6 ordinal levels.** `WeightedRandomSampler` / class-balanced batches so `{very_short,short,medium,long,very_long,absurdly_long}` are ~equinumerous; oversample medium + the long/very_long boundary (Cui CVPR'19 effective-number or plain inverse-frequency). **This is the single most important debias knob** — DFR's robustness comes entirely from the balanced reweighting set. But note: balanced sampling WITHOUT the gamma_neg drop (step 4) and bias recal (step 7) is a **silent no-op** — the balanced batch shows the positives but gamma_neg=7 zeroes their recovery gradient.

**☐ 6. Head LR / schedule.** DFR: **one constant high LR (~1e-3)** on the head, AdamW, **WD kept at 0.05** (per project rule — never inverse-sqrt-by-dataset-size it). **Skip the cosine-restart machinery entirely** for tiny gold. Do NOT reuse the 6M schedule. 50-200 epochs of the gold set, early-stop on **held-out gold macro-F1 over the 6 levels** (not training loss). Refit 5-10× on different balanced draws and average the head weights (Kirichenko's variance reduction — 1.5-3K is high-variance). *(LP-FT only: you'd extend `get_parameter_groups` for per-group LRs — but the layer-wise path emits `lr_scale` not a literal `'lr'` at `:2781/2788`, and the scheduler/logging read `param_groups[0]['lr']` at `train_direct.py:1842,2283`; with no per-group `base_lrs` the cosine rescales the backbone group UP to the head's max_lr, silently turning LP-FT into full-FT. Avoid by using DFR + constant LR.)*

**☐ 7. Re-init the head bias from BALANCED gold priors, NOT the dirty 6M.** This is the most direct lever on the resting-threshold tilt. Either re-run `initialize_tag_head_bias` (`model_architecture.py:65`) on the **gold/balanced** class counts, or apply Menon-style additive logit adjustment `tau*log(balanced_prior)` on the 6 length logits at inference. **Do NOT re-call it with the dirty `tag_frequencies`** (`:88`) or you re-inject the exact tilt you're removing, even under a perfect sampler. Compromise: warm-start the weight rows, re-init the bias rows. (If column-restricted per step 2, only the 6 rows move anyway.)

**☐ 8. Encode ordinal implications as label propagation in the GOLD targets, not as a loss constraint.** Keep per-tag sigmoid/BCE. `absurdly_long ⇒ very_long ⇒ long`; only `{short, medium, long}` are mutually exclusive, modifiers ride along. **Non-negotiables baked in: when `very_long` fires, ADD it and KEEP `long`; never set `medium` positive→negative when a longer modifier fires (protect medium recall — bias the threshold toward medium); never softmax the 6 levels (that would force exclusivity and break the ride-along rule); never force colors exclusive (irrelevant here but a standing rule).**

## 5. EVALUATION THAT PROVES IT MOVED (and the silent-failure guards)

**The master guard: every decision metric on a held-out, group-balanced GOLD split — NEVER on the dirty 6M val.** The dirty val encodes the down-bias *as ground truth*, so a fine-tune that preserves the bias scores BETTER there than one that fixes it (Wang ICCV'23). Watching val loss/mAP inverts the signal; debiasing reads as regression. Nothing in the repo computes a balanced-gold metric today — it must be built or the run is uninterpretable.

**Primary success metrics (held-out balanced gold):**
- **Confusion-matrix off-diagonal shrinkage:** `P(predict long | true very_long)` drops; the two **medium** off-diagonal cells drop.
- **long→very_long recall RISES** and **medium recall RECOVERS** (the collapsed class).
- **Ordinal error** (mean absolute ordinal distance, off-by-one rate) decreases.
- A **stall** — improves then plateaus well short of clean — is **diagnostic of the feature ceiling (iii)**, not a tuning problem. That stall is the empirical signal to switch to DINOv2/v2, not to train DFR harder.

**The no-regression tripwire (the silent-forgetting check):** overall **mAP on a general val slice** and **per-tag F1 on the other ~19K tags** must NOT drop. Without this, a full-head refit silently rots the other columns and the dev never notices because they only look at hair tags. With frozen backbone + column-mask this is structurally guaranteed, but **run the tripwire anyway** as proof.

**Silent-failure configs to actively avoid (each looks like it worked, didn't):**
1. Evaluating on dirty val (inverted signal).
2. Balanced sampler + gamma_neg=7 unchanged (no-op on bias; loss drops, boundary cells don't move).
3. LP-FT/partial + cosine scheduler with no per-group base_lrs (silent full-FT via `lr_scale`/max_lr rescale).
4. Full-head refit without column-mask (silent 19K regression via gamma_neg=7 hard-negative channel).
5. Soft labels without the `:269` fix (physically wrong gradients).
6. Re-running `initialize_tag_head_bias` with dirty frequencies (silent bias re-injection).

## 6. THE HONEST VERDICT

**Does v1-gold-FT clean the poison GT? NO.** It changes what the *deployed model suggests*; it does not rewrite a single on-disk GT label. v1 is structurally disqualified as the *cleaner/detector* by circularity — it memorized the exact convention you want surfaced (fires long_hair on 3.12M imgs with ~0 hair-length false-negatives; it literally cannot believe the GT length is wrong). Using v1 to find v1's bad labels re-introduces the circularity the gold set exists to break (Xia NeurIPS'20 / Yao NeurIPS'21: instance-dependent noise is non-identifiable from the noisy corpus alone, and v1's features ARE that corpus). **Treating an improved gold confusion matrix as evidence the corpus is cleaner is the central distraction — it is neither cleaning nor detection.**

**Does it improve the suggestion engine? YES — capped by v1's features.** A cheap, safe, reversible, ship-this-week recalibration of the live model: better medium/very_long suggestions in the labeling UI, which *assist* a human-in-the-loop cleaning pass and reduce annotator reflexive-long clicks. Capped at v1's feature ceiling (§3).

**When each tool is right, and how they compose:**

| Tool | Right when | Role |
|---|---|---|
| **DFR-on-v1-gold** | You need a better deployed *suggestion engine* cheaply this week, AND the probe greenlights (v1 features encode the distinction) | Interim **patch on the symptom**. Cheap, safe, reversible, ships. Cannot clean GT, cannot beat the ceiling. |
| **Frozen DINOv2 + tiny ordinal head** | The goal is to **clean the corpus / surface bad labels** | The **detector**. Must be OOD precisely because v1's memorized convention makes it blind to its own bias. |
| **v2 trained fresh on repaired GT** (SPLC/Hill on the noisy residual) | You want to actually **raise the ceiling** | The **cure**. Only a from-scratch model can learn the medium/very_long features the dirty GT denied v1. |

**Pipeline order:** DINOv2 detector finds + helps clean → v2 trains fresh on clean GT. **DFR-on-gold runs in parallel as the cheap interim patch for the live model — never instead of the detector, never as the cure.**

**The single deciding condition for whether to do DFR at all:** the frozen-v1 probe must separate long↔very_long and lift medium to within a small margin of the frozen-DINOv2 control on a held-out gold split.
- **PASS** → DFR is a cheap, safe, ship-this-week patch; do it in parallel with the DINOv2 detector.
- **FAIL** (v1 collapses while DINOv2 separates) → v1-gold-FT is dead on arrival at every point on the freeze-spectrum; skip it entirely, put all effort into the DINOv2 specialist and v2.

Run the probe first. It costs minutes and is the one experiment that turns this from a judgment call into a decision. **Full fine-tuning is off the table in every branch** (maximal feature distortion per Kumar ICLR'22; severe 19K-tag forgetting amplified by gamma_neg=7; bias persists anyway per Wang ICCV'23 — it pays the most and buys the least).

**Key code anchors:** head `model_architecture.py:405`; CLS read-out `:665-669`; dirty log-prior bias `:65/:88`; requires_grad-aware param groups `training_utils.py:2722/2727/2760`; single-group LR logging/scheduler `train_direct.py:1842/2283`; optimizer build point `:797`; ASL ignore-mask path `loss_functions.py:46/:252-263`; gamma_neg gating `:308-315`; soft-label leak `:269`.