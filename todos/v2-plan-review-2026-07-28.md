# V2 Plan — Independent Soundness Review (2026-07-28)

> **STATUS (2026-07-29): point-in-time evidence snapshot — do not edit.** This doc is the evidence
> base for [v2-plan.md](v2-plan.md), which is the only doc that governs the run and wins wherever
> the two disagree (e.g. §4.7's `thresholds=None` → superseded by plan §8.3's `ap_thresholds: 200`).
> Code line refs here are stale (code has drifted up to ~586 lines; the plan carries refreshed
> refs). The three documents it reviews were retired 2026-07-29 — full text in git history — as was
> `overfitting-risk-assessment.md`, which §4.3 cites.

> **Scope.** Reviews `v2-plan-correction-2026-07-28.md`, `progressive-training-plan.md`, and
> `ASL_plan.md` (all retired 2026-07-29, git history)
> against the goal: **a noticeably improved model over V1**.
>
> **Method.** Every quantitative claim is recomputed from `vocabulary.json`, the TensorBoard event
> files, `logs/training.log`, `experiments/run1_vit/checkpoints/pr_threshold_*.json`,
> `L:\Dab\DataCleaning Project\corrections_report.json`, and the training code — or read out of the
> primary paper PDFs. Numbers stated in the plans are treated as unverified until reproduced.
>
> **Status:** complete. Loss, architecture, recipe, and evaluation all reviewed.

---

## 0. Headline

The correction doc is the most rigorous document in the set, and its *negative* claims hold: the
γ_neg descent had no valid rationale, and the "V1 lacked boundaries" diagnosis rested on a broken
metric. **Those corrections stand.**

Everything built on top of them does not. Two independent failures:

> **1. The premise is not supported by the evidence.** V1 was never shown to hit a label-noise
> ceiling. It was **stopped by hand** at 33/40 epochs (Phase 1) and 6/15 (Phase 2), with mAP rising
> monotonically at every validation event, the LR schedule 18%/58% unfinished, and **zero
> train/val generalization gap**. No early-stopping mechanism ever fired. The 0.674 figure is the
> last value logged before someone pressed stop, not an asymptote.
>
> **2. The chosen method does not transfer.** P-ASL is a *partial-annotation* method whose gain
> depends on human-verified negatives. Booru tagging has none. About half its measured gain is
> definitionally unavailable to us, its prior-estimation stage is mathematically degenerate in our
> regime, and its two ignore sets are predicted to fail in specific ways at N=∅.

Both failures point the same direction: **the plan spends its effort on the loss function, which is
the lever with the weakest evidence and the smallest published effect size, while the levers with
real evidence — finishing training, completing labels, and the prediction head — are deferred,
de-scoped, or cut.**

The plan's own budget said *~35% loss / ~55% data / ~10% data growth / ~0% capacity*. The correction
doc then removed the data workstream (§4: "do not gate V2 on the cleaning effort") and kept a loss
change worth **+0.87 mAP** in a regime that doesn't match ours. **As written, V2 has no mechanism
capable of a noticeable improvement.**

**What a noticeable improvement would actually be built from**, with the evidence attached (details
and citations in §6–§7):

| Lever | Est. Δ mAP | Status in the plan |
|---|---|---|
| Finish training (V1 got ~60% of DeiT-B's from-scratch budget for a 2.9× larger model) | **+1 to +4** | Correctly identified in §3, then contradicted by the ceiling framing |
| Pretrained init (LiGO-grow, distil, or adopt ViT-L) | **+1 to +2.5** | **Never argued — asserted in one table cell** |
| Weight EMA | +0.2 (clean-ImageNet analogue) to **+9** (measured under 40% real label noise) | **Absent from plan and codebase** |
| Run 512px instead of gating it | **+0.8 to +1.4** | Gated behind a trigger that will not fire |
| Label completion, tail-targeted | **+3.6** on rare classes (RAM++) | **De-scoped by the correction doc** |
| LayerScale at depth 18 | **+0.5 to +1.0** | Absent from plan and codebase |
| 2D RoPE across three resolution changes | +1.4 to +2.5 top-1-equivalent | Absent |
| Attention-pooling head, K≈100 | +0.7 | Deferred behind the wrong gate |
| **P-ASL selective ignore** | **+0.87, discounted for regime mismatch and a pretrained-backbone measurement** | **The centrepiece** |

The loss is the smallest, least-transferable item on this list, and it is the one the plan is about.
Note also that **most of these deltas are below the resolution of the current instrument** — which
is why §4 and §7-Tier-0 come before everything else.

---

## 1. What is sound and should not be relitigated

| Claim | Verdict |
|---|---|
| γ_neg descent 7→5 withdrawn; V1-P1 ran γ=4 | **Sound.** Checkpoint-verified; the "reverse V1" rationale was void. |
| `val/f1_macro` at a single global 0.2653 is uninterpretable | **Sound, and confirmed quantitatively** — see §4.1. |
| Selection/early-stop must move off f1_macro | **Sound — and already done.** `unified_config.yaml:412` sets `selection_metric: val_mAP`; dispatched at `train_direct.py:1331`/`:2431`, early-stop message at `:2537`. **Strike it from the launch-blocker list** (§4.2). Only `early_stopping_threshold` 5e-7 → a measured value remains. |
| Cleaning has not moved global ρ (0.65% of tags, 0.017% of label mass) | **Sound** as a statement of fact. The *conclusion* drawn from it is wrong — §7. |
| No negative-branch loss knob touches wrong-positive noise (ASL_plan §7 fence) | **Sound.** Keep the fence. |
| γ_neg = 7.0 fixed | **Right answer, wrong reason** — §3.3. |

---

## 2. The premise fails: V1 was under-trained, not noise-limited

### 2.1 It was not early-stopped — it was stopped by hand

`grep -c "Early stopping triggered" logs/training.log` → **0**. The string the code emits on early
stop (`train_direct.py:2537`) never appears. What does appear, at the end of both phases:

```
logs/training.log:14030  Soft stop requested - waiting for accumulation to finish
logs/training.log:14035  Soft stop checkpoint saved at global_step=85517
logs/training.log:14036  Soft stop engaged. Exiting training loop before validation.
```

Phase 1 ended the same way (`:13353-13359`, step 38923 of that leg). Both phases ended by
**user-initiated soft stop mid-epoch** and were never resumed. No crash, no OOM, no criterion.

`huggingface_release/README.md:186` is candid about the real reason — *"Continuing was unlikely to
buy enough real gain to justify the extra training time"* — a cost/benefit judgement call. The plan
hardened it into a measured fact.

### 2.2 Every metric was still improving at the stop

| | last-5-epoch mAP slope | value at stop | budget used | LR at stop |
|---|---|---|---|---|
| **Phase 1** (320px) | **+0.0065 / epoch** | 0.6518 | **33/40 epochs**, 81.8% of cosine cycle | **14% of peak** |
| **Phase 2** (448px) | **+0.0028 / epoch** | 0.6744 | **6/15 epochs**, 41.8% of cycle | **76% of peak** |

Val loss and mAP both improved **monotonically at every single validation event** in both phases.
The claimed signature — *"mAP growth flattened while val loss kept falling"* — does not exist in the
data; the two moved in proportion (−2.8% loss vs +2.3% mAP across P2). There is no "then" after
Phase 2 epoch 5; that is simply where the data ends.

Naive linear extrapolation over Phase 1's seven unrun epochs is **+0.041 mAP**, before any
cosine-tail bonus — and cosine schedules deliver a disproportionate share of final gain in the
anneal tail that was never run.

### 2.3 Zero generalization gap — the decisive number

From `pr_threshold_*.json` (999-point sweep, 19,292 tags), micro F1-optimal:

| eval set | composition | micro F1 |
|---|---|---|
| 296,056 images | ~90% **were moved into training** (`logs/training.log:13127`) | **0.70044** |
| 30,000 images | genuinely held out | **0.70088** |

After 39 epochs, a 248M-parameter ViT scores **identically on data it trained on and data it never
saw** — marginally *worse* on the training-seen set. It has memorized nothing.

This is the fingerprint of a model with **unused capacity and unfinished convergence**. It is
directly incompatible with the plan's foundational claim that *"the ~250M model carried unusable
headroom, so cutting to ~192M costs ~nothing."* That headroom was never tested.

### 2.4 The frequency-bucket claim is inverted in the data

Progressive-plan §2.1 asserts V1 showed a *"uniform plateau across frequency buckets; rare bucket
still climbing."* `val_bucketed/*/mAP` was logged. Phase 2 E4→E5:

| bucket | tags | mean support in 30K val | Δ mAP |
|---|---|---|---|
| 500–999 (rare) | 7,619 | **2.45** | **+0.00211** ← slowest |
| 1K–5K | 7,873 | 8.5 | +0.00191 |
| 5K–10K | 1,482 | 30.3 | +0.00239 |
| 10K+ | 2,314 | 399 | +0.00217 |
| 300–499 | **0** | — | never measured |

**No bucket plateaued** — all four move at the same +0.0019–0.0024/epoch. The specific differential
the plan claims (rare climbing while others flatten) is absent, and the rare bucket is if anything
the *slowest*. The genuinely-rare bucket contains zero tags because the vocabulary is floored at
500. The measurement existed and was mis-summarized.

### 2.5 Consequence

The plan contains its own contradiction: §3 correctly identifies *"V1's root failure: Phase 1
stopped at 33/40 epochs"* while §0/appendix attribute the same run's endpoint to label noise. The
evidence supports the first reading only.

**A label-noise ceiling has never been observed in this project.** It may well exist — the corpus
demonstrably has missing positives. But it has not been measured, and three major V2 decisions rest
on it: the 248M→192M cut, the loss-centric workstream, and the de-scoping of data work.

---

## 3. The method fails: P-ASL does not transfer to a positive-only regime

Verified against the paper PDF ([arXiv:2110.10955](https://arxiv.org/abs/2110.10955), CVPR 2022).

**The plan's factual claims about the paper are all correct** (9,600 classes; ASL-Negative 85.85 →
P-ASL Selective 86.72; the Ω_L ∪ Ω_P mechanism; γ⁺=1, γ⁻=2, γᵘ=7, K=200, η=0.05; Ignore-mode prior
estimation). The problem is the transfer argument, in four places.

### 3.1 Half the gain requires verified negatives we don't have

Table 1 decomposes as:

| transition | Δ mAP(C) | mechanism | needs verified negatives? |
|---|---|---|---|
| ASL-Neg 85.85 → P-ASL-Neg 86.28 | **+0.43** | γ⁻/γᵘ **decoupling** | **YES** |
| P-ASL-Neg 86.28 → P-ASL-Selective 86.72 | **+0.44** | Ω_L ∪ Ω_P selective ignore | No |

OpenImages V6 training data contains **37.7M human-verified negative labels — 1.9× its positives.**
The paper's stated rationale is explicit: *"the negative annotated samples are verified ground-truth
[so] we are interested in preserving their contribution… allowing us to set a lower decay rate for
the annotated negative labels: γ⁻ < γᵘ."*

With N = ∅ the γ⁻ term vanishes and **~49% of the measured gain is structurally unavailable.** The
paper also names our regime as out of scope: *"Positive Unlabeled (PU) … use only positive and
un-annotated labels without any negative annotations."* It runs no positive-only experiment.

### 3.2 The prior estimator is degenerate at N = ∅

§3.2 of the plan adopts the paper's `P̂(c)` estimated from a model trained in Ignore mode. At N = ∅,
Ignore mode's loss is `Σ_{c∈P} L_F(p_c, γ⁺)` only — whose global optimum is **p_c = 1 for every
class**. The estimator returns 1 everywhere. The paper's headline §4.2 contribution **cannot be run
as specified.**

*(The plan's proposed substitute — averaging `best_model.pt` predictions over 6.8M images — also
isn't needed. Where Ω_P is active at η=0.05, measured ρ is ~0.4%, so empirical tag frequency from
`vocabulary.json` is already within half a percent of the true prior. Delete that work item; it is
an L-sized offline job buying nothing.)*

### 3.3 γᵘ = 7 is right — for a completely different reason than the plan gives

The plan reads γᵘ=7 as "the published value for exactly our label regime." It is the value for the
branch the paper **distrusts** — set high precisely so the trusted γ⁻=2 branch does the
boundary-setting. Figure 8 varies γ⁻ with γᵘ pinned at 7 and states: *"The case of γ⁻ = 7 represents
the standard ASL. As can be seen, the mAP score increases as we lower γ⁻, up to 2."* In our regime
U **is** the branch that carries the real learning signal — so the plan imports the paper's *worst
tested configuration* for that role. The paper never ablates γᵘ at all.

**But γ=7 is independently and correctly supported** — by ASL's own Appendix F
([arXiv:2009.14119](https://arxiv.org/abs/2009.14119)): *"we set all untagged labels as negative…
Since the level of positive-negative imbalancing is significantly higher than MS-COCO, we increased
the level of loss asymmetry: For ASL, we trained with γ⁻ = 7, γ⁺ = 0."* That is assume-negative,
~5,400–9,600 classes, extreme imbalance — a structurally correct match.

**Right number, wrong citation** — and it matters, because ASL Appendix F pairs γ⁻=7 with **γ⁺ = 0**
(P-ASL uses γ⁺=1, which would down-weight our already-scarce positives) and with the ASL margin
**m = 0.05**. So the correction doc's clip 0.2 → 0.05 move lands in the right place, but its stated
justification (ignore-set-as-better-armor) is not the reason; the reason is that ASL's own
assume-negative recipe at this scale uses m=0.05.

### 3.4 Both ignore sets are predicted to misfire at N = ∅

**Ω_P (prior > η) is actively harmful.** Recomputed against `vocabulary.json`:

| η | tags selected | % of vocab | % of label mass | mean ρ̂ of selected |
|---|---|---|---|---|
| **0.05 (paper)** | **107** | **0.55%** | **42.4%** | **0.38%** |
| 0.01 | 492 | 2.6% | 66.0% | 0.63% |
| 0.002 | 1,832 | 9.5% | 83.1% | 1.08% |

At OpenImages, masking a high-prior class is safe because it still receives verified negative
gradients from N. At N = ∅ it receives **none** — so `1girl`, `solo`, `long_hair`, `highres` would
get positive gradients only and collapse to always-on. Ω_P selects the 107 tags with the *lowest*
missingness in the vocabulary (ρ̂ ≈ 0.38%) and 42.4% of the label mass. It is anti-correlated with
our actual problem and degenerate in our annotation regime.

**Ω_L (top-K predicted) is an unchecked feedback loop.** At OpenImages a verified negative sits in N
and can never enter Ω_L — N is a hard anchor against runaway confirmation. At N = ∅ every class is
in U, so Ω_L unconditionally deletes the loss on the model's own top-K predictions; confident false
positives are never corrected.

Ω_L also inherits the exact objection the plans used to **reject** Hill/SPLC — progressive-plan
§1.2: *"SPLC's self-relabel flip assumes a calibrated (pretrained) backbone and is unsafe in
from-scratch Phase 1."* Ω_L is equally prediction-dependent, and P-ASL runs on pretrained backbones
throughout. The skepticism was applied to the rejected option and not to the adopted one.

### 3.5 In the closest published match to our regime, ASL loses to plain focal loss

Hill/SPLC ([arXiv:2112.07368](https://arxiv.org/abs/2112.07368)) Table V — **OpenImages
single-label: 567 classes, 1.74M images, exactly one positive per image, no verified negatives**:

| BCE | Focal | **ASL** | Hill | SPLC |
|---|---|---|---|---|
| 60.83 | 62.14 | **61.95** | **62.71** | **62.86** |

ASL is *beaten by plain focal loss* when negatives are genuinely absent, and by Hill/SPLC by
+0.76/+0.91. Their stated reason: *"ASL⁻ still puts too much focus on these possibly false
negatives."* This is the best regime match in the literature and it argues against the plan's
sole-loss decision. Hill's negative re-weighting is also structurally safer than Ω_L — it
*down-weights* probable false negatives continuously rather than *zeroing* them, so it has no
confirmation-bias loop, needs no prior, and needs no change_epoch.

### 3.6 Two levers with better evidence, both currently deferred

- **ML-Decoder** ([arXiv:2111.12933](https://arxiv.org/abs/2111.12933), WACV 2023), **at 9,600
  OpenImages classes**: GAP head 86.0 → **86.8**. A head swap alone delivers P-ASL's *entire*
  loss-side gain, at our exact class count, composable with any loss, O(N) in class count.
  Progressive-plan §2 mentions it and defers it as *"not needed at the 896w default."*
- **RAM++** ([arXiv:2310.15200](https://arxiv.org/abs/2310.15200)) data-engine ablation: scaling
  tags 12.0M → 41.7M moves OpenImages-**rare** 63.54 → **67.17 (+3.6)**. The strongest published
  large-vocabulary tagger attacks missing labels with a **data engine, not a loss**, for 1.4–4× the
  gain of any loss change here. This directly validates the cleaning/label-completion workstream
  that correction-doc §4 de-scoped.

---

## 4. The instrument is broken more deeply than §3.4 diagnoses

§3.4 is right that the instrument is the launch blocker. It identifies the wrong defects.

### 4.1 The f1_macro artifact is confirmed — and larger than stated

The in-training metric is hard-coded at θ = 0.2653. The measured F1-optimal threshold is
**θ ≈ 0.76–0.81**. Properly thresholded, V1-P2 posts **micro F1 0.7004** and **per-tag macro F1
0.6751** — against the ~0.013 the training loop was reporting. E1's conclusion is fully vindicated,
and it means the manual stop decision (§2.1) was made while watching a number with no relationship
to model quality.

Also measured: P1-e27 → P2-e7 moved per-tag macro F1 **0.6218 → 0.6751 (+8.6% relative)** on an
identical eval set. A model at a noise ceiling does not gain 8.6% macro F1.

### 4.2 Enabling `threshold_calibration: per_tag` is a no-op for its stated purpose

The calibrator exists and works (`evaluation_metrics.py:601-622`), but its output goes to
`thresholds.json` and TensorBoard only (`train_direct.py:2319-2342`). It is never fed back into the
metric that drives selection, which stays bound to the fixed global threshold at
`train_direct.py:1308`. **Flipping the flag leaves the failure fully intact while appearing fixed.**

The real fix is ~20 lines: `ThresholdCalibrator._compute_f1_grid` already returns a
`(num_thresholds, C)` grid; `grid.max(axis=0).mean()` over supported columns *is* per-tag-optimal
macro-F1. (Also gate the 19,292-entry log line and TB write that `per_tag` mode would emit
per epoch.)

### 4.3 `max_val_samples: 30000` cannot support three of the four launch-blocker items

| val size | tags <5 positives | tags <10 | decile-10 median positives | decile-10 mAP 95% CI |
|---|---|---|---|---|
| **30,000 (current)** | **44.5%** | **65.8%** | **2.4** | **±0.0125** |
| 100,000 | 0% | 0% | 8.9 | — |
| 276,000 (full 5% split) | 0% | 0% | 21.8 | ±0.0045 |

Aggregate macro-mAP survives (simulated 95% CI ±0.0027 — adequate to detect a 1-point move).
**Everything below aggregate does not.** Tail-decile mAP carries a ±1.25-point CI, so no tail
improvement smaller than that is detectable — and the tail is where the plan locates the problem.
Confirmed against the real artifact: on the 30K set only **10,406 of 19,292 tags** have support ≥5,
and 958 have zero positives.

Per-tag thresholds and per-tag isotonic calibration are not estimable for ~2/3 of the vocabulary at
2–10 positives. `_calibrate_per_tag` has **no support floor**.

Flagged in `overfitting-risk-assessment.md` Risk Factor 4 (April 2026; retired 2026-07-29, git
history) and never fixed. Raising the cap costs validation time only.

### 4.4 Val does triple duty, and 266K val images were moved into training

`logs/training.log:13127`: *"Validation limited to 30,000 samples at split time (was 296,056, moved
266,056 to training)."* Consequences:

- The genuine held-out set is **30,000 images = 0.51% of corpus**.
- The `pr_threshold_*_full296k` artifacts evaluate 296K images of which ~90% were trained on. Those
  numbers must not be quoted as held-out performance. *(They happen to match the clean 30K number —
  which is itself the §2.3 finding.)*
- No held-out **test** set exists at all. Val simultaneously drives early stopping, threshold
  calibration, and reporting (also flagged April 2026, Risk Factor 3). Fitting per-tag thresholds on
  the same draw used for selection makes every reported number optimistically biased.

### 4.5 The split is not group-aware

`dataset_loader.py:2664-2678` — plain uniform random 95/5 over individual JSON sidecars, `seed=42`.
No perceptual-hash, artist, series, or post-id grouping. `logs/dedup_hashes/dedup_clusters.json`
measured a 0.26% near-duplicate cluster rate over 5.54M images — but the corpus at split time was
5.92M, so **~382K images were never dedup-scanned**, and whether the 14,230 flagged deletions were
applied is not verifiable from the artifacts. Expected leakage is order 10²–10³ of the 30K val set:
a real hygiene defect worth fixing with group-aware splitting, **not** large enough to explain
0.674.

### 4.6 The break the plan misses entirely: evaluation bias points *against* V2

This is larger than the threshold problem and cannot be fixed by thresholds, mAP, deciles, or
calibration.

For tag `c` with missing-positive rate ρ_c, the val set contains ρ_c·π_c·N images that are truly
positive but labelled negative. A model that has learned the concept ranks those images **high**, and
in AP a false positive at rank 1 costs far more than one at rank 500. Therefore:

> **Measured AP is biased downward, and the bias grows with the model's true quality.**

This is a model-dependent systematic bias, not variance that averaging removes. And our measured ρ
makes it worst exactly where we most want to measure: 0.3–0.5% at head, ~5% median at the tail
decile, **23–64% on individual confusable tags**. The deciles the plan wants to report are the
deciles whose measurement is most corrupted — and corrupted in the direction of *hiding*
improvement.

Northcutt et al. ([arXiv:2103.14749](https://arxiv.org/abs/2103.14749)) show this **flips model
rankings** on real benchmarks at only 3.4% average label error, with the specific conclusion that
*"lower capacity models may be practically more useful than higher capacity models in real-world
datasets with high proportions of erroneously labeled data."*

**The sharp consequence for the plan's own method:** selective ignore works by removing the gradient
that trains the model to suppress high-scoring un-annotated labels. Its *intended* effect is that V2
fires on more unlabelled true positives than V1. On a validation set that scores those exact firings
as false positives, **the intended improvement registers as a loss.** Three concrete failure modes:

1. **V2 can be genuinely better and measure worse**, especially in the tail deciles.
2. **The §3.2 K/η calibration pass, scored on noisy val, will systematically select K too small** —
   tuning the ignore set back toward the ASL-Negative baseline it was adopted to beat.
3. Early stopping on noisy val will peak and decline as the model begins exceeding the annotation —
   not from overfitting. *(Note: this did **not** cause V1's stop, which was manual — §2.1.)*

**Two mitigations, one cheap and one unavoidable:**

- **Cheap — approximate the Open Images protocol.** OpenImages evaluation ignores unannotated
  classes entirely and does not penalize false positives on them. We can't apply that directly
  (positive-only annotation ⇒ the ignore set would be everything), but we *can* restrict evaluation
  to tag×image cells where we have **evidence of negativity**: tags in a mutually-exclusive group
  where a sibling is positively labelled. For hair colour/length and the other confusable groups —
  precisely where ρ is 23–64% and where review budget is being spent — a sibling-positive label
  *is* reliable evidence of negativity. Low-noise evaluation on the hardest tags at zero annotation
  cost.
- **Unavoidable — a bias-controlled slice built by pooled stratified adjudication.** Select ~200–300
  tags stratified across deciles, oversampling confusables; for each, pool **V1's and V2's top-N
  scored images plus a random corpus sample with recorded inclusion probabilities**; adjudicate
  double-reviewed with measured κ; estimate per-tag AP with the **statAP / infAP** estimators
  (Yilmaz & Aslam SIGIR 2008; Aslam & Yilmaz CIKM 2006), which are unbiased under known inclusion
  probabilities and come with variance estimates. **Pooling both systems is the critical design
  choice** — it removes the better-model-penalized-more bias for the compared pair. This is standard
  TREC methodology and it is what Schultheis et al. (KDD 2022 §4.1) explicitly recommend.

### 4.7 Three further instrument defects

- **`val/mAP` is computed with the binned estimator.** `train_direct.py:1356`:
  `MultilabelAveragePrecision(..., thresholds=200)` — an integer selects torchmetrics' *"binned
  version that is less accurate but more memory efficient"*, on 200 uniform bins over [0,1]. Since
  a γ_neg change shifts where score mass sits relative to that fixed grid, the estimator's bias can
  move between two measurements being compared. **Set `thresholds=None`.** *(The measured
  F1-optimal thresholds are ~0.76–0.81, so score mass is not crammed into the bottom bins and the
  practical error is likely modest — but this is a one-word fix on the metric that now drives
  checkpoint selection.)*
- **The macro average floats.** `train_direct.py:2240`: `keep_classes = (val_pos_counts > 0)`. On the
  30K draw ~958 tags have zero positives and are silently dropped. Freeze the tag list and log it
  with the run, or cross-run comparisons average over different denominators.
- **Per-tag isotonic calibration (ASL_plan §5) is not viable and should be replaced.**
  Niculescu-Mizil & Caruana (ICML 2005): isotonic **overfits below ~2,000 calibration points**;
  Platt wins there. Our median tag has ~6. Ullah et al.
  ([arXiv:2411.04276](https://arxiv.org/abs/2411.04276)) — the only calibration study at XMC scale —
  use a **global** calibrator and state per-label calibration at this scale is *future work*, i.e.
  an open problem. They also show marginal per-label ECE is misleading (**ECE 0.05 vs ECE@5 =
  9.25** on the same model) and that *"marginal calibration… does not imply top-k calibration."*
  **Replacement:** global isotonic on top-k scores (their validated recipe: ECE@1 17.02% → 0.17%
  with no accuracy loss), plus per-decile isotonic (~1,900 tags each = ample support), and per-tag
  **Platt** only for the ~1,230 tags with ≥100 calibration positives, shrunk toward the decile fit.

---

## 5. Errata in the correction doc's own measurement section

**(a) The 4.9:1 missing:wrong ratio is a review-budget artifact.** Recomputed from
`corrections_report.json`:

| pool | size | converted | mining rate |
|---|---|---|---|
| FP candidates (→ missing positives) | **24,765,099** | 38,463 adds | **0.155%** |
| FN candidates (→ wrong positives) | **11,279** | 7,852 removes | **69.6%** |

Mined at rates differing by ~450×. The ratio measures how review budget was spent, not noise
composition. The *direction* (missing ≫ wrong) is almost certainly right given the 2,200× pool-size
difference, but §4's recommendation leans on "wrong positives are only 1/4.9 of errors by count."
Note also that the FN pool is model-defined, so wrong positives the model has successfully
*memorized* are structurally invisible to it.

**(b) The `gt_current = 0` claim is wrong.** §0.2 states `exercise`, `gibson_les_paul`,
`beam_saber`, `white_wristband` are *"concepts present in the corpus with zero labels anywhere."*
In the training vocabulary they have **559 / 1,410 / 998 / 705** occurrences. The zero is in the
DataCleaning Project's separate GT store, which disagrees with `vocabulary.json` by a median factor
of 1.115× (range 0.73–1.77×). The report's own `training_count_oppai` column matches
`vocabulary.json` exactly for all 126 tags — that is the correct denominator, and §0.2 did not use
it. Recomputed per-bucket **median** ρ is **0.47 / 1.59 / 3.30 / 5.11%**, not the aggregate
0.32 / 3.73 / 4.51 / 12.10% quoted.

**(c) Per-tag missingness IS estimable**, contra ASL_plan §7 (*"they require known per-tag
missingness rates we don't have"*). Fitting the 123 usable campaign tags against their training
counts:

```
ρ̂(c) ≈ 0.50 × freq(c)^−0.365       corr = −0.52,  R² = 0.27,  n = 123
```

→ 0.19% at 4M, 0.75% at 100K, 1.73% at 10K, 4.02% at 1K, 5.17% at 500. A lower bound with a loose
fit, but structurally the same object Jain et al. (KDD 2016) use — enough to drive a
frequency-conditioned ignore set or negative-branch weight. If an ignore set is wanted, select on
**ρ̂, not prior**: ρ̂ ≥ 5% covers 1,330 tags (6.9% of vocab) but only **0.3% of label mass** —
precisely targeted and nearly free, the exact inverse of Ω_P's profile.

**(d) Corpus size is a lower bound of unknown tightness.** E2 infers N ≥ 6.77M from `highres`. If
`highres` covers ~70% rather than ~100% of the corpus, N ≈ 9.7M and the positive rate falls to
~0.12%. The direction reinforces the plan's argument, but no independent corpus count exists in the
repo.

**(e) Two code footguns the plan doesn't flag.** `asl_telemetry.py:106-115` makes a checkpoint's
persisted γ_neg **win over YAML** (warning only) — under "γ_neg fixed at 7.0" any checkpoint that
persisted a different γ silently overrides config on resume. And `train_direct.py:1109` does
`max(best_metric, loaded_best)`; existing checkpoints hold F1-scale values (0.013–0.045) while mAP
is ~0.67, so switching the selection metric without resetting `best_metric` either marks every epoch
best or never does.

**(f) The V2 architecture change has not been applied.** `configs/unified_config.yaml` still holds
`hidden_size: 1024`, `num_attention_heads: 16`, `intermediate_size: 4096` — the V1 Phase-2 config
with only the loss and comment blocks rewritten.

---

## 6. Architecture and recipe — the strongest part of the plan, and the most under-claimed

The architecture half survives scrutiny better than the plan itself claims. Verified sound:

- **896×18 is not a non-standard shape — it is a published compute-optimal one.** SoViT-150m/14
  ([arXiv:2305.13035](https://arxiv.org/abs/2305.13035)) is width **880**, depth **18**. The plan
  matches to within 2%. Aspect ratio 49.8 sits between ViT-L (42.7) and ViT-B (64).
- **Generalized Neural Collapse ([arXiv:2310.05351](https://arxiv.org/abs/2310.05351)) genuinely
  covers K ≫ d.** The width-896-for-19K-classes argument stands.
- **patch16 is right, and the plan's central claim is provable.** Attention is a *minority* of
  compute at every planned resolution (7.5% at 320, 14.6% at 448, 19.1% at 512), so patch14@448 is
  **1.36× FLOPs** (the plan's original figure was right; the revised "1.4–1.6×" conflates the
  activation-memory term). 512px/patch16 gives N=1025 — **identical token cost to 448px/patch14** —
  while delivering real pixels instead of a finer mesh over the same downscaled raster.
- **The 320→448 schedule has direct in-regime precedent the plan doesn't cite**: ASL itself trains
  Open Images (~5,400 classes, partial labels) at 224 for 30 epochs then fine-tunes at 448.
- **Phase-2 LR is not too low.** DeiT III's high-res fine-tune is AdamW 1e-5 @ batch 512 × 20 epochs;
  the plan's 2.6e-5 @ 768 × ~106K steps is higher in both. Rule this hypothesis out explicitly —
  V1's 1e-5 was also inside DeiT III's range, so P2 LR did **not** cap V1 at 0.672.

### 6.1 What the plan gets wrong here

- **Param count vs data is not a constraint, so the cut has no justification.** DeiT III trains a
  **304M** ViT-L from scratch on **1.28M** ImageNet images to 84.9%. The plan proposes 192M on
  6.8M — an order of magnitude more headroom. Combined with §2.3's zero generalization gap, **the
  248M → 192M cut is unmotivated in both directions**: nothing showed the capacity was unusable, and
  nothing suggests 248M was too large for the data.
- **The MLP ratio contradicts its own citation.** SoViT-150m uses MLP **2320** (ratio 2.64), not
  4×896 = 3584; SoViT's fitted exponents put s_MLP ≈ 0.60 > s_depth ≈ 0.45 > s_width ≈ 0.22, i.e.
  MLP should be *relatively narrow* at this scale. Following the paper exactly gives **~150M total,
  −22% params and compute** for the shape actually validated. If the goal is right-sizing, this is
  the principled version of it.
- **Drop the plain dropout; keep drop_path at 0.25.** DeiT III and big_vision both use **stochastic
  depth only, no plain dropout**, and Steiner et al.
  ([arXiv:2106.10270](https://arxiv.org/abs/2106.10270)) find *"when using the 10× larger
  ImageNet-21k dataset and keeping compute fixed, any kind of AugReg hurts performance for all but
  the largest models"* — at 6.8M images we are in that regime. So `hidden_dropout_prob: 0.10` /
  `attention_dropout: 0.05` should go to 0.
  **But drop_path 0.25 is correct and should NOT be lowered.** `model_architecture.py:398` uses
  `torch.linspace(0.0, rate, num_hidden_layers)` — the **timm linear-ramp convention**, so 0.25
  means a *mean* rate of 0.125 across 18 blocks, not 0.25 everywhere. DeiT III's table is in the
  same convention (ViT-B 0.1, ViT-L 0.4), so 0.125 mean for a 192M model sits correctly between
  them. *(An earlier draft of this review recommended cutting it; that was based on reading 0.25 as
  a uniform rate. Verify the convention before touching this number.)*
  Also resolve the WD conflict: plan says 0.08/0.04, live config and the standing project decision
  say **0.05 fixed**. And note DeiT III's explicit rule — **if you extend training you must raise WD
  and drop_path with it** (+0.05 drop_path per 200 epochs); the 800-epoch gains are not reachable at
  the 400-epoch regularization setting.
- **Head expressivity is definitively not a bottleneck.** *Taming the Sigmoid Bottleneck* (AAAI
  2024, [arXiv:2310.10443](https://arxiv.org/abs/2310.10443)) Theorem 4: all *k*-active label
  assignments are argmaxable given **2k+1** dimensions. At k ≈ 36 that is **73 ≪ 896**. Their
  empirical failures at n ≈ 9,000 appear only at d ≤ 200. This closes the §2 "separability" worry
  from the other direction and removes it as a reason to want a decoder head.
- **Two miscitations.** `arXiv:2501.02364` is about *intrinsic data dimension*, not class count —
  it does not support "required width grows sub-linearly with class count." `arXiv:2601.20994`
  ("Depth Delusion") is a **language-model** study whose exponents contradict SoViT's vision ones;
  cite it as cross-modal tension, not ViT evidence. Also `unified_config.yaml:256-257` still labels
  the 2-epoch re-warmup a "FixRes recommendation" — it is generic hygiene, same class of error the
  plan has been cleaning up elsewhere.
- **FixRes doesn't describe this pipeline at all.** FixRes's discrepancy is caused by
  RandomResizedCrop; this pipeline is downscale-only letterbox
  (`dataset_loader.py:282-306`), so there is no apparent-size discrepancy to fix, and
  train-res = test-res = 448 is correct.

### 6.2 Levers the plan omits entirely

| Lever | Evidence | Est. Δ | Cost |
|---|---|---|---|
| **Run 512px unconditionally** (don't gate Phase 3) | ASL COCO 448→640 **+1.4 mAP**; ML-Decoder **+1.1**; Query2Label **+1.1** — unusually consistent across three multi-label papers | **+0.8 to +1.4** | compute |
| **2D RoPE** instead of interpolated learned pos-embeds | RoPE-ViT (ECCV 2024, [arXiv:2403.13298](https://arxiv.org/abs/2403.13298)): +1.4 @384, +2.5 @512 for ViT-L, at **0.01% of FLOPs** | large, for a plan whose spine is 320→448→512 | small code |
| **Weight EMA** | **The payoff is bimodal and the upside is large.** On clean ImageNet EMA is worth +0.1–0.2 (DeiT-S 80.0→80.2, DeiT-B 81.0→81.1 — which is why DeiT III, big_vision and ResNet-strikes-back all *drop* it). Under **40% real human label noise** it is worth **+9 points** (CIFAR-100N ResNet-34 55.50 → **65.15**; CIFAR-10N 77.69 → **86.71**) — TMLR 2024 ([arXiv:2411.18704](https://arxiv.org/abs/2411.18704)). It also cuts prediction churn 18.84→11.69 and post-temperature ECE 4.67→3.13, which feeds the threshold-calibration problem directly. Verified absent from the codebase | **+0.2 to +2** | trivial — one weight copy; keep both checkpoints, so downside is zero |
| **LayerScale (ε = 0.1)** | CaiT ([arXiv:2103.17239](https://arxiv.org/abs/2103.17239)) Table 1 measures **+1.0 top-1 at exactly depth 18** (80.7 → 81.7) against a *drop-path-tuned* baseline. DeiT III uses it in **every** configuration. Verified absent from the codebase — the only `layer_scale` hits are dead LLRD code | **+0.5 to +1.0** — best-evidenced item at our exact depth | one learned diagonal per residual branch |
| **fp32 optimizer state for the 19,294-way head + pos-embeds** | 8-bit AdamW is *not* verified lossless for ViT-from-scratch: [arXiv:2309.01507](https://arxiv.org/abs/2309.01507) Table 2 measures **Swin-T IN-1k from scratch, 8-bit AdamW 81.0 vs fp32 81.2 (−0.2, exceeding seed std)**. Dettmers ([arXiv:2110.02861](https://arxiv.org/abs/2110.02861)) contains **no ViT and no 8-bit-*Adam* vision result** — its ImageNet row is 8-bit *momentum-SGD*. A 19K head at 0.18% positive rate is structurally the sparse-embedding shape that motivated Dettmers' Stable Embedding Layer | +0.1 to +0.2, plus tail-risk insurance | ~140 MB via `GlobalOptimManager` |
| **Longer warmup (→ ~10K steps ≈ 1.5 epochs)** | Beyer et al. use 10K warmup steps at exactly batch 1024; Wortsman Fig. 5 — longer warmup flattens LR sensitivity; σReparam's grid diverges on 7/8 configs at ViT-B, batch 1024–2048, LR 5e-4–1e-3 — our exact box | +0 to +0.3, plus variance reduction | free |
| **QK-norm** | Wortsman ICLR 2024 ([arXiv:2309.14322](https://arxiv.org/abs/2309.14322)); ViT-22B. Flattens LR sensitivity across orders of magnitude | 0 direct; removes a run-ending risk on a multi-week run | trivial |
| **Attention-only fine-tune at the resolution switch** | *Three things everyone should know about ViT* ([arXiv:2203.09795](https://arxiv.org/abs/2203.09795)): MHSA-only fine-tune lands within ±0.1 of full, −10% memory/time — and fewer trainable params means **slower false-negative memorization** | ~0 quality, real cost saving | small |
| **Aspect-bucketed batching** | NaViT ([arXiv:2307.06304](https://arxiv.org/abs/2307.06304)). Square letterboxing typical portrait booru art spends ~25–35% of 784 tokens on gray bars | 0 to +0.3, ~25% throughput | medium |

### 6.3 Pretraining — the plan's single largest unexamined decision

The plan asserts "from scratch" in the §0 TL;DR and never argues for it.

*For from-scratch:* the domain gap is real (flat shading, line art, no photographic texture
statistics — Geirhos et al. photographic texture priors are largely irrelevant); Kornblith et al.
([arXiv:1805.08974](https://arxiv.org/abs/1805.08974)) find fine-tuning gives no substantial benefit
on fine-grained targets; He/Girshick/Dollár ([arXiv:1811.08883](https://arxiv.org/abs/1811.08883))
find ImageNet pretraining speeds convergence but doesn't raise final accuracy given enough data; and
no public checkpoint exists at 896×18/patch16.

*Against:* Steiner et al. Table 3 — ViT-L/16 IN-21k-pretrained **87.08%** vs from-scratch **74.01%**;
even against DeiT III's much better from-scratch recipe (84.9%), 21k pretraining still adds
**+2.1pp**. ASL reports that swapping IN-1k for IN-21k pretraining raises **multi-label mAP by
"almost 2%"** — same task family, same metric, same loss family. Illustration2Vec (SIGGRAPH Asia
2015), the closest in-domain scholarly precedent, is ImageNet-pretrained VGG-16 fine-tuned.

**And the sharpest point: P-ASL's +0.87 was measured on an ImageNet-21k-pretrained TResNet.** By the
plan's own out-of-regime discipline, that number must be discounted for a from-scratch run — and the
discount is unestimated. The same applies to the Ω_L top-K mechanism, which needs a calibrated model
to rank (§3.4).

**Cost context:** Phase 1 alone is ≈1.2e20 FLOPs (~2/3 of the total budget), multiple weeks on one
GPU. Pretraining is the only lever that buys a large fraction of that back.

Four published routes that reconcile pretraining with the non-standard shape: **LiGO** growth
operators (ICLR 2023, [arXiv:2303.00980](https://arxiv.org/abs/2303.00980) — grow ViT-B 768×12 into
896×18, "55% savings in FLOPs with no performance drop"); **patient distillation** from a public
ViT-L (Beyer et al., [arXiv:2106.05237](https://arxiv.org/abs/2106.05237) — preserves the shape and
the ONNX path exactly); **adopt ViT-L/16 outright** and init from DeiT III / SigLIP / DINOv2; or
**in-domain MAE/DINO SSL** on the same 6.8M images.

**This decision is worth more than every architectural lever in §6 combined and it is currently
undocumented. It should be argued, not assumed.** The cheap falsifiable test — a DINOv2-L / SigLIP
linear probe on a 100K slice against the 19K vocabulary, ~1 GPU-day — has never been run, and no
scholarly evaluation of foundation-model features on booru-style illustration appears to exist.

---

## 7. Recommendations

### Tier 0 — Settle the premise before committing anything (blocking, cheap)

1. **Resume V1 from `experiments/run1_vit/checkpoints/last.pt` and run Phase 2 to its planned E14.**
   ~9 epochs. This is the cheapest decisive experiment available and it settles the question three
   major V2 decisions rest on. If mAP keeps climbing past ~0.69, the ceiling premise is dead and the
   192M cut, the loss-first workstream, and the de-scoping of cleaning all need rewriting. If it
   genuinely flattens, you will have — for the first time — **measured** the thing the plan assumes.

2. **Fix the validation set** (§4.3–§4.5). Raise `max_val_samples` from 30,000 to the full ~276K
   split (or ≥100K if validation time bites); stop folding 266K val images into training; carve a
   separate held-out **test** set so val isn't simultaneously doing selection, calibration, and
   reporting; make the split **group-aware** on perceptual-hash cluster ID.

3. **Fix the selection metric properly, not as §3.4 specifies it** (§4.2, §4.7). Note first that
   **"move selection to val/mAP" is already done** (`unified_config.yaml:412`) — strike it from the
   blockers. What actually remains: set `MultilabelAveragePrecision(thresholds=None)`
   (`train_direct.py:1356`); **freeze the macro-average tag list** (`:2240` currently floats);
   compute per-tag-optimal macro-F1 *inside* the val loop (~20 lines via `_compute_f1_grid`);
   set `early_stopping_threshold` to ~1 measured bootstrap SE rather than a guessed 1e-3; assert
   config γ_neg over the checkpoint's persisted value (`asl_telemetry.py:106-115`).

4. **Adopt xCOLUMNs for thresholding.** Schultheis et al.
   ([arXiv:2401.16594](https://arxiv.org/abs/2401.16594), ICLR 2024;
   [arXiv:2311.05081](https://arxiv.org/abs/2311.05081), NeurIPS 2023;
   [library](https://github.com/mwydmuch/xCOLUMNs)) give **statistically consistent, post-hoc,
   retraining-free** algorithms for optimizing macro-F1/macro-recall at a per-instance budget k,
   built for extreme label spaces. Two independent literature sweeps converged on this as the
   answer to the instrument problem, and it is the only candidate whose validation scale matches
   ours. It drops onto a 19,294-way sigmoid head with no retraining and no verified negatives.
   Their NeurIPS'23 Table 1 also gives a **free sanity test of the metric panel**: score a
   head-only-capable model with every metric you plan to use — P@5 drops only 8.7% and PSP@5 18.4%
   under total tail collapse, while macro-F1@5 drops 88.5%. Any metric that doesn't move cannot
   detect tail failure.

5. **Build the bias-controlled evaluation slice** (§4.6). This is the only thing that can answer
   "is V2 actually better," and it is weeks of work, so start it now in parallel with everything
   else. Do the free sibling-negative approximation immediately.

### Tier 1 — The levers that actually move mAP

6. **Do not cut to 192M on the current rationale.** The premise (unusable headroom) is refuted by
   the zero generalization gap. If you still want a smaller/faster model, take **SoViT-150m's real
   shape** (MLP ratio ~2.64 → ~150M) rather than ratio-4 at 192M — same citation, −22% compute.

7. **Argue the pretraining decision explicitly** (§6.3). Worth **+1 to +2.5 mAP** and a large
   fraction of a multi-week Phase 1. Run the DINOv2/SigLIP linear probe first — one GPU-day to
   replace an assumption with a measurement.

8. **Run 512px unconditionally**; delete the fine-bucket-plateau gate. Three multi-label papers
   agree on ~+1 mAP, and the gate will under-trigger precisely because tail-metric noise masks the
   benefit it's supposed to detect.

9. **Finish training.** V1 received ~230M sample presentations for a 248M-param from-scratch ViT —
   about 60% of what DeiT-B (2.9× smaller) uses. Keep the plan's "run to a genuine plateau, not a
   fixed epoch" gate; it is correct and it is exactly what V1 failed to do.

10. **Reinstate label completion — but be precise about which data lever.** RAM++
   ([arXiv:2310.15200](https://arxiv.org/abs/2310.15200)) scales **tags** 12.0M → 41.7M for
   **OpenImages-rare 63.54 → 67.17 (+3.6)**. That is *label completion*, not more images — and the
   distinction matters, because Zhai et al. measured that going **JFT-300M → JFT-3B (10× more
   noisy images) buys only ~1%**, equally for small and large models. So: completing labels on the
   corpus you have is the high-value lever; adding more equally-noisy images is not. Correction-doc
   §4's *facts* are right (cleaning has touched 0.017% of label mass) but the *conclusion* is
   backwards — that says the campaign is under-resourced and mis-targeted, not that label completion
   is the wrong lever. Retarget it per §4 of the correction (tail and confusable groups, not head).

### Tier 2 — Near-free adds (§6.2), in cost order

11. **Longer warmup** (→ ~10K steps ≈ 1.5 epochs) — free.
12. **Weight EMA** with horizon ≈ 1% of total steps (α ≈ 0.9998, *not* the copy-pasted 0.999 —
    [arXiv:2502.06761](https://arxiv.org/abs/2502.06761)); evaluate online and EMA weights every
    validation so the downside is exactly zero. This is the highest expected value in the tier: the
    payoff is +0.1–0.2 if this run behaves like clean ImageNet and **+9 points** if late-training
    noise memorization is active — and the plan's own thesis is that it is.
13. **LayerScale, ε = 0.1** — +1.0 measured at exactly depth 18, absent from the codebase.
14. **Plain dropout → 0** (keep drop_path 0.25 — it's linear-ramp, mean 0.125; §6.1).
15. **QK-norm** — only if you intend to actually raise LR. Free diagnostic first: **log max
    attention logits**; Wortsman's threshold is *"all points with attention logits above 1e4
    diverged."* One epoch converts this from speculation to measurement.
16. **2D RoPE** — small code, and the plan's spine is three resolution changes.
17. **fp32 optimizer state for the 19K head and pos-embeds** (~140 MB) — 8-bit AdamW is not verified
    lossless for ViT-from-scratch (§6.2).

**Do not** add SWA on top of EMA (near-substitutes, measured within noise), model soups (needs a
shared pretrained init), or LAMB (its entire measured advantage is at batch ≥16K; batch 1024 sits at
the *bottom* of ImageNet's critical-batch range of 1,000–15,000, and Beyer et al. measure batch
1024→4096 *dropping* accuracy 76.5→74.7).

### Tier 3 — The loss (i.e. most of what the current plan is about)

18. **Keep γ_neg = 7.0 fixed, γ_pos = 0, clip = 0.05 — but cite ASL Appendix F, not P-ASL.** ASL's
    own assume-negative recipe at 5,400–9,600 classes is γ⁻=7, γ⁺=0, m=0.05. Same numbers, correct
    regime, one stage, no degenerate prior step. This makes the clip→0.05 move well-founded and
    deletes the 6.8M-image prior-estimation job entirely.
    **Three independent corroborations for this exact configuration:** ASL Appendix F itself;
    **RAM** (CVPRW 2024, [arXiv:2306.03514](https://arxiv.org/abs/2306.03514)) tags **6,449
    categories** from image-text-derived incomplete labels using **plain ASL at γ⁻=7, γ⁺=0,
    clip=0.05** — bit-for-bit this config, and the closest true regime match published; and
    **ML-Decoder's** OpenImages loss table, **CE 84.8 · Focal 84.9 · ASL 86.3**. Additionally
    **SINR** (ICML 2023, [arXiv:2306.02564](https://arxiv.org/abs/2306.02564)) — the largest
    positive-only multi-label study in existence at **47,000 classes** with no verified absences —
    finds full assume-negative is the workhorse and performance "scales gracefully." That is the
    best available evidence about how these losses behave as C grows from 80 to 47,000.

19. **Drop Ω_P.** It is degenerate at N=∅ (§3.4) — head tags would receive no negative gradient from
    any source and collapse to always-on.

20. **Drop or heavily gate Ω_L.** If kept, it needs the same `change_epoch` delay the plans used to
    reject SPLC, and K must be calibrated, not copied — and per §4.6, **calibrating K on noisy val
    will drive it too small**, so calibrate on the gold slice. Note zeroing is strictly more
    dangerous than down-weighting when there is no verified-negative anchor.

21. **Reinstate Hill/SPLC as a live candidate, not a deferred fallback.** In the closest published
    regime match (OpenImages single-label, 567 classes, no verified negatives) ASL is beaten by
    plain focal loss and loses to Hill/SPLC by +0.76/+0.91. Hill's continuous down-weighting of
    probable false negatives does Ω_L's job without the confirmation-bias loop, without a prior, and
    without a schedule.

22. **If an ignore set is still wanted, select on ρ̂ rather than prior** (§5c). ρ̂ ≥ 5% covers 1,330
    tags — 6.9% of the vocabulary but only **0.3% of label mass** — precisely targeted and nearly
    free, the exact inverse of Ω_P's profile.

*Dissent recorded:* a second literature sweep, scoped to scale rather than annotation regime,
concluded P-ASL should stay primary on the grounds that nothing published 2023–2026 beats it above
1,203 classes. That is true, but it is an argument from absence — nobody else evaluates at that
scale, and P-ASL's evaluation is on a dataset with 37.7M verified negatives. The structural argument
in §3 (half the gain needs those negatives; the prior estimator is degenerate without them) is not
addressed by the scale argument, and RAM's 6,449-tag plain-ASL result is the closer regime match.

### Tier 4 — Structural (one ablation, not a commitment)

23. **Attention-pooling head — but expect ~+0.7, not a step change, and run it last.** The right
    framing is not "linear vs decoder" but **average/CLS pooling vs learned attention pooling**, and
    the evidence is more modest than it first looks:
    - ML-Decoder's gain **does not grow with class count**: 80 classes +1.1, 1,000 +0.6/+0.8,
      **9,600 +0.7/+0.8**. And K=100 → 200 → 400 at 9,600 classes moves 86.7 → 86.8 → 86.8, so
      quadrupling query resolution buys +0.1. Queries carry no class semantics (learnable, fixed
      random, and word-based queries all score identically).
    - **CaiT Table 2, from scratch at matched capacity: attention class-pooling exactly ties average
      pooling (80.3 = 80.3).** That is the closest published from-scratch analogue, and it says the
      decoder head is a FLOPs optimization, not an accuracy win.
    - All ML-Decoder's large gains used OpenImages-pretrained backbones fine-tuned on ≤118K images —
      the opposite of our under-trained-backbone regime, which inverts the expected ordering.
    - **Query2Label is not merely expensive, it is impossible here:** one query per class at 19,294
      classes is ~1.33 TFLOPs/layer of self-attention (~4.3× the whole backbone) and ~10 GB of
      attention memory per image. ML-Decoder measured a full transformer-decoder head **OOM at
      9,600** classes; we are 2× beyond that.
    - Head *expressivity* is settled and is not the reason to do this (§6.1: 2k+1 = 73 ≪ 896).
    So: an ML-Decoder-style layer with **K ≈ 100** queries cross-attending patch tokens, feeding the
    existing 896→19,294 linear head — ~1.4% of backbone FLOPs, zero added head params. **Do not
    scale K.** And run it only after the instrument is fixed: +0.7 pooled mAP is below the
    resolution of the current metric, and a pooled gain could be entirely head-class movement given
    the top decile is 83.7% of label mass.

### What to stop doing

- Stop treating "V1 hit a label-noise ceiling" as established. It has never been measured.
- Stop investing in γ_neg control machinery. `ASLDriveManager` → telemetry-only is right; its
  per-decile EPR / sibling-gap / non-GT histogram outputs are worth keeping as observables.
- Stop citing P-ASL as the anchor for a positive-only regime. Use ASL Appendix F.

