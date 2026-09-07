# V2 Plan Review — frozen evidence appendix (2026-07-28)

> **This is a frozen 2026-07-28 evidence appendix to [`v2-plan.md`](../v2-plan.md), not a plan.**
> Its conclusions have all been absorbed into the plan; what remains is the measurement,
> recomputation and paper-quotation detail the plan cites without reproducing. **Do not edit.**
>
> - **`v2-plan.md` wins on every disagreement**, without exception.
> - **Code line refs here are as-of 2026-07-28 and have drifted** (the codebase has moved by up to
>   ~586 lines). The plan carries refreshed refs; use those.
> - Sections fully carried by the plan were **deleted 2026-07-29**, so the numbering has gaps
>   intentionally. Every recommendation and verdict was removed with them — instructions live in
>   `v2-plan.md` only. Two deleted instructions are named so nobody acts on a memory of them:
>   **"set `thresholds=None`"** (superseded — plan §8.3 keeps `ap_thresholds: 200`; exact costs 93 GB
>   and ~103 min/pass, and the binning bias is measured at ~−0.0031 and near-uniform) and **"set
>   `early_stopping_threshold` to ~1 measured bootstrap SE"** (superseded — plan §8.3 uses **3e-3**
>   from three converging sources; the SE heuristic is 2–7× too small because it ignores the bias
>   term).
> - It reviewed `v2-plan-correction-2026-07-28.md`, `progressive-training-plan.md` and `ASL_plan.md`
>   — **all three retired 2026-07-29, full text in git history**. Internal section refs below
>   ("progressive-plan §2.1", "ASL_plan §7", "the correction doc's §3.2") point into those **retired**
>   docs, **not** into `v2-plan.md`, whose numbering differs. Verdicts on the retired docs' framing
>   are gone; `v2-plan.md` does not share that framing.
> - **Method:** every quantitative claim was recomputed from `vocabulary.json`, the TensorBoard event
>   files, `logs/training.log`, `pr_threshold_*.json`, `corrections_report.json` and the training
>   code — or read out of the primary paper PDFs.

---

## 0. The lever inventory

**These Δ-mAP figures are JUDGEMENTS/ESTIMATES, not measurements.** Each is a reviewer's
extrapolation from the cited out-of-domain result to our regime; the *citations* are measured, the
*mapping onto this run* is not. `v2-plan.md` deliberately does not carry most of these numbers.

| Lever | Est. Δ mAP (**estimate, not measured**) | Status at review time |
|---|---|---|
| Finish training (V1 got ~60% of DeiT-B's from-scratch budget for a 2.9× larger model) | **+1 to +4** | Identified, then contradicted by the ceiling framing |
| Pretrained init (LiGO-grow, distil, or adopt ViT-L) | **+1 to +2.5** | Never argued — asserted in one table cell |
| Weight EMA | +0.2 (clean-ImageNet analogue) to **+9** (measured under 40% real label noise) | Absent from plan and codebase |
| Run 512px instead of gating it | **+0.8 to +1.4** | Gated behind a trigger that will not fire |
| Label completion, tail-targeted | **+3.6** on rare classes (RAM++) | De-scoped |
| LayerScale at depth 18 | **+0.5 to +1.0** | Absent from plan and codebase |
| 2D RoPE across three resolution changes | +1.4 to +2.5 top-1-equivalent | Absent |
| Attention-pooling head, K≈100 | +0.7 | Deferred behind the wrong gate |
| **P-ASL selective ignore** | **+0.87, discounted for regime mismatch and a pretrained-backbone measurement** | The centrepiece |

Most of these deltas are **below the resolution of the current instrument**, which is why the
instrument work precedes everything.

---

## 2. V1 was under-trained, not noise-limited

### 2.1 It was not early-stopped — it was stopped by hand

`grep -c "Early stopping triggered" logs/training.log` → **0**. The string the code emits on early
stop never appears anywhere in the log.

The two **terminal** stops (corrected 2026-07-29 — an earlier draft of this section misidentified
them):

| leg | log lines | date | `image_size` | checkpoint | resumed? |
|---|---|---|---|---|---|
| **Phase 1 terminal** | `logs/training.log:12904-12910` | 2026-05-05 12:30 | **320** | `checkpoint_epoch_33_step_209899.pt` | **No** |
| **Phase 2 terminal** | `logs/training.log:14030-14036` | 2026-05-09 07:37 | 448 | `checkpoint_epoch_7_step_85517.pt` | **No** |

Both are `Soft stop requested` → `Soft stop checkpoint saved` → `Soft stop engaged. Exiting training
loop before validation.` — user-initiated, mid-epoch, no crash, no OOM, no criterion.

**Not** terminal, and previously misread as Phase 1's ending: `:13353-13358` (2026-05-07,
`image_size: 448`, `checkpoint_epoch_3_step_38923.pt`) is a **Phase-2** soft stop at epoch 3, and
Phase 2 **was** resumed from it and ran on to epoch 7 / step 85517. The log contains several further
routine soft-stop/resume cycles (`:10437`, `:12785`, `:13007`, `:13103`). So "never resumed" holds
for the two terminal events only — but the conclusion is unaffected: **no early-stopping criterion
ever fired, and both phases ended on a manual stop.**

`huggingface_release/README.md:186` is candid about the real reason — *"Continuing was unlikely to
buy enough real gain to justify the extra training time"* — a cost/benefit judgement call, later
hardened into a measured fact.

### 2.2 Every metric was still improving at the stop

| | last-5-epoch mAP slope | value at stop | budget used | LR at stop |
|---|---|---|---|---|
| **Phase 1** (320px) | **+0.0065 / epoch** | 0.6518 | **33/40 epochs**, 81.8% of cosine cycle | **14% of peak** |
| **Phase 2** (448px) | **+0.0028 / epoch** | 0.6744 | **6/15 epochs**, 41.8% of cycle | **76% of peak** |

Val loss and mAP both improved **monotonically at every single validation event** in both phases. The
claimed signature — *"mAP growth flattened while val loss kept falling"* — does not exist in the
data; the two moved in proportion (−2.8% loss vs +2.3% mAP across P2).

Naive linear extrapolation over Phase 1's seven unrun epochs is **+0.041 mAP**, before any
cosine-tail bonus — and cosine schedules deliver a disproportionate share of final gain in the anneal
tail that was never run.

### 2.4 The frequency-bucket claim is inverted in the data

The retired progressive plan's §2.1 asserted V1 showed a *"uniform plateau across frequency buckets;
rare bucket still climbing."* `val_bucketed/*/mAP` was logged. **Phase 2 E4→E5, per-epoch Δ:**

| bucket | tags | mean support in 30K val | Δ mAP |
|---|---|---|---|
| 500–999 (rare) | 7,619 | **2.45** | **+0.00211** ← slowest |
| 1K–5K | 7,873 | 8.5 | +0.00191 |
| 5K–10K | 1,482 | 30.3 | +0.00239 |
| 10K+ | 2,314 | 399 | +0.00217 |
| 300–499 | **0** | — | never measured |

**No bucket plateaued** — all four move at the same **+0.0019–0.0024/epoch**. The specific
differential claimed (rare climbing while others flatten) is absent, and the rare bucket is if
anything the *slowest*. The genuinely-rare bucket contains zero tags because the vocabulary is
floored at 500. *(This is a different measurement from plan §8.5's table, which reports P1-final
**levels** per bucket, not per-epoch deltas.)*

---

## 3. P-ASL does not transfer to a positive-only regime

Verified against the paper PDF ([arXiv:2110.10955](https://arxiv.org/abs/2110.10955), CVPR 2022).
**The reviewed plans' factual claims about the paper are all correct** (9,600 classes; ASL-Negative
85.85 → P-ASL Selective 86.72; the Ω_L ∪ Ω_P mechanism; γ⁺=1, γ⁻=2, γᵘ=7, K=200, η=0.05;
Ignore-mode prior estimation). The problem is the transfer argument, in four places.

*(`v2-plan.md` §1 delegates the P-ASL detail to this section — it is load-bearing, keep it.)*

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

The retired correction doc's §3.2 adopted the paper's `P̂(c)`, estimated from a model trained in
Ignore mode. At N = ∅, Ignore mode's loss is `Σ_{c∈P} L_F(p_c, γ⁺)` only — whose global optimum is
**p_c = 1 for every class**. The estimator returns 1 everywhere. The paper's headline §4.2
contribution **cannot be run as specified.**

*(The proposed substitute — averaging `best_model.pt` predictions over 6.8M images — also isn't
needed. Where Ω_P is active at η=0.05, measured ρ is ~0.4%, so empirical tag frequency from
`vocabulary.json` is already within half a percent of the true prior.)*

### 3.3 γᵘ = 7 is right — for a completely different reason

The reviewed plans read γᵘ=7 as "the published value for exactly our label regime." It is the value
for the branch the paper **distrusts** — set high precisely so the trusted γ⁻=2 branch does the
boundary-setting. Figure 8 varies γ⁻ with γᵘ pinned at 7 and states: *"The case of γ⁻ = 7 represents
the standard ASL. As can be seen, the mAP score increases as we lower γ⁻, up to 2."* In our regime U
**is** the branch that carries the real learning signal — so the plans imported the paper's *worst
tested configuration* for that role. **The paper never ablates γᵘ at all.**

**But γ=7 is independently and correctly supported** — by ASL's own Appendix F
([arXiv:2009.14119](https://arxiv.org/abs/2009.14119)), which is assume-negative at ~5,400–9,600
classes and extreme imbalance, a structurally correct match. **Right number, wrong citation** — and
it matters, because Appendix F pairs γ⁻=7 with **γ⁺ = 0** (P-ASL uses γ⁺=1, which would down-weight
our already-scarce positives) and with margin **m = 0.05**, so clip 0.2 → 0.05 is right for a reason
other than the one originally given. *(Quote and full corroboration: `v2-plan.md` §4.)*

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

Ω_L also inherits the exact objection the plans used to **reject** Hill/SPLC — retired
progressive-plan §1.2: *"SPLC's self-relabel flip assumes a calibrated (pretrained) backbone and is
unsafe in from-scratch Phase 1."* Ω_L is equally prediction-dependent, and P-ASL runs on pretrained
backbones throughout. The skepticism was applied to the rejected option and not to the adopted one.

---

## 4. Instrument measurements

### 4.1 The f1_macro artifact is confirmed — and larger than stated

The in-training metric is hard-coded at θ = 0.2653. The measured F1-optimal threshold is
**θ ≈ 0.76–0.81**. Properly thresholded, V1-P2 posts **micro F1 0.7004** and **per-tag macro F1
0.6751** — against the ~0.013 the training loop was reporting. It means the manual stop decision
(§2.1) was made while watching a number with no relationship to model quality.

Also measured: P1-e27 → P2-e7 moved per-tag macro F1 **0.6218 → 0.6751 (+8.6% relative)** on an
identical eval set. A model at a noise ceiling does not gain 8.6% macro F1.

### 4.3 Val-set size vs what it can support

| val size | tags <5 positives | tags <10 | decile-10 median positives | decile-10 mAP 95% CI |
|---|---|---|---|---|
| **30,000 (then current)** | **44.5%** | **65.8%** | **2.4** | **±0.0125** |
| **100,000** | **0%** | **0%** | **8.9** | — |
| 276,000 (full 5% split) | 0% | 0% | 21.8 | ±0.0045 |

Aggregate macro-mAP survives (simulated 95% CI ±0.0027). Confirmed against the real artifact: on the
30K set only **10,406 of 19,292 tags** have support ≥5, and 958 have zero positives. Also flagged in
`overfitting-risk-assessment.md` Risk Factor 4 (April 2026; retired 2026-07-29, git history).

### 4.5 The split is not group-aware

`dataset_loader.py:2664-2678` (**stale ref** — see plan §8.2) — plain uniform random 95/5 over
individual JSON sidecars, **`seed=42`**. No perceptual-hash, artist, series, or post-id grouping.
`logs/dedup_hashes/dedup_clusters.json` measured a **0.26%** near-duplicate cluster rate over 5.54M
images — but the **corpus at split time was 5.92M**, so **~382K images were never dedup-scanned**,
and **whether the 14,230 flagged deletions were applied is not verifiable from the artifacts**.
**Expected leakage is order 10²–10³ of the 30K val set:** a real hygiene defect worth fixing with
group-aware splitting, **not** large enough to explain 0.674.

*(Denominator note: 30,000 is 0.51% of the 5.92M corpus at split time and 0.44% of the ≥6.8M corpus
today. Plan §8.2 states both.)*

---

## 5. Errata in the reviewed measurement sections

**(b) The `gt_current = 0` claim is wrong.** `exercise`, `gibson_les_paul`, `beam_saber`,
`white_wristband` were called *"concepts present in the corpus with zero labels anywhere."* In the
training vocabulary they have **559 / 1,410 / 998 / 705** occurrences. The zero is in the
DataCleaning Project's separate GT store, which disagrees with `vocabulary.json` by a median factor
of **1.115×** (range 0.73–1.77×). The report's own `training_count_oppai` column matches
`vocabulary.json` **exactly for all 126 tags** — that is the correct denominator, and it was not
used. Recomputed per-bucket **median** ρ is **0.47 / 1.59 / 3.30 / 5.11%**, not the aggregate
**0.32 / 3.73 / 4.51 / 12.10%** quoted.

**(c) Per-tag missingness IS estimable**, contra the retired ASL_plan §7 (*"they require known
per-tag missingness rates we don't have"*). Fitting the 123 usable campaign tags against their
training counts:

```
ρ̂(c) ≈ 0.50 × freq(c)^−0.365       corr = −0.52,  R² = 0.27,  n = 123
```

→ 0.19% at 4M, 0.75% at 100K, 1.73% at 10K, 4.02% at 1K, 5.17% at 500. A lower bound with a loose
fit, but structurally the same object Jain et al. (KDD 2016) use. If an ignore set is wanted, select
on **ρ̂, not prior**: ρ̂ ≥ 5% covers 1,330 tags (6.9% of vocab) but only **0.3% of label mass** —
precisely targeted and nearly free, the exact inverse of Ω_P's profile.

**(d) Corpus size is a lower bound of unknown tightness.** N ≥ **6.77M** is inferred from `highres`.
If `highres` covers ~70% rather than ~100% of the corpus, **N ≈ 9.7M and the positive rate falls to
~0.12%**. No independent corpus count exists in the repo.

**(f) The V2 architecture change has not been applied.** Still true as of 2026-07-29:
`configs/unified_config.yaml:33` `hidden_size: 1024`, `:35` `num_attention_heads: 16`, `:36`
`intermediate_size: 4096` — the V1 Phase-2 config with only the loss and comment blocks rewritten.

---

## 6. Architecture and recipe

Verified sound at review time:

- **896×18 is not a non-standard shape — it is a published compute-optimal one.** SoViT-150m/14
  ([arXiv:2305.13035](https://arxiv.org/abs/2305.13035)) is width **880**, depth **18**; the plan
  matches to within 2%. Aspect ratio 49.8 sits between ViT-L (42.7) and ViT-B (64).
- **Generalized Neural Collapse ([arXiv:2310.05351](https://arxiv.org/abs/2310.05351)) genuinely
  covers K ≫ d.** The width-896-for-19K-classes argument stands.
- **patch16 is right, and the central claim is provable.** Attention is a *minority* of compute at
  every planned resolution — **7.5% at 320, 14.6% at 448, 19.1% at 512** — so patch14@448 is
  **1.36× FLOPs** (the original figure was right; the revised "1.4–1.6×" conflates the
  activation-memory term). 512px/patch16 gives N=1025 — **identical token cost to 448px/patch14** —
  while delivering real pixels instead of a finer mesh over the same downscaled raster.
- **The 320→448 schedule has direct in-regime precedent**: **ASL itself trains Open Images (~5,400
  classes, partial labels) at 224 for 30 epochs then fine-tunes at 448.**
- **Phase-2 LR is not too low.** DeiT III's high-res fine-tune is AdamW 1e-5 @ batch 512 × 20 epochs;
  2.6e-5 @ 768 × ~106K steps is higher in both. V1's 1e-5 was also inside DeiT III's range, so P2 LR
  did **not** cap V1 at 0.672.

### 6.1 The MLP ratio, and the drop_path self-correction

- **The MLP ratio contradicts its own citation.** SoViT-150m uses MLP **2320** (ratio **2.64**), not
  4×896 = 3584; SoViT's fitted exponents put **s_MLP ≈ 0.60 > s_depth ≈ 0.45 > s_width ≈ 0.22**,
  i.e. MLP should be *relatively narrow* at this scale. Following the paper exactly gives **~150M
  total, −22% params and compute** for the shape actually validated.
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
  them. ***An earlier draft of this review recommended cutting it; that was based on reading 0.25 as
  a uniform rate. Verify the convention before touching this number.***
  Also note DeiT III's explicit rule — **if you extend training you must raise WD and drop_path with
  it** (+0.05 drop_path per 200 epochs); the 800-epoch gains are not reachable at the 400-epoch
  regularization setting.

### 6.2 Levers the reviewed plans omitted entirely

| Lever | Evidence | Est. Δ (**estimate**) | Cost |
|---|---|---|---|
| **Run 512px unconditionally** (don't gate Phase 3) | ASL COCO 448→640 **+1.4 mAP**; ML-Decoder **+1.1**; Query2Label **+1.1** — unusually consistent across three multi-label papers | **+0.8 to +1.4** | compute |
| **2D RoPE** instead of interpolated learned pos-embeds | RoPE-ViT (ECCV 2024, [arXiv:2403.13298](https://arxiv.org/abs/2403.13298)): +1.4 @384, +2.5 @512 for ViT-L, at **0.01% of FLOPs** | large, for a plan whose spine is 320→448→512 | small code |
| **Weight EMA** | **The payoff is bimodal and the upside is large.** On clean ImageNet EMA is worth +0.1–0.2 (DeiT-S 80.0→80.2, DeiT-B 81.0→81.1 — which is why DeiT III, big_vision and ResNet-strikes-back all *drop* it). Under **40% real human label noise** it is worth **+9 points** (CIFAR-100N ResNet-34 55.50 → **65.15**; CIFAR-10N 77.69 → **86.71**) — TMLR 2024 ([arXiv:2411.18704](https://arxiv.org/abs/2411.18704)). It also cuts prediction churn 18.84→11.69 and post-temperature ECE 4.67→3.13. Verified absent from the codebase | **+0.2 to +2** | trivial — one weight copy; keep both checkpoints, so downside is zero |
| **LayerScale (ε = 0.1)** | CaiT ([arXiv:2103.17239](https://arxiv.org/abs/2103.17239)) Table 1 measures **+1.0 top-1 at exactly depth 18** (80.7 → 81.7) against a *drop-path-tuned* baseline. DeiT III uses it in **every** configuration. Verified absent from the codebase — the only `layer_scale` hits are dead LLRD code | **+0.5 to +1.0** — best-evidenced item at our exact depth | one learned diagonal per residual branch |
| **fp32 optimizer state for the 19,294-way head + pos-embeds** | 8-bit AdamW is *not* verified lossless for ViT-from-scratch: [arXiv:2309.01507](https://arxiv.org/abs/2309.01507) Table 2 measures **Swin-T IN-1k from scratch, 8-bit AdamW 81.0 vs fp32 81.2 (−0.2, exceeding seed std)**. Dettmers ([arXiv:2110.02861](https://arxiv.org/abs/2110.02861)) contains **no ViT and no 8-bit-*Adam* vision result** — its ImageNet row is 8-bit *momentum-SGD*. A 19K head at 0.18% positive rate is structurally the sparse-embedding shape that motivated Dettmers' Stable Embedding Layer | +0.1 to +0.2, plus tail-risk insurance | ~140 MB via `GlobalOptimManager` |
| **Longer warmup (→ ~10K steps ≈ 1.5 epochs)** | Beyer et al. use 10K warmup steps at exactly batch 1024; Wortsman Fig. 5 — longer warmup flattens LR sensitivity; σReparam's grid diverges on 7/8 configs at ViT-B, batch 1024–2048, LR 5e-4–1e-3 — our exact box | +0 to +0.3, plus variance reduction | free |
| **QK-norm** | Wortsman ICLR 2024 ([arXiv:2309.14322](https://arxiv.org/abs/2309.14322)); ViT-22B. Flattens LR sensitivity across orders of magnitude | 0 direct; removes a run-ending risk on a multi-week run | trivial |
| **Attention-only fine-tune at the resolution switch** | *Three things everyone should know about ViT* ([arXiv:2203.09795](https://arxiv.org/abs/2203.09795)): MHSA-only fine-tune lands within ±0.1 of full, −10% memory/time — and fewer trainable params means **slower false-negative memorization** | ~0 quality, real cost saving | small |
| **Aspect-bucketed batching** | NaViT ([arXiv:2307.06304](https://arxiv.org/abs/2307.06304)). Square letterboxing typical portrait booru art spends ~25–35% of 784 tokens on gray bars | 0 to +0.3, ~25% throughput | medium |

### 6.3 Pretraining — the single largest unexamined decision

*(At review time the reviewed docs asserted "from scratch" in a TL;DR table cell and never argued
for it. **`v2-plan.md` §5 is now that argument** and holds the decision open — the material below is
its evidence base, not an outstanding criticism of the current plan.)*

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

The cheap falsifiable test — a DINOv2-L / SigLIP linear probe on a 100K slice against the 19K
vocabulary, ~1 GPU-day — has never been run, and **no scholarly evaluation of foundation-model
features on booru-style illustration appears to exist.**

---

## 7. Dissent on the P-ASL rejection (recorded)

*Dissent recorded:* a second literature sweep, scoped to **scale** rather than annotation regime,
concluded P-ASL should stay primary on the grounds that nothing published 2023–2026 beats it above
1,203 classes. That is true, but it is an argument from absence — nobody else evaluates at that
scale, and P-ASL's evaluation is on a dataset with 37.7M verified negatives. The structural argument
in §3 (half the gain needs those negatives; the prior estimator is degenerate without them) is not
addressed by the scale argument, and RAM's 6,449-tag plain-ASL result is the closer regime match.

*(This is the only record that the rejection was contested. `v2-plan.md` §4 carries the same
dissent + rebuttal.)*

---

## 8. Attention-pooling / ML-Decoder — the detailed scaling evidence

Expect **~+0.7, not a step change.** The right framing is not "linear vs decoder" but
**average/CLS pooling vs learned attention pooling**, and the evidence is more modest than it looks:

- ML-Decoder's gain **does not grow with class count**: 80 classes **+1.1**, 1,000 **+0.6/+0.8**,
  **9,600 +0.7/+0.8**. And K=100 → 200 → 400 at 9,600 classes moves **86.7 → 86.8 → 86.8**, so
  quadrupling query resolution buys **+0.1**. Queries carry **no class semantics** (learnable, fixed
  random, and word-based queries all score identically).
- **CaiT Table 2, from scratch at matched capacity: attention class-pooling exactly ties average
  pooling (80.3 = 80.3).** That is the closest published from-scratch analogue, and it says the
  decoder head is a **FLOPs optimization, not an accuracy win**.
- **All of ML-Decoder's large gains used OpenImages-pretrained backbones fine-tuned on ≤118K
  images** — the opposite of our under-trained-backbone regime, which inverts the expected ordering.
- **Query2Label is not merely expensive, it is impossible here:** one query per class at 19,294
  classes is ~1.33 TFLOPs/layer of self-attention (~4.3× the whole backbone) and ~10 GB of attention
  memory per image. ML-Decoder measured a full transformer-decoder head **OOM at 9,600** classes; we
  are 2× beyond that.
- Head *expressivity* is settled and is not the reason to do this (§6.1's companion result:
  2k+1 = 73 ≪ 896).

So if ever built: an ML-Decoder-style layer with **K ≈ 100** queries cross-attending patch tokens,
feeding the existing 896→19,294 linear head — ~1.4% of backbone FLOPs, zero added head params.
**Do not scale K.** And only after the instrument is fixed: +0.7 pooled mAP is below the resolution
of the current metric, and a pooled gain could be entirely head-class movement given the top decile
is 83.7% of label mass. *(`v2-plan.md` §11 rejects it for this run on exactly these grounds.)*
