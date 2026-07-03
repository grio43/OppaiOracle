# Progressive-Resolution ViT Training Plan — V2

> **Scope.** This is the **active guide for the V2 model**. V1 (the ~248M run that early-stopped at a **label-noise ceiling ~0.674 mAP**, not a capacity ceiling) is preserved as a compact **historical appendix** at the bottom — footnotes, not the plan. Everything above the appendix is what we are building next.
>
> **Verification status (2026-06-04).** Every setting below was re-checked against current scholarly literature (incl. 2024–2026 work) and against the live codebase. Corrections from that pass are folded in and marked **[corrected]** where a previous version of this plan was wrong. The biggest correction: **the old H1 "lower `clip` 0.2→0.05" was backwards** (see §1.3).

---

## 0. What V2 is (TL;DR)

| | V2 default |
|---|---|
| Backbone | **896 width × 18 layers**, patch16, head_dim 64 (14 heads), CLS-pool, learned pos-embed |
| Params | **~192.3M** total (175.0M backbone + 17.3M head @ 19,294 tags) — **−23%** vs V1's 248.1M |
| Data | **6.3M** anime images (+17% vs V1's 5.4M), vocab **19,294** tags, ~0.6% positive rate/image |
| Training | **from scratch**, bf16, AdamW8bit, single-GPU VRAM-bound |
| Resolution | progressive **320 → 448** ( → optional **512** Phase 3, gated) |
| Loss | **ASL** (Asymmetric Loss) — the **sole** loss: γ_pos=0, clip≥0.2, γ_neg schedule **Phase 1 = 7 → Phase 2 sweep 7→5(→4)** gated on a clean slice. Hill/SPLC evaluated and **not** pursued (§1.2). |

**The reframe that drives V2.** V1's wall was **label noise (missing positives), not capacity** — val loss kept falling while mAP flattened, with no overfitting reversal. Two consequences:

1. The ~250M model carried **unusable headroom**, so cutting to ~192M costs ~nothing in achievable mAP.
2. A smaller architecture will **not by itself** raise mAP. mAP only moves by attacking the ceiling.

So V2 splits into **two orthogonal workstreams that ship in the same run**:

- **(A) Right-size the architecture** — a cost/efficiency decision (cut ~23% params, keep detail). §2.
- **(B) Attack the label-noise ceiling** — the *only* lever that moves mAP: **loss → data → convergence**. §1.

**Expected attribution of next-iter lift** (loss-lens estimate, *[medium]* confidence): with the loss lever now scoped to **ASL γ_neg-schedule + clip tuning** — a narrower lever than a loss-family swap — the mAP budget shifts toward data: **~35% loss (ASL γ_neg recovery), ~55% data/instrumentation (clean slice + missing-positive completion + wrong-positive cleaning), ~10% the +17% data growth, ~0% capacity.** Do not expect a smaller model to beat V1 on mAP unless (B) is executed. Architecture is **second-order**; the loss + data levers are where the mAP budget is.

---

# 1. Workstream B — Tag noise / missing positives (the headline)

> This is where mAP actually moves, and it is the workstream we are actively improving. The dataset has **missing-positive noise**: a tag is genuinely present but unlabeled, and the loss treats it as a negative. As the model converges it learns to *suppress* unlabeled-but-correct tags, which looks like healthy mAP on a noisy val set but stalls true mAP. The literature below is validated **under missing-positive noise**, not on clean ImageNet — that distinction is the crux.

## 1.1 The mechanism (well-grounded)

- Networks **learn clean labels first, then memorize noisy negatives** (false negatives = unlabeled-correct tags). Kim et al., *Large Loss Matters in Weakly Supervised Multi-Label Classification* (CVPR 2022, [arXiv:2206.03740](https://arxiv.org/abs/2206.03740)) — false negatives are separable as **large-loss** samples once early learning ends; their LL-R / LL-Ct / LL-Cp methods use a **per-epoch linearly-increasing rejection rate** (`delta_rel`), not a static loss cutoff.
- The early-learn-then-memorize phase transition is fundamental even in linear models: Liu et al., *Early-Learning Regularization (ELR)* (NeurIPS 2020, [arXiv:2007.00151](https://arxiv.org/abs/2007.00151)).
- **[corrected attribution]** "Reduced regularization / more capacity *extends* the memorization window" is **not** shown in Kim 2022 — it is an extrapolation from Arpit et al., *A Closer Look at Memorization in Deep Networks* (ICML 2017, [arXiv:1706.05394](https://arxiv.org/abs/1706.05394)) and ELR. It is defensible but untested for missing-positive multi-label. Practical consequence: a **large from-scratch model raises the FN-memorization stakes**, making a large-loss/ELR-style mitigation *more* important, not less.

## 1.2 The loss decision — ASL (chosen); Hill/SPLC considered, not pursued

**Decision: ASL is the sole loss.** We tune ASL's existing knobs (γ_neg schedule, clip, γ_pos) **by config only** — no new loss module, no `loss_type` switch, no SPLC self-relabel plumbing. The full ASL spec is §1.3; the per-phase schedule is §3.1.

**Why ASL and not Hill/SPLC.** Hill/SPLC (Zhang et al., *Simple and Robust Loss Design for Multi-Label Learning with Missing Labels*, [arXiv:2112.07368](https://arxiv.org/abs/2112.07368)) is, on paper, the stronger *missing-positive* tool: ASL's high-γ_neg focusing only down-weights the `p>0.9` region, leaving the **`p∈[0.5,0.9]` semi-hard-negative band — exactly where missing positives concentrate — at large weight**, and Hill/SPLC target that band directly (MS-COCO @40% labels: **BCE 70.49 → ASL 72.70 → Hill 75.15 → Focal-margin+SPLC 75.69 mAP**). We are **deliberately not taking that gain** this iteration:
1. Those gains are **COCO-80, pretrained-backbone** numbers — **none validated at 19K tags / 0.6% positive / from-scratch**, so the headline margin is uncertain in our regime.
2. SPLC's self-relabel flip assumes a **calibrated (pretrained) backbone** and is unsafe in from-scratch Phase 1 — it needs a `change_epoch` delay, per-class/percentile τ, and a clean-slice flip-precision gate before it can be trusted, i.e. genuine **net-new code + risk** (a `HillSPLCLoss` module, a `loss_type` dispatch, and `current_epoch` threading).
3. **ASL is already the live, validated criterion** ([loss_functions.py](../loss_functions.py)), so the ASL re-tune is **zero-code and immediately runnable**.

The cost we accept: we forgo the semi-hard-band recovery and instead attack the same missing-positive ceiling with the **γ_neg schedule (§1.3)** plus the **data-side levers (§1.5, §1.7)**. **If the clean-slice measurement later shows the ASL schedule plateauing below target, Hill/SPLC is the documented fallback to revisit — but it is out of scope for this plan.**

- Keep `label_smoothing=0.0` (Xia 2025; the code already isolates focal gating from smoothing — §1.6 — so this is a non-issue). *(loss-agnostic; stays regardless of the loss choice.)*

## 1.3 The ASL configuration — the loss we ship

ASL is the loss (§1.2). The settings below are the **entire** loss plan; the per-phase schedule is §3.1.

**γ_pos = 0 (all phases, no A/B).** ASL optimal ("simply setting gamma_pos=0 leads to the best results"); raising it *hurts* — Ridnik Fig 6 shows mAP↓ as γ_pos rises from 0, and γ_pos<0 fails to converge, measured at MS-COCO **~3.6%** positive. At our **0.6%** the case for 0 is **stronger**, not weaker. **Double-justified** by sparse positives *and* missing-positive noise. (Code consequence: with γ_pos=0 every annotated positive flows through plain BCE at full gradient — which is also why ASL structurally cannot touch *wrong*-positive noise; §1.7/§4.)

**clip ≥ 0.2 — keep high; do NOT lower to 0.05** [corrected — this reverses the old plan]. The code's `clip` **is** the official ASL negative-branch margin `m` (verified against [loss_functions.py:294](../loss_functions.py#L294), [:301-303](../loss_functions.py#L301-L303): `probs_neg=(p−clip)₊`, log term `log(1−p+clip)` — identical to official ASL on the negative branch only). The margin protects an already-learned missing positive (a true-positive mislabeled negative, hence high `p`) by **capping its negative loss at ~`−log(clip)`** instead of letting `−log(1−p)` diverge — **raising** `clip` tightens that protection; lowering it removes it. ASL Fig 7: focal-optimal margin is **0.3–0.4** vs CE-optimal 0.05. Keep clip at **0.2** (optionally raise toward 0.3–0.4); never drop it. *(clip is a negative-branch knob — it has **no** bearing on wrong-positive/aqua noise either way; the old "lower clip → aqua fires more" claim was backwards, §1.7/§4.)*

**γ_neg — keep HIGH in Phase 1, sweep DOWN in Phase 2** [corrected — the single most important schedule fix in this pass; it **reverses** the old plan's ascending 4→5 into a descending 7→5→4]:
- **Phase 1 (from-scratch, 320px): γ_neg = 7 (keep high). Do NOT drop it.** With γ_pos=0 the positive gradient mass is *fixed*; at 0.6% positive, negatives outnumber positives **~165:1**, so lowering γ_neg *multiplies* every surviving negative's weight (~37× at p=0.5 for 7→4). On an **uncalibrated from-scratch** model many negatives clear the clip with diffuse probabilities, so a γ_neg drop flips the gradient balance **toward** negatives and **suppresses** the rare positives — the opposite of the goal (toy aggregate neg/pos mass ≈3.4 early-from-scratch vs ≈0.2 clean).
- **Phase 2 (mature/cleaned, 448px): sweep γ_neg 7 → 5 → 4** toward ASL's published **2–4** sweet spot (Ridnik et al., ICCV 2021, [arXiv:2009.14119](https://arxiv.org/abs/2009.14119)), **gated on the clean slice** (group-wise recall on golden positives vs FP-rate on golden negatives, §1.5). Recommended landing **5**; go to **4 only with clip kept ≥0.2**. **Hold at 7 if the clean slice shows no benefit.** Lower γ_neg only as cleaning lowers the *global* missing-positive rate — a global drop relaxes M1 protection on the ~19K still-noisy tags, the blunt-instrument tradeoff we accept by staying config-only/global (no per-group vector). *(Context: γ_neg=7 was the paper's value only for Open Images, and there the untagged negatives were down-weighted; our full-weight γ_neg=7 is strictly more aggressive — another reason to back off in P2 once cleaning justifies it.)*

**Confidence / prerequisite.** **HIGH** that the *direction* is right (high in P1, descending in P2); only **MEDIUM** that any single global γ_neg is near-optimal at 19K/0.6% — nothing cited exceeds ~9,600 classes, so 5 vs 4 is **directional, not paper-calibrated**. **The Phase-2 sweep is unmeasurable** until the clean held-out slice exists *and* model-selection/early-stop runs on a clean metric (today it selects on noisy `val/f1_macro`) — see §1.5 and §5. Until then, run Phase 1 at γ_neg=7 and do not sweep.

## 1.4 Instrumentation (H5) — per-epoch missing-positive diagnostics

Without these the next run is as unfalsifiable as V1 ("looks like the noise ceiling" with nothing to measure it). Add to the validation pass:

| Diagnostic | Signal | Citation |
|---|---|---|
| `pred_pos_ratio` = mean_active / expected positives | falls <1.0 = suppression collapse | Cole et al., *Single Positive Labels* (CVPR 2021, [arXiv:2106.09708](https://arxiv.org/abs/2106.09708)) Expected-Positive-Regularizer |
| `mean_sigmoid_topK_unlabeled` (K≈10, tags **not** in GT) | falls = suppressing co-occurring/unlabeled-correct tags | ELR (Liu 2020) |
| `cooccur_jaccard@K` (pred-pair vs train-set pair freqs) | falls = predicted co-occurrence diverging from real | Xia et al., *Holistic Label Correction* (ICCV 2023); Zhao & Gomes 2021. **Estimate the prior on the CLEAN slice**, else the noise contaminates the reference. |
| `rare_bucket_ECE` (<1000-freq tags) | rises while rare-mAP plateaus = calibration drift before metric drift | **Use smoothed / equal-sample-bin ECE** (SmoothECE, Błasiok 2023; Dual-TS [arXiv:2308.08366](https://arxiv.org/abs/2308.08366)) — vanilla equal-width ECE is biased at 0.6% positive rate |
| `logit_std_rank_11_50` | compresses = second-tier predictions collapsing onto suppression mode | Kim 2022; Arpit 2017 |
| **`gamma_neg_sweep_recall/FP`** (new) | per cleaned group: recall on golden positives vs FP-rate on golden negatives at each γ_neg in the P2 sweep | directly measures whether the γ_neg down-step is recovering recall without inflating false positives (§1.3) |

`pred_pos_ratio` is ~free (mean_active already computed). The rest need a one-time validation-loop hook to retain top-K logits. Treat all as **direction-only** signals (like a calibration canary); two consecutive monotonic "problem"-direction moves at E5+ is the trigger to spot-check predictions early.

## 1.5 Data-side — the highest-leverage instrumentation (H4 and beyond)

The val metric itself is biased: noisy-val mAP **under-estimates** true mAP and can **mis-rank** checkpoints, because a model that *recovers* missing positives is scored as producing false positives. Zhao & Gomes, *Evaluating Multi-label Classifiers with Noisy Labels* ([arXiv:2102.08427](https://arxiv.org/abs/2102.08427)); Northcutt et al., *Pervasive Label Errors* (NeurIPS 2021 D&B, [arXiv:2103.14749](https://arxiv.org/abs/2103.14749)) — test-label errors can **invert model rankings**. So **do model selection and early-stopping on a clean slice, never on noisy-val mAP.**

**Ordered data-side plan (do these in order):**

1. **Hand-relabel a clean val slice (~2–5K images) — do this FIRST.** It gates everything else and de-confounds the "0.674 = noise ceiling" claim by letting *true* mAP be measured directly. Highest-leverage single action in the whole plan.
2. **VLM / tagger teacher to propose missing positives.** Accept proposals only **above a precision threshold tuned on the clean slice.** Options:
   - **RAM / Tag2Text** (Zhang et al., *Recognize Anything*, [arXiv:2306.03514](https://arxiv.org/abs/2306.03514)) — an **open-vocab tagging** teacher, the most domain-apt for a 19K-tag vocabulary.
   - CDUL-style CLIP global+local pseudo-labels (ICCV 2023, [arXiv:2307.16634](https://arxiv.org/abs/2307.16634)) — but note **CLIP has weak coverage of booru/anime tags**, so RAM-style tagging teachers are likely stronger here.
3. **Multi-label confident learning + co-occurrence cleaning.** cleanlab multi-label (Thyagarajan/Northcutt, [arXiv:2211.13895](https://arxiv.org/abs/2211.13895)) + clean-slice-estimated co-occurrence to prioritize which proposed positives to actually add.

**Clean vs grow.** Given the ceiling is label noise (not data volume), **completing labels beats adding more equally-noisy images.** *Scaling Laws for Data Filtering* (CVPR 2024, [arXiv:2404.07177](https://arxiv.org/abs/2404.07177)) and DataComp show curation can beat raw scale. **Gate any param re-growth on effective *clean-label* signal, not raw image count** (see §2 forward-compat).

## 1.6 Newer (2024–2026) work — and why most of it does NOT transfer

The headline 2024–2026 "SOTA" on missing/partial labels is mostly **single-positive multi-label (SPML)** + **CLIP-pseudo-label** driven. **That regime does not transfer to this project** (we have *many* positives per image, and CLIP has no 19K-anime-tag coverage). State this explicitly so reviewers don't "correct" the plan toward the wrong benchmark.

- **GR Loss** — *Boosting Single Positive Multi-label Classification with Generalized Robust Loss* (IJCAI 2024, [arXiv:2405.03501](https://arxiv.org/abs/2405.03501)). Unifies ASL/Hill/SPLC; replaces SPLC's **hard τ flip** with a **soft, logistic FN-probability estimate** `k̂(p)`. We run **neither** SPLC nor any flip (§1.2), and it is **SPML-specific and weak under high label correlation / many labels per image**, so **context-only, not adopted.**
- **APL** — *Asymmetric Polynomial Loss* (ICASSP 2023, [arXiv:2304.05361](https://arxiv.org/abs/2304.05361)) and **RAL** — *Robust Asymmetric Loss for Multi-Label Long-Tailed Learning* (ICCVW 2023, [arXiv:2308.05542](https://arxiv.org/abs/2308.05542)) extend the Hill term, and RAL targets the long tail (closer to our 19K tags). **Not adopted** — we run plain ASL, not Hill, so these are out-of-scope alternatives; they are the documented **next-loss-family direction** if the ASL γ_neg schedule is later shown insufficient on the clean slice.
- **CSL** — *Incomplete Multi-Label Recognition by Co-learning Semantic-Aware Features and Label Recovery* ([arXiv:2510.10055](https://arxiv.org/abs/2510.10055), 2025): joint feature-learning + iterative missing-label recovery.
- **AdaGC** — *Adaptive Gradient Calibration for SPML* ([arXiv:2510.08269](https://arxiv.org/abs/2510.08269), 2025): student-teacher EMA with **early-learning-indicator-gated** gradient calibration — directly operationalizes the early-learning theory §1.1 relies on.

**Honest frontier note:** *no* method here is validated at 19K-tag scale; all SPML/VLM benchmarks are ≤925 classes, single-positive — far easier. **Expect headline mAP gains to shrink, and validate any adopted method's flip/pseudo-label precision on the clean slice before trusting it.** That 19K-tag-scale gap is the project's real research frontier.

**Backlog (not next-iter):** Distribution-Balanced loss needs class-aware resampling this pipeline doesn't run.

## 1.7 Wrong-positive noise — HLC vs a golden dataset (the *other* half of the ceiling)

> Everything in §1.1–§1.6 attacks **missing positives** (false negatives). The aqua↔green↔blue hair-color confusion is the **opposite** noise: a **wrong positive** — an adjacent class labeled instead of / in addition to the correct one. **No loss in §1.2–§1.3 touches it.** Verified (deep-research pass, 2026-06-13, 25 claims 3-0): ASL feeds any observed positive straight into plain CE (`γ_pos=0`); Hill/SPLC, LL-R/Ct/Cp, SPML-EPR/ROLE, Entropy-Max are **all** missing-positive methods that re-weight/flip *negatives* only and assume the annotated positive is clean (GALC-SLR [arXiv:2108.02032](https://arxiv.org/abs/2108.02032) exists *because* ASL isn't false-positive-robust — Open Images is 26.6% FP). So "add more green" and "tune the loss" both leave a wrong `aqua=1` untouched: ASL's `γ_pos=0` sends every annotated positive through plain BCE regardless, and `clip` is a **negative-branch** knob that cannot affect a positive-branch wrong positive **at all** (the earlier "lowering `clip` makes aqua fire *more*" claim was **backwards** and is dropped — §1.3/§4; clip stays ≥0.2 for M1 protection, with zero bearing on M2). Two — and only two — scholarly levers target wrong positives directly: **HLC** (correct it in training) and **a golden dataset** (anchor detection + targeted removal). This section reviews the choice.

**The two are not substitutes — they fix different things, and one is a prerequisite for the other.**

**Option A — HLC (Holistic Label Correction, Xia et al., ICCV 2023; [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Xia_Holistic_Label_Correction_for_Noisy_Multi-Label_Classification_ICCV_2023_paper.pdf), [code](https://github.com/xiaoboxia/HLC)).** Training-time correction of *class-dependent pairflip* noise — its transition matrix `P[i,i]=1−n, P[i,i+1]=n` is literally the aqua→green→blue ring. It computes a co-occurrence-aware GCN "holistic score" (on an ADD-GCN backbone) and, when a noisy positive's score ratio falls below a scheduled threshold (δ≈0.4 with epoch decay), **replaces** the noisy positive set with the top-k predictions — so it can swap a label between adjacent classes **while preserving legitimate co-occurrence** (multi-colored hair). It never forces hard exclusivity — exactly the property §4 / the color-group constraint demands.
- **Evidence:** VOC/COCO, **≤80 classes**, *synthetic* clean i→i+1 ring at 20/30/40% noise.
- **Cost / risk:** needs a **GCN correction head + retrain** (net-new code, changes the ONNX/export path — interacts with the Query2Label/ML-Decoder decision in §2); the δ schedule is hand-set, not learned; and **its scaling to 19,294 real tags with genuine non-ring noise is unestablished** — the same 19K-scale gap §1.6 flags for every method. Long-tail behavior unknown. **Promising but unproven; do not adopt corpus-wide on faith.**

**Option B — a golden dataset (human-anchored clean labels).** Three distinct things hide under this name; separate them, because their leverage differs:
1. **Clean *eval* slice** — already mandatory in §1.5 (~2–5K hand-relabeled images). Gates model selection / early-stop, de-confounds the "0.674 = noise ceiling" claim, measures *true* mAP. Non-negotiable regardless of A.
2. **Clean *anchor* set for automated cleaning** — the slice that lets you **tune and prove the precision threshold** of any detector (Confident-Learning multi-label, [arXiv:2211.13895](https://arxiv.org/abs/2211.13895), §1.5; or HLC's δ) before trusting it on 6.3M images. Without it, A is unfalsifiable at 19K — V1's exact mistake in a new costume.
3. **Synthetic golden set** (the in-progress prompt-controlled Anima/ComfyUI buckets) — scalable clean labels, but **distribution-shifted** from real booru imagery; use it to *probe* color discrimination and seed rubrics, **not** as the sole real-data anchor.

**Why this is sequenced, not either/or.** HLC, Confident Learning, and the §3.1 ASL γ_neg down-sweep **all** need a clean anchor to set their correction threshold (or, for the sweep, to judge recall vs FP) *and* to prove precision on the rare, visually-adjacent tags — where the model's own probabilities are weakest (the research flagged detector reliability as **worst exactly there**). So a golden dataset is **load-bearing for HLC**, not an alternative to it. Conversely, a golden slice alone does **not** fix the labels on 6.3M training images — it anchors and measures; at-scale correction still needs an automated lever (CL detect-and-remove, or HLC correct-in-training).

**Ranked recommendation (for "what do I build for the next model").**

| Rank | Action | Why | Risk |
|---|---|---|---|
| **1 — do first** | **Golden eval + anchor slice** (extend §1.5's clean slice; oversample the color group) | Prerequisite for *every* wrong-positive method; measures true mAP; de-confounds the ceiling. Highest-leverage, lowest global risk. | Human labor; the gold set is the single point of failure (rubric-first, medium-oversampled, held-out). |
| **2 — scalable, low-risk fix** | **Confident-Learning multi-label** detect-and-**remove** on the color group, gated on the slice, human-reviewed | Model-agnostic (runs on the *current* ViT/ASL probs), **no retrain**, finally extends past additive-only by removing confirmed wrong `aqua=1`. Reversible. | Weakest precision on rare adjacent tags → human-review, never auto-remove. |
| **3 — pilot, not commit** | **HLC** on a color **sub-vocabulary** only | Only *training* method that auto-corrects adjacent pairflip while preserving co-occurrence. | Unproven >80 classes; GCN head + retrain + export change; δ schedule may not transfer. |
| **avoid** | Expecting *any* loss change — the ASL we ship, or a Hill/SPLC swap — to fix aqua/green | All are missing-positive levers, zero on-target effect on wrong positives; `clip` is negative-branch only and does not touch M2 (§1.3/§4). | No benefit on wrong positives; global long-tail risk. |

**Bottom line for the immediate situation** ("adding missing labels, won't cover everything"): the missing-positive half is handled by the ASL γ_neg schedule (§1.3) + §1.5 teachers. The **wrong-positive half needs the golden anchor first**, then CL-based removal as the scalable, reversible fix, with **HLC reserved as a later sub-vocab pilot** once the anchor can actually measure whether its 19K-scale correction is net-positive. **Build the golden dataset; treat HLC as the experiment the golden dataset *unlocks*, not a substitute for it.**

---

# 2. Workstream A — Architecture (right-size, keep detail)

**Verified exact parameter counts** (num_tags=19,294, MLP ratio 4, patch16 @ 448). Confirmed three ways incl. live `SimplifiedTagger` instantiation — diff = 0 vs closed form. Per-block = `12·d² + 13·d`; head = `width·num_tags + num_tags`.

| Config | Backbone | Head | **Total** | vs V1 248.1M |
|---|---|---|---|---|
| **896w × 18L** (default) | 175.0M | 17.3M | **192.3M** | **−23%** |

**Default: `896w × 18L`, patch16, 448px (~192.3M).**
- Cuts width 1024→896 (**88%** of the residual stream — where fine color/accessory features and 19K-tag head separability live) while keeping **all 18 layers**.
- **[corrected reasoning]** Keeping 18 layers adds **negligible** new convergence risk **because we are *not increasing* depth** — 18L is firmly within the routinely-trainable range, well below the extreme-depth regime where attention-entropy/rank collapse appears (ViT-22B, Dehghani 2023, [arXiv:2302.05442](https://arxiv.org/abs/2302.05442); *Depth Delusion* 2026). The only thing we cut, **width, is the benign/easy-to-optimize axis.** (The earlier phrasing "depth-fragility makes 18L zero risk" was muddled — depth fragility is a reason not to *add* depth, which we don't.)
- **head_dim stays the standard 64** (896 / 14 heads). ✓ canonical since Vaswani 2017; not a bottleneck at this scale.
- Data-justified: ~1.31× images/param @40ep on 6.3M (vs V1's 0.72×). Genuine 23% cut, well above the **160M floor**; **reject shallow-wide 16×1024.**

**[corrected] The "separability bottleneck" fear at width 896 is overstated.**
- Width 896 for 19,294 labels sits in the well-studied **K≫d regime**, which **Generalized Neural Collapse** (NeurIPS 2023, [arXiv:2310.05351](https://arxiv.org/abs/2310.05351)) shows supports good class separation; required width grows only **sub-linearly** with class count ([arXiv:2501.02364](https://arxiv.org/abs/2501.02364)). Real ViT/ConvNeXt taggers and CLIP heads operate at d≈768–1024 over tens of thousands of classes.
- **Drop the "(DeiT III separability argument)" citation** — DeiT III contains no such theorem; the attribution was wrong.
- To fully de-risk, a **Query2Label / ML-Decoder** cross-attention head ([arXiv:2107.10834](https://arxiv.org/abs/2107.10834); Ridnik WACV 2023) removes the single-`896→19,294`-linear-head dimensionality limit entirely. It is **new code (changes the ONNX/export path)** — prototype before committing the param budget. **Not needed at the 896w default**; revisit only if width is cut to 768.

**Open architectural question (not settled).** The arch-sizing lens favored deep-narrow `832×22` (~200M; SoViT "depth scales ~2× width" — verified exponents s_depth≈0.45, s_width≈0.22). **[corrected caveat]** Those exponents were derived **IsoFLOP-optimal on clean JFT-3B** (pretrained, FLOP-bound) — **out-of-regime** for our 6.3M noisy, VRAM-bound, from-scratch setting. The **direction** (deep-narrow ≥ shallow-wide; reject 16×1024) holds; the precise **2× ratio does not transfer**, so don't treat it as a target. 2026 work (*Depth Delusion*, [arXiv:2601.20994](https://arxiv.org/abs/2601.20994)) even argues width is the *safer* axis, mildly tensioning deep-narrow. **Net: the conservative default keeps width 896 / depth 18.** `832×22` remains defensible if you want to lean into capacity-efficiency, accepting the deeper-from-scratch optimization risk and thinner evidence. **No scaling law is validated on few-million-image noisy multi-label data — `896×18` is principled extrapolation, not SOTA-validated.**

## 2.1 Detail lever — KEEP patch16 (patch14 refuted)

patch16→14 was **refuted** by a 3-lens adversarial research pass and re-confirmed here:
- **Tiny, clean-label-only gain.** **[corrected magnitude]** *Scaling Laws in Patchification* ([arXiv:2502.03738](https://arxiv.org/abs/2502.03738), 2025, Table 1) shows the 16→8 step buys **~+1.3–1.7pp** ImageNet top-1 (not the +0.3pp this plan previously claimed), and the curve **does not saturate below 16** — it diminishes monotonically to pixel tokenization. *But the conclusion survives:* patch14 adds only **~31% tokens** (785→1025; vs +300% for 16→8), so it captures only a small fraction (~**+0.2–0.5pp** clean-label, extrapolated), which the missing-positive noise ceiling then discounts toward ~0.
- **patch14 adds ZERO real pixels.** Preprocessing is **downscale-only letterbox** ([dataset_loader.py:282-306](../dataset_loader.py#L282-L306)); patch14 is a finer mesh over the *same* raster. (Note: the downscale is now **LANCZOS** [corrected — see §4/§5], which preserves near-Nyquist detail better than the old bilinear, so the "already-blurred" framing is weaker — but the "no new pixels" point stands on downscale-only grounds alone.)
- **Cost is superlinear.** **[corrected precision]** the **attention term** is ~1.71× (quadratic, (1025/785)²); the **linear** MLP/proj term is ~1.31× (fix the old "1.36×"). End-to-end at 896w/18L the MLP term is comparable, so realized cost is **~1.4–1.6×**, not a flat 1.7×. patch14 also forces a stem re-init (no warm-start) and breaks the 320px (320/14=22.86) and 512px (512/14=36.6) grids.
- **V1 was not detail-limited** (uniform plateau across frequency buckets; rare bucket still climbing — the missing-positive signature, not a detail bottleneck). More tokens = more effective capacity = **faster FN memorization** (Kim 2022) — wrong direction under missing-positive noise.

**If real detail is ever wanted, the lever is `512px / patch16`, not patch14.** Same ~1025-token budget but **genuinely more real pixels** (sources are high-res), keeps the stem **warm-startable** (only pos-embed interpolation needed — FlexiViT, [arXiv:2212.08013](https://arxiv.org/abs/2212.08013), exists precisely because a patch-*size* change needs PI-resize/retraining), reuses the existing bicubic pos-embed interp (20×20→32×32 is clean), integer grids throughout. This is the **optional Phase 3** (§3), **gated on a fine-bucket-specific plateau** (rare/fine tags lagging head buckets) — not the uniform label-noise plateau V1 had.

*Aside:* 2024–2026 SOTA for "wanting detail" has moved to **content-adaptive tokenization** (APT [arXiv:2510.18091](https://arxiv.org/abs/2510.18091); Grc-ViT [arXiv:2511.19021](https://arxiv.org/abs/2511.19021)) and **native/variable resolution** (NaViT, NeurIPS 2023, [arXiv:2307.06304](https://arxiv.org/abs/2307.06304)) — all of which argue *against* a uniformly finer fixed patch14 mesh. No located study validates patch size under missing-positive noise (open gap; all cited gains are clean-label upper bounds).

## 2.2 Forward-compat (anticipated larger dataset)

Grow by **adding layers at 896 width** (896×18 → 896×24 → only then widen toward 1024). Low-risk (more identical blocks; pos-embed/patch-embed/head geometry untouched, head_dim stays 64). Narrow-but-deep taxes vocabulary growth less (~1.0M params per +1,300 tags at 896w vs ~1.3M at 1024w). **Gate param re-growth on effective *clean-label* signal, not raw image count.** Rough triggers (plan's own ~1.5–2.0× @40ep exposure basis): ~9–10M clean images → ~248M; ~12–13M → ~300M; ~16–20M → ~320M.

---

# 3. Schedule — converge Phase 1 (the biggest single fix)

V1's root failure: Phase 1 stopped at **33/40 epochs ≈ 0.72× images/param exposure**, which cascaded into the Phase-2 regularization mess. **[corrected framing]** the "images/param exposure floor" is an **internal sanity heuristic, not a citeable law** — Zhai et al., *Scaling Vision Transformers* (CVPR 2022, [arXiv:2106.04560](https://arxiv.org/abs/2106.04560)) actually find ViT scaling is **model-bound** and big models are *more* sample-efficient, so the real V1 risk was **under-compute + label noise**, not literally too-few-images-per-param. Keep the ~1.8–2.0× target as a "train long enough to not be compute-limited" guardrail, **not** a derived constant — and note **more exposure will not break the label-noise ceiling** by itself (that's §1's job).

Per-phase schedule for the ~192M default (base LR shown for 896w). **At every resolution switch:** interpolate pos-embeds (bicubic), reset optimizer state, short re-warmup.

| Phase | Res | Epochs | Eff. batch | Warmup | Base→Peak LR | drop_path | WD | Loss |
|---|---|---|---|---|---|---|---|---|
| **1 (from-scratch)** | 320 | **40, run to a genuine plateau** | ~1024 | 4 ep | 2.7e-4 → ~5.4e-4 | 0.25 | 0.08 | ASL γ_neg=**7**, clip≥0.2, γ_pos=0 |
| **2 (fine-tune)** | 448 | ~12 | ~768 | 2 ep | 1.5e-5 → ~2.6e-5 | **0.15** | **0.04** | ASL γ_neg **swept 7→5(→4)** (clean-slice gated), clip≥0.2, γ_pos=0 |
| 3 (optional detail) | **512 / patch16** | 3–4 | — | 1 ep | 8e-6 → ~1.3e-5 | 0.10 | 0.03 | unchanged |

> The table is the at-a-glance summary. The **full, runnable per-phase config** is in §3.1–§3.5 below. Firm *rules* (resolution, LR-coupling, loss schedule, transition steps) are distinguished from *VRAM-dependent starting points* (batch / grad-accum) that must be tuned to the GPU on day one.

### 3.1 Loss schedule across phases (the rule)

**One loss, one schedule.** ASL throughout; the only thing that changes across phases is **γ_neg** (clip and γ_pos are fixed). Phase 1 keeps γ_neg **high** because a from-scratch model is too uncalibrated to lower it safely (§1.3 gradient-balance caveat — dropping γ_neg early flips the gradient balance toward negatives and suppresses the rare positives). Phase 2 **lowers** γ_neg to recover rare-positive recall, **gated on the clean slice**.

| | Phase 1 (from scratch) | Phase 2 (mature) | Phase 3 |
|---|---|---|---|
| **ASL** (§1.3) | γ_neg=**7**, clip **0.2**, γ_pos=0 | γ_neg **swept 7→5(→4)** (gated on clean-slice recall/FP), clip **≥0.2 (NOT 0.05)**, γ_pos=0 | unchanged from P2 |

The γ_neg schedule **descends** (7→5→4) as cleaning lowers the missing-positive rate — the **reverse** of V1's ascending 4→7. Two hard rules: **do not lower γ_neg in Phase 1**, and **do not lower clip in any phase.** Hold γ_neg at 7 if the clean slice shows no benefit from the down-step.

### 3.2 Phase 1 — 320px from-scratch (full config)

**Goal:** learn coarse features from scratch; **run to a genuine mAP plateau, not a fixed epoch** (V1's fatal mistake — it stopped at 33/40). Gate on clean-slice mAP growth <1.005×/epoch for 2 consecutive epochs.

| Config key | Value | Source / why |
|---|---|---|
| `model.hidden_size / num_attention_heads / num_hidden_layers / intermediate_size` | 896 / 14 / 18 / 3584 | §2 (head_dim 64) |
| `data.image_size` | **320** | 20×20=400 patches +CLS = 401 tokens; ~2× faster/step than 448 |
| `data.batch_size` | **~96** *(start; tune to VRAM)* | V1 P1 ran 96 ([unified_config.yaml:96](../configs/unified_config.yaml#L96)); 896w is narrower → ≥ that headroom. OOM → lower batch, raise grad-accum to **hold eff batch** |
| `training.gradient_accumulation_steps` | set so batch×accum ≈ **1024** (e.g. 96×11) | eff batch drives LR |
| `training.learning_rate` (base) | **2.7e-4** | muP: narrower→slightly higher than V1's 2.5e-4 (a *prior*, validate; §3.6) |
| peak LR (derived) | **~5.4e-4** | **Rule:** peak = base × √(eff/256) = 2.7e-4 × √(1024/256) = ×2 (Malladi, [arXiv:2205.10287](https://arxiv.org/abs/2205.10287)) |
| `training.scheduler / lr_end` | cosine / 1e-6 | single cosine decay |
| `training.warmup_epochs` | **4** | ~10% of 40, linear from 1e-6 (DeiT range 1.7–17%) |
| `training.num_epochs` | **40 (cap; gate on plateau)** | run to plateau; target ~1.8–2.0× total exposure across P1+P2 |
| `model.drop_path_rate` | **0.25** | strong reg from-scratch; below DeiT III ViT-L's 0.4 because this model is smaller |
| `model.hidden_dropout_prob / attention_dropout` | **0.10 / 0.05** | V1 P1 values ([unified_config.yaml:35,37](../configs/unified_config.yaml#L35)) |
| `training.weight_decay` | **0.08** | |
| `training.adam_beta2 / adam_epsilon` | 0.999 / 1e-7 | DeiT convention; ε safe for bf16 (and ε→ε/√κ under sqrt scaling, §3.6) |
| **loss** | **ASL** γ_neg=**7** / clip 0.2 / γ_pos=0 (keep γ_neg high from-scratch — §1.3) | §1.3, §3.1 |
| early stopping | **disabled** (or patience high) | transitions on plateau regardless |

**Phase-1 augmentation — full strength** (config lines [134–214](../configs/unified_config.yaml#L134-L214)):
- horizontal flip 0.5 + orientation-aware tag swap
- color jitter brightness **0.30** / contrast **0.20** / saturation **0.08** @ p=0.5 (DeiT III `--color-jitter 0.3`; saturation held ≈¼ per BYOL; §4)
- random rotation ±[2°,8°] @ p=**0.50**, bicubic
- gaussian blur p=**0.30**, kernel 3, σ∈[0.1,**1.5**] (DeiT-III-derived blur-only; §4)
- **no** mixup / cutmix / randaugment / random-erasing / hue rotation

### 3.3 Phase 1 → 2 transition checklist (do every step, in order)

1. **Select the best Phase-1 checkpoint on CLEAN-slice mAP** (§1.5) — *not* noisy-val mAP (it mis-ranks).
2. **Interpolate position embeddings** bicubic **20×20 → 28×28** ([training_utils.py:1787-1841](../training_utils.py#L1787-L1841), already implemented). (Patch16 unchanged, so the conv stem warm-starts directly — only pos-embed needs interpolation.)
3. **Reset optimizer state** (stale Adam 2nd-moment from 320 causes instability — generic phase-transition hygiene, §3.6).
4. **Update config:** `image_size=448`, batch/grad-accum (§3.4), `num_epochs≈12`, `warmup_epochs=2`, `drop_path_rate=0.15`, `weight_decay=0.04`, dropouts (§3.4), reduce augmentation (§3.4), and **begin the ASL γ_neg down-sweep (7→5(→4)), gated on clean-slice recall/FP** (§3.1) — only if the clean slice exists (§1.5); otherwise hold γ_neg=7.
5. **Re-warmup 2 epochs**; `torch.compile` auto-recompiles on first forward (sequence length 401→785).

### 3.4 Phase 2 — 448px fine-tune (full config)

**Goal:** fine-grained spatial detail **+ recover missing positives by lowering γ_neg (7→5(→4), gated on the clean slice)**. Because Phase 1 *actually converged this time*, reg reduction is now **legitimate** (FixRes/DeiT III) — the reduction V1 was forced to abandon.

| Config key | Value | Source / why |
|---|---|---|
| `data.image_size` | **448** | 28×28=784 patches +CLS = 785 tokens |
| `data.batch_size` | **~48–56** *(start; tune to VRAM)* | V1 P2 ran 48 at 1024w; 896w narrower → 56 plausible |
| `training.gradient_accumulation_steps` | set so batch×accum ≈ **~768** | |
| `training.learning_rate` (base) | **1.5e-5** | continuation ≈ 2.8% of P1 peak ([corrected] not "1/300 DeiT III"; §3.6) |
| peak LR (derived) | **~2.6e-5** | **Rule:** 1.5e-5 × √(768/256) ≈ ×1.73 |
| `training.warmup_epochs` | **2** | re-warmup at resolution switch |
| `training.num_epochs` | **~12 (gate on clean-slice mAP plateau)** | |
| `model.drop_path_rate` | **0.15** | legitimate reduction (P1 converged) |
| `model.hidden_dropout_prob / attention_dropout` | **0.05 / 0.02** | reduced (legit now; V1 was forced to hold 0.10/0.05) |
| `training.weight_decay` | **0.04** | |
| **loss** | **ASL** γ_neg **swept 7→5(→4)** (gated on clean-slice recall/FP), clip≥0.2, γ_pos=0 | §1.3, §3.1 |
| **early stopping** | **on `val/mAP` (clean slice)**, patience 4, burn_in 2, threshold 1e-3 | §6 item 2; F1 is calibration-floored |

**Phase-2 augmentation — reduced** (now a *legitimate* FixRes/DeiT III reduction, not V1's forced hold):
- horizontal flip 0.5 + tag swap (lossless, unchanged)
- color jitter brightness **0.22** / contrast **0.15** / saturation **0.06**
- random rotation ±[2°,5°] @ p=**0.30**
- gaussian blur p=**0.15**, σ∈[0.1,**1.0**]
- **no** mixup / cutmix / randaugment / random-erasing / hue rotation

### 3.5 Phase 3 — 512px / patch16 (OPTIONAL, gated)

**Only run this if** a converged Phase-2 run (after the γ_neg down-sweep) shows a **fine-bucket-specific plateau** (rare/fine tags lagging head buckets) — *not* the uniform label-noise plateau (§2.1). Transition mirrors §3.3 but interpolates pos-embed **28×28 → 32×32** (512/16=32, integer grid). 3–4 epochs, warmup 1, drop_path 0.10, WD 0.03, LR 8e-6→~1.3e-5, **loss unchanged from P2** (ASL at the P2 final γ_neg, clip≥0.2, γ_pos=0). **Drop it if it OOMs or the plateau stays uniform.**

**Cross-phase notes / corrections:**
- **Gate the Phase 1→2 transition on a real plateau** (mAP growth <1.005×/epoch for 2 consecutive epochs), **not a fixed epoch count** — the precise inverse of V1's mistake. Target ~1.8–2.0× total exposure (~50–56 epochs; the smaller/faster model makes the extra epochs affordable).
- Because Phase 1 now actually converges, **Phase 2 can *legitimately* reduce drop_path (0.25→0.15) and WD** per the FixRes/DeiT III reduce-reg-at-fine-tune rule — the reduction V1 was forced to abandon. (V1 had to *hold* reg high because its base was under-converged → the EfficientNetV2 "reg-up with resolution" regime applied instead; see appendix.) A right-sized 192M model also needs **less** drop_path than ViT-L.
- **LR scaling [confirmed, with nuance].** Use **sqrt** scaling for AdamW: LR ∝ √(batch/256) (Malladi et al., NeurIPS 2022, [arXiv:2205.10287](https://arxiv.org/abs/2205.10287)) — linear is SGD-only. The full rule **also rescales ε → ε/√κ** (and betas); for κ≈4 the prescribed ε is halved, consistent with AdamW8bit's small ε. The `/256` base is a free reference — keep it consistent with where you actually tuned the base LR.
- **muP/width [confirmed direction, softened].** A narrower 896 model takes a *slightly higher* base LR than the 1024 prior (~1024/896 ≈ 1.14×). Treat the **1.14× as a prior, not a constant** — under proper muP the optimal LR is width-*invariant*, and LR transfer hinges on **decoupled weight decay** ([arXiv:2510.19093](https://arxiv.org/abs/2510.19093)); since we run a single global LR (SP), validate the bump rather than deriving it.
- **Fine-tune LR ratio [corrected].** Don't cite "DeiT 1/200, DeiT III 1/300" — those are loose/not apples-to-apples (DeiT higher-res finetune is ~1/50–1/100 of pretrain base; DeiT III's "1/300" mixes a LAMB pretrain with an AdamW finetune). Frame Phase 2 as a **progressive-resolution continuation** (~1–3% of peak), which **1.5e-5 / 5.4e-4 ≈ 2.8% satisfies.**
- **Resolution-switch attribution [corrected].** **Position-embedding bicubic interpolation is from the ViT paper** (Dosovitskiy 2021, [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)) / DeiT — **not FixRes.** FixRes (Touvron 2019, [arXiv:1906.06423](https://arxiv.org/abs/1906.06423)) is a **BatchNorm CNN** method (low-LR last-layer/BN re-adaptation; no pos-embeds), and its BN mechanism **does not transfer** to our LayerNorm ViT. Cite FixRes only for the high-level finding "train low-res, fine-tune at target res helps." Optimizer-state reset and re-warmup are **generic phase-transition hygiene**, not FixRes prescriptions — both still correct to do.
- **LR guard:** if grad_norm drifts >2× in the first 5 epochs, fall back to base 2.5e-4 (peak ~5.0e-4). **No QK-norm in the code, so do not chase DeiT III's 3e-3.**
- **Forward note — QK-norm is the SOTA fix, not a low LR cap.** The 2024–2026 move (Wortsman 2024, [arXiv:2309.14322](https://arxiv.org/abs/2309.14322); *ViT-5* 2026, [arXiv:2602.08071](https://arxiv.org/abs/2602.08071)) is to **add QK-norm** — it suppresses attention-logit-growth instability and **flattens LR sensitivity across orders of magnitude**, letting you safely run a higher peak LR / shorter schedule. Cheap, ~no quality cost, often a small speedup. **Consider QK-norm (± RMSNorm/LayerScale) as a high-value, low-risk V2 arch add** rather than permanently capping LR. Do not jump LR 5× without it.

---

# 4. Augmentation — corrected justifications

The augmentation *choices* are sound; two *justifications* were wrong and are fixed here.

- **Horizontal flip:** 50% with orientation-aware tag swapping (lossless, unchanged).
- **Exclude naive Mixup/CutMix/RandAugment** — **[corrected justification]**: do **NOT** cite DeiT/DeiT III for this (DeiT III *uses* mixup 0.8 + cutmix 1.0). The exclusion is a **domain-specific deviation** justified by 2024 multi-label work: naive CutMix erases/adds labels (LogicMix-class label noise — [arXiv:2405.13451](https://arxiv.org/abs/2405.13451)) and naive Mixup **injects false-negatives** in partial-label training ([arXiv:2405.15860](https://arxiv.org/abs/2405.15860)) — i.e. it actively amplifies our exact missing-positive failure mode. *Future lever:* **label-aware mixing** (LogicMix logical-OR; CutMix-with-label-propagation) is the principled way to re-introduce mixing if a reg lever is wanted — it **fights** the missing-positive ceiling rather than worsening it.
- **DeiT-III-derived blur-only aug** — **[corrected label]**: DeiT III "3-Augment" (grayscale / solarization / Gaussian blur) beats RandAugment for from-scratch ViT (ECCV 2022, [arXiv:2204.07118](https://arxiv.org/abs/2204.07118), Table 3, +0.3–0.4pt). We **drop grayscale + solarization** (they destroy color-tag signal) and keep **only blur** — so call it "DeiT-III-derived blur-only," and **don't claim 3-Augment's measured +0.4pt** (that was the full 3-transform set).
- **Color jitter** — brightness 0.30 (DeiT III `--color-jitter 0.3`); saturation held ≈¼ (BYOL asymmetric design, Grill 2020, [arXiv:2006.07733](https://arxiv.org/abs/2006.07733)). **[corrected — fabricated claim removed]**: **delete** "Kirichenko: raising frequency is safer than magnitude" — *that claim is not in the paper.* Kirichenko et al. (NeurIPS 2023, [arXiv:2401.01764](https://arxiv.org/abs/2401.01764)) actually find color-jitter/RandAugment **disproportionately harm color/texture-distinguished class pairs** and recommend **class-conditional *magnitude* reduction** on the affected (color-named) classes. Cite **Planckian Jitter** (ICLR 2023, [arXiv:2202.07993](https://arxiv.org/abs/2202.07993)) for the "color jitter destroys color discrimination" mechanism, and consider it as a **color-tag-safe partial substitute** for color jitter.
- **Gaussian blur** — kernel=3, σ≤1.5 (Phase 1) — **[corrected status]**: these are **engineering defaults, not literature constants** (DeiT III's reference σ range is [0.1, 2.0]; ≤1.5 is the conservative, fine-detail-preserving side). Drop any implication that "frequency over intensity" is a cited principle — it's a domain heuristic.
- **EfficientNetV2 reg-up [confirmed].** For *from-scratch progressive* training, regularization scales **up** with resolution (Tan & Le, ICML 2021, [arXiv:2104.00298](https://arxiv.org/abs/2104.00298)) — opposite of the FixRes/DeiT III reduce-reg rule, which applies only to a **short fine-tune of an already-converged base**. Since V2's Phase 1 *will* converge, Phase 2 can reduce reg legitimately (§3). **Caution:** EfficientNetV2 scales **mixup** up as part of reg-up; since we exclude mixup, our reg-up lever is restricted to **dropout + stochastic-depth + blur/jitter** — make sure there's enough non-mixup reg headroom at 448.
- No random erasing; no hue rotation. **Class-conditional aug** (Kirichenko 2023) and **ForAug** ([arXiv:2503.09399](https://arxiv.org/abs/2503.09399), 2025, occlusion-free fg/bg recombination) are optional better-targeted levers.

---

# 5. Config deltas vs the live `unified_config.yaml`

> **[corrected]** The live config is the **Phase-2 (448px) V1 fine-tune** state. Several "current" values in the old plan table were stale. These are the **verified current** values.

| Parameter | Live now (V1 P2) | V2 default | Note |
|---|---|---|---|
| `model.hidden_size` | 1024 | **896** | width cut; set heads=14 (head_dim 64) |
| `model.num_attention_heads` | 16 | **14** | 896/14 = 64 |
| `model.num_hidden_layers` | 18 | **18** | unchanged (keep depth) |
| `model.intermediate_size` | 4096 | **3584** | MLP ratio 4 × 896 |
| `model.patch_size` | 16 | **16 (unchanged)** | patch14 refuted; 512px/patch16 is the conditional detail lever |
| `data.image_size` | 448 | 320 P1 / 448 P2 (/512 P3 cond.) | progressive |
| `data` resampler | **LANCZOS (already landed)** | LANCZOS | **[corrected]** the BILINEAR→LANCZOS switch is **DONE** at [dataset_loader.py:296](../dataset_loader.py#L296); not a pending win. Match the **inference/serving** preprocessor to LANCZOS to avoid train/serve skew. |
| `data.batch_size` | **48** | per-phase (P1 ~96) | (old plan said 58 — stale) |
| `training.num_epochs` | **15** | per-phase (P1 40) | (old plan said 36 — stale) |
| `training.warmup_epochs` | **2** | 4 (P1) / 2 (P2) | (old plan said 10 — stale) |
| `training.learning_rate` | 1.0e-5 (P2) | 2.7e-4 P1 / 1.5e-5 P2 | per-phase; narrower → slightly higher (prior, validate) |
| `training.tag_loss.loss_type` | **(field does not exist)** | **(not added)** | ASL (`AsymmetricFocalLoss`) stays the criterion — no new `loss_type`/module needed |
| `training.tag_loss.gamma_neg` | 7.0 | **P1 = 7.0; P2 sweep 7→5(→4)** (clean-slice gated) | live 7.0 already = the Phase-1 target; lower only in P2 (§1.3, §3.1) |
| `training.tag_loss.clip` | **0.2** | **keep ≥0.2** (do NOT lower) | **[corrected]** old plan's 0.2→0.05 was backwards (§1.3) |
| `training.tag_loss.gamma_pos` | 0.0 | **0.0** (no A/B) | ASL optimal; raising it hurts mAP (§1.3) — keep pinned at 0 |
| early-stop metric | `val/f1_macro` | **`val/mAP`** | F1 is calibration-floored; mAP is ranking-based |
| `training.early_stopping_threshold` | 5e-7 | **1e-3** | mAP-scale margin |

---

# 6. Implementation — code work required

> **[corrected line numbers]** — verified against the live repo this pass.

**Already done / no code needed:**
- **Resampler:** LANCZOS downscale already live at [dataset_loader.py:296](../dataset_loader.py#L296) (BICUBIC for rotation at [:180](../dataset_loader.py#L180)/[:330](../dataset_loader.py#L330)). ✓
- **Pos-embed bicubic interpolation:** already implemented at [training_utils.py:1787-1841](../training_utils.py#L1787-L1841) (interpolate call at :1832-1833). ✓
- **Architecture/width/depth/patch are config-driven** — `model_architecture.py` needs no change. ✓
- **`label_smoothing` "leak" — non-issue [corrected].** The old claim was wrong: [loss_functions.py:269](../loss_functions.py#L269) snapshots a **clean `targets_for_focal`** *before* smoothing, the focal pos/neg weights use that clean tensor ([:314-315](../loss_functions.py#L314-L315)), and smoothing feeds **BCE only** ([:278](../loss_functions.py#L278)). It is also `0.0` in config, so it's a no-op. (Cosmetic only: the `__init__` default at [loss_functions.py:45](../loss_functions.py#L45) is still 0.05 — consider matching it to 0.0.)

**Loss work — none (config-only).** ASL (`AsymmetricFocalLoss`) is **already** the live criterion ([train_direct.py:694-705](../train_direct.py#L694-L705) builds it from `tag_loss_cfg`), so the entire loss plan is **config edits to `training.tag_loss`** — γ_neg per phase, clip≥0.2, γ_pos=0. **No new loss module, no `loss_type` dispatch, no `current_epoch`/`set_epoch` plumbing** (those were only needed for an SPLC flip, which is dropped — §1.2). The γ_neg phase sweep is realized by setting `training.tag_loss.gamma_neg` per phase; stepping it inside Phase 2 (7→5→4) can ride the existing per-phase config-reload boundary rather than a code change.

**Net-new work for instrumentation + metric swap (§1.4, §1.5, §5):**
1. **H5 validation-loop hook** to retain top-K logits and compute the missing-positive scalars; `pred_pos_ratio` is ~free.
2. **Move auto-stop `val/f1_macro` → `val/mAP`.** The trigger is at [train_direct.py:2384-2386](../train_direct.py#L2384-L2386); best-metric/patience bookkeeping at [:2305-2306](../train_direct.py#L2305-L2306) and [:2346-2353](../train_direct.py#L2346-L2353). **[corrected]** the 0.2653 F1 threshold is **not** hard-coded in `train_direct.py` — it is **config-driven** ([train_direct.py:1317-1318](../train_direct.py#L1317-L1318) reads `threshold_calibration.default_threshold`; literal in [unified_config.yaml:330](../configs/unified_config.yaml#L330)/[:348](../configs/unified_config.yaml#L348) and [evaluation_metrics.py:484](../evaluation_metrics.py#L484)). Thread mAP through the best-metric comparison at :2305/:2346 — **not** through `find_optimal_threshold` ([evaluation_metrics.py:216-264](../evaluation_metrics.py#L216-L264)), which **explicitly rejects mAP** as threshold-independent. Bump `early_stopping_threshold` 5e-7→1e-3 ([unified_config.yaml:319](../configs/unified_config.yaml#L319)). Field is named `early_stopping_burn_in_epochs` (not `burn_in`). `ThresholdCalibrator` is at [evaluation_metrics.py:469](../evaluation_metrics.py#L469)-~640.
3. **Build the clean val slice (H4)** — labeling effort, not code; gates 1 and 2's interpretation **and the entire γ_neg sweep** (§1.3/§3.1 are unmeasurable without it).

---

# 7. What to monitor (V2)

- **`val/mAP` on the CLEAN slice** — primary progress + early-stop signal (noisy-val mAP under-estimates and mis-ranks; §1.5).
- **Frequency-bucketed mAP** — rare-tag buckets (<1000 occurrences); a *fine-bucket-specific* plateau (rare/fine lagging head) is the only trigger for the optional 512px Phase 3. A *uniform* plateau = label-noise ceiling (§1).
- **Missing-positive diagnostics** (§1.4) — `pred_pos_ratio`, `mean_sigmoid_topK_unlabeled`, `cooccur_jaccard@K`, `rare_bucket_ECE` (smoothed), `logit_std_rank_11_50`, and the **γ_neg-sweep recall/FP** on the clean slice (gates the P2 down-step).
- **Phase-1 convergence** — gate the 320→448 transition on a genuine plateau, not an epoch count.
- **Stability** — grad_norm drift, logit-distribution drift, NaN/Inf flags (turbulence expected for 1–2 epochs after each resolution switch).

---

# 8. Scholarly basis (key papers, V2-relevant)

| Topic | Paper | Use |
|---|---|---|
| **Loss (primary) — ASL** | Ridnik et al., ICCV 2021, [arXiv:2009.14119](https://arxiv.org/abs/2009.14119) | The loss we ship: γ_pos=0; focal margin (clip) **0.3–0.4** (Fig 7) → keep clip≥0.2; γ_neg sweet-spot **2–4** is the **Phase-2** target — keep γ_neg=7 from-scratch (P1) then sweep **down** toward it (§1.3) |
| Missing-positive loss (considered, **not adopted**) | Zhang et al., Hill/SPLC, [arXiv:2112.07368](https://arxiv.org/abs/2112.07368) | Stronger semi-hard-band tool on paper (beats ASL ~2.5–3 mAP @COCO-40%) but unvalidated at 19K/0.6%/from-scratch; documented fallback (§1.2) |
| Long-tail robust loss (**not adopted**) | Park et al. RAL, ICCVW 2023, [arXiv:2308.05542](https://arxiv.org/abs/2308.05542); APL [arXiv:2304.05361](https://arxiv.org/abs/2304.05361) | Polynomial-robust Hill extensions for the 19K tail — out of scope (we run ASL, not Hill); next-loss-family direction if ASL plateaus (§1.6) |
| Loss generalization (SPML, context-only) | GR Loss, IJCAI 2024, [arXiv:2405.03501](https://arxiv.org/abs/2405.03501) | soft generalization of an SPLC-style flip; SPML-specific — not adopted (§1.6) |
| **Wrong-positive / adjacent-class correction** | Xia et al. **HLC**, ICCV 2023 ([paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Xia_Holistic_Label_Correction_for_Noisy_Multi-Label_Classification_ICCV_2023_paper.pdf), [code](https://github.com/xiaoboxia/HLC)); cleanlab-ML [arXiv:2211.13895](https://arxiv.org/abs/2211.13895) | pairflip correction w/ co-occurrence (no hard exclusivity) vs golden-anchored CL detect-remove — the wrong-positive half (§1.7) |
| Early-learning / memorization | Kim CVPR 2022 [arXiv:2206.03740](https://arxiv.org/abs/2206.03740); ELR NeurIPS 2020 [arXiv:2007.00151](https://arxiv.org/abs/2007.00151); Arpit ICML 2017 [arXiv:1706.05394](https://arxiv.org/abs/1706.05394) | the FN-memorization mechanism (§1.1) |
| Noisy-val bias | Zhao & Gomes [arXiv:2102.08427](https://arxiv.org/abs/2102.08427); Northcutt NeurIPS 2021 [arXiv:2103.14749](https://arxiv.org/abs/2103.14749) | clean slice for selection/early-stop (§1.5) |
| Label completion teachers | RAM/Tag2Text [arXiv:2306.03514](https://arxiv.org/abs/2306.03514); CDUL [arXiv:2307.16634](https://arxiv.org/abs/2307.16634); cleanlab-ML [arXiv:2211.13895](https://arxiv.org/abs/2211.13895) | propose missing positives, gate on clean-slice precision |
| Data curation vs scale | Scaling Laws for Data Filtering, CVPR 2024, [arXiv:2404.07177](https://arxiv.org/abs/2404.07177) | complete labels > add noisy images |
| ViT scaling shape | SoViT, NeurIPS 2023, [arXiv:2305.13035](https://arxiv.org/abs/2305.13035); Zhai CVPR 2022 [arXiv:2106.04560](https://arxiv.org/abs/2106.04560) | direction only (out-of-regime exponents); model-bound scaling |
| Separability at K≫d | Generalized Neural Collapse, NeurIPS 2023, [arXiv:2310.05351](https://arxiv.org/abs/2310.05351) | width 896 for 19K classes is fine |
| Patch/detail | Scaling Laws in Patchification, [arXiv:2502.03738](https://arxiv.org/abs/2502.03738); FlexiViT [arXiv:2212.08013](https://arxiv.org/abs/2212.08013) | keep patch16; 512/patch16 is the detail lever |
| LR / optimizer | Malladi NeurIPS 2022 [arXiv:2205.10287](https://arxiv.org/abs/2205.10287) (sqrt); Wortsman 2024 [arXiv:2309.14322](https://arxiv.org/abs/2309.14322) (QK-norm) | sqrt scaling; QK-norm as a future LR-stability add |
| Resolution adaptation | ViT [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) (pos-embed interp); FixRes [arXiv:1906.06423](https://arxiv.org/abs/1906.06423) (low-res→high-res helps) | correct attribution (§3) |
| Augmentation | DeiT III ECCV 2022 [arXiv:2204.07118](https://arxiv.org/abs/2204.07118); EfficientNetV2 [arXiv:2104.00298](https://arxiv.org/abs/2104.00298); Kirichenko NeurIPS 2023 [arXiv:2401.01764](https://arxiv.org/abs/2401.01764); LogicMix [arXiv:2405.15860](https://arxiv.org/abs/2405.15860); Planckian Jitter [arXiv:2202.07993](https://arxiv.org/abs/2202.07993) | blur-only, reg-up, class-conditional, no naive mixup |

---
---

# Appendix — V1 history (footnotes)

> V1 is no longer the plan. This is kept only so V2's decisions are traceable. The full original V1 document (Phase-1/Phase-2 tables, the EfficientNetV2 reversal saga, the H1–H5 hedge table, the per-epoch monitoring runbook) lives in git history and in [TRAINING_HEALTH_TRACKER.md](../TRAINING_HEALTH_TRACKER.md).

**What V1 was.** A ViT-L/16-class model, **1024w × 18L, ~248M total**, trained from scratch on 5.4M images / ~19K tags, progressive 320→448. Loss: pure ASL, `gamma_neg=4` (P1) → `7` (P2), `gamma_pos=0`, `clip=0.2`, no class weights.

**What happened.**
- **Phase 1 stopped under-converged at 33/40 epochs (~0.72× images/param exposure).** This is the root failure V2 fixes by gating the phase transition on a real plateau (§3).
- Because the base was under-converged, the planned FixRes/DeiT III "reduce regularization at fine-tune" recipe was **illegitimate** (it assumes a *converged* base). The **EfficientNetV2** "reg-up with resolution" regime applied instead, so Phase 2 **held regularization high** (drop_path 0.20, dropouts unchanged, WD 0.05, color jitter ¼-cut, rotation/blur kept reduced) rather than slashing it.
- **Phase 2 early-stopped at a label-noise ceiling: mAP 0.652 → ~0.674 by E5**, then mAP growth flattened while val loss kept falling (no overfitting reversal). The wall was **missing-positive label noise, not capacity or over/under-fitting.**

**The P2 label-noise review (2026-05-06)** flagged that `gamma_neg=7` + reduced regularization is the configuration most exposed to **missing-positive bias** (Kim 2022; Park RAL 2023; Zhang Hill/SPLC 2021; Zhao & Gomes 2021). In the end P2 ran with reg *held high* (a stronger form of the review's H2 hedge), so it was *less* exposed than the worst case — but it still hit the ceiling. The review's conclusion is what drives V2 §1: the **γ_neg schedule** (keep high from-scratch, sweep down on a clean slice) is the loss lever carried forward (the review's **H3 — Hill/SPLC — was evaluated and *not* adopted**; §1.2), plus the H4 clean slice and H5 diagnostics that were never implemented for V1.

**Lessons carried into V2:**
1. The ceiling is **label noise** → attack it with loss + data, not capacity (§0, §1).
2. **Converge Phase 1** before the resolution switch (§3).
3. **Instrument missing-positive bias** and **measure true mAP on a clean slice** — V1's ceiling was unfalsifiable without them (§1.4, §1.5).
4. Right-size the model (the 248M headroom was unusable) — but expect **0 mAP** from the cut itself (§2).
