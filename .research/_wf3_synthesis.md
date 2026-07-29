# 1. DIRECT VERDICT

**YES-BUT.** Training a fresh, tiny ordinal head on very-clean tags and ranking the dirty GT by its disagreement WILL surface your M4 hair-length errors to the top of the pipeline2 queue and make the long→very_long cleanup materially easier — but only if you build the detector as a *frozen non-Danbooru backbone (DINOv2 ViT-L) + new CORN/SORD ordinal head trained ONLY on a rubric-first, medium-and-boundary-OVERSAMPLED gold set*, never as a continued fine-tune of V1 and never on proportionally-sampled gold. That one condition is load-bearing twice over: it is the only thing that makes the detector's errors orthogonal to the convention bias (so disagreement ranks anything at all), and the only thing that keeps the medium cell from being *anti-sorted to the bottom* (worse than random review).

---

# 2. WHY IT WORKS (the mechanism)

The defect is invisible to every GT-only check (implication violations ~0%): one well-formed tag is present, just wrong. So the only way to surface it is **model-vs-GT disagreement**, and the only way disagreement carries truth-signal is if the model never learned the wrong labels in the first place.

The mechanism, grounded:

- **A clean-trained model never fit the systematic wrong labels**, so it has no memorized "long-is-the-default" prior to reproduce. When it sees a long-only image whose hair is actually past-the-hips, it confidently predicts `very_long`. That high-confidence disagreement with the dirty GT *is* the suspicion signal. Because the noise is systematic + directional-down, the disagreement is **consistent and monotone** — it points the same way every time (predict-up vs GT) — which is exactly what makes it rankable rather than random.

- **This is Confident Learning's out-of-sample requirement made concrete** (Northcutt JAIR'21; multi-label extension Thyagarajan et al. 2211.13895). CL's per-tag self-confidence `s = y·p + (1−y)·(1−p)` only ranks errors if `p` comes from a probe that *never trained on that item's noisy label*. A clean-trained head satisfies this by construction. CL's recovery *theorem* assumes class-conditional noise and does NOT formally hold here (M4 is instance-dependent, non-identifiable from noisy labels alone — Xia NeurIPS'20, Berthon 2001.03772) — so keep CL's **ordering**, discard its noise-rate estimate.

- **Trusted-data methods say the clean anchor is the whole ballgame.** GLC (Hendrycks NeurIPS'18) shows a tiny trusted set carries the correction signal; Yu et al. ICML'23 ("Delving into Noisy Label Detection with Clean Data") frames detection as hypothesis-testing against a trusted clean subset and beats clean-free baselines by +19% F1. The instance-dependent-noise literature proves *why* you cannot skip it: this noise is mathematically non-identifiable from noisy labels alone — truth must enter from outside, via the gold set.

**This is the SAME orthogonal-anchor logic as the DINOv2 specialist, just reframed.** "Clean-finetune → rank dirty" and "frozen-DINOv2 ordinal specialist → rank by signed ordinal residual" are not two strategies — they are one strategy described from two ends. The specialist *is* the clean-finetune: the only "finetune" is fitting a small head on the gold set; the backbone stays frozen precisely so its features remain bias-orthogonal. The word "fine-tune" is the only thing that makes them sound different.

---

# 3. WHICH MODEL TO FINE-TUNE

**Build (b): frozen DINOv2 ViT-L + a new, tiny rank-monotone ordinal head (CORN/SORD) trained from scratch on gold.** Cache embeddings once; the head re-fits in seconds.

Ranking, with reasons:

**(b) Frozen DINOv2 + new ordinal head — DO THIS.**
- DINOv2 never saw a single Danbooru tag, so its residual errors are *image-content* failures, random w.r.t. whether an annotator typed `long_hair` — the error-orthogonality the detector needs.
- DFR (Kirichenko ICLR'23): retraining only the last layer on a small *class-balanced* set recovers worst-group performance — *if the frozen features encode the true signal*. True for DINOv2 (features never encoded the convention); contaminated for V1.
- Linear-probe-on-frozen beats full FT under high noise (arXiv:2310.17668, IJCAI'24); DACoN (2025) shows frozen DINOv2 carries part-level anime line-art semantics (57.5% vs CLIP 36.7%) — prefer DINOv2 over CLIP.
- Per-tag / multi-label: the head is a per-attribute hair-length ordinal head, not a whole-set classifier. The disagreement is computed per-attribute against the specific length tag, so it slots straight into a per-image/per-tag queue.

**(a) Full-fine-tune biased V1 — THE TRAP. Do not.** This is what "a fine-tune based on very clean tags" sounds like in plain English, and it is the single worst choice *and the one that fails silently*:
- Wang ICCV'23's three persistence conditions ALL fire for M4 (high correlation: `long` is the reflexive default; low salience: the long|very_long boundary; small clean set: ~1.5–3K). With 1024 clean samples, 4/11 attributes — including hair attributes — kept the bias.
- Kumar ICLR'22: full FT distorts features toward the in-distribution convention you need to disagree with (~7% worse OOD; detection is OOD-shaped). Liu NeurIPS'20: the memorized down-bias is sticky and re-emerges.
- Empirically confirmed in your own data: V1 fires `long_hair` on 3.12M images, length `fn_count` ≈ 0. Its disagreement with dirty GT collapses to ~1x lift on exactly the long-only M4 pool. You ship a detector that confidently finds nothing, and validation on a convention-calibrated head looks clean. **Do not trust V1 to find its own bias.**

**(c) From-scratch on clean-only — correct but data-starved.** A few-K clean images cannot train a ViT, and it throws away the free bias-free DINOv2 features. Theoretically orthogonal, practically wasteful.

**If you insist on using V1 at all:** only the DFR pattern (freeze V1 backbone, fresh last layer on balanced gold), and only as a **low-trust secondary** annotator in the fusion — never primary — because V1's features are themselves convention-shaped.

---

# 4. HOW TO SURFACE THE BAD DATA

**Ranking signal:** per-tag, out-of-sample, ordinal — not nominal CE margin.

1. Score every length-tagged image with the clean head's ordinal posterior over `{very_short < short < medium < long < very_long < absurdly_long}`.
2. Build the suspicion score as a **signed ordinal residual**, exploiting the known downward direction:
   - **Long→very_long (the trusted direction):** for the M4 pool (solo + long-only, ~76% of `long_hair`), rank descending by `P(very_long)` (and `P(absurdly_long)`). High `P(very_long)` where GT = long-only is the consistent, monotone suspicion signal.
   - **Medium recovery:** flag where GT ∈ {short, long} but head argmax = `medium` (the collapsed cell).
3. Pool into a per-image quality score with the **multi-label CL EMA** (Thyagarajan 2211.13895), but feed it the *specialist's* probabilities, not the tagger's. Add **Deep k-NN feature-space disagreement** (Bahri ICML'20) on the cached DINOv2 embeddings as a second, orthogonal signal — it needs no class-conditional matrix and is robust to IDN.
4. **Emit rows directly into pipeline2's ranked corrections queue:**
   `{image_id, tag=long_hair, action=ADD very_long / KEEP long, suspicion=0.82, evidence="specialist_p(very_long)=0.71; knn_disagree=0.6"}`.
5. Order the queue by **suspicion × error-mass** with **BADGE/core-set diversity** (Bernhardt, Nat. Commun. 2022) so the human isn't shown 500 near-duplicate ponytails. Bernhardt's selector also separates fixable-mislabel from inherently-ambiguous (abstain), so you don't burn reviewer minutes on coin-flips.

**Direction-awareness is the design, not a tweak.** The error is unidirectional-down, so the recovery signal is unidirectional-up; a symmetric detector wastes power. **Non-negotiables baked into the action column:** ADD `very_long` / KEEP `long` (never auto-downgrade long); protect existing `medium_hair` from removal (a long-default reviewer will strip a correct medium toward long — guard against it explicitly); base-triple-only exclusivity (modifiers ride along); never force colors exclusive.

---

# 5. WHERE IT BREAKS

This does **NOT** escape the gold-set single-point-of-failure — it *concentrates* it, because queue order is a deterministic function of the head, which is a deterministic function of gold coverage. Three break surfaces:

**(a) Fuzzy-boundary false-positive flood — the most insidious failure.** The long|very_long boundary ("past the hips") is genuinely fuzzy, and a small head on frozen features is **overconfident off the gold manifold**. It will emit high `P(very_long)` on a large mass of *correctly* tagged long-only images. Worse: by ranking one direction you **deliberately removed your only symmetry check** — you cannot distinguish "found systematic down-bias" from "uncertain at a fuzzy boundary, and uncertainty is one-sided by construction." Concrete risk: true-very_long base rate among length-tagged solo is ~16%, so the ceiling on genuine long→very_long upgrades in the long-only pool is bounded; if boundary precision on top flags is even mediocre (~60%, optimistic), ~40% of your top-of-queue is a human being nudged to add `very_long` that doesn't belong — and under a long-default habit some reviewers *accept* it, injecting reverse-direction noise. **Hard gate:** measure the head's precision at the top of the long→very_long queue AND its FP rate at the long|very_long boundary on the held-out gold slice. If boundary FP is high, cap auto-surface to high-separability long→canonical-very_long and route the boundary entirely to **abstain/SORD-soft review** — never auto-apply.

**(b) The MEDIUM cell is binary on gold coverage — can go WORSE than random.** Corpus: medium 2,083 vs long 24,032 (~11.5:1), eaten from both sides. PLC's (Zhang ICLR'21) region condition is literal: the clean head places `medium = argmax` in medium-feature regions *only if gold makes medium dense enough to form a decision region*. Proportional gold inherits the ~5.8% starvation → no medium region → medium suspicion ≈ 0 → and the head, now confidently saying long/short in medium-feature regions, **anti-sorts medium images to the bottom**. An unranked reviewer hits medium errors at the 5.8% base rate; a gold-starved ranked queue drives that toward ~0% at the top. **Mandate:** oversample medium AND both boundaries (short|medium, medium|long) in gold; grow via active-learning mining (BADGE/core-set on DINOv2 space) of those two cells each round. There is no in-between.

**(c) Gold non-representativeness (style/source skew).** Danbooru hair-length appearance is conditioned on art style/archetype/resolution. Narrow gold → boundary calibrated for that slice, errors correlated with style, and a same-slice held-out gold looks fine (Goodhart) while the queue surfaces "images unlike gold" rather than "mislabeled images." Require a **rubric-first** gold (reviewers never see existing tags) with a **held-out, never-seen Goodhart slice** for all detection-quality measurement, plus an "all-signals-said-keep" probe to bound the silent false-negative floor on the diagonal.

---

# 6. THE ITERATIVE LOOP

The bootstrap is real and standard (active label correction: Kremer AISTATS'18; Active Label Cleaning: Bernhardt 2022) — but self-training on the dirty distribution amplifies bias (Co-teaching provably converges to consensus, Han NeurIPS'18; Arazo IJCNN'20 names confirmation bias). The loop converges **only** with an external immutable anchor each round.

**The loop:**
1. Fit specialist head on gold (backbone frozen).
2. Score dirty GT out-of-sample → rank by suspicion × error-mass with BADGE/core-set diversity.
3. Human verifies top-of-queue in pipeline2.
4. **Only human-verified items** join the gold; re-fit the tiny head (seconds); re-score.

**Three immutable rules — non-negotiable, not best-effort:**
- **Held-out gold slice the loop NEVER trains on, measured every round.** Track the off-diagonal — `P(predict long | true very_long)` and the two medium cells. It must *shrink* each round. **STOP when it stops shrinking or starts growing** (semantic-drift guard).
- **Human-only promotion to clean.** Never auto-promote the model's own high-confidence predictions — that turns the loop into self-training on pseudo-labels and drifts (Arazo). Model output only RANKS, never relabels.
- **Queue built from the orthogonal specialist, never tagger/V1 consensus** — or the defect stays on the diagonal and never enters the queue.

Each round, the active-learning selection must **mine the medium and boundary cells**, not add more easy long examples — otherwise the gold drifts back toward the corpus's 5.8% medium starvation and you reopen the §5(b) blind spot.

---

# 7. EXPECTED PAYOFF

**Realistic, and honestly bucketed — it re-allocates human effort toward higher yield rather than reducing it:**

- **Genuine front-loaded speedup on long→very_long (the big win, 76% of `long_hair`).** Generic detectors give ~4.5x top-of-queue lift (CelebA tagging analogue, Thyagarajan); because your specialist is bias-orthogonal and you pre-filter to the solo long-only M4 pool, early-round precision runs materially higher — roughly **3–5x fewer items reviewed per correction at the head of the queue**. This gain **decays as easy errors drain** — stop on marginal-yield, not on exhausting the list (Bernhardt).
- **Medium: neutral-to-harder, gold-bound.** Lift is ≤1x (and sub-1x / anti-sorted with proportional gold). Only oversampled-gold + AL mining makes it positive.
- **Boundary: correctly relocated, not faster.** SORD-soft/abstain flags the fuzzy ones honestly instead of guessing — every abstained item is still a human decision.
- **The illusion to police:** raw flag volume looks like productivity even when a chunk are correct-GT "keep" confirmations producing zero corrections (and risking reverse noise). **Measure corrections-per-item-reviewed at queue-top vs the unranked base rate, per cell, every round** — not flag throughput. You front-load the hardest, highest-skill labeling (rubric design, boundary adjudication into gold) and make the bulk cheaper; that is a good trade, but it is *re-allocation*, not less total work.

**Where it fits the broader strategy:** this clean-finetune-and-rank *is* the detector node of your established orthogonal-error plan — identical to the frozen-DINOv2 + CORN/SORD specialist, just framed as "rank the dirty GT." It is the **surfacing** half. The complement is the **fixing/training** half: feed human-verified corrections back as cleaned GT, and train v2 with noise-robust ordinal loss (SPLC/Hill, SORD soft targets at the fuzzy boundary, rank-monotone CORN) so the next model doesn't re-memorize the down-bias. Detector surfaces and ranks; humans verify; clean labels flow to v2 training. The gold set is the single point of failure for both halves — rubric-first, medium/boundary-oversampled, held-out Goodhart slice — and that is exactly where to spend your annotation budget.