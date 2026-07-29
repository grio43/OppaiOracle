This is a synthesis-and-strategy task. I have a complete evidence package (recon of three on-disk projects plus a literature survey) and an adversarial critique. The deliverable is a decision-ready strategy document. I don't need to run code or read files — the evidence is self-contained and I've been asked to produce the synthesized strategy verbatim as my response. Let me write it.

# M4 Ground-Truth Cleaning Strategy — Decision-Ready

## 1. THE CORE INSIGHT

M4 is **systematic, instance-dependent, shared-bias noise**, the provably-hardest regime: the Xia/Yao unidentifiability theorems mean you mathematically cannot recover the truth by re-fitting anything derived from the Danbooru convention — the v1 model, flip-TTA, and the three Danbooru taggers all *deconvolve to the bias, not the truth*. Truth can only enter through an **external anchor** that was produced by a different process than the one that created the bias. **Rank the anchors strongest→weakest: (1) human gold set labeled against a written rubric** — the only place real truth enters, and the load-bearing dependency of the whole system; **(2) non-Danbooru VLM with rubric+CoT** — genuinely bias-*decorrelated* (never trained on Danbooru tag conventions) but not bias-*free* (it has an opposite, upward overshoot: 85% of its accepted `long` flags were truly `medium`), so it brackets the truth from the other side; **(3) current community Danbooru re-pull via post-id** — semi-independent at best (same site's convention) but a live, human-edited signal, weight by edit-recency/editor-count; **(4) Danbooru-tagger consensus (WD-ViT/ML-Danbooru/Camie)** — weakest, three positively-correlated echoes of one bias, usable only for random-error suppression and review-queue ranking, never as the truth gate. The single highest-leverage correction to the handoff is to **invert the trust order**: the handoff makes (4) the corpus-scale gate and treats (1)/(2)/(3) as footnotes; the evidence says (1)/(2)/(3) are the only signals that can break the bias and (4) must be demoted.

## 2. THE FUSION ARCHITECTURE

**Method (concrete, from B4/B1): a Bayesian Dawid-Skene / IBCC backbone (Kim & Ghahramani 2012) with three project-specific augmentations**, calibrated semi-supervised on the human gold set (Raykar-style: gold items pin the latent truth `z_i`):

- **(a) Ordinal-adjacency prior** (Zhou et al. minimax-conditional-entropy 2014) — concentrate each source's Dirichlet confusion mass on the diagonal + adjacent off-diagonals, so the 6×6 matrix is estimable from a modest gold set and the model "knows" errors are adjacent.
- **(b) Dependency block** (DBCC / Snorkel-MeTaL dependency graph) — declare the 3 Danbooru taggers a single correlated block so their shared bias is discounted, not triple-counted. The VLM and community re-pull are declared independent.
- **(c) Informative asymmetric Dirichlet priors** encoding direction-aware trust *before* data: prior mass favoring the `long→very_long` fix-up entry, tight/skeptical priors on the `very_long→long` downgrade entry. After gold calibration this is no longer asserted — it's measured.

**Sources = "annotators," one ordinal vote per image:** the 3 Danbooru taggers (argmax+conf, collapsed to one correlated block), the VLM (its own confusion matrix), the **original GT snapshot treated as just another biased annotator** (the conceptual unlock — GT is not truth), the community re-pull, and the human gold (observed `z_i` on the calibration subset). v1 OppaiOracle may be a low-trust down-weight vote only, never a tiebreaker.

**Outputs:** a per-image **posterior over the 6 ordinal levels** + a **GLAD/minimax-entropy item-difficulty** term + a **CrowdTruth item-ambiguity** score.

**Routing (thresholds set on gold ROC, not guessed):**
- **AUTO-APPLY** — only the FIX-UP direction (`GT=long` → ADD `very_long`, KEEP `long`), only when posterior mass on the single adjacent fine value clears a high gold-calibrated threshold AND item-difficulty is low AND the independent (VLM/provenance) view agrees. Even here, **gate hard**: the VLM's empirical ceiling on the long channel is 71.7% precision *with* human review, so auto-apply should be reserved for the narrow high-posterior tail, not the bulk.
- **VLM-ARBITRATED** — Danbooru block vs GT disagree; route to the independent oracle for a tie-break vote into the fusion (k-sample self-consistency, see §3).
- **HUMAN-REVIEW** — middle posterior band, ranked by `P(mislabeled) × error-mass` (Bernhardt active-cleaning 2022). The **downgrade direction (`very_long→long`) is reachable ONLY through human review by construction** (its skeptical prior makes auto-apply mathematically unreachable).
- **ABSTAIN / do nothing** — flat posterior or high item-difficulty/ambiguity (the genuinely fuzzy `long`/`very_long` boundary). Abstention is a first-class outcome, not a failure.

**Critical wiring rule:** the "confidence gate" must NOT use VLM self-reported confidence (degenerate: 99.98% "high," evidence-insensitive per Groot 2024). Replace with **VLM self-consistency** (k samples at temp>0 / paraphrased prompts; agreement = gate) and gold-calibrated per-direction reliability.

## 3. ROLE OF EACH EXISTING ASSET

**DataCleaning Project — REUSE as execution/eval/review backbone; RETIRE its detection method.** Its ingest is a single-oracle v1-judging-its-own-GT engine with no ordinal/adjacency logic and no hair-length mutex — it *cannot* detect M4 and must not be the triage. What to reuse verbatim:
- `apply_corrections_fast.py` + `export_corrections.py` — the safe atomic, encoding/BOM/line-ending/indent-preserving sidecar writeback (the Phase-4 applier). **Extend:** it does flat add/remove only, so (i) compute implication closure upstream before export, and (ii) batch every `long→medium`-style change (remove+add) **transactionally per image** so a crash can't strand an image with no base length tag (a new M1).
- `screening_*.py` + `screening/gold/screen_v1.sqlite` + `screening_grade.py` — the gold/honeypot/MCC-grader suite. **Extend:** clone the COLOR family map to a **LENGTH family with ordinal adjacency**; this becomes the Phase-1 confusion-matrix/trust-weight estimator.
- `image_analysis` + `tag_details` join + `baseline_snapshot` — **materialize a small M4-candidate table** (GT length tag co-occurring with a high-conf predicted length tag) so you never re-scan the 3.47B-row / 1.1TB DB. Use `baseline_snapshot` as the per-tagger consensus prior (for the correlated block) only.
- `pipeline2_review` UI — the abstention-first per-image human review surface; activate the unused `tag_decisions` (model_correct/gt_correct) path for ordinal decisions.
- `MUTUAL_EXCLUSION_GROUPS` — **extend with the base triple {short,medium,long} ONLY. Fence this to length.** Do not touch colors.

**VLM assist — PROMOTE from footnote to primary independent oracle; EXTEND substantially.**
- **Scope:** `TARGET_TAG="short_hair"` → all length tags; seed jobs from the **triaged subset** (Danbooru-vs-GT disagreement or `long`-only high-risk images), NOT the full corpus (cost).
- **Class scheme:** 3-class → acknowledge **6 levels in the prompt** so it stops collapsing `very_long` into `long` — but **do NOT expect reliable `very_long`-vs-`absurdly_long` discrimination** (granularity collapse: CUB-200 ~67% fine). The VLM is a reliable **base-triple {short,medium,long} adjudicator**; the fine boundary goes to humans.
- **Prompt:** inject the Phase-0 rubric verbatim + 1-2 reference images/class (Prometheus-Vision); **mandate landmark-first CoT** ("state where the visible hair endpoint falls relative to a named body landmark, THEN map to class") to fight modality-neglect; ask a discrete landmark bucket, never a length scalar.
- **Calibration:** replace the degenerate confidence field with **k-sample self-consistency**; estimate its **per-direction confusion matrix on gold** before fusion (it overshoots upward — that bias must be measured, not trusted).
- **Coverage:** route the **NSFW/moderation-blocked slice entirely to humans** (138 blocks at 6.83% → systematic hole on a non-random, often-long-hair slice).
- **Reuse:** `batch_processor.py` (rate limiter/retry/content-filter), `phase2_review` UI, `query_arrow.py`. Mine the existing **6,636 human-accepted + 1,304 rejected** verdicts as a short-end calibration seed before re-collecting.
- **SECURITY (do first):** revoke the hardcoded OpenRouter key in `VLM assist/config.py:18`, move to env var.

**Tag Improvement Pipeline (M1) — RETIRE its method for M4; REUSE infra only; HEED its hazard.**
- It is **strictly additive** and its applied rule (`very_long/absurdly_long → long`) *reinforces* the coarse `long` default — it can never produce the corrective swap M4 needs.
- **Reuse:** `cooccurrence.py` engine + the 14GB cache (ordinal co-occurrence priors); `json_updater.py` / `backup_manager.py` (DB-backed per-file backup + auto-rollback — the safest writeback of the three; extend with a remove/replace path); `implication_applicator.py::compute_vocabulary_hash` (vocab-integrity guard for the correction matrix).
- **Closure consequence to counter:** its `very_long→long` closure (and your Phase-4 closure) **inflate the `long` base rate further** — logically correct, statistically pours fuel on the collapse. This makes the **v2 soft-ordinal loss a co-requirement, not a footnote.**
- **SECURITY:** revoke the second hardcoded key in `Tag Improvement Pipeline/config.py:93`.

## 4. PRIORITIZED PLAN

**Step 0 — Revoke both hardcoded API keys; fix `loss_functions.py:269`.** Purpose: kill the live secret exposure, and re-binarize targets *before* the focal-weight mask so soft ordinal labels survive (confirmed live alias `targets_for_focal = targets`; any fractional `>0` leaks into the wrong focal branch). Trap avoided: shipping soft labels that get silently clamped back to hard, invalidating the whole v2 loss design. *(Automated fix + config.)*

**Step 1 — Write the operational RUBRIC as a LIVING document with an abstain escape-hatch.** Purpose: the single highest-value artifact — `long`-vs-`very_long` has no operational definition; without a written one, reviewers reproduce the original noise. Include reference images per class, a **logged exception list** for hard cases (seated/back-view/occluded-hips/chibi/windswept), and an explicit **"ambiguous → abstain"** rule. Trap avoided: a one-shot static rubric that forces a pick on 50/50 images, converting honestly-uncertain GT into confidently-wrong GT (strictly worse). *(Human; user signs off.)*

**Step 2 — Build the GOLD set deliberately (the load-bearing step).** Purpose: the unidentifiability anchor. Non-negotiables: **(a)** size to estimate the rare *off-diagonal* cells (`long→very_long`, `very_long→long`), not overall accuracy — **oversample boundary/ambiguous images and the collapsed middle**, do NOT sample proportionally (or it inherits medium-starvation); **(b)** carve a **second, held-out gold slice the correction pipeline never sees** — your only defense against Goodharting the calibration gold; **(c)** include **"all signals agreed keep"** images to measure the false-negative floor; **(d)** run reviewers through the existing **screening/honeypot/MCC gating** before they label gold. Trap avoided: clean-looking confusion matrices over a corpus that's actually getting worse (the invisible §4 failure). *(Human, gated.)*

**Step 3 — Materialize the M4-candidate table from the existing DB.** Purpose: fast triage substrate without re-scanning 3.47B rows; join `image_analysis.ground_truth_tags` × `tag_details` for (GT length tag, high-conf predicted length tag) co-occurrence. Trap avoided: a cold 1.1TB scan per query. *(Automated.)*

**Step 4 — Re-run the VLM (extended) on the triaged candidate subset.** Purpose: inject the genuinely independent signal where it matters. 6-level-aware prompt, landmark-first CoT, k-sample self-consistency, NSFW→human. Trap avoided: spending VLM budget on the whole corpus, and trusting its degenerate confidence field. *(Automated inference + human on flags.)*

**Step 5 — Re-pull current community Danbooru tags via post-id** (post_id == filename stem confirmed; `id_index.json` maps 5.92M posts). Weight by edit-recency/editor-count. Trap avoided: treating same-site tags as a clean co-equal independent signal. *(Automated.)*

**Step 6 — Calibrate the fusion model on gold; estimate per-source per-direction confusion + true M4 prevalence.** Purpose: turn "direction-aware trust" from assertion into measured numbers; **prove or disprove Camie's independence** (does its `long→very_long` off-diagonal differ structurally from WD-ViT/ML-Danbooru? until proven, it's one more correlated vote). Run the IBCC/DBCC + ordinal-adjacency + asymmetric-prior model. **Empirically select** among {DS, IBCC/DBCC, ordinal-minimax-entropy, Raykar} on gold (Zheng 2017: none dominates). Trap avoided: triple-counting correlated taggers; unjustified Camie trust. *(Automated, gold-anchored.)*

**Step 7 — Route by posterior; humans review the middle band, abstain the fuzzy, auto-apply only the narrow high-trust fix-up tail.** Purpose: corrections enter only through the calibrated posterior + human confirmation. **Protect existing `medium_hair` tags from removal** (rare+noisy → naive noisiness ranker treats true-medium as noise and strips it toward `long`, worsening the collapse). Defer medium *recovery* on Danbooru-only signals. Trap avoided: making the medium collapse worse; over-aggressive auto-apply. *(Mostly human; narrow auto.)*

**Step 8 — Apply via the safe writeback, implication-closure LAST, transactional per image, with backup/rollback.** Purpose: format-preserving, reversible edits. Run closure *after* M2 resolution and re-validate (don't orphan a modifier by resolving a base-tag contradiction first). Trap avoided: crash-stranding an image with no base length tag; orphaned modifiers. *(Automated, reversible.)*

**Step 9 — v2 retrain with soft ordinal labels (SORD), unimodality calibration (ORCU), loss countering the `long` base rate.** Purpose: emit cleaned labels as **soft ordinal distributions** (mass on the true rank bleeding to adjacent ranks, width from the gold confusion matrix), carve hair-length into an ordinal sub-head, keep implication parents hard at 1.0. The loss must actively counter the `long` base rate that closure inflated. Trap avoided: hard flips that memorize coin-flips; a v2 that re-learns the collapse. *(Automated.)*

**Step 10 — Iterate as ASYMMETRIC co-teaching with an immutable anchor.** Purpose: residual denoising that converges instead of amplifying. **Rules (must be explicit):** View A = Danbooru block; View B = non-Danbooru (VLM + provenance + gold). A label flips only on **A∧B agreement in the high-trust direction**; A-only never auto-flips. The **v2 model may only down-weight-vote, never flip alone.** **Re-query the VLM and re-pull provenance on the NEW residual every round** (fresh external signal). Measure corpus improvement on the **held-out** gold slice. Trap avoided: self-training drift / confirmation-bias amplification / Goodharting the calibration gold. *(Automated + human on flags.)*

## 5. SUCCESS METRICS & STOPPING

- **Primary (per round):** the **held-out gold confusion matrix off-diagonal mass shrinks** — specifically the `long→very_long` and `medium`-collapse cells. Measure on the held-out slice the pipeline never trained/calibrated on; the calibration gold will always look good (Goodhart).
- **False-negative floor:** error rate on the **"all signals agreed keep"** gold subset — must not be ignored; a corpus can look cleaner on the reviewed slice and be unchanged on the silent majority.
- **Distributional health:** corpus `medium` share rising toward a plausible target and `long`-only fraction (currently ~76-78%) falling — re-run `a4_corpus_signal_scan.py` after each writeback to track. Watch that closure-driven `long` inflation is being countered by the v2 loss.
- **Per-source drift:** each source's gold-calibrated confusion matrix re-measured each round; a **non-shrinking or growing off-diagonal = the loop is amplifying, STOP.**
- **Reviewer health:** honeypot integrity + pooled MCC stay above gate (active selection amplifies a noisy annotator).
- **Stopping rule:** stop a round when the held-out off-diagonal stops shrinking, OR reviewer agreement on newly-queued items decays (rubric exhausted its resolving power → abstain the rest), OR the marginal human-review yield per item drops below threshold.

## 6. TOP RISKS & MITIGATIONS

1. **The gold set is non-representative / under-sized (invalidates everything, invisibly).** Every threshold, trust weight, and stopping rule is computed against it; a bad gold set produces clean matrices over a worsening corpus. **Mitigate:** oversample boundary + collapsed-middle; size for the off-diagonal cells; build a **held-out Goodhart-defense slice**; include "all-keep" cases; gate reviewers first.
2. **Danbooru-tagger consensus laundering bias by omission.** A true-`very_long` jointly called `long` by 2/3 taggers never enters the queue. **Mitigate:** demote the taggers to a single correlated block (DBCC) used only for ranking/random-error suppression; make the VLM/provenance the gate-openers; treat **cross-family (VLM-vs-tagger) agreement as strong, intra-family agreement as weak.**
3. **VLM mis-wired or over-trusted.** Degenerate confidence as a gate = no-op (everything passes); anime-OOD + relative-spatial = its two worst regimes stacked; upward overshoot. **Mitigate:** self-consistency gate (not self-reported confidence), gold-calibrated per-direction confusion, landmark-first CoT, base-triple scope, abstention-first, NSFW→human, never auto-apply raw VLM flags.
4. **Iterate loop re-launders / closure re-inflates `long`.** **Mitigate:** v2 may only down-weight-vote; fresh VLM+provenance every round; measure on held-out gold; and make the **v2 soft-ordinal loss counter the `long` base rate** closure inflates (co-requirement, not footnote — fix `loss_functions.py:269` first).
5. **Worsening the rare classes / breaking structure.** Naive noisiness ranking strips true-`medium`; flat add/remove can strand base tags. **Mitigate:** explicitly **protect existing `medium_hair` from removal** and inject rare classes into the human queue (ALR imbalance handling); transactional per-image apply; closure last + re-validate; fence the mutex extension to length only (never force colors exclusive).

## 7. OPEN DECISIONS FOR THE USER

1. **Rubric boundaries.** The exact landmark cutoffs (e.g., is `very_long` = past hips or past mid-thigh? how to handle seated/back-view/occluded-hips) are judgement calls only you can set — the entire system calibrates to your definition. This is the one decision that cannot be delegated to any model.
2. **VLM model tier.** Stay on Gemini flash-lite (cheap, but weakest anime perception — the likely weak link) or pay for a stronger frontier VLM for the calibration study? Test both on gold before committing.
3. **Gold-set budget vs corpus ambition.** How many human-hours for gold labeling (deeper gold = more trustworthy off-diagonals = safer auto-apply) vs corpus review? The gold set is where to spend first.
4. **Auto-apply appetite.** How narrow is the high-trust fix-up auto-apply tail — or is *every* write human-confirmed in round 1? (The evidence supports human-confirmed-everything early; loosen only after the held-out metric proves it safe.)
5. **medium recovery timing.** Defer medium recovery until v2 (asymmetric co-teaching) provides a less-biased vote — or attempt a VLM-base-triple-led medium recovery earlier, accepting the VLM's coarse-adjudicator limits?
6. **Scope discipline.** Hold strictly to hair-length M4 this cycle, or allow the (tempting, dangerous) reuse of the ordinal machinery for colors? Recommendation: **fence to length**; colors stay detect-and-abstain, never force-exclusive.