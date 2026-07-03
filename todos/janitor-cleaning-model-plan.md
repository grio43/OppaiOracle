# Janitor Model — Standalone Fine-Tune & Data-Cleaning Plan (throwaway model)

**Date:** 2026-07-02
**Status:** Decision/build-out plan. New pipeline, distinct from both the V2 production run and V2.1.1.
**Revision:** 2026-07-02 — hardened after a five-lens adversarial review (literature / statistics / codebase / domain-shift / ops) + judge. Skeleton unchanged; gates, calibration, replay, writeback, and Stage-A irreversibles substantially reworked. Judge verdicts archived at [notes/janitor_plan_adversarial_review_2026-07-02.md](../notes/janitor_plan_adversarial_review_2026-07-02.md).
**Revision 2 (same day):** added the **20K adjudicated real slice** (§3.1.2) as the real-domain backbone for calibration/thresholds/gates, after establishing that the Anima screening was Gemma 4 31B IT render-verification, not human review.
**Revision 3 (same day):** **gold is TUNE-only.** Anima removed from CAL and SEALED entirely; all calibration, thresholds, and gates run on real data (portal + 20K slice). Rare siblings the real data can't fund are human-queue, never Anima-backfilled. The full ~9.5K Anima pool goes to the refit.

**One sentence:** train a model through **normal Phase 1** (per [progressive-training-plan.md](progressive-training-plan.md) + [ASL_plan.md](ASL_plan.md)), pull it **out of the normal cycle** at the end of P1, **tune it on the golden data** so its boundaries on the noisy tag groups are anchored to the rubric instead of booru convention, run it **across the 6.3M corpus for one cleaning pass**, then **discard it**. The next production model starts **from scratch on the cleaned data**.

---

## 0. Identity — what this model IS and IS NOT

**IS:**
- A **purpose-built data-cleaning instrument**. Its only deliverables are (a) corrected label sidecars for the corpus, (b) an applied-edit journal that can invert every write, and (c) a measured precision/recall report for every correction rule it ran. The checkpoint itself is a consumable.
- The **probability engine for `Pipeline2_review`** — this plan supplies the calibrated, gold-anchored scores that pipeline's detect stage needs.
- A **full-cost rehearsal of the V2 Phase-1 recipe** — the ASL §8 machinery (γ setter, checkpoint persistence, telemetry, Anima canary) gets shaken down on a run whose failure is cheap.

**IS NOT:**
- ❌ A deployment model. No Phase 2 training, no export, no serving thresholds beyond what cleaning needs. (It *does* operate at 448px post-branch — via FixRes-style head adaptation, not a P2 fine-tune; see §2.)
- ❌ V2.1.1 (that is a *training* run on gold-augmented data, downstream of cleaning).
- ❌ A replacement for the external-anchor (DINOv2) track on the ordinal axes — see the scope fence in §7. A model whose Phase 1 was trained on the noisy corpus has the **diagonal blind spot** ([golden_set_plan §2.2](../.research/golden_set_plan.md)): it learned the very convention we want to clean, and gold tuning can only move its boundaries as far as its features allow.

**Reconciling the "fine-tune trap."** V2.1.1 §5 forbids fine-tuning a biased checkpoint on small gold because feature distortion (Kumar 2022) poisons a *deployment* model. For a throwaway cleaner the trap is re-scoped, not voided: distortion doesn't hurt a model we discard, but it *does* hurt the cleaning signal (distorted features → miscalibrated scores on real booru images). The resolution is the tuning regime in §3: **frozen backbone, head-refit-first (DFR)** — exactly what [golden_set_plan §2.1](../.research/golden_set_plan.md) prescribes for few-K gold vs 192M params — with full-FT explicitly off the table.

---

## 1. Lifecycle

```
Stage B  Gold assembly & splits (TUNE / CAL / SEALED — disjoint), frozen and
         manifest-committed BEFORE Stage A launches; includes the 20K
         adjudicated real slice (§3.1.2: Gemma-assisted, disagreement-routed
         to human); gold is TUNE-only — CAL and SEALED are REAL-only (§3.2)
            ▼
Stage A  Phase 1, by the book (progressive plan §3.2 + ASL plan §3, Option A)
            │  ~40 ep @ 320px, γ_neg 7 → floor 5, clip per ASL §0 measurement
            │  + Stage-A deltas §2: checkpoint retention, EMA copy, online AUM
            ▼
         BRANCH POINT — model leaves the normal cycle (no P2, ever)
            │
         RESOLUTION SWITCH 320 → 448: bicubic pos-embed interp 20×20 → 28×28;
         backbone stays frozen — everything below runs at 448px, uniformly
            │
Stage B′ FixRes WHOLE-HEAD re-adaptation @448: all 19.3K head rows briefly
         refit on ~200–500K corpus images, original labels + original ASL loss
         (resolution adaptation only — same data regime as training; §2 delta 1)
            ▼
Stage C  Boundary tuning: frozen backbone, per-group head refit on gold
         (+filtered/down-weighted replay, §3.3) @448 — applied ON TOP of the
         resolution-adapted head
            ▼
Stage D  Calibration (per-group parametric + per-tag intercept on CAL)
         + ρ-corrected prior-shift offset; thresholds validated on REAL data (§4)
            ▼
Stage E  Corpus scoring (6.3M @ 448px, calibrated in-line) → per-mode candidate lists
            ▼
Stage F  SEALED qualifying gates → PRE-WRITEBACK certifying human audit (user
         decision point #1) → journaled writeback → cleaned sidecars → Arrow
         cache rebuild
            ▼
         DISCHARGE: checkpoint archived-then-retired; report + cleaned data +
         applied-edit journal survive
            ▼
         Next production model: FROM SCRATCH on cleaned corpus (normal V2 plan)
         — launch gated on discharge-report review (user decision point #3)
```

---

## 2. Stage A — Phase 1 training (delegated, with 6 deltas)

Run Phase 1 **exactly** per the standing docs: architecture and schedule from [progressive-training-plan.md §3.2](progressive-training-plan.md) (896w×18L, 320px, ~40 ep to genuine plateau), loss from [ASL_plan.md](ASL_plan.md) (γ_pos=0, clip frozen at the §0-decided value, γ_neg 7 → floor 5 via manual guarded steps, Option A telemetry). The ASL plan's §0 pre-launch blockers **still block this run** — the ρ measurement doubles as this plan's cleaning prior (§5). Loss-state checkpoint persistence is **already implemented in the working tree** ([asl_telemetry.py](../asl_telemetry.py): γ_neg restores from checkpoint with checkpoint-wins-over-YAML semantics, state shared by reference into `TrainingState.loss_state`) — the remaining task is commit + a kill/resume test (§8 item 1).

Deltas vs. the normal cycle:

1. **No Phase 2 — but the branch operates at 448px, with a whole-head FixRes re-adaptation.** The branch point is the P1 plateau; there is no P2 backbone fine-tune. At the branch, construct the scoring model with `image_size=448` and let checkpoint load interpolate pos-embeds bicubic 20×20 → 28×28 once ([training_utils.py:1868-1925](../training_utils.py#L1868-L1925); note 1787-1841 is the vocab-SHA resume guard, and [model_architecture.py:456-481](../model_architecture.py#L456-L481) also interpolates at runtime — prefer the one-time construct-at-448 path for determinism and zero per-batch cost). Freeze the backbone and run **everything downstream — feature cache, head refits, calibration, thresholds, corpus scoring — uniformly at 448**.
   FixRes (Touvron 2019) is a *per-classifier* recipe, and the §3.3 gold refit only touches ~100 group rows — so **before** the gold refit, re-adapt the **entire head**: sample ~200–500K corpus images, run the frozen backbone once at 448, cache pooled CLS features (~0.9 GB fp16, captured via a forward hook on `model.tag_head` — `forward()` returns logits only), and briefly refit **all 19.3K head rows** on those features with the **original labels and the original ASL loss**. This is pure resolution re-adaptation in the same data regime as training (no new label-noise exposure), costing a few-hour partial feature pass + a linear fit — negligible next to the 6.3M scoring bill. Without it, the M1 add rule thresholds 448 scores through a 320-trained head for 99.5% of tags, and per-tag monotone calibration cannot repair FixRes's image-dependent within-tag ranking distortion. The §3.3 gold group refit applies **on top of** the resolution-adapted head. Benefit bought: fine-detail tags (multi-color streaks/tips, eye color on small faces, small accessories) come back into cleaning range. **Never mix resolutions**: no 320-fit threshold may be applied to a 448 score or vice versa.
2. **Checkpoint selection for the branch — on TUNE+CAL only, real ranking wins.** Noisy-val f1 mis-ranks (Northcutt); select the branch checkpoint on: Anima recall canary (hold/high) + stable per-decile EPR + widest sibling-gap among plateau-region checkpoints. Ties → later checkpoint. Guards against the Anima-style shortcut (2,400 same-generator images sit in the P1 training pool): (a) all selection metrics are computed on **TUNE+CAL only — never SEALED** (SEALED is real-only per §3.2, so the Anima canary cannot touch it by construction); (b) add one real-data criterion — per-group sibling-gap/off-diagonal on the TUNE+CAL portal reals; (c) report Anima-vs-real rank agreement across the plateau checkpoints, and where they disagree, **the real ranking wins**.
3. **Keep TWO checkpoints, not one — and pin them against the retention policy.** The plateau checkpoint (primary cleaner) **and** one early-learning checkpoint (≈ epoch 8–10, pre-descent). Rationale: missing positives are *suppressed* progressively as training memorizes them as negatives (Kim 2022) — the early checkpoint still scores many unlabeled-true-positives high and is a **required** second opinion for the M1 add-list (§5). ⚠️ `save_total_limit: 3` ([unified_config.yaml:276](../configs/unified_config.yaml#L276)) prunes the epoch-8-10 checkpoint ~30 epochs before the branch and leaves only ~3 checkpoints to "select among": **copy the early checkpoint OUT of CheckpointManager's managed directory** when it lands, and archive model-only (optimizer-stripped, ~770 MB) copies every epoch from plateau onset. Benign divergence from the V2 rehearsal; note it in the report.
4. **Log online AUM (was: "trajectories not required" — reversed).** The two-checkpoint diff is the documented-unreliable version of this signal, and trajectories cannot be retrofitted after the run — the old "revisit if §5 precision comes up short" clause was unexecutable. Log a running mean margin per (image, GT-positive tag) and per registered group column, computed from logits already in the training forward pass (~2 GB memmap, zero extra passes; a per-image extension of the sibling-gap telemetry [asl_telemetry.py](../asl_telemetry.py) already computes). AUM (Pleiss 2020) is the primary suppression-resistance feature for M1 and the primary ranker for wrong-positive candidates. **Hook guarded fail-open** — a telemetry bug must not compromise the run's V2-rehearsal value.
5. **Maintain an EMA weight copy through the plateau region.** The plateau checkpoint is selected at maximum memorization (Kim 2022); snapshot/EMA self-ensembles substantially beat a single converged model at label-error detection (SELF, Nguyen ICLR 2020; ensemble uncertainty > vanilla CL, Jakubik 2024). One weight copy, ~free at train time. Decide on CAL whether EMA replaces or averages with the plateau scorer for candidate **generation** (recall); the gold-refit head stays the engine for gated **decisions** (precision).
6. *(unchanged)* All other Stage-A behavior per the standing docs — this is still the full-cost V2-P1 rehearsal.

---

## 3. Stage B + C — Gold assembly and boundary tuning

### 3.1 Gold inventory and honesty about "best effort"

Sources, in trust order:
1. **Portal-corrected real booru images** — Tag Review Portal `portal_data/edits.db` → exported JSON sidecars. Real-domain, human-reviewed. Highest value per image; smallest set.
2. **20K adjudicated real slice** (added 2026-07-02 — the real-domain backbone for calibration, thresholds, and gates). ~20K images sampled from the 6.3M corpus: a **prevalence-representative random core** (~8–10K; funds priors and absolute thresholds) plus **boundary strata** oversampling the registered groups' rarer siblings (~10–12K; funds group gates and refit signal). Group-axis labels via **disagreement routing** between two independent signals per axis — the existing sidecar label and a constrained per-axis Gemma 4 31B IT judgment ("what color is this character's hair?"; nominal axes are the easy VLM case — the M4 VLM rejection was specifically ordinal/relative-spatial). **Agree** → accepted as gold, residual correlated error measured by the audit stratum below. **Disagree** → human adjudication in the portal (expected ~10–20% → ~2–4K human calls; these adjudicated boundary cases are prime TUNE material — human effort lands exactly on the contested cases). A **random human audit stratum** (~200–400 per axis, agreement cases included) measures per-axis α/β — including the correlated mode where sidecar and Gemma are wrong the same way — and doubles as the per-axis validation of Gemma itself. Hygiene: these images stay in P1 training with their **original noisy labels** (no gold leakage into the run), and they are **excluded from janitor writeback** — their adjudicated corrections are applied directly at Stage F, never scored against themselves.
3. **Anima synthetic golden set** — 12,135 images, of which **9,465 are earmarked FINE-TUNE / GOLD-EVAL** per `TARGET_DATA.md` (the 2,400 rare-bucket floor images are already destined for the main pool and are NOT gold here). Prompt-controlled labels: near-perfect on **presence** of the prompted attribute, convention-shared (i.e. weaker) on **ordinal** boundary placement ([golden_set_collection_targets §8 caveat](../.research/golden_set_collection_targets.md)). **Screening was Gemma 4 31B IT render-verification, not human review** — verification of a known prompted target is a much easier task than open judging, and the labels themselves come from prompt control, but ε_gold for Anima is (render-failure rate × Gemma miss rate) and is **measured** by the spot-audit below, not assumed. Anima's role is **TUNE-only** (user decision, Rev 3): it anchors *what the tag means* in the head refit — guaranteed per-bucket floors for rare group siblings that a 20K real sample barely contains (an 0.5%-prevalence color yields ~50–100 random-core positives; rarer siblings fewer) — and appears in **no** calibration, gate, or certification set. The full ~9.5K pool is available to the refit.

**Anima gold is axis-scoped, not globally complete.** Each Anima sidecar carries ONLY the controlled attribute (+`ai-generated`); other attributes are rendered but unrecorded (collection-targets, manifest.jsonl). A breast-bucket image with rendered aqua hair is **not** a verified negative for `aqua_hair`. Rule, enforced everywhere: **group G consumes only Anima images from G's own bucket-axis folders** (folder = complete-for-that-axis label); on every other Anima image, G's tags are **masked** (missing) — in the refit loss, and excluded from CAL/SEALED denominators — never treated as negatives. Optional recovery: parse `.research/golden_gen/manifest.jsonl` prompts to harvest prompted secondary attributes as **low-trust positives only** (uncontrolled attributes were never screened; never as verified negatives). Consequence for gates: an M1 add for tag *t* is verifiable on Anima only within *t*'s own axis stratum — reinforcing per-group pooling (§6) and real-audit certification (§6).

**Best-effort accounting (the user's constraint, made operational).** No label source is 100% gold. The **α/β that feed the gates** (Rogan–Gladen, §6.1) come from the **real-slice random audit strata** (§3.1.2): **α** = label false-positive rate and **β** = label false-negative rate, measured per axis (same audit, two counters — covering reviewer error and the correlated sidecar×Gemma mode). The **Anima spot-audit** (100–200 images/axis) is a lighter, TUNE-quality diagnostic: render/screening error estimate plus a **cross-axis completeness check** (e.g. does a prompted-hair-color image contain a second, unrecorded hair color?) so within-axis completeness is *measured*, not assumed — it informs refit trust, and feeds no gate. Then:
- Gold error propagates into measured precision via **Rogan–Gladen correction**, not a symmetric band: `p_true = (p_meas − α) / (1 − α − β)` — applied to the *lower confidence bound* of the measurement (§6). This is usually **less** pessimistic than blanket ±ε subtraction, buying back automation while being more correct.
- Do **not** chase disagreements inside the gold-noise band — that's fitting the gold's noise, the exact failure golden_set_plan §8 warns about.
- No gold image is "cleaned" by this pipeline against itself.

### 3.2 Splits (fixed BEFORE Stage A launches; disjoint; never revisited)

| Split | Contents | Use | Size guide |
|---|---|---|---|
| **TUNE** | the **full ~9.5K Anima gold-eval pool** (gold is fine-tune-only) + ~50% of portal-corrected reals + ~50% of the real slice (all adjudicated-disagreement cases land here) | head refit (§3.3); checkpoint-selection metrics (§2.2) | ~20K |
| **CAL** | **REAL-only:** ~25% of portal reals + ~25% of the real slice (random-core weighted) | calibration + threshold fitting (§4); refit-variant selection (§3.3) | ~5K |
| **SEALED** | **REAL-only:** ~25% of portal reals + ~25% of the real slice (random core **and** boundary strata), **adjudicated/audited independently** where possible | qualifying-gate measurement (§6); the only pre-audit numbers reported | sized bottom-up, see below — ~5K; carries ALL auto-apply eligibility |

Rules:
- Split **by image**, stratified by axis/bucket. Splits are frozen and manifest-committed **before Stage A launches**. With SEALED real-only, the training-time Anima canary needs no carve-out — it may use the full Anima pool, and the branch checkpoint *cannot* be selected on its own certification set, by construction (a simplification bonus of Rev 3).
- Nothing in Stages B′–E may read SEALED except the final qualifying-gate evaluation. The SEALED slice must also stay disjoint from anything later promoted into V2.1.1 training (one sealed slice serves both programs — do not mint two overlapping "sealed" sets; SEALED being real-only suits V2.1.1's off-diagonal yardstick as well).
- **Size SEALED bottom-up from the enumerated gate list** (§6): ≥40 expected evaluable decisions per gated unit (per-group, per-mode). A gate left with n<29 is a decorative gate (§6 min-n rule). Gates are funded by **real data only**: any group tag or edge whose real n can't certify is pre-registered human-queue-only ("undecidable") — decided now, not discovered later, and **never backfilled from Anima**.

### 3.3 Tuning regime — DFR head refit, per confusable group

**Default (the plan):** freeze the backbone; refit **only the classifier-head rows of the registered cleaning groups** (see §5 registry — nominal color groups, ordinal groups, multi-color set, penis axis each get different *rule types*) on TUNE, **on top of the §2-delta-1 resolution-adapted head**. With independent sigmoids each tag's head row is decoupled given features, so a scoped refit cannot touch the other ~19.2K tags at all — the no-regression gate is satisfied by construction for everything outside the groups.

Mechanics:
- **Loss for the refit: plain BCE, group-balanced sampling — on gold columns only.** ASL's asymmetry exists to survive missing positives; the *gold* is (near-)complete within each axis (§3.1 axis-scoping), so the armor is unnecessary there and would re-suppress exactly the boundary signal we're buying. This touches nothing in the main training config — the refit is a separate small script over cached backbone features.
- **Replay mix — redesigned; never raw noisy labels at full weight.** As originally drafted (~50% replay with original noisy labels under plain BCE), half of each refit batch would train the *only trainable rows* on the exact M1/M3 noise being cleaned, with the missing-positive armor removed — a DFR-assumption violation (Kirichenko 2023 assumes a clean reweighting set); the refit would converge to a gold/convention compromise. Replay images **stay** (real-domain anchoring); raw noisy supervision on contested columns **goes**: (i) down-weight the replay loss term to λ≈0.25 (~4:1 gold:replay effective weight); (ii) for registered-group columns, keep a replay supervision pair only when the **pre-refit model already agrees with its label** (labeled sibling in the top raw-score quantile ≈ p>0.9-equivalent, unlabeled siblings low — small-loss selection per Co-teaching, Han 2018; raw-score quantiles because Stage-D calibration runs *after* Stage C), **or** replace group-column replay targets with pre-refit soft outputs (self-distillation) — both pin the real-domain operating point without pulling the contested boundary; (iii) because refits run on cached features, **ablate {raw, filtered+down-weighted, distilled} and select on CAL** (never SEALED) by off-diagonal metric + Anima-vs-real precision gap. Non-group columns of replay images may keep original labels (they are not being refit). Kirichenko group-balance guidance still applies: balance beats volume; average 5–10 refit seeds and average the resulting head weights.
- **Features once, refits cheap:** run the frozen backbone over TUNE + replay pool once **at 448** (post pos-embed interpolation), cache pooled features via the `tag_head` forward hook, iterate refits on features only. The gold tune pool is small (~6–8K images incl. replay), so the 448 feature pass is trivial.
- **Bias/offset double-count trap:** the §2-delta-1 whole-head refit and the §3.3 group refit both re-derive head biases; the §4 prior-shift intercept is applied **once, after all refits** — never baked into a refit and then applied again at scoring.
- **Escalation (gated, default OFF):** if the SEALED-slice sibling-gap / off-diagonal metric shows a group's refit is feature-limited (head refit can't separate what the backbone never encoded — expected for `medium_hair`), do **not** unfreeze the backbone. That group is out of this model's reach → route it to the external-anchor (DINOv2) track per the M4 plan. Full or partial backbone fine-tuning on ~6K gold is prohibited in this pipeline, throwaway or not — it degrades the real-domain calibration everything in §5 depends on.

**Prerequisite code (real tasks, not "confirmation"):** gold-directory loader that bypasses the 95/5 auto-split (same task as V2.1.1 §3); the **off-diagonal confusion metric** (V2.1.1 §4 — needed here first, it's this plan's per-group gate); feature-cache + head-refit script; and **authoring `configs/cleaning_groups.json`** — [confusable_groups.json](../configs/confusable_groups.json) covers only 4 of the 6 named groups (hair_color, eye_color, hair_length, breast_size; no multi-color set, no penis axis) and is scoped to sibling-gap telemetry semantics. Keep it for telemetry; the new cleaning registry carries per-group **rule-type metadata** `{nominal-swap | label-set | PU-add-only | ordinal-columns-only}` plus allowed edges, so the rule runner *cannot* silently pick up ordinal edges (§5).

---

## 4. Stage D — Calibration and thresholds

1. **Calibration: per-GROUP parametric + per-tag intercept — per-tag isotonic only where it's actually fittable.** CAL supplies ~10–65 positives per tag; isotonic is unreliable below high hundreds of points and its flat-extrapolation tails sit exactly in the extreme-score region where τ_add lives (the original draft's "Platt fallback for positive-starved tags" had it inverted). Default: per-group temperature/vector scaling (beta calibration if reliability curves are non-sigmoid) + a **per-tag additive intercept**, merged with the prior-shift offset into **one** per-tag logit shift. Per-tag isotonic is permitted only at n_pos ≥ 200 on CAL — i.e. approximately nowhere. **CAL is real-only** (gold is TUNE-only): group calibration is fit on the real-slice boundary strata + portal reals; a group tag with too few real CAL positives to calibrate is flagged and its rules demoted to human queue (consistent with the §6.1 min-n rule). Calibrate untouched tags on the least-noisy labels available (portal-corrected reals; accept noisy-val calibration for tags with no coverage, flagged as such). Per ASL plan §5, asymmetric-loss outputs are systematically shifted; cleaning decisions need probabilities, not raw sigmoids.
2. **Prior-shift correction — ρ-corrected target prior, offset demoted to initialization.** The closed-form per-tag logit offset `−log P_gold(y) + log P_corpus(y)` (Saerens 2002/Menon 2013) has two problems here: (a) the sidecar-count "corpus prior" is understated by exactly the missing positives being cleaned — for a ρ=25% tag that's a one-directional ~−0.29-logit error suppressing adds precisely where ρ is highest — so correct it first: `P̂_true(y) = P_sidecar(y) / (1 − ρ̂_decile)` from the §0 ρ measurement (or EM/BBSE on calibrated corpus scores, Alexandari 2020); (b) boundary-oversampled gold is a **conditional** shift (selection on x within class), outside the label-shift assumption the closed form needs — so treat the offset as **initialization/diagnostic only**, and log the per-tag divergence between offset-implied and real-slice-implied thresholds as a gate input.
3. **Decision thresholds are fit for target precision and validated on REAL data.** Absolute thresholds (τ_add for M1 additions, τ_swap for grouped swaps) are validated on: CAL-real portal images + the CAL share of the §3.1.2 real-slice **random core** (prevalence-representative by construction; supersedes the earlier ~300–500-image mini-slice) + the §6 audit labels as they accumulate. Within-group relative quantities (swap margin δ, sibling ranking) are likewise fit on the real-slice boundary strata and adjudicated disagreements — gold is TUNE-only and funds no CAL quantity; where real n is insufficient for a sibling or edge, its rule is undecidable → human queue (§6.1), never Anima-backfilled. (Anima-fit absolute thresholds were never transferable anyway: locked generator config, single style scaffold; post-hoc calibration degrades under covariate shift — Ovadia 2019.)
4. **Winner's-curse guard:** thresholds chosen so the point estimate just clears the bar on CAL select on upward noise and fail the one-shot SEALED gate at roughly coin-flip rates. Choose τ so the CAL **Jeffreys/Clopper–Pearson lower confidence bound** on precision clears the bar — not the point estimate. Accept and report the recall cost.
5. **§7.2 fence, upgraded but capped:** if a group's SEALED-real qualifying results diverge badly from its CAL fit (calibration didn't transfer), **one** recalibration on accumulated real audit labels is permitted, then demote to human queue — no iterative refit-and-recheck loops (that's peeking, which the one-pass discipline forbids).

---

## 5. Stage E — Corpus scoring and candidate generation

**Scoring run:** frozen janitor (interpolated pos-embeds + resolution-adapted head + refit group rows + calibration + prior offsets) over all 6.3M images **@ 448px**, no augmentation. **Flip-TTA default OFF** — decide by an A/B of gate precision on CAL at 448 (a few-thousand-image job, minutes), not "decide once" by fiat; it doubles a real bill.

**Scorer build (this is a build, not an adaptation):** [Inference_Engine.py:596-604](../Inference_Engine.py#L596-L604) force-overrides `config.image_size` back to the checkpoint's trained 320 — using it as-is would violate never-mix-resolutions on the first batch. The scorer loads via CheckpointManager into a **448-config SimplifiedTagger directly** (bypassing ModelWrapper's preprocessing override), reusing InferenceDataset/preprocess_batch, with a new parquet emit stage. Hygiene: mask special indices `{PAD_idx, UNK_idx}` **resolved from the vocab** — never positional `logits[:, 2:]` slicing (a silent off-by-two across every stored column and the asl_telemetry group index tensors); all parquet columns store **original vocab indices**. fp32 sigmoid per ASL plan §5. Budget: days-scale task (§8 item 4).

**Cost, priced:** ~315 GFLOP/image at 448 → roughly 100–200 img/s on the RTX 5090 (bf16, no grads, batch ≥192) → **~10–18 h per full 6.3M pass**. The early-checkpoint pass (below) adds only a candidate-subset re-score, not a second full pass.

**Storage — persist threshold-relative CALIBRATED scores, not a raw top-K.** Stage D precedes Stage E, so the harness applies calibration + prior offsets **in-line** and persists, per image: (a) all registered cleaning-group columns, (b) all columns where GT=1, (c) every (tag, p_cal) with `p_cal > τ_add(tag) − 0.1` slack — so thresholds can be revised post-SEALED without a 6.3M re-score. A raw-score top-256 cut would silently starve M1, the plan's strongest mode: per-tag intercepts and heterogeneous prior offsets reorder tags across images, so a rare tag raw-ranked 300th can be the top *calibrated* candidate and never reach the artifact. Calibrated top-K kept only as a diagnostic. Parquet, keyed by image id; ~10–15 GB.

**Early-checkpoint cross-check — candidates-first, dense gather, required.** The plateau/EMA scorer emits M1 candidates first; the early checkpoint (§2 delta 3, its own interpolated pos-embeds, identical 448 setup) then scores **only candidate-bearing images** and gathers **exactly the candidate (image, tag) cells dense from its full logits**. (An identical top-K schema from the early model would be missing the plateau model's candidates precisely on the suppressed tags this signal exists to rescue — the boost would silently degrade to "absent = no boost".) It is a *relative* boost signal only, never thresholded on its own — but it is a **required input** of the M1 rule, not optional.

**In-sample caveat, stated honestly:** CL prescribes out-of-sample probabilities; K-fold janitors are unaffordable. We accept in-sample scores with these mitigations: EMA/snapshot self-ensembling for candidate generation (§2 delta 5 — the published cheap substitute; SELF, Jakubik 2024), online AUM as the suppression-resistance ranker (§2 delta 4), calibration on held-out gold (§4), the clip/γ-descent armor, and the early-checkpoint cross-check. Report this as a known bias source, and report **estimated recall** too (fraction of known M1 errors — SEALED-real adjudicated adds — surfaced by each scorer) — precision-only reporting hides exactly the memorization failure mode. One aligned property: the SEALED-real images are themselves in the P1 training set — which is the *deployment condition* (the janitor only ever scores its own training set), so the real-slice gates measure the deployed setting, not a generalization gap.

### Candidate lists, per noise mode — rule types from the cleaning registry (§3.3)

| Mode | Rule | Signal strength |
|---|---|---|
| **M1 missing positive** (add) | GT=0, calibrated p > τ_add(tag); AUM + early-checkpoint boost required inputs — suppression-resistant evidence | **Strongest.** This is the mode the model's whole training run is aligned with; the §0-P0 ρ measurement predicts list sizes per decile |
| **Wrong positive, confusable swap** — **NOMINAL color groups only** (`hair_color`, `eye_color`) | GT sibling *i*=1 within the group, but calibrated p(i) low AND p(j) of one sibling high (margin δ fit on CAL-real boundary strata, §4.3); emit as a **swap proposal** i→j, never a bare removal | **Medium.** Gold-refit head + CL-style ranking (cleanlab multi-label on the group columns as the ranker); evaluated per §6 (synthetic corruption + real audit) |
| **Ordinal groups** (`hair_length`, `breast_size`) | **NO auto-apply swaps** — adjacent ordinal swaps are M2 in disguise (the janitor adjudicating its own convention against convention-shared Anima ordinal gold — the §7.1 fence applies). Emit calibrated columns for the DINOv2 fusion track; at most human-queue proposals for FAR (non-adjacent) disagreements, which are gross mislabels rather than convention calls | **Columns + queue only** |
| **Penis axis** | **PU add-only** ([golden_set_plan §4.3](../.research/golden_set_plan.md): 92.2% unlabeled-normal — there is no "normal" tag to swap to; swap semantics structurally inapplicable) | **M1-style only** |
| **Multi-color set** | **Label-set track**: set-overlap rule (missing/extra member), not pairwise swap | **Medium; per-set gate** |
| **Wrong positive, non-grouped** | GT=1, calibrated p ≪ τ (bottom-percentile self-confidence), cleanlab + AUM ranking | **Weak-to-medium.** Human-queue only |
| **Ordinal convention bias** (M2: reflexive `long`, `medium` starvation) | **OUT OF SCOPE for this model** — diagonal blind spot. Emit the group columns to feed the external-anchor (DINOv2) fusion track; do not let the janitor adjudicate its own convention | **Structurally unreachable from inside** |

---

## 6. Stage F — Two-stage gates, journaled writeback, verification, discharge

### 6.1 Gate statistics (applies to every gate below)

The original gates carried **no sampling-error term** — at n=30 the binomial band (~±0.12) dwarfs ε_gold (~0.02–0.03). Redefined:

- Every gate is a **one-sided 95% Clopper–Pearson (or Jeffreys) LOWER confidence bound** on measured precision, then gold-error-corrected via Rogan–Gladen (§3.1): `p_true = (p_L − α) / (1 − α − β)`.
- **Minimum-evidence rule:** a rule whose evaluable decision count cannot mathematically certify its bar even at zero errors (n < 29 for a 0.90 bar; n < 59 for 0.95 — from `0.05^(1/n)`) is **ineligible for auto-apply** and pre-registered human-queue-only. Reported as **"undecidable," never "failed."**
- Gates pool **per-GROUP** (never per-edge below the n floor — 548 hair-color edges vs ~400 SEALED images makes per-edge gating arithmetic nonsense).
- **Damage-weighted multiplicity control:** rules proposing >100K edits certify at the **99%** lower bound (α=0.01); small rules keep 95%. This bounds expected bad-edit *volume* — the actual loss function — without Bonferroni-starving every small gate.

**Swap-rule evaluation needs synthetic corruption.** On truth-labeled SEALED, any swap-rule firing is by construction a wrong proposal (the GT sibling *is* correct there), so organic SEALED swap "precision" measures ≈0 — structurally meaningless. Build the swap-eval set by **synthetic corruption**: take truth-labeled SEALED-real images (adjudicated/audited labels) of true class *j*, flip the presented sidecar label to sibling *i* (sampling flips from the corpus confusion prior), and score the rule's proposals against the adjudicated truth — this manufactures per-group n from the real-slice boundary strata. Caveat, reported: random flips are easier than real correlated errors, so synthetic precision is optimistic — the §6.3 real audit remains the certifying gate.

### 6.2 Two-stage gate structure

| Stage | What it does | Data |
|---|---|---|
| **QUALIFYING (SEALED)** | Eligibility, threshold verification, within-group structure, per §6.1 statistics. **SEALED is real-only (~5K) and carries all qualification**; rare siblings the real data can't fund are undecidable → human queue, never Anima-backfilled. Gemma-derived labels may **qualify** but never **certify**: the janitor and Gemma could share an error mode on real art, and a shared mode must not self-certify | SEALED, one shot |
| **CERTIFYING (pre-writeback human audit)** | For each qualified rule: stratified human review of the rule's **actual corpus candidates**, with a fixed quota per tag-frequency decile (e.g. 40 per super-decile, ≥200 total; 500 for rules proposing >100K edits — a uniform sample is blind in the tail deciles where §0-ρ says the noise lives), reviewed in the portal **before any sidecar write**. Auto-apply only if the audit's one-sided exact-binomial lower bound (95%; 99% for >100K-edit rules) clears the bar after §3.1 gold/reviewer-error correction. **Pre-register the rejection threshold as a COUNT** (illustrative: n=200 vs a 0.95 bar at α=0.01 → reject at ≥19 failures), not a proportion-vs-band comparison. The audit labels double as real-domain calibration data (§4.3/§4.5) | Real corpus output |

This **replaces** the original post-gate 200-random-edit audit — the human look moves *before* the writes, which is the strongest cheap protection against the plan's one irreversible failure mode.

| Action tier | Bar | Disposition |
|---|---|---|
| **Auto-apply** | Certifying-audit lower bound ≥ 0.95 (M1 adds) / ≥ 0.90 (nominal-group swaps), AND per-tag volume sanity vs the ρ/decile prior | journaled writeback (§6.3) |
| **Human queue** | 0.7 ≤ bound < bar, or any non-grouped removal, or "undecidable" per §6.1 | Pipeline2_review / portal queue, ranked by score; budget-capped |
| **Drop** | below 0.7, or tag has no gold coverage and no calibration | logged in the report, not acted on |

### 6.3 Journaled writeback (the "existing path" does not exist — this is a build)

The cited DataCleaning-Project applier (`apply_corrections_fast.py`) is verbatim "no backups, no DB writes" — the original §6 reversibility claim was unbudgeted new code, and an *intent* log cannot invert edits (rolling back a no-op add would delete a pre-existing tag). Build `tools/janitor_apply.py` (port the format-preserving sidecar editor):

- **Append-only APPLIED-delta journal** (NDJSON): one record per **actual mutation** — `{image_id, tag, op(add/swap_from/swap_to), rule_id, calibrated_score, model='janitor-v1', registry+calibration version, batch_id, full pre-edit tag list, timestamp}` — no-ops logged as no-ops.
- **Idempotent and resumable** (6.3M small NTFS writes take hours and can be interrupted mid-flight).
- **Rollback = inverse replay** of the journal filtered by rule_id/batch_id, using the recorded pre-edit tag lists. Recovery runbook: rollback → cache rebuild → restart next run.
- **Retention:** journal + pre-clean state retained until the NEXT production run's first Anima-canary and per-decile-EPR evals look sane.
- **Additive bias preserved:** removals/swaps only inside nominal groups that cleared their gate; everything else stays additive (consistent with the Tag Improvement Pipeline precedent).
- **Cache:** sidecars are edited **between runs only** — cleaned corpus → one Arrow cache rebuild → next model. (Consistent with the build-once cache decision; no mid-run edits exist in this design.)

### 6.4 Discharge checklist

1. Cleaning report: per-rule qualifying + certifying precision bounds (per domain), volumes applied/queued/dropped/undecidable, **estimated recall per scorer** (§5), per-decile ρ before/after estimate, off-diagonal confusion table before/after on SEALED.
2. Checkpoints + refit heads + calibration tables + journal archived (cold storage — kept so a bad edit batch can be re-derived/rolled back), then the model is **retired: it seeds nothing.** The next production model is from scratch on the cleaned corpus, per the normal V2 plan.
3. Post-mortem note into the V2 plan: measured ρ after cleaning decides whether the next run's clip re-opens the §0 decision (a materially lower ρ strengthens the clip=0.1 case) and whether the dormant γ 5→4 contingency becomes viable (the sealed slice now exists and has been exercised).
4. **Next-run launch gate:** the from-scratch production run does not start until the discharge report is reviewed (user decision point #3) — the catastrophic path is a bad decile discovered weeks into the next P1 with the journal already discarded.

---

## 7. Scope fences (what this plan refuses to claim)

1. **The janitor cannot clean its own convention (M2/ordinal).** Its P1 features encode booru convention; gold head-refit moves thresholds, not feature geometry. `medium_hair` recovery and boundary *relocation* on ordinal axes stay with the external-anchor DINOv2 fusion track. The janitor's contribution there is measurement plumbing (scores for the fusion), not judgment. **Corollary now enforced in the rules:** ordinal groups are columns-and-queue only — no auto-apply swaps (§5).
2. **Synthetic domain shift is managed, not solved — and now contained to TUNE.** Anima gold anchors *what* a tag means in the refit; everything decision-facing (calibration, thresholds, gates, certification) runs on real data (§3.2, §4). The residual synthetic exposure is the refit itself: the CAL-based refit-variant ablation (§3.3) is where an Anima-skewed decision surface shows up, and any group whose SEALED-real qualifying results diverge badly from its CAL fit gets one capped recalibration on real audit labels, then demotes to human-queue (§4.5).
3. **No loss-side cleaning claims.** This plan is the data lever from progressive-plan §1.7; it does not modify ASL, and it does not substitute for the noise-robust-loss track — they compose.
4. **One pass.** No iterative self-training loops (janitor cleans → retrain janitor on its own cleaning → clean again) — that's confirmation-bias amplification with no fresh anchor. One model, one pass, one report, discharge.

## 8. Sequenced task list

1. ☐ ASL §0 blockers (shared with the normal plan): ρ measurement, V1 γ-history peek. Loss-state checkpoint persistence: **already implemented in the uncommitted working tree** ([asl_telemetry.py](../asl_telemetry.py)) — task is now *commit + resume test*: kill a run mid-descent (γ=6), resume, assert the criterion's γ_neg equals the checkpoint value, not YAML.
2. ☐ **Author `configs/cleaning_groups.json`** (not "confirm coverage" — [confusable_groups.json](../configs/confusable_groups.json) has 4 of 6 groups and telemetry semantics): per-group rule-type metadata `{nominal-swap | label-set | PU-add-only | ordinal-columns-only}` + allowed edges; freeze the registry version for this pipeline.
3. ☐ **20K real slice built** (§3.1.2): stratified sample drawn (random core + boundary strata), Gemma 4 31B IT per-axis labeling run (~1 GPU-day), disagreement queue adjudicated in the portal (~2–4K images), random audit strata reviewed (~200–400/axis — doubles as per-axis Gemma validation and feeds α/β); slice images flagged as writeback-excluded. Then splits (TUNE/CAL/SEALED) frozen and manifest-committed **before Stage A** — gold TUNE-only, CAL/SEALED real-only; SEALED sized bottom-up per §3.2 with the undecidable list pre-registered; Anima spot-audit done per axis (render/screening error + cross-axis completeness — TUNE-quality diagnostic only, feeds no gate). *(Supersedes the earlier ~300–500 random-slice task.)*
4. ☐ Code: gold-dir loader (no 95/5 split), off-diagonal metric, feature-cache (tag_head hook) + head-refit script (whole-head FixRes + group refit + replay-variant ablation), calibration/threshold fitter (§4), **corpus scoring harness** (448-config direct load bypassing ModelWrapper override, in-line calibration, threshold-relative parquet emit, vocab-index masking) — days-scale, candidate-rule runner, **`tools/janitor_apply.py` + applied-delta journal** — days-scale.
5. ☐ Stage A run (normal P1 + §2 deltas: early-checkpoint pin, plateau archival, EMA copy, online AUM fail-open hook) → branch-point checkpoint selection on TUNE+CAL (+ real criterion, rank-agreement report).
6. ☐ Stage B′ whole-head FixRes refit → Stages C–D → SEALED qualifying gates → **pre-writeback certifying audit** → gate+audit report reviewed by user before any auto-apply write.
7. ☐ Stage E–F → journaled writeback → cache rebuild → discharge report.

**User decision points (three):** (1) sign-off on the qualifying-gate + certifying-audit report before auto-apply writes (step 6→7), (2) the discharge post-mortem's clip/γ recommendation for the next run, (3) **next-run launch** after discharge-report review (§6.4.4).
