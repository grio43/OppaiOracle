# Janitor Model — Standalone Fine-Tune & Data-Cleaning Plan (throwaway model)

[Roadmap](README.md) · [Shared production recipe](v2-plan.md) ·
[Off-diagonal metric owner](v2.1.1-gold-training-pipeline.md#4-off-diagonal-confusion-metric--spec-owned-here---blocking)

**Date:** 2026-07-02 · **Status:** Decision/build-out plan. New pipeline, distinct from both the V2 production run and V2.1.1.
**Revision history:** hardened by a five-lens adversarial review + 2 same-day revisions (gold→TUNE-only; 20K adjudicated real slice added); full narration at `36dc11e:notes/janitor_plan_adversarial_review_2026-07-02.md` (git history; removed from the tree 2026-07-29). Condensed 2026-07-29; deferred ordinal track split out into `todos/ordinal-fusion-track.md` in `2c1cfa1` (2026-07-29), retired from the tree 2026-07-30 — recover via `git show 2c1cfa1:todos/ordinal-fusion-track.md`; living summary at §7.1.

**One sentence:** train a model through **normal Phase 1** (per [v2-plan.md](v2-plan.md)), pull it **out of the normal cycle** at the end of P1, **tune it on the golden data** so its boundaries on the noisy tag groups are anchored to the rubric instead of booru convention, run it **across the ~5.92M corpus (5,921,102 measured — v2-plan §10) for one cleaning pass**, then **discard it**. The next production model starts **from scratch on the cleaned data**.

---

## 0. Identity — what this model IS and IS NOT

**IS:** a **purpose-built cleaning instrument** — deliverables: corrected sidecars, an applied-edit journal that can invert every write, a per-rule precision/recall report; checkpoint = consumable · the **probability engine for the human review process** (Tag Review Portal + queue) · a **full-cost V2-P1 rehearsal** (γ setter, checkpoint persistence, telemetry, Anima canary) where failure is cheap.

**IS NOT:** ❌ a deployment model — no P2, no export, no serving thresholds (448px post-branch = FixRes head adaptation, §2) · ❌ V2.1.1 (a *training* run, downstream of cleaning) · ❌ the external-anchor (DINOv2) track on ordinal axes (§7.1; full spec `git show 2c1cfa1:todos/ordinal-fusion-track.md`) — P1 on the noisy corpus ⇒ **diagonal blind spot** (golden_set_plan §2.2): it learned the convention being cleaned.

**Fine-tune trap, re-scoped:** feature distortion (Kumar 2022) hurts the **cleaning signal**, not the discarded weights (Kirichenko/DFR regime and the full-FT prohibition: §3.3).

---

## 1. Lifecycle

```
Stage B  Gold assembly & splits — frozen and manifest-committed BEFORE
         Stage A launches (§3.1, §3.2)
            ▼
Stage A  Phase 1, by the book (v2-plan §3/§4/§6/§8) + the 6 deltas (§2)
            ▼
         BRANCH POINT — model leaves the normal cycle (no P2, ever)
            │
         RESOLUTION SWITCH 320 → 448 (§2 delta 1) — everything below it
         (B′, C, D, E) runs at 448px, uniformly
            │
Stage B′ Whole-head FixRes re-adaptation @448, all 19.3K rows (§2 delta 1)
            ▼
Stage C  Boundary tuning — per-group head refit on gold, on top of B′ (§3.3)
            ▼
Stage D  Calibration + thresholds, validated on REAL data (§4)
            ▼
Stage E  Corpus scoring, ~5.92M @448, calibrated in-line → candidate lists (§5)
            ▼
Stage F  SEALED qualifying gates → PRE-WRITEBACK certifying human audit (user
         decision point #1) → journaled writeback → cache rebuild (§6)
            ▼
         DISCHARGE: checkpoint archived-then-retired; report + cleaned data +
         applied-edit journal survive (§6.4)
            ▼
         Next production model: FROM SCRATCH on cleaned corpus — launch gated
         on discharge-report review (user decision point #3)
```

---

## 2. Stage A — Phase 1 training (delegated, with 6 deltas)

Run Phase 1 per [v2-plan.md](v2-plan.md) §3/§4/§6/§8 **exactly**. **Fresh from-scratch Stage A CONFIRMED by user 2026-07-30** (vs reusing V1) — the programme therefore contains two multi-week P1 runs; budget both (v2-plan §12). Inherited decisions (2026-07-30): sizing = **option C, 896×18 MLP 2320, ~150M** (v2-plan §3) · loss = ASL γ_neg **7.0 fixed**, γ_pos=0, clip=0.05, **focal weight detached** (v2-plan §4 — implemented and checked after resume) · schedule = **WSD** (v2-plan §6), so "branch at the P1 plateau" now means: plateau gate fires on the EMA weights → run the 1-sqrt cooldown → **branch the cooled-down (annealed) checkpoint**, never the raw stable one. Telemetry: v2-plan Appendix A. Launch blockers: v2-plan §8. Cleaning prior ρ̂: v2-plan §10 (used in §5). γ persistence is committed (`a872e9f`). `test_training_audit.py` rejects conflicting persisted gamma and checks the fixed loss in real completion/halt checkpoints; `test_softstop_resume_e2e.py` now observes the resumed live criterion (gamma=7, clip=.05, detached focal weights). Telemetry cannot change the loss.


**Training split inheritance:** Stage A follows V2 §8.1’s 30K-only budget. V2 now prepares a fresh fixed holdout from the combined E:\Dataset corpus after all subsets arrive. Do not automatically carve additional CALIB/TEST images. Shared Phase 1 code now implements WSD, QK-norm/LayerScale, namespaced ingestion and fp32 head/position optimizer states. The new 60-epoch ceiling and `v2_phase1_vit` artifacts are for production V2; Stage A still needs its own isolated experiment, budget and campaign-specific launch checks.

**Inheritance boundary:** production V2 §6 cut weight EMA on 2026-08-01. This campaign
still explicitly specifies EMA in delta 5 and its branch-selection readouts in delta 2;
that is a janitor-only delta, not an inherited production requirement. Shared correctness
and launch-readiness work, including the September audit closure checks, is tracked once
in [V2 §12](v2-plan.md#12-order-of-work) and applies before Stage A.

**Verification inherited from V2 §4/§10 (2026-09-05):** retain detached ASL (7, 0, .05)
for Stage A; the paper also supports (4, 0, .05), but existing mining does not establish
that residual omissions are low enough to favor 4. The 896-wide shape is an adaptation of
SoViT's 880-wide model. Score-mass EPR has no target near 1/(1−ρ); diagnose formation with
ranking and reviewed-real evidence. The γ4 restart remains a fresh-contract contingency,
not a YAML-only switch. Completed Stage A includes cooldown; numerical aborts stop immediately.
Production's two-model GOLD-SLICE freezes its protocol before training and its realized
pool after V2 inference; janitor's independently assembled TUNE/CAL/SEALED manifests still
freeze before Stage A. Do not share fitting labels with the production gate.

**Prior-correction prerequisite:** V2 §10's mined addition ratios are not identified
population missingness. Before using `P_sidecar/(1−ρ̂)` below, estimate residual ρ on
probability-sampled reviewed real data, state assumptions and uncertainty, and propagate
that uncertainty. Until then the historical power-law fit is a diagnostic only; it cannot
certify an offset or auto-apply rule.

Deltas vs. the normal cycle:

1. **No P2 — branch at the P1 plateau, then 448px via whole-head FixRes re-adaptation.** Construct scoring model at `image_size=448`; one-time bicubic pos-embed interp 20×20 → 28×28 at load (training_utils.py:1990-2047; runtime interp at model_architecture.py:456-481 — prefer one-time). Backbone frozen; ALL downstream at 448 — **never mix resolutions**. Before the §3.3 group refit (~100 rows), refit **all 19.3K head rows** (FixRes is per-classifier, Touvron 2019): ~200–500K corpus images → one frozen 448 pass → pooled CLS features → **original labels + original ASL loss**; few-hour pass + linear fit (else 99.5% of tags threshold 448 scores through a 320 head).
2. **Branch checkpoint on TUNE+CAL only; real ranking wins** (noisy-val f1 mis-ranks — Northcutt). Criteria: Anima recall canary (hold/high) + stable per-decile EPR + per-group sibling-gap/off-diagonal **computed on portal-corrected real labels only**; ties → later. ⚠️ The noisy-telemetry sibling-gap is **veto-only** (v2-plan §9.2) — never select a checkpoint for the *widest* gap; a model that memorized a wrong positive maximizes it. Under WSD, selection ranks the **EMA readouts of stable checkpoints** to choose the cooldown launch point; exactly **one** 1-sqrt cooldown runs, and its output is the branch checkpoint (no multi-cooldown comparisons — v2-plan's single-configuration rule). Anima-shortcut guards (2,400 same-generator images in the P1 pool): never SEALED; one real criterion (per-group sibling-gap/off-diagonal on portal reals); report Anima-vs-real rank agreement — **real ranking wins**.
3. **TWO checkpoints, pinned:** plateau (primary) + early ≈ epoch 8–10 (suppression grows with training, Kim 2022; **required** M1 second opinion, §5). The early checkpoint is a **diagnostic-only raw-stable scorer, explicitly exempt from the branch-after-cooldown rule** (it is never the branch, never thresholded alone — relative-boost input only), and it **gets its own few-hour B′ whole-head refit at 448** before scoring (else its M1 boost runs 448 features through a 320 head — the exact failure delta 1 exists to prevent). ⚠️ `save_total_limit: 3` (unified_config.yaml:308) prunes it ~30 epochs pre-branch → **copy OUT of CheckpointManager's managed directory**; archive model-only (~770 MB) copies each epoch from plateau onset. Note divergence in report.
4. **Online AUM** (not retrofittable; two-checkpoint diff = unreliable substitute): running mean margin per (image, GT-positive tag) + per group column from in-pass logits (~2 GB memmap, zero extra passes; extends asl_telemetry sibling-gap). Primary M1 suppression-resistance feature + wrong-positive ranker (Pleiss 2020). **Hook fail-open.**
5. **EMA weight copy through the plateau** (SELF, Nguyen ICLR 2020; Jakubik 2024; plateau = max memorization, Kim 2022). ~Free. Decide on CAL: replace vs average for candidate **generation**; the gold-refit head decides gated **actions**.
6. *(unchanged)* Everything else per the standing docs.

---

## 3. Stage B + C — Gold assembly and boundary tuning

### 3.1 Gold inventory

Sources, in trust order:
1. **Portal-corrected reals** — `portal_data/edits.db` → exported JSON sidecars. Human-reviewed, real-domain; smallest set, highest value per image.
2. **20K adjudicated real slice** — real-domain backbone for calibration/thresholds/gates. **Random core** ~8–10K (prevalence-representative → priors, absolute thresholds) + **boundary strata** ~10–12K (rarer group siblings → group gates, refit signal). Per-axis **disagreement routing**: sidecar label vs constrained per-axis Gemma 4 31B IT judgment (nominal axes = easy VLM case; the M4 VLM rejection was ordinal/relative-spatial). Agree → gold; disagree → portal adjudication (~10–20% → ~2–4K human calls; prime TUNE material). **Random human audit stratum** ~200–400/axis (incl. agreements) → per-axis α/β incl. the correlated sidecar×Gemma mode; doubles as Gemma validation. Slice stays in P1 with original noisy labels; **writeback-excluded** — corrections applied directly at Stage F, never scored against themselves.
3. **Anima golden set** — 12,135 images; **9,465 FINE-TUNE / GOLD-EVAL** (2,400 rare-bucket floor images → main pool, not gold). Counts: `F:/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/ComfyUI/output/golden_set/BREAKDOWN.md`; authoritative per-image label record: `.research/golden_gen/labels.jsonl`. (The `TARGET_DATA.md` this line used to cite is no longer generated — `.research/golden_gen/gen_target_md.py:13-14` still targets that path, but the output root now holds `BREAKDOWN.md`/`PROGRESS.md`.) Near-perfect on **presence**, convention-shared on **ordinal** placement (golden_set_collection_targets §8). Screened by Gemma 4 31B IT render-verification, not human review; ε_gold = render-failure × Gemma miss rate, **measured** by spot-audit. **TUNE-only** (user decision, Rev 3): per-bucket floors for rare siblings (0.5%-prevalence color ⇒ ~50–100 random-core positives); in **no** calibration, gate, or certification set. Full ~9.5K pool to the refit.

**Axis-scoped, not globally complete.** Anima sidecars carry ONLY the controlled attribute (+`ai-generated`); rendered-but-unrecorded is **not** a verified negative (breast-bucket image with aqua hair ≠ negative for `aqua_hair`). Rule: **group G consumes only Anima images from G's own bucket-axis folders**; elsewhere G's tags are **masked** (refit loss + CAL/SEALED denominators), never negatives. Optional: `.research/golden_gen/manifest.jsonl` prompted secondaries = **low-trust positives only**. M1 adds Anima-verifiable only within the tag's own axis stratum → per-group pooling + real-audit certification (§6).

**Best-effort accounting.** Gate **α/β** (§6.1) = real-slice audit-strata label FP/FN rates per axis (incl. the correlated sidecar×Gemma mode). **Anima spot-audit** 100–200/axis (render/screening error + cross-axis completeness) = TUNE-quality diagnostic, feeds no gate. Gold error → **Rogan–Gladen** correction applied to the *lower confidence bound* (formula and gate use: §6.1). Never chase disagreements inside the gold-noise band (golden_set_plan §8). No gold image is cleaned against itself.

**Reviewer QC — reuse, don't rebuild.** `DataCleaning Project/screening_*.py` + `screening_calibrate.py` give gold-set honeypots, per-reviewer MCC/specificity/recall, and isolated disposable DBs → use directly to vet reviewers and to measure inter-annotator agreement on the §3.1.2 audit strata that produce α/β. **Do NOT use OppaiOracle V1.1** (`DataCleaning Project/datacleaning/models/oppai/V1.1/`) as an oracle anywhere in this plan — same bias as the GT being fixed; a down-weighted 4th vote at most, and never in a gate.

### 3.2 Splits (fixed BEFORE Stage A launches; disjoint; never revisited)

| Split | Contents | Use | Size guide |
|---|---|---|---|
| **TUNE** | the **full ~9.5K Anima gold-eval pool** (gold is fine-tune-only) + ~50% of portal-corrected reals + ~50% of the real slice (all adjudicated-disagreement cases land here) | head refit (§3.3); checkpoint-selection metrics (§2.2) | ~20K |
| **CAL** | **REAL-only:** ~25% of portal reals + ~25% of the real slice (random-core weighted) | calibration + threshold fitting (§4); refit-variant selection (§3.3) | ~5K |
| **SEALED** | **REAL-only:** ~25% of portal reals + ~25% of the real slice (random core **and** boundary strata), **adjudicated/audited independently** where possible | qualifying-gate measurement (§6); the only pre-audit numbers reported | sized bottom-up, see below — ~5K; carries ALL auto-apply eligibility |

Rules:
- Split **by image**, stratified by axis/bucket. Splits frozen and manifest-committed **before Stage A launches**. With SEALED real-only, the training-time Anima canary needs no carve-out — it may use the full Anima pool, and the branch checkpoint *cannot* be selected on its own certification set, by construction.
- Nothing in Stages B′–E may read SEALED except the final qualifying-gate evaluation. The SEALED slice must stay disjoint from anything later promoted into V2.1.1 training. This SEALED slice is also V2.1.1's eval slice (v2.1.1-gold-training-pipeline.md §3) — one sealed slice serves both programs; do not mint two overlapping "sealed" sets (SEALED being real-only suits V2.1.1's off-diagonal yardstick as well).
- **Size SEALED bottom-up from the enumerated gate list** (§6): ≥40 expected evaluable decisions per gated unit (per-group, per-mode). A gate left below the §6.1 min-evidence floor is decorative. Gates are funded by **real data only**: any group tag or edge whose real n can't certify is pre-registered human-queue-only ("undecidable") — decided now, not discovered later, and **never backfilled from Anima**.

### 3.3 Tuning regime — DFR head refit, per confusable group

**Default:** frozen backbone; refit **only the registered groups' head rows** (§5 registry) on TUNE, on top of the §2-delta-1 head. Independent sigmoids ⇒ the other ~19.2K tags untouched; no-regression by construction.

- **Loss:** plain BCE, group-balanced, gold columns only (ASL's missing-positive armor would re-suppress the boundary signal; gold near-complete per axis, §3.1). Separate script over cached features; main training config untouched.
- **Replay — never raw noisy labels at full weight** (DFR clean-set assumption, Kirichenko 2023): (i) replay loss λ≈0.25 (~4:1 gold:replay effective); (ii) group-column replay pairs only where the pre-refit model agrees with the label (labeled sibling in top raw-score quantile ≈ p>0.9-equivalent, unlabeled siblings low — small-loss selection, Co-teaching Han 2018; raw quantiles: calibration is post-Stage-C) **or** self-distilled (pre-refit soft outputs); (iii) **ablate {raw, filtered+down-weighted, distilled}, select on CAL** (never SEALED) by off-diagonal metric + Anima-vs-real precision gap. Non-group replay columns keep original labels. Balance beats volume (Kirichenko); 5–10 refit seeds, average head weights.
- **Features once:** TUNE + replay pool at 448, `tag_head` hook, iterate on features (~6–8K images incl. replay).
- **Double-count trap:** the §4 prior-shift intercept applies **once, after all refits** (both refits re-derive head biases).
- **Escalation (default OFF):** feature-limited group on the SEALED sibling-gap/off-diagonal metric (expected: `medium_hair`) → do **not** unfreeze; route to the DINOv2 track (§7.1; full spec `git show 2c1cfa1:todos/ordinal-fusion-track.md`). Backbone FT on ~6K gold prohibited — degrades the real-domain calibration §5 depends on.

**Prerequisite code:** gold-dir loader bypassing the 95/5 auto-split (= V2.1.1 §3); **off-diagonal confusion metric** (V2.1.1 §4 — needed here first; the per-group gate); feature-cache + head-refit script; **author `configs/cleaning_groups.json`** — `configs/confusable_groups.json` has 4 of 6 groups (hair_color, eye_color, hair_length, breast_size; no multi-color set, no penis axis; telemetry semantics — keep for telemetry). New registry: per-group rule-type `{nominal-swap | label-set | PU-add-only | ordinal-columns-only}` + allowed edges ⇒ the rule runner cannot silently pick up ordinal edges (§5).

---

## 4. Stage D — Calibration and thresholds

1. **Per-GROUP temperature/vector scaling** (beta calibration if non-sigmoid reliability) + **per-tag additive intercept**, merged with the prior-shift offset into **one** per-tag logit shift. **No per-tag isotonic** (v2-plan §8.3; CAL has ~10–65 positives/tag). Too few real CAL positives → flagged, rules → human queue (§6.1 min-n). Untouched tags: least-noisy labels available (portal reals; flagged noisy-val).
2. **Prior shift — ρ-corrected target prior; closed-form offset demoted to initialization.** Offset `−log P_gold(y) + log P_corpus(y)` (Saerens 2002/Menon 2013) fails twice: (a) **understated sidecar prior** (ρ=25% ⇒ ~−0.29-logit one-directional error) → use `P̂_true(y) = P_sidecar(y) / (1 − ρ̂_decile)` (or EM/BBSE on calibrated scores, Alexandari 2020); (b) **conditional, not label, shift** (boundary-oversampled gold) → offset is initialization/diagnostic only; log offset-implied vs real-slice-implied threshold divergence per tag as a gate input.
3. **Thresholds fit for target precision, validated on REAL data.** τ_add (M1) / τ_swap (grouped swaps): CAL-real portal images + CAL share of the random core + accumulating §6 audit labels. Within-group δ margins / sibling rankings: boundary strata + adjudicated disagreements. Gold funds no CAL quantity; insufficient real n ⇒ undecidable → human queue (§6.1). (Anima absolute thresholds non-transferable: locked generator, single style scaffold; calibration degrades under covariate shift — Ovadia 2019.)
4. **Winner's-curse guard:** choose τ so the CAL **Jeffreys/Clopper–Pearson lower confidence bound** clears the bar — not the point estimate (point-estimate winners fail one-shot SEALED at ~coin-flip rates). Report the recall cost.
5. **Capped recalibration — EXACTLY ONE** (definition of the §7.2 fence): bad SEALED-vs-CAL divergence ⇒ **one** recalibration on accumulated real audit labels, then human queue. No refit-and-recheck loops (peeking; one-pass discipline).

---

## 5. Stage E — Corpus scoring and candidate generation

**Scoring run:** frozen janitor (interpolated pos-embeds + adapted head + refit group rows + calibration + prior offsets), all ~5.92M **@ 448px**, no augmentation. **Flip-TTA default OFF** — decide by measuring gate precision on CAL at 448 with TTA on vs off (a minutes-scale check on cached scores, not a training experiment); it doubles a real bill.

**Scorer build (a build):** Inference_Engine.py:605-613 force-overrides `config.image_size` back to the trained 320 (violates never-mix-resolutions) → load via CheckpointManager into a **448-config SimplifiedTagger directly** (bypassing ModelWrapper's preprocessing override), reuse InferenceDataset/preprocess_batch, new parquet emit stage. Mask special indices `{PAD_idx, UNK_idx}` **resolved from the vocab** — never `logits[:, 2:]` slicing (silent off-by-two incl. the asl_telemetry group index tensors); parquet stores **original vocab indices**; fp32 sigmoid. Days-scale (§8 item 4).

**Cost:** ~315 GFLOP/image at 448 → ~100–200 img/s on the RTX 5090 (bf16, no grads, batch ≥192) → **~10–18 h per full ~5.92M pass**; the early-checkpoint pass is a candidate-subset re-score only.

**Storage — threshold-relative CALIBRATED scores, not raw top-K.** Calibration + offsets applied **in-line**; persist per image: (a) all cleaning-group columns, (b) all GT=1 columns, (c) every (tag, p_cal) with `p_cal > τ_add(tag) − 0.1` slack ⇒ thresholds revisable post-SEALED without a re-score. Raw top-K would starve M1 (offsets reorder tags across images; raw-rank-300 can be the top calibrated candidate); calibrated top-K = diagnostic only. Parquet by image id; ~10–15 GB.

**Early-checkpoint cross-check — candidates-first, dense gather, required.** Plateau/EMA emits M1 candidates first; the early checkpoint (§2 delta 3, own interpolated pos-embeds, identical 448 setup) scores **only candidate-bearing images**, gathering **the candidate (image, tag) cells dense from its full logits** (a same-schema top-K would miss the suppressed tags this signal exists to rescue). Relative boost only, never thresholded alone — but a **required input** of the M1 rule.

**In-sample caveat:** CL prescribes out-of-sample probabilities; K-fold janitors unaffordable. Mitigations: §2.4, §2.5, §4, and the early-checkpoint cross-check above. Report as a known bias source + report **estimated recall** per scorer (fraction of SEALED-real adjudicated adds surfaced) — precision-only hides the memorization failure mode. SEALED-real images sit in the P1 train set = the deployment condition (the janitor only ever scores its own train set), so the gates measure the deployed setting.

### Candidate lists, per noise mode — rule types from the cleaning registry (§3.3)

| Mode | Rule | Signal strength |
|---|---|---|
| **M1 missing positive** (add) | GT=0, calibrated p > τ_add(tag); AUM + early-checkpoint boost required inputs — suppression-resistant evidence | **Strongest.** This is the mode the model's whole training run is aligned with; the ρ measurement predicts list sizes per decile |
| **Wrong positive, confusable swap** — **NOMINAL color groups only** (`hair_color`, `eye_color`) | GT sibling *i*=1 within the group, but calibrated p(i) low AND p(j) of one sibling high (margin δ fit on CAL-real boundary strata, §4.3); emit as a **swap proposal** i→j, never a bare removal | **Medium.** Gold-refit head + CL-style ranking (cleanlab multi-label on the group columns as the ranker); evaluated per §6 (synthetic corruption + real audit) |
| **Ordinal groups** (`hair_length`, `breast_size`) | **NO auto-apply swaps** — adjacent ordinal swaps are M2 in disguise (the janitor adjudicating its own convention against convention-shared Anima ordinal gold — the §7.1 fence applies). Emit calibrated columns for the DINOv2 fusion track (the ordinal-fusion track (git history `2c1cfa1`; §7.1)); at most human-queue proposals for FAR (non-adjacent) disagreements, which are gross mislabels rather than convention calls | **Columns + queue only** |
| **Penis SIZE axis** | **PU add-only** ([golden_set_plan §4.3](../.research/golden_set_plan.md): 92.2% unlabeled-normal — there is no "normal" tag to swap to; swap semantics structurally inapplicable) | **M1-style only** |
| **Penis TYPE axis** (`animal_penis` ⊃ `horse_penis`/`dog_penis`/`knotted_penis`; `penis_tentacle` separate) | **Generic M1 add path — no special handling, and the size row's PU exemption does NOT extend here**: unlabeled type is a TRUE negative (2026-08-10 200k-sidecar scan: 98.5% of `penis` images carry no type tag; default = human). Danbooru implications are fully materialized in the sidecars (0/119 subtype-without-umbrella; `canine_penis` is a Danbooru alias→`dog_penis`) → **no hierarchy-completion rule to build**. **NO swaps, structurally**: `knotted` is a feature not a species (85% of `dog_penis` co-carries it) and species↔species n is unfundable (SEALED ~5K ⇒ <2 subtype positives → §6.1 undecidable). CAL ≈ 72/34/24/20/9 positives (animal/horse/knotted/dog/tentacle): umbrella rides the generic M1 gate; subtype adds and species-specification of the ~17% unspecified-umbrella pool (~700 images) are human-queue per §6.1 | **M1-style only; umbrella fundable, subtypes human-queue** |
| **Multi-color set** | **Label-set track**: set-overlap rule (missing/extra member), not pairwise swap | **Medium; per-set gate** |
| **Wrong positive, non-grouped** | GT=1, calibrated p ≪ τ (bottom-percentile self-confidence), cleanlab + AUM ranking | **Weak-to-medium.** Human-queue only |
| **Ordinal convention bias** (M2: reflexive `long`, `medium` starvation) | **OUT OF SCOPE for this model** — diagonal blind spot. Emit the group columns to feed the external-anchor (DINOv2) fusion track (§7.1; full spec `git show 2c1cfa1:todos/ordinal-fusion-track.md`); do not let the janitor adjudicate its own convention | **Structurally unreachable from inside** |

---

## 6. Stage F — Two-stage gates, journaled writeback, verification, discharge

### 6.1 Gate statistics (applies to every gate below)

At n=30 the binomial band (~±0.12) dwarfs ε_gold (~0.02–0.03), so every gate carries a sampling-error term:

- Every gate is a **one-sided 95% Clopper–Pearson (or Jeffreys) LOWER confidence bound** on measured precision, then gold-error-corrected via Rogan–Gladen (§3.1): `p_true = (p_L − α) / (1 − α − β)`.
- **Minimum-evidence rule:** a rule whose evaluable decision count cannot mathematically certify its bar even at zero errors (n < 29 for a 0.90 bar; n < 59 for 0.95 — from `0.05^(1/n)`) is **ineligible for auto-apply** and pre-registered human-queue-only. Reported as **"undecidable," never "failed."**
- Gates pool **per-GROUP** (never per-edge below the n floor — 548 hair-color edges vs ~400 SEALED images makes per-edge gating arithmetic nonsense).
- **Damage-weighted multiplicity control:** rules proposing >100K edits certify at the **99%** lower bound (α=0.01); small rules keep 95%. This bounds expected bad-edit *volume* — the actual loss function — without Bonferroni-starving every small gate.

**Swap-rule evaluation needs synthetic corruption.** On truth-labeled SEALED, any swap-rule firing is by construction a wrong proposal (the GT sibling *is* correct there), so organic SEALED swap "precision" measures ≈0. Build the swap-eval set by **synthetic corruption**: take truth-labeled SEALED-real images (adjudicated/audited labels) of true class *j*, flip the presented sidecar label to sibling *i* (sampling flips from the corpus confusion prior), and score the rule's proposals against the adjudicated truth — this manufactures per-group n from the real-slice boundary strata. Caveat, reported: random flips are easier than real correlated errors, so synthetic precision is optimistic — the §6.2 real pre-writeback audit remains the certifying gate.

### 6.2 Two-stage gate structure

| Stage | What it does | Data |
|---|---|---|
| **QUALIFYING (SEALED)** | Eligibility, threshold verification, within-group structure, per §6.1 statistics. **SEALED is real-only (~5K) and carries all qualification**; rare siblings the real data can't fund are undecidable → human queue, never Anima-backfilled. Gemma-derived labels may **qualify** but never **certify**: the janitor and Gemma could share an error mode on real art, and a shared mode must not self-certify | SEALED, one shot |
| **CERTIFYING (pre-writeback human audit)** | Stratified human review of each qualified rule's **actual corpus candidates** — full spec immediately below | Real corpus output |

**Certifying-audit spec.** For each qualified rule, review the rule's actual corpus candidates in the portal **before any sidecar write**. The human look sits *before* the writes — the strongest cheap protection against the plan's one irreversible failure mode (replaces an earlier post-gate random-edit audit).

- **Quota:** fixed per tag-frequency decile — **40 per super-decile, ≥200 total; 500 for rules proposing >100K edits**. A uniform sample is blind in exactly the tail deciles where the ρ measurement says the noise lives.
- **Bar:** auto-apply only if the audit's one-sided exact-binomial lower bound (**95%; 99% for >100K-edit rules**) clears the bar *after* §3.1 gold/reviewer-error correction.
- **Pre-register the rejection threshold as a COUNT**, never a proportion-vs-band comparison. Illustrative: n=200 against a 0.95 bar at α=0.01 → **reject at ≥19 failures**.
- **Reuse:** the audit labels double as real-domain calibration data (§4.3/§4.5).

| Action tier | Bar | Disposition |
|---|---|---|
| **Auto-apply** | Certifying-audit lower bound ≥ 0.95 (M1 adds) / ≥ 0.90 (nominal-group swaps), AND per-tag volume sanity vs the ρ/decile prior | journaled writeback (§6.3) |
| **Human queue** | 0.7 ≤ bound < bar, or any non-grouped removal, or "undecidable" per §6.1 | portal / human review queue, ranked by score; budget-capped |
| **Drop** | below 0.7, or tag has no gold coverage and no calibration | logged in the report, not acted on |

### 6.3 Journaled writeback

The DataCleaning-Project applier (`apply_corrections_fast.py`) has **no backups and no DB writes** — its own line-1 docstring says so. (`json_backups` rollback exists only in a *different* pipeline the applier never touches: `datacleaning/pipeline3_deletion/backup/manager.py`, `pipeline1_ingest/cli.py`. Do not read it as pre-existing reversibility.) Reversibility is therefore new code, and an *intent* log cannot invert edits (rolling back a no-op add would delete a pre-existing tag). Build `tools/janitor_apply.py`, porting the one genuinely reusable part — the **format-preserving sidecar editor**: BOM/encoding detection (`apply_corrections_fast.py:34-60`), indent detection (`:66-82`), trailing-newline preservation (`:156-157`); plus `core/tag_normalizer.py`'s `normalize_tag` (`lower().strip().replace(' ','_')`) — use it everywhere. Reference DB: `DataCleaning Project/datacleaning/data/datacleaning.db` (schema `database/schema.sql`).

- **Append-only APPLIED-delta journal** (NDJSON): one record per **actual mutation** — `{image_id, tag, op(add/swap_from/swap_to), rule_id, calibrated_score, model='janitor-v1', registry+calibration version, batch_id, full pre-edit tag list, timestamp}` — no-ops logged as no-ops.
- **Idempotent and resumable** (~5.92M small NTFS writes take hours and can be interrupted mid-flight).
- **Rollback = inverse replay** of the journal filtered by rule_id/batch_id, using the recorded pre-edit tag lists. Recovery runbook: rollback → cache rebuild → restart next run.
- **Retention:** journal + pre-clean state retained until the NEXT production run's first Anima-canary and per-decile-EPR evals look sane.
- **Additive bias preserved:** removals/swaps only inside nominal groups that cleared their gate; everything else stays additive (consistent with the Tag Improvement Pipeline precedent).
- **Cache:** sidecars are edited **between runs only** — cleaned corpus → one Arrow cache rebuild → next model. (Consistent with the build-once cache decision; no mid-run edits exist in this design.)

### 6.4 Discharge checklist

1. Cleaning report: per-rule qualifying + certifying precision bounds (per domain), volumes applied/queued/dropped/undecidable, **estimated recall per scorer** (§5), per-decile ρ before/after estimate, off-diagonal confusion table before/after on SEALED.
2. Checkpoints + refit heads + calibration tables + journal archived (cold storage — kept so a bad edit batch can be re-derived/rolled back), then the model is **retired: it seeds nothing.** The next production model is from scratch on the cleaned corpus, per the normal V2 plan.
3. Post-mortem note into the V2 plan: measured ρ after cleaning decides whether the next run's clip decision re-opens (a materially lower ρ strengthens the clip=0.1 case) and whether a γ contingency becomes viable (the sealed slice now exists and has been exercised).
4. **Next-run launch gate:** the from-scratch production run does not start until the discharge report is reviewed (user decision point #3) — the catastrophic path is a bad decile discovered weeks into the next P1 with the journal already discarded.

---

## 7. Scope fences (what this plan refuses to claim)

1. **Cannot clean its own convention (M2/ordinal)** — the diagonal blind spot (§0). `medium_hair` recovery + ordinal boundary relocation belong to a **separate program** (spec split out in `2c1cfa1` 2026-07-29, retired from the tree 2026-07-30 — recover via `git show 2c1cfa1:todos/ordinal-fusion-track.md`; **the five lines below are the living summary**); the janitor contributes calibrated columns to that fusion, not judgment (enforced in §5: ordinal groups are columns-and-queue only, no auto-apply swaps). That program, in five lines:
   - Signals: **M-D** frozen-DINOv2 CORN/SORD ordinal specialist · **M-B** entity-canonical directional-UPPER consensus · **community-DELTA** · **M-A** geometry (optional confirmer) · **M-E** human.
   - Auto-apply **only when ≥2 orthogonal signals agree in-direction** on a low-ambiguity item.
   - **Medium recovery is gold-and-specialist-bound** or it does not happen (medium false-removal floor = 0).
   - Fusion posterior from the orthogonal signals — **never tagger consensus** (tagger-fed posteriors put the mislabels on the diagonal, so they never queue).
2. **Synthetic domain shift managed, not solved — contained to TUNE.** Everything decision-facing runs on real data (§3.2, §4); residual exposure is the §3.3 CAL refit-variant ablation, and divergence is handled by the capped recalibration defined at §4.5.
3. **No loss-side cleaning claims.** Does not modify ASL; composes with the noise-robust-loss track (v2-plan §4 owns that scope).
4. **One pass.** No self-training loops (confirmation-bias amplification, no fresh anchor) — one model, one pass, one report, discharge (§6.4).

## 8. Sequenced task list

1. ☐ Shared blockers: V1 γ-history inspected; **residual population ρ remains unmeasured** (v2-plan §10). Loss-state persistence is committed (`a872e9f`). Fixed-loss verification is implemented: conflicting checkpoint gamma aborts, and the audit/resume regressions check the live fixed criterion. The remaining shared V2 launch gates (final dataset preparation, campaign-specific controls and full-scale validation) remain in v2-plan §12.

2. ☐ **Author `configs/cleaning_groups.json`** per §3.3 — *not* "confirm coverage"; freeze the registry version for this pipeline.
3. ☐ **Build the 20K real slice** per §3.1.2, then freeze and manifest-commit the §3.2 splits **before Stage A**, with the undecidable list pre-registered and the Anima spot-audit done per axis. Sizing datum not in §3.1.2: the Gemma per-axis labeling pass is **~1 GPU-day**. *(Supersedes the earlier ~300–500 random-slice task.)*
4. ☐ Code per §3.3 (gold-dir loader, off-diagonal metric, feature-cache + head-refit script), §4 (calibration/threshold fitter), §5 (corpus scoring harness, candidate-rule runner) and §6.3 (`tools/janitor_apply.py` + applied-delta journal). Effort datum: the scoring harness and the applier are each **days-scale**.
5. ☐ Stage A run (§2 deltas) → branch-point checkpoint selection (§2 delta 2).
6. ☐ Stage B′ → C → D → SEALED qualifying gates → **pre-writeback certifying audit** (§6.2); gate+audit report reviewed by user before any auto-apply write.
7. ☐ Stage E–F → journaled writeback (§6.3) → cache rebuild → discharge report (§6.4).

**User decision points (three):** (1) sign-off on the qualifying-gate + certifying-audit report before auto-apply writes (step 6→7), (2) the discharge post-mortem's clip/γ recommendation for the next run, (3) **next-run launch** after discharge-report review (§6.4.4).
