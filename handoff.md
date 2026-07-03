# Handoff — Ground-Truth Cleaning for Ordinal/Perceptual Tag Noise

**Date:** 2026-06-08
**Owner focus (this handoff):** clean the **ground truth**, not the model. The model/loss/inference fixes are downstream and secondary — they cannot fix what the labels get wrong.
**Scope:** hair-length ordinals first (`very_short → short → medium → long → very_long → absurdly_long`), then hair/eye colors.

---

## 0. The mission, in one paragraph

The dataset (~6M Danbooru-sourced images at `L:/Dab/Dab/shard_*/*.json`) has a **deep, systematic ground-truth defect** in ordinal tag groups. The headline case: **a large fraction of images tagged `long_hair` are actually `very_long_hair`** (and `medium_hair` is routinely mislabeled `short`/`long`, fine refinements are omitted, etc.). This is *active mislabeling*, not just missing labels. Missing labels are a known, tolerable secondary problem; **wrong labels are the priority** because they teach the model the wrong answer and then the model launders that bias back into every suggestion. We must repair the GT, feed the cleaned labels to the v2 retrain, and iterate.

---

## 1. The problem is THREE distinct defects — and the worst one is invisible

The earlier audit (`.research/tag_noise_audit.py`, 44,983 solo images) found three modes. **The mode the user is calling out is M4 below, which the audit could not directly see.**

| ID | Defect | How it looks in GT | Audit signal | Severity |
|----|--------|--------------------|--------------|----------|
| **M1** | **Missing label** | image has *no* hair-length tag | ~20% of solo chars have none | secondary (known/tolerable) |
| **M2** | **Visible contradiction** | image has ≥2 length tags | 18.5% of present; 92% adjacent | visible, partly legit (implication chain) |
| **M4** | **Silent ordinal mis-assignment** ⚠️ | image has **exactly one** length tag, and it is the **wrong** one (`long` when truly `very_long`) | **INVISIBLE** — looks clean to every missing/contradiction metric | **PRIMARY — biggest + hardest** |
| (M3) | Perceptual color confusion | aqua/blue/green | inference-time, mostly not a GT defect | handle at decode, not here |

**Why M4 is the hard one:**
- It is undetectable by counting (one tag present, well-formed, passes implication checks).
- Its root cause is **annotator coarse-default bias**: `long_hair` is the reflexive default; annotators don't "upgrade" to `very_long_hair`; `medium_hair` is skipped (corpus shows `long` 24,032 vs `medium` 2,083 vs `short` 10,610 on solo — the middle of the scale is collapsed). The boundary `long` vs `very_long` has **no crisp definition**, so disagreement is structural.
- **The v1 OppaiOracle model learned this exact bias from this exact GT.** It will *also* under-predict `very_long`/`medium`. So **the model cannot be the sole oracle for finding its own bias** — that is circular. Breaking the circularity is the central design problem of this handoff.

**Implication structure (must respect during repair):** `very_long ⇒ long`, `very_short ⇒ short`, `absurdly ⇒ very_long`. The chain is clean in GT (99.98%). So the correct fix for "tagged `long` but is `very_long`" is **ADD `very_long_hair` and KEEP `long_hair`** — never remove `long`. Only the **base triple `{short, medium, long}` is mutually exclusive**; the modifiers ride along via implication.

---

## 2. Assets already on disk (reuse — do not rebuild)

### 2a. Three architecturally-independent taggers (the circularity-breakers)
All cover the 6 hair-length ordinals + colors. **All are Danbooru-trained**, so they share the *convention* bias to a degree (see caveat) — but their *architectural/idiosyncratic* error is independent, and their **consensus disagreeing with GT** is a strong triage signal.

| Tagger | Path | Input | Output | Vocab |
|--------|------|-------|--------|-------|
| **WD-ViT-Large-v3** (SmilingWolf) | `DataCleaning Project/datacleaning/models/wd-vit-large-tagger-v3/model_int8.onnx` (+`selected_tags.csv`); also `OppaiOracle/exported_model/wd-vit-large-tagger-v3/` | 448px, **white pad, RGB→BGR**, [0,255] | sigmoid | 10,862 |
| **ML-Danbooru** (TResnet-D, deepghs) | `DataCleaning Project/datacleaning/models/TResnet-D-FLq_ema_6-30000.onnx` (+`tags.csv`) | 512px, black pad, [0,1] | **raw logits → sigmoid** | 12,548 |
| **Camie-Tagger-v2** (p1atdev) | `OppaiOracle/.research/camie/camie-tagger-v2.onnx` (+`camie-tagger-v2-metadata.json`) | 512px | sigmoid, per-category thresholds | 70,527 |

Runner already exists: `DataCleaning Project/datacleaning/core/ml_danbooru_inference.py` (`MLDanbooruInference.infer_batch` / `decode_predictions`). WD-ViT is already wired as a reference baseline in the project's `compare_models.py`.

> **Do NOT use OppaiOracle V1.1 as the oracle** (`DataCleaning Project/datacleaning/models/oppai/V1.1/`). Same bias as the GT we're fixing. It may be a *down-weighted* 4th vote at most, never a tiebreaker.

> ⚠️ **Caveat that defines the whole approach:** all three external taggers were trained on Danbooru, where the same `long`-as-default convention lives. So multi-tagger consensus kills **random** per-model error but only **partially** escapes the **systematic** convention bias. Camie-v2 is the most valuable vote here — newer (2024 snapshot), long-tail-aware sampling (IRFS), so it is the least likely to share the under-tagging of `very_long`/`medium`. **The only true bias-break is a human-labeled gold set (§4, Phase 1).** Consensus is *triage*; humans *establish* truth.

### 2b. DataCleaning Project infrastructure (the review→correct→writeback machine — already built)
| Stage | Location | What it gives you |
|-------|----------|-------------------|
| **Baseline stats** | `pipeline0_baseline/` (`thresholds.py`, `baseline_tag_summary` table) | per-tag score distributions → adaptive confidence floors (`compute_missing_threshold` etc.). **Prereq: must be run so per-tag p75/median/stddev exist, else thresholds fall back to clamps.** |
| **Compare/score** | `pipeline1_ingest/tag_comparison.py` (`TagComparator.compare_tags`) | FN/FP/inability/`gt_suspicion_score`/`frequency_weighted_score`, and **`MUTUAL_EXCLUSION_GROUPS`** (extend with the base length triple) |
| **Review UI** | `pipeline2_review/` (FastAPI, `api.py`, `static/`) | per-image: thumbnail + GT tags + model predictions w/ confidence; queues by error type; add/remove + bulk corrections → `tag_corrections` table |
| **Writeback** | `export_corrections.py` → `apply_corrections_fast.py` | NDJSON corrections → **encoding/BOM/indent/line-ending-preserving atomic** sidecar edits, with `json_backups` rollback |
| **Reviewer QC** | `screening_*.py`, `screening_calibrate.py` | gold-set honeypots, MCC/specificity/recall per reviewer, isolated disposable DBs — **reuse directly for inter-annotator agreement + reviewer vetting** |
| **Normalization** | `core/tag_normalizer.py` (`normalize_tag`) | `lower().strip().replace(' ','_')` — reuse everywhere |

DB: `DataCleaning Project/datacleaning/data/datacleaning.db` (schema in `database/schema.sql`). Correction row format:
```json
{"analysis_id": 12345, "action": "add|remove", "tag_name": "very_long_hair", "namespace": 0, "created_at": "..."}
```

### 2c. The provenance lever (the GT JSONs are otherwise bare)
GT sidecars contain **only** `filename`, `rating`, `tags` — **no score, fav count, uploader, or post link**. There is no built-in trust signal. **BUT the filename stem appears to be the Danbooru post ID** (e.g. `1546994.jpg` → Danbooru post 1546994). **Verify this on a 20-file sample first.** If it holds, you can re-pull, for the *flagged subset only*:
- authoritative **current** Danbooru tags (the community keeps editing — current tags are often cleaner than the snapshot),
- the post **score / fav_count** (a real trust/quality signal the JSONs lack),
- the authoritative **tag implication table** (pull once, globally).

This is a powerful, cheap second non-model signal. Mind API rate limits — flagged subset only, never all 6M.

---

## 3. Strategy: consensus **triages**, the gold rubric **defines truth**, humans **decide**

The detector for M4 (silent ordinal mislabel) is:

> For each **solo** image with exactly one base-length tag in GT, compute each independent tagger's **argmax over the hair-length ordinal scale**. Flag when **≥2 independent taggers agree on a *different, adjacent* value than GT**, each above its per-tag confidence floor.

Two non-negotiables that make this work instead of laundering bias:

1. **A written, image-anchored boundary rubric (Phase 0 deliverable).** The root cause of M4 is that `long` vs `very_long` (and `medium` vs everything) has no operational definition. You must **write one** (e.g. *very_long = hair extends past hips/below; long = shoulder-to-hip; medium = chin-to-shoulder; short = above shoulder; very_short = above ears/pixie*) with example images. Without it, human reviewers disagree exactly as much as the original annotators did, and you re-inject noise. **This rubric is also the target the v2 model will learn — it is the single most valuable artifact this project produces.**

2. **Direction-aware trust, calibrated on a gold set.** The bias is *asymmetric*: under-tagging the fine/rare end. So "GT=`long`, consensus=`very_long`" is the **high-trust** direction (matches the known bias), while "GT=`very_long`, consensus=`long`" is **suspect** (could be the taggers sharing the under-tag bias). Quantify this with a **per-tagger, per-direction confusion matrix measured on the human gold set** — that matrix sets the auto-flag thresholds and per-direction trust weights, and *estimates the true prevalence of M4* (which the original audit could not).

---

## 4. Phased plan

### Phase 0 — Define truth (no images cleaned yet; this gates everything)
- [ ] Write `configs/tag_groups.yaml` (shared registry; promote the dicts in `.research/tag_noise_audit.py:32-76`): ordinal chain, implications, `exclusive_base: [short,medium,long]`, color adjacency, `hair_color.exclusive_base: false`.
- [ ] Write the **operational boundary rubric** (markdown + example images per class). Get the user to sign off — these are *their* definitions.
- [ ] Verify filename = Danbooru post ID on a 20-file sample. Pull the Danbooru tag-implication table once.

### Phase 1 — Gold calibration set (breaks the circularity)
- [ ] Human-label ~200–500 **solo** images **per ordinal class** against the rubric (use the existing `screening` gold-set tooling as the harness). This is the only bias-free reference.
- [ ] Run all 3 (+ down-weighted OppaiOracle) taggers on gold → **per-tagger, per-direction confusion matrices**. Outputs: (a) real M4 prevalence estimate, (b) per-direction trust weights, (c) auto-flag thresholds, (d) reviewer honeypots.

### Phase 2 — Detect candidate mislabels at corpus scale
- [ ] Ensure `pipeline0_baseline` has run (per-tag floors exist for all ~30 group tags).
- [ ] New `pipeline4_ordinal_repair/`: score the solo corpus with the 3 taggers on the ~30 group tags; store each tagger's ordinal argmax + confidence. (Flip-TTA optional — length is flip-invariant — but remember flip shares the model's bias, so it's a stability check, not a bias check.)
- [ ] Detect: consensus-vs-GT adjacent disagreement, **direction-aware** (Phase-1 weights), confidence-gated. Three candidate types:
  - **FIX-UP** (`long`→add `very_long`): high-trust direction, KEEP `long` (implication). The main M4 repair.
  - **FILL** (no length tag → add consensus top-1 + implied parents): M1. **Known limit: defer `medium` recovery** — every Danbooru-trained tagger under-tags it; recover short/long/very_long now.
  - **RESOLVE** (≥2 base tags, M2): keep argmax winner only if it also clears its own absolute floor; else abstain.
- [ ] Prioritize the review queue by **expected error mass** (reuse `frequency_weighted_score` × group-noise-prior; hair-length ≫ eye-color).

### Phase 3 — Human review (establish truth, abstention-first)
- [ ] Extend `pipeline2_review` UI to show, per image: the rubric, GT, **and all taggers' ordinal votes + confidences** side by side. Reviewer picks the correct ordinal value.
- [ ] Every write human-confirmed. Below-confidence model/tagger suggestions must be **visibly marked "low-confidence/unverified"** — that is exactly where the system "strongly recommends incorrect tags"; unmarked, it re-poisons the GT.
- [ ] Gate reviewers + measure inter-annotator agreement with the `screening`/honeypot/MCC system before trusting their edits at scale.

### Phase 4 — Apply + iterate
- [ ] `export_corrections.py` → `apply_corrections_fast.py` (encoding-preserving, backed up). Corrections honor implications (ADD `very_long`, keep `long`).
- [ ] Implication closure as a separate near-auto pass (`very_long ∈ GT, long ∉ GT → add long`; 99.98%-safe).
- [ ] Feed cleaned GT to the v2 retrain. **The cleaned model becomes a better oracle next round** → re-run Phase 2 on the residual (co-teaching / iterative denoising). Track the gold-set confusion matrix shrinking across rounds as the success metric.

---

## 5. Do NOT do this (traps that re-inject noise)

- **Do NOT auto-apply any tagger's argmax to GT.** That launders model bias straight into the labels — the exact thing we're escaping. Taggers triage; humans decide.
- **Do NOT trust OppaiOracle v1 (or flip-TTA of it) to find its own bias.** Circular. Especially do not let it recover `medium_hair`.
- **Do NOT remove `long_hair` when adding `very_long_hair`.** `very_long ⇒ long`; both are correct. Only the base triple is exclusive.
- **Do NOT treat the two directions symmetrically.** `long→very_long` (fine-upgrade) is the trusted fix; `very_long→long` (downgrade) is suspect (shared bias) and needs stronger evidence / human eyes.
- **Do NOT clean without the written boundary rubric.** Reviewers will disagree as much as the original annotators and you'll spend labor adding noise.
- **Do NOT force color groups exclusive.** 71% of hair-color multi (black+red, black+white) and 74% of eye multi (heterochromia) are legitimate. Colors are detect-and-**abstain** only here.
- **Do NOT clean the full 6M blind.** Prioritize by expected-error mass; the clean long tail isn't worth review time. Log what you deprioritized.
- **Do NOT hammer the Danbooru API for all 6M.** Re-pull provenance only for the flagged subset; pull the implication table once.

---

## 6. Open decisions for the user
1. **Rubric boundaries** — confirm the operational `short/medium/long/very_long` definitions (§3.1). These are judgment calls only the user can anchor.
2. **Gold-set size & who labels it** — how many per class, and is it the user or vetted reviewers (the `screening` system can gate either).
3. **Provenance re-pull** — OK to hit the Danbooru API for the flagged subset using filename-as-post-ID? (verify the ID mapping first).
4. **Where `pipeline4_ordinal_repair` lives** — confirm it belongs in `DataCleaning Project/datacleaning/` (recommended; all the infra is there).

---

## 7. Cross-references
- Audit script + numbers: `.research/tag_noise_audit.py`, `.research/hairlen_noise_sample*.py`.
- Full 3-tier plan (decode-now / GT-repair / v2-loss), incl. the loss-side fixes that ride the v2 run: see project memory `project_tag_group_noise.md`. **This handoff is the deepened, top-prioritized version of that plan's Tier B**, because GT is the deep root.
- Loss correctness constraint for when v2 adds ordinal soft labels: re-binarize `targets_for_focal = (targets > 0.5)` at `loss_functions.py:269` (currently aliases). Not a GT task, but the v2 run that consumes this cleaned GT depends on it.
