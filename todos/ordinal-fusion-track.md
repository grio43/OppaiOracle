# Deferred Ordinal Track — DINOv2 External-Anchor Fusion (M2 / ordinal cleaning)

**Date:** 2026-07-29 · **Status:** Deferred program spec. Relocated from [janitor-cleaning-model-plan.md](janitor-cleaning-model-plan.md) Appendix A (originally `.research/_wf2_synthesis.md`, `36dc11e`).

**Why this is a separate program, not a janitor stage.** The janitor is a Phase-1 model trained on the noisy corpus, so it *learned the convention being cleaned* — it is structurally blind to the diagonal (golden_set_plan §2.2). A frozen backbone + head refit moves thresholds, not feature geometry, so no amount of janitor tuning reaches M2 (reflexive `long`, `medium` starvation) or ordinal boundary relocation. That work needs an anchor **outside** the booru label distribution, which is what this track is. The janitor plan therefore declares it out of scope in §0 and §7.1 and contributes only **calibrated ordinal columns** to the fusion — scores, not judgment.

Interface with the janitor: janitor plan §5 emits calibrated `hair_length` / `breast_size` columns (no auto-apply swaps, human-queue proposals for FAR/non-adjacent disagreements only); janitor plan §3.3's escalation path routes a feature-limited group (expected: `medium_hair`) here rather than unfreezing the backbone.

---

## 1. Signals and roles

A gold-anchored, direction-aware fusion of orthogonal-error signals:

- **M-D — DINOv2-frozen-backbone + CORN/SORD ordinal specialist**: the corpus-scale workhorse. Frozen DINOv2 ViT-L/14 features (registers; not CLIP — DACoN 2025: 57.5% zero-shot anime line-art part-matching vs CLIP 36.7%), never trained on tags = the orthogonality anchor; CORN rank-monotone head, SORD soft targets; gold grown via BADGE/core-set active learning mining boundary + medium cases.
- **M-B — entity-canonical directional-UPPER consensus**: near-free first pass. Character bags (Danbooru category-4 tags; exclude the alternate-hairstyle/wig/cosplay AU set; abstain on bags <10–15 or bimodal); a character is canonically ≥`very_long` iff its non-AU very_long rate clears a gold-calibrated threshold. **Aggregate UPPER, not mode — "the mode launders the bias"** (the unidirectional-DOWN annotator bias structurally cannot fabricate `very_long`; noisy-OR aggregation). Fenced to upgrade-only; never strips medium.
- **Community-DELTA**: re-pull current Danbooru tags via post-id (`id_index.json`); **use the delta (current minus snapshot), not static current tags** (same-site same-convention; only the change-over-time is editor-attention-driven and orthogonal). Weight by edit recency + distinct-editor count; canonicalize via `tag_aliases`, enforce `tag_implications` closure on write.
- **M-A geometry**: optional confirmer only, on the ~9.4% cleanly-measurable slice (solo + longish + full_body/cowboy_shot, not tied/back/seated/chibi); abstains aggressively; its *downward* occlusion artifact opposes the up-bias — brackets rather than launders. Build last or skip.
- Fusion posterior from the **orthogonal signals, never tagger consensus** (tagger-fed posteriors put M4 mislabels on the diagonal — they never queue). Auto-apply only when ≥2 orthogonal signals agree in-direction on a low-ambiguity item.

## 2. Confusion-cell ownership (verbatim, synthesis §2.6)

| Cell | Primary | Confirmer | Direction policy |
|---|---|---|---|
| long↔very_long | M-B-upper + M-D | community-delta, M-A(clean slice) | ADD very_long, KEEP long; auto only on ≥2-signal agree |
| medium↔long | **M-D only** | human (M-E) | recover medium; protect existing medium from removal |
| short↔medium | **M-D only** | human (M-E) | recover medium; AL must mine these |
| very_long↔absurdly | human/fusion | — | never auto-write fine call |

## 3. Medium recovery (§2.7)

"Medium recovery is **gold-and-specialist-bound or it does not happen**." Only M-D + medium-oversampled human gold can recover it; M-B, community-delta, and all Danbooru-trained taggers under-tag medium and are fenced out. Protect existing `medium_hair` from removal (rare-class protection in routing; medium false-removal floor = 0).

## 4. Gold SPOF

No signal is individually sufficient — all five are *transducers that spend an external gold anchor across the corpus*. The program lives or dies on **rubric-sampled (not convention-sampled) gold**: rubric-first labeling (reviewers never see existing tags), boundary + medium deliberately oversampled (never proportional), a held-out Goodhart slice the pipeline never sees, honeypot/MCC reviewer gating.

---

## Appendix — Danbooru-trained tagger inventory (FENCED OUT of the fusion posterior)

Relocated with this track from janitor plan Appendix B (`48302d4:handoff.md` §2). **These are the taggers §1 and §3 fence out**, recorded here so the fence names locatable assets rather than abstractions. They are inventory for the M4 / DataCleaning consensus program; they may **never** feed this track's fusion posterior (§1, last bullet), because all three are Danbooru-trained and share the convention bias being corrected — consensus kills random per-model error but only partially escapes the systematic bias, and all three under-tag medium (§3).

| Tagger | Path | Input | Output | Vocab |
|--------|------|-------|--------|-------|
| **WD-ViT-Large-v3** (SmilingWolf) | `DataCleaning Project/datacleaning/models/wd-vit-large-tagger-v3/model_int8.onnx` (+`selected_tags.csv`); also `OppaiOracle/exported_model/wd-vit-large-tagger-v3/` | 448px, **white pad, RGB→BGR**, [0,255] | sigmoid | 10,862 |
| **ML-Danbooru** (TResnet-D, deepghs) | `DataCleaning Project/datacleaning/models/TResnet-D-FLq_ema_6-30000.onnx` (+`tags.csv`) | 512px, black pad, [0,1] | **raw logits → sigmoid** | 12,548 |
| **Camie-Tagger-v2** (p1atdev) | `OppaiOracle/.research/camie/camie-tagger-v2.onnx` (+`camie-tagger-v2-metadata.json`) | 512px | sigmoid, per-category thresholds | 70,527 |

Camie-v2 is the most valuable vote for that other program (2024 snapshot, IRFS long-tail sampling). Runner exists: `DataCleaning Project/datacleaning/core/ml_danbooru_inference.py` (`MLDanbooruInference.infer_batch` / `decode_predictions`); WD-ViT is wired as reference baseline in that project's `compare_models.py`. The remaining DataCleaning inventory (`pipeline0_baseline/thresholds.py` adaptive floors, `pipeline1_ingest/tag_comparison.py` + `MUTUAL_EXCLUSION_GROUPS`, the `pipeline2_review/` FastAPI UI, correction-row NDJSON format) belongs to that same program and is recoverable from `48302d4:handoff.md` §2; neither this track nor the janitor plan uses it (the janitor's review UI is the Tag Review Portal, janitor plan §3.1.1).
