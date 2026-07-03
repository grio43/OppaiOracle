# Agent prompt — update OppaiOracle HF model card draft

## Context

You are updating an anime-tagger model card draft. The user has just done a head-to-head comparison vs `SmilingWolf/wd-swinv2-tagger-v3` and `Camais03/camie-tagger-v2` and surfaced facts that are missing from the current draft. Your job is to fold those facts in without breaking the existing voice.

**Voice to preserve:** the current card is direct, doesn't oversell, openly discloses noise/bias, and uses one consistent operating point (P=R). Keep that. Don't add marketing language.

## Files to read first (in order)

1. [notes/huggingface_release_draft.md](notes/huggingface_release_draft.md) — the draft the user is editing. This is your write target.
2. [huggingface_release/V1.1_safetensors/README.md](huggingface_release/V1.1_safetensors/README.md) — the more complete of the two existing READMEs. Use as structural reference; don't blindly copy.
3. [huggingface_release/V1_safetensors/pr_thresholds.json](huggingface_release/V1_safetensors/pr_thresholds.json) and [huggingface_release/V1.1_safetensors/pr_thresholds.json](huggingface_release/V1.1_safetensors/pr_thresholds.json) — source of truth for the P=R numbers below. Both are at parity on the full 296,056-image held-out split. **Note:** if the V1.1 release file still shows `val_samples: 30000` (the pre-recompute snapshot from before 2026-05-09), the authoritative full-recompute file lives at [experiments/run1_vit/checkpoints/pr_threshold_last.json](experiments/run1_vit/checkpoints/pr_threshold_last.json) — same V1.1 checkpoint (epoch 7, step 85517), full val split, run on 2026-05-09 — use that file's numbers and flag the staleness in your diff summary.
4. [huggingface_release/V1.1_safetensors/config.json](huggingface_release/V1.1_safetensors/config.json) — architecture spec.
5. [huggingface_release/V1.1_safetensors/selected_tags.csv](huggingface_release/V1.1_safetensors/selected_tags.csv) — confirms all 19,294 tags are category 0 (general).

Ask the user before assuming the draft should match the V1.1 README structure — it may diverge intentionally.

## Facts to add (each with source)

### 1. Head-to-head comparison table

The current card says "best P=R I've measured" but shows no competitor numbers. Add a section "Comparison vs other open anime taggers" with:

**Re-read both `pr_thresholds.json` files before writing this section.** Both V1 and V1.1 are now on the full 296,056-image val split with the same threshold sweep (0.001 → 0.999, step 0.001), so the numbers below are stable at the values quoted — but pull from the file at execution time as a sanity check.

**Macro-F1 at P=R (the metric WD v3 publishes):**
- OppaiOracle V1.1 (support≥1): 0.646 — `macro_single_threshold.support_ge_1.pr_breakeven.f1`. On the 296K val split, support≥0 / ≥1 / ≥5 collapse to essentially the same number (0.646 in all three) because 19,290/19,292 non-PAD/UNK tags have ≥5 positives.
- OppaiOracle V1.1 (full vocab): 0.646 — `macro_single_threshold.support_ge_0.pr_breakeven.f1`
- OppaiOracle V1: 0.588 — `V1_safetensors/pr_thresholds.json` `support_ge_0`
- camie-tagger-v2 (Macro-OPT, thr 0.492): 0.506 — model card on HF
- wd-eva02-large-tagger-v3: 0.4772 — model card
- wd-vit-large-tagger-v3: 0.4674 — model card
- wd-swinv2-tagger-v3: 0.4541 — model card

**Micro-F1 at P=R:**
- OppaiOracle V1.1: 0.699 — `micro.pr_breakeven.f1`
- camie-tagger-v2 (Micro-OPT, thr 0.614): 0.673 — *flag that this is at a different threshold from his macro headline*
- OppaiOracle V1: 0.659 — `V1_safetensors/pr_thresholds.json` `micro.pr_breakeven`
- WD v3: not reported

If you find the V1.1 file at `huggingface_release/V1.1_safetensors/pr_thresholds.json` still has `val_samples: 30000` (pre-recompute snapshot), use the authoritative full-recompute file at [experiments/run1_vit/checkpoints/pr_threshold_last.json](experiments/run1_vit/checkpoints/pr_threshold_last.json) and flag in your diff summary that the release file needs to be synced.

### 2. Vocabulary clarification

Add a short sub-table. The "70K" headline for Camie is misleading because most of it is named-entity tags. Apples-to-apples general-tag vocab:

| Model | General tags | Total |
|---|---|---|
| OppaiOracle V1/V1.1 | 19,294 | 19,294 (100% general) |
| camie-tagger-v2 | 30,841 | 70,527 |
| wd-vit-large-tagger-v3 | 8,106 | 10,861 |

Sources: your `selected_tags.csv` (all category 0); WD's `selected_tags.csv` in `exported_model/wd-vit-large-tagger-v3/` (8,106 cat-0 + 2,751 cat-4 + 4 cat-9); Camie's model card.

### 3. Architecture spec

Currently absent. Add a brief block citing `V1.1_safetensors/config.json`:
- ViT, 18 layers, 1024 hidden, 16 heads, 4096 FFN, patch 16
- drop_path 0.2, attention_dropout 0.05, hidden_dropout 0.1
- ~247M params total (rough — verify by counting if the user wants exact)
- Compare to WD swinv2-base ~99M and Camie v2 143M — note that the lead is *not* at parameter parity.

### 4. Eval set methodology

Current card says "my evaluation set" and stops. Add:
- V1: evaluated on 296,056-image held-out split (`V1_safetensors/pr_thresholds.json` → `val_samples`)
- V1.1: same 296,056-image held-out split as V1 (full-val recompute on epoch 7, step 85517, completed 2026-05-09). Pull `val_samples` from the file at execution time as a sanity check; if it still shows ~30,000 the V1.1 release file is the pre-recompute snapshot, fall back to [experiments/run1_vit/checkpoints/pr_threshold_last.json](experiments/run1_vit/checkpoints/pr_threshold_last.json).
- Threshold sweep: 0.001 → 0.999 step 0.001
- `skip_indices`: [0, 1] — `<PAD>` and `<UNK>` excluded from metrics

### 5. Reframe the early-stop section

The current "Why V1.1 stopped at 6 of 15 planned epochs" reads as defensive. Reframe it as a **deliberate noise-robust choice**, not a regret. The argument:

- mAP growth decelerated +0.7%/epoch → +0.3%/epoch by epoch 5.
- γ_neg=7.0 + reduced regularization is the WSML-literature regime most exposed to missing-positive memorization (Park et al. RAL ICCVW 2023; Liu 2020 ELR; Kim 2022 Large Loss Matters).
- Validation has the same missing-positive pattern as training, so a model that learns to suppress unlabeled-but-correct tags will *raise* noisy-reference mAP even after true ranking quality plateaus.
- Competitors who report higher convergence almost certainly trained past this phase transition (WD: 50+ epochs; Camie: not disclosed). So the comparison is between OppaiOracle's pre-memorization checkpoint and their likely-post-memorization checkpoints — meaning the gap is real and likely understated.

Keep the existing "would have been operating in the regime where it is no longer cleanly distinguishable…" sentence — it's accurate. Add the comparison-context implication after it.

### 6. `pr_thresholds.json` pointer

Card mentions "prefer per-tag thresholds" but doesn't tell users where they are. Add: each variant directory ships `pr_thresholds.json` with per-tag P=R thresholds for tags with support≥5. Pull the actual covered-tag count from `per_tag.pr_summary.tags_with_support` in each file at execution time — V1 covers 19,290/19,292 tags, V1.1 covers 19,290/19,292 (essentially every non-PAD/UNK tag has ≥5 positives in 296K samples).

### 7. License section

V1 README has `_TODO: fill in license_`. The V1.1 README has the full Apache-2.0 block. Make sure the draft uses the V1.1 version, not the TODO stub.

## Recommendations to surface to the user (don't auto-apply)

Present these as a numbered list of options at the end of your response. Don't edit the file for these without asking:

1. **Add a "what this card does that the others don't" callout** — single threshold, missing-positive disclosure, per-bucket mAP, full-vocab macro. Risks sounding boastful; let the user decide.
2. **Cite the WSML papers inline** in the early-stop section (Park 2023 RAL, Liu 2020 ELR, Kim 2022 LLM). Strengthens the noise-robust framing but adds academic weight the rest of the card doesn't carry.
3. **Add a "comparison caveats" subsection** — different val splits, different vocab sizes, different label-noise regimes. Honest, but may dilute the headline gap. Recommend including; the existing card's disclosure tone supports it.
4. **Compute exact param count** by loading `V1.1_safetensors/model.safetensors` and summing tensor sizes. Replaces "~247M" with the real number. Cheap; recommend doing it.
5. **Quote Camie's own admission** — "WD is considerably more accurate for common tags" (from his v1 card) — to support the vocab-asymmetry framing. Risks looking petty; user's call.

## Don't

- Don't add competitor screenshots or images.
- Don't write "OppaiOracle is the best anime tagger" — keep the "on this eval set, with these caveats" framing the existing card uses.
- Don't remove the existing "I think it performs well because the dataset is cleaner, not because of a special training regimen" paragraph. That's load-bearing honesty.
- Don't update the V1 / V1.1 READMEs in `huggingface_release/` — only the draft in `notes/`. The user is staging changes there before pushing.
- Don't create new files. Edit only [notes/huggingface_release_draft.md](notes/huggingface_release_draft.md).

## Output expectation

After editing, give the user a short diff summary: which sections you added, which you reworded, and which of the five recommendations above you held back for them to decide on. Under 200 words.
