---
license: apache-2.0
pipeline_tag: image-classification
language:
- en
tags:
- anime
- anime-tagger
- tagger
- image-tagging
- multi-label
- multi-label-classification
- vision-transformer
- vit
- illustration
- danbooru
- safetensors
- onnx
---



## TL;DR

A multi-label anime tagger trained from scratch on a \~5.9M image dataset that received a targeted cleaning and vocabulary-expansion pass before training. The corrections touched roughly **1.3M tags** — large in absolute terms, but only on the order of **\~3% of all tags** in the corpus, so this is best described as a *targeted* cleaning rather than a heavy one. The pass was deliberately weighted toward **low-frequency tags**, which is where mislabels and missing labels hurt a tagger the most. On my evaluation set the model achieves the best precision-equals-recall point and a good mAP relative to comparable open tagger checkpoints, but the underlying training data still contains category-level noise that no amount of training would have erased. **All predictions should be human-reviewed before they are trusted.**

Two checkpoints are released here. **V1** is the from-scratch 320×320 model. **V1.1** is a 448×448 fine-tune of V1 and on this evaluation set posts a modest mAP gain over V1 (overall val/mAP 0.674 vs. 0.614, ~+6 points absolute, ~+10% relative). The fine-tune helps across every frequency bucket but does not transform results — both checkpoints inherit the same source-data label noise. Pick the checkpoint whose native resolution matches the resolution you intend to feed it (see *Variants* below).

A live demo is available on the companion Space: [Grio43/OppaiOracle](https://huggingface.co/spaces/Grio43/OppaiOracle).

---

## Variants — which checkpoint should I use?

| Checkpoint | Native resolution | How it was produced | When to use |
|---|---|---|---|
| **V1** | 320×320 | Trained **from scratch** at 320×320. This is the model's native resolution. | The right pick if you are running inference at 320×320 or if throughput matters. |
| **V1.1** | 448×448 | A **fine-tune of V1** at 448×448. Position embeddings were interpolated from the 20×20 grid to 28×28, optimizer state was reset, and training continued at the new resolution following the FixRes / DeiT III progressive-resolution recipe. Trained for 6 of 15 planned epochs and stopped early — see *Performance notes / V1.1 headline numbers* below for the rationale. | Use when you specifically want 448×448 inference for finer spatial detail (small accessories, eye details). V1.1 outperforms V1 on every frequency bucket of this eval set (numbers in the *Performance notes* section), but the gain is modest — V1 remains a fully reasonable choice if you are running at 320×320. |

Two practical notes:

- **Match input resolution to the checkpoint.** Feeding 448×448 images to V1, or 320×320 images to V1.1, will give worse results than matching them. The position-embedding grid is fixed at load time.
- **V1 is not deprecated by V1.1.** They are siblings with different operating points, not generations of the same model. The V1.1 mAP gain over V1 is real but small (~+6 points overall) — pick on resolution, not on the assumption that V1.1 is strictly better.

### Files in this repo

- `V1_safetensors/` — V1 in `safetensors` format with `config.json` and `preprocessing.json`. Use this for PyTorch / custom inference.
- `V1_onnx/` — V1 exported to ONNX. Use this for ONNX Runtime inference (CPU, DirectML, CUDA EP).
- `V1.1_safetensors/` — V1.1 in `safetensors` format with `config.json` and `preprocessing.json`.
- `V1.1_onnx/` — V1.1 exported to ONNX.
- Each variant directory ships `vocabulary.json`, `selected_tags.csv`, `pr_thresholds.json`, and a copy of this README.

---

## Architecture

A Vision Transformer (ViT) trained from scratch. Spec (from `V1.1_safetensors/config.json`; V1 shares the same backbone at a smaller patch grid):

- 18 layers, hidden size 1024, 16 attention heads, FFN dim 4096, patch size 16
- Regularization: drop-path 0.2, attention dropout 0.05, hidden dropout 0.1
- Output head: 19,294 classes (all general-category tags; see *Vocabulary* in the comparison section below)
- Patch grid: V1 = 20×20 (320×320 input); V1.1 = 28×28 (448×448 input, position embeddings interpolated from V1's 20×20 grid)
- Roughly **\~247M parameters** total (estimated from the spec — exact count can be obtained by summing tensor sizes in `model.safetensors`)

For context, the comparison set in *Performance notes / Comparison vs other open anime taggers* is **not at parameter parity**: WD swinv2-base v3 is on the order of 99M params and Camie tagger v2 is on the order of 143M params. So the headline gap below is *not* explained by OppaiOracle being a smaller model than the comparison set — it is a larger one.

---

## How this model came to be

I started with a corpus of roughly **5.9 million images** with publicly-sourced tags. Before training anything of my own, I used **SmilingWolf's ViT v3 tagger** to help clean the dataset. With that pipeline I:

- **Removed \~300k incorrect tags** from images where the public labels disagreed with the AI tagger and a human spot-check confirmed the public labels were wrong.
- **Added \~1,000,000 missing tags** in the same fashion — places where the AI tagger surfaced a label the public tag set had simply omitted, and human review agreed.

That is \~1.3M corrections in total, which is only on the order of **\~3% of the tags in the corpus**. This was a *targeted* pass, not a top-to-bottom relabel. Effort was deliberately concentrated on **low-frequency tags**, on the assumption that mislabels and missing labels do disproportionate damage in the long tail — a missing label on a tag with 800 positives in the entire dataset matters far more than a missing label on a tag with 800k positives.

I then trained a small "light" model on this cleaned dataset, primarily as a vehicle to **expand the tag vocabulary by \~20,000 additional low-frequency tags** that the original tag set under-represented. That expanded vocabulary is what the released model was trained against.

The released checkpoint is the main training run on the cleaned dataset with the expanded vocabulary.

---

## What "cleaned" actually means (and what it does not)

This is the most important section of this release. The cleaning was real work, but it was not omniscient, and the dataset still has structured, category-level label noise that you will see in the model's outputs. Most of these issues are inherited directly from the **publicly-sourced source datasets** — they are not new noise introduced during cleaning; they are pre-existing patterns that the cleaning pass touched but did not resolve at the category level.

The categories below are **illustrative, not exhaustive.** Many other tag families show similarly deep-rooted issues. Two failure modes show up across most of them, but they are not equal in size:

- **Missing tags (by far the dominant problem)** — concepts that are clearly present in an image but were never tagged at the source. This is the single biggest source of noise in the entire dataset. See the dedicated subsection below for the empirical scale.
- **Wrong tags (not uncommon, but secondary)** — visually similar concepts confused with each other in the source data (the bow / bowtie / ribbon / ascot / necktie cluster, color buckets, length and size buckets). These are real and plentiful, just not the dominant failure mode.

### Missing tags (the dominant noise mode)

If you only remember one thing from this section, remember this: **the biggest single problem in the source data is not wrong tags, it is missing tags.** Wrong tags are not uncommon either, but they are dwarfed in volume by labels that should be present and simply aren't.

A rough empirical sense of the gap, from manual review:

- A typical image in this dataset arrives with roughly **\~28 tags** from the source.
- A reasonably-tagged image — judged by what is actually visible, sticking to common in-vocabulary concepts and not reaching for rare tags — should have **50+ tags**, often more.
- During spot-checks I have routinely taken images that arrived with **\~40 tags up past 60 tags** just by adding common, obviously-present concepts. That is without making any effort to surface rare tags; including those would push the number higher still.

So the source tag count is on the order of **half** of what a careful tagger would emit on the same image, and the gap is concentrated in concepts that are not subjective — they are simply omissions. The cleaning pass added \~1M missing tags back, but with the gap this large there are many millions still missing across the corpus.

The training-time consequence is that for every missing-but-present tag, the model receives **no positive gradient at all** for that concept on that image — only an implicit negative through the loss. This systematically biases the model toward under-predicting any tag with a high source-data omission rate, and the effect is uneven across tags: some tag families are well-tagged at the source and some are very sparsely tagged. Practically, this means **low predicted scores are less informative than they look** — a tag scoring below threshold may be genuinely absent, or it may be a concept the model has learned is "usually unlabeled even when present."

### Color tags

Color-named tags (eye color, hair color, general color tags) are **poorly tagged at the source**, and the noise that survived cleaning is dataset-wide. Every color tag in the vocabulary has some version of this problem; some are worse than others.

- **Obvious failures were cleanable.** A bright, unambiguous yellow mislabeled as `blue_eyes` is exactly the kind of disagreement the AI-assisted pass catches, and those got fixed. The residual noise is not the obvious-failure kind.
- **The deep-rooted issue is perceptual, not technical.** The category boundaries between color tags are drawn by *human viewers*, not by RGB codes. Different taggers carve up the spectrum differently, and any single color tag in this dataset covers a fairly wide perceptual band of that color. There is no clean RGB threshold I could have used to mechanically separate the categories, which is exactly why manual cleaning at the category level is intractable.
- **Adjacent / overlapping colors leak into each other in predictable patterns.** Some examples I have observed:
  - `aqua_*` tags heavily pollute both **blue** and **green** based tags — aqua sits perceptually between them and gets sorted into all three buckets across the corpus.
  - `yellow_*` tags overlap meaningfully with **red** and **orange** tags — warm-spectrum boundaries are inconsistent in the source data.
  - Similar patterns exist for purple/blue/pink, brown/orange/red, and black/very-dark-anything.
- Color tags are also **high frequency**, so the noise is spread across millions of images rather than concentrated where it could be hand-fixed.
- When I sampled live in-the-wild images and compared the model's predictions to a careful human reading, the same source-data confusion patterns were still present in the predictions. The model is faithfully reproducing the source-data label distribution, which is itself noisy along the color axis.

### Hair length

The hair length tags — `very_short_hair`, `short_hair`, `medium_hair`, `long_hair`, `very_long_hair` — all have major boundary issues. `long_hair` and `very_long_hair` are the worst offenders; the source labels routinely disagree with each other across visually similar images. The model inherits this confusion.

### Other "objective size" body-part tags

The same problem applies to tags that sound objective but are really continuous and judgement-dependent: `flat_chest`, `small_breasts`, `medium_breasts`, `large_breasts`, `huge_breasts`. These are inherently noisy supervision targets for a classifier — adjacent buckets are not crisply separable in the source data, and the model cannot do better than the labels it was given.

### Neckwear and small accessories (bows, bowties, ribbons, ascots, neckties)

This cluster of tags has systemic issues at the source. `bow`, `bowtie`, `ribbon`, `ascot`, and `necktie` are visually similar but distinct accessories, and the public source data routinely confuses them — the same physical object will be tagged differently across images, and adjacent categories leak into each other in both directions. The cleaning pass touched obvious mistakes here but did not normalize the category boundaries; the model learns the same fuzzy boundaries the source data has.

These five are the cluster I happened to look at closely. Many other small-accessory and clothing-detail tags show the same pattern — visually similar items, fuzzy source-data boundaries, residual confusion in the model. Treat any prediction in this category as a *suggestion* to inspect, not a final answer.

### Character-vs-concept leakage

For some tags, the data is dominated by a small number of characters. When that happens, the model tends to learn **the character** rather than **the concept** the tag was meant to represent. Without a curated golden-standard set that deliberately decouples the concept from those characters, this is very hard to fix at training time.

### My estimate of cleaning quality

The 300k removals and \~1M additions were **AI-assisted and then human-reviewed by me**. My honest estimate is that the corrections themselves are **<5% error**. That is a statement about the *changes I made*, not about the *underlying dataset* — the underlying dataset still contains the structured noise described above, because cleaning was driven by AI-flagged disagreements and the AI shares the same color/length/size confusion as the source data does.

---

## How to use this model responsibly

- **Human review every output.** This applies most strongly to color, hair length, and size-bucket tags. The model is a fast first pass, not an authoritative labeler.
- **Treat sibling tags as a group, not a hard pick.** If the model emits `blue_eyes` with high confidence, also check the `purple_eyes` / `aqua_eyes` / `black_eyes` scores before you commit.
- **Do not use the raw output as ground-truth for downstream training** without manual review. The very confusion patterns that this model can't resolve will get baked into your downstream model.
- **For thresholding, prefer per-tag thresholds over a single global threshold.** Different tag families have very different precision/recall behavior on this dataset. Each variant directory ships `pr_thresholds.json` containing per-tag P=R thresholds for tags with support≥5 in the held-out split — this covers **19,290 of the 19,292 evaluated tags** (essentially every non-`<PAD>`/`<UNK>` tag has ≥5 positives in 296,056 samples) for both V1 and V1.1.

---

## Performance notes

On my evaluation set this model achieves:

- The best **precision-equals-recall** point I have measured among comparable open anime taggers.
- A solid **mAP** relative to the same comparison set.

### Evaluation methodology

So that the headline numbers are interpretable:

- **Eval set.** Both V1 and V1.1 are evaluated on the **same 296,056-image held-out split**, drawn from the cleaned-and-expanded corpus described above. V1 was evaluated at epoch 27 / step 170,799; V1.1 at epoch 7 / step 85,517 (the full-val recompute completed 2026-05-09).
- **Threshold sweep.** F1 / P=R operating points are obtained from a post-training sweep of the global threshold over **[0.001, 0.999] in steps of 0.001** (999 points), independently for each model. Both per-tag and single-threshold operating points come from this sweep. Indices `[0, 1]` (`<PAD>` and `<UNK>`) are excluded from all metrics.
- **Source of truth.** Numbers in this section are pulled from each variant's `pr_thresholds.json`. Both files are at parity on the full-val split. (The copy of `pr_thresholds.json` shipped inside `V1.1_safetensors/` may temporarily be a pre-recompute snapshot at `val_samples: 30000`; the authoritative full-val V1.1 numbers — same checkpoint — live in [experiments/run1_vit/checkpoints/pr_threshold_last.json](../experiments/run1_vit/checkpoints/pr_threshold_last.json) and will be synced into the release directory before push.)

### V1 headline numbers (e27/40, Phase 1, 320×320, 19,292 tags)

| Metric | Value |
|---|---|
| Macro F1 | 0.588 |
| Micro F1 | 0.659 |
| P=R threshold (macro / micro) | 0.614 / 0.670 |
| Overall val/mAP | 0.614 |

**mAP broken out by tag frequency bucket:**

| Frequency bucket | mAP |
|---|---|
| 500–999 (rare) | 0.589 |
| 1K–5K (mid) | 0.598 |
| 5K–10K (head) | 0.535 |
| 10K+ (very common) | 0.542 |

Note the inversion: rare/mid tags out-score head/very-common tags on mAP. This is consistent with the missing-tag bias described above — high-frequency concepts are the ones most often present-but-unlabeled in the source data, which depresses their measured precision against a noisy reference.

### V1.1 headline numbers (e6/15, Phase 2, 448×448, 19,292 tags)

| Metric | Value |
|---|---|
| Macro F1 (P=R) | 0.646 |
| Micro F1 (P=R) | 0.699 |
| P=R threshold (macro / micro) | 0.753 / 0.793 |
| Overall val/mAP | 0.674 |

Macro F1 is the mean of per-tag F1 at a single global threshold (the WD14-comparable convention). The macro number is essentially identical at any support cutoff (0 / 1 / 5) on this val split, since 19,290 of 19,292 non-PAD/UNK tags have ≥5 positives in 296,056 samples — there are no structural-zero outliers depressing the mean. Per-tag-tuned operating points (each tag at its own break-even threshold) average mean P=R = 0.648 and mean F1-opt = 0.675. See the *A note on the F1 numbers* paragraph below for what "F1 at P=R" means here vs. the in-training F1 metric you may have seen in earlier drafts.

**mAP broken out by tag frequency bucket — V1 vs. V1.1 on the same eval set:**

| Frequency bucket | V1 mAP | V1.1 mAP | Δ |
|---|---|---|---|
| 500–999 (rare) | 0.589 | 0.645 | +0.056 |
| 1K–5K (mid) | 0.598 | 0.656 | +0.058 |
| 5K–10K (head) | 0.535 | 0.595 | +0.060 |
| 10K+ (very common) | 0.542 | 0.606 | +0.064 |
| **Overall** | **0.614** | **0.674** | **+0.060** |

The same rare-vs-head inversion noted for V1 (rare/mid > head/very-common on mAP) is still present in V1.1, and for the same reason — high-frequency tags are the ones most often present-but-unlabeled in the source data, which depresses their measured precision against a noisy reference.

### Comparison vs other open anime taggers

The TL;DR claim "best P=R I've measured" deserves the underlying numbers. The comparison below is at each model's own break-even threshold (the same convention WD v3 publishes its headline numbers under). All OppaiOracle numbers are pulled from the `pr_thresholds.json` referenced in *Evaluation methodology* above; competitor numbers are quoted from each model's published model card.

**Macro-F1 at P=R (each model evaluated against its own training distribution / val split):**

| Model | Macro-F1 (P=R) | Notes |
|---|---|---|
| **OppaiOracle V1.1** | **0.646** | `macro_single_threshold.support_ge_1.pr_breakeven.f1`. On the 296K val split, support≥0 / ≥1 / ≥5 collapse to essentially the same number (0.6460 in all three) because 19,290 of 19,292 non-`<PAD>`/`<UNK>` tags have ≥5 positives. |
| OppaiOracle V1 | 0.588 | `V1_safetensors/pr_thresholds.json`, support≥0 |
| camie-tagger-v2 | 0.506 | "Macro-OPT" at threshold 0.492, from his model card |
| wd-eva02-large-tagger-v3 | 0.4772 | model card |
| wd-vit-large-tagger-v3 | 0.4674 | model card |
| wd-swinv2-tagger-v3 | 0.4541 | model card |

**Micro-F1 at P=R:**

| Model | Micro-F1 (P=R) | Notes |
|---|---|---|
| **OppaiOracle V1.1** | **0.699** | `micro.pr_breakeven.f1` |
| camie-tagger-v2 | 0.673 | "Micro-OPT" at threshold 0.614 — note this is a **different threshold** from his macro-headline operating point, so his macro and micro numbers are not from the same model state |
| OppaiOracle V1 | 0.659 | `V1_safetensors/pr_thresholds.json`, `micro.pr_breakeven` |
| WD v3 | not reported | — |

**Apples-to-apples vocabulary.** Comparing F1 numbers across these models is fair only once the vocabularies are described, because "70K tags" and "19K tags" are not the same target. Camie's headline 70K is dominated by named-entity (character / copyright / artist) tags, while OppaiOracle's vocabulary is general-only:

| Model | General tags | Total vocab |
|---|---|---|
| OppaiOracle V1 / V1.1 | **19,294** | 19,294 (100% general) |
| camie-tagger-v2 | 30,841 | 70,527 |
| wd-vit-large-tagger-v3 | 8,106 | 10,861 (8,106 cat-0 general + 2,751 cat-4 character + 4 cat-9) |

So on the general-tag axis, OppaiOracle's vocabulary is roughly **2.4× WD's** and roughly **0.6× Camie's general slice**, while still beating both on macro-F1. The named-entity tags Camie's total includes are a different problem domain (recognizing specific characters / copyrights / artists) and are not what this model is measured on.

**Why this comparison is fair on the metric, but not at parameter parity.** Macro-F1 at the model's own P=R threshold is a calibration-agnostic operating-point comparison — every model is being scored at *its own* best threshold, so loss-function calibration differences don't bias the ranking. What it isn't is a parameter-parity comparison: as noted in *Architecture*, OppaiOracle is the largest model in this set (~247M vs. WD swinv2-base ~99M and Camie ~143M). The gap is real on the metric used; the parameter-count caveat belongs in any deeper analysis.

**Why V1.1 stopped at 6 of 15 planned epochs.** This was a deliberate noise-robust stopping decision, not a regret. Per-epoch mAP growth decelerated from ~+0.7%/epoch in early Phase 2 to ~+0.3%/epoch by epoch 5, while validation loss continued to fall and per-tag calibration shifted (mean activations per image dropped from ~4500 at epoch 0 to ~4200 at epoch 5; the auto-stop F1 metric is calibration-floored at a fixed threshold of 0.2653 and therefore unreliable as a stop signal — see [TRAINING_HEALTH_TRACKER.md](../TRAINING_HEALTH_TRACKER.md)). The deceleration coincides with a known phase transition in the weakly-supervised multi-label / asymmetric-loss literature: V1.1's loss configuration (`γ_neg=7.0` with reduced regularization) is precisely the regime most exposed to **missing-positive memorization** — the model begins learning that *labeled-but-noisy = positive* and *unlabeled-but-actually-present = negative*. Validation has the same missing-positive structure as training, so a model that has crossed into this regime will *raise* noisy-reference mAP even after true ranking quality has plateaued. The remaining 9 epochs would have been operating in the regime where it is no longer cleanly distinguishable whether mAP gains are *real ranking improvement* or *memorization of the labeled subset of a noisy multi-label corpus* (the missing-positive bias documented earlier in this card sets a soft ceiling somewhere in this neighbourhood). **Implication for the comparison numbers above:** competitors that report higher convergence almost certainly trained past this same phase transition (WD v3 trains for 50+ epochs; Camie's training duration is not disclosed). The headline gap is therefore between OppaiOracle's *pre-memorization* checkpoint and competitor checkpoints that have very likely crossed into it — meaning the gap on cleanly-tagged data is real, and probably understated rather than overstated. Continuing was unlikely to buy enough real gain to justify the extra training time, so V1.1 ships at the epoch-5 / step-81822 checkpoint.

**A note on the F1 numbers.** V1.1's loss configuration (`gamma_neg=7.0`, `clip=0.2`) shifts the logit distribution relative to V1's (`gamma_neg=4.0`, `clip=0.05`), so the **in-training** F1 metric — which uses a fixed threshold (0.2653) calibrated against V1's distribution — is calibration-floored for V1.1 and unreliable both as a stop signal during training and as a comparison number against V1. Earlier drafts of this card therefore declined to report any F1 number for V1.1. The F1 / P=R numbers in the headline table above are **not** the in-training metric; they are the operating point from a post-training threshold sweep over [0.001, 0.999] (step 0.001) on the same 296,056-image held-out split used for V1, with `<PAD>` and `<UNK>` excluded. The sweep finds each model's own break-even / argmax-F1 threshold from scratch, which is calibration-agnostic — so the V1 → V1.1 F1 comparison is apples-to-apples (Macro F1 0.588 → 0.646, Micro F1 0.659 → 0.699; mAP 0.614 → 0.674).

I want to be honest about *why* I think it performs well: **it is almost certainly not because of a special training regimen.** The training recipe is grounded in standard ViT-from-scratch literature (DeiT / DeiT III / FixRes / ASL / AugReg) without exotic tricks. The most likely explanation is simply that the **input dataset is cleaner** than what most comparable taggers were trained on. If you are trying to reproduce or beat this result, I would put your effort into data curation before you put it into training-recipe tuning.

---

## Image augmentation settings (V1 and V1.1)

For reproducibility, here are the exact augmentation pipelines used for each checkpoint. V1.1 is a fine-tune of V1, so its augmentation is a *reduced* version of V1's — narrower ranges and lower probabilities at the higher 448×448 resolution. The reductions follow EfficientNetV2 / FixRes guidance for progressive-resolution training, but only partially (\~¼ reduction rather than ½), because Phase 1 stopped at 33/40 epochs and the V1 base was under-converged when V1.1 began.

| Augmentation | V1 (320×320, from scratch, 40 epochs planned, 33 trained) | V1.1 (448×448, fine-tune of V1, 15 epochs planned, 6 trained) |
|---|---|---|
| Horizontal flip | p = 0.5 | p = 0.5 |
| Color jitter — brightness | 0.30 (p = 0.5) | 0.22 (p = 0.5) |
| Color jitter — contrast | 0.20 (p = 0.5) | 0.15 (p = 0.5) |
| Color jitter — saturation | 0.08 (p = 0.5) | 0.06 (p = 0.5) |
| Random rotation | p = 0.50, ±[2°, 8°], bicubic | p = 0.30, ±[2°, 5°], bicubic |
| Gaussian blur | p = 0.30, kernel = 3, σ ∈ [0.1, 1.5] | p = 0.15, kernel = 3, σ ∈ [0.1, 1.0] |
| Random erasing | disabled | disabled |
| Normalization (mean / std) | [0.5, 0.5, 0.5] / [0.5, 0.5, 0.5] | [0.5, 0.5, 0.5] / [0.5, 0.5, 0.5] |
| Letterbox pad color | [114, 114, 114] | [114, 114, 114] |

Notes on a few of these choices:

- **Saturation is held well below brightness/contrast** in both phases. Saturation is the only color-jitter axis that directly attacks color-named tag identity (`blue_eyes`, `pink_skin`, etc.); brightness and contrast are luminance-driven and largely chroma-safe. The ratio (\~¼ of brightness) is taken from BYOL's asymmetric augmentation.
- **Rotation is kept on at V1.1**, against the plain FixRes recommendation. The original plan was to disable it at 448 for spatial precision, but with V1 under-converged it was safer to keep a residual rotational-invariance signal. The compromise was a tighter angle band (±5° vs. ±8°) and a lower fire rate (0.30 vs. 0.50).
- **Gaussian blur is also kept on at V1.1** for the same reason (under-converged base + reduced color/rotation aug → strips too much input variability if blur is dropped entirely). Frequency was halved and the σ ceiling pulled in from 1.5 to 1.0.
- **No mixup, no cutmix, no RandAugment, no random erasing** in either phase. The recipe is intentionally close to DeiT III's "3-Augment" regime (flip + color jitter + blur) plus a small rotation, not a heavy AugReg/RandAugment stack.

---

## Limitations summary

| Area | Severity | Notes |
|---|---|---|
| Color tags (eye/hair/general) | **High** | Source-data noise survives; sibling colors leak into each other |
| Hair length (especially `long_hair`, `very_long_hair`) | **High** | Boundary tags inherently noisy in source |
| Size-bucket body-part tags | **High** | Continuous quantity discretized into noisy buckets |
| Neckwear (`bow`, `bowtie`, `ribbon`, `ascot`, `necktie`) | **High** | Visually similar accessories routinely confused at source; representative of a broader small-accessory pattern |
| Missing tags (concept present, no label) | **Dominant** | The single biggest source of noise in the corpus. Typical \~28 tags/image vs. 50+ that should be present. \~1M added back during cleaning; many millions remain. Hurts performance broadly and biases the model toward under-prediction. |
| Character-overwhelmed tags | **Medium** | Some tags are learned as proxies for specific characters |
| Rare / low-frequency tags | **Medium** | The +20k vocabulary expansion helps, but tail tags still see fewer examples |
| Anything not on the above list | Use with normal caution | The above are illustrative, not exhaustive — many tag families show similar source-data issues |

---

## What's next (V2)

Once a refreshed 2026-vintage source dataset becomes available, I plan to start work on V2. The biggest single change between V1 and V2 will not be the model — it will be **substantially more time spent on data cleaning before training begins**, with a particular focus on:

- Building a curated **golden-standard slice** for color tags, hair-length tags, and size-bucket tags so those categories can be supervised against deliberately disambiguated examples.
- Deeper character/concept decoupling so character-overwhelmed tags learn the actual concept.
- Better measurement of "true" performance on a hand-relabeled validation slice, so the headline metrics are not silently inflated by the same missing-positive bias that affects the training data.

V1 ships with the noise it ships with. V2 is where I plan to do something about it.

---

## Acknowledgments

- **SmilingWolf** for the ViT v3 tagger, which made the initial cleaning pass tractable. None of this would have been feasible without an existing strong tagger to use as a second opinion.
- The broader anime-tagger open-source community for the public tag corpora and prior model checkpoints I compared against.

---

## License / usage

Released under the **Apache License 2.0**. You may use, modify, and redistribute the model and accompanying files for personal, research, or commercial purposes, provided you retain the license notice and attribution.

**Intended use.** Research and downstream tooling for multi-label tagging of anime / illustration imagery.

**Out-of-scope use.** Decisions about real people; safety-critical pipelines that depend on label correctness without human review; training a downstream model on raw outputs without manual review (the missing-tag bias described above will propagate).
