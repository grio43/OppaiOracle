# Overfitting Risk Assessment: OppaiOracle

**Date:** 2026-04-01
**Status:** Assessment complete, pending implementation decisions
**Overall Risk Level:** MODERATE-HIGH

## Context

Deep review of overfitting risk for a 244M-parameter ViT trained **from scratch** on ~5.4M anime images with ~28K tags. Image augmentation is intentionally limited due to anime art style constraints -- **mixup/cutmix and random erasing are deliberately disabled** because they would obscure fine-grained features like eye color, hair accessories, and other detail-level tags critical for anime tagging accuracy.

The dataset is large (5.4M images), the regularization stack is strong (dropout, drop path, weight decay, LR-aware early stopping), and the augmentation constraints are well-reasoned for the domain. However, training a 244M-param ViT from scratch without pretraining is the primary structural risk, compounded by validation set limitations that make overfitting harder to detect.

---

## Findings

### Risk Factor 1: SEVERE -- Training 244M-param ViT from Scratch Without Pretraining

**Verified TRUE** -- All weights initialized via truncated normal at `model_architecture.py:509`. No pretrained backbone loading exists for the ViT path.

| Training Setup | Params | Images | Images/Param | Augmentation |
|---|---|---|---|---|
| **OppaiOracle** | **244M** | **5.4M** | **22** | **Flip + mild jitter** |
| ViT-B on ImageNet-21K | 86M | 14M | 163 | Moderate |
| DeiT-B on ImageNet-1K | 86M | 1.28M | 15 | Heavy (mixup, cutmix, erasing) |

22 images/param is in the DeiT range, but DeiT compensates with heavy augmentation that isn't viable here. The patch embedding (Conv2d 3->1024, 16x16, ~786K params) must learn pixel-to-feature mappings from scratch. The original ViT paper showed ViTs trained from scratch underperform CNNs on datasets under ~100M images.

### Risk Factor 2: MODERATE-HIGH -- Augmentation Constraints are Valid but Create a Gap

Mixup, cutmix, and random erasing are **intentionally disabled** to preserve fine-grained visual features (eye color, small accessories, etc.). This is a sound domain decision.

Current augmentation (`configs/unified_config.yaml:186-219`):
- Horizontal flip: 50% with orientation-aware tag swapping (well implemented)
- Color jitter: brightness +/-6%, contrast +/-4%, saturation +/-4% (very conservative)
- Random erasing: disabled (would obscure fine features)
- Random rotation: disabled but implemented (0.5-3 degrees)

#### Color Jitter Deep Dive

Color jitter was intentionally de-escalated from aggressive values (commit 413c4ce: *"Conservative defaults tuned for anime colour fidelity"*). The current +/-4-6% is nearly imperceptible and provides minimal regularization. ~2,365 tags (2.76% of vocab) are color-dependent, but they're among the most important (hair color, eye color, clothing). The `IndependentColorJitter` class applies each parameter separately with 50% probability, preventing compound over-augmentation.

Per-parameter analysis:

| Parameter | Current | Natural Dataset Variation | Recommended | Rationale |
|---|---|---|---|---|
| **Brightness** | +/-6% | +/-30-50% across artist styles | **+/-12-15%** | Doesn't affect hue/color identity at all. Dark moody vs bright cheerful scenes vary far more than this. |
| **Contrast** | +/-4% | Wide (cel shading vs watercolor) | **+/-8-10%** | Adjusts dynamic range around mean, doesn't change color identity. Different art techniques naturally vary more. |
| **Saturation** | +/-4% | Moderate | **+/-6-8%** | Most sensitive -- desaturation can make `red_hair` look brownish. But +/-8% is within perceptual boundary. Fine-grained pairs like `light_blue_hair` vs `dark_blue_hair` need care. |
| **Hue** | Disabled | N/A | **Stay disabled** | CRITICAL -- hue rotation is categorical for anime. Even 20 degrees can turn `blue_eyes` into `green_eyes`. Correctly removed in commit history. |

Additional augmentation notes:
- **Small-angle rotation (0.5-3 degrees)** would not affect tag accuracy and is already implemented -- just toggled off.
- The gap left by no mixup/cutmix makes dropout (10%), drop path (20%), and weight decay (0.1) more load-bearing than usual.

### Risk Factor 3: VERIFIED HIGH -- No Held-Out Test Set

**Verified TRUE** -- Only train/val split exists. No test loader or separate evaluation anywhere.

The validation set simultaneously serves:
1. Early stopping criterion (patience=4 on `val_f1_macro`)
2. Per-tag threshold calibration (`train_direct.py:2348-2352`)
3. Performance reporting

Reported metrics are optimistic. A 30-50K held-out test set (from the 240K excess val samples currently moved to training) would provide unbiased generalization estimates.

### Risk Factor 4: VERIFIED MODERATE-HIGH -- Validation Cap at 30K Hides Rare-Tag Overfitting

**Verified TRUE** -- `max_val_samples: 30000` at `configs/unified_config.yaml:153`. Frequency-bucketed F1 also runs on the capped 30K, not the full set.

For rare tags near the 300-occurrence threshold:
- 300 / 5.4M = 0.006% positive rate
- In 30K val (5% split): **~0.84 expected positives** -- cannot measure anything

Increasing to 100K costs ~3x validation time (negligible vs training) and gives ~2.8 expected positives for the rarest tags.

### Risk Factor 5: VERIFIED MODERATE -- No Class Weighting for Rare Tags

**Verified TRUE** -- `class_weight_strategy` commented out at `configs/unified_config.yaml:311-313`. Full implementation exists at `train_direct.py:195-248` but never triggered.

Focal loss (gamma_neg=4.0) handles global imbalance but doesn't scale inversely with tag frequency. Tags at 300 occurrences get the same focal treatment as tags at 3M. Inverse-sqrt weighting would provide ~5-10x gradient amplification for the rarest tags.

### Risk Factor 6: MODERATE -- 36 Epochs with Limited Augmentation Diversity

Each image seen ~36 times with ~2x effective diversity from flips = each unique view seen ~18 times. Early stopping should catch plateau, but only if validation is reliable (see #4).

---

## Investigated and Cleared: CLS Token is NOT a Practical Bottleneck

The initial concern was that predicting 28K tags from a single 1024-dim CLS vector creates an information bottleneck. **Three independent investigations concluded this is not a practical risk:**

1. **Full dense attention from layer 0** -- CLS attends to all 784 patches in every layer. `flex_attention` with `block_size=128` is a compute optimization, not a sparsity pattern. `token_ignore_threshold=0.9` only masks padding.

2. **17 layers of iterative refinement with 16 attention heads** -- CLS doesn't statically compress. It learns a dynamic query that selectively attends via 16 independent 64-dim subspaces. This is ~481M parameters of information transformation.

3. **Standard production practice** -- CLIP ViT-L, DINOv2, MAE all use single CLS readout for classification without bottleneck issues.

**Note:** Config declares `use_style_token`, `use_line_token`, `use_color_token` with `num_special_tokens: 4`, and `num_groups: 20`, `tags_per_group: 10000`, but these are all **ghost config** -- explicitly filtered out as unused keys at `train_direct.py:771-776`. The validation loop has 3D output handling code suggesting grouped prediction was planned but never shipped. These are vestigial, not missing features.

---

## What's Working Well (Existing Protections)

| Mechanism | Value | Assessment |
|---|---|---|
| Dropout | 10% hidden, 5% attention | Standard, adequate |
| Drop path | 20% linearly increasing | Strong, well-configured |
| Weight decay | 0.1 (excluding norms/biases) | Aggressive, good |
| Gradient clipping | max_norm=1.0 | Standard protection |
| Focal loss | gamma_neg=4.0, alpha=0.75 | Well-tuned for 0.6% positive rate |
| Early stopping | patience=4, burn-in=3, LR-aware | Sophisticated |
| Frequency-bucketed eval | 6 frequency buckets | Good diagnostic (limited by 30K cap) |
| Tag head bias init | Log-prior from frequencies | RetinaNet technique, helps gradient flow |
| Orientation-aware flips | Tag swapping on flip | Domain-appropriate |

---

## Action Items

### Tier 1: High Impact, Low Effort

- [ ] **A. Increase val samples to 100K+**
  - File: `configs/unified_config.yaml:153`
  - Change `max_val_samples: 30000` to `max_val_samples: 100000`
  - Effort: Trivial

- [ ] **B. Create held-out test set** (30-50K from val pool)
  - File: `dataset_loader.py:3151-3178`
  - Reserve samples that are never used for early stopping or threshold calibration
  - Effort: Small

- [ ] **C. Enable class weighting** -- uncomment inverse_sqrt
  - File: `configs/unified_config.yaml:311-313`
  - Uncomment `class_weight_strategy: "inverse_sqrt"` and clip bounds
  - Effort: Trivial

- [ ] **D. Strengthen color jitter** (hue stays disabled)
  - File: `configs/unified_config.yaml:200-205`
  - brightness: 0.06 -> 0.12
  - contrast: 0.04 -> 0.08
  - saturation: 0.04 -> 0.06
  - Effort: Trivial

- [ ] **E. Enable random rotation** (0.5-3 degrees, already implemented)
  - File: `configs/unified_config.yaml:216`
  - Set `random_rotation_enabled: true`
  - Effort: Trivial

### Tier 2: High Impact, Medium Effort

- [ ] **F. Use pretrained ViT backbone** if compatible checkpoint available
  - File: `model_architecture.py`
  - Would need to find/adapt a pretrained ViT-L/16 with 1024-dim, 17-layer architecture
  - Effort: Medium

### Tier 3: Good Practice

- [ ] **G. Reduce max epochs to 25, warmup to 5-7**
  - File: `configs/unified_config.yaml:224,272`
  - Effort: Trivial

- [ ] **H. Clean up ghost config** (special tokens, num_groups, tags_per_group)
  - File: `configs/unified_config.yaml:57-61,76-77`
  - Remove vestigial config that is explicitly filtered out and never used
  - Effort: Trivial

---

## Overfitting Monitoring Checklist

### Every Epoch
- [ ] **Train loss vs val loss gap** -- widening gap after epoch 10+ = overfitting
- [ ] **Val F1 macro trajectory** -- <0.05% improvement for 2+ epochs while train loss drops = overfitting
- [ ] **Frequency-bucketed F1 divergence** -- high-freq improving while low-freq degrades = rare-tag overfitting

### Every 2-3 Epochs
- [ ] **Per-tag threshold drift** -- >0.05 absolute change between epochs = unstable calibration

### Red Flags (Consider Stopping)
- Val loss increases 2+ consecutive epochs while train loss decreases
- Rare-tag F1 (bucket <1000) drops below its epoch-15 value
- Training recall on any frequency bucket approaches 100%

---

## Key Files Reference

| File | Relevance |
|---|---|
| `configs/unified_config.yaml` | Augmentation, val samples, class weights, epochs |
| `train_direct.py` | Training loop, class weight computation, ghost config filtering (L771-776) |
| `dataset_loader.py` | Augmentation pipeline, split logic, IndependentColorJitter (L157-224) |
| `model_architecture.py` | ViT architecture, tag head, bias init (L167-197) |
| `loss_functions.py` | AsymmetricFocalLoss, class weight integration |
| `validation_loop.py` | Has 3D output handling for planned-but-unshipped grouped prediction |
