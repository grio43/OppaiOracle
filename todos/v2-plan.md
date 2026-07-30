# V2 Plan — the authoritative document

> **This is THE V2 plan.** It holds every decision and the runnable per-phase config. Where a
> decision needs more than a paragraph of justification, it cites into the detail docs rather than
> repeating them.
>
> **Doc set, and what each one is for now:**
>
> | Doc | Role |
> |---|---|
> | **`v2-plan.md`** (this file) | **Decisions + config. The only doc that governs the run.** Self-contained as of 2026-07-29 — no other doc is needed to execute it. |
> | [`v2-plan-review-2026-07-28.md`](v2-plan-review-2026-07-28.md) | Evidence base. Every number here is measured or cited there. Point-in-time snapshot: its code line refs are stale, and where it and this file disagree, this file wins. |
> | `progressive-training-plan.md` · `ASL_plan.md` · `v2-plan-correction-2026-07-28.md` | **Retired 2026-07-29** (full text in git history). Decisions superseded by this file; still-live content was inlined first: per-phase augmentation → §7, telemetry spec → Appendix A, scope fence → §4. |
>
> **Rule for future edits:** amend *this* file. Do not add another dated correction layer — the
> supersession chain is what let stale summaries revert settled decisions twice already.
>
> **Created 2026-07-28**, consolidating the four-layer plan set after the independent review.

---

## 1. What changed, and why the plan needed rewriting

Two load-bearing beliefs turned out to be unmeasured:

**The label-noise ceiling was never observed.** V1 did not early-stop — `grep -c "Early stopping
triggered" logs/training.log` returns **0**. Both phases ended with a manual `Soft stop engaged`
mid-epoch. Phase 1 stopped at **33/40** epochs with LR at 14% of peak and mAP climbing
+0.0065/epoch; Phase 2 at **6/15** with LR at 76% of peak. mAP and val loss improved *monotonically
at every logged validation*. And there is **zero generalization gap**: micro-F1 **0.70044** on 296K
images (~90% of which were folded into training) vs **0.70088** on the 30K genuinely held out. A
248M ViT memorized nothing in 39 epochs.

Three V2 decisions rested on that ceiling — the 248M→192M cut, the loss-first workstream, and
de-scoping label cleaning. All three are re-opened below.

**P-ASL does not transfer to positive-only annotation.** ~49% of its published gain comes from
γ⁻/γᵘ decoupling, which requires the **37.7M human-verified negatives** OpenImages has and booru has
none of; its Ignore-mode prior estimator is degenerate at N=∅; and its Ω_P ignore set would select
the 107 tags in our vocabulary with the *lowest* missingness and strip their only negative gradient.
Details: [review §3](v2-plan-review-2026-07-28.md).

**What survives unchanged:** the γ_neg descent stays withdrawn. γ_neg = 7.0 fixed is still correct —
via a different citation. The wrong-positive scope fence stays. The instrument is still the launch
blocker, but for different reasons.

---

## 2. The plan at a glance

| | V2 |
|---|---|
| **Backbone** | 896w × 18L, patch16, head_dim 64 (14 heads), CLS-pool — **pending the §3 sizing decision** |
| **Params** | ~192M (or ~150M at SoViT's true MLP ratio, or ~248M unchanged — §3) |
| **Init** | **OPEN DECISION — §5.** From-scratch is the current default and is *unargued*. |
| **Data** | ≥6.8M images, 19,294 tags, positive rate ~0.18%, neg:pos ≈ 530–640:1 |
| **Resolution** | 320 → 448 → **512 (now unconditional, not gated)** |
| **Loss** | **ASL**, γ_neg = 7.0 fixed, γ_pos = 0, clip = **0.05**, α = 1.0, label_smoothing = 0 — per **ASL Appendix F**, not P-ASL |
| **Budget** | **~400–700M samples seen** (V1 got ~230M) |
| **New this run** | Weight EMA · LayerScale · plain dropout → 0 · fp32 optimizer state on the head · longer warmup |
| **Selection** | `val/mAP` over **all** tags, frozen list, on a **≥276K** val set; binned at `ap_thresholds: 200` (bias −0.0031, near-uniform); resolution floor **3e-3** |
| **Decision procedure** | Three-gate protocol, §9. Gate 3 (bias-controlled slice) is the actual arbiter. |

---

## 3. Architecture — the cut is re-opened

The 248M→192M cut was justified by *"the ~250M model carried unusable headroom."* That headroom was
never tested (§1). Separately, 192M on 6.8M images is **conservative**, not aggressive: DeiT III
trains a **304M** ViT-L from scratch on **1.28M** ImageNet images to 84.9%.

**Three defensible options. Pick one; the run cannot start without it.**

| Option | Shape | Params | Case for it |
|---|---|---|---|
| **A — keep 1024×18** | current config, no change | 248M | The cut's premise is void. Zero config risk, zero re-derivation, and V1's weights become a warm-start option. |
| **B — 896×18, MLP 3584** | plan as written | 192M | Throughput. Matches SoViT-150m's width/depth to within 2%. |
| **C — 896×18, MLP 2320** | SoViT-150m's *actual* shape | **~150M** | If you want a smaller model, this is the principled version: SoViT's fitted exponents put s_MLP ≈ 0.60 > s_depth ≈ 0.45 > s_width ≈ 0.22, so MLP should be *relatively narrow* at this scale. −22% params/compute vs B, same citation. |

**Recommendation: A or C, not B.** B takes the throughput hit of a cut without taking the principled
shape, and its stated justification no longer holds. If throughput matters, C. If it doesn't, A —
and the compute saved goes into §6's epoch budget, which has better-evidenced returns.

**Forward-compat (carried from the retired progressive plan §2.2):** if the corpus grows (the ~12M
same-taxonomy expansion), grow **depth-first at 896 width** (896×18 → 896×24 → only then widen
toward 1024), and **gate param re-growth on effective *clean-label* signal, not raw image count**.
Its rough triggers (~9–10M clean → ~248M; ~12–13M → ~300M; ~16–20M → ~320M) assumed option B's
shape — re-derive for whichever of A/B/C is chosen.

**Settled, do not revisit:**
- **patch16.** patch14 buys +31% tokens for 1.36× FLOPs, forces a stem re-init, and breaks the 320
  and 512 grids. `512px/patch16` gives N=1025 — *identical token cost to 448px/patch14* — with real
  pixels instead of a finer mesh over the same downscaled raster.
- **The linear head stays.** *Taming the Sigmoid Bottleneck* (AAAI 2024,
  [arXiv:2310.10443](https://arxiv.org/abs/2310.10443)) Thm 4: all *k*-active assignments are
  argmaxable given **2k+1** dims. At k ≈ 36 that is **73 ≪ 896**. Head expressivity is not a
  bottleneck, and Generalized Neural Collapse settles separability at K ≫ d.
- **Query2Label is impossible here** — one query per class at 19,294 classes is ~1.33 TFLOPs/layer
  (~4.3× the whole backbone) and ~10 GB attention memory per image. ML-Decoder measured a full
  decoder head **OOM at 9,600** classes.

**Architecture adds (all new, all cheap):**

| Add | Setting | Evidence |
|---|---|---|
| **LayerScale** | ε = **0.1** | CaiT Table 1 measures **+1.0 top-1 at exactly depth 18** (80.7 → 81.7) vs a drop-path-tuned baseline. DeiT III uses it in every config. Absent from the codebase. |
| **QK-norm** | **measure first, then decide** | Log max attention logits from epoch 1. Wortsman's threshold: *"all points with attention logits above 1e4 diverged."* Add only if you intend to raise LR — ViT's instability regime is ~8B params, 40× above us. |
| **2D RoPE** | optional, P1 only if adopted | RoPE-ViT (ECCV 2024): +1.4 @384, +2.5 @512 for ViT-L vs interpolated learned pos-embeds, at **0.01% of FLOPs**. Attractive for a plan whose spine is three resolution changes. Changes the ONNX path — prototype before committing. |

---

## 4. Loss — ASL, cited correctly

```yaml
training:
  tag_loss:
    gamma_neg: 7.0        # FIXED, all phases. No descent, no dwell, no guarded steps.
    gamma_pos: 0.0
    clip: 0.05            # changed from 0.2
    alpha: 1.0
    label_smoothing: 0.0
    class_weight_strategy: null
```

**The citation is ASL Appendix F, not P-ASL.** Ridnik et al.
([arXiv:2009.14119](https://arxiv.org/abs/2009.14119)): *"we set all untagged labels as negative…
Since the level of positive-negative imbalancing is significantly higher than MS-COCO, we increased
the level of loss asymmetry: For ASL, we trained with γ⁻ = 7, γ⁺ = 0."* That is assume-negative,
~5,400–9,600 classes, extreme imbalance — structurally our regime, and it pairs γ⁻=7 with **γ⁺=0**
and margin **m = 0.05**. *(Quote nuance: the elided clause is "with reduced weights" — Appendix F
also down-weights the assumed negatives, a knob this plan does not currently replicate.)*

Corroborated three ways: **RAM** (CVPRW 2024, [arXiv:2306.03514](https://arxiv.org/abs/2306.03514))
tags **6,449 categories** from incomplete image-text labels using **plain ASL at γ⁻=7, γ⁺=0,
clip=0.05** — bit-for-bit this config (verified in the official RAM code; the paper text omits the
hyperparameters), and the closest published regime match. **ML-Decoder's**
OpenImages loss table: CE 84.8 · Focal 84.9 · **ASL 86.3**. And **SINR** (ICML 2023,
[arXiv:2306.02564](https://arxiv.org/abs/2306.02564)) at **47,000 positive-only classes** finds full
assume-negative is the workhorse and scales gracefully — the best evidence available about how these
losses behave as C grows from 80 to 47,000.

**clip 0.2 → 0.05.** This lands where the correction doc put it, for a sounder reason: it is ASL's
own assume-negative value at our class count, not a consequence of an ignore set we are no longer
building. Frozen for the whole run; never moved in the same step as anything else.

**Implementation caveat (carried from the retired ASL_plan §1; re-verified 2026-07-29):** our focal
weight is **not detached** (`loss_functions.py:317-329`), unlike the official ASL default — the
extra ∂(focal-weight)/∂p term scales gradient magnitudes ~2–5× in the semi-hard band. Never flip
detach mid-run, and tune any constants against this code's gradient scale, not the paper's.

**P-ASL selective ignore is NOT adopted.**
- **Ω_P is dropped outright.** At η=0.05 it selects **107 of 19,288 tags** (0.55% of vocab, **42.4%
  of label mass**) whose mean estimated missingness is **0.38%** — the tags needing it least — and at
  N=∅ they would receive no negative gradient from any source and collapse to always-on.
- **Ω_L is not adopted this run.** It is an unchecked confirmation loop without a verified-negative
  anchor, and it inherits the exact objection used to reject SPLC (*"assumes a calibrated backbone,
  unsafe in from-scratch Phase 1"*). If revisited: it needs a `change_epoch` delay, and **K must be
  calibrated on the gold slice, not on val** — §9 explains why noisy val drives K too small.
- The **6.8M-image prior-estimation pass is deleted.** It was for Ω_P, and where Ω_P was active
  empirical tag frequency is already within ~0.5% of the true prior.

**Hill/SPLC is a live A/B, not a deferred fallback.** In the closest published regime match —
OpenImages **single-label**, 567 classes, 1.74M images, **no verified negatives** — ASL is beaten by
plain focal loss: **BCE 60.83 · Focal 62.14 · ASL 61.95 · Hill 62.71 · SPLC 62.86**
([arXiv:2112.07368](https://arxiv.org/abs/2112.07368) Table V). Hill's continuous down-weighting of
probable false negatives does Ω_L's job with no confirmation loop, no prior, and no schedule. Run it
as a Phase-2 branch once the instrument is fixed.

**Scope fence (unchanged; carried from the retired ASL_plan §7).** No negative-branch knob touches **wrong-positive**
noise: a mislabeled `aqua_hair=1` trains the wrong side of the boundary at full BCE gradient
regardless of γ_neg, clip, or any ignore set. That is the cleaning track's job, permanently.

**Telemetry, not control.** `ASLDriveManager` → telemetry-only. Keep its per-decile EPR, sibling-gap
and non-GT histogram outputs (full observable definitions + measurement hygiene: **Appendix A**);
remove its authority (its `set_gamma_neg` calls are
`asl_telemetry.py:190` and `:248`, plus a pass-through delegation at `loss_functions.py:473`).
**Invert the precedence at `asl_telemetry.py:146-160`** — a
checkpoint's persisted γ_neg currently *wins over YAML* with only a warning, which under "γ_neg
fixed" silently overrides config on resume.

---

## 5. OPEN DECISION — pretrained initialization

**This is worth more than every architectural lever combined and it has never been argued.** The
plan asserts "from scratch" in a table cell.

*For from-scratch:* the domain gap is real (flat shading, line art, no photographic texture
statistics); Kornblith et al. find fine-tuning gives no substantial benefit on fine-grained targets;
He/Girshick/Dollár find ImageNet pretraining speeds convergence without raising final accuracy given
enough data; and no public checkpoint exists at 896×18/patch16.

*Against:* Steiner et al. Table 3 — ViT-L/16 IN-21k-pretrained **87.08%** vs from-scratch **74.01%**;
even against DeiT III's far better from-scratch recipe (84.9%) it still adds **+2.1pp**. ASL reports
IN-21k over IN-1k pretraining raises **multi-label mAP by "almost 2%"**. Illustration2Vec, the
closest in-domain scholarly precedent, is a fine-tuned ImageNet VGG-16. And Phase 1 alone is
≈1.2e20 FLOPs — multiple weeks on one GPU, ~2/3 of the total budget. Pretraining is the only lever
that buys a large fraction of that back.

**Estimated stake: +1 to +2.5 mAP, plus a large convergence-time saving.**

**Do this before committing (≈1 GPU-day, and it blocks nothing else):**
> Linear-probe **DINOv2-L** and **SigLIP** features on a 100K-image slice against the 19,294-tag
> vocabulary. Compare against V1's per-decile mAP. **No scholarly evaluation of foundation-model
> features on booru-style illustration appears to exist** — the domain-gap argument is plausible and
> currently unmeasured, which is the exact failure mode this plan set criticises elsewhere.

If the probe says the features transfer, four published routes preserve the shape:
**LiGO** growth ([arXiv:2303.00980](https://arxiv.org/abs/2303.00980) — grow ViT-B 768×12 into
896×18, *"55% savings in FLOPs with no performance drop"*); **patient distillation** from a public
ViT-L ([arXiv:2106.05237](https://arxiv.org/abs/2106.05237) — preserves the ONNX path exactly);
**adopt ViT-L/16 outright** and init from DeiT III / SigLIP / DINOv2 (pairs naturally with §3
option A); or **in-domain MAE/DINO SSL** on the same 6.8M images.

---

## 6. Schedule — finish training this time

V1 received **~230M sample presentations** for a 248M from-scratch ViT — about **60%** of what
DeiT-B (2.9× smaller) uses. That, not label noise, is the measured root failure.

**Target: 400–700M samples seen.** Normalized: 1 epoch = 6.8M samples = 6,641 steps at eff batch
1024.

| Phase | Res | Epochs | Samples | Eff. batch | Warmup | Base → peak LR | drop_path | dropout | WD |
|---|---|---|---|---|---|---|---|---|---|
| **1** from-scratch | **320** | **~55 (cap 60); gate on plateau** | 374M | ~1024 | **10K steps (~1.5 ep)** | 2.7e-4 → ~5.4e-4 | **0.25** | **0.0 / 0.0** | 0.05 |
| **2** fine-tune | **448** | ~12; gate on plateau | 82M | ~768 | 2 ep | 1.5e-5 → ~2.6e-5 | 0.15 | 0.0 / 0.0 | 0.05 |
| **3** detail | **512** | 3–4 — **run it, don't gate it** | 27M | — | 1 ep | 8e-6 → ~1.3e-5 | 0.10 | 0.0 / 0.0 | 0.05 |

**≈483M samples total — 2.1× V1.**

**Phase-transition checklist (every step, in order):** select best P1 checkpoint on the §9 metric →
interpolate pos-embeds bicubic 20×20 → 28×28 (`training_utils.py:1990-2047`, implemented; grid
sizes are derived, not hardcoded) → **reset
optimizer state** → update config → re-warmup. `torch.compile` recompiles on first forward
(401 → 785 tokens).

**Notes on the numbers:**
- **drop_path 0.25 stays. Do not lower it.** `model_architecture.py:398` uses
  `torch.linspace(0.0, rate, num_hidden_layers)` — the **timm linear-ramp convention** — so 0.25 is
  a *mean* of 0.125 across 18 blocks. DeiT III's table is in the same convention (ViT-B 0.1,
  ViT-L 0.4), so 0.125 mean for a ~192M model sits correctly between them. Verify the convention
  before ever changing this number.
- **Plain dropout → 0.** DeiT III and big_vision both use stochastic depth *only*. Steiner et al.:
  *"when using the 10× larger ImageNet-21k dataset and keeping compute fixed, any kind of AugReg
  hurts performance for all but the largest models."* At 6.8M images we are in that regime.
  Current: `hidden_dropout_prob: 0.10`, `attention_dropout: 0.05`.
- **WD 0.05 fixed**, per the standing project decision — overriding the old plan's 0.08/0.04.
  (V1 checkpoint record, from the retired correction doc §0.1: P1 ran WD **0.1**, P2 **0.05**.)
  *Documented tension:* DeiT III's long-schedule rule raises WD and drop_path when extending
  training (+0.05 drop_path per 200 epochs). If P1 runs materially past 55 epochs, revisit.
- **Phase-2 LR is not the failure mode.** DeiT III's high-res fine-tune is AdamW 1e-5 @ batch 512 ×
  20 epochs; 2.6e-5 @ 768 is higher in both LR and step count. V1's 1e-5 was also inside DeiT III's
  range. Rule this hypothesis out explicitly.
- **Warmup 10K steps**, not 4 epochs — Beyer et al. use exactly this at batch 1024, and σReparam's
  grid diverges on 7 of 8 configs at ViT-B / batch 1024–2048 / LR 5e-4–1e-3, our exact box.
- **The batch→LR rule (carried from the retired progressive plan §3.5):** **sqrt** scaling for
  AdamW — peak = base × √(eff batch / 256) (Malladi et al.,
  [arXiv:2205.10287](https://arxiv.org/abs/2205.10287)); linear scaling is SGD-only. The full rule
  also rescales **ε → ε/√κ** (κ≈4 halves it, consistent with AdamW8bit's small ε). The /256 base is
  a free reference — keep it consistent with where the base LR was actually tuned. Re-apply
  whenever eff batch is retuned to real VRAM.
- **Base-LR bookkeeping to pin down before the run:** the "2.7e-4 ≈ 1.14× V1 (muP)" bump only holds
  if V1's 2.5e-4 was a /256-normalized *base*. If it was the actual optimizer LR at eff batch 96,
  sqrt-scaling to 1024 gives ~8.2e-4 and we are 35% *below* our own anchor. Check V1-P1's
  `gradient_accumulation_steps`. **LR guard:** grad_norm drift >2× in the first 5 epochs → fall back
  to base 2.5e-4.
- **Phase 3 is no longer gated.** ASL COCO 448→640 **+1.4 mAP**; ML-Decoder **+1.1**; Query2Label
  **+1.1** — unusually consistent across three multi-label papers, and it targets exactly the
  small-detail tags (hair ornaments, accessories). The old fine-bucket-plateau trigger would
  under-fire, because the same tail-metric noise that hides the plateau also hides the benefit.
  Sanity check first: what fraction of the corpus is natively below 512px?

**Optimizer:** AdamW8bit, **except fp32 optimizer state for the 19,294-way head and position
embeddings** (~140 MB via bitsandbytes `GlobalOptimManager`); verify block-wise *dynamic* (not
linear) quantization. 8-bit AdamW is not verified lossless for ViT-from-scratch —
[arXiv:2309.01507](https://arxiv.org/abs/2309.01507) Table 2 measures Swin-T IN-1k from scratch at
**81.0 (8-bit) vs 81.2 (fp32)**, exceeding seed std, and Dettmers' paper contains no ViT and no
8-bit-*Adam* vision result at all. A 19K head at 0.18% positive rate is structurally the sparse
embedding shape that motivated the Stable Embedding Layer.

**Do not adopt LAMB.** Its entire measured advantage is at batch ≥16K; ImageNet's critical batch
size is 1,000–15,000, so batch 1024 sits at the bottom of the range where AdamW is not the
bottleneck. Beyer et al. measure batch 1024 → 4096 *dropping* accuracy 76.5 → 74.7.

**Weight EMA — add it.** α ≈ **0.9998** (horizon ≈ 1% of total steps, per
[arXiv:2502.06761](https://arxiv.org/abs/2502.06761)), **not** the copy-pasted 0.999. Evaluate the
online *and* EMA weights at every validation and keep both checkpoints, so the downside is exactly
zero. The payoff is bimodal: **+0.1–0.2** if this run behaves like clean ImageNet (which is why
DeiT III drops EMA), **+9 points** if late-training noise memorization is active (CIFAR-100N
ResNet-34 55.50 → **65.15**, [arXiv:2411.18704](https://arxiv.org/abs/2411.18704)). This plan's whole
thesis is that it *is* active. It also cuts prediction churn 18.84 → 11.69 and post-temperature ECE
4.67 → 3.13, which feeds §9 directly. *(Non-transfer flag: the noise evidence is CNN/single-label;
the ViT evidence is clean-label ImageNet where the gain is near-nil.)*
**Do not** add SWA on top (near-substitute, measured within noise) or pursue model soups (needs a
shared pretrained init).

---

## 7. Augmentation — unchanged, and better justified than before

Phase 1 at full strength; Phase 2/3 reduced (a *legitimate* reduction now that P1 actually
converges). Per-phase values (inlined 2026-07-29 from the retired progressive plan §3.2/§3.4):

| Aug | Phase 1 (320) | Phase 2 (448) | Phase 3 (512) |
|---|---|---|---|
| horizontal flip + orientation-aware tag swap | p=0.5 | p=0.5 (lossless, unchanged) | as P2 |
| colour jitter brightness / contrast / saturation | 0.30 / 0.20 / 0.08 @ p=0.5 | 0.22 / 0.15 / 0.06 | as P2 |
| random rotation, bicubic | ±[2°,8°] @ p=0.50 | ±[2°,5°] @ p=0.30 | as P2 |
| gaussian blur, kernel 3 | p=0.30, σ ∈ [0.1, 1.5] | p=0.15, σ ∈ [0.1, 1.0] | as P2 |
| mixup / cutmix / randaugment / random erasing / hue rotation | **none** | **none** | **none** |

*(The retired plan specced no separate P3 augmentation; "as P2" is this plan's reading of its
"Phase 2/3 reduced" rule.)*

**The exclusion is now the better-supported choice, not merely acceptable.** Plain Mixup under
partial labels measures **−12.9 mAP at 10% labels and is still negative at 90%**
([arXiv:2405.15860](https://arxiv.org/abs/2405.15860) Table). Label interpolation is justified only
by softmax-CE's linearity in the target and directly contradicts sigmoid + asymmetric focusing.
CutMix's area∝semantics assumption is undefined for global anime tags (`1girl`, `long_hair`, style
and meta tags). Hue rotation is excluded because hue is *categorical* for anime — even ~20° can
turn `blue_eyes` into `green_eyes` (carried from the retired overfitting assessment). And every large-scale ViT recipe at ≥14M images uses augmentation *no richer than
ours* — Scaling ViT's entire JFT-3B pipeline is `inception_crop + flip_lr`.

*Honest counterweight:* the accurate claim is "expected gain is small and the interaction risk with
partial labels is large," not "augmentation cannot help at scale." Steiner's "hurts" result is at
fixed 30-epoch compute; augmentation recovers at 300 epochs.

**If a mixing method is ever wanted:** SpliceMix (but a 2×2 splice at 448 means each source is seen
at 224, which attacks precisely our resolution-sensitive confusables), or LogicMix — *gated on
building a per-image unknown-label mask*, without which it degenerates to plain hard union and
buys nothing. Not BalanceMix (+0.2 over ASL on clean data, and its noise mode is wrong-positive,
our minority). Not CutMix+LP (needs pixel maps for 19,294 tags).

---

## 8. LAUNCH BLOCKERS — the instrument

**Nothing below §8 is measurable until these clear.** Most of the plan's expected deltas (0.2–1.0
mAP) are at or below the current instrument's resolution.

### 8.1 Already done — strike from the list
`selection_metric: val_mAP` is **live** (`unified_config.yaml:467`; dispatch table
`train_direct.py:59-63`, read `:1680-1695`, applied `:2971-2973`, early-stop message `:3098`). The
correction doc's item 2 was stale.

### 8.2 Val set — the biggest defect, and cheap
`max_val_samples: 30000` (0.44% of corpus) gives:

| | tags <5 val positives | tags <10 | decile-10 median | decile-10 mAP 95% CI |
|---|---|---|---|---|
| **30,000 (now)** | **44.5%** | **65.8%** | **2.4** | **±0.0125** |
| 276,000 (full 5%) | 0% | 0% | 21.8 | ±0.0045 |

Aggregate mAP survives (±0.0027 — fine). **Everything below aggregate does not.** Confirmed against
the real artifact: on 30K only **10,406 of 19,292** tags have support ≥5.

- [ ] **Raise `max_val_samples` to the full ~276K split.** Cost is validation time only.
- [ ] **Stop folding the val excess into training** (`logs/training.log:13127`: *"moved 266,056 to
      training"*). The `pr_threshold_*_full296k` artifacts are ~90% train-contaminated and must not
      be quoted as held-out.
- [ ] **Carve a held-out TEST set.** Val currently does triple duty — selection, threshold fitting,
      and reporting — so every reported number is optimistically biased.
- [ ] **Group-aware split** on perceptual-hash cluster ID. Current split is uniform random over
      individual sidecars (`dataset_loader.py:2727-2741`); measured near-dup rate 0.26%, and ~382K
      images added after the dedup scan were never checked.

### 8.3 Metric correctness
- [x] **Binned AP — resolved, keep `ap_thresholds: 200`.** *(Corrected: an earlier draft said set
      `thresholds=None`. That is impractical and unnecessary.)* The exact estimator retains every
      update's preds/targets (~7 GB resident at 30K × 19.3K labels), and the transient peak scales
      brutally — measured on RTX 5090 @ B=48: **200 → 9.3 GB / 40 ms per update; 500 → 23.3 GB;
      2000 → 93 GB and ~103 min per validation pass** (`train_direct.py:1702-1738`,
      `configs/unified_config.yaml:553-560`). The binning bias has been **measured at ~−0.0031 vs
      exact, near-uniform across support buckets**, so it largely cancels in checkpoint-to-checkpoint
      comparison. Nothing to change.
- [ ] **Set `early_stopping_threshold` from the measured resolution floor, not a guess.** Three
      independent sources agree on ~0.003: the binning bias (−0.0031), the 30K sampling CI
      (±0.0027), and the config's own note *"treat val_mAP differences below ~0.003 as noise."*
      Use **3e-3**. Currently 5e-7 — i.e. every epoch counts as an improvement.
      *(At the §8.2 full 276K val split the sampling CI drops to ±0.0009 and the binning bias
      dominates, so 3e-3 still holds.)*
- [ ] **Freeze the macro-average tag list.** `train_direct.py:2826` masks to `val_pos_counts > 0`,
      dropping ~958 tags on the 30K draw (logged at debug level only — silent in practice).
- [ ] **Per-tag-optimal macro-F1 inside the val loop** (~20 lines). Note that flipping
      `threshold_calibration: per_tag` is a **no-op for selection** — its output goes to
      `thresholds.json`/TensorBoard only and never reaches the metric.
      `ThresholdCalibrator._compute_f1_grid` already returns a `(num_thresholds, C)` grid;
      `grid.max(axis=0).mean()` over supported columns *is* the number. Gate the 19,292-entry log
      line and TB write.
- [ ] **Reset `best_metric` on any metric change** — `train_direct.py:1254-1255` does
      `max(best_metric, loaded_best)`, and old checkpoints hold F1-scale values (0.013–0.045)
      against mAP ~0.67.
- [ ] **Per-decile mAP** (deciles from `vocabulary.json`; assignment already exists at
      `asl_telemetry.py:331-348`, per-group mAP at `evaluation_metrics.py:491-493`). Un-gate from
      `use_tensorboard`.

### 8.5 Support floor — selection and reporting are different metrics

Macro-averaging over all 19,292 tags is a *choice*, not a default, and an unqualified one puts
roughly half the metric's weight on tags that cannot be measured. Even at the full 276K val split,
**47.8% of tags have <50 positives**. Rare tags are genuinely learnable — the vocab is floored at
500 occurrences and tail tags ride on features the head built, so they are not learned from scratch
— but *learnable* and *measurable* are different problems, and only the second one is ours here.

**The V1 evidence, extracted from the TensorBoard event files (Phase 1 final, step 209135):**

| bucket | tags | **mean support (30K val)** | mAP | f1_macro @ 0.2653 |
|---|---|---|---|---|
| 500–999 | 7,619 | **2.45** | **0.6250** | 0.0279 |
| 1,000–4,999 | 7,873 | 8.52 | **0.6345** | 0.0421 |
| 5,000–9,999 | 1,482 | 30.3 | 0.5713 | 0.0567 |
| 10,000+ | 2,314 | 399 | 0.5809 | 0.0782 |
| 300–499 | **0** | 0 | 0 | bucket is structurally empty (vocab floored at 500) |

**Rare tags are demonstrably being learned — this is not a small-sample artifact.** For a tag with
2.45 positives in 30,000 images, **random ranking gives AP ≈ 0.00042**. The observed 0.6250 is
~1,500× that, and it rose monotonically across Phase 1 (0.6028 → 0.6153 → 0.6250). Concretely, the
model is placing a 500-occurrence tag's true positive at about **rank 2 of 30,000**. Per-*tag* AP at
2.45 positives is high-variance, but it is not *biased* — the mean over 7,619 tags has SE ≈ 0.005,
so the bucket figure is trustworthy even though no individual tag's is.

Mechanism: tail tags do not learn from scratch. `fox_hair_ornament` inherits hair/ornament/fox
representations built from millions of head-tag examples and only learns a small delta. This is why
500 examples suffices in a 19K-way multi-label setting and would not for an isolated binary
classifier — the same reason Tencent ML-Images (11,166 classes) and RAM (6,449 tags) work.

The rare buckets also score *lower* on F1 than the head while scoring *higher* on mAP. That
divergence is the fixed-threshold artifact again (§4.1), not a statement about the tail.

*(This does dispose of progressive-plan §2.1's "rare bucket still climbing while others plateau" —
all four buckets climbed at comparable rates. But the rare bucket's *level* is real.)*

| support floor (val positives, 276K split) | tags kept | % of vocab | **% of label mass** |
|---|---|---|---|
| none | 19,288 | 100% | 100% |
| ≥25 | 16,556 | 85.8% | **99.3%** |
| **≥50 ← use this** | **10,074** | **52.2%** | **96.9%** |
| ≥100 | 6,226 | 32.3% | 93.9% |

**The floor belongs on per-tag operations, NOT on the selection metric.** Averaging is the whole
point: per-tag AP is noisy but unbiased, so a mean over thousands of tags is precise even when no
individual term is.

- [ ] **Selection metric = macro-mAP over ALL tags, on a frozen list.** No support floor. Simulated
      95% CI at 30K val is **±0.0027**, and at the full 276K split **±0.0009** — ample to resolve
      the deltas this plan is chasing. What matters is that the tag list is **frozen**, so the
      denominator cannot drift between checkpoints (`train_direct.py:2826` currently lets it).
- [ ] **Reporting = per decile, all tags.** Reliable at decile granularity (~1,929 tags → 95% CI
      ±0.0045 at 276K). Do not over-read single-epoch deltas in deciles 9–10; do trust the level
      and the trend.
- [ ] **Support floor ≥50 applies to per-TAG operations only** — per-tag thresholds
      (`_calibrate_per_tag` has only a zero-support fallback, no *minimum*-support floor —
      `evaluation_metrics.py:605-626`),
      per-tag calibration (§8.4 uses ≥100 for Platt), and any per-tag claim. Below the floor, fall
      back to the tag's decile-level operating point rather than fitting a free parameter on ~5
      events. `tools/find_pr_threshold.py` already has the `--min-support` pattern (flag at `:85`,
      default 5; consumed at `:310`).

**Hygiene backlog (non-blocking, carried 2026-07-29):** ghost config keys `use_style_token` /
`num_special_tokens` / `num_groups` / `tags_per_group` are filtered as unused
(`train_direct.py:771-776`; `tags_per_group: 10000` still in the live config) and
`validation_loop.py`'s 3D-output handling is vestigial — delete when convenient.
`loss_functions.py:45`'s `label_smoothing` default is 0.05 while the operative value is 0.0
(config-driven, §4 pins 0.0) — fix the default. Dead-code candidates (from the retired
`deprecated_candiates.md`, re-verified 2026-07-29): `log_index_order_hash`,
`LearningRateSchedulerFactory`, `TrainingMetricsTracker` in `training_utils.py` are referenced only
by its `__main__` self-test block.

**Deployment consequence.** The bucket mean says the tail is learned; it does **not** say *which*
tail tags are individually reliable, and 2.45 val positives cannot tell you. Surfacing rare
predictions therefore needs a decile-level operating point plus the gold slice, not a per-tag
threshold fitted on noise. Gate 3 is where per-tag tail reliability actually gets established —
the slice is stratified across deciles with confusables oversampled, which is the only affordable
way to measure the tail *per tag*.

**Not recommended, but worth knowing the shape of:** raising `vocab_min_frequency` 500 → 2000 would
drop **62.6% of tags for 5.1% of label mass**. That is a *product* decision, not a metric one — tag
coverage is what distinguishes this tagger, so leave the floor at 500.

### 8.4 Thresholding and calibration — replace the specified approach
**Per-tag isotonic across 19K tags is not viable.** Niculescu-Mizil & Caruana (ICML 2005): isotonic
**overfits below ~2,000 calibration points**. Our median tag has ~6. Ullah et al.
([arXiv:2411.04276](https://arxiv.org/abs/2411.04276)) — the only calibration study at XMC scale —
use a **global** calibrator and name per-label calibration as *future work*. They also show marginal
per-label ECE is misleading (**ECE 0.05 vs ECE@5 = 9.25** on the same model) and that *"marginal
calibration… does not imply top-k calibration."*

Calibrate after training, not via the loss: asymmetric losses are not strictly proper scoring
rules, so outputs stay systematically shifted regardless of γ tuning (Cheng & Vasconcelos,
CVPR 2024 — carried from the retired ASL_plan §5).

- [ ] **Global isotonic on top-k scores** (their validated recipe: ECE@1 17.02% → **0.17%**, no
      accuracy loss), **plus per-decile isotonic** (~1,900 tags each = ample support), **plus
      per-tag Platt only for the ~1,230 tags with ≥100 calibration positives**, shrunk toward the
      decile fit.
- [ ] **Fit on a disjoint CALIB split**, never on the reported eval split. Thresholds fit on noisy
      labels skew **high** (§9.1's bias at the operating point: firing on unlabeled true positives
      is scored as FP); before shipping, cross-check per-group thresholds against the Anima slice
      (probe, not anchor — §9.4).
- [ ] **ECE estimator:** use smoothed / equal-sample-bin ECE (SmoothECE, Błasiok 2023; Dual-TS,
      [arXiv:2308.08366](https://arxiv.org/abs/2308.08366)) — vanilla equal-width ECE is biased at
      our ~0.18% positive rate. Applies to Gate 2's ECE@k and per-decile reliability diagrams.
- [ ] **Adopt xCOLUMNs** ([arXiv:2401.16594](https://arxiv.org/abs/2401.16594) ICLR 2024;
      [arXiv:2311.05081](https://arxiv.org/abs/2311.05081) NeurIPS 2023;
      [library](https://github.com/mwydmuch/xCOLUMNs)) — statistically consistent, **post-hoc,
      retraining-free** optimization of macro-F1/macro-recall at a per-instance budget k, built for
      extreme label spaces. Drops onto a 19,294-way sigmoid head with no retraining and no verified
      negatives. Two independent literature sweeps converged on this.
- [ ] **Validate the metric panel** using their NeurIPS'23 Table 1 test: score a head-only-capable
      model with every metric you plan to use. Under total tail collapse P@5 drops 8.7% and PSP@5
      18.4%, while **macro-F1@5 drops 88.5%**. Any metric that doesn't move cannot detect tail
      failure.

---

## 9. How we decide V2 is actually better

### 9.1 The bias that points against V2

For tag `c` with missing rate ρ_c, the val set contains truly-positive-but-labelled-negative images.
A model that has learned the concept ranks them **high**, and in AP a false positive at rank 1 costs
far more than at rank 500. So **measured AP is biased downward, and the bias grows with the model's
true quality.** This is model-dependent systematic bias, not variance that averaging removes.

Measured ρ makes it worst exactly where we want to look: ~0.3–0.5% at head, ~5% median at the tail
decile, **23–64% on individual confusable tags**. Northcutt et al. show this **flips model rankings**
at only 3.4% average label error.

**Three consequences to pre-empt:**
1. V2 can be genuinely better and measure worse, especially in tail deciles.
2. **Any ignore-set / relabeling hyperparameter tuned on noisy val will be tuned too conservatively**
   — this is why §4 says calibrate on the gold slice, not val.
3. Early stopping on noisy val can peak and decline as the model exceeds the annotation. *(This did
   not cause V1's stop, which was manual.)*

### 9.2 Free mitigation: sibling-negative evaluation
OpenImages evaluation ignores unannotated classes and does not penalize false positives on them. We
can't apply that directly, but we **can** restrict evaluation to cells with *evidence of negativity*:
tags in a mutually-exclusive group where a sibling is positively labelled. For hair colour/length and
the other confusable groups — where ρ is 23–64% and where review budget goes — a sibling-positive
label *is* reliable evidence of negativity. Low-noise evaluation on the hardest tags at zero
annotation cost.

### 9.3 The three-gate protocol — pre-register before the run

**Frozen artifacts** (hashed, versioned, reused identically for V1 and V2): `EVAL-SET` (≥276K,
never used for fitting), `CALIB-SET` (disjoint), `TAGSET` (frozen list + decile assignments),
`GOLD-SLICE` (§9.4). One **offline scorer**, not the training loop.

**Gate 1 — ranking quality on the noisy set.** Exact macro-mAP over `TAGSET`, **per decile**, with
paired **image-level** bootstrap 95% CIs (bootstrap over images, not tag×image cells — cells are
correlated within an image).
*Pass:* V2 ≥ V1 aggregate, CI on Δ excludes 0 in at least the top 5 deciles.
**Do not fail V2 on the bottom deciles here** — §9.1 predicts adverse bias exactly there. Record and
adjudicate at Gate 3.

**Gate 2 — deployment quality at the shipping operating point.** Calibrate on `CALIB-SET` only
(§8.4), then report macro-F1@k, macro-recall@k, coverage@k, ECE@k and per-decile reliability
diagrams on `EVAL-SET`.
*Pass:* no regression in macro-F1@k or coverage@k; ECE@k improved or flat.
This catches "mAP went up but the shipped tagger got worse" — Gate 1 is structurally blind to it,
because a per-tag monotone recalibration changes every deployed output while leaving every AP
*exactly* unchanged.

**Gate 3 — the arbiter.** On `GOLD-SLICE`, statAP-estimated per-tag AP/precision/recall for V1 and
V2 on the **shared pool**, with variance estimates and paired bootstrap.
***Decision: V2 > V1 iff the paired CI on Δ excludes 0 on the gold slice, with the improvement
present in the tail-decile stratum.*** Also report the off-diagonal confusable-group metric (§9.2).

**Pre-committed falsifier:** "Gate 1 passes but Gate 3's tail stratum includes 0" ⇒ the aggregate
gain is head-driven, and the response is to revisit the loss/data, **not** to ship.

### 9.4 The gold slice — start it now, it is the long pole
Exhaustive 19K annotation is impossible and unnecessary. **Pooled stratified adjudication:**
- ~200–300 **tags**, stratified across frequency deciles, oversampling confusable groups.
- Per tag, pool = **V1's top-N ∪ V2's top-N ∪ a random corpus sample**, with **recorded inclusion
  probabilities**.
- Adjudicate **double-reviewed with measured κ**. The 2.2× inter-reviewer spread on identical
  candidates would otherwise dominate — reviewer reliability is the prerequisite to the prerequisite.
- Estimate with **statAP / infAP** (Yilmaz & Aslam SIGIR 2008; Aslam & Yilmaz CIKM 2006) — unbiased
  under known inclusion probabilities, with variance estimates.
- Record an **ε_gold** for the slice itself. A screened-not-adjudicated slice cannot certify.

**Pooling both systems is the critical design choice** — it removes the better-model-penalized-more
bias for the compared pair. Standard TREC methodology, and exactly what Schultheis et al. (KDD 2022
§4.1) recommend. **Do not use the Anima synthetic set as the arbiter** — it measures
model-precision-on-synthetic-data. (The edits.db verdict store is ~705K verdicts, all on synthetic
Anima IDs; the live db was wiped 2026-07-27 — 58,207 predictions, 0 verdicts.)

---

## 10. Data — re-scoped, not de-scoped

The correction doc's *facts* are right: cleaning has touched 0.65% of tags and **0.017% of label
mass**, and will not move global ρ on V2's timeline. Its *conclusion* — drop the workstream — is
backwards. That number says the campaign is under-resourced and mis-targeted.

**Be precise about which data lever.** RAM++
([arXiv:2310.15200](https://arxiv.org/abs/2310.15200)) scales **tags** 12.0M → 41.7M for
**OpenImages-rare 63.54 → 67.17 (+3.6)**. That is *label completion*. By contrast Zhai et al.
measured **JFT-300M → JFT-3B (10× more noisy images) buys only ~1%**, equally for small and large
models. **Completing labels on the corpus we have ≫ adding more equally-noisy images.**

**Retargeting, highest value first:**
1. **Fix reviewer reliability before scaling review.** 2.2× spread on identical candidates means
   throughput currently buys variance. Rubrics + adjudication + measured κ. Everything downstream —
   gold slice, thresholds, gates — inherits this. (Measured basis, from the retired correction doc:
   four screening sessions over the identical 743 FP candidates converted at 28.8 / 53.7 / 56.8 /
   62.7% — `L:\Dab\DataCleaning Project\screening\runs\*.db`.)
2. **Spend review on the confusable groups.** Wrong positives are the entire share that *no* loss
   can address (§4 scope fence), and they are where §9.2's free evaluation lives. **Method of
   record (carried from the retired progressive plan §1.7):** Confident-Learning multi-label
   detect-and-remove, gated on the gold slice and human-reviewed — never auto-remove on rare
   adjacent tags; HLC only ever as a colour sub-vocabulary pilot (unproven >80 classes).
3. **Stop spending review on head tags.** 76% of adds went to 14 very-high-frequency tags with
   ρ ≈ 0.3–0.5%, on labels the model already learns well.

**Per-tag missingness is estimable** — contra the retired ASL_plan §7's *"we don't have per-tag missingness
rates."* Fitting the 123 usable campaign tags against their actual training counts:
`ρ̂(c) ≈ 0.50 × freq(c)^−0.365` (corr −0.52, R² 0.27) → 0.19% at 4M, 1.73% at 10K, 5.17% at 500. A
lower bound with a loose fit, but enough to drive a frequency-conditioned ignore set or negative
weight **if** one is ever wanted: selecting on ρ̂ ≥ 5% covers 1,330 tags (6.9% of vocab) but only
**0.3% of label mass** — the exact inverse of Ω_P's profile.

*Measurement caveats on the campaign's own numbers:* the 4.9:1 missing:wrong ratio compares a pool
mined at 0.155% (24.8M FP candidates → 38,463 adds) against one mined at 69.6% (11,279 → 7,852) —
a ~450× difference, so it measures review budget, not noise composition. And the `gt_current = 0`
tags (`exercise`, `beam_saber`, `white_wristband`) **do** have labels in the training vocabulary
(559 / 998 / 705); the zero is in the DataCleaning Project's separate store, which disagrees with
`vocabulary.json` by a median 1.115×. Use the report's `training_count_oppai` column as the
denominator — it matches `vocabulary.json` exactly for all 126 tags.

---

## 11. Explicitly rejected, with the reason

| Rejected | Why |
|---|---|
| γ_neg descent (7→6→5) | V1-P1 ran γ=4, so "reverse V1" was void; gating premise (falling global ρ) unreachable; no published multi-label work anneals γ_neg down |
| P-ASL Ω_P prior ignore | Degenerate at N=∅; selects the 107 lowest-ρ tags holding 42.4% of label mass |
| P-ASL Ω_L top-K ignore | Unchecked confirmation loop without a verified-negative anchor; inherits the objection used to reject SPLC |
| 6.8M-image prior-estimation pass | Was for Ω_P; empirical frequency is already within ~0.5% where Ω_P was active |
| Propensity-scored losses / metrics | Schultheis KDD 2022: A=0.55/B=1.5 are not fitted for any dataset; unnormalized PSP@1 = 326% refutes the model; selecting on it picked a model at 48.58% actual P@1 vs 60.83% for doing nothing |
| Per-tag isotonic across 19K tags | Isotonic overfits below ~2,000 points; median tag has ~6; open problem at XMC scale |
| Query2Label | ~4.3× the backbone in self-attention, ~10 GB attention memory/image at 19,294 queries |
| LAMB | Entire measured advantage is at batch ≥16K; batch 1024 is at the bottom of ImageNet's critical-batch range |
| Naive mixup / cutmix / randaugment | −12.9 mAP at 10% partial labels; interpolation contradicts sigmoid+ASL; area∝semantics undefined for global tags |
| SWA on top of EMA; model soups | Near-substitute; soups needs a shared pretrained init we don't have |
| patch14 | +31% tokens for 1.36× FLOPs, stem re-init, breaks the 320 and 512 grids |
| ML-Decoder head *this run* | +0.7 at 9,600 classes and the gain doesn't grow with class count (80→+1.1, 9,600→+0.7); CaiT shows attention pooling *ties* average pooling from scratch at matched capacity. Revisit after the instrument is fixed — it's below current measurement resolution. |
| RAL ([arXiv:2308.05542](https://arxiv.org/abs/2308.05542)) | Documented next-loss-family fallback in the retired ASL plan; not adopted |
| APL ([arXiv:2304.05361](https://arxiv.org/abs/2304.05361)) | Context only, not adopted |
| Schultheis & Babbar unbiased-BCE estimators ([arXiv:2109.11282](https://arxiv.org/abs/2109.11282)) | Require known per-tag missingness; §10's ρ̂ fit now partially weakens that objection — recorded, still not adopted |
| COMIC | Strongest rejected joint long-tail + missing-label framework — setting-mismatched |

**Citation corrections carried forward:** `arXiv:2501.02364` is about *intrinsic data dimension*, not
class count — drop it from the width argument. `arXiv:2601.20994` ("Depth Delusion") is a
**language-model** study; cite as cross-modal tension, not ViT evidence. FixRes describes a
RandomResizedCrop discrepancy that a downscale-only letterbox pipeline does not have — cite it only
for "train low-res, fine-tune at target res helps." *(The config's old "(FixRes recommendation)"
label on the re-warmup was already corrected 2026-07-28 — `unified_config.yaml:262-271` — nothing
left to fix there.)*

---

## 12. Order of work

**Now, in parallel:**
1. §8 launch blockers — val set, metric correctness, calibration. Nothing else is measurable first.
2. §9.4 gold slice — the long pole; weeks. Start reviewer-reliability work immediately.
3. §5 pretraining probe — ~1 GPU-day, blocks nothing, worth +1 to +2.5 mAP.
4. **Resume `experiments/run1_vit/checkpoints/last.pt` and run Phase 2 to its planned E14.** ~9
   epochs. This is the cheapest decisive experiment available: if mAP keeps climbing past ~0.69 the
   ceiling premise is dead for good; if it genuinely flattens, we will have *measured* it for the
   first time. Either way §3's sizing decision stops being a guess.

**Then decide:** §3 (sizing) and §5 (init). Both need the above.

**Then run:** §6 Phase 1 → 2 → 3, with §4's loss and §8's instrument, judged by §9's three gates.

---

## Appendix A — ASL telemetry spec (carried from the retired ASL_plan §5, 2026-07-02 re-spec)

*Inlined 2026-07-29 before ASL_plan.md was retired. Context shift: the γ_neg descent is withdrawn
(§4, §11), so every "γ step" gate role in the original is **inert** — these are the observables for
a fixed-γ run, surfaced by the telemetry-only `ASLDriveManager`.*

| Metric | Definition | Role under fixed γ |
|---|---|---|
| `Δp_mean` | mean(p_pos) − mean(p_neg) | Coarse health only — arithmetically insensitive to γ at 19K tags |
| `Δp_hard` | mean(p_pos) − mean(top-10 non-GT scores) | Boundary-relevant gap. Standing caveat: crushing unlabeled true positives *lowers* top non-GT scores and therefore **widens** it — it can veto, never green-light |
| `pred_pos_ratio` (EPR) | **threshold-free**: Σᵢ pᵢ / expected positives (Cole 2021's formulation, not a thresholded count), per tag-frequency decile, EMA-smoothed | Suppression tripwire. Healthy operating point ≈ 1/(1−ρ) **> 1** (missing positives inflate it). Alarm: sustained >5–10% relative drop in any decile within ~2 epochs of any loss/config change |
| non-GT score histogram | bucket counts of non-GT scores over [0.05, 0.95] per validation pass; watch band **[0.2, 0.5]** | Clip-cost observable: pile-up in-band + stable EPR = clip too high (log it for a future clip run); pile-up + falling EPR = over-suppression |
| sibling-gap | per registered confusable group (hair colour/length etc.): score of the labeled sibling − max score among unlabeled siblings | The *direct* boundary observable, robust to missing positives outside the group; feeds §9.2 |
| Anima recall canary | recall on the 7,805 known-positive synthetic images (prompt-controlled GT), per eval | The only label-clean recall signal; **probe, not anchor** — it never certifies (§9.4) |

**Measurement hygiene (all of the above):** compute on `logits[:, 2:]` — columns 0–1 (PAD/UNK) are
live, loss-free, drifting outputs, and a naive top-K non-GT capture can be dominated by them (val
metrics already special-case this via `skip_metric_cols`). Decide rating-tag handling explicitly
(they inflate mean(p_pos)). fp32-upcast before sigmoid (bf16 granularity ~0.004 near p=0.5). Sample
only on optimizer-update boundaries (reuse `is_update_boundary`) so gradient accumulation doesn't
alias the EMA. Val-side variants are pure consumers of the already-accumulated prob/target
matrices.

Optional supplementary channel: **Label Wave** (ICLR 2024) — the prediction-fluctuation minimum on
a fixed training subset marks the onset of fitting mislabeled data, validation-free. Single-label /
CIFAR-scale evidence: bookkeeping channel and sanity check only, never a gate.

**Supplementary observables (carried from the retired progressive plan §1.4; secondary — log,
don't gate):** `rare_bucket_ECE` (<1000-freq tags) computed with the §8.4 smoothed-ECE estimator;
`cooccur_jaccard@K` (co-occurrence sanity of top-K predictions — estimate the prior on the clean
slice, not noisy val); `logit_std_rank_11_50` (spread of the just-below-top-10 band). Trigger rule:
two consecutive monotonic problem-direction moves at E5+ → spot-check predictions by hand.

*Not carried forward (dead with the descent):* the clean-slice 5→4 gate set, the mid-run Hill/SPLC
fallback trigger (Hill/SPLC is now a live Phase-2 A/B, §4), and ASL_plan §5's per-tag-isotonic
calibration default (superseded by §8.4's global-isotonic + per-decile + shrunk-Platt recipe).

---

## 13. Sources

**Loss.** ASL, ICCV 2021 — [arXiv:2009.14119](https://arxiv.org/abs/2009.14119) (**Appendix F is the
citation for γ⁻=7 / γ⁺=0 / m=0.05**) · RAM, CVPRW 2024 —
[arXiv:2306.03514](https://arxiv.org/abs/2306.03514) · RAM++ —
[arXiv:2310.15200](https://arxiv.org/abs/2310.15200) · SINR, ICML 2023 —
[arXiv:2306.02564](https://arxiv.org/abs/2306.02564) · Hill/SPLC —
[arXiv:2112.07368](https://arxiv.org/abs/2112.07368) · P-ASL/CSL, CVPR 2022 —
[arXiv:2110.10955](https://arxiv.org/abs/2110.10955) (*not adopted — see §4*) · Large Loss Matters —
[arXiv:2206.03740](https://arxiv.org/abs/2206.03740) · ELR —
[arXiv:2007.00151](https://arxiv.org/abs/2007.00151) · HLC, ICCV 2023 (wrong-positive track)

**Architecture / recipe.** DeiT III, ECCV 2022 —
[arXiv:2204.07118](https://arxiv.org/abs/2204.07118) · Steiner et al. —
[arXiv:2106.10270](https://arxiv.org/abs/2106.10270) · Scaling ViT, CVPR 2022 —
[arXiv:2106.04560](https://arxiv.org/abs/2106.04560) · SoViT, NeurIPS 2023 —
[arXiv:2305.13035](https://arxiv.org/abs/2305.13035) · CaiT/LayerScale —
[arXiv:2103.17239](https://arxiv.org/abs/2103.17239) · EMA dynamics, TMLR 2024 —
[arXiv:2411.18704](https://arxiv.org/abs/2411.18704) · When/where to average —
[arXiv:2502.06761](https://arxiv.org/abs/2502.06761) · RoPE-ViT, ECCV 2024 —
[arXiv:2403.13298](https://arxiv.org/abs/2403.13298) · QK-norm —
[arXiv:2309.14322](https://arxiv.org/abs/2309.14322) · σReparam, ICML 2023 —
[arXiv:2303.06296](https://arxiv.org/abs/2303.06296) · 4-bit optimizer states —
[arXiv:2309.01507](https://arxiv.org/abs/2309.01507) · Beyer et al. plain-ViT baselines —
[arXiv:2205.01580](https://arxiv.org/abs/2205.01580) · Malladi et al., sqrt LR scaling for Adam —
[arXiv:2205.10287](https://arxiv.org/abs/2205.10287) · Sigmoid bottleneck, AAAI 2024 —
[arXiv:2310.10443](https://arxiv.org/abs/2310.10443) · ML-Decoder, WACV 2023 —
[arXiv:2111.12933](https://arxiv.org/abs/2111.12933) · LiGO, ICLR 2023 —
[arXiv:2303.00980](https://arxiv.org/abs/2303.00980) · Patient distillation —
[arXiv:2106.05237](https://arxiv.org/abs/2106.05237) · Scaling laws in patchification —
[arXiv:2502.03738](https://arxiv.org/abs/2502.03738)

**Evaluation.** xCOLUMNs / macro-at-k, ICLR 2024 —
[arXiv:2401.16594](https://arxiv.org/abs/2401.16594) · Generalized test utilities, NeurIPS 2023 —
[arXiv:2311.05081](https://arxiv.org/abs/2311.05081) · Missing labels & propensities, KDD 2022 —
[arXiv:2207.13186](https://arxiv.org/abs/2207.13186) · Calibration at XMC scale —
[arXiv:2411.04276](https://arxiv.org/abs/2411.04276) · Pervasive label errors, NeurIPS 2021 —
[arXiv:2103.14749](https://arxiv.org/abs/2103.14749) · Zhao & Gomes —
[arXiv:2102.08427](https://arxiv.org/abs/2102.08427) (*context only: despite the title it studies
training-time robustness, not noisy-val selection — the mis-ranking claim rides on Northcutt*) ·
Cole et al., single-positive multi-label (EPR formulation) —
[arXiv:2106.09708](https://arxiv.org/abs/2106.09708) · Cheng & Vasconcelos, CVPR 2024 (asymmetric
losses are not strictly proper — §8.4) · SmoothECE, Błasiok 2023 · Dual-TS —
[arXiv:2308.08366](https://arxiv.org/abs/2308.08366) · Label Wave, ICLR 2024 —
[arXiv:2502.07551](https://arxiv.org/abs/2502.07551) · Thresholding for F1 —
[arXiv:1402.1892](https://arxiv.org/abs/1402.1892) · Niculescu-Mizil & Caruana, ICML 2005 · statAP
(Yilmaz & Aslam SIGIR 2008) / infAP (Aslam & Yilmaz CIKM 2006) · Sakai, SIGIR 2006 (paired
bootstrap)

**Augmentation.** LogicMix — [arXiv:2405.15860](https://arxiv.org/abs/2405.15860) · SpliceMix —
[arXiv:2311.15200](https://arxiv.org/abs/2311.15200) · BalanceMix —
[arXiv:2312.07087](https://arxiv.org/abs/2312.07087) · Kirichenko et al., NeurIPS 2023 —
[arXiv:2401.01764](https://arxiv.org/abs/2401.01764) · Planckian Jitter —
[arXiv:2202.07993](https://arxiv.org/abs/2202.07993)

**On-disk evidence.** `experiments/run1_vit/checkpoints/*.pt`, `pr_threshold_*.json` ·
`logs/training.log` · `tensorboard/**/events.out.tfevents.*` · `vocabulary.json` ·
`L:\Dab\DataCleaning Project\corrections_report.json` · `configs/unified_config.yaml`
