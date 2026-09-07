<!-- ARCHIVED SNAPSHOT 2026-07-30 — the pre-rewrite working-tree state of todos/v2-plan.md,
     preserved so the 2026-07-30 verification-pass audit trail (never committed) survives in git
     history. Reconstructed at archive time from the same session that performed the rewrite.
     Superseded by the clean rewrite of todos/v2-plan.md the same day. DO NOT EDIT, DO NOT CITE
     as current — every decision here is restated (or deliberately dropped) in the live plan. -->

# V2 Plan — the authoritative document

> **Doc set, and what each one is for now:**
>
> | Doc | Role |
> |---|---|
> | **`v2-plan.md`** (this file) | **Decisions + the runnable per-phase config. The only doc that governs the run.** Self-contained as of 2026-07-29 — no other doc is needed to execute it. |
> | [`v2-plan-review-2026-07-28.md`](v2-plan-review-2026-07-28.md) | Frozen evidence appendix. Every number here is measured or cited there. Its code line refs are as-of 2026-07-28 and have drifted; where it and this file disagree, **this file wins**. |
> | `ordinal-fusion-track.md` | **Retired to git history 2026-07-30** (`2c1cfa1`; file deleted from the tree). Living summary of the DINOv2 ordinal-fusion / medium-recovery track: janitor plan §7.1. Not on V2's critical path. |
> | [`v2-monitoring.md`](v2-monitoring.md) | Canary checks, decision rules and the TensorBoard reduction procedure, relocated out of `TRAINING_HEALTH_TRACKER.md`. |
> | `TRAINING_HEALTH_TRACKER.md` | Reduced to a pure V1-P2 per-epoch run archive — data, no decisions. |
> | `progressive-training-plan.md` · `ASL_plan.md` · `v2-plan-correction-2026-07-28.md` | **Retired 2026-07-29** (full text in git history). Decisions superseded by this file; still-live content was inlined first: per-phase augmentation → §7, telemetry spec → Appendix A, scope fence → §4. |
>
> **Rule for future edits:** amend *this* file. Do not add another dated correction layer — the
> supersession chain is what let stale summaries revert settled decisions twice already.
>
> ---
>
> **Verification pass 2026-07-30 — corrections applied inline, per the rule above.** All 52 arXiv
> citations, every `file:line` reference, and every quantitative claim were independently checked and
> then adversarially judged. **The central thesis survived** (§1: V1 never early-stopped, mAP rose
> monotonically, ~230M samples ≈ 60% of DeiT-B, zero micro-F1 gap) and so did most of §3, §4 and §11.
> **What did not, in order of consequence:** the corpus size and every figure normalized to it (§10,
> §6); the existence of a shared held-out eval set (§9.3, §8.2); the third refutation's independence
> (§1); the sibling-gap arbiter's sign (§9.2); the propensity rejection (§11); the drop_path,
> sigmoid-bottleneck, Illustration2Vec, xCOLUMNs and Scaling-ViT-augmentation citations.
> **Load-bearing claims that are still UNVERIFIED and are flagged as such in place:** §3's attention
> percentages, §8.2's simulated CIs and the −0.0031 binning bias, the `ap_thresholds` memory/time
> figures, and Gate 3's statistical power. Nothing was run on a GPU, so no throughput, MFU or memory
> figure in this document is measured.
>
> **DECISIONS 2026-07-30 (user, after the external review + literature sweep):** §3 sizing =
> **option C** (896×18, MLP 2320, ~150M) · §5 init = **from-scratch** (closed; probe retired) ·
> schedule = **WSD** (constant LR per phase + 1-sqrt cooldown, replacing cosine — §6) · loss =
> **plain ASL γ⁻=7 fixed with the focal weight DETACHED**, **single configuration — no experiment
> arms of any kind** (§4) · the janitor keeps
> its **fresh from-scratch Stage A** (its plan §2). Plans updated same day; **every implied
> code/config change is still UNAPPLIED** — see §12 item 3. More research precedes launch; nothing
> in this document starts a run yet.
>
> **Created 2026-07-28**, consolidating the four-layer plan set after the independent review.

---

## 1. What changed, and why the plan needed rewriting

Two load-bearing beliefs turned out to be unmeasured:

**The label-noise ceiling was never observed.** V1 did not early-stop — `grep -c "Early stopping
triggered" logs/training.log` returns **0**. Both phases ended with a manual `Soft stop engaged`
mid-epoch. Phase 1 stopped at **33/40** epochs at mAP **0.6518**, having run **81.76%** of its cosine
cycle (209,899 / 256,720 steps), with mAP climbing +0.0065/epoch; Phase 2 at **6/15** at mAP
**0.6744**, **41.81%** of cycle (85,517 / 204,555), LR at **77.5%** of peak (`last.pt`
`scheduler_state_dict`, max_lr 1.2990e-5) with 58% of the cosine unspent. mAP and val loss improved
*monotonically at every logged validation* — all 19 P1 and all 6 P2 events in the tfevents files —
and the val draw is deterministic (`shuffle=False`, no flip, cached list), so the same 30,000 images
and the same 958 zero-support tags recur every epoch: the slope is **not** denominator drift.
Naive linear extrapolation over Phase 1's unrun epochs is **+0.041 mAP** (= 6.3 × 0.0065; the full
7 epochs would be +0.0455).

*(Corrected 2026-07-30. The LR at P1's terminal step 209,899 is **10.0%** of peak, not the 14%
previously stated, and P2's is 77.5%, not 76% — both errors were in the conservative direction.
**Withdrawn as unsupported:** "cosine schedules deliver a disproportionate share of their final gain
in the anneal tail." It carried no citation, none was found, and it was doing real work as an addend
on top of the +0.041. Treat the tail bonus as unquantified.)*

And there is **zero generalization gap**: micro-F1 **0.70044** on 296K images (~90% of which were
folded into training) vs **0.70088** on the 30K genuinely held out. A 248M ViT memorized nothing in
39 epochs.

**A third refutation — real, but weaker than this section previously claimed (corrected 2026-07-30).**
Per-tag macro F1 moved **0.6218 → 0.6751** from P1-e27 to P2-e7. Three defects, all verified:
- Both endpoints are `per_tag.f1_summary.mean_f1` on the **same 296K artifact §8.2 forbids quoting as
  held-out** (`New folder/pr_threshold_last.json` and `pr_threshold_e7_step85517_full296k.json`).
- Both are per-tag **oracle**-thresholded on that same data (median θ 0.641 → 0.773), which §8.3 now
  flags as a fit-on-eval statistic.
- The interval contains a **γ_neg 4 → 7 change** (`logs/training.log:12828` vs `:12945`), the
  320→448 switch, and an optimizer/scheduler reset (max_lr 4.73e-4 → 1.299e-5) as well as ~13
  epochs — so it is **not** a clean test of "more training", and citing it against the *loss*-first
  workstream is self-defeating.

**What survives:** the de-oracled single-threshold pair **0.5591 → 0.6212 (+11.1%)**, both on 296K,
preserves the direction and the magnitude. A *hard* ceiling at 0.6218 is still refuted. The claims of
**independence** and of **held-out measurement** are withdrawn — this shares the mAP-slope argument's
eval set and its noisy-label assumption.

Three V2 decisions rested on that ceiling — the 248M→192M cut, the loss-first workstream, and
de-scoping label cleaning. All three were re-opened, and all three are now **closed on new grounds
(2026-07-30):** the cut lands at option C for throughput reasons that do not need the ceiling (§3);
the loss is fixed ASL, detached, with no descent and no experiment arms (§4); cleaning is the janitor
programme, whose fresh Stage A is confirmed.

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
| **Backbone** | **896w × 18L, patch16, MLP 2320, head_dim 64 (14 heads), CLS-pool — DECIDED 2026-07-30 (§3, option C)** |
| **Params** | **~150M** (SoViT-150m's fitted shape) |
| **Init** | **From-scratch — DECIDED 2026-07-30 (§5 closed)** |
| **Data** | **~6.2M images (operative planning figure)** — on-disk count at `L:/Dab/Dab` measured 2026-07-30 is **5,921,102**; reconcile before the run (**§10**). 19,294 tags, positive rate ~0.18%, neg:pos ≈ 530–640:1 |
| **Resolution** | 320 → 448 → **512 (unconditional, not gated)** |
| **Loss** | **ASL**, γ_neg = 7.0 fixed, γ_pos = 0, clip = **0.05**, α = 1.0, label_smoothing = 0, **focal weight DETACHED** — per **ASL Appendix F** + RAM; **single configuration, no experiment arms** (§4) |
| **Schedule** | **WSD per phase: warmup → constant LR → plateau gate on the EMA weights → 1-sqrt cooldown of ~10–20% of elapsed steps** (§6). Replaces cosine; epoch counts are planning estimates, not commitments. |
| **Budget** | **~400–700M samples seen** as the planning envelope (V1 got ~230M); the gate + cooldown decide the actual |
| **New this run** | Weight EMA · LayerScale · plain dropout → 0 · fp32 optimizer state on the head · bf16/no-GradScaler recorded as policy (§6) |
| **Selection** | `val/mAP` over **all** tags, frozen list, on a **≥276K** val set; binned at `ap_thresholds: 200` (bias −0.0031, near-uniform); resolution floor **3e-3** |
| **Decision procedure** | Three-gate protocol, §9. Gate 3 (bias-controlled slice) is the actual arbiter. |

---

## 3. Architecture — DECIDED 2026-07-30: option C (896×18, MLP 2320, ~150M)

**The decision.** Smaller-than-V1 was set as a requirement (excluding A — keep 1024×18 / 248M), and
between the two cuts **C is the one with a citation**: SoViT-150m's fitted exponents put
s_MLP ≈ 0.60 > s_depth ≈ 0.45 > s_width ≈ 0.22 — at this compute scale the MLP should be
*relatively narrow*, and C **is** SoViT-150m's shape. B (896×18, MLP 3584, 192M) kept the
conventional 4× MLP ratio, a shape no paper has trained, citing SoViT for the cut while ignoring
its finding — **rejected**. C is ~22% less compute per image than B (≈ a week of Phase 1 on the
5090, or ~28% more epochs at equal wall-clock — and §6's epoch budget is this plan's
best-evidenced lever).

*Context kept for the record:* the original cut's premise ("the ~250M model carried unusable
headroom") was never tested (§1), and DeiT III's from-scratch ViT-L benchmark took ~1.02B sample
presentations — >2× §6's ~418M planning figure. The parameter comparison favours us; the compute
comparison does not. WSD (§6) is what closes that gap now: the epoch count is no longer
pre-committed.

**Capacity note (2026-07-30).** C does not meaningfully lower the image cap: the 19K head
(896×19,294 ≈ 17.3M params, ~11.5% of the model) is identical to B's; an 86M ViT-B is still
unsaturated between 30M and 300M JFT images; and V1 showed zero train/val gap at 248M. Useful
runway ≈ **8–30M images** — the ~12M same-taxonomy expansion fits inside it, and the binding
constraint stays label quality (10× equally-noisy images ≈ +1%, Zhai et al.).

**Forward-compat (carried from the retired progressive plan §2.2):** if the corpus grows, grow
**depth-first at 896 width** (896×18 → 896×24 → only then widen toward 1024), gating re-growth on
effective *clean-label* signal, not raw image count. The old triggers assumed B's shape; for C they
sit ~20% earlier — first re-growth review at ~**8M clean-label images**.

**Settled, do not revisit:**
- **patch16.** patch14 buys +31% tokens for ~1.36× FLOPs, forces a stem re-init, and breaks the 320
  and 512 grids. Attention stays a minority of compute at every planned resolution — analytic share
  at the decided C shape (d=896, MLP 2320): **~8.9% at 320, ~16.0% at 448, ~19.9% at 512**
  *(re-derived 2026-07-30 for MLP 2320; the earlier 6.94/12.74/16.01% were MLP-3584 figures, and the
  once-quoted "measured 7.5/14.6/19.1%" matched no model and no profiler output on disk)*.
  `512px/patch16` gives N=1025 — *identical token cost to 448px/patch14* — with real pixels instead
  of a finer mesh over the same downscaled raster.
- **The linear head stays — on the compute argument below.** Its two former citations are withdrawn
  (2026-07-30): *Taming the Sigmoid Bottleneck* ([arXiv:2310.10443](https://arxiv.org/abs/2310.10443))
  was cited inverted — its actual headline (d ≪ n has exponentially many *unargmaxable* label
  combinations) argues the head *could* bind at d=896 / n=19,294 — and Generalized Neural Collapse
  ([arXiv:2310.05351](https://arxiv.org/abs/2310.05351)) is single-label CE with no multi-label
  claim. The decision survives on Query2Label/ML-Decoder economics alone; if head capacity ever
  looks binding, the sigmoid-bottleneck paper is a reason to look harder.
- **Query2Label is impossible here** — one query per class at 19,294 classes is ~1.33 TFLOPs/layer
  (~4.3× the whole backbone) and ~10 GB attention memory per image. ML-Decoder measured a full
  decoder head **OOM at 9,600** classes.

**Architecture adds (all new, all cheap):**

| Add | Setting | Evidence |
|---|---|---|
| **LayerScale** | ε = **0.1** | CaiT Table 1 measures **+1.0 top-1 at exactly depth 18** (80.7 → 81.7) vs a drop-path-tuned baseline. DeiT III uses it in every config. Absent from the codebase. *(The old warm-start incompatibility note is moot — §3 decided C from scratch; no warm start exists.)* |
| **QK-norm** | **lean ADOPT — final call before P1** | Updated 2026-07-30 (review): a modern supervised-ViT recipe measures QK-norm default-ON at 87–449M (removing it costs ~0.12 top-1; "significantly enhances training stability"), and Wortsman's divergence threshold is **LR-dependent, not scale-gated**, so scale buys no safety margin. Near-free FLOPs; one mid-run divergence on a weeks-long single-GPU run costs more than the norm ever will. Whatever the call, log max attention logits from epoch 1 (Wortsman: *"all points with attention logits above 1e4 diverged"*). |
| **2D RoPE** | optional, P1 only if adopted | RoPE-ViT (ECCV 2024): +1.4 @384, +2.5 @512 for ViT-L vs interpolated learned pos-embeds, at **0.01% of FLOPs**. *(Regime caveat added 2026-07-30: those gains are **zero-shot resolution extrapolation** on models never fine-tuned at the target resolution — i.e. exactly the degradation §6's recipe already repairs by re-warming and fine-tuning at each rung. Expect materially less than +1.4/+2.5 here.)* Changes the ONNX path — prototype before committing. |

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
    detach_focal_weight: true   # DECIDED 2026-07-30 — new key, see the detach paragraph below. NOT YET IMPLEMENTED in loss_functions.py.
```

**The citation is ASL Appendix F, not P-ASL.** Ridnik et al.
([arXiv:2009.14119](https://arxiv.org/abs/2009.14119)): *"we set all untagged labels as negative…
Since the level of positive-negative imbalancing is significantly higher than MS-COCO, we increased
the level of loss asymmetry: For ASL, we trained with γ⁻ = 7, γ⁺ = 0."* That is assume-negative,
~5,400–9,600 classes, extreme imbalance — structurally our regime, and it pairs γ⁻=7 with **γ⁺=0**
and margin **m = 0.05**. *(Quote nuance: the elided clause is "with reduced weights" — Appendix F
also down-weights the assumed negatives, a knob this plan does not currently replicate.)*

**Corroborated — but by one strong source, not three (corrected 2026-07-30).**
- **RAM** (CVPRW 2024, [arXiv:2306.03514](https://arxiv.org/abs/2306.03514)) tags **6,449
  categories** from incomplete image-text labels using **plain ASL at γ⁻=7, γ⁺=0, clip=0.05** —
  bit-for-bit this config (verified in the official RAM code; the paper text omits the
  hyperparameters). **This is the real corroboration and the closest published regime match.**
- ~~**ML-Decoder's** OpenImages loss table (CE 84.8 · Focal 84.9 · **ASL 86.3**)~~ — the numbers are
  right, but this is **ASL's own Table 7 re-quoted by the same lab**. It is not an independent
  replication and must not be counted as one.
- ~~**SINR** at 47,000 positive-only classes~~ — SINR ([arXiv:2306.02564](https://arxiv.org/abs/2306.02564))
  is a **lat/lon MLP for species range mapping with no images**, and it **mentions no ASL, focal,
  Hill or SPLC**. It supports the narrow claim *"assume-negative training stays viable at 47K
  labels"* and **nothing about how these particular losses behave** as C grows. Cite it for the
  former only.

**clip 0.2 → 0.05.** This lands where the correction doc put it, for a sounder reason: it is ASL's
own assume-negative value at our class count, not a consequence of an ignore set we are no longer
building. Frozen for the whole run; never moved in the same step as anything else.

**Focal-weight detach — DECIDED 2026-07-30: detach at V2 start (re-adjudicated same day,
adversarial panel: KEPT, unanimous).** The repo currently does **not** detach
(`loss_functions.py:317-329`). Provenance corrected 2026-07-30: cite detach to **RAM's released
code only** (`utils.py` defaults `disable_torch_grad_focal_loss=True`, never overridden). Appendix
F is **indeterminate** — no OpenImages training script was ever released, both ASL classes
defaulted *non*-detached until 2021-03, and the author's issue-#31 comment treats non-detached as
the paper's form. Magnitude corrected: non-detach multiplies the negative gradient by
1 + γ(1−p+m)(−ln(1−p+m))/(p−m) — monotone *decreasing*, **7.8× at p=0.1 → 3.3× at p=0.9** (verified
against the repo loss via autograd), not "~2–5× in a band". Detach wins on **both** constraints
(measured): it raises missing-label tolerance (a truly-present tag at p=0.7 survives up to ~91%
unlabeled occurrences vs ~67% non-detached; the confusable band is 23–64%) *and* keeps more ρ-wise
boundary separation (93% of V1-P1's logit spread vs 83%); the lower non-detached firing level is
absorbed by threshold re-derivation and is not a real advantage. V2 is a fresh run; this is not a
mid-run flip. **Consequences:** (a) the
code change in `loss_functions.py` is required before any V2/janitor run — **NOT yet applied**
(plans-only day); (b) V1's grad-norm baselines do not transfer — re-derive the monitoring bands on
V2's own first epochs (v2-monitoring.md already mandates this); (c) "never flip detach mid-run"
stands.

**P-ASL selective ignore is NOT adopted.**
- **Ω_P is dropped outright.** At η=0.05 it selects **107 of 19,288 tags** (0.55% of vocab, **42.4%
  of label mass**) whose mean estimated missingness is **0.38%** — the tags needing it least — and at
  N=∅ they would receive no negative gradient from any source and collapse to always-on.
- **Ω_L is not adopted this run.** It is an unchecked confirmation loop without a verified-negative
  anchor, and it inherits the exact objection used to reject SPLC (*"assumes a calibrated backbone,
  unsafe in from-scratch Phase 1"*). If revisited: it needs a `change_epoch` delay, and **K must be
  calibrated on the gold slice, not on val** — §9 explains why noisy val drives K too small.
- The **full-corpus prior-estimation pass is deleted.** It was for Ω_P, and where Ω_P was active
  empirical tag frequency is already within ~0.5% of the true prior.

*Dissent recorded:* a scale-scoped sweep argued P-ASL should stay primary (nothing published beats
it above 1,203 classes). Argument from absence — half its gain needs verified negatives we don't
have, its prior estimator is degenerate without them, and RAM's 6,449-tag plain-ASL result is the
closer regime match. Rejection stands.

**Fallbacks, not experiments (user decision 2026-07-30: V2 trains a single loss configuration —
nothing runs alongside it or against it).** The loss is plain ASL as specified above, full stop.
Kept for the record: on OpenImages 567 classes with no verified negatives, BCE 60.83 · Focal
62.14 · ASL 61.95 · Hill 62.71 · SPLC 62.86
([arXiv:2112.07368](https://arxiv.org/abs/2112.07368) Table V, cells verified — an r=1
single-positive reconstruction with ASL at author defaults, not γ⁻=7, so it cannot predict our
margin). **Hill is the documented first fallback**, invoked only if the §9 gates fail the shipped
loss: it does Ω_L's job with no confirmation loop, no prior, no schedule. The 2023–2026 sweep
(review, 2026-07-30) found nothing that beats ASL/Hill/SPLC in this regime; LL-R/LL-Ct — the only
≥5K-class-validated missing-label mechanism — is recorded in §11 as the fallback after Hill.

**Scope fence (unchanged; carried from the retired ASL_plan §7).** No negative-branch knob touches **wrong-positive**
noise: a mislabeled `aqua_hair=1` trains the wrong side of the boundary at full BCE gradient
regardless of γ_neg, clip, or any ignore set. That is the cleaning track's job, permanently.

**Telemetry, not control.** `ASLDriveManager` → telemetry-only. Keep its per-decile EPR, sibling-gap
and non-GT histogram outputs (full observable definitions + measurement hygiene: **Appendix A**);
remove its authority (its `set_gamma_neg` calls are
`asl_telemetry.py:190` and `:248`, plus a pass-through delegation at `loss_functions.py:473`).
**Invert the precedence at `asl_telemetry.py:136-158`** *(line ref corrected 2026-07-30 from
`:146-160`)* — a checkpoint's persisted γ_neg currently *wins over YAML* with only a warning, which
under "γ_neg fixed" silently overrides config on resume.

> **⚠ This action list is incomplete — three live config sites keep the γ_neg ladder alive
> (added 2026-07-30).** Neutralize all of them, or "γ_neg = 7.0 FIXED, no descent, no dwell" is
> aspirational rather than true:
> - **`training.tag_loss.asl_schedule` (`unified_config.yaml:421-428`) is still live**, with
>   `phase1: {gamma_neg_min: 5.0, gamma_neg_max: 7.0, hold_epochs: 8}` and phase2/phase3 bands down
>   to 5.0. A band with a minimum of 5.0 is not a fixed γ of 7.0.
> - **`gamma_neg_override: null` (`:384`)** — the documented manual-step hatch. Keep it null and say
>   so, or remove it.
> - **`unified_config.yaml:352-354` documents checkpoint-wins-over-YAML as *intended* behaviour.**
>   So the "precedence inversion" above is not a bug the code is unaware of; it is a decision
>   recorded in the config that this plan reverses. Update the comment in the same commit, or the
>   next reader will restore it — which is exactly the stale-artifact failure this document's header
>   warns about.
> *(Note the config is currently **more** disciplined than this plan on one point: `clip: 0.2` at
> `:404` carries an explicit `NOT YET APPLIED … pending the §8 instrument work` comment. That is the
> right call and §12 item 4 now depends on it.)*

---

## 5. Initialization — DECIDED 2026-07-30: from-scratch (closed)

User decision, 2026-07-30. The linear-probe tie-breaker is **retired un-run**, and the pretrained
routes (LiGO growth · patient distillation · adopt-ViT-L-outright · in-domain MAE/DINO SSL) are
**closed** — rejection rows in §11, full text in git history.

Kept for the record, so the evidence state is not re-litigated from stale premises:
- **The regime is unmeasured in both directions.** No published controlled init comparison exists
  at ≥5M target images / 10K+ tags for ViT classification. Steiner's +13pp for IN-21k pretraining
  (87.08 vs 74.01) is at **1.28M target images** — the steep part of the data curve — and
  overstates the pretraining case at 6M; the strongest for-from-scratch evidence (Zoph et al.,
  [arXiv:2006.06882](https://arxiv.org/abs/2006.06882): **−1.0 AP negative transfer** at full COCO
  with strong augmentation) is detection-only.
- **Two evidence corrections from the 2026-07-30 review.** The old "no scholarly evaluation of
  foundation-model features on booru-style illustration exists" claim was **false**: Chen & Zwicker
  (WACV 2022, [arXiv:2108.01819](https://arxiv.org/abs/2108.01819)) built a Danbooru-tagger
  backbone that beat ImageNet init in-domain (0.8827 vs 0.8218 OKS@50), and DAF:re
  ([arXiv:2101.08674](https://arxiv.org/abs/2101.08674)) measured ViT-L pretrained 85.95% vs
  scratch 59.39% (compute-starved from-scratch arm — upper bound, not a fair-budget number). And
  **Illustration2Vec is frozen-feature evidence, not init evidence** — its 32.2-vs-10.1 compares
  from-scratch *training* against *frozen* ImageNet-VGG features, so it argues against probes, not
  against pretrained init + full fine-tune. Neither correction re-opens the decision; both stop bad
  citations from recurring.
- **Standing consequence (unchanged):** every prediction-dependent mechanism in the literature
  (P-ASL, Ω_L, SPLC self-relabel) was measured with a pretrained backbone doing the calibrating —
  from-scratch discounts that evidence base by an unestimated amount. All such mechanisms are
  rejected in §4/§11 anyway; remember this if any is ever revisited.
- **Compute anchor at the decided shape:** Phase 1 ≈ **1.3e20 FLOPs at option C** — multiple weeks
  on one GPU (§12 wall-clock gap). WSD (§6) is the mitigation: no pre-committed epoch count.

---

## 6. Schedule — finish training this time (WSD, decided 2026-07-30)

V1 received **~230M sample presentations** for a 248M from-scratch ViT — about **60%** of what
DeiT-B (2.9× smaller) uses. That, not label noise, is the measured root failure.

**Target: 400–700M samples seen — as a planning envelope, not a commitment (see WSD below).**

**The schedule is WSD (warmup–stable–decay), replacing cosine. DECIDED 2026-07-30.** Per phase:
linear warmup → **constant LR** through the stable body → plateau gate evaluated on the **EMA
weights'** `val/mAP` (raw constant-LR weights understate annealed performance) → a **1-sqrt
cooldown of ~10–20% of that phase's elapsed steps** → the phase transition. Why: both V1 phases
were hand-stopped mid-cosine (P1 at 82% of cycle, P2 at 42%, LR still at 10% / 77.5% of peak),
forfeiting the anneal both times. Under WSD the anneal happens **after** the stop decision — a hand
stop costs ~10–20% extra steps instead of the tail — and a run that wants to go *longer* than
planned just keeps going at constant LR (under cosine the LR has already collapsed). Evidence:
constant+cooldown matches cosine at equal compute with cooldowns of 10–20% of steps (Hägele et al.,
NeurIPS 2024; Tissue et al. 2024; MiniCPM: 10% suffices, 2.5% falls short) — LLM-scale parity, with
the supervised-ViT precedent being Zhai et al.'s rsqrt+cooldown "infinite schedule", built for
exactly this stop-when-ready property. Two riders: (1) mid-run EMA narrows but does **not** replace
the cooldown (Hägele) — always run the decay; (2) a cooldown can be **branched** off any stable
checkpoint for an annealed readout while the stable run continues — this is also the janitor's
Stage-A branch mechanism (branch *after* a cooldown, never the raw stable checkpoint).
**Scheduler implementation: NOT yet built** (§12 item 3).

**Normalization — corrected 2026-07-30, and it moves everything below.** An epoch is the **training
pool**, not the corpus: `dataset_loader.py:2780-2794` carves val out first. The old
"1 epoch = 6.8M samples = 6,641 steps" was wrong twice — it used the void 6.8M corpus figure (§10)
*and* it counted the held-out split as trainable.

| | operative (6.2M) | measured on disk (5,921,102) |
|---|---|---|
| holdout @5% | 310,000 | 296,056 |
| **training pool** | **5,890,000** | **5,625,046** |
| **1 epoch @ eff batch 1024** | **5,752 steps** | **5,493 steps** |

The old 6,641 is **15–21% high**. Every sample figure in the table is restated:

| Phase | Res | Epochs (est. — gate decides) | Samples (6.2M) | Samples (measured) | Eff. batch | Warmup | Base → stable LR | drop_path | dropout | WD |
|---|---|---|---|---|---|---|---|---|---|---|
| **1** from-scratch | **320** | **~55; gate on plateau** | 324M | 309M | ~1024 | **10K steps** | 2.7e-4 → ~5.4e-4 | **0.25** | **0.0 / 0.0** | 0.05 |
| **2** fine-tune | **448** | ~12; gate on plateau | 71M | 68M | ~768 | 2 ep | 1.5e-5 → ~2.6e-5 | 0.15 | 0.0 / 0.0 | 0.05 |
| **3** detail | **512** | 3–4 — **run it, don't gate it** | 24M | 23M | ~676 | 1 ep | 8e-6 → ~1.3e-5 | 0.10 | 0.0 / 0.0 | 0.05 |

**≈418M samples total at 6.2M (≈399M measured) — 1.8× V1, not the 2.1× previously claimed.**

**Two notes on the numbers above:**
1. **The epoch counts are estimates, not caps — the old "cap 60 → 66 → 78?" arithmetic is retired
   by WSD.** It existed because cosine forces the horizon at launch. Now the stable phase runs
   until the plateau gate fires, the cooldown adds ~10–20%, and the 400–700M band is a planning
   envelope for wall-clock/storage, not a scheduler input.
2. **Phase 3's effective batch is filled in at ~676**, which was blank. It is over-determined by this
   plan's own sqrt rule: 8e-6 → 1.3e-5 implies √(B/256) = 1.625, so B ≈ 676.

**The 320→448 spine has direct in-regime precedent:** **ASL itself trains Open Images (~5,400
classes, partial labels) at 224 for 30 epochs, then fine-tunes at 448** — same loss family, same
assume-negative annotation regime, same two-stage low-res-then-high-res shape. This is the closest
published precedent for the whole progressive-resolution decision and it was previously uncited here.

**Phase-transition checklist (every step, in order):** select best P1 checkpoint on the §9 metric →
interpolate pos-embeds bicubic 20×20 → 28×28 (`training_utils.py:1990-2047`, implemented; grid
sizes are derived, not hardcoded) → **reset
optimizer state** → update config → re-warmup. `torch.compile` recompiles on first forward
(401 → 785 tokens).

**Notes on the numbers:**
- **drop_path — the justification was inverted, and 0.25 is a raise, not a hold (corrected
  2026-07-30).**
  - **The convention claim is backwards.** `model_architecture.py:398` does use
    `torch.linspace(0.0, rate, num_hidden_layers)` (timm linear ramp → mean 0.125 across 18 blocks);
    that half is right. But **DeiT III does not use that convention.** Its §3.1 reads verbatim:
    *"We use a **uniform** drop rate across all layers"*, confirmed in `facebookresearch/deit`
    `models_v2.py` (`dpr = [drop_path_rate for i in range(depth)]`). So its ViT-B 0.1 / ViT-L 0.4 are
    **means**, and our mean of 0.125 sits barely above its **ViT-B** figure for a model 2.2× ViT-B.
    It does **not** "sit correctly between them", and the error points toward
    **under**-regularization on a multi-week phase. Matching DeiT III's ViT-L mean under a linear
    ramp would need `rate = 0.8`.
  - **The status quo it claims to protect does not exist.** `unified_config.yaml:44` is
    `drop_path_rate: 0.20`, and `last.pt` records 0.2 for the real V1-P2 run. "0.25 stays. Do not
    lower it" is a **+25% raise**. State it as a raise and justify it, or set 0.20.

  The old instruction ("verify the convention before ever changing this number") was carried out for
  our code and **not** for the cited table. Do both.
- **Plain dropout → 0.** *(Support corrected 2026-07-30: big_vision's plain-ViT baseline uses
  **neither** dropout **nor** stochastic depth — Beyer et al. verbatim — so it is not a second vote
  for "stochastic depth only". The conclusion now rests on **DeiT III alone**, plus Steiner.)*
  Steiner et al.: *"when using the 10× larger ImageNet-21k dataset and keeping compute fixed, any
  kind of AugReg hurts performance for all but the largest models."* At ~6.2M images we are near
  that regime — though note the quote contrasts 14M IN-21k against 1.3M IN-1k, and 6.2M is below
  both, so the extrapolation is directional rather than exact.
  Current: `hidden_dropout_prob: 0.10`, `attention_dropout: 0.05`.
- **WD 0.05 fixed**, per the standing project decision — overriding the old plan's 0.08/0.04.
  (V1 checkpoint record, from the retired correction doc §0.1: P1 ran WD **0.1**, P2 **0.05**.)
  *Documented tension:* DeiT III's long-schedule rule raises WD and drop_path when extending
  training (+0.05 drop_path per 200 epochs). If P1 runs materially past 55 epochs, revisit.
- **Phase-2 LR is not the failure mode.** DeiT III's high-res fine-tune is AdamW 1e-5 @ batch 512 ×
  20 epochs; 2.6e-5 @ 768 is higher in both LR and step count. V1's 1e-5 was also inside DeiT III's
  range. Rule this hypothesis out explicitly.
- **Warmup 10K steps — a floor, not a target.** V1-P1's 4-epoch warmup was 25,672–27,276 optimizer
  updates, so 10K is a 61% cut in updates (the old "longer warmup" framing was wrong and is gone
  from §2). Beyer et al. use 10K at batch 1024, so the number is defensible — but the σReparam grid
  (7 of 8 divergent configs in our exact box: ViT-B / batch 1024–2048 / LR 5e-4–1e-3) argues for
  more, and §3's QK-norm adoption lean is the other half of that insurance. Under WSD the warmup
  hands off directly into the constant stable LR.
- **The batch→LR rule (carried from the retired progressive plan §3.5):** **sqrt** scaling for
  AdamW — stable LR = base × √(eff batch / 256) (Malladi et al.,
  [arXiv:2205.10287](https://arxiv.org/abs/2205.10287)); linear scaling is SGD-only. **Do NOT apply
  the rule's β-rescaling** (β₁′ = 1 − κ(1−β₁) → β₁ ≈ 0.6 at κ≈4): that clause is needed only for
  exact SDE-dynamics matching, and no strong vision recipe (ViT, DeiT, DeiT-III, AugReg,
  big_vision) rescales betas across batch 256–4096 *(settled 2026-07-30 — the previous "currently
  unapplied" framing read as a to-do)*. ε → ε/√κ is harmless and consistent with AdamW8bit's small
  ε. Re-apply the LR half whenever eff batch is retuned to real VRAM; keep β = (0.9, 0.95–0.999)
  fixed.
- **The base-LR question this section left open is ANSWERED on disk, in the plan's favour (resolved
  2026-07-30).** `learning_rate` is unambiguously a **/256 base, sqrt-scaled at runtime**:
  `unified_config.yaml:244-259`, `train_direct.py:768-777`, and the log's own
  *"base 2.50e-04 × sqrt(918/256) = 4.73e-04"*. So the "~8.2e-4 / 35% below our own anchor" branch
  **cannot obtain** and needs no pre-run check. The stated **1.14× is correct** as a peak-to-peak
  ratio (5.4e-4 / 4.73e-4 = 1.14, which equals 1.08 × √(1024/918)). **LR guard retained:** grad_norm
  drift >2× in the first 5 epochs → fall back to base 2.5e-4 — though see the abort-criterion gap
  below, since 5 epochs covers under 10% of Phase 1.
- **Phase 3 is no longer gated.** ASL COCO 448→640 **+1.4 mAP**; ML-Decoder **+1.1**; Query2Label
  **+1.1** — unusually consistent across three multi-label papers, and it targets exactly the
  small-detail tags (hair ornaments, accessories). The old fine-bucket-plateau trigger would
  under-fire, because the same tail-metric noise that hides the plateau also hides the benefit.
  Sanity check **done 2026-07-29** (3,000-image random sample of `L:/Dab/Dab`): long side is
  **exactly 512 for 93.5%**, below 512 for **6.5%**, and **above 512 for 0%**; nothing is below 448.
  So the ladder terminates exactly at native — P2 (448) is a uniform gentle 1.14× downscale and P3
  (512) is a true native pass with no interpolation for 93.5% of the corpus. This is what the
  "real pixels instead of a finer mesh" argument above assumed, and it holds. There is no rung
  above 512: any P4 would be upscaling. Median short/long aspect is 0.713 (IQR 0.705–0.799), so the
  measured letterbox pad fraction is 0.287 — consistent with the 25–35% figure in §6.

**Considered at the resolution switches, not adopted this run (recorded so they are not re-litigated
from scratch):**
- **MHSA-only fine-tune** at the resolution switches — *Three things everyone should know about ViT*
  ([arXiv:2203.09795](https://arxiv.org/abs/2203.09795)): within **±0.1 mAP of full fine-tuning at
  −10% memory and time**. **Not adopted** — a full fine-tune is what every §6 precedent (DeiT III,
  ASL, FixRes-style) actually measured — but P2+P3 are **~40% of the budget by FLOPs** (not the
  ~23% by samples), so **keep it genuinely live as the fallback if P2/P3 memory binds**; the
  quality evidence says it is close to free.
- **Matching the serving preprocessor to training's two-stage resample** — training is
  original→512→448 (LANCZOS∘LANCZOS) vs serving's single original→448 LANCZOS. **Not adopted: the
  measured skew is too small to act on** — ~1.1/255 mean |Δ| on-corpus (~2.05/255 projected onto
  true originals), under the 3.69/255 bar of the already-fixed BILINEAR serving bug. Upstream
  kernel confirmed Pillow LANCZOS (`L:/Dab/Dab/resize_images.py:52`; the q95/4:4:4 signature comes
  from the `downsample_*.py` pair, and JPEG-container forensics ties 94% of corpus headers to it).
  If ever wanted, the exact fix is "at serving, if long side > 512: resample to 512 LANCZOS first,
  then to 448" — correct for 100% of cases — but it touches five serving sites plus the ONNX
  metadata and release manifest.
- **Aspect-bucketed batching / NaViT** ([arXiv:2307.06304](https://arxiv.org/abs/2307.06304)) —
  square letterboxing of typical portrait booru art spends **~25–35% of 784 tokens on gray bars**,
  so this is worth ~25% throughput and 0 to +0.3 mAP. **Not adopted:** it changes the letterbox
  preprocessing contract that the ONNX/inference path, the flip pipeline and the dedup artifacts all
  share, which is a broad change to make in the same run as three resolution switches. Revisit as a
  standalone throughput project.

**Optimizer:** AdamW8bit, **except fp32 optimizer state for the 19,294-way head and position
embeddings** (~140 MB via bitsandbytes `GlobalOptimManager`); verify block-wise *dynamic* (not
linear) quantization. 8-bit AdamW is not verified lossless for ViT-from-scratch —
[arXiv:2309.01507](https://arxiv.org/abs/2309.01507) Table 2 measures Swin-T IN-1k from scratch at
**81.0 (8-bit) vs 81.2 (fp32)**, exceeding seed std, and Dettmers' paper
([arXiv:2110.02861](https://arxiv.org/abs/2110.02861)) contains no ViT and no 8-bit-*Adam* vision
result at all — its ImageNet row is 8-bit momentum-SGD. A 19K head at 0.18% positive rate is
structurally the sparse embedding shape that motivated the Stable Embedding Layer.

**Do not adopt LAMB.** Its entire measured advantage is at batch ≥16K; ImageNet's critical batch
size is 1,000–15,000 (McCandlish et al., [arXiv:1812.06162](https://arxiv.org/abs/1812.06162) —
*citation added 2026-07-30; this figure was previously uncited*), so batch 1024 sits at the bottom of
the range where AdamW is not the bottleneck.

*Two caveats added 2026-07-30, neither fatal but both previously unstated:*
- **DeiT III itself trains with LAMB at batch 2048 for all three of its own columns.** This plan
  imports DeiT III's drop_path, LayerScale and dropout-free recipe while rejecting its optimizer.
  That may well be right at batch 1024, but the tension should be on the record rather than silent.
- **The Beyer 1024 → 4096 row (76.5 → 74.7) is confounded and is not clean batch-size evidence:** it
  is an absolute-step config, so warmup moves from 8.9% to 35.5% of training, and it is **ViT-S/16
  only**. Beyer never uses LAMB, so it is also not an AdamW-vs-LAMB comparison. Keep the conclusion,
  drop this row as its support.

**Precision — recorded as policy (2026-07-30):** **bf16 autocast, no GradScaler**, fp32 master
weights; compute the 19,294-way loss reduction in fp32 (bf16's 7 mantissa bits are the only real
cost, and a 19K-term per-sample sum is where it would show). The config already runs `amp_dtype:
bfloat16`; this paragraph exists so the choice is a decision, not an accident — fp16 loss-scale
drift is a documented divergence mode for large ViT runs, and dropping the scaler removes the
stale-scale-after-resume failure class on a resume-heavy phase structure.

**Weight EMA — add it; under WSD it is also the plateau-gate signal (2026-07-30).** α ≈ **0.9998**.
Evaluate the online *and* EMA weights at every validation and keep both checkpoints (downside
exactly zero) — and the §6 plateau gate reads the **EMA** curve, since mid-stable raw weights
understate annealed performance. The α is now positively supported rather than merely
self-consistent: the ~1%-of-budget averaging-horizon rule is vision-verified (AlgoPerf ViT rows,
[arXiv:2502.06761](https://arxiv.org/abs/2502.06761)), and 1/(1−α) = 5,000 steps ≈ 1% of a
~470–500K-step run; extend toward 0.9999 if the run materially exceeds that. Expected upside on
clean data ≈ +0.2 (same source). The once-quoted **+9 was a wrong-noise-mode CIFAR-100N figure —
do not budget for it.** EMA does **not** replace the cooldown (Hägele).

*(Standing non-transfer flag, unchanged: the noise evidence is CNN/single-label; the ViT evidence is
clean-label ImageNet where the gain is near-nil.)*
**Do not** add SWA on top (near-substitute, measured within noise) or pursue model soups (needs a
shared pretrained init).

---

## 7. Augmentation — unchanged, and defensible, but thinner evidence than claimed

Phase 1 at full strength; Phase 2/3 reduced (a *legitimate* reduction now that P1 actually
converges). Per-phase values, from the retired progressive plan §3.2/§3.4:

| Aug | Phase 1 (320) | Phase 2 (448) | Phase 3 (512) |
|---|---|---|---|
| horizontal flip + orientation-aware tag swap | p=0.5 | p=0.5 (lossless, unchanged) | as P2 |
| colour jitter brightness / contrast / saturation | 0.30 / 0.20 / 0.08 @ p=0.5 | 0.22 / 0.15 / 0.06 | as P2 |
| random rotation, bicubic (mask rotates too — see note) | ±[2°,8°] @ p=0.50 | ±[2°,5°] @ p=0.30 | as P2 |
| gaussian blur, kernel 3 | p=0.30, σ ∈ [0.1, 1.5] | p=0.15, σ ∈ [0.1, 1.0] | as P2 |
| mixup / cutmix / randaugment / random erasing / hue rotation | **none** | **none** | **none** |

*(The retired plan specced no separate P3 augmentation; "as P2" is this plan's reading of its "Phase 2/3 reduced" rule.)*

**Rotation now rotates the padding mask (fixed 2026-07-29, `dataset_loader.apply_random_rotation`).**
It previously rotated the canvas only, on the argument that the extra corner pad-fill was within the
letterbox distribution the model already ignores. That covered the wrong term: rotating the canvas
tilts the *whole* letterbox, so the larger error was real content rotating up into a region the
static mask still called a bar, which `token_ignore_threshold=0.9` then dropped from attention
entirely. Measured over 400 corpus images at the mean 3.5° angle: **1.83% of tokens
content-dropped vs 1.02% flat-gray-attended** (at 8°: 3.26% / 3.80%) — ~0.9% of token slots after a
p=0.30 gate, **or ~1.5% at Phase 1's actual p=0.50** *(the 0.9% figure was computed against the
Phase-2 rate while sitting under the Phase-1 row — corrected 2026-07-30; still too small to expect a
visible mAP delta)*. Too small to expect a visible mAP delta, fixed because it was an uncontrolled
confound sitting underneath the loss measurement this plan exists to make. Note it does **not**
recover content rotated off the non-expanded canvas (~13px per side at 5° for the median 0.713
aspect); that clipping is the intended semantics of rotate-in-place. Regression test:
`test_rotation_mask.py`. **Not adopted alongside it:** translation jitter (unmeasurable at the
current val-set size, and a confound on the ASL measurement).

**The exclusion is a defensible choice, not a well-supported one** *(downgraded from "the
better-supported choice" 2026-07-30 — three of the four supporting claims were overstated).*

Plain Mixup under partial labels measures **−12.9 mAP at 10% labels**
([arXiv:2405.15860](https://arxiv.org/abs/2405.15860) Table) — that part holds, and at low label
completeness it is decisive. *(But **"still negative at 90%" is false as a general claim**: at 90%
known labels the table reads −1.3 COCO / −0.7 VOC / **+0.2 VG-200** — and VG-200 is the
**highest-class-count** dataset in it, i.e. the row closest to us flips sign. Also note 2405.15860 is
an unrefereed preprint and is the sole quantitative support for this exclusion.)*

CutMix's area∝semantics assumption is undefined for global anime tags (`1girl`, `long_hair`, style
and meta tags). Hue rotation is excluded because hue is *categorical* for anime — even ~20° can turn
`blue_eyes` into `green_eyes` (carried from the retired overfitting assessment).

*(Two claims **withdrawn** 2026-07-30:*
- *"Label interpolation is justified only by softmax-CE's linearity in the target and directly
  contradicts sigmoid + asymmetric focusing" — **BCE and ASL are equally affine in the target**, so
  this argument does not distinguish them and proves nothing. The sound objection is **semantic**:
  under multi-label the union is the correct target, which is LogicMix's own argument.*
- *"Every large-scale ViT recipe at ≥14M images uses augmentation no richer than ours — Scaling
  ViT's entire JFT-3B pipeline is `inception_crop + flip_lr`." The pipeline string is right
  (Zhai et al. Appendix B), **but `inception_crop` IS a random-resized-crop**, and this repo has
  **zero** crop, scale or translate augmentation — verified, no `RandomResizedCrop` / `random_crop` /
  `inception_crop` anywhere in `*.py`. The two sets are **non-nested**, not ordered, so "no richer
  than ours" is not a statement that can be made. More decisively, **DeiT III's own ImageNet-21k
  (14M) column uses CutMix α=1.0 + ColorJitter 0.3 + 3-Augment** — a ≥14M recipe strictly richer
  than ours. **The open question this leaves is real: we have no scale augmentation at all on a plan
  whose spine is three resolution changes.**)*

*Honest counterweight:* the accurate claim is "expected gain is small and the interaction risk with
partial labels is large," not "augmentation cannot help at scale." Steiner's "hurts" result is at
fixed 30-epoch compute; augmentation recovers at 300 epochs.

**If a mixing method is ever wanted:** SpliceMix (but a 2×2 splice at 448 means each source is seen
at 224, which attacks precisely our resolution-sensitive confusables), or LogicMix — *gated on
building a per-image unknown-label mask*, without which it degenerates to plain hard union.
**BalanceMix — rejected for this run (single-configuration decision, 2026-07-30), with the record
corrected:** its noise modes *do* include missing positives and its largest ASL margins are exactly
there (COCO 73.3 → 77.4 — but at 80 classes, single-positive). Unproven above 80 classes, and it
needs a warm-up model whose confidence is exactly what's unreliable from scratch at 0.18% positive
rate. Not CutMix+LP (needs pixel maps for 19,294 tags).

---

## 8. LAUNCH BLOCKERS — the instrument

**Nothing below §8 is measurable until these clear.** The plan's *smaller* expected deltas are at or
below the current instrument's resolution.

*(Overstatement corrected 2026-07-30: this said "most of the plan's expected deltas (0.2–1.0 mAP) are
at or below the instrument's resolution", which is **false over the upper half of its own range** —
the 30K aggregate half-width is ±0.0027 = **0.27 points**, so a +0.7 effect is ~2.6× resolvable even
today. The same error underlies §11's ML-Decoder rationale. The instrument work is still the right
first move — for the **per-decile and per-tag** claims, where the 30K CIs genuinely cannot resolve
anything — but do not use "unmeasurable" to defer decisions about large aggregate effects. **Note
also this document mixes fractions (0.6518, 3e-3) with points (+1.0, +9) throughout; pick one.**)*

### 8.1 Already done — strike from the list
`selection_metric: val_mAP` is **live** (`unified_config.yaml:474`; dispatch table
`train_direct.py:59-63`, read `:1680-1695`, applied `:2971-2973`, early-stop message `:3101`). The
correction doc's item 2 was stale.

### 8.2 Val set — the biggest defect, and cheap
`max_val_samples: 30000` (**0.51%** of the measured 5,921,102 corpus) gives:

| | tags <5 val positives | tags <10 | decile-10 median | decile-10 mAP 95% CI |
|---|---|---|---|---|
| **30,000 (now)** | **46.06%** | **65.8%** | **2.4** | **±0.0125** |
| 100,000 | 0% | 0% | 7.9 | — |
| **296,056 (the actual full split)** | 0% | 0% | 21.8 | ±0.0045 |

Aggregate mAP survives. **Everything below aggregate does not.** Confirmed against the real
artifact: on 30K only **10,406 of 19,292** tags have support ≥5.

*(Three numbers corrected 2026-07-30. **44.5% → 46.06%**: 44.5% was a frequency projection,
contradicted five lines later by the measured 8,886 of 19,292 — quote the measurement. **100K decile-10
median 8.9 → 7.9.** **"276,000 (full 5%)" → 296,056**: the split is 5,625,046 + 296,056, confirmed
by the split-file bodies, `logs/training.log:13126`, and `_preds_cache_last_full.pt` being exactly
296,056 × 19,294 × 3 bytes. `dataset_loader.py:2784` hardcodes `split_ratio = 0.95` two-way.)*

**Resolution reference, used throughout §8** (simulated 95% CIs on macro-mAP): aggregate **±0.0027**
at 30K and **±0.0009** at 276K; per decile (~1,929 tags) **±0.0045** at 276K; the tail decile alone
at 30K, **±0.0125**. Binning bias adds a further ~−0.0031 (§8.3).

> **⚠ UNVERIFIED — flagged 2026-07-30, and these numbers are load-bearing throughout §8 and §9.**
> - **No script, artifact or benchmark in the repo produces any of them**, and the binning bias is
>   cited only to a retired review doc. **Commit the simulation and re-measure the bias on the actual
>   30K × 19.3K matrix**, or stop quoting them to three decimal places.
> - **They are unpaired *level* CIs, and §9.3 quotes them as the resolution of a paired-Δ test.**
>   0.0027/0.0009 = 3.00 against √(276/30) = 3.03 is pure 1/√n scaling of a single model's level,
>   with no correlation term. At zero pairing correlation the per-decile paired half-width is
>   0.0045·√2 = **0.0064** (MDE₈₀ ≈ **0.0091**) — against a target effect range of 0.002–0.010, i.e.
>   Gate 1 may be underpowered for most of what this plan is chasing. **Measure ρ from the V1/V2 pair
>   and restate these as paired quantities.**
> - They are also hard to reconcile with each other: ten uniform ±0.0045 deciles imply a ±0.0028
>   aggregate, not ±0.0009. Either ±0.0045 is the *widest* decile rather than a uniform one, or one
>   of the two is wrong. Say which.

> **⚠ READ THIS BEFORE §9. Only 30,000 images in the entire corpus were never seen by V1.**
> `dataset_loader.py:2801-2806` folds the val excess into training (`train_list = train_list +
> excess_val`), logged as *"was 296,056, moved 266,056 to training"* on **every launch of both
> phases**; the Arrow table is then filtered to **5,891,102 rows** for training. So the 296,056-image
> split is **89.9% V1 training data**, and **no ≥100K V1-vs-V2 held-out comparison can be
> constructed after the fact** — the images are spent. Fixing the fold helps V2 and every model
> after it; it cannot un-train V1. §9.3's frozen `EVAL-SET` is rewritten accordingly.

- [ ] **Raise `max_val_samples` to the full 296,056 split** (or **≥100K** if validation time bites —
      the 100K row above is what makes that fallback safe: it already clears every support floor).
      **Cost is NOT validation time only** *(corrected 2026-07-30)*: `train_direct.py:3080-3084`
      accumulates fp16 probs + bool targets per batch whenever `use_tensorboard or
      calibration_enabled` (`:2984`; TB is on), and `:3195` does `torch.cat(...).float()` — an fp32
      (N, 19294) allocation **while the fp16 list is still live**, ≈**32–42 GB of host RAM at 296K**.
      The code already says so at `:2987-2990` (*"will not fit once max_val_samples grows to the full
      ~276K split"*), and on disk `_preds_cache_last_30000.pt` is 1.74 GB against
      `_preds_cache_last_full.pt` at **17.1 GB**. 255 GB host RAM means it survives, but this must be
      sized deliberately, it pulls against §8.3's "un-gate per-decile mAP from `use_tensorboard`",
      and disabling the accumulation would kill **all** Appendix-A val telemetry.
- [ ] **Stop folding the val excess into training** (`dataset_loader.py:2801-2806`;
      `logs/training.log:13127`). The `pr_threshold_*_full296k` artifacts are ~90% train-contaminated
      and must not be quoted as held-out — **including by §1**, which did exactly that until
      2026-07-30.
- [ ] **Carve a held-out TEST set.** Val currently does triple duty — selection, threshold fitting,
      and reporting — so every reported number is optimistically biased.
- [ ] **Group-aware split** on perceptual-hash cluster ID. Current split is uniform random over
      individual sidecars (`dataset_loader.py:2732-2746`, `seed=42` default at `:2556`), with no
      perceptual-hash, artist, series or post-id grouping. Measured near-dup cluster rate 0.26% over
      5.54M images — but the corpus at split time was 5.92M, so ~382K images were never
      dedup-scanned, and whether the 14,230 flagged deletions were applied is not verifiable from
      the artifacts. **Magnitude bound, and why this is hygiene rather than a blocker: expected
      leakage is order 10¹–10² images of the 30K val set** (0.26% × 30,000 ≈ 78) — *corrected
      2026-07-30 from "10²–10³", i.e. an order of magnitude smaller than stated, which strengthens
      the "hygiene not blocker" call.* A real defect worth fixing, nowhere near large enough to
      explain 0.674. Note `logs/dedup_hashes/dedup_clusters.json` is ~6.5 months stale.

### 8.3 Metric correctness
- [x] **Binned AP — resolved, keep `ap_thresholds: 200`.** *(Corrected: an earlier draft said set
      `thresholds=None`. That is impractical and unnecessary.)* The exact estimator retains every
      update's preds/targets (~7 GB resident at 30K × 19.3K labels), and the transient peak scales
      brutally — measured on RTX 5090 @ B=48: **200 → 9.3 GB / 40 ms per update; 500 → 23.3 GB;
      2000 → 93 GB and ~103 min per validation pass** (`train_direct.py:1702-1738`,
      `configs/unified_config.yaml:553-567`). The binning bias has been **measured at ~−0.0031 vs
      exact, near-uniform across support buckets**, so it largely cancels in checkpoint-to-checkpoint
      comparison. Nothing to change.
- [x] **Set the stop threshold from the measured resolution floor, not a guess — done 2026-07-29.**
      Three independent sources agree on ~0.003: the binning bias (−0.0031), the 30K sampling CI
      (§8.2), and the config's own note *"treat val_mAP differences below ~0.003 as noise."* Applied
      as the **new** key `early_stopping_min_delta: 0.003`, read by
      [stop_conditions.py](../stop_conditions.py); at the full 276K split the sampling CI shrinks but
      the bias term dominates, so 3e-3 still holds. *(A "one measured bootstrap SE" heuristic would
      give ~1.4e-3 at 30K and ~4.5e-4 at 276K — 2–7× too small, because it ignores the bias term.)*
      > **Two notes added 2026-07-30. The value is safe; the derivation is not.**
      > - **`stop_conditions.py`'s own docstring contradicts this derivation**: it says 3e-3 is the
      >   **between-run** floor and is conservative *within* a run ("well under 0.0015"). The error
      >   direction is safe — a too-large floor stops late, not early — but the plan and the code
      >   disagree about what the number *means*, and only one of them can be right.
      > - **The plateau rule does not do what this item implies.** `stop_conditions.py:406-417`'s
      >   `_window_gain` is `max(recent w) − max(prior w)`, which on a **monotone** curve is the
      >   **sum** of the last w deltas — so it lengthens the horizon rather than denoising it. The
      >   operator-relevant bar is therefore `min_delta / window`: at phase 2's `window=2` that is
      >   **1.5e-3**, which is still above V1-P2's smallest real gain (+0.00148). Check this against
      >   the intended semantics before P1.
      **Note what changed from this item's original framing:** `early_stopping_threshold` was *not*
      raised to 3e-3. It still governs best-checkpoint selection, where near-zero is correct (you
      want to keep the argmax). Raising it there would have stopped `best_model.pt` tracking the
      actual best. And 3e-3 exceeds a single epoch's real gain often enough that a *per-epoch*
      comparison against it trips on a healthy curve — V1-P2's gains were +0.0041 +0.0045 +0.0015
      +0.0031 +0.0022 — which is why the plateau rule compares window maxima instead.
- [x] **Freeze the macro-average tag list — done 2026-07-29.** `validation.freeze_macro_tag_list`
      (default true) freezes the denominator at the first validated epoch of the phase and persists
      the index list in the checkpoint, so it survives a soft stop. Later drift against the frozen
      list is logged at WARNING (it was masking to `val_pos_counts > 0` per epoch, dropping ~958 tags
      on the 30K draw, logged at debug only — silent in practice).
- [ ] **Per-tag-optimal macro-F1 inside the val loop (~20 lines) — but log it as an ORACLE, and never
      select on it.** Flipping `threshold_calibration: per_tag` is a **no-op for selection**: its
      output goes to `thresholds.json`/TensorBoard only and never reaches the metric.
      `ThresholdCalibrator._compute_f1_grid` returns a `(num_thresholds, C)` grid
      (`evaluation_metrics.py:582`), and `grid.max(axis=0).mean()` over supported columns is the
      number. Gate the 19,292-entry log line and TB write.
      > **⚠ Added 2026-07-30.** `grid.max(axis=0)` argmaxes **per column on the evaluation data** —
      > it is a **fit-on-eval per-tag oracle**, exactly what **§8.4 forbids** ("fit on a disjoint
      > CALIB split, never on the reported eval split"), and `_calibrate_per_tag`
      > (`evaluation_metrics.py:605-626`) has only a zero-support fallback, no minimum-support floor.
      > Measured inflation on the same checkpoint and draw: `per_tag.f1_summary.mean_f1` **0.6751** vs
      > single-threshold `support_ge_5.f1_optimal` **0.6212** — **+0.054** — and the same checkpoint
      > reads **0.6410** on the 30K draw, so the statistic is **not comparable across eval-set
      > sizes**. This is the same defect that inflated §1's third refutation. Useful as an upper
      > bound; never as a gate, a selection metric, or a reported headline.
- [x] **Reset `best_metric` on any metric change — already done, this item is STALE-OPEN** *(verified
      2026-07-30)*. The cross-metric guard is in HEAD at ~1165-1211 and re-seeds from
      `best_model.pt`'s named metrics; verified live against the artifacts. The cited
      `train_direct.py:1254-1255` `max(best_metric, loaded_best)` no longer governs. Strike from the
      blocker list.
- [ ] **Per-decile mAP** (deciles from `vocabulary.json`; assignment already exists at
      `asl_telemetry.py:331-348`, per-group mAP at `evaluation_metrics.py:491-493`). Un-gate from
      `use_tensorboard`.

### 8.4 Thresholding and calibration — replace the specified approach
**Per-tag isotonic across 19K tags is not viable.** Niculescu-Mizil & Caruana (ICML 2005): isotonic
**overfits below ~200–1,000 calibration cases** *(corrected 2026-07-30 from "~2,000"; note it counts
**cases**, not positives)*. Our median tag has ~6 positives, so the conclusion is untouched — the
margin is three orders of magnitude either way. Ullah et al.
([arXiv:2411.04276](https://arxiv.org/abs/2411.04276)) — the only calibration study at XMC scale —
use a **global** calibrator and name per-label calibration as *future work*. They also show marginal
per-label ECE is misleading (**ECE 0.05 vs ECE@5 = 9.25** on the same model) and that *"marginal
calibration… does not imply top-k calibration."*

Calibrate after training, not via the loss: asymmetric losses are not strictly proper scoring
rules, so outputs stay systematically shifted regardless of γ tuning (Cheng & Vasconcelos,
CVPR 2024 — carried from the retired ASL_plan §5). Bound noted 2026-07-30: post-hoc repair is
limited by finite-net deviation, not by the margin — Cheng's exact inverse map only ~halves
calibration error (their Table 3), and the m-induced non-invertible region is just [0, 0.05],
below any shipping threshold. Trace-ρ tags equilibrate at ~0.12–0.35 and no monotone transform
re-separates them — keep operating thresholds above that region (0.79-class thresholds already
are).

- [ ] **Global isotonic on top-k scores** (their recipe: ECE@1 17.02% → **0.17%**, no accuracy loss —
      *but note added 2026-07-30: Ullah produce that by k-fold cross-fitting the isotonic map **on the
      reported test set**, an easier design than this plan's own disjoint-CALIB rule, and all their
      datasets are text XMLC*), **plus per-decile isotonic** (~1,900 tags each = ample support),
      **plus per-tag Platt only for tags with ≥100 calibration positives**, shrunk toward the decile
      fit.
      > **Sizing corrected 2026-07-30.** "~1,230 tags" and §8.5's "≥100 → 6,226 tags" are **both
      > right, at calibration-set sizes 9× apart**: 1,230 reproduces exactly at **S = 30,000**, 6,226
      > at **S = 276,000**. Since this whole recipe is support-gated, **`CALIB-SET` must be sized
      > before it can be specified** — and at a 296,056 total holdout it cannot be 276K without
      > starving EVAL and TEST (§9.3). Fix the split budget first, then re-derive the tag count.
      > *Also: the shrinkage weight, the composition order of the three calibrators, and whether they
      > compose monotonically are all unspecified.*
- [ ] **Fit on a disjoint CALIB split**, never on the reported eval split. Thresholds fit on noisy
      labels skew **high** (§9.1's bias at the operating point: firing on unlabeled true positives
      is scored as FP); before shipping, cross-check per-group thresholds against the Anima slice
      (probe, not anchor — §9.4).
- [ ] **ECE estimator:** use smoothed / equal-sample-bin ECE (SmoothECE, Błasiok 2023; Dual-TS,
      [arXiv:2308.08366](https://arxiv.org/abs/2308.08366)) — vanilla equal-width ECE is biased at
      our ~0.18% positive rate. Applies to Gate 2's ECE (at-threshold since 2026-07-30) and
      per-decile reliability diagrams.
- [ ] **Benchmark xCOLUMNs, then decide — do NOT adopt unconditionally** *(downgraded from
      "**Adopt**" 2026-07-30)*. ([arXiv:2401.16594](https://arxiv.org/abs/2401.16594) ICLR 2024;
      [arXiv:2311.05081](https://arxiv.org/abs/2311.05081) NeurIPS 2023;
      [library](https://github.com/mwydmuch/xCOLUMNs)) — post-hoc, retraining-free optimization of
      macro-F1/macro-recall at a per-instance budget k, built for extreme label spaces, needing no
      verified negatives. Three problems, all verified:
      - **It optimizes a mechanism the product does not have.** ICLR'24 §3 verbatim: *"we assume that
        the predictions are budgeted at k … ‖ŷ‖₁ = k with probability 1."* The shipped tagger is
        **threshold**-based (`unified_config.yaml:569-570`: `prediction_threshold: 0.7927`,
        `top_k: null`). **k is given no numeric value anywhere in this plan**, and `coverage@k` is
        never defined. *Resolved 2026-07-30: Gate 2 is restated at the threshold (§9.3).* xCOLUMNs'
        budgeted-@k mechanism remains product-mismatched unless a top-k product is ever adopted.
      - **At our scale the paper's own results go the other way.** Its largest evaluated label space
        is AmazonCat **m = 13,330**, where it **loses** to TOP-K+wPOW (40.90 vs 45.70 macro-F1@3),
        and the authors attribute the failure to **labels with <10 positives** — which is 65.8% of
        our vocabulary at 30K val.
      - **"Statistically consistent" is asymptotic**, and Thm 5.1's m²·√(m log m log n / n) term is
        ~1e9 at m = 19,294, n = 276K. Algorithm 1 also needs a labelled held-out S₂ whose
        false-positive entries missing positives inflate, and it returns a **randomized** classifier.

      It is still the best candidate in this space and it is not in `requirements.txt` — budget the
      integration, run it against a plain per-decile threshold baseline, and let the benchmark decide.
- [ ] **Validate the metric panel** using their NeurIPS'23 Table 1 test: score a head-only-capable
      model with every metric you plan to use. Under total tail collapse P@5 drops 8.7% and PSP@5
      18.4%, while **macro-F1@5 drops 88.5%**. Any metric that doesn't move cannot detect tail
      failure.

### 8.5 Support floor — selection and reporting are different metrics

Macro-averaging over all 19,292 tags is a *choice*, not a default, and an unqualified one puts
roughly half the metric's weight on tags that cannot be measured. Even at the full 276K val split,
**47.8% of tags have <50 positives**. *Learnable* and *measurable* are different problems, and only
the second one is ours here.

**The V1 evidence, extracted from the TensorBoard event files (Phase 1, last logged validation
event, step 209135 — note the run's *terminal* checkpoint is step 209899,
`logs/training.log:12904-12910`, so this table is the last measured epoch, not the last trained
one):**

| bucket | tags | **mean support (30K val)** | mAP | f1_macro @ 0.2653 |
|---|---|---|---|---|
| 500–999 | 7,619 | **2.45** | **0.6250** | 0.0279 |
| 1,000–4,999 | 7,873 | 8.52 | **0.6345** | 0.0421 |
| 5,000–9,999 | 1,482 | 30.3 | 0.5713 | 0.0567 |
| 10,000+ | 2,314 | 399 | 0.5809 | 0.0782 |
| 300–499 | **0** | 0 | 0 | bucket is structurally empty (vocab floored at 500) |

**Rare tags are demonstrably being learned — this is not a small-sample artifact.** For a tag with
2.45 positives in 30,000 images, **random ranking gives AP ≈ 0.00042**. The observed 0.6250 is
~1,500× that, and it rose monotonically across Phase 1 (0.6028 → 0.6153 → 0.6250): the model is
placing a 500-occurrence tag's true positive at about **rank 2 of 30,000**. Mechanism: tail tags do
not learn from scratch. `fox_hair_ornament` inherits hair/ornament/fox representations built from
millions of head-tag examples and only learns a small delta — which is why 500 examples suffices in
a 19K-way multi-label setting and would not for an isolated binary classifier, and the same reason
Tencent ML-Images (11,166 classes) and RAM (6,449 tags) work. Per-*tag* AP at 2.45 positives is
high-variance, but it is **not biased**, so the bucket mean (SE ≈ 0.005 over 7,619 tags) is
trustworthy even though no individual tag's is.

**So the floor belongs on per-tag operations, NOT on the selection metric** — averaging over
thousands of noisy-but-unbiased terms is the whole point. And note what the bucket mean does *not*
say: it says the tail **is** learned, not **which** tail tags are individually reliable, and 2.45
val positives cannot tell you. Surfacing rare predictions therefore needs a decile-level operating
point plus the gold slice, not a per-tag threshold fitted on noise. **Gate 3 is where per-tag tail
reliability actually gets established** — its slice is stratified across deciles with confusables
oversampled, the only affordable way to measure the tail *per tag*.

The rare buckets score *lower* on F1 than the head while scoring *higher* on mAP. That divergence is
the fixed-threshold artifact, not a statement about the tail: the in-training metric is hard-coded at
**θ = 0.2653** while the measured F1-optimal threshold is **θ ≈ 0.76–0.81**. Properly thresholded,
V1-P2 posts **micro F1 0.7004** and **per-tag macro F1 0.6751** — against the **~0.013** the training
loop was reporting. That is also the measurement basis for §1's +8.6%-macro-F1 refutation, and it
means V1's manual stop was decided while watching a number with no relationship to model quality.

*(This does dispose of progressive-plan §2.1's "rare bucket still climbing while others plateau" —
all four buckets climbed at comparable rates, +0.0019–0.0024 mAP/epoch. But the rare bucket's
*level* is real.)*

| support floor (val positives, 276K split) | tags kept | % of vocab | **% of label mass** |
|---|---|---|---|
| none | 19,288 | 100% | 100% |
| ≥25 | 16,556 | 85.8% | **99.3%** |
| **≥50 ← use this** | **10,074** | **52.2%** | **96.9%** |
| ≥100 | 6,226 | 32.3% | 93.9% |

- [x] **Selection metric = macro-mAP on a frozen list — DONE, and this item contradicted §8.3's own
      `[x]` until 2026-07-30.** The tree already has `keep_classes = frozen_macro_mask`
      (`train_direct.py:3139`, config `:645` default true); the old text's "`keep_classes =
      (val_pos_counts > 0)` currently lets it [drift]" was accurate only at HEAD, not in the working
      tree. Two clarifications that matter:
      - **"over ALL tags" is not what is frozen.** The frozen list is the **first-epoch supported
        set** (`train_direct.py:3107-3113`) — **18,334 of 19,292** on the 30K draw, not 19,292. That
        is the right behaviour, but state it, because it is also the denominator that makes
        §8.5's bucket mAPs and the headline mAP reconcile (0.6518 × 18,334/19,292 = 0.6195).
      - **"§8.2's CIs are ample" is true for the aggregate and false per decile** — ±0.0027 vs
        ±0.0125 at 30K. Since the very next bullet mandates per-decile reporting, do not carry this
        sentence as blanket reassurance.
- [ ] **Reporting = per decile, all tags** — reliable at decile granularity (§8.2). Do not over-read
      single-epoch deltas in deciles 9–10; do trust the level and the trend. **Note the decile
      assignment inherits §10's occurrence-inflated tag frequencies — fix the counter before freezing
      `TAGSET`.**
- [ ] **Support floor ≥50 applies to per-TAG operations only** — per-tag thresholds
      (`_calibrate_per_tag` has only a zero-support fallback, no *minimum*-support floor —
      `evaluation_metrics.py:605-626`), per-tag calibration (§8.4 uses ≥100 for Platt), and any
      per-tag claim. Below the floor, fall
      back to the tag's decile-level operating point rather than fitting a free parameter on ~5
      events. `tools/find_pr_threshold.py` already has the `--min-support` pattern (flag at `:85`,
      default 5; `per_tag_operating_points` defined at `:310`, applied at `:627`, reported at
      `:651`).

**Hygiene backlog (non-blocking, re-verified 2026-07-29):** the ghost config keys `num_groups` /
`tags_per_group` are filtered as unused at `train_direct.py:615-619` (`tags_per_group: 10000` is
still in the live config at `unified_config.yaml:53`); `use_style_token` appears nowhere in the repo
and `num_special_tokens` only in a comment at `training_utils.py:2013`, so neither is a real key.
`validation_loop.py`'s 3D-output handling is vestigial — delete when convenient.
`loss_functions.py:45`'s `label_smoothing` default is 0.05 while the operative value is 0.0
(config-driven, §4 pins 0.0) — fix the default. Dead-code candidates (from the retired
`deprecated_candiates.md`, re-verified 2026-07-29): `log_index_order_hash`,
`LearningRateSchedulerFactory`, `TrainingMetricsTracker` in `training_utils.py` are referenced only
by its `__main__` self-test block.

**Not recommended, but worth knowing the shape of:** raising `vocab_min_frequency` 500 → 2000 would
drop **62.6% of tags for 5.1% of label mass**. That is a *product* decision, not a metric one — tag
coverage is what distinguishes this tagger, so leave the floor at 500.

---

## 9. How we decide V2 is actually better

### 9.1 The bias that points against V2

For tag `c` with missing rate ρ_c, the val set contains truly-positive-but-labelled-negative images.
A model that has learned the concept ranks them **high**, and in AP a false positive at rank 1 costs
far more than at rank 500. So **measured AP is biased downward, and the bias grows with the model's
true quality.** This is model-dependent systematic bias, not variance that averaging removes.

Measured ρ makes it worst exactly where we want to look: ~0.3–0.5% at head, ~5% median at the tail
decile, **23–64% on individual confusable tags** (per-bucket medians in §10). Northcutt et al. show
label errors can **flip model rankings** — *though the citation is weaker than stated (corrected
2026-07-30): the average error rate is **3.3%**, the flips require a **further +5–6pp** of injected
prevalence on top of that, and the noise is **wrong-positive, single-label, accuracy**-based — not
missing-positive AP. It establishes that the mechanism is real; it does not establish that our
regime is past the threshold.*

**Three consequences to pre-empt:**
1. V2 can be genuinely better and measure worse, especially in tail deciles.
2. **Any ignore-set / relabeling hyperparameter tuned on noisy val will be tuned too conservatively**
   — this is why §4 says calibrate on the gold slice, not val.
3. Early stopping on noisy val can peak and decline as the model exceeds the annotation. *(This did
   not cause V1's stop, which was manual.)*

### 9.2 Sibling-negative evaluation — useful, but NOT a clean-label arbiter
OpenImages evaluation ignores unannotated classes and does not penalize false positives on them. We
can't apply that directly, but we **can** restrict evaluation to cells with *some* evidence of
negativity: tags in a confusable group where a sibling is positively labelled. On hair colour/length
and the other confusable groups — where ρ is 23–64% and where review budget goes — this is a cheap
low-noise-ish signal on the hardest tags.

> **⚠ CORRECTED 2026-07-30 — the sign is wrong, and this was load-bearing.** The old text called a
> sibling-positive label *"reliable evidence of negativity"*. It is not:
> - `configs/confusable_groups.json`'s **own header** states colours are **NOT treated as
>   exclusive**, and the only filter is that **exactly one sibling be *labelled***. That does not
>   exclude a sibling which is genuinely *present but unlabelled* — the dominant failure on exactly
>   these tags.
> - The metric is `gap = p(labelled sibling) − max p(unlabelled siblings)`. So a model that has
>   **memorized a wrong positive** — confidently agreeing with a mislabelled `aqua_hair=1` and
>   suppressing the true `green_hair` — **maximizes the gap.** The observable is *increasing* in the
>   one noise mode §4's scope fence says no loss knob can touch.
> - The loop is closed the wrong way in code: `stop_conditions.py`'s `CLEAN_MARGIN_NOTE` asserts the
>   opposite, and `:792-802` **demotes a tripped `peak_decline` to advisory** whenever this arbiter
>   held or rose. A rising sibling gap therefore *silences* the very stop signal that
>   wrong-positive memorization would trip.
>
> **Demote it to the same status Appendix A already gives `Δp_hard`: it can veto, never
> green-light.** Remove its authority in `stop_conditions.py`. It remains a genuine boundary
> observable and still feeds Gate 3's off-diagonal metric — it just cannot certify cleanliness, and
> the ρ = 23–64% on these tags is precisely why.

### 9.3 The three-gate protocol — pre-register before the run

**Frozen artifacts** (hashed, versioned): `EVAL-SET`, `CALIB-SET` (disjoint), `TAGSET` (frozen list +
decile assignments), `GOLD-SLICE` (§9.4). One **offline scorer**, not the training loop.

> **⚠ REWRITTEN 2026-07-30. The old spec — "`EVAL-SET` (≥276K, never used for fitting), reused
> identically for V1 and V2" — describes an artifact that cannot exist.** Per §8.2, V1 trained on
> 5,891,102 of 5,921,102 images; **exactly 30,000 were never seen.** A ≥276K set reused for both arms
> is ~90% V1-training-data, which biases Gates 1 and 2 *in V1's favour* — the wrong direction for a
> protocol whose job is to stop us shipping a worse model.
>
> **There is no way to recover a large clean V1 baseline.** Choose one and record it:
> - **(a) Honest-but-small.** Score both arms on the 30,000 V1-clean images. Unbiased, and the CIs
>   are §8.2's own: **±0.0027 aggregate, ±0.0125 tail decile.** That resolves the aggregate for
>   effects ≳0.006 and **cannot resolve per-decile deltas at all** — so Gate 1's per-decile pass rule
>   must be dropped or explicitly declared underpowered.
> - **(b) Large-but-biased, with the bias stated.** Use the full 296,056 and report V1's number as an
>   **upper bound on V1**, so a V2 win on it is conservative and a V2 loss is uninformative.
> - **(c) Retrain a V1-equivalent** on the corrected split. Honest and fully powered; costs a second
>   multi-week run and is almost certainly not worth it.
>
> **Recommendation: (b) for Gate 1 with the bias stated, (a) as the confirmatory check, and let
> Gate 3 carry the decision** — which is what §2 already says it does. What is *not* acceptable is
> the current text, which promises a clean shared artifact and would be read as one.
>
> **A split budget must accompany this.** No block in this plan enumerates
> `N_train / N_EVAL / N_CALIB / N_TEST / N_GOLD` as disjoint sets summing to N, and at a 296,056
> holdout an `EVAL-SET` of ≥276K consumes 93% of it — leaving ~20K for CALIB **and** TEST, against
> this plan's own finding that 100K is the minimum clearing every support floor. `CALIB-SET` is never
> sized anywhere. The loader cannot express any of this: `dataset_loader.py:2784` is a hardcoded
> two-way `split_ratio = 0.95` and there is no TEST or CALIB concept in the repo.

**Gate 1 — ranking quality on the noisy set.** Macro-mAP over `TAGSET`, **per decile**, with
paired **image-level** bootstrap 95% CIs (bootstrap over images, not tag×image cells — cells are
correlated within an image).
*Pass:* V2 ≥ V1 aggregate, CI on Δ excludes 0 in at least the top 5 deciles.

> **Two fixes, 2026-07-30.**
> - **Score both arms with the SAME estimator.** This said "**exact** macro-mAP" while §8.3 keeps the
>   training loop at `ap_thresholds: 200` with a ~−0.0031 binning bias. Every V1 number quoted in
>   this plan is **binned**; scoring V2 exact would hand V2 a free ~0.003 — the same size as the
>   stated resolution floor and larger than several effects being chased. Binned-vs-binned is fine
>   (the bias largely cancels); mixed is not.
> - **The pass rule has no stated power.** The conjunction over 5 deciles *over*-controls Type I
>   error, so no multiplicity correction is needed — but Type II is uncontrolled and **no power or
>   MDE statement appears anywhere in §8 or §9**. Given the paired-CI correction above, state the
>   MDE per decile before pre-registering, or this gate rejects good models silently.
**Do not fail V2 on the bottom deciles here** — §9.1 predicts adverse bias exactly there. Record and
adjudicate at Gate 3.

**Gate 2 — deployment quality at the shipping operating point.** Restated at a **threshold**
2026-07-30 (the product ships threshold-based — `prediction_threshold: 0.7927`, `top_k: null` — and
k was never given a value; §8.4's @k objection is resolved accordingly). Calibrate on `CALIB-SET`
only (§8.4), re-derive each arm's operating threshold by the **same procedure**, then report on
`EVAL-SET` at that operating point: macro-F1, precision, recall, `mean_active` (mean tags fired per
image), and per-decile reliability diagrams (smoothed ECE, §8.4).
*Pass:* no regression in macro-F1; **precision and `mean_active` no worse than V1 at V1's own
re-derived operating point**; ECE improved or flat. The firing comparison is the point: V1's actual
defect (`mean_active` 1322 → 4493 at a frozen threshold while mAP *rose* —
TRAINING_HEALTH_TRACKER.md P1-E32/P2-E0) passes Gate 1 and the old @k form of this gate undetected;
this clause is what fails it.
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
- Estimate with **statAP** (Aslam & Pavlu) or **infAP** (Yilmaz & Aslam, CIKM 2006) / **xinfAP**
  (Yilmaz, Kanoulas & Aslam, SIGIR 2008), with variance estimates. *(Attributions corrected
  2026-07-30 — statAP was credited to the wrong authors, SIGIR 2008 is xinfAP not statAP, and the
  infAP author order was reversed.)* **And soften "unbiased":** statAP's AP is a **ratio estimator**
  which the paper itself says is *"not guaranteed to be unbiased"*, calling the bias negligible at
  TREC per-query scale. **Unverified for us:** whether that holds at Gate 3's per-tag relevant-set
  sizes of ~5–20. Simulate the estimator at R = 5, 20, 100 before trusting a tail-stratum verdict.
- Record an **ε_gold** for the slice itself. A screened-not-adjudicated slice cannot certify.

**Pooling both systems is the critical design choice** — it removes the better-model-penalized-more
bias for the compared pair. This is **standard TREC methodology**. *(Citation split 2026-07-30:
Schultheis et al. KDD 2022 §4.1 is "**Bias-controlled validation and test sets**" and recommends
**nothing about pooling two systems** — it supports the bias-controlled *slice*, which is a different
and also-needed thing. Do not attribute the pooling design to it.)* **Do not use the Anima synthetic set as the arbiter** — it measures
model-precision-on-synthetic-data. (The edits.db verdict store is ~705K verdicts, all on synthetic
Anima IDs; the live db was wiped 2026-07-27 — 58,207 predictions, 0 verdicts.)

---

## 10. Data — re-scoped, not de-scoped

The correction doc's *facts* are right: cleaning has touched 0.65% of tags and **0.017% of label
mass**, and will not move global ρ on V2's timeline. Its *conclusion* — drop the workstream — is
backwards. That number says the campaign is under-resourced and mis-targeted.

**Be precise about which data lever.** **RAM Table 5**
([arXiv:2306.03514](https://arxiv.org/abs/2306.03514)) scales **tags** 12.0M → 41.7M for
**OpenImages-rare 63.54 → 67.17 (+3.6)**. That is *label completion*. *(Attribution corrected
2026-07-30 — this was cited to RAM++ (2310.15200). And note the added tags are **machine-generated**,
which is a real caveat for an argument being used to justify a **human** review budget.)* By contrast Zhai et al.
measured **JFT-300M → JFT-3B (10× more noisy images) buys only ~1%**, equally for small and large
models. **Completing labels on the corpus we have ≫ adding more equally-noisy images.**

**How big is the corpus — MEASURED 2026-07-30. The previous "N ≥ 6.77M lower bound" was wrong, and
wrong in the direction nobody was guarding against: it was an over-estimate, not an under-estimate.**

**Operative planning figure: ~6.2M images.** **On-disk count: 5,921,102** — reconcile the two before
the run; every arithmetic below is given at both.

The measurement takes ~6 seconds and agrees four independent ways:
- Live enumeration of `L:/Dab/Dab`: **5,921,102** `.json` sidecars and **5,921,102** images across
  798 shards. `data.storage[0].path` (`unified_config.yaml:67`) is the **only** configured root.
- `logs/splits/2df7eb3c2bc0d784.{train,val}.txt` headers: `# FILE_COUNT=5921102`, bodies
  5,625,046 + 296,056.
- `logs/metadata_cache/2df7eb3c2bc0d784.metadata.arrow.meta`: `{"count": 5921102}`.
- `logs/training.log`: "Arrow cache from 5,921,102 total files", on every launch.

**Why the `highres` inference failed, so it is never repeated.** `vocabulary.py:149` counts tag
*occurrences* (`tag_counts[tag] += 1`), and ~80% of sidecars duplicate their trailing
meta/character/copyright tags. So `highres: 6,773,251` **exceeds the file count** — impossible for a
per-image tag, which is the tell that should have caught it. §8.2 of this same document was already
using 5.92M.

**Companion defect, still live:** `vocabulary.json` frequencies are occurrence-inflated **~1.6–1.9×
on meta/copyright/character tags** but exactly **1.00× on general tags** (`1girl`, `solo`,
`long_hair`). Training labels are unaffected (`vocabulary.py:635-659` does `vector[indices] = 1.0`),
but **decile membership is biased along a category axis** — which touches §8.5's reporting deciles,
§10's ρ̂ fit, and the decile assignment at `asl_telemetry.py:331-348`. Fix the counter before
freezing `TAGSET`.

**What this changes.** Every "6.8M" in this plan is ~10% high against the operative 6.2M and ~15%
high against the disk. Specifically: the §2 data row (fixed), §6's samples-per-epoch and the ≈483M
budget (fixed below), §8.2/§8.5's support projections, and the AdamW8bit sparse-head argument. The
old "if N is really ~9.7M" hedge is **void** — it pointed the wrong way.

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

**Per-bucket ρ, recomputed:** **median ρ = 0.47 / 1.59 / 3.30 / 5.11%** across the four frequency
buckets (500–999 / 1K–5K / 5K–10K / 10K+, reading tail→head as 5.11%→0.47%). This **replaces an
earlier aggregate of 0.32 / 3.73 / 4.51 / 12.10%**, which was wrong because it used the
DataCleaning Project's `gt_current = 0` values as the denominator. Quote the medians, not the
aggregate.

*Measurement caveats on the campaign's own numbers:* the 4.9:1 missing:wrong ratio compares a pool
mined at 0.155% (24.8M FP candidates → 38,463 adds) against one mined at 69.6% (11,279 → 7,852) —
a ~450× difference, so it measures review budget, not noise composition. **And the FN pool is
model-defined**, so any wrong positive the model has successfully *memorized* is structurally
invisible to it — the wrong-positive share is therefore a floor, and the cleaning track's error
budget must not treat 1/4.9 as the true composition. Separately, the `gt_current = 0`
tags (`exercise`, `beam_saber`, `white_wristband`) **do** have labels in the training vocabulary
(559 / 998 / 705); the zero is in the DataCleaning Project's separate store, which disagrees with
`vocabulary.json` by a median 1.115×. Use the report's `training_count_oppai` column as the
denominator — it matches `vocabulary.json` exactly for all 126 tags.

---

## 11. Explicitly rejected, with the reason

| Rejected | Why |
|---|---|
| Propensity-scored losses / metrics | **Rejection rewritten 2026-07-30 — the old reason was an inverted reading and does not support the conclusion.** What holds: Schultheis KDD 2022 shows A=0.55/B=1.5 are not fitted for any dataset, and unnormalized PSP@1 = 326% refutes the model. What does **not** hold: the "48.58% vs 60.83%" line described Table 4 as *selecting on a PSP metric*. It is a propensity-weighted **training** comparison, and the full column is φ₁ (do nothing) 60.83 · φJPV 66.03 · φJPV(fit.) **48.58** · φP(fit.) 63.53 · φR(fit.) 71.23 · φ_direct **73.72** · φPEJL 68.09 — i.e. **every variant except the degenerate JPV beats doing nothing**, and the plan quoted the single worst outlier as the paper's verdict. The honest rejection is narrower: *the unfitted propensity constants (A=0.55, B=1.5) are unusable here, and we have no fitted propensities.* Note the best variant (φ_direct) fits propensities from a **bias-controlled set** — exactly what §9.4's gold slice would supply, so this is a **revisit-after-gold-slice**, not a kill. |
| ML-Decoder / attention-pooling head *this run* | **The gain does not grow with class count:** 80 classes **+1.1**, 1,000 **+0.6/+0.8** *(note 2026-07-30: this row is **single-label ImageNet**, Table 13 — not multi-label, so the trend is assembled across task types)*, 9,600 **+0.7/+0.8** — and K=100→200→400 at 9,600 moves **86.7 → 86.8 → 86.8**, so quadrupling query resolution buys +0.1. Queries carry no class semantics (learnable, fixed-random and word-based queries all score identically). **CaiT Table 2, from scratch at matched capacity: attention class-pooling exactly *ties* average pooling, 80.3 = 80.3** — the closest published from-scratch analogue, i.e. the decoder head is a FLOPs optimization, not an accuracy win. All of ML-Decoder's gains used OpenImages-pretrained backbones fine-tuned on ≤118K images, the inverse of our under-trained-backbone regime. If ever revisited: **K ≈ 100** queries cross-attending patch tokens into the existing 896→19,294 linear head (~1.4% of backbone FLOPs, zero added head params), **do not scale K**, and only after the instrument is fixed — +0.7 is below current measurement resolution and a pooled gain could be entirely head-class movement (top decile = 83.7% of label mass). |
| RAL ([arXiv:2308.05542](https://arxiv.org/abs/2308.05542)) | Documented next-loss-family fallback in the retired ASL plan; not adopted |
| APL ([arXiv:2304.05361](https://arxiv.org/abs/2304.05361)) | Context only, not adopted |
| Schultheis & Babbar unbiased-BCE estimators ([arXiv:2109.11282](https://arxiv.org/abs/2109.11282)) | Require known per-tag missingness; §10's ρ̂ fit now partially weakens that objection — recorded, still not adopted |
| COMIC | Strongest rejected joint long-tail + missing-label framework — setting-mismatched |
| **Pretrained initialization — any route** (linear probe, LiGO growth, patient distillation, adopt-ViT-L outright, in-domain MAE/DINO SSL) | **User decision 2026-07-30: from-scratch (§5).** Routes retired to git history. LiGO note for the record: an independent NeurIPS 2024 evaluation finds trivial depthwise stacking dominates the learned operator and width is the weak growth axis — if growth is ever revisited under §3's forward-compat, stack depth-first. |
| **Experiment arms of any kind** (loss arms, the init bake-off, mid-run comparisons) | **Rejected outright 2026-07-30 (user): V2 trains a single configuration.** Fallback ladder — invoked only by §9 gate failure, never run alongside: **Hill** first (§4), then **LL-R/LL-Ct** (Large Loss Matters, CVPR 2022, [arXiv:2206.03740](https://arxiv.org/abs/2206.03740) — the only missing-label mechanism validated at 5,000 classes; loss-agnostic, needs no verified negatives). |
| **BalanceMix** | Unproven above 80 classes + warm-up-model dependence + the single-configuration rule (§7; rejection re-argued 2026-07-30). |

**Also rejected, argued upstream, and deliberately not restated here:** γ_neg descent (§1, §4) ·
P-ASL Ω_P (§4) · P-ASL Ω_L (§4) · the full-corpus prior-estimation pass (§4) · per-tag isotonic
across 19K tags (§8.4) · Query2Label (§3) · LAMB (§6) · naive mixup/cutmix/randaugment (§7) ·
SWA-on-top-of-EMA and model soups (§6) · patch14 (§3). Deferred rather than killed, with reasons at
§6: MHSA-only fine-tune at the resolution switch, and NaViT aspect-bucketed batching.

**Citation corrections carried forward:** `arXiv:2501.02364` is about *intrinsic data dimension*, not
class count — drop it from the width argument. `arXiv:2601.20994` ("Depth Delusion") is a
**language-model** study; cite as cross-modal tension, not ViT evidence. FixRes describes a
RandomResizedCrop discrepancy that a downscale-only letterbox pipeline does not have — cite it only
for "train low-res, fine-tune at target res helps." *(The config's old "(FixRes recommendation)"
label on the re-warmup was already corrected 2026-07-28 — `unified_config.yaml:262-271` — nothing
left to fix there.)*

---

## 12. Order of work

**0. DONE 2026-07-30 — count the corpus.** It was never a research task: a 6-second enumeration, and
the answer was already on disk in three places. **N = 5,921,102** (§10). This should have been item 0
from the start; it is recorded here so the pattern is visible — *measure before inferring, especially
when the inference is cheap to check.*

**Now, in parallel:**
1. §8 launch blockers — val set, metric correctness, calibration. Nothing else is measurable first.
   **Add: a split budget (`N_train / N_EVAL / N_CALIB / N_TEST / N_GOLD` as disjoint sets summing to
   N) — §9.3 and §8.4 are mutually unsatisfiable without it, and the loader cannot currently express
   it.**
2. §9.4 gold slice — the long pole; weeks. Start reviewer-reliability work immediately.
   **Gate 3, not the GPU, sets the go/no-go date. Say so in the schedule.**
3. **Implement the 2026-07-30 decisions — all currently plans-only, code and config untouched:**
   the WSD scheduler (§6) · `detach_focal_weight` in `loss_functions.py` (§4) · option C's shape in
   the config (§3) · neutralize the live `asl_schedule` block, `gamma_neg_override`, and the
   checkpoint-wins-over-YAML comment (§4's ⚠ list — all three sites verified still live
   2026-07-30) · **fix the sibling-gap arbiter sign** (`stop_conditions.py:792-802` still demotes
   `peak_decline` on a held/rising margin, `test_stop_conditions.py:254` cements it with the
   withdrawn rationale, and v2-monitoring.md carried the old text until today — §9.2 says
   veto-only).
4. **Resume `experiments/run1_vit/checkpoints/last.pt` and run Phase 2 to its planned E14** (~8–9
   epochs). Still the cheapest experiment available, and the quantitative case is unchanged: §1's
   +0.041-mAP extrapolation plus 58% of Phase 2's cosine cycle unspent. **Its decision role has
   narrowed (2026-07-30):** §3/§5 no longer wait on it — it now informs the janitor's timing and
   the cleaning track's value (does continued training alone keep buying mAP?), and it remains a §1
   claim-check. **Added precondition: pin the code SHA and enumerate code drift** — the 2026-07-29
   rotation-mask fix changed the augmentation distribution and the grad-accounting changed the
   logging; epochs 7–14 on today's tree are not clean continuation of 0–6 unless the deltas are
   recorded.

   > **⚠ It is NOT decisive as previously sequenced (corrected 2026-07-30). Six problems; fix them
   > before running it or the result is uninterpretable:**
   > - **Item 1 destroys the baseline this experiment is read against.** Raising `max_val_samples`
   >   (still `30000` at `unified_config.yaml:118`), stopping the val fold, and re-seeding the frozen
   >   macro tag list all change the instrument that produced the **0.674373** bar — *and* removing
   >   266,056 images from training mid-phase changes the run. **Either freeze the instrument for the
   >   duration of this experiment, or do item 1 first and re-baseline. Do not interleave.**
   > - **A resumed V1 cannot be scored on an expanded val set at all** (§8.2/§9.3: only 30,000
   >   images are V1-clean).
   > - **The "~0.69" bar is undocumented and question-begging.** It is *exactly* linear continuation
   >   of the observed rate (0.674373 + 9 × 0.00224 = 0.6945). A saturating fit to the same data fits
   >   **better** (rms 4.3e-4 vs 8.2e-4) and predicts **0.682** — the opposite verdict, from data
   >   that cannot distinguish the two models. **Pre-register the bar and the model before running.**
   > - **"Genuinely flattens" is never defined, and no fail-branch consequence is stated.** §3 and §5
   >   are said to be unblocked "either way", but only the pass branch has a stated implication.
   >   Write the fail branch.
   > - **`clip` is not persisted in `loss_state`**, so applying §4's YAML would change the loss margin
   >   mid-run (both V1 phases logged `clip 0.2`) — which §4 explicitly forbids. `unified_config.yaml`
   >   correctly still reads `clip: 0.2` with a `NOT YET APPLIED` comment; **keep it that way for this
   >   experiment.** Same for `gamma_neg`: the checkpoint-over-YAML precedence inversion (§4) is inert
   >   for `last.pt` (no `loss_state`), but the live `asl_schedule` block
   >   (`unified_config.yaml:421-428`, phase1 `gamma_neg_min: 5.0 / max: 7.0`) contradicts "γ_neg
   >   FIXED at 7.0" and must be neutralized first.
   > - **`last.pt` carries no `eval_history`**, so the stop system starts blind and needs ~6 of the
   >   ~8 remaining epochs before it can confirm anything.
   > - **Name the V1 baseline checkpoint explicitly.** `best_model.pt` is **epoch 5 / val_mAP
   >   0.6722102**, not the E6 0.674373 checkpoint, because selection ran on `val_f1_macro`. §9 never
   >   says which one is "V1".

**Gaps needing an owner before P1 launches** (updated 2026-07-30; none is a research question, all
are scheduling):
a **split budget** (item 1) · a **power/MDE statement per gate** — without it §9's pre-registration is
unenforceable · a **wall-clock and throughput budget** — §6 has no time column, v2-monitoring's
not-to-track list now routes throughput to the run log, and the programme now **firmly contains two
multi-week from-scratch P1 runs** (the janitor's fresh Stage A was confirmed 2026-07-30): at option
C each P1 is ≈1.3e20 FLOPs ≈ **2.5–7 weeks on the 5090 at 25–40% MFU**, before P2/P3 (+~40% FLOPs)
and before human review — budget both explicitly · a **mid-phase abort/rollback criterion** (the LR
guard covers 5 of ~55 epochs and `early_stopping_mode: advise` halts nothing) · an **in-run
observable with the right sign for wrong-positive noise** (§9.2 — there is currently none) ·
**sizing for ~15 unbuilt components** (offline scorer, statAP, pooling harness, xCOLUMNs,
isotonic/SmoothECE, cleanlab, group-aware split, TEST/CALIB carve, weight EMA, **the WSD
scheduler**, LayerScale, RoPE, `GlobalOptimManager`, in-loop macro-F1, artifact hashing) · a
**gold-slice plan with headcount, judgment count and a date** (order-of-magnitude estimate from
the 2026-07-30 review: ~250 tags × ~100–130 pooled images × 2 reviewers ≈ **50–65K judgments**;
consider sharing the janitor's Gemma-assisted disagreement-routing + reviewer-reliability
infrastructure rather than pure double-review) · an **ONNX/export validation step** (the 1024→896
width change re-triggers V1's head-split dynamic-batch repair) · a **seed/reproducibility
statement** (every §9 comparison is single-seed on a run that will be soft-stopped and resumed).

**Decided 2026-07-30, no longer gating:** §3 = option C · §5 = from-scratch · schedule = WSD ·
focal-weight detach = on · janitor Stage A = fresh P1.

**Then run:** §6 Phase 1 → 2 → 3, with §4's loss and §8's instrument, judged by §9's three gates.

---

## Appendix A — ASL telemetry spec

*Carried from the retired ASL_plan §5 (2026-07-02 re-spec), inlined 2026-07-29 before ASL_plan.md
was retired. Context shift: the γ_neg descent is withdrawn (§4, §11), so every "γ step" gate role in
the original is **inert** — including its clean-slice 5→4 gate set and its mid-run Hill/SPLC fallback
trigger (Hill is now a documented gate-failure fallback only, §4/§11 — nothing runs alongside the
shipped loss), and its per-tag-isotonic calibration default (superseded by §8.4). What follows is
the observable set for a fixed-γ run, surfaced by the telemetry-only `ASLDriveManager`.*

| Metric | Definition | Role under fixed γ |
|---|---|---|
| `Δp_mean` | mean(p_pos) − mean(p_neg) | Coarse health only — arithmetically insensitive to γ at 19K tags |
| `Δp_hard` | mean(p_pos) − mean(top-10 non-GT scores) | Boundary-relevant gap. Standing caveat: crushing unlabeled true positives *lowers* top non-GT scores and therefore **widens** it — it can veto, never green-light |
| `pred_pos_ratio` (EPR) | **threshold-free**: Σᵢ pᵢ / expected positives (Cole 2021's formulation, not a thresholded count), per tag-frequency decile, EMA-smoothed | Suppression tripwire. Healthy operating point ≈ 1/(1−ρ) **> 1** (missing positives inflate it). Alarm: sustained >5–10% relative drop in any decile within ~2 epochs of any loss/config change |
| non-GT score histogram | bucket counts of non-GT scores over [0.05, 0.95] per validation pass; watch bands **[0.2, 0.5]** and **[0.6, 0.9]** (second band added 2026-07-30 — straddles the shipping operating point) | Clip-cost observable in [0.2, 0.5]: pile-up + stable EPR = clip too high (log it for a future clip run); pile-up + falling EPR = over-suppression. **[0.6, 0.9] is the overguess watch**: growing mass there = a firing regression forming at the operating point (V1's defect lived above 0.5, where the old band never looked) |
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
[arXiv:2103.17239](https://arxiv.org/abs/2103.17239) · Generalized Neural Collapse (K ≫ d) —
[arXiv:2310.05351](https://arxiv.org/abs/2310.05351) · *Three things everyone should know about ViT*
(MHSA-only fine-tune) — [arXiv:2203.09795](https://arxiv.org/abs/2203.09795) · NaViT, NeurIPS 2023
(aspect-bucketed batching) — [arXiv:2307.06304](https://arxiv.org/abs/2307.06304) · EMA dynamics,
TMLR 2024 — [arXiv:2411.18704](https://arxiv.org/abs/2411.18704) · When/where to average —
[arXiv:2502.06761](https://arxiv.org/abs/2502.06761) · RoPE-ViT, ECCV 2024 —
[arXiv:2403.13298](https://arxiv.org/abs/2403.13298) · QK-norm —
[arXiv:2309.14322](https://arxiv.org/abs/2309.14322) · σReparam, ICML 2023 —
[arXiv:2303.06296](https://arxiv.org/abs/2303.06296) · 4-bit optimizer states —
[arXiv:2309.01507](https://arxiv.org/abs/2309.01507) · Dettmers, 8-bit optimizers —
[arXiv:2110.02861](https://arxiv.org/abs/2110.02861) · Beyer et al. plain-ViT baselines —
[arXiv:2205.01580](https://arxiv.org/abs/2205.01580) · Malladi et al., sqrt LR scaling for Adam —
[arXiv:2205.10287](https://arxiv.org/abs/2205.10287) · Sigmoid bottleneck, AAAI 2024 —
[arXiv:2310.10443](https://arxiv.org/abs/2310.10443) · ML-Decoder, WACV 2023 —
[arXiv:2111.12933](https://arxiv.org/abs/2111.12933) · Scaling laws in patchification —
[arXiv:2502.03738](https://arxiv.org/abs/2502.03738)

**Schedule (WSD, §6).** Hägele et al., NeurIPS 2024 —
[arXiv:2405.18392](https://arxiv.org/abs/2405.18392) · Tissue et al., LR-annealing law —
[arXiv:2408.11029](https://arxiv.org/abs/2408.11029) · MiniCPM (WSD origin) —
[arXiv:2404.06395](https://arxiv.org/abs/2404.06395) · vision precedent: Zhai et al.'s
rsqrt+cooldown "infinite schedule" (Scaling ViT, above)

**Transfer / pretraining (§5 — closed).** Zoph et al., "Rethinking Pre-training and Self-training" —
[arXiv:2006.06882](https://arxiv.org/abs/2006.06882) · Chen & Zwicker, WACV 2022 —
[arXiv:2108.01819](https://arxiv.org/abs/2108.01819) · DAF:re —
[arXiv:2101.08674](https://arxiv.org/abs/2101.08674) · Illustration2Vec, SIGGRAPH Asia 2015
(frozen-feature evidence, not init evidence — §5)

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
losses are not strictly proper — §8.4) · SmoothECE, Błasiok & Nakkiran 2023 —
[arXiv:2309.12236](https://arxiv.org/abs/2309.12236) *(ID and co-author added 2026-07-30)* · Dual-TS —
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
