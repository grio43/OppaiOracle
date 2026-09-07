# V2 Plan — the authoritative document

> **Status:** Phase 1 recipe implemented; dataset completion/preparation and full-scale launch checks remain.
> The active YAML uses E:\Dataset and a fresh 30K holdout after all subsets arrive.
> Phase 1 is limited to 60 epochs including cooldown; Phase 2/3 setup is deferred.
> Operational details: [Phase 1 setup](v2-phase1-setup.md). This file still owns the production recipe.
>
> **Navigation:** [Roadmap and document ownership](README.md) ·
> [Monitoring](v2-monitoring.md) · [Augmentation decisions](v2-augmentation.md) ·
> [Janitor campaign](janitor-cleaning-model-plan.md) ·
> [Gold-training pipeline](v2.1.1-gold-training-pipeline.md).
> Supporting reviews and historical snapshots are indexed in the roadmap; they do not
> override this plan.
>
> **Rule for future edits:** amend *this* file. Do not add another dated correction layer — the
> supersession chain is what let stale summaries revert settled decisions twice already. Any
> amendment to the P1 recipe propagates to
> [`janitor-cleaning-model-plan.md`](janitor-cleaning-model-plan.md) Stage A explicitly — it trains
> by this recipe, so every defect is paid twice.
>
> **Provenance:** created 2026-07-28 consolidating the four-layer plan set; full verification pass
> (52 citations, every `file:line`, every quantitative claim) 2026-07-30; **rewritten clean same day**
> after all open decisions closed. The correction-by-correction audit trail lives in the archived
> pre-rewrite snapshot ([`archive/v2-plan-pre-rewrite-2026-07-30.md`](archive/v2-plan-pre-rewrite-2026-07-30.md))
> and in this file's git history — never cite the snapshot as current.
>
> **Adversarially re-reviewed 2026-08-01** (four pillars × advocate/adversary briefs with fresh
> 2025–26 literature sweeps → three-judge panel → citation-checked verification): **every pillar
> KEEP — no switch**; the surviving amendments are integrated inline below (marked *2026-08-01*).
> Two panel proposals were **refuted in verification and do not ship**: step-0 translation/scale
> jitter (§7, routed to v2.1.1) and the frozen-feature init probe (stays retired per §11 —
> reopening it is a user decision, not a review output).
>
> **DECISIONS — all closed 2026-07-30 (user):** §3 sizing = **option C** (896×18, MLP 2320, ~150M) ·
> §5 init = **from-scratch** · schedule = **WSD** (§6) · loss = **plain ASL γ⁻=7 fixed, focal weight
> DETACHED, single configuration — no experiment arms of any kind** (§4) · **QK-norm ON · RoPE OUT ·
> drop_path 0.20 · warmup 10K** (§3/§6) · split budget **EVAL 30,000 only; no additional CALIB/TEST reservation (user revised)**
> (§8.1) · current baseline comparison = **V1 and V2 on the same existing 30K validation set** (§9.3) · statistics = **lean** (paired
> bootstrap at gate time; no pre-run simulations, no xCOLUMNs) · the **resume-V1-P2 experiment is
> CUT** (§11) · janitor keeps its **fresh from-scratch Stage A** · gold slice **shares the janitor's
> review infrastructure** (§9.4). Micro-decisions recorded inline: Appendix F's assumed-negative
> down-weighting not replicated (§4) · rating tags excluded from telemetry `mean(p_pos)`
> (Appendix A) · plateau-rule window-maxima semantics stands (§8.2).
>
> **The core recipe decisions are closed.** Later augmentation proposals remain pending in
> [the consolidated decision record](v2-augmentation.md); they do not change this recipe.
> Launch readiness is governed by §12 with the **30K-only validation budget**. The larger
> calibration/test allocation and full gold release protocol are deferred (§8.1/§9.3). The
> [2026-09-05 implementation audit](reviews/vit-major-issues-audit-2026-09-05.md) records
> additional correctness/resource findings and confirms several recipe migrations remain
> unapplied. Those dated checks are evidence, not a claim that implementation is complete.
>
> **Verification scope (2026-09-05):** primary-source and implementation review retains
> detached ASL (7, 0, .05), but distinguishes it from the paper's usual (4, 0, .05).
> Remaining missing positives still matter after cleaning. Architecture, statistical,
> calibration and launch claims below have been amended; historical corpus/checkpoint
> measurements remain dated evidence, not freshly remeasured results or causal proof.

---

## 1. Why V2 — the measured case

**V1 stopped while measured validation was improving.** V1 never early-stopped (`grep -c "Early stopping
triggered" logs/training.log` → 0). Both phases ended with a manual `Soft stop engaged` mid-epoch:
Phase 1 at **33/40** epochs, mAP **0.6518**, 81.76% of its cosine cycle (209,899 / 256,720 steps);
Phase 2 at **6/15**, mAP **0.674373**, 41.81% of cycle, LR still at 77.5% of peak. mAP and val loss
improved **monotonically at every logged validation** — all 19 P1 and all 6 P2 events — on a
deterministic val draw (`shuffle=False`, cached list), so the slope is not denominator drift.
This supports finishing the anneal and allowing more training; it does not identify the
noise ceiling or predict the gain from unrun epochs. No linear-extrapolation gain is budgeted.

**Similar aggregate F1, not proof of zero memorization.** Micro-F1 **0.70044** on the 296K
artifact (~90% folded into training) vs **0.70088** on the 30K held out. These overlapping,
noisily labeled sets do not measure a clean train/test gap; micro-F1 can hide per-tag
memorization and confusable-group failures. Use the clean-label evaluation in §9.

**Macro-F1 rose through the supposed ceiling.** Single-threshold macro-F1 **0.5591 → 0.6212
(+11.1%)** from P1-e27 to P2-e7, both on the 296K artifact. Direction and magnitude only: the set is
not held-out, the interval also contains the γ_neg 4→7 change and the 320→448 switch, and the
once-quoted per-tag-oracle pair (0.6218 → 0.6751) is **withdrawn** as a fit-on-eval statistic —
never re-cite it (§8.2).

**V1 got ~230M sample presentations** — about 60% of what DeiT-B (2.9× smaller) uses from scratch.
This is a scale comparison, not a sample-complexity law across datasets and losses.
§6 addresses unfinished training; §9 and §10 address label quality independently.

**P-ASL does not transfer to positive-only annotation.** ~49% of its published gain comes from
γ⁻/γᵘ decoupling, which requires OpenImages' 37.7M human-verified negatives; its Ignore-mode prior
estimator is degenerate at N=∅; and its Ω_P ignore set would select the 107 tags with the *lowest*
missingness and strip their only negative gradient. Details: [review §3](reviews/v2-plan-review-2026-07-28.md).

**Scope fence:** no negative-branch loss knob touches **wrong-positive** noise (a mislabeled
`aqua_hair=1` trains the wrong side at full BCE gradient regardless of γ_neg or clip). That is the
cleaning track's job, permanently.

---

## 2. The plan at a glance

| | V2 |
|---|---|
| **Backbone** | 896w × 18L, patch16, MLP 2320, head_dim 64 (14 heads), CLS-pool (§3, option C) |
| **Params** | ~151.2M including the tag head; adapted from SoViT-150m (§3) |
| **Init** | From-scratch (§5) |
| **Data** | **5,921,102 images** (measured on disk; §10), 19,294 vocab entries (= 19,292 tags + `<PAD>`/`<UNK>`; tables quoting 19,288 also exclude the 4 `rating:*` tags), positive rate ~0.18%, neg:pos ≈ 530–640:1 |
| **Resolution** | 320 → 448 → 512 (unconditional; 512 is the corpus-native terminus) |
| **Loss** | ASL, γ_neg = 7.0 fixed, γ_pos = 0, clip = 0.05, α = 1.0, label_smoothing = 0, **focal weight DETACHED** — single configuration, no experiment arms (§4) |
| **Schedule** | WSD: warmup → constant LR → 1-sqrt cooldown. P1/P2 use the plateau gate; P3 reserves cooldown inside its fixed budget (§6). |
| **Budget** | ~400–700M samples seen as the planning envelope (V1 got ~230M); the gate + cooldown decide the actual |
| **New this run** | LayerScale ε=0.1 · QK-norm · plain dropout → 0 · fp32 optimizer state on the head · bf16/no-GradScaler policy (§3, §6) |
| **Selection** | `val/mAP`, frozen tag list, on **EVAL (30,000)**; binned at `ap_thresholds: 200`; provisional plateau min_delta=.003, window=3 (§8) |
| **Validation budget** | **30,000 images total**, same fixed subset across P1/P2/P3; both training and standalone validation capped at 30K; no extra CALIB/TEST reservation (§8.1) |
| **Decision** | Compare V1/V2 on the same 30K validation set; the full three-gate independent release protocol is deferred (§9.3) |

---

## 3. Architecture — option C (896×18, MLP 2320, ~150M)

**The decision.** Smaller-than-V1 was set as a requirement (excluding A — keep 1024×18 / 248M), and
between the two cuts **C has a shape-scaling motivation**: SoViT-150m's fitted exponents put
s_MLP ≈ 0.60 > s_depth ≈ 0.45 > s_width ≈ 0.22 — at this compute scale the MLP should be
*relatively narrow*. **SoViT §4.1 actually specifies 880×18, MLP 2320, patch14**, trained on
4B JFT-3B presentations and evaluated by ImageNet few-shot transfer
([paper](https://arxiv.org/pdf/2305.13035)). Our **896×18/patch16** adapts that shape to
14 heads of width 64 and the existing image grids; it is not a reproduced optimum for anime.
B (MLP 3584) keeps the conventional 4× ratio and costs more; C remains the chosen compute
tradeoff. A meta-device construction of the current C model at 320px/19,294 outputs counts
**151,181,310 parameters**, before the small LayerScale/QK-norm additions. Wall-clock gains
must be profiled; FLOP ratios do not establish a week saved on the 5090.

**Capacity is an empirical question.** The 19K head (≈17.3M parameters) is identical to B's,
but its inputs come from a narrower MLP. Neither JFT scaling nor V1's aggregate F1 comparison
establishes an 8–30M-image capacity limit. Retain C, then assess underfitting and clean-label
quality as the corpus grows; no guaranteed runway or universal gain-per-image is claimed.

**Forward-compat:** review capacity when the clean-label corpus grows materially. Depth-first
growth at width 896 is an engineering candidate, not a consequence of SoViT (which scales MLP
fastest). The former 8M trigger was a planning heuristic, not a measured capacity boundary.

**Settled, do not revisit:**
- **patch16.** patch14 buys +31% tokens for ~1.36× FLOPs, forces a stem re-init, and breaks the 320
  and 512 grids. Attention stays a minority of compute at every planned resolution — analytic
  estimate at the C shape (not profiled): **~8.9% at 320, ~16.0% at 448, ~19.9% at 512**.
  `512px/patch16` gives N=1025 — identical token cost to 448px/patch14 — with real pixels instead of
  a finer mesh over the same downscaled raster.
- **The linear head stays — on compute economics** (Query2Label/ML-Decoder, §11). Its two former
  citations are withdrawn: *Taming the Sigmoid Bottleneck* was cited inverted (its actual result —
  d ≪ n has exponentially many unargmaxable label combinations — argues the head *could* bind at
  d=896 / n=19,294) and Generalized Neural Collapse is single-label CE with no multi-label claim.
  If head capacity ever looks binding, the sigmoid-bottleneck paper is a reason to look harder.
- **A full per-class Query2Label decoder is impractical at the planned budget** — one query per class at 19,294 classes is ~1.33 TFLOPs/layer
  (~4.3× the whole backbone) and ~10 GB attention memory per image. ML-Decoder measured a full
  decoder head OOM at 9,600 classes.

**Architecture adds (implemented for Phase 1):**

| Add | Setting | Evidence |
|---|---|---|
| **LayerScale** | ε = **0.1** | CaiT Table 1 measures +1.0 top-1 at exactly depth 18 (80.7 → 81.7) vs a drop-path-tuned baseline. DeiT III uses it in every config. |
| **QK-norm** | **ON — decided 2026-07-30** | A modern supervised-ViT recipe measures QK-norm default-ON at 87–449M (removing it costs ~0.12 top-1; "significantly enhances training stability"), and Wortsman's divergence threshold is LR-dependent, not scale-gated. Near-free FLOPs; one mid-run divergence on a weeks-long single-GPU run costs more than the norm ever will. **Log max attention logits from epoch 1 regardless** (Wortsman: *"all points with attention logits above 1e4 diverged"*). |

(2D RoPE is **rejected for this run** — §11.)

---

## 4. Loss — ASL with remaining missing positives

```yaml
training:
  tag_loss:
    gamma_neg: 7.0        # Retained fixed choice; not a proven optimum over 4.
    gamma_pos: 0.0
    clip: 0.05
    alpha: 1.0            # This implementation uses 1 to disable alpha weighting.
    label_smoothing: 0.0
    class_weight_strategy: null
    detach_focal_weight: true
```

**4 and 7 are both paper-backed, in different settings.** ASL Table 2 / Appendix B uses
**(γ⁻, γ⁺, m) = (4, 0, .05)**. Appendix F changes γ⁻ to **7** for OpenImages, treating
untagged labels as negatives **with reduced weights**; other settings follow the COCO
recipe. That supports 7 as an extreme-imbalance choice, not an isolated demonstration
that 7 beats 4 or that reduced weights are unnecessary
([ASL paper, Table 2 and Appendices B/F](https://arxiv.org/pdf/2009.14119)).

**Retain 7 for this single-configuration run.** Our remaining missing positives and very
sparse labels still justify attenuation of the negative branch. Cleaning does not make
absent sidecar tags verified negatives, and the campaign's mined additions do not measure
residual missingness (§10). There is no controlled OppaiOracle comparison establishing
4 or 7 as superior. Moving to 4 would strengthen negative learning, including penalties
on unannotated true positives; it is a plausible alternative, not a correction required
by the paper. Keep γ⁺=0 and clip=.05; no schedules, extra loss terms or parallel arms.

**Released-code precedent and its limits.** RAM explicitly instantiates (7, 0, .05), and
its copied loss defaults to detached focal weights
([call site](https://github.com/xinyu1205/recognize-anything/blob/main/ram/models/ram.py),
[loss](https://github.com/xinyu1205/recognize-anything/blob/main/ram/models/utils.py)).
Its pretrained backbone and additional objectives do not establish the best from-scratch
recipe here. The official ASL repository's ordinary loss defaults to detach **on**, its
optimized loss to **off**; both constructors default to (4, 1, .05), whereas the paper's
main experiment uses γ⁺=0
([official losses](https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py)).
Thus “paper default,” “constructor default” and “OpenImages recipe” are distinct. V2 is
neither a bit-for-bit RAM training reproduction nor the full differentiated Eq. 8 of ASL.

**What the knobs actually do — local analytic check, confirmed by autograd.** For an
observed negative with score p>m, detached focal weights give

`g_neg = (p-m)^γ_neg × p(1-p)/(1-p+m)`; for p<m the gradient is zero.

At γ⁺=0, the observed-positive gradient is `g_pos = p-1`. Holding p and m=.05 fixed,
`g_neg(7)/g_neg(4) = (p-.05)^3`:

| Score p | Negative gradient, γ4 | Negative gradient, γ7 | γ7 / γ4 |
|---|---|---|---|
| .3 | .001094 | .00001709 | .015625 |
| .5 | .018639 | .001698 | .091125 |
| .7 | .107104 | .029413 | .274625 |
| .9 | .313204 | .192346 | .614125 |

These are unreduced per-logit derivatives, not measured recall improvements. Detach
removes the derivative through the focal multiplier and further reduces negative
pressure compared with differentiating the whole loss. Larger γ and detach can protect
missing positives **and** weaken rejection of genuinely absent confusable tags. Positive
margin in negative-branch probability shifting makes its gradient tend to zero as p→1 even with γ4;
changing .05 to .2 would attenuate more, not provide stronger false-positive correction.
The previous “91% missing-label tolerance” and “93% retained separation” claims were
conditional fixed-score calculations, not end-to-end tolerance guarantees; do not use
those percentages as expected performance.

**Reduction and precision are part of the contract.** Our loss averages over batch ×
nonignored labels; the reference implementations sum. This changes gradient scale,
clipping and optimizer-ε interactions, so importing their LR or grad-norm bands unchanged
is unjustified. Retain the local mean reduction, ignored PAD/UNK columns and fp32
probability/log/reduction math under bf16 autocast. Keep class weighting disabled as a
recipe choice; ASL does not mathematically eliminate every long-tail imbalance.

**Formation canary, without a fictitious probability target.** During P1 epochs 1–5,
record supported-tag ranking progress, positive-score distributions, per-decile score
mass, sibling gap and non-GT histogram; interpret trends after warmup. Raw ASL scores
are not calibrated prevalences: never require EPR≈1/(1−ρ) or restart just because it is
large or falling. A constant predictor at observed prevalence .0018 has a stationary
score ≈.307 under detached γ4 and ≈.468 under detached γ7 (solving the expected-gradient
balance above), despite learning no image features. This is a counterexample to an EPR
level gate, not a prediction of trained-model scores. Failure requires persistent lack
of ranking/positive learning, corroborated on reviewed real examples, after ruling out
loader, optimizer and numerical faults. Freeze the observation window and corroborating
criteria before launch; raw-score bands are diagnostics, not autonomous loss controls.

**Restart contingency:** the previously allowed fresh-P1 restart at γ4 remains a
formation-failure contingency after that diagnosis, not a mid-run descent or an automatic
response to EPR. It is **not implemented**: `ASLDriveManager` currently hard-requires 7.
A restart requires an explicit new fixed contract in schema/YAML/manager/regressions,
a fresh experiment and reset monitoring/selection history; changing YAML alone aborts.

**P-ASL selective ignore remains unadopted.** The reviewed Ω_P proposal would remove
negative gradients from high-frequency tags without verified-negative anchors; Ω_L and
SPLC add prediction-based confirmation risks. Missing positives and wrong positives
remain separate problems: negative-branch settings cannot repair an incorrectly labeled
positive. **Gate-failure contingencies** remain Hill, then LL-R/LL-Ct, assessed only after
an informative clean-label failure; an inconclusive CI is not proof that ASL failed.
The fallback evidence is not a demonstrated win over this exact recipe.

**Implementation status:** fixed-loss launch/resume verification and telemetry-only
control are implemented in `ASLDriveManager._reconcile_gamma` / `request_gamma_step`.
Conflicting checkpoint gamma, a manual override, or an enabled schedule aborts. The old
checkpoint-wins precedence and active gamma ladder are no longer outstanding work.

---

## 5. Initialization — from-scratch (closed)

User decision 2026-07-30. The linear-probe tie-breaker was retired un-run; all pretrained routes
(LiGO growth · patient distillation · adopt-ViT-L outright · in-domain MAE/DINO SSL) are closed —
rejection rows in §11.

Kept so the evidence state is not re-litigated from stale premises: **the regime is unmeasured in
both directions** — no published controlled init comparison exists at ≥5M target images / 10K+ tags
for ViT classification (Steiner's +13pp for IN-21k pretraining is at 1.28M target images; the
strongest for-from-scratch result, Zoph's −1.0 AP negative transfer, is detection-only). Two
citation guards: Chen & Zwicker (WACV 2022) and DAF:re **do exist** — the old "no scholarly
evaluation on illustration exists" claim was false — and Illustration2Vec is frozen-feature
evidence, not init evidence. Neither re-opens the decision.

**Standing consequence:** every prediction-dependent mechanism in the literature (P-ASL, Ω_L, SPLC
self-relabel) was measured with a pretrained backbone doing the calibrating — from-scratch
discounts that evidence base by an unestimated amount. All such mechanisms are rejected in §4/§11
anyway; remember this if any is ever revisited.

**Compute anchor:** Phase 1 ≈ 1.3e20 FLOPs at option C — multiple weeks on one GPU (§12). WSD is
the mitigation: no pre-committed epoch count.

---

## 6. Schedule — finish training this time (WSD)

**Target: 400–700M samples seen, as a planning envelope, not a commitment.** For scale: DeiT III's
from-scratch ViT-L benchmark used **~1.02B sample presentations** — >2× this envelope's midpoint —
so the gate, not the envelope, decides; running long is the well-precedented direction.

**The schedule is WSD (warmup–stable–decay), replacing cosine.** Per phase: linear warmup →
**constant LR** through the stable body → plateau gate on `val/mAP` (`stop_conditions.py`'s
window-over-window rule, unchanged) → a **1-sqrt cooldown of ~10–20% of that
phase's elapsed steps** → the phase transition. Why: both V1 phases were hand-stopped mid-cosine
(P1 at 82% of cycle, P2 at 42%), forfeiting the anneal both times. Under WSD the anneal happens
**after** the stop decision — a hand stop costs ~10–20% extra steps instead of the tail — and a run
that wants to go longer just keeps going at constant LR. Evidence: constant+cooldown matches cosine
at equal compute with 10–20% cooldowns (Hägele et al., NeurIPS 2024; Tissue et al. 2024; MiniCPM:
10% suffices, 2.5% falls short) — LLM-scale parity, with Zhai et al.'s rsqrt+cooldown "infinite
schedule" as the supervised-ViT precedent. Two riders: (1) **always run the decay** — no mid-run
device (weight averaging included) substitutes for it; (2) a cooldown can be **branched** off any stable
checkpoint for an annealed readout while the stable run continues — this is also the janitor's
Stage-A branch mechanism (branch *after* a cooldown, never the raw stable checkpoint).
**Phase 1 scheduler implemented:** `WarmupStableDecayLR` persists update position, cooldown start/length/reason and cooldown-best metric. The active config uses a 15% elapsed-update fraction and the user-approved **60-epoch ceiling including cooldown**. Plateau dispatch is disabled until the stable-phase jitter review; budget cooldown is automatic. See [setup](v2-phase1-setup.md).

**Gate-noise pre-flight (added 2026-08-01, judged review).** min_delta 3e-3 (§8.2) was triangulated
from V1 jitter measured under a cosine decayed to ~76% of peak; WSD gates at **100% of peak**,
where window jitter is unmeasured — and V1-P2's real window gains (+0.0015…+0.0031) sit at the
bar. Before the plateau gate is armed in P1: measure stable-phase `val/mAP` window-over-window
jitter on V2's own first ~10 eval events as **residuals about a smoothed trend** (the method the
`stop_conditions.py` docstring already mandates — raw early gains are trend-dominated and would
over-read jitter), confirm min_delta ≥ ~2× the measured jitter, and pre-register the
widened-window value to use if it is not. The pre-flight must also pin ONE (min_delta, window)
pair — use the code's window=3 provisionally, then freeze it before enabling the gate.
A fixed EVAL set removes resampling between epochs, but does not make population-estimation
error cancel exactly: different checkpoints make different errors on those same images.
Distinguish paired evaluation uncertainty, binning error and temporal weight fluctuations.

**Cooldown is a written procedure (added 2026-08-01), not an implementation-time choice:** 1-sqrt
shape `lr(u)=lr_end+(lr_start-lr_end)*(1-sqrt(u))`, u∈[0,1]; fraction picked inside
10–20% of elapsed pre-cooldown updates and recorded *before* cooldown; terminal LR ≈ 0;
checkpoint cadence unchanged; **β₂ during cooldown unchanged — named non-decision** (the
raise-β₂-in-cooldown finding, arXiv:2508.01483, is LLM-scale; declined under single-config).
**Expectation note (mirror in v2-monitoring.md):** annealing can lower loss and improve mAP;
neither change is guaranteed. Do not diagnose failure from the expected difference alone;
the branched cooldown is the sanctioned annealed readout.
The [WSD paper](https://arxiv.org/pdf/2405.18392) defines its cooldown fractions over
**total** steps. Our elapsed-step convention is a local choice: r=10–20% of elapsed
means r/(1+r)=9.1–16.7% of the total; do not call those fractions identical.
Persist warmup/stable/cooldown state and the cooldown start/length on resume.

**Normalization.** An epoch is the **training pool**, not the corpus (`dataset_loader.py:2780-2794`
carves val out first): training pool **5,891,102** = 5,921,102 − 30,000 holdout; **1 epoch ≈ 5,753
steps at eff batch 1024**. (The retired 6,641 steps/epoch figure was 15–21% high — never reuse it.)

| Phase | Res | Epochs (est. — gate decides) | Samples | Eff. batch | Warmup | Base → stable LR | drop_path | dropout | WD |
|---|---|---|---|---|---|---|---|---|---|
| **1** from-scratch | **320** | ~55; gate on plateau | ~324M | ~1024 | **10K steps** | 2.7e-4 → ~5.4e-4 | **0.20** | 0.0 / 0.0 | 0.05 |
| **2** fine-tune | **448** | ~12; gate on plateau | ~71M | ~768 | 2 ep | 1.5e-5 → ~2.6e-5 | 0.15 | 0.0 / 0.0 | 0.05 |
| **3** detail | **512** | 3–4 — run it, don't gate it | ~24M | ~676 | 1 ep | 8e-6 → ~1.3e-5 | 0.10 | 0.0 / 0.0 | 0.05 |

**≈419M samples total ≈ 1.8× V1.** Epoch counts are estimates, not caps — WSD retired the cosine
horizon arithmetic; the 400–700M band is a wall-clock/storage envelope, not a scheduler input.
The table budgets complete phases including cooldown; estimate the stable body accordingly.
Phase 3 uses a fixed 3–4-epoch budget with a reserved terminal cooldown, rather than the
plateau gate used by P1/P2. Choose achievable integer microbatch × accumulation products
after profiling; ~676 is an LR-derived placeholder, not a measured feasible batch size.

**The 320→448 spine has direct in-regime precedent:** ASL itself trains OpenImages (~5,400 classes,
partial labels) at 224, then fine-tunes at 448 — same loss family, same assume-negative regime,
same two-stage shape.

**Phase-transition checklist (every switch, in order):** finish cooldown, then select the
best finite checkpoint **from that cooldown** on the §8.4 selection metric →
interpolate pos-embeds bicubic (`training_utils.py:1990-2047`, implemented; grid sizes derived) →
**reset optimizer state** → update config → re-warmup. `torch.compile` recompiles on first forward.

**Notes on the numbers:**
- **drop_path 0.20 — decided 2026-07-30 (status quo).** The convention mismatch is real and
  recorded: our `torch.linspace(0.0, rate, 18)` ramp (`model_architecture.py:398`) gives mean 0.10,
  while DeiT III uses a **uniform** rate (its ViT-B 0.1 / ViT-L 0.4 are means), so we sit at DeiT
  III's ViT-B level for a model 1.7× ViT-B — nominally under-regularized. Chosen anyway: V1 ran
  0.20 with similar aggregate F1 on its compared sets (§1), and Steiner reports extra reg can hurt at fixed compute
  near our data scale. **Revisit only if P1 runs materially past ~55 epochs** (DeiT III's
  long-schedule rule raises drop_path +0.05 per +200 epochs).
- **Plain dropout → 0.** DeiT III (stochastic depth only) plus Steiner (*"any kind of AugReg hurts…
  for all but the largest models"* at 14M images, fixed compute). Current config: `hidden_dropout_prob:
  0.10`, `attention_dropout: 0.05` — both go to 0.
- **WD 0.05 fixed**, per the standing project decision. (V1 record: P1 ran 0.1, P2 0.05.) Same
  long-schedule revisit trigger as drop_path.
- **Phase-2 LR is not the failure mode.** DeiT III's high-res fine-tune is 1e-5 @ batch 512 × 20
  epochs; our higher LR and different update count are not a controlled comparison. Retain the
  planned LR as a starting recipe; neither citation proves it optimal or rules out LR effects.
- **Warmup 10K steps — decided 2026-07-30.** Beyer et al. use 10K at batch 1024. It is a 61% cut in
  warmup updates vs V1-P1's 4-epoch warmup; the σReparam divergence grid argued for more, and
  **QK-norm (§3, adopted) is that insurance**. Under WSD the warmup hands off directly into the
  constant stable LR.
- **Batch→LR rule: sqrt** scaling for AdamW — stable LR = base × √(eff batch / 256) (Malladi et al.).
  This is a practical heuristic here, not the full SDE-preserving transformation. Keep
  β=(0.9,0.999) and the configured ε fixed; re-apply the LR heuristic whenever eff batch
  is retuned to real VRAM, then verify stability. No universal SGD-only claim is made.
- **Base LR is a /256 base, sqrt-scaled at runtime** — confirmed on disk
  (`unified_config.yaml:244-259`, `train_direct.py:768-777`, and the log's own *"base 2.50e-04 ×
  sqrt(918/256) = 4.73e-04"*). Peak-to-peak vs V1: 5.4e-4 / 4.73e-4 = **1.14×**. **LR guard:**
  grad-norm drift >2× in the first 5 epochs → fall back to base 2.5e-4 (the §12 abort rule covers
  the rest of the phase).
- **Phase 3 is not gated.** ASL COCO 448→640 +1.4 mAP; ML-Decoder +1.1; Query2Label +1.1 —
  consistent across three multi-label papers, and it targets exactly the small-detail tags.
  Measured corpus geometry (3,000-image sample, 2026-07-29): long side exactly 512 for **93.5%**,
  below 512 for 6.5%, above for **0%** — so P2 (448) is a uniform gentle 1.14× downscale, P3 (512)
  is a true native pass, and **there is no rung above 512** (any P4 would be upscaling). Median
  aspect 0.713 → letterbox pad fraction 0.287.

**Considered, not adopted this run (recorded so they are not re-litigated):**
- **MHSA-only fine-tune** at the resolution switches (*Three things…*, ±0.1 accuracy points of full FT at −10%
  memory/time; that source reports classification accuracy, not mAP). Full fine-tune is what every §6 precedent measured — but P2+P3 are ~40% of the
  budget by FLOPs, so **keep it live as the fallback if P2/P3 memory binds**.
- **Serving-preprocessor match to training's two-stage resample** — measured skew ~1.1/255 mean |Δ|
  on-corpus, under the 3.69/255 bar of the already-fixed BILINEAR bug; upstream kernel confirmed
  Pillow LANCZOS. Exact fix recorded (at serving, long side > 512 → resample to 512 then 448) but
  it touches five serving sites + ONNX metadata; not worth it at this skew.
- **Aspect-bucketed batching / NaViT** — worth ~25% throughput (letterbox waste) but changes the
  preprocessing contract shared by ONNX/inference, the flip pipeline and dedup artifacts. Standalone
  throughput project, not this run.

**Optimizer:** AdamW8bit, **except fp32 optimizer state for the 19,294-way head and position
embeddings** (~140 MB via bitsandbytes `GlobalOptimManager`); verify block-wise *dynamic* (not
linear) quantization. 8-bit AdamW is not verified lossless for ViT-from-scratch (measured Swin-T
gap 81.0 vs 81.2 exceeds seed std; Dettmers' paper contains no ViT 8-bit-Adam vision result), and a
19K head at 0.18% positive rate is structurally the sparse-embedding shape that motivated the
Stable Embedding Layer. **No LAMB** — its measured advantage is at batch ≥16K; batch 1024 sits at
the bottom of ImageNet's critical-batch range (McCandlish et al.), where AdamW is not the
bottleneck. (Recorded tension: DeiT III itself trains with LAMB at batch 2048; we import its recipe
while rejecting its optimizer — judged right at batch 1024.)

**Precision policy:** **bf16 autocast, no GradScaler**, fp32 master weights; compute the 19,294-way
loss reduction in fp32 (bf16's 7 mantissa bits are the only real cost, and a 19K-term per-sample
sum is where it would show). Config already runs `amp_dtype: bfloat16`; this paragraph makes it a
decision, not an accident — fp16 loss-scale drift is a documented divergence mode, and no scaler
removes the stale-scale-after-resume failure class on a resume-heavy phase structure.

**No weight averaging — decided 2026-08-01. Weight EMA is cut; no SWA, no model soups.** It was
carried through earlier drafts on two claims, and the first does not survive contact with the
implemented gate. (1) *Gate signal:* `stop_conditions.py`'s plateau rule is
`best(last k) − best(prior k) < min_delta` (`:406-450`, k=3, confirm=2). `best`-of-window reduces sensitivity to downward excursions but also selects upward noise.
Only a constant additive offset cancels exactly; EMA changes variance and lag. (2) Accuracy
gains from weight averaging are regime-dependent, not a verified +0.2 mAP here. The choice
remains no EMA to limit implementation and checkpoint cost, with no claim of equivalence.
Final checkpoints are selected **after** a
cooldown, where LR ≈ 0 and the weights are no longer bouncing — EMA's weakest point — and the
branched cooldown (rider 2 above) already yields a *true* annealed readout rather than an
approximation of one. **Revisit only if** stable-phase `val/mAP` proves noisy enough that the
window-over-window rule cannot confirm (symptom: `plateau` oscillating ok↔tripped across
consecutive evaluations); the fix order is then widen `window` first, EMA second.

---

## 7. Augmentation — unchanged, and defensible

Phase 1 at full strength; Phase 2/3 reduced (legitimate now that P1 actually converges):

| Aug | Phase 1 (320) | Phase 2 (448) | Phase 3 (512) |
|---|---|---|---|
| horizontal flip (no directional-tag swap) | p=0.5 | p=0.5 | as P2 |
| colour jitter brightness / contrast / saturation | 0.30 / 0.20 / 0.08 @ p=0.5 | 0.22 / 0.15 / 0.06 | as P2 |
| random rotation, bicubic (mask rotates too) | ±[2°,8°] @ p=0.50 | ±[2°,5°] @ p=0.30 | as P2 |
| gaussian blur, kernel 3 | p=0.30, σ ∈ [0.1, 1.5] | p=0.15, σ ∈ [0.1, 1.0] | as P2 |
| mixup / cutmix / randaugment / random erasing / hue rotation | **none** | **none** | **none** |

**Label caveat:** flip has four documented chirality exceptions: `left-handed`,
`left-to-right_manga`, `right-over-left_kimono`, and `right-to-left_comic`. The pipeline
performs no directional-tag swap. Name-based exclusion is the pending **D1** proposal in
[the augmentation decision record](v2-augmentation.md), not part of the current recipe.
That record consolidates both reviews, including the later recommendation against P2/P3
rotation widening. The table above remains the adopted per-phase policy.

**Rotation rotates the padding mask** (fixed 2026-07-29, `dataset_loader.apply_random_rotation`;
regression test `test_rotation_mask.py`; the fix is **uncommitted** — HEAD still has the static
mask and the test is untracked; committing it is a §12 item). The old canvas-only rotation content-dropped ~1.5% of
token slots at Phase-1 rates — too small for a visible mAP delta, fixed because it was an
uncontrolled confound under the loss measurement. Edge clipping (~13 px/side at 5° for the median
aspect) is the intended rotate-in-place semantics, not a residual bug. **Translation jitter not
adopted** (unmeasurable at current val size, and a confound on the ASL measurement).

**The mixing exclusion is a defensible choice, not a well-supported one.** Plain Mixup under
partial labels measures **−12.9 mAP at 10% known labels** (unrefereed preprint, sole quantitative
support; note its highest-class-count row, VG-200 at 90% labels, flips to +0.2). CutMix's
area∝semantics assumption is undefined for global anime tags (`1girl`, style/meta). Hue rotation is
excluded because hue is *categorical* for anime — ~20° can turn `blue_eyes` into `green_eyes`.
Honest counterweight: the accurate claim is "expected gain small, interaction risk with partial
labels large," not "augmentation cannot help at scale" — and DeiT III's own IN-21k (14M) column
uses CutMix + ColorJitter + 3-Augment, a recipe strictly richer than ours.

**Scale/crop question CLOSED 2026-08-01 (judged review).** The scope defect is confirmed:
Steiner's every arm (including "none") sits on an always-on Inception-style scale-jittered crop,
DeiT III retunes crop to SRC rather than deleting it, and both scholarly anime-domain taggers keep
mild crop/translation — so this recipe sits *below* every recipe in its own evidence base, and the
Steiner quote licenses removing *extra* AugReg, not all geometric variation. Step-0 adoption of
the two label-exact jitters was adjudicated and **refuted in verification**: no identified deploy
pathway for the feared geometry brittleness (serving shares the identical deterministic letterbox;
V1 ran the same rigid pipeline without a documented incident). The old 0…+.005 benefit
range was speculative and is not uniformly below .003; paired uncertainty must be measured.
Keep the scope decision without claiming the effect is unmeasurable.
**Disposition: ship this table as written; letterbox translation jitter and
downscale-only scale jitter (s∈[0.75,1.0], P2/P3) are routed to v2.1.1 BY NAME — never introduce
either mid-run.** No-regret monitoring riders (added 2026-08-01): add the luminance-coded pairs
(`white_hair`/`grey_hair`, `dark_skin`/`pale_skin` — audits the already-live brightness/contrast
jitter) and the letterboxed/border tag group to per-tag-group validation monitoring.

**If a mixing method is ever wanted:** SpliceMix (but 2×2 at 448 shows each source at 224 —
attacks our resolution-sensitive confusables) or LogicMix (gated on a per-image unknown-label
mask). **BalanceMix rejected** (§11).

---

## 8. The instrument — launch blockers

**Complete the instrument before relying on §9.** The historical 30K aggregate CI (~±.003)
is not a universal detection threshold. Compute paired uncertainty for the actual comparison;
rare per-tag support remains limited. Units: .003 mAP means 0.3 percentage points.

### 8.1 Split budget — 30K only, revised by user

**Do not hold out additional images.** The new `E:\Dataset` corpus gets a **fresh,
deterministic 30,000-image validation set**, per the user's V2 setup decision.
Build it once, after Danbooru and the additional top-level subset are ready, using
`tools/prepare_v2_dataset.py`; all other images train. The old V1 split lists were
removed, without deleting source images. Corpus/vocabulary counts are pending the
combined-data preparation; old 5,921,102/19,294 figures elsewhere are V1 evidence.

**Keep this same holdout through all three phases.** A resolution switch does not create
a new validation draw or enlarge the reserved pool. The planned configuration is:

```yaml
data:
  max_val_samples: 30000   # In-training validation and total reserved holdout.
validation:
  max_samples: 30000       # Standalone validation cap.
```

The former EVAL 146,056 / CALIB 100,000 / TEST 50,000 allocation is **deferred**, not an
instruction for the next launch. Do not carve extra sets or expand duplicate clusters into
additional held-out rows without a new user decision. CALIB/TEST-dependent procedures in
§8.3/§9 remain future methodology; those sets are not currently provisioned and validation
must not be described as an independent final test set.

- [x] Cap is the actual holdout budget; unused candidates return to training.
- [x] Preserve train/validation disjointness, including worker row-index mapping.
- [x] Implement membership/label hashing and source-sidecar snapshots in preparation; generation awaits the additional subset.

The fp32/bool validation buffer is about **2.89 decimal GB** at 30,000 × 19,294.
Bucket diagnostics use bounded column chunks. A larger holdout is not a launch prerequisite.
The prior support-floor caution still applies to any future calibration proposal: recount
distinct-image support and use fallbacks for unsupported tags rather than assuming a sample
budget guarantees minimum per-tag support.

### 8.2 Metric correctness
- **Binned AP: keep `ap_thresholds: 200` for routine validation**, subject to the actual
  pass's memory check. Increasing threshold bins is still binned AP; it is not exact AP.
  The historical ~−.003 discrepancy is not a universal bias or cancellation guarantee:
  models with different score calibration have different quantization/tie errors.
  Use the same estimator for V1 and V2. At gate time, compute exact per-label AP in bounded
  column chunks as a sensitivity check; do not infer a ranking win from a bin-sensitive delta.
- **Stop threshold .003 is provisional practical significance, not an instrument constant.**
  A single-model sampling CI, systematic binning discrepancy and temporal jitter are different
  quantities. Validate the chosen `(min_delta=.003, window=3)` against §6's stable-phase
  observations and freeze before arming. Best-checkpoint tracking keeps its near-zero threshold.
  Window maxima remain the chosen statistic. Only for a linear trend is `min_delta/window`
  an approximate per-epoch slope bar; raising min_delta makes a plateau easier to trigger,
  so the old claim that this threshold necessarily errs toward longer training was wrong.
- **Frozen macro tag list — DONE** (`validation.freeze_macro_tag_list: true`; survives soft stops).
  Note precisely: the frozen list is the **first-epoch supported set** (18,334 of 19,292 on the 30K
  draw — recount on the frozen launch labels), not all tags; that is correct behaviour and also the
  denominator that reconciles headline and bucket mAPs.
- [ ] **Per-decile mAP** un-gated from `use_tensorboard` (deciles from `vocabulary.json`; assignment
      exists at `asl_telemetry.py:331-348`, per-group mAP at `evaluation_metrics.py:491-493`).
- [ ] **Per-tag-optimal macro-F1 in the val loop — log as ORACLE only, never select on it.**
      `grid.max(axis=0)` argmaxes per column on the evaluation data — a fit-on-eval oracle
      (measured inflation **+0.054**: 0.6751 oracle vs 0.6212 single-threshold on the same
      checkpoint; and the same statistic reads 0.6410 on 30K, so it is not comparable across eval
      sizes). Useful as an upper bound; never a gate, selection metric, or headline.

### 8.3 Thresholding and calibration
**Deferred methodology under the current 30K-only policy (§8.1).** CALIB/TEST are not
provisioned. The procedure below describes a future independently calibrated comparison;
its extra reservations are not current launch blockers. Until then, name metrics from the
30K subset as validation results, not independent calibrated test estimates. Any fitting
on those same labels must be disclosed and cannot also be reported as held-out performance.

**Fit calibration after training, using CALIB only.** Raw ASL sigmoid outputs are scores,
not true-label posteriors. Neither .7927 nor a theoretical fixed-score region is a V2
threshold prescription. Fit thresholds to a declared deployment objective, with ≥50
CALIB positives for per-tag thresholds and ≥100 for per-tag Platt; otherwise back off.
Support counts must come from the frozen CALIB labels, not old 30K measurements.

- [ ] **Calibration stack:** global isotonic fallback, decile-level isotonic where supported,
  and monotone per-tag Platt shrunk toward the decile map for eligible tags. Pick a single
  composition/shrinkage rule with internal CALIB cross-fitting; refit on all CALIB and
  freeze before EVAL/Gate 2. Low-support per-tag isotonic stays excluded.
- **Sampling must match deployment.** A calibrator fitted only to top-k scores estimates
  correctness conditional on top-k selection; it cannot be assumed valid for a product
  emitting every score above a threshold. Fit/evaluate over the threshold-emission candidate
  distribution, retaining a weighted random background sample and its inclusion probabilities.
  Keep top-k reliability as a separately named diagnostic. All-cell ECE can be dominated by
  trivial negatives; report emitted-tag reliability and positive/tail coverage too.
- **Noisy-label calibration targets observed annotations.** Without reviewed real labels or
  identified missingness assumptions, it does not recover true correctness probabilities.
  Use disjoint reviewed calibration data for any true-label probability/precision claim;
  GOLD-SLICE gate judgments cannot also fit thresholds. Synthetic Anima checks are probes.
- [ ] **ECE estimator:** distinguish kernel-based SmoothECE from equal-count binned ECE;
  they are different estimators. Freeze estimator, score population and weighting for both
  models. The text-XMC study's cross-fit ECE@1 improvement is precedent, not our expected result
  ([XMC calibration](https://arxiv.org/abs/2411.04276),
  [SmoothECE](https://arxiv.org/abs/2309.12236)).
- **Standing rule: nothing is ever fitted on the reported split.** Thresholds fit on noisy labels skew high
      (§9.1); before shipping, cross-check per-group thresholds against the Anima slice (probe, not
      anchor — §9.4).

- **Before an independently calibrated release comparison:** implement scoring, support fallback and calibration; fit final maps only after model selection. Gate 2 checks firing errors at
  each model's fitted operating point, and Gate 3 checks true-label behavior.

(xCOLUMNs is **rejected** — §11.)

### 8.4 Support floor — selection and reporting are different metrics

**V1 evidence (P1, last logged validation): the tail is genuinely learned.**

| bucket | tags | mean support (30K val) | mAP | f1_macro @ 0.2653 |
|---|---|---|---|---|
| 500–999 | 7,619 | 2.45 | **0.6250** | 0.0279 |
| 1,000–4,999 | 7,873 | 8.52 | **0.6345** | 0.0421 |
| 5,000–9,999 | 1,482 | 30.3 | 0.5713 | 0.0567 |
| 10,000+ | 2,314 | 399 | 0.5809 | 0.0782 |

Random ranking at 2.45 positives gives AP ≈ 0.00042; the observed 0.6250 is ~1,500× that and rose
monotonically — tail tags inherit representations from head-tag examples and learn a small delta
(the reason 500 examples suffices in a 19K-way setting, and the same reason Tencent ML-Images and
RAM work). Per-tag AP at tiny support has high variance and finite-sample bias; missing annotations
add another bias. Bucket averaging reduces some variance but does not remove either bias
or dependence between tags. Treat this as evidence of learned ranking on observed labels,
not a certified true-label tail score. The tail's *lower F1* at the hardcoded
θ=0.2653 vs *higher mAP* is a fixed-threshold artifact (measured F1-optimal θ ≈ 0.76–0.81): V1's
manual stop was decided watching a number with no relationship to model quality (~0.013 — the
logged `val/f1_macro` at θ=0.2653 over the all-tag denominator, hence lower than the frozen-list
figures in the table above).

**Consequences:**
- **Selection metric: macro-mAP over the frozen list, no support floor** — averaging thousands of
  terms gives tail tags a voice; uncertainty and label bias still need reporting. *(Why a macro metric: under total tail collapse macro-F1@5
  drops 88.5% while P@5 drops 8.7% and PSP@5 18.4% — NeurIPS'23 Table 1; macro-averaged metrics are
  the only panel members that move when the tail fails.)*
- **Reporting: per decile, all tags** (reliable at decile granularity on EVAL). **Fix the
  occurrence-inflating counter (§10) before freezing `TAGSET`** — decile membership is currently
  biased along a category axis.
- **Support floor ≥50 applies to per-TAG operations only** (per-tag thresholds, per-tag Platt ≥100,
  any per-tag claim); below it, fall back to the tag's decile-level operating point.
- [ ] **Add the minimum-support floor to `_calibrate_per_tag`** (`evaluation_metrics.py:605-626`
  has only a zero-support fallback; `tools/find_pr_threshold.py` has the `--min-support` pattern).
- Gate 3 is where per-tag tail reliability is actually established — its slice is stratified with
  confusables oversampled, the only affordable way to measure the tail per tag.

| support floor (val positives; measured at the old ~276K-scale draw — re-derive at the final split sizes, §8.3) | tags kept | % of vocab | % of label mass |
|---|---|---|---|
| none | 19,288 | 100% | 100% |
| ≥25 | 16,556 | 85.8% | 99.3% |
| **≥50** | **10,074** | **52.2%** | **96.9%** |
| ≥100 | 6,226 | 32.3% | 93.9% |

**Not recommended:** raising `vocab_min_frequency` 500 → 2000 would drop 62.6% of tags for 5.1% of
label mass — tag coverage is the product; leave the floor at 500.

**Hygiene backlog (non-blocking):** ghost keys `num_groups`/`tags_per_group` filtered as unused
(`train_direct.py:617-622`; still in config at `:52-53`); `use_style_token`/`num_special_tokens`
are not real keys; `validation_loop.py` 3D handling vestigial; `loss_functions.py:45`
`label_smoothing` default 0.05 → fix to 0.0; dead code `log_index_order_hash`,
`LearningRateSchedulerFactory`, `TrainingMetricsTracker` (referenced only by `__main__` self-test).

---

## 9. How we decide V2 is actually better

### 9.1 Missing labels can distort the comparison in either direction
For a missing-positive tag, a correct high score is counted as a false positive against
sidecars. This can penalize improved concept recognition, but **AP bias is not universally
downward or monotone in model quality**: deleting positives also changes AP's denominator
and can remove hard positives. Example with descending scores: true labels `[1,0,1]` have
AP=5/6; hiding the last positive gives observed `[1,0,0]`, AP=1. Hiding a top-ranked positive
can instead lower AP. These are exact finite examples, not a model of corpus missingness.

Campaign additions suggest heterogeneous omissions, particularly confusables, but do not
identify residual ρ (§10). Neither a noisy-val decline nor an increase proves the model
outgrew the annotations. Log noisy-label metrics, then adjudicate with independently
reviewed real labels; do not tune ignore/relabel mechanisms to the gate judgments.

### 9.2 Sibling-negative evaluation — veto only, never a green light
Restricting evaluation to confusable-group cells where exactly one sibling is *labelled* gives a
cheap boundary observable on the hardest tags — but it is **not** a clean-label arbiter, and its
sign is wrong for exactly one noise mode: a model that has **memorized a wrong positive**
(confidently agreeing with a mislabeled `aqua_hair=1`, suppressing the true `green_hair`)
**maximizes** the gap. `confusable_groups.json`'s own header says colours are not exclusive and the
filter does not exclude a present-but-unlabelled sibling — the dominant failure on these tags
(historical mined addition ratios are large here, but do not identify ρ; §10).

**Standing rule: the sibling gap can veto, never green-light.** A falling gap corroborates a stop
signal; a rising gap must never silence one. The implementation and regression now enforce
this rule: holding/rising margins cannot demote `peak_decline`; falling margins may corroborate
it. It remains a boundary observable and feeds Gate 3's off-diagonal metric.

### 9.3 Current comparison and deferred three-gate protocol
**Current launch:** freeze the actual TRAIN/30K EVAL membership and label snapshots,
vocabulary/TAGSET and V1 baseline hash. Evaluate both models on the same 30K subset with
the same scorer and report paired validation deltas. This is one validation set used for
selection and comparison, not an independent final test set.

**Deferred scope:** the remainder of this section is the future release-comparison protocol.
The larger EVAL, independent CALIB/TEST and their gates below are deferred by §8.1;
do not reserve additional rows or claim those gates passed without their data.


**If the full protocol is later adopted, before its P1 freeze and hash:** image/group manifests and label snapshots for `EVAL` /
`CALIB` / `TEST`; vocabulary and `TAGSET` support/decile policy; the **gold sampling protocol**,
random seed, candidate universe, tag strata, adjudication rules and estimator (§9.4).
The supported macro list can be derived from fixed EVAL labels before training; verify the
trainer's first-epoch list agrees. Freeze V1's model/vocab/preprocessing hashes as well.
The **final two-model GOLD-SLICE cannot exist before V2 predictions**; construct and hash
that realized pool after both checkpoints are fixed, before revealing judgments or scoring.

Use one offline scorer. Historical V1 baseline: `experiments/run1_vit/checkpoints/last.pt`
was P2-E6 / .674373 on 30K; `best_model.pt` was E5 / .6722102 selected by F1. These mutable
paths are not pins: copy/resolve the intended artifact and record its hash before use.

**The V1-contamination fact that shapes everything:** V1 trained on 89.9% of the 296,056 holdout
(the fold, §8.1); only the 30K subset is V1-clean, and no large clean V1 baseline can be
constructed after the fact — this includes every `pr_threshold_*_full296k` artifact on disk: none
is held-out, never quote one as such. **Decided 2026-07-30 — option (b) with (a) confirmatory:** score both
arms on **EVAL**, reporting V1's exposure explicitly. Its EVAL score is potentially optimistic, **not a
mathematical upper bound**; a V2 win is encouraging but does not remove the confound.
**Confirm on the 30K V1-clean subset** with a paired CI; its historical ±.003 single-model
CI does not determine the uncertainty of a paired difference; **Gate 3 carries the decision**. (Option (c), retraining a V1-equivalent, rejected —
a second multi-week run for a baseline.)

**Gate 1 — ranking quality on the noisy set.** Macro-mAP over `TAGSET`, per decile, on EVAL —
**binned-vs-binned, same `ap_thresholds` for both arms** (§8.2). **Paired group-level bootstrap** (same resampled image/duplicate clusters for both models)
CIs computed at gate time from the cached prediction matrices (never independent tag×image cells; lean decision 2026-07-30 — no pre-run power simulations, the measured paired CI at gate time
is the operative resolution). *Pass:* the paired 95% CI for aggregate ΔmAP=V2−V1 lies above 0,
confirmed in sign on the 30K subset; report its paired CI too. Per-decile CIs are descriptive
unless multiplicity is handled in pre-registration, rather than a search for winning deciles.
**Bottom-decile regressions require investigation**; missingness is one possible cause,
not automatic exoneration. Gate 3 adjudicates. A CI including zero is inconclusive.

**Gate 2 — deployment quality at the shipping operating point.** The product ships
threshold-based (`prediction_threshold: 0.7927`, `top_k: null`). Calibrate each arm on CALIB by the
**same procedure** (§8.3), re-derive each arm's operating threshold, then report on EVAL at that
operating point: macro-F1, precision, recall, **`mean_active`**, false tags per image and
per-decile reliability on the declared candidate population. *Pass:* no material macro-F1
or precision regression under pre-registered paired tolerances, with reviewed-real recall
and false-positive burden checked at Gate 3. **More emitted tags is not itself a failure**:
recovering missing true tags should increase `mean_active`. Investigate extra emissions
using true-label precision/false positives, rather than cap tag count at V1's level.

Strictly increasing per-tag calibration preserves **exact** AP. Isotonic maps can create
ties; uniform-bin AP can change even under a strictly increasing map. Score ranking
quality on the original scores and deployment quality on the calibrated outputs.

**Gate 3 — the arbiter.** Estimate AP over the declared candidate universe from the shared
sampled judgments; do not compute ordinary AP only over the enriched pool. Primary
statistic: macro-AP with tag-stratum weights restoring the declared target tag population
(or explicitly limit claims to the sampled tags). Require paired 95% CIs for V2−V1 to lie
**above zero both overall and in the predeclared tail stratum**. Report true-label precision,
recall and false-positive burden at frozen operating points with design-aware uncertainty.
Also report the off-diagonal metric (canonical spec: v2.1.1 §4).

**Interpretation:** require informative positive aggregate and tail-stratum gold deltas
for the superiority claim, with the statistic, weighting, confidence level and margins
pre-registered. A CI spanning zero is **inconclusive**, not proof of a head-only gain or
a failed loss. Extend judgments only under a predeclared fixed extension or sequentially
valid inference rule (no repeat-until-significant sampling), or withhold superiority;
do not automatically change loss. A significant clean-label regression invokes diagnosis
and the §4 fallback review. Report paired uncertainty conditional on this single training
seed; it does not estimate training-seed variability.

**TEST is scored exactly once, after all gates and all decisions, for the shipping report.**

**Accepted limitations, recorded (lean decision):** every §9 comparison is single-seed on a
soft-stop/resume run; no in-run observable has the right sign for wrong-positive noise (§9.2 —
wrong-positive regressions are caught offline at Gate 3, not mid-run); statAP's ratio estimator is
not guaranteed unbiased at Gate 3's per-tag relevant-set sizes (~5–20) — report its variance
estimates and treat single-tag tail verdicts with caution rather than pre-simulating.

### 9.4 The gold slice — start it now, it is the long pole
Exhaustive 19K annotation is impossible and unnecessary. **Pooled stratified adjudication:**
- ~200–300 **tags**, stratified across frequency deciles, oversampling confusable groups.
- Per tag, pool = V1's top-N ∪ V2's top-N ∪ a random sample from a **predeclared held-out
  EVAL candidate universe**, with **recorded inclusion probabilities**, including overlaps.
  Exclude CALIB/TEST, V2 training and janitor TUNE/CAL images. Track V1 exposure and report
  the V1-clean stratum; if too small, state that comparative generalization remains uncertain.
  Two-system pooling reduces preferential coverage (standard TREC methodology; the bias-controlled *slice* idea is Schultheis KDD 2022
  — the two-system pooling design is not from that paper).
- Adjudicate **double-reviewed with measured κ** — reviewer reliability first (§10): the measured
  2.2× inter-reviewer spread on identical candidates would otherwise dominate everything downstream.
  **Decided 2026-07-30: share the janitor's review infrastructure** (Gemma-assisted screening with
  disagreement → portal routing) rather than building a parallel pure-double-review pipeline.
  Order-of-magnitude: ~250 tags × ~100–130 pooled images × 2 reviewers ≈ **50–65K judgments**.
- Before training freeze the sampling **design**, not the V2-dependent top-N images.
  Keep nonzero sampling probability outside both top lists. The estimator and its variance/
  paired resampling must match the sampling design (including strata and duplicate clusters),
  not an ordinary unweighted bootstrap over the enriched pool. A small uniform background
  sample may find almost no rare positives: report effective support and uncertainty of the
  recall denominator, and do not promise per-tag recall precision from 100–130 judgments.
- Estimate with **statAP** (Aslam & Pavlu) or **infAP** (Yilmaz & Aslam, CIKM 2006) / **xinfAP**
  (Yilmaz, Kanoulas & Aslam, SIGIR 2008), with variance estimates.
- Record an **ε_gold** for the slice itself. A screened-not-adjudicated slice cannot certify.
- **The Anima synthetic set is never the arbiter** — it measures model-precision-on-synthetic-data;
  probe only. (The edits.db verdict store is ~705K verdicts, all on synthetic Anima IDs; the live
  portal db was wiped 2026-07-27 — 58,207 predictions, 0 verdicts — so the shared-infrastructure
  route starts from zero real-image verdicts.)

**Gold adjudication controls the full-protocol superiority claim.** Freeze its design early;
the realized two-model pool and final judgments necessarily follow fixed V2 inference.

---

## 10. Data — re-scoped, not de-scoped

**Corpus: N = 5,921,102, measured 2026-07-30** and consistent four independent ways (live
enumeration of `L:/Dab/Dab` — the only configured root; the split-file headers; the Arrow cache
meta; the training log). Holdout 30,000, training pool 5,891,102 (before exclusions). If the corpus grows before
launch, re-derive §6/§8 arithmetic and the split carve from the new count — do not run mixed
figures. **The retired "≥6.8M" figure was an occurrence-count artifact** (`vocabulary.py:149`
counts tag *occurrences*, and ~80% of sidecars duplicate trailing meta/character/copyright tags —
`highres: 6,773,251` exceeding the file count was the tell). Never infer corpus size from tag
counters again.

**Companion defect, still live:** `vocabulary.json` frequencies are occurrence-inflated ~1.6–1.9×
on meta/copyright/character tags but exactly 1.00× on general tags. Training labels are unaffected
(binary vectors), but **decile membership is biased along a category axis**.

- [ ] **Fix the occurrence counter before freezing `TAGSET`** (§8.4, §9.3).

**Label completion is a priority, not evidence that more images cannot help.** Continue
fixing omissions and wrong positives, especially confusables. Published data/annotation
scaling results are regime-specific; they do not establish a universal “10× images = 1%”
ceiling or a quantitative ranking of our next data investments. **Expect substantial
residual omissions after this campaign**, as the user reports. Freeze the post-cleaning
training snapshot and remeasure its distinct-image frequencies before production P1.

**Review-budget retargeting, highest value first:**
1. **Reviewer reliability before scaling review.** The 2.2× spread on identical candidates
   (measured: four sessions over the same 743 FP candidates converted at 28.8/53.7/56.8/62.7%)
   means throughput currently buys variance. Rubrics + adjudication + measured κ; everything
   downstream (gold slice, thresholds, gates) inherits this.
2. **Spend review on the confusable groups.** Wrong positives are the share no loss can address
   (§1 scope fence). Method of record: Confident-Learning multi-label detect-and-remove, gated on
   the gold slice and human-reviewed — never auto-remove on rare adjacent tags; HLC only ever as a
   colour sub-vocabulary pilot (unproven >80 classes).
3. **Retarget routine review from well-learned head tags.** Historically 76% of adds went
   to 14 frequent tags with low mined addition ratios. Preserve probability-sampled head
   coverage for an honest audit; mining yield does not identify residual missingness.

**Campaign yield is measurable; residual missingness is not yet identified.** The historical
fit on 123 usable, model-mined campaign tags was `0.50 × freq(c)^−0.365` (R²=.27), with
reported bucket ratios .47 / 1.59 / 3.30 / 5.11%. These are historical discovery statistics,
not post-cleaning population ρ estimates. Define true missingness as
`ρ_c = P(observed label=0 | true label=1)`. If A previously missing positives are found
against P correct observed positives, A/P is an addition ratio; even exhaustive discovery
would give ρ=A/(P+A), not A/P. With incomplete mining, category-dependent counter inflation
and unmeasured wrong positives, no unconditional ρ estimate follows.

Measure residual omissions on probability-sampled, reviewed **real** images with an
explicit denominator and uncertainty, separately by frequency/confusable group. Until then,
do not insert the fitted ratios into prior correction, expected EPR, or loss guarantees.
Keep train-only corrections versioned; adjudicated gate labels stay held out from fitting.

**Measurement caveats on the campaign's own numbers:** the 4.9:1 missing:wrong ratio compares
pools mined at 0.155% vs 69.6% conversion (~450× difference) — it measures review budget, not
noise composition; and the FN pool is model-defined, so memorized wrong positives are structurally
invisible to it — treat the wrong-positive share as a floor.

---

## 11. Explicitly rejected, with the reason

| Rejected | Why |
|---|---|
| **Resume-V1-P2 experiment** (run `last.pt` E6→E14) | **CUT 2026-07-30 (user).** Since §3/§5 closed it gated nothing; it needed six precondition fixes (instrument freeze, pre-registered bar, pinned SHA, …) or the result was uninterpretable — and its own analysis showed the data cannot distinguish linear continuation (→0.694) from saturation (→0.682). V2 and the janitor's Stage A answer "does more training keep buying mAP" anyway. |
| **2D RoPE** | **Rejected this run 2026-07-30.** RoPE-ViT's +1.4/+2.5 gains are zero-shot resolution *extrapolation* on models never fine-tuned at the target resolution — exactly the degradation §6's re-warm-and-fine-tune recipe already repairs, so expect materially less here; and it changes the ONNX path, requiring a prototype (testing this plan no longer carries). Revisit only for a future zero-shot-resolution product need. |
| **xCOLUMNs / budgeted-at-k optimization** | **Rejected 2026-07-30.** Product-mismatch: it assumes ‖ŷ‖₁ = k while the product ships threshold-based; at the nearest label-space scale it evaluated (AmazonCat m=13,330) it *loses* to TOP-K+wPOW, with the failure attributed to labels with <10 positives (65.8% of our vocab at 30K); its consistency guarantee is asymptotic (the bound's m²-term is ~1e9 for us); and it returns a randomized classifier. Revisit only if a top-k product ships. |
| Propensity-scored losses / metrics | The unfitted A=0.55/B=1.5 constants are unusable (unnormalized PSP@1 = 326%) and we have no fitted propensities. Honest scope: Schultheis' Table 4 shows most *fitted* propensity-trained variants beat doing nothing (the once-quoted 48.58% was the single degenerate outlier, not the verdict), and the best variant fits propensities from a bias-controlled set — exactly what the gold slice supplies. **Revisit after the gold slice exists; not before.** |
| ML-Decoder / attention-pooling head *this run* | The gain does not grow with class count (80→9,600 classes: +1.1 → +0.7; K=100→400 queries buys +0.1); queries carry no class semantics; CaiT Table 2 from-scratch at matched capacity: attention pooling exactly ties average pooling. All its gains used OpenImages-pretrained backbones — the inverse of our regime. If revisited: K≈100 into the existing linear head, after the instrument work. |
| **Pretrained initialization — any route** (probe, LiGO, distillation, ViT-L adopt, in-domain SSL) | User decision 2026-07-30: from-scratch (§5). LiGO note for the record: trivial depthwise stacking dominates the learned operator — if growth is ever revisited, stack depth-first. |
| **Experiment arms of any kind** (loss arms, init bake-off, mid-run comparisons) | User decision 2026-07-30: V2 trains a single configuration. Fallback ladder (Hill → LL-R/LL-Ct) is invoked only by §9 gate failure, never run alongside. |
| **BalanceMix** | Unproven above 80 classes; needs a warm-up model whose confidence is exactly what's unreliable from scratch at 0.18% positive rate; single-configuration rule. (Its noise modes do include missing positives — recorded fairly.) |
| **Consistency regularization — Mean Teacher / FixMatch / UDA second loss term** | **Declined 2026-07-31 (user): no teacher model.** A teacher forward pass adds compute and a second loss term beside the locked single-config ASL. Production weight EMA is also **cut** (§6, 2026-08-01); the historical review discussion is not a surviving EMA requirement. Full mechanism record: [augmentation round 1 §4.5](reviews/v2-augmentation-review.md). |
| RAL / APL | Documented context / next-loss-family fallbacks in the retired ASL plan; not adopted. |
| Schultheis & Babbar unbiased-BCE estimators | Require identified missingness assumptions; the mined-campaign fit in §10 does not supply them. Not adopted. |
| COMIC | Strongest rejected joint long-tail + missing-label framework — setting-mismatched. |

**Also rejected, argued upstream:** γ_neg descent (§1/§4 — formation-failure carve-out
2026-08-01: the §4 canary's pre-registered restart-at-γ4 is a written *restart* path, not a
descent or any mid-run γ movement; the rejection stands for all in-run changes) · P-ASL Ω_P / Ω_L · full-corpus prior
pass (§4) · per-tag isotonic at 19K (§8.3) · Query2Label (§3) · LAMB (§6) · naive
mixup/cutmix/randaugment (§7) · weight EMA, SWA, model soups (§6) · patch14 (§3). **Deferred with
reasons at §6:** MHSA-only fine-tune (fallback if P2/P3 memory binds) · NaViT aspect-bucketed
batching (standalone throughput project).

**Citation guards (kept so bad cites don't recur):** `arXiv:2501.02364` is intrinsic-dimension, not
class count. "Depth Delusion" is a language-model study. FixRes = "train low-res, fine-tune at
target" only (no RandomResizedCrop discrepancy in a letterbox pipeline). Northcutt establishes
noisy-val mis-ranking is *possible*, not that our regime is past the threshold. The oracle macro-F1
pair 0.6218→0.6751 is fit-on-eval — never quote it (§1, §8.2).

---

## 12. Order of work

**1. Implementation — complete and verify before a V2 or janitor Stage-A launch.**
The July/August reviews recorded unapplied recipe changes; the September audit confirms
several are still outstanding and adds correctness findings. Do not infer completion from
passing legacy-policy tests or from loading the current YAML.

**Recipe and evaluation implementation, in dependency order:**

- **Fixed-loss conformance — implemented:** launch/resume checks effective γ_neg=7,
  γ_pos=0, clip=.05, alpha=1, smoothing=0, detached focal weights, no class weights,
  no gamma override or enabled schedule. Conflicting checkpoint gamma aborts. Telemetry
  cannot mutate gamma. fp32 probability math preserves confidently wrong negative gradients.
  Mean reduction/ignore indices and production Phase 1 shape/precision/schedule assertions are also implemented.
- **Sibling-gap veto-only — implemented:** holding/rising margins never demote a stop trigger;
  falling margins may corroborate it. Production replay regressions cover both directions.
- **Phase 1 WSD implemented:** warmup/stable/1-sqrt cooldown, budget reserve, plateau dispatch once armed, exact state restore, and best-checkpoint selection during cooldown. Phase 2/3 orchestration remains deferred.
- **Option C implemented:** 896×18/MLP2320, dropout zero, QK-norm and LayerScale 0.1. The 64×16 microbatch/accumulation choice is provisional until profiling with the final vocabulary.
- **Data preparation implemented:** new namespaced sidecars, all categories except 77 rejected tags, per-image distinct counting including known ratings, unrated images retained with all four rating labels masked from loss/metrics/calibration (including optional telemetry, sparse validation and all-unrated worker/trainer regressions), fresh exact 30K split, TRAIN/EVAL label and membership hashes, and guarded startup. Do not build until the second subset arrives.
- **Optimizer precision verified on a small CUDA model:** head/position states fp32, eligible backbone states dynamic blockwise 8-bit. Full-shape throughput/VRAM profiling remains a launch check.
- **Instrumentation implemented:** binned per-decile mAP and reporting-only binned oracle F1, exact-versus-binned diagnostic mAP, and sampled attention-logit maxima; these run without requiring TensorBoard. The original exact per-tag oracle search remains separate from the cheaper threshold-grid diagnostic.
- **Deferred independent-release tooling (§8.1):** CALIB support/fallbacks, candidate weighting,
  calibration cross-fitting and the §9 offline scorer/statAP pooled-judgment harness.
  These procedures must be implemented before claiming the full release protocol passed;
  extra holdout allocation is not a current P1 launch requirement.
- Cleaning-track cleanlab integration remains in §10; gate labels cannot fit the cleaner.
- ONNX/export validation — the 1024→896 width change re-triggers V1's head-split dynamic-batch
  repair.
- Stop-conditions reconciliation (§8.2): docstring meaning of 3e-3 + `_window_gain` semantics.
- Commit the rotating-padding-mask fix + `test_rotation_mask.py` (§7 cites it; HEAD still has
  the static mask, the test is untracked).
- Checkpoint-resume compat: add `data.*` augmentation keys to the compat diff
  (`training_utils.py:471-488`) — **warning tier only**, gated to same-phase resume (phase boundaries legitimately
  change aug via flip-phase-and-resume). Full rationale: [augmentation round 1 §4.6](reviews/v2-augmentation-review.md).
  **Implemented: critical compatibility mismatches hard-fail** on `resume_from=latest/best`
  with the diff printed. Missing explicit checkpoint paths also abort.

**Audit closure checks — part of this implementation queue.** The
[2026-09-05 audit](reviews/vit-major-issues-audit-2026-09-05.md) owns reproductions and
verification limits. IDs below refer to that audit; overlapping findings belong to the
existing work above, rather than a second implementation backlog.

| Audit IDs | Work / closure required | Relationship to the recipe work |
|---|---|---|
| **1** | Compute ASL negative-branch probabilities and logarithms in fp32 before bf16 saturation; verify nonzero finite gradients on confident negative examples | Extends the precision requirement: casting only the final reduction is insufficient |
| **2** | Upcast validation logits before sigmoid; verify the selection metric retains the score distinctions in the audit reproduction | Instrument correctness, §8.2 |
| **3** | User revised scope: keep only 30K validation held out; return excess candidates to training and verify worker disjointness | Larger CALIB/TEST reservation deferred, §8.1 |
| **4** | Incompatible `latest`/`best` resume must abort with its diff, never begin fresh in the same experiment | Existing fail-fast resume work |
| **5** | Save the final committed state on ordinary completion and policy halt, including completed-epoch status and latest evaluation history | Required for reliable phase/cooldown handoff, §6 |
| **6** | Preserve checkpoint pointer order under async queue saturation; a stale queued save must not overwrite newer `last.pt` / `best_model.pt` pointers | Checkpoint lifecycle; exercise a deterministic backlog |
| **7** | Enforce fixed detached ASL on launch and resume; test the effective criterion, including checkpoint/config precedence | Existing loss and conformance work, §4 |
| **8** | A rising sibling margin must not demote a tripped stop rule; update the regression that currently enforces demotion | Existing veto-only stop-policy work, §9.2 |
| **9–10** | Budget worker-private Arrow selections and expanded validation diagnostics at the intended split/worker count; reduce copies or peak allocations if needed | Resource readiness, not a claim of a measured production OOM |

WSD now has schema support **and trainer dispatch**. The earlier audit recorded direct
cosine construction, so changing `training.scheduler` alone cannot implement the recipe.
The audit now contains a remediation status table and additional training findings.
`test_training_audit.py` exercises the fixes; old measurement history is reset on resume via
`TrainingState.measurement_contract`. Actual split fingerprinting and production-size memory measurement
remain full-scale launch checks; the Phase 1 WSD and architecture implementations now have focused regressions.


**2. Gold protocol (§9.4):** prepare reviewer reliability and the sampling design early.
The realized V2-dependent pool follows training; gold adjudication gates the superiority
claim, not the existence of a V2 checkpoint. Extra CALIB/TEST reservations remain deferred.

**3. Pre-registration:** hash the actual TRAIN/30K EVAL image and label membership,
vocabulary/TAGSET and baseline checkpoint before this launch. For the deferred full
§9 protocol, freeze the gold sampling design before training and the realized two-model
pool after fixed V2 inference, before adjudication/scoring. No extra holdout is implied.

**4. Budgets, before P1 launches:**
- **Freeze a real compute ceiling and cooldown reserve.** The 400–700M envelope is a
  planning range, not authorization for unbounded training until an uncertain gate fires.
  A ceiling reached without convergence is reported as budget-limited; it is not evidence
  of a plateau. Profile actual throughput and accumulation before committing calendar dates.
- **Wall-clock:** the programme contains **two multi-week from-scratch P1 runs** (V2 + the
  janitor's fresh Stage A): each ≈1.3e20 FLOPs ≈ **2.5–7 weeks on the 5090 at 25–40% MFU**
  (estimate — nothing GPU-measured yet), before P2/P3 (+~40% FLOPs) and before human review.
- **Abort rule (mid-phase):** hard-halt on NaN loss · max attention logits > 1e4 (Wortsman's
  divergence line; logged from epoch 1 per §3) · grad-norm > the §4-rederived band for 3
  consecutive epochs. The LR guard (§6) covers the first 5 epochs; this rule covers the rest.
  `early_stopping_mode: advise` halts nothing by itself — these three are the halts.
- **Storage:** one checkpoint stream (no EMA copy, §6); size `save_total_limit` accordingly.
- **Budget to the envelope TOP (added 2026-08-01):** size wall-clock, storage
  (`save_total_limit`) and the calendar to ~700M presentations (the 6–12-week P1 branch), not
  §6's ~419M row-sum — the row-sum is the envelope bottom. Per-parameter normalization
  against DeiT III is a budget analogy, not a required presentation count. Confirm the
  real ceiling before launch and reserve the anneal within it.
- **Operator stop rule (added 2026-08-01, mirror in v2-monitoring.md):** the only sanctioned
  phase completions follow a cooldown (plateau, planned P3 end or budget-driven completion).
  A user soft-stop may pause training at a safe boundary and resume the same schedule later;
  it does not certify phase completion. **NaN/Inf or divergence aborts halt immediately**:
  never train through invalid state to satisfy cooldown. Diagnose first, recover a finite
  checkpoint if appropriate, then decide whether to resume or restart. A discarded failed
  run does not need an annealed checkpoint.

**5. Run:** §6 Phase 1 → 2 → 3, with §4's loss and §8's instrument, judged by §9's three gates.

---

## Appendix A — ASL telemetry spec

*Observable set for a fixed-γ run, surfaced by the telemetry-only `ASLDriveManager`; every γ-step
gate role from the retired ASL plan is inert.*

| Metric | Definition | Role under fixed γ |
|---|---|---|
| `Δp_mean` | mean score on observed positives minus mean score on observed negatives | Coarse score gap, dominated on the negative side by easy negatives; not ASL paper §2.6's target-confidence gap and not a loss-selection gate |
| `Δp_hard` | mean(p_pos) − mean(top-10 non-GT scores) | Boundary-relevant gap. Caveat: crushing unlabeled true positives *widens* it — can veto, never green-light |
| per-decile score-mass ratio (ASL telemetry EPR) | threshold-free Σ scores / observed positive count, with matching non-rating columns and sample population | Descriptive trend only; no universal healthy level or missingness inversion. Interpret jointly with supported ranking and reviewed-real recall (§4). Cole's expected-positive regularizer is not a calibration theorem for this ASL model |
| non-GT score histogram | bucket counts over [0.05, 0.95] per validation pass; watch **[0.2, 0.5]** and **[0.6, 0.9]** | [0.2, 0.5]: clip-cost observable (pile-ups are diagnostic; neither clip nor suppression is identified by this histogram alone). **[0.6, 0.9] is an emission watch** — growing mass can be true-label recovery or false positives; investigate on reviewed labels |
| sibling-gap | per confusable group: score of labeled sibling − max score among unlabeled siblings | Direct boundary observable, robust to missing positives outside the group; feeds §9.2 — **veto-only** |
| Anima recall canary | recall on the 7,805 known-positive synthetic images, per eval | The only label-clean recall signal; probe, never certifies (§9.4) |

**Measurement hygiene:** compute on `logits[:, 2:]` — columns 0–1 (PAD/UNK) are live, loss-free,
drifting outputs (val metrics already special-case via `skip_metric_cols`). **Rating tags are
excluded from `mean(p_pos)`** (decided 2026-07-30 — they inflate it). fp32-upcast before sigmoid
(bf16 granularity ~0.004 near p=0.5). Sample only on optimizer-update boundaries (reuse
`is_update_boundary`) so gradient accumulation doesn't alias the EMA. Val-side variants are pure
consumers of the accumulated prob/target matrices.

**Supplementary (log, don't gate):** Label Wave (single-label/CIFAR evidence — bookkeeping only) ·
`rare_bucket_ECE` (<1000-freq tags, SmoothECE estimator) · `cooccur_jaccard@K` (prior from the
clean slice, not noisy val) · `logit_std_rank_11_50`. Trigger rule: two consecutive monotonic
problem-direction moves at E5+ → spot-check predictions by hand.

---

## 13. Sources

**Loss.** ASL, ICCV 2021 — [arXiv:2009.14119](https://arxiv.org/abs/2009.14119) (**Table 2 / Appendix B: 4/0/.05; Appendix F: γ⁻=7 with reduced-weight assumed negatives; §4 distinguishes released-code detach**) · RAM, CVPRW 2024 —
[arXiv:2306.03514](https://arxiv.org/abs/2306.03514) · RAM++ —
[arXiv:2310.15200](https://arxiv.org/abs/2310.15200) · SINR, ICML 2023 —
[arXiv:2306.02564](https://arxiv.org/abs/2306.02564) · Hill/SPLC —
[arXiv:2112.07368](https://arxiv.org/abs/2112.07368) · P-ASL/CSL, CVPR 2022 —
[arXiv:2110.10955](https://arxiv.org/abs/2110.10955) (*not adopted — §1, §4*) · Large Loss Matters —
[arXiv:2206.03740](https://arxiv.org/abs/2206.03740) · ELR —
[arXiv:2007.00151](https://arxiv.org/abs/2007.00151) · HLC, ICCV 2023 (wrong-positive track)

**Architecture / recipe.** DeiT III, ECCV 2022 —
[arXiv:2204.07118](https://arxiv.org/abs/2204.07118) · Steiner et al. —
[arXiv:2106.10270](https://arxiv.org/abs/2106.10270) · Scaling ViT, CVPR 2022 —
[arXiv:2106.04560](https://arxiv.org/abs/2106.04560) · SoViT, NeurIPS 2023 —
[arXiv:2305.13035](https://arxiv.org/abs/2305.13035) · CaiT/LayerScale —
[arXiv:2103.17239](https://arxiv.org/abs/2103.17239) · Generalized Neural Collapse —
[arXiv:2310.05351](https://arxiv.org/abs/2310.05351) · *Three things everyone should know about
ViT* — [arXiv:2203.09795](https://arxiv.org/abs/2203.09795) · NaViT —
[arXiv:2307.06304](https://arxiv.org/abs/2307.06304) · EMA dynamics, TMLR 2024 —
[arXiv:2411.18704](https://arxiv.org/abs/2411.18704) · When/where to average —
[arXiv:2502.06761](https://arxiv.org/abs/2502.06761) · RoPE-ViT, ECCV 2024 —
[arXiv:2403.13298](https://arxiv.org/abs/2403.13298) (*rejected — §11*) · QK-norm —
[arXiv:2309.14322](https://arxiv.org/abs/2309.14322) · σReparam, ICML 2023 —
[arXiv:2303.06296](https://arxiv.org/abs/2303.06296) · 4-bit optimizer states —
[arXiv:2309.01507](https://arxiv.org/abs/2309.01507) · Dettmers, 8-bit optimizers —
[arXiv:2110.02861](https://arxiv.org/abs/2110.02861) · Beyer et al. plain-ViT baselines —
[arXiv:2205.01580](https://arxiv.org/abs/2205.01580) · Malladi et al., sqrt LR scaling —
[arXiv:2205.10287](https://arxiv.org/abs/2205.10287) · Sigmoid bottleneck, AAAI 2024 —
[arXiv:2310.10443](https://arxiv.org/abs/2310.10443) · ML-Decoder, WACV 2023 —
[arXiv:2111.12933](https://arxiv.org/abs/2111.12933) · Patchification scaling —
[arXiv:2502.03738](https://arxiv.org/abs/2502.03738) · McCandlish et al., critical batch —
[arXiv:1812.06162](https://arxiv.org/abs/1812.06162)

**Schedule (WSD).** Hägele et al., NeurIPS 2024 —
[arXiv:2405.18392](https://arxiv.org/abs/2405.18392) · Tissue et al. —
[arXiv:2408.11029](https://arxiv.org/abs/2408.11029) · MiniCPM —
[arXiv:2404.06395](https://arxiv.org/abs/2404.06395) · vision precedent: Zhai et al.'s
rsqrt+cooldown "infinite schedule" (Scaling ViT, above)

**Init (§5, closed).** Zoph et al. — [arXiv:2006.06882](https://arxiv.org/abs/2006.06882) ·
Chen & Zwicker, WACV 2022 — [arXiv:2108.01819](https://arxiv.org/abs/2108.01819) · DAF:re —
[arXiv:2101.08674](https://arxiv.org/abs/2101.08674) · Illustration2Vec, SIGGRAPH Asia 2015
(frozen-feature evidence, not init evidence)

**Evaluation.** xCOLUMNs, ICLR 2024 — [arXiv:2401.16594](https://arxiv.org/abs/2401.16594)
(*rejected — §11*) · Generalized test utilities, NeurIPS 2023 —
[arXiv:2311.05081](https://arxiv.org/abs/2311.05081) · Missing labels & propensities, KDD 2022 —
[arXiv:2207.13186](https://arxiv.org/abs/2207.13186) · Calibration at XMC scale —
[arXiv:2411.04276](https://arxiv.org/abs/2411.04276) · Pervasive label errors, NeurIPS 2021 —
[arXiv:2103.14749](https://arxiv.org/abs/2103.14749) · Zhao & Gomes —
[arXiv:2102.08427](https://arxiv.org/abs/2102.08427) (*training-time robustness; the mis-ranking
claim rides on Northcutt*) · Cole et al., single-positive multi-label (EPR) —
[arXiv:2106.09708](https://arxiv.org/abs/2106.09708) · Cheng & Vasconcelos, CVPR 2024 (asymmetric
losses are not strictly proper) · SmoothECE, Błasiok & Nakkiran 2023 —
[arXiv:2309.12236](https://arxiv.org/abs/2309.12236) · Dual-TS —
[arXiv:2308.08366](https://arxiv.org/abs/2308.08366) · Label Wave, ICLR 2024 —
[arXiv:2502.07551](https://arxiv.org/abs/2502.07551) · Thresholding for F1 —
[arXiv:1402.1892](https://arxiv.org/abs/1402.1892) · Niculescu-Mizil & Caruana, ICML 2005 ·
**statAP (Aslam & Pavlu) · infAP (Yilmaz & Aslam, CIKM 2006) · xinfAP (Yilmaz, Kanoulas & Aslam,
SIGIR 2008)** · Sakai, SIGIR 2006 (paired bootstrap)

**Augmentation.** LogicMix — [arXiv:2405.15860](https://arxiv.org/abs/2405.15860) · SpliceMix —
[arXiv:2311.15200](https://arxiv.org/abs/2311.15200) · BalanceMix —
[arXiv:2312.07087](https://arxiv.org/abs/2312.07087) (*rejected — §11*)

**On-disk evidence.** `experiments/run1_vit/checkpoints/*.pt`, `pr_threshold_*.json` ·
`logs/training.log` · `tensorboard/**/events.out.tfevents.*` · `vocabulary.json` ·
`logs/splits/2df7eb3c2bc0d784.{train,val}.txt` · `L:\Dab\DataCleaning Project\corrections_report.json` ·
`configs/unified_config.yaml`
