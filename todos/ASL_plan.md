# ASL Plan — Driving the Loss for V2

> **Status.** Standalone loss plan, split out of [progressive-training-plan.md](progressive-training-plan.md) (§1.3/§3.1) on 2026-07-02. Where this doc and the progressive plan disagree, **this doc wins for loss settings**; the deltas vs. the older plan are listed explicitly in §6. Everything else (architecture, LR, aug, schedule) stays in the progressive plan.
>
> **2026-07-02 — Option A adopted: no golden/clean slice during this run.** The always-on telemetry (§5) is the sole γ driver. P1 descends 7 → floor 5 as planned; **P2 holds and lands at 5** — the 5→4 step and the §5 clean-slice gates become dormant contingencies, activated only if a sealed clean slice ever exists.
>
> **Research basis.** ASL paper mechanics re-verified against the arXiv/ICCV text (incl. the Adaptive Asymmetry section — §2.7 in the arXiv numbering, previously cited here as §4.4 — and the per-dataset hyperparameter table) plus the 2026-06 verification pass already folded into the progressive plan. Key new finding (2026-07-02): **V1's γ_neg=7 + clip=0.2 combination was never published** — the paper ran γ_neg=7 only on Open Images (m is not printed there; 0.05 is inferred from the code default and every other dataset), and its MS-COCO optimum was γ_neg=4, m=0.05.
>
> **Verification pass (2026-07-02).** Every paper-mechanics claim below was re-checked against the paper text, the official code, and a 2021–2026 literature sweep. All core claims held; four corrections are folded in: the §1 table arithmetic (the corrected numbers *strengthen* the diagnosis), the published adaptive-γ trajectory (it **ascends** — §4 caveat), Kim 2022's schedule direction (§3 caveat), and the noisy-val citation (now Northcutt, §3). Confirmed absence: **no published work anneals γ_neg downward in multi-label training** (Weber 2019 is a single-label-detection downward-γ precedent — closest published mechanism) — this schedule is publication-novel in multi-label, so the §5 gates are load-bearing, not belt-and-suspenders. None of the drive machinery exists in code yet (§8).
>
> **Adversarial review (2026-07-02, later same day — five competitive angles + judge; verdicts folded in throughout).** The skeleton held: 7→5 trajectory, γ_pos=0, sole loss, Option A, P2 lands 5, epoch-8-earliest descent. Four material changes: (1) **clip=0.2 is now conditional on a pre-launch measurement of the missing-positive rate** (§0, §1) — the clip freeze is irreversible at run start, so the measurement is a launch blocker; (2) the **§4 controller is demoted to shadow/logging-only** — Δp_mean is arithmetically insensitive to γ at 19K tags, so the feedback law degenerates to an open-loop slam to the clamp floor; γ is driven by the manual guarded steps with dwell (§3); (3) the **§5 gates are re-specced** — the thresholded `pred_pos_ratio < 1.0` alarm fires only post-mortem, and Δp_hard *widens* when unlabeled true positives get crushed (blind to the descent's primary failure mode); replaced with a threshold-free per-decile EPR trend alarm, a non-GT score histogram + sibling-gap metric, and a per-step Anima recall canary; (4) **§8 gains a checkpoint-persistence row** — today any γ change silently reverts to the YAML value on resume. Open evidence item: the §1 V1 history must be verified against V1 checkpoint configs (§0) — the config's own comment says V1's P1 ran γ=4, not 7.

---

## 0. Pre-launch blockers (added 2026-07-02 — the run does not start until these clear)

1. **P0 — Measure the empirical missing-positive rate.** The clip decision (§1, §2) hinges entirely on this number, it sits unmeasured, and the data to measure it is already on disk: tags **added** during human review in the Tag Review Portal (`portal_data/edits.db`) and the 7,805-image Anima golden-set corrections. Compute ρ overall, per tag-frequency decile, and per registered confusable group. Decision rule (from the §1 break-even table): measured tail/confusable-group ρ **< ~25% → set clip=0.1 pre-run**; ρ **≳ 30% or unmeasurable in time → hold clip=0.2**. Write the measured rate and the ρ* table into §2 as the armor's load rating. This blocks launch because clip is frozen irreversibly at run start (§1 rule + §2).
2. **P1 — Recover V1's true γ trajectory.** [unified_config.yaml:297](../configs/unified_config.yaml#L297)'s own comment says V1's **Phase 1 ran γ_neg=4.0** and only P2 ran 7 — contradicting §1's "entire run" framing. Peek the embedded configs of an early-P1 and a P2 checkpoint (`CheckpointManager.peek_checkpoint_config`, [train_direct.py:994](../train_direct.py#L994)), rewrite §1 with the actual history, and re-apportion γ-vs-clip blame **before** finalizing the clip decision. Fix the stale comments at yaml:297/:300.
3. **P1 — Loss-state checkpoint persistence (§8 row 2) must land before the first descent step.** Without it, any restart silently reverts γ to the YAML value — V1's loss again.

## 1. Diagnosis — why V1 "lacked boundaries"

> **History caveat (2026-07-02):** per §0 item 2, the "γ7 for the entire run" framing below is unverified — the config comment attributes γ7 to P2 only. The "no boundaries" artifact was measured on the post-P2 model, so the *mechanism* stands either way, but the γ-vs-clip blame split stays ambiguous until the V1 checkpoints are peeked. If P1 genuinely ran γ4+clip0.2 for ~40 epochs and the band still died, clip's share of the blame rises materially.

V1 ran ASL with `gamma_neg=7.0`, `clip=0.2`, `gamma_pos=0` ([configs/unified_config.yaml:296-300](../configs/unified_config.yaml#L296-L300)). In the implementation ([loss_functions.py:294-329](../loss_functions.py#L294-L329)) a negative's loss is:

```
(p − clip)₊^γ_neg · −log(1 − p + clip)
```

*(Implementation note, 2026-07-02 audit: our focal weight is **not detached** ([loss_functions.py:317-329](../loss_functions.py#L317-L329)), unlike the official ASL default — the extra ∂(focal-weight)/∂p term scales gradient magnitudes ~2–5× in the semi-hard band. All tables in this doc compare loss ratios like-for-like, so every ratio and conclusion survives; but never flip detach mid-run, and tune any constants against the code's gradient scale, not the paper's.)*

Both knobs suppress the negative branch and they **multiply**. The damage concentrates in the semi-hard band p ∈ [0.3, 0.7] — exactly where boundaries between confusable sibling tags are learned:

| negative at p | γ=7, clip=0.2 (V1) | γ=4, clip=0.05 (paper COCO optimum) | signal ratio |
|---|---|---|---|
| 0.3 | ~1e-8 | ~1.1e-3 | ~100,000× |
| 0.5 | ~8e-5 | ~2.5e-2 | ~300× |
| 0.7 | ~5e-3 | ~0.19 | ~35× |

A wrong-but-plausible sibling tag sitting at p≈0.5 received effectively **zero push-down for the entire run**. Consequences: sibling scores compress upward into an undifferentiated blob, decision boundaries never form, threshold-based F1 degrades while ranking metrics (mAP) partially hide it. This is structural, not a tuning nuance.

**What the paper actually shipped** (per-dataset table, ICCV 2021):

| Dataset | γ+ | γ− | m |
|---|---|---|---|
| MS-COCO | 0 | 4 | 0.05 |
| Open Images (~9.6K classes) | 0 | 7 | 0.05† |
| Pascal-VOC / NUS-WIDE | 0 | 4 | 0.05 |

† The paper changes only γ for Open Images and never prints m there; 0.05 is inferred from the code default and every other dataset.

The m≈0.2 figure has no published home at all: Fig. 7's **margin-only ablation** swept m on top of CE (γ=0) and *symmetric* focal γ=2/4, finding optima at m=0.05 (CE) and m=0.3–0.4 (focal) — m=0.2 was never anyone's optimum or default, and was never combined with γ_neg=7 in any recipe. The comment at [unified_config.yaml:300](../configs/unified_config.yaml#L300) calling m=0.2 the "paper default for the focal variant" overstates it. (Fig. 7's *direction* — optimal m grows with γ — was previously cited as mild support for parking a large clip next to a large γ_neg; the adversarial review withdrew that defense as a category error — the sweep was CE/*symmetric* focal, and every asymmetric recipe the paper shipped used m=0.05. The deviation below now rests solely on the break-even arithmetic against the measured missing-positive rate.)

**Why clip=0.2 is the default (deliberate deviation — made conditional on the §0 measurement, 2026-07-02).** Clip is the missing-positive armor: it caps the loss a memorized-but-unlabeled positive (high p, labeled negative) can absorb instead of letting −log(1−p) diverge. Implementation-exact numbers (the earlier ~−log(clip) figure overstated the cap ~3×, in the safe direction): the saturation cap at p→1 is (1−m)^γ·(−log m) ≈ **0.53** (γ5, m0.2) vs **1.36** (γ5, m0.1) vs **2.44** (γ4, m0.05); at p=0.95 a missing positive absorbs 2.6× more at m=0.1 and 4.6× more at the paper recipe.

The decision-relevant quantity (adversarial review) is **per-tag gradient balance**, not aggregate loss mass: the break-even miss rate ρ* above which a tag's own true positives receive net-*downward* gradient at p=0.8 is:

| Config | ρ* at p=0.8 |
|---|---|
| γ=5, m=0.2 | **~61%** |
| γ=5, m=0.1 | ~38% |
| γ=4, m=0.05 (paper) | ~26% |

The cost, equally quantified: **after the γ descent, clip — not γ — binds boundary formation below p≈0.5.** At γ=5/m=0.2 the remaining suppression deficit vs. the paper's boundary-forming recipe (γ4, m0.05) is ~1067× / ~90× / ~28× at p=0.3/0.4/0.5, and moving m 0.2→0.1 buys more band signal (68× / 12× / 6×) than the entire γ descent does anywhere below p≈0.5. Which side wins hinges **entirely** on the measured missing-positive rate (§0 item 1): at booru-plausible tail ρ=30–50%, clip=0.1 puts the noisiest tags' true positives into net score collapse — and under Option A that failure is invisible to every gate we have (observability asymmetry favors 0.2 as the default); at ρ<~25%, 0.1 is the better trade and is set pre-run. **If 0.2 is held: boundary suppression below p≈0.45 is forfeited this run** — it stays 1–3 orders of magnitude below published boundary-forming recipes, and separation there is carried by positive-side gradient plus per-tag thresholds, not negative push-down.

The two knobs still divide labor:

- **clip** = shift → protection for high-p mislabeled negatives (missing positives). **Frozen all run at the §0-decided value** (0.2 default, 0.1 if the measurement clears it).
- **γ_neg** = exponent → carries the suppression above ≈clip+0.15. **This is the drive knob.**

Rule: **never move clip and γ_neg in the same step** — attribution dies otherwise.

## 2. Fixed settings (all phases, not swept)

| Setting | Value | Grounding |
|---|---|---|
| `gamma_pos` | **0** | Ridnik Fig. 6: mAP degrades monotonically as γ_pos rises from 0, measured at ~3.6% positive; at our 0.6% the case is stronger. Every annotated positive flows through plain BCE at full gradient |
| `clip` | **0.2 default; 0.1 if the §0 measurement clears it. Frozen all run either way** | §1 above. Negative-branch knob only — zero bearing on wrong-positive (aqua/green) noise. **Adversarial review 2026-07-02: the earlier "held" verdict is downgraded to conditional.** The Fig. 7 "mid-range for our γ" defense is withdrawn — category error (that sweep was CE/*symmetric* focal; every asymmetric recipe the paper shipped used m=0.05, including Open Images at γ7/9.6K classes). The honest ground for 0.2 is the §1 break-even arithmetic (ρ* ≈ 61/38/26% at m=0.2/0.1/paper) against the *measured* missing-positive rate (§0). Known cost, corrected: zero gradient for p≤clip and near-zero through p≈0.45 at γ=5, so negatives park anywhere in **[0.2, ~0.45]** (wider than the "just below 0.3" previously stated) — thin margin for rare tags whose true positives also score low. The old "too high" signature (top-10 non-GT pile-up at 0.25–0.35) is **dead text**: high-scoring missing positives crowd all ten top-10 non-GT slots precisely when the armor works (parked siblings sit at rank 11+, and the parking band is wider than the watch window), and even when it fires it cannot distinguish clip-parked siblings from descent-crushed missing positives. Replaced by the §5 **non-GT score histogram** (watch band [0.2, 0.5]) + **per-confusable-group sibling-gap**. A clip-0.1 follow-up run is authorized only on band pile-up **AND stable pred_pos_ratio** — pile-up with *falling* pred_pos_ratio means step γ back up, never lower clip. Always a **separate single-knob run** — never mid-run, never together with a γ move (§1 rule) |
| `alpha` | **1.0** (disabled) | Paper Table 2: either ASL mechanism alone (86.3 mAP) beats FL + linear weighting (85.3); weighting is redundant next to dynamic asymmetry (it helped plain FL only +0.2 — "insufficient", not toxic) |
| `label_smoothing` | **0.0** | Smoothing corrupts focal weights on sparse multi-label targets |

## 3. γ_neg schedule — the drive plan

### Phase 1 (320px, from scratch, ~40 ep)

| Window | γ_neg | Why |
|---|---|---|
| Epochs 1–8 (warmup + early learning) | **7.0, hard hold** | From-scratch probabilities are diffuse; many negatives clear the clip; lowering γ_neg here multiplies surviving-negative mass (~37× at p=0.5 for 7→4) and flips gradient balance against the 165:1-outnumbered positives (ELR early-learning; Kim 2022 confirms the same clean-first/memorize-late order in multi-label WSML — but see the direction caveat below). The 8-epoch hold ≈ warmup + the ~10–20% of training the ASL paper's own dynamic scheme needed before γ− stabilized |
| Epochs 9 → end | **descend 7 → ~5 via manual guarded steps 7→6→5 (primary driver — the §4 controller is shadow-only), minimum dwell ~3 epochs per unit step, floor 5 in P1.** Each step gated on the §5 always-on set: per-decile EPR trend stable, Δp_hard holds-or-widens, Anima recall canary holds. Epoch 8 is the *earliest* unfreeze; optionally gate the unfreeze on a measurable instead — EMA frac(negatives with p > clip) < ~5%, stable for 2 epochs (one cheap reduction on tensors the loss already computes) | After early learning, most easy negatives sit below clip (zero loss regardless), so a guarded descent only re-engages the semi-hard boundary band — 40 epochs of boundary formation are too expensive to postpone to P2. Mechanism precedent: the paper's adaptive scheme (§4) — though its *published* trajectory ran the other way; see the caveat below |

**Direction-of-descent caveat (2026-07-02 sweep; softened same day after the adversarial review's literature pass).** No published work anneals γ_neg *downward* during **multi-label** training — in multi-label every published γ trajectory ascends (ASL's own adaptive scheme initializes γ low and climbs, stabilizing within ~10% of training; annealed/cyclical focal variants likewise), and Kim 2022's LL-R *raises* its noise-rejection rate epoch-by-epoch precisely because missing-positive noise is memorized late. **Closest published precedent for feedback-driven γ descent: Weber 2019 (arXiv:1904.09048, single-label object detection — γ = −log(p̂) falls as confidence grows); AdaFocal (NeurIPS 2022, single-label calibration) is secondary.** Neither is multi-label, so the multi-label novelty stands. Descending γ late runs mildly against the memorization literature: γ 7→4 raises the loss a memorized missing positive (p≈0.9, labeled negative) absorbs by ~2.9× (0.7⁴ vs 0.7⁷), against a clip-capped log term. Bounded, but real — note the timing: the epoch-9 descent start *coincides with*, not precedes, the memorization-active phase (Kim 2022's FN-loss peaks right after early learning), which is exactly why each step carries a ≥3-epoch dwell, the per-decile EPR trend gate, and the Anima recall canary (§5) rather than a bare Δp_hard check. Any gate failure steps γ back up.

### Phase 1 → 2 transition

γ_neg **frozen at its P1-final value (~5)** through the 2-epoch re-warmup. The resolution switch (401→785 tokens) + optimizer reset transiently distorts the probability distribution, so the gap signal is garbage — phase-transition hygiene, same logic as the LR re-warmup.

### Phase 2 (448px, fine-tune, ~12 ep)

| Window | γ_neg | Why |
|---|---|---|
| Re-warmup (2 ep) | frozen at ~5 | above |
| After re-warmup | **hold 5, land 5** (Option A, 2026-07-02) | No clean slice is used this run, and the 5→4 decision is exactly where noisy val mis-ranks in the *wrong direction*: a boundary-recovering model starts firing on unlabeled true positives, which noisy val scores as FP inflation — the better model loses the comparison. γ_neg=5 + clip=0.2 already recovers ~11× of V1's boundary signal at p=0.5 (p=0.5 only — below p≈0.45 the band stays clip-bound regardless of γ; see §1); the last step is deliberately forfeited. **Contingency (dormant):** if a sealed clean slice ever exists, sweep 5 → 4 per the §5 gates — γ_neg=4 is the paper's COCO/VOC/NUS-WIDE optimum, and with clip 0.2 retained it still exceeds any published suppression combo |

**Hard rule (now the operative branch):** any step below 5 requires the §5 clean-slice gates. **No clean slice ⇒ hold at 5, don't sweep** — model selection and γ decisions on noisy val mis-rank in general (Northcutt NeurIPS 2021: +6% test-label errors flips ResNet-50 vs ResNet-18 rankings), and for this specific step the bias is anti-correlated with the thing being measured.

### Phase 3 (optional 512px)

Loss unchanged from P2 landing (γ_neg=5 under Option A — 4 only if the clean-gated contingency ran; clip at its §0-decided value, γ_pos=0).

## 4. Adaptive-asymmetry controller (DEMOTED 2026-07-02: shadow/logging-only — it never sets γ this run)

The ASL paper's own Adaptive Asymmetry mechanism (§2.7 in the arXiv numbering) was the preferred driver in the previous revision. The adversarial review killed it as a driver on arithmetic:

```
Δp   = mean(p_positives) − mean(p_negatives)      # batch-level, EMA-smoothed
γ_neg ← γ_neg + λ · (Δp_target − Δp)              # gap too small → raise γ; gap wide → descend
```

- **Why it cannot drive at 19K tags: Δp_mean is a dead signal.** mean(p_neg) is averaged over ~19.2K negatives per sample, the overwhelming majority sub-clip with zero gradient — its response to a γ step in the clamp range is ~5e-4, while the error term (Δp_target − Δp) ≈ −0.3 is large and constant-sign. At λ=0.05–0.1 with an update every ~100 steps (~146 updates/epoch at effective batch 432 on 6.3M images), the "controller" completes 7→5 in **~0.5–1 epoch after unfreeze** — an open-loop slam to the clamp floor, not feedback. The paper's COCO equilibrium (γ parked at ~5.2, Table 3 target=0.2) was only possible at 80 classes, where mean(p_neg) actually responds to γ. "Descends on its own timetable" is illusory here.
- **This run:** implement it (if at all) in **shadow mode** — compute Δp on the same EMA cadence and log the γ it *would* set next to the actual γ. Zero authority. The log is cheap calibration data for whether any future controller is worth reviving.
- **The operative driver is §3's manual guarded steps** (7→6→5, ≥3-epoch dwell, gated on the §5 set). The clamps ([5,7] P1 / [5,6] P2) and warmup freezes still apply to the *manual* schedule.
- Kept for the record: **Δp_target = 0.2** is the paper's best-tested value (Table 3: targets {0, 0.1, 0.2, 0.3} → mAP {85.8, 86.1, 86.4, 86.3}); on COCO the controller landed γ− ~5.2, *above* the fixed optimum 4 — which already argued for clamping over trusting the landing point, and now doubles as evidence that even a live-signal controller lands imprecisely.

## 5. Telemetry and gates

### Always-on (no clean slice needed — live from epoch 1; re-specced 2026-07-02)

| Metric | Definition | Role |
|---|---|---|
| `Δp_mean` | mean(p_pos) − mean(p_neg) | Shadow-controller input and coarse health only. **Not a gate** — arithmetically insensitive to γ at 19K tags (§4) |
| `Δp_hard` | mean(p_pos) − mean(top-10 non-GT scores) | **The boundary-relevant gap — but necessary-NOT-sufficient.** Trend rule stands: each γ step must widen or hold it. But it is **blind to the descent's primary failure mode**: crushing unlabeled true positives *lowers* top-10 non-GT scores and therefore **widens** Δp_hard — the gate passes on the failure (the same anti-correlation that disqualifies noisy val for the 5→4 step, §3). It can veto a step; it can never green-light one alone. Computable from the §1.4 top-K validation hook |
| `pred_pos_ratio` (EPR) | **threshold-free**: Σᵢ pᵢ / expected positives (Cole 2021's actual formulation — not a thresholded count), computed **per tag-frequency decile**, EMA-smoothed | Suppression tripwire. Healthy operating point is ≈ 1/(1−ρ) **> 1** (missing positives inflate it), so the old "falls below 1.0" alarm fires only post-mortem. **Alarm: sustained >5–10% relative drop in any decile within ~2 epochs of a γ step → step back up.** This is the signal that disambiguates Δp_hard's blind spot |
| non-GT score histogram | bucket counts of non-GT scores over [0.05, 0.95] per validation pass; watch band **[0.2, 0.5]** | The clip-cost observable (§2) — replaces the blind top-10 pile-up signature. Pile-up in-band + stable EPR = clip too high (log for a future clip-0.1 run); pile-up + falling EPR = γ overshoot, step γ back up |
| sibling-gap | per registered confusable group (hair color/length etc., from the tag-group registry): score of the labeled sibling − max score among unlabeled siblings | The *direct* boundary observable, robust to missing positives outside the group. Must not shrink across a γ step |
| **Anima recall canary** | recall on the 7,805 known-positive synthetic images (prompt-controlled ground truth), per eval and **mandatorily per γ step** | The only label-clean recall signal that exists under Option A (§5 post-training already licenses the slice as probe-not-anchor). **Each γ step must hold it.** A drop = missing-positive-style crushing reaching real positives |

Optional supplementary channel: **Label Wave** (ICLR 2024) — the prediction-fluctuation minimum on a fixed training subset marks the onset of fitting mislabeled data, validation-free. Single-label/CIFAR-scale evidence, so: bookkeeping channel and a sanity-check on the noisy-val f1_macro early-stop, never a gate.

**Measurement hygiene (all of the above):** compute on `logits[:, 2:]` — columns 0–1 (PAD/UNK) are live, loss-free, drifting outputs ([train_direct.py:701](../train_direct.py#L701) `ignore_indices`, [loss_functions.py:234-263](../loss_functions.py#L234-L263)); a naive top-K non-GT capture can be dominated by them (val metrics already special-case this via `skip_metric_cols`, [train_direct.py:2122-2124](../train_direct.py#L2122-L2124)). Decide rating-tag handling explicitly (they inflate mean(p_pos)). fp32-upcast before sigmoid (bf16 granularity ~0.004 near p=0.5). Sample only on optimizer-update boundaries (reuse `is_update_boundary`, [train_direct.py:1443](../train_direct.py#L1443)) so gradient accumulation doesn't alias the EMA. The val-side variants are pure consumers of the already-accumulated prob/target matrices ([train_direct.py:2129-2134](../train_direct.py#L2129-L2134), [2191-2202](../train_direct.py#L2191-L2202)).

### Clean-slice gates (dormant contingency under Option A — required only for the 5→4 step, which this run does not take)

A γ_neg step 5→4 stands only if, on the sealed clean slice:
1. **group-wise recall on golden positives** rises or holds, **and**
2. **FP-rate on golden negatives** does not inflate, **and**
3. **off-diagonal sibling confusion** (the V2.1.1 yardstick) does not worsen.

Any gate fails → step back up and hold.

### Post-training calibration

Even correctly-tuned asymmetric losses are not proper scoring rules — outputs stay systematically shifted (formally: asymmetric losses lack the strictly-proper property — Cheng & Vasconcelos, CVPR 2024). Do **not** chase deployment-threshold quality with further loss tinkering; calibrate after training on the least-noisy labels available. Default calibrator: **per-tag isotonic regression**, evaluated with **calibration@k** rather than plain ECE (ECE is misleading long-tailed; isotonic improves XMC-scale calibration without accuracy loss — arXiv:2411.04276); Platt scaling as the fallback for positive-starved tags. Under Option A there is no clean slice — note that fitting thresholds on noisy val skews them **high** (firing on unlabeled true positives is scored as FP), so cross-check per-group thresholds against the synthetic Anima slice (probe, not anchor) before shipping.

## 6. Deltas vs. progressive-training-plan.md §1.3/§3.1

1. **γ_neg begins descending late in Phase 1** (epoch ~9+, floor 5) — the old plan held 7 for all of P1. The P1-hold rationale (gradient balance) is correct only during early learning; after it, easy negatives are under clip anyway and the descent re-engages only the boundary band. Tripwires: `pred_pos_ratio`, `Δp_hard`.
2. **Phase 2 lands at γ_neg=5, reached without any clean slice (Option A, 2026-07-02).** The old plan froze γ at 7 unless a clean-gated sweep ran; this plan reaches 5 on always-on telemetry alone (P1 descent) and forfeits only the final 5→4 step, which stays documented as a clean-gated contingency. (An earlier revision of this doc landed P2 at 4; superseded by the no-golden-set decision.)

3. **2026-07-02 adversarial-review deltas:** §0 pre-launch blockers added (miss-rate measurement decides clip; V1-history verification; checkpoint persistence); §4 controller demoted to shadow; §5 gates re-specced (per-decile threshold-free EPR trend, histogram, sibling-gap, Anima canary); "clip≥0.2 never lowered" relaxes to **"clip frozen all run at the §0-decided value (0.2 or 0.1), never lowered mid-run."**

Unchanged from the old plan: ASL is the **sole** loss (no Hill/SPLC/per-group/masking), γ_pos=0, clip never moved mid-run, clean-slice gating for any step below 5. **Hill/SPLC fallback trigger, now defined in Option-A-measurable terms** (the old "plateaus below target on the clean slice" referenced a slice that doesn't exist this run): if after two consecutive dwelled γ steps the **sibling-gap metric fails to improve AND Δp_hard fails to widen while per-decile EPR is stable** — i.e., suppression is stable but buying zero boundary gain — the schedule has plateaued; evaluate Hill/SPLC for the *next* run (never mid-run; sole-loss stands for this one). Context for the fallback's standing: Hill's +2.45 mAP over ASL at COCO-40%-missing (arXiv:2112.07368) is the strongest published claim in this noise regime — stronger than anything supporting a large clip — it just isn't validated at 19K tags or against our noise structure.

## 7. What this schedule cannot fix (scope fence)

- **Wrong-positive noise** (aqua↔green↔blue): a mislabeled `aqua_hair=1` trains the wrong side of the boundary at **full BCE gradient** (γ_pos=0) regardless of γ_neg. The γ schedule sharpens boundaries only up to the ceiling the wrong-positive rate allows; raising that ceiling is the golden-anchor + Confident-Learning cleaning track (progressive plan §1.7).
- **Do not** fix boundaries by lowering clip mid-run or below the §0-decided value. (The 2026-07-02 review corrected the old parenthetical here: lowering clip *does* buy band signal — 68×/12×/6× at p=0.3/0.4/0.5 for 0.2→0.1 — but at the cost of missing-positive armor, and the trade is decided once, pre-run, by the §0 measurement, never reactively.) No per-class γ vectors either (nothing validated at 19K tags; config-only decision stands).
- Nothing cited exceeds ~9.6K classes, and the >10K-class literature (XMC) runs mostly on propensity-scored/ranking losses (Schultheis & Babbar's unbiased BCE estimators are the loss-level exception, but they require known per-tag missingness rates we don't have; COMIC is the strongest rejected joint long-tail+missing-label framework — setting-mismatched) — ASL has zero published evidence at 19K tags. Likewise no published downward γ_neg schedule exists **in multi-label training** (2021–2026 sweep; Weber 2019 is the single-label-detection exception). 5-vs-4 is directional, not paper-calibrated — the gates in §5, not the citations, are the final arbiter.

## 8. Implementation prerequisites (audited 2026-07-02 — none of this exists yet)

| Piece | Status | Work |
|---|---|---|
| Mutable γ_neg | ✗ — set once at criterion construction ([train_direct.py:694-704](../train_direct.py#L694-L704)); no setter, no schedule hook | Add a `set_gamma_neg()` (or per-step argument). 2026-07-02 audit: `torch.compile` wraps only the model ([train_direct.py:1183](../train_direct.py#L1183)) — the loss runs **eager** today, so a plain attribute works and the stop/edit-YAML/resume manual path is viable now. If the loss is ever pulled into a compile region, store γ as a 0-dim tensor/buffer (a Python-float mutation would recompile every update) |
| **Loss-state checkpoint persistence** | ✗ — **any γ change silently reverts to YAML 7.0 on restart.** Criterion is built from live config *before* resume logic ([train_direct.py:694-704](../train_direct.py#L694-L704)); the checkpoint dict carries model/optimizer/scheduler/scaler/RNG/sampler/config but **no loss state** ([training_utils.py:1259-1308](../training_utils.py#L1259-L1308)); `resume_from: latest` ([unified_config.yaml:282](../configs/unified_config.yaml#L282)) makes restarts routine on a ~40-epoch run | Save current γ_neg + telemetry EMAs + gate state into `training_state` at every `save_checkpoint`; restore into the criterion after checkpoint load, **overriding the YAML value**; log γ_neg at every save/load so any reset is visible in TensorBoard. **Must land before the first descent step (§0 item 3)** |
| `Δp_hard` + histogram + sibling-gap telemetry | ✗ — nothing in the codebase computes any of them | Batch hooks + EMA per the §5 hygiene block: `logits[:, 2:]` only, fp32 upcast, optimizer-update boundaries (`is_update_boundary`, [train_direct.py:1443](../train_direct.py#L1443)); top-K non-GT capture (progressive plan §1.4 hook) must exclude cols 0–1; val-side variants consume the existing accumulated prob/target matrices ([train_direct.py:2129-2134](../train_direct.py#L2129-L2134)) |
| `pred_pos_ratio` (EPR) | ✗ | **Threshold-free** Σp/expected-positives (not a thresholded count), per tag-frequency decile, EMA-smoothed, logged every N update steps; trend alarm per §5 |
| Anima recall canary | ✗ | Per-eval recall harness over the 7,805 known-positive synthetic images; wired to run mandatorily before/after each γ step |
| Controller (§4) | ✗ | **Shadow/logging-only this run** — log the γ it would set; zero authority. The manual guarded steps (§3) need only the setter + persistence + telemetry above |
| Clean-slice gate wiring | ✗ — selection/early-stop still run on noisy val f1_macro | **Not needed under Option A** (no step below 5 this run). Only required if the dormant 5→4 contingency is ever activated |

Manual-fallback minimum: the setter, **checkpoint persistence**, the always-on telemetry set (per-decile EPR, Δp_hard, histogram, sibling-gap), and the Anima canary. Without those, this plan silently degrades to fixed γ_neg=7 — i.e., V1's loss again.

## Sources

- Ridnik et al., *Asymmetric Loss for Multi-Label Classification* (ICCV 2021) — [arXiv:2009.14119](https://arxiv.org/abs/2009.14119) · [ICCV PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.pdf) · [official code](https://github.com/Alibaba-MIIL/ASL)
- Zhang et al., *Simple and Robust Loss Design for Multi-Label Learning with Missing Labels* (Hill/SPLC) — [arXiv:2112.07368](https://arxiv.org/abs/2112.07368)
- Kim et al., *Large Loss Matters in Weakly Supervised Multi-Label Classification* (CVPR 2022) — [arXiv:2206.03740](https://arxiv.org/abs/2206.03740) (supports the memorization-order premise; NB its own rejection schedule ramps *up* over training — §3 caveat)
- Liu et al., *Early-Learning Regularization* (NeurIPS 2020) — [arXiv:2007.00151](https://arxiv.org/abs/2007.00151)
- Cole et al., *Multi-Label Learning from Single Positive Labels* (CVPR 2021) — [arXiv:2106.09708](https://arxiv.org/abs/2106.09708)
- Zhao & Gomes, *Evaluating Multi-label Classifiers with Noisy Labels* — [arXiv:2102.08427](https://arxiv.org/abs/2102.08427) (context only: despite the title it studies training-time robustness, not noisy-val model selection — the mis-ranking claim rides on Northcutt)
- Northcutt et al., *Pervasive Label Errors* (NeurIPS 2021 D&B) — [arXiv:2103.14749](https://arxiv.org/abs/2103.14749)
- Weber et al., *Automated Focal Loss for Image based Object Detection* — [arXiv:1904.09048](https://arxiv.org/abs/1904.09048) (closest published feedback-driven **downward**-γ precedent — γ = −log(p̂) falls as confidence grows; single-label detection, mechanism-level support only)
- Ghosh et al., *AdaFocal: Calibration-aware Adaptive Focal Loss* (NeurIPS 2022) — [arXiv:2211.11838](https://arxiv.org/abs/2211.11838) (secondary feedback-driven-γ precedent; single-label calibration)
- *Label Wave* (ICLR 2024) — [OpenReview CMzF2aOfqp](https://openreview.net/forum?id=CMzF2aOfqp) · [arXiv:2502.07551](https://arxiv.org/abs/2502.07551) (validation-free memorization-onset signal via training-set prediction-fluctuation minimum; optional §5 channel, single-label evidence)
- Cheng & Vasconcelos, calibration of asymmetric multi-label losses (CVPR 2024) (formal grounding for §5's proper-scoring-rule claim)
- *Calibration of XMC models* — [arXiv:2411.04276](https://arxiv.org/abs/2411.04276) (isotonic regression + calibration@k at extreme scale; §5 post-training default)
- Huang et al., *Robust Asymmetric Loss for Multi-Label Long-Tailed Learning* (ICCVW 2023) — [arXiv:2308.05542](https://arxiv.org/abs/2308.05542) (documented next-loss-family fallback, not adopted)
- Huang et al., *Asymmetric Polynomial Loss* (ICASSP 2023) — [arXiv:2304.05361](https://arxiv.org/abs/2304.05361) (context, not adopted)
