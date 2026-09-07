# V2 Augmentation Review — judged research sweep (2026-07-31)

> **Reading status:** supporting research. Start with the
> [consolidated augmentation decision record](../v2-augmentation.md) for adopted policy,
> pending proposals, and the latest cross-round disposition. Historical recommendations
> below do not override [the production plan](../v2-plan.md).

**What this is.** A six-family literature sweep for additional image augmentations compatible with
(a) the locked detached-ASL loss (γ_neg=7, γ_pos=0, clip=0.05, binary focal gate at
`loss_functions.py:269`), (b) multi-label tagging over the 19,294-tag booru vocabulary, and
(c) anime tag semantics. Every researcher finding was adversarially judged before inclusion:
citations verified against the actual papers, tag-safety claims stress-tested against
`vocabulary.json`, loss-interaction claims checked against `loss_functions.py`. A completeness
critic then audited the whole sweep. 37 candidates were evaluated; 15 survived in some form,
22 are documented dead ends. Scholarly sources only.

**Status: research deliverable, awaiting user review. Nothing here is adopted**, with one
decision now recorded: §4.5 consistency regularization is **DECLINED** (user, 2026-07-31; also
logged in v2-plan §11). Several survivors conflict with recorded v2-plan decisions (§5);
adopting any of them is a plan amendment, not a config edit. The 2026-07-31 adjudication
corrections are folded in below in place.

---

## 1. The one systemic correction the judges forced (read first)

Multiple researchers claimed "augmentation that erases evidence of a labeled tag manufactures a
missing positive that γ_neg=7 amplifies." **The judges verified this against the code and it is
wrong-sided.** Per `loss_functions.py`: γ_neg gates only labeled *negatives*
(`(1 - targets_for_focal)`, L329). A labeled tag whose evidence an augmentation removed has
target=1 and trains through the γ_pos=0 shortcut (`pos_weights = targets_for_focal`, L317–319):
a **full-weight, ungated positive BCE pull toward predicting a tag with no visible evidence** —
wrong-positive pressure that teaches hallucination, the same direction as V1's measured
operating-point collapse. No focal mechanism attenuates it at any γ.

So every augmentation must be scored on **two channels**:

- **Positive branch (ungated):** label kept, evidence removed → full-weight hallucination training.
  This is the binding constraint on occlusion, cropping, and severity-stacked degradation.
- **Negative branch ((p−0.05)₊^7):** augmentation makes an *unlabeled* tag visually true → the
  model's correct perception is punished hardest exactly when it is most confident. This is the
  binding constraint on anything that *mimics a vocabulary tag* (§4.2).

Also confirmed for the whole sweep: any augmentation keeping hard {0,1} targets leaves the L269
focal gate untouched — the decisive structural advantage of everything below over the
MixUp/CutMix family, whose fractional targets contaminate *both* branches simultaneously
(four cross-terms per tag, verified against L278/L302–303/L319/L329).

**Code-state precondition for any ablation:** the V2 loss config is still unapplied in yaml
(clip 0.2, detach off — v2-plan §12 item 1). Any aug A/B run before that lands measures the
augmentation against the wrong loss.

---

## 2. Survivors — adopt-grade tier (judge: recommend, with corrections)

Ranked by risk-adjusted value. All three keep hard binary targets and require no loss change.

### 2.1 Letterbox translation jitter (geometric) — recommend / judge REVISED
Place the letterboxed content at a random offset inside the canvas instead of centering;
translate the padding mask rigidly with it. No content pixel is ever removed → label-exact,
same safety class as the verified hflip. Evidence: DeiT III's mild-crop-beats-RRC at 13M-image
scale (ECCV 2022); Bouchacourt et al. (NeurIPS 2021) — translation is the single most
load-bearing component of standard geometric aug.
**Judge corrections:** (1) it is *milder* than DeiT III's SRC, so cite Bouchacourt, not the SRC
table; benefit is plausible-but-unbounded, not "tenths of mAP". (2) Geometry: under the no-clip
constraint, slack exists only along the padded (short) axis (~128px at 448); the long axis has
zero slack for 93.5% of images. Either accept 1-axis jitter or add 4–8px reflect-pad slack
(SRC-style). (3) *(adjudicated 2026-07-31, supersedes the earlier fold-into-warp advice)*
Implement as a randomized integer paste offset at letterbox time, **never** folded into the
rotation warp: letterboxing is a centered integer `canvas.paste` plus a tensor-slice mask
(`dataset_loader.py:317-323`), and rotation acts afterward on the full canvas+mask without
assuming centering — folding translation into the warp saves nothing (rotation resamples the
whole canvas either way) while rebuilding tested machinery. The Arrow cache is metadata-only
(images letterbox per-sample per-epoch in `__getitem__`), so per-epoch random offsets need zero
cache work. Draw the offset in `__getitem__`/`process_image_cpu`, not inside
`apply_random_rotation`, to preserve the pinned RNG draw count in `test_rotation_mask.py`.
**Open hazards:** conflicts with an explicit v2-plan rejection (§5 item 1) and needs the
border/letterboxed-tag audit (§4.2).

### 2.2 Downscale-only scale jitter, s ∈ [0.75, 1.0] within letterbox — recommend / judge CONFIRMED
Resize content to s× its normal fit inside the fixed canvas, growing the masked padding
(LSJ restricted to its downscale branch; Ghiasi CVPR 2021, ViTDet ECCV 2022, NaViT NeurIPS 2023).
No content removed, proportions isotropic-invariant → framing, counting, and proportion-ordinal
tags all safe. s ≥ 0.75 keeps content ≥ ~0.65× source scale so small-evidence tags stay above
detectability; no upscale branch (s > 1 clips content and, on this corpus, reveals only
interpolated pixels). Distribution anchored at s=1.0 (FixRes apparent-size argument), p≈0.3–0.5;
compose with rotation as a single warp when both fire (translation stays a paste offset, §2.1).
**Only candidate whose ASL analysis the judge marked exactly right per the code.**
**Adjudicated phase correction (2026-07-31):** the s ≥ 0.75 detectability bound is
*source-relative* and only reproduces at 448 (448/512 × 0.75 ≈ 0.656). At 320, content already
sits at 0.625× source before any jitter — below the bound, zero downscale headroom (even s=0.85
lands at 0.531×). The only internally consistent setting is **off in phase 1; s ∈ [0.75, 1.0]
in phases 2–3 only**, which confines the op to ~23% of training samples (phase-table estimate).
The §4.2 mimicry audit and the step-0 rule (§5 item 1) still gate adoption.
**Open hazard:** at s≈0.75 the result is content windowboxed in grey on all four sides — the
literal visual definition of the in-vocab `letterboxed`/`pillarboxed`/`border` tags (§4.2). Needs
that audit before adoption, same as 2.1.

### 2.3 Rotation-range extension to at most ±7–8° — recommend / judge REVISED (capped)
The two scholarly anime-domain multi-label works (Chen & Zwicker Danbooru tagger; Lan et al.,
Sensors 2023) independently converge on exactly the live stack — hflip + small rotation + mild
crop + **no hue** — which is strong negative-space confirmation that nothing obvious is missing.
The one supported knob is widening rotation from ±5°.
**Judge correction (decisive):** the researcher's ±10–15° proposal is KILLED by `dutch_angle`
(high-frequency, ~rank 380): a 10–15° canvas rotation *is* a dutch angle — manufactured
camera-tilt evidence on dutch_angle-negatives, punished by the γ_neg branch precisely when the
model correctly perceives the tilt. Cap at ±7–8° (below dutch-angle perceptibility, consistent
with the yaml's own RandAugment M=9 note), keep p≈0.3 and mask co-rotation, and engage the
in-repo 5-vs-8 rationale in the config comment before changing it.

---

## 3. Survivors — conditional tier (gated experiments, not defaults)

### 3.1 JPEG recompression — conditional / judge REVISED (was recommend)
Round-trip through JPEG at random quality; simulates the actual deploy distribution
(web-recompressed images; corpus itself is Pillow q95/4:4:4). Ehrlich ICCV-W 2021: clean-trained
models pay a real penalty on compressed inputs, largely recoverable by training on them.
**Judge-corrected parameters:** quality ∈ [70, 95] (floor raised from 60), 4:2:0 subsampling only
at q ≥ 80 (chroma halving is the dominant color-evidence risk — the "flat regions keep mean
chroma" defense fails for *small* color evidence like eyes), p ≤ 0.25, applied after geometric ops.
**Verified vocab collisions:** `jpeg_artifacts` (33,995), `blurry` (179,644), `lowres` (53,553).
The clip=0.05 "safe harbor" is an outcome to monitor, not a bound — so either zero the loss on
the ~5 degradation meta-tags for augmented samples (needs per-sample-per-class masking, which
the loss does not currently have) or gate adoption on per-tag AP for those tags.
**Precondition:** first measure the current model's penalty on a JPEG-recompressed copy of the
sealed EVAL slice. If negligible, do not ship. (This doubles as the falsification harness §4.3.)

### 3.2 Random Erasing re-enable — conditional / judge REVISED (two families converged)
Both the occlusion and domain researchers independently landed on re-enabling the existing
disabled implementation; both judges kept it conditional. Direct multi-label precedent:
Query2Label (ASL-variant loss + Cutout 0.5 + RandAugment, COCO SOTA) — compatibility evidence,
not gain-size evidence.
**Converged parameters:** p = 0.25, scale (0.02, 0.05), ratio (0.3, 3.3), keep mean-fill and
pre-normalization placement. Honest benefit: 0 to +0.3 mAP plus occlusion robustness; nearly
free since the code path exists.
**Judge-added conditions (all mandatory):**
- **Censor-tag mimicry** (the family-wide booru-specific hazard no researcher saw): a flat
  rectangle over anatomy is manufactured `bar_censor`/`censored`/`mosaic_censoring` evidence on
  censor-negative images in an NSFW-heavy corpus (all high-frequency: ~ranks 121/257/307). Add
  the censor family to per-tag-group monitoring, or switch to a custom uniform-noise fill
  (torchvision `value='random'` is unusable pre-normalize per the existing comment at
  `dataset_loader.py:1877`).
- Monitor small-evidence tag groups (jewelry, eye-detail) per Balestriero/Kirichenko; abort if
  group F1 drops while aggregate mAP is flat. At the recommended scale, a ~30px evidence region
  is fully covered in ~1–1.5% of image-epochs (decorrelated across epochs, unlike persistent
  label noise — that decorrelation is the qualitative safety argument).
- Either clamp erase boxes to the content region (mask-aware sampling) or document the ~29%
  padding no-op dilution of effective p.
- Fix the config drift first: yaml live values (0.35 / 0.007–0.03) disagree with code defaults.

### 3.3 Downscale-upscale cycle, scale ∈ [0.5, 1.0], mixed kernels — conditional / judge REVISED
Thumbnail/re-host simulation (BSRGAN ICCV 2021, APISR CVPR 2024 degradation models). Hue-safe in
flat regions (not at edges — Lanczos/bicubic overshoot shifts local hue on hard line edges).
**Judge additions:** never nearest-kernel upscale (`pixelated` 897 / `pixel_art` 6,399 mimicry);
shared severity budget with JPEG (p_total ≤ 0.3, never max-JPEG + max-downscale on one sample);
same meta-tag mask-or-gate as 3.1. **Ship JPEG first; add this only if the recompressed-EVAL
harness shows residual resolution-axis penalty.**

### 3.4 Gamma/tone-curve jitter — conditional / judge REVISED (tightened)
One more entry in `IndependentColorJitter`: identical monotone curve on R,G,B jointly.
Channel-order preservation makes hue-*sector* flips impossible, and judge-computed intra-sector
drift at the proposed bounds is ~1–2° (aqua-vs-green safe — the M2 adjacent-color mode survives,
but because the drift is tiny, not because of the sector argument).
**Judge corrections:** tighten to γ ∈ [0.85, 1.18] log-uniform; make gamma and brightness
mutually exclusive per sample (shared tone budget); gate on a luminance-coded-pair eval slice
(§4.1 — this correction implicates the *live* jitter too, which is the more important finding).
Cheapest candidate to implement; smallest expected benefit.

### 3.5 Restricted-op-pool TrivialAugment — conditional / judge REVISED
Replace hand-tuned per-op probabilities with TA's one-op-per-image draw over a curated pool
(rotate/shear/translate at capped magnitudes + existing bounded brightness/contrast; every
color-semantics op dropped). Published support for pruned pools retaining benefit: TA ICCV 2021
ablations; RandAugment per-op ablation (several color ops negative on average, posterize
consistently hurts); MedAugment.
**Judge corrections:** explicit rotate cap = current ±5° inside the pool (`dutch_angle` again);
translate ≤ 2–3% canvas; name sharpness excluded; hflip and blur stay outside the draw; the A/B
must control *total augmentation intensity* — one-op-per-image likely lowers net per-image aug
vs the current independently-fired stack, so a naive A/B confounds policy shape with intensity.
Mask co-transform machinery exists only for rotation (`apply_random_rotation`,
`test_rotation_mask.py`) — shear/translate need the analogous code + test before this is even
runnable. Larger refactor, small expected gain; lowest urgency of the conditional tier.

### 3.6 WSD decay-phase aug anneal-off — conditional / judge REVISED (ablation arm only)
Anneal stochastic aug probabilities (not magnitudes) toward zero across the WSD decay phase;
finish on clean images. Central evidence is arXiv-only (He et al. 2019); DeiT/DeiT III/AugReg all
use constant policies — honest counter-evidence. Cheap (probability knobs already exist).
**Judge corrections:** key the anneal to the WSD decay *fork*, not global step — with multiple
decays from stable-phase checkpoints and the resume machinery, schedule state must be
checkpointed fork-relative or a re-forked decay silently gets the wrong aug strength. Exempt
hflip (label-exact; annealing it only shrinks diversity). Whitelist the resulting loss-statistic
drift in asl_telemetry/stop_conditions monitoring or it will read as a training anomaly.
Strongest project-specific rationale: the final model is calibrated on clean images
(CALIB split), so consolidating on clean inputs aligns train with the calibration
operating point.

### 3.7 Parked / experiment-only
- **Union-label hard mixing (RSB/LogicMix-style OR targets)** — the only mixing variant that
  keeps L269 valid by construction, but the judge showed the researcher's high-alpha-blend
  variant is internally inconsistent (minority-partner tags asserted at target 1.0 with ~20%
  alpha evidence = manufactured full-weight wrong-positives, *worse* per-tag than fractional),
  and the counting-tag failure (union(1girl,1girl)=1girl on a two-figure composite; 2girls
  unreachable by any OR scheme) is unfixable without per-sample-per-class loss masking the
  codebase does not have. LogicMix's missing-positive repair channel is the one interesting
  idea, but at 0.18% positive rate the partner-carries-the-tag probability is negligible for
  tail tags. Park unless the masking infrastructure gets built for other reasons.
- **RegMixup-style auxiliary branch** — the only structurally ASL-safe mixing integration
  (clean branch runs the locked loss untouched; aux branch is separate plain BCE). Zero
  multi-label precedent, ~2× step cost. Only justified if a measured style-shift/calibration
  problem appears on CALIB/TEST. Judge CONFIRMED as honest.
- **Batch repeated augmentation (DeiT repeated-aug)** — judge CONFIRMED the park: pays off only
  in overfitting-limited regimes; V1 measured zero train/val gap. Revisit only if V2 develops a
  gap. Note: "20-line change" underestimates ResumableSampler integration.
- **Mild bounded RRC (area ≥ 0.85, framing-tag-gated)** — judge kept conditional but weakened:
  the label-gate fails open on under-labeled images (label-conditional gating in a
  missing-positive-dominant corpus concentrates damage exactly where labels are worst — a
  general principle worth remembering), and translation+scale jitter already capture the
  load-bearing RRC components without either noise channel. Adopt only if it beats 2.1+2.2 in a
  controlled ablation.
- **Line-art elastic warp (Bezier-pivot, sketch lineage)** — the only family with published
  gains specifically in line-drawn domains (Sketch-a-Net IJCV 2017; SSDA Neurocomputing 2021),
  but at 20K-image single-label scale. Judge added `bad_anatomy`/`bad_hands`/`bad_proportions`/
  `bad_perspective` mimicry (warped hands = manufactured evidence for real in-vocab quality
  tags) and notes it is "one adverse monitoring result from KILLED." Mask co-warp is new
  engineering. Lowest priority; likely never.

---

## 4. Cross-cutting findings (these change how everything above is applied)

### 4.1 The live jitter has an unaudited exposure the sweep found by accident
The photometric judge's decisive point: the family design rule "hue is sacred, luminance is
free" is **wrong for this vocabulary**. `white_hair` (503,877) vs `grey_hair` (490,380),
`dark_skin`/`tan`/`pale_skin`, `light_blue_hair` vs `blue_hair` are distinguished substantially
or purely by *lightness* — among the largest tags in the vocabulary — and the already-live
brightness 0.22 / contrast 0.15 jitter has never been per-tag audited against these pairs.
Action independent of any new augmentation: add a luminance-coded-pair slice to per-tag-group
validation monitoring.

### 4.2 The vocabulary-mimicry audit is a mandatory pre-adoption step for ANY augmentation
Verified pattern across four independent judge findings: an augmentation whose visual artifact
*is* an in-vocabulary tag manufactures wrong-evidence on tag-negatives, punished by the
(p−0.05)₊^7 branch exactly when the model correctly perceives the artifact.
Confirmed instances (all tags verified in `vocabulary.json`):
- rectangular erasing → `censored`, `bar_censor`, `mosaic_censoring`
- rotation > ~8° → `dutch_angle`
- JPEG/downscale/noise → `jpeg_artifacts`, `blurry`, `lowres`, `aliasing`, `pixelated`
- warping → `bad_anatomy`, `bad_hands`, `bad_proportions`, `bad_perspective`
- **letterbox/scale/translation jitter → `letterboxed`, `pillarboxed`, `border`,
  `black_border`/`grey_border`/`white_border`, `grey_background`** (critic finding — applies to
  the two top *recommended* candidates. Adjudicated 2026-07-31: the shielding is stronger than
  first written. Padding is attention-masked end-to-end (pmask passed at `train_direct.py:2162`
  → `pixel_to_token_ignore`, `model_architecture.py:629` → key-padding mask, consumed as a
  flex_attention BlockMask in training; the SDPA bool-mask path runs only in ONNX mode), and
  under axis-aligned
  letterboxing pure-pad tokens are *always* excluded (pad fraction 1.0 ≥ threshold 0.9). The
  1.02% figure in the `apply_random_rotation` docstring (`dataset_loader.py:338-342`) measures
  the **old static-mask rotation error**, not letterbox leakage — do not cite it here. The real
  residual is **boundary-straddling tokens below threshold 0.9, which see grey fill**; that
  residual is what justifies the border-tag monitoring. Caveat: the rotating-mask fix is
  **uncommitted** (HEAD still has the static mask; `test_rotation_mask.py` untracked) — commit
  it before any plan doc leans on it.)
- live Gaussian blur (p 0.15, σ≤1.0) → probably sub-threshold for `blurry`, but belongs in the
  monitoring set
**Rule for the future: before adopting any augmentation, grep the vocabulary for tags that
describe the augmentation's own artifact.** Mitigations, in order of strength: per-sample loss
masking of the mimicked tag family on augmented samples (infrastructure does not exist yet —
`AsymmetricFocalLoss` has only per-sample `sample_weights` and class-level `ignore_indices`),
per-tag AP gates on the mimicked family, artifact kept below perceptibility bounds.

**Implementation spec for the (B,C) mask, if it is ever built (adjudicated 2026-07-31):**
apply it inside `MultiTaskLoss.forward` — verified as the sole route to the criterion
callsites. `ignore_indices` column-filters logits/targets *before* the loss
(`loss_functions.py:260-263`), so the (B,C) mask must be sliced by the same cached keep-mask
(mirror the `class_weights` pattern at `loss_functions.py:362-366`). The reduction is a plain
`.mean()`, so a masked mean is required to keep loss scale invariant across batches. And one
prerequisite the loss work does not cover: the batch carries no per-sample augmentation flags
today — dataset/collate plumbing must land before a (B,C) mask can even be constructed.
Companion edits if the `MultiTaskLoss.forward` signature changes: the second construction and
callsite in `tools/run_validation_for_epoch.py:188-189/:254`, and the monkeypatch in
`test_softstop_resume_e2e.py:298-307`.

### 4.3 No falsification plan existed — build the degraded-EVAL harness first
By the researchers' own predictions, clean mAP will not move for the robustness candidates
(3.1/3.3), so their claimed benefit is unmeasurable as proposed. Required before any of §3.1/3.3
ships: JPEG-recompressed and downscale-upscaled copies of the sealed EVAL slice as permanent
robustness benchmarks, and a measurement of the *current* model's penalty on them. If the
penalty is already negligible, the candidates are moot. This is also the cheapest first action
in the whole review.

### 4.4 Per-phase parameters and the composed-stack budget (critic; unanswered by the sweep)
- Every candidate was analyzed at phase-2 448 only, but the plan's spine is a 320→448→512
  ladder with a phase-scheduled aug table and per-phase drop_path (0.20/0.15/0.10).
  Detail-destroying ops are resolution-relative: severity floors negotiated at 448 are ~1.6×
  harsher at 320, where most epochs run. Any adopted op needs a per-phase parameter column
  mirroring the plan's own table format.
- The survivors were judged one-at-a-time, but four occupy the same detail-destruction axis
  (scale jitter × down-up cycle × JPEG × live blur) and rotation appears in three places.
  Required at adoption time: mutual-exclusion groups — at most one detail-destroying op per
  sample from a shared severity budget (BSRGAN-style), and an explicit statement of whether
  drop_path is co-tuned (AugReg: aug and stochastic depth trade off in the same regularization
  budget; the plan currently fixes drop_path).
- TrivialAugment's safety argument is its exactly-one-op property, which is voided if layered
  on top of always-on independent jitters. TA-as-composed was never scored, only TA-in-isolation.

### 4.5 Families the sweep did not cover (critic; statuses recorded per bullet as of 2026-07-31)
- **Consistency regularization (FixMatch/UDA/Mean-Teacher style) — DECLINED (user decision
  2026-07-31; recorded in v2-plan §11).** No teacher model and no second loss term in V2 (the
  parked RegMixup aux branch, §3.7, would need a plan amendment on the same grounds if its
  measured-problem trigger ever fires). The
  mechanism note stands for the record: the consistency target is a prediction, not a label, so
  the L269 precondition and both label-noise channels do not apply, and an EMA teacher's
  weak-view prediction can partially shield missing positives from the negative branch (a
  denoising channel adjacent to SPLC). But the original costing was wrong: the plan's weight
  EMA (α=0.9998) is a plateau-gate *signal*, not a ready-made Mean Teacher — it is itself
  unimplemented (v2-plan §12: the only EMA in the repo is telemetry scalar smoothing), and the
  teacher's forward pass on a second view is new compute — minimal Mean Teacher ≈ +33% FLOPs
  (~25% throughput loss); FixMatch/UDA-style two-view ≥ 50%. Declined per this bullet's own
  else-branch. (Superseded detail, 2026-08-01: v2-plan §6 has since cut the weight EMA outright,
  so there is no EMA in the run to repurpose as a teacher — this decline only hardens.)
- **Generative augmentation:** the project already owns this track (v2.1.1 Anima gold-training
  pipeline) and it is the *only* augmentation-adjacent intervention that attacks the dominant
  noise mode (missing positives) rather than merely perturbing pixels. One concrete verified
  hazard to carry into v2.1.1: **`ai-generated` is a live tag** — synthetic images entering the
  train pool labeled 0 manufacture systematic wrong-negatives on it; labeled 1, they teach an
  Anima-style shortcut. That labeling decision needs to be made explicitly in the v2.1.1 doc.
- **PatchDropout-style token dropping:** judge-KILLED, but the critic argues the kill hit the
  candidate's false compute claim (mask extension ≠ token removal; no FLOPs saved), not the
  mechanism. True token *removal* (gather to a shorter sequence) renders no fill pixels — the
  censor-mimicry hazard of §3.2 cannot occur at all — can reuse the prod token-ignore
  *decision* logic (`model_architecture.py:627`; the gather itself is new work, see the filing
  correction below), and has *negative* cost (PatchDropout WACV 2023; FLIP CVPR
  2023). Residual risk (dropped evidence of a labeled tag → ungated positive pull) is the same
  kind and magnitude as Random Erasing at matched area, which survived. The train/serve token-
  density mismatch has a built-in anneal path (phase-3 512 native + WSD decay). **Flagged as
  wrongly-killed-in-part; deserves a proper re-evaluation if occlusion-style regularization is
  wanted without the fill-artifact hazard.** Filing correction (adjudicated 2026-07-31): any
  re-evaluation is **V2 model-side work**, not a v2.1.1 item — v2.1.1 is the gold-data pipeline
  doc and owns no model changes. True token removal is an architecture/forward-pass change:
  `model_architecture.py:627` is the token-ignore threshold lookup inside the attention-mask
  pipeline; the sequence is never gathered or shortened, and repo-wide grep finds zero
  token-removal machinery.

### 4.6 Aug config is invisible to resume compat checking (adjudicated 2026-07-31)

Applies to every knob in this review. The checkpoint compat diff (`training_utils.py:471-488`)
covers only num_labels / patch_size / architecture_type / grad-accum / image_size plus ad-hoc
normalize/color-order checks — **zero augmentation keys**. Editing any `data.*` aug knob and
resuming silently changes training behavior mid-run, which corrupts exactly the
controlled-ablation discipline this review's adoption path depends on. Required alongside any
adopted knob (now a v2-plan §12 item):

- Put aug keys in the **warning** tier, not critical — a strict-critical mismatch on
  `resume_from=latest/best` is caught at `train_direct.py:1108-1115` and **silently discards
  the checkpoint, restarting from scratch** (the image_size footgun, again).
- Gate the check on **same-phase resume**: the plan legitimately changes aug values at every
  phase boundary via flip-phase-and-resume; a naive check fires on every intended transition.

---

## 5. Plan reconciliation (critic; blocking for adoption)

The sweep was run as research; the plan is decisions-only and currently claims "no open research
questions remain." Conflicts that must be resolved *in the plan* before any survivor ships:

1. **`todos/v2-plan.md:375-376` explicitly rejects translation jitter** ("unmeasurable at
   current val size, and a confound on the ASL measurement"). The sweep's top recommendation
   contradicts this without having engaged either objection. Both are answerable — EVAL is
   146,056, and the confound objection holds only if the aug is introduced mid-run (adopt from
   step 0 or not at all) — but answering them is a plan amendment.
2. **`todos/v2-plan.md:369` sets random erasing to "none" across all three phases.** §3.2
   reopens that line.
3. **`todos/v2-plan.md:389` records SpliceMix as the "if a mixing method is ever wanted"
   fallback.** The sweep CONFIRMED-rejected SpliceMix (counting-tag corruption of head tags +
   small-evidence destruction at 224-effective resolution; IEEE TMM 2025's own framing). That
   plan line should be struck or re-pointed at the RegMixup-aux pattern (§3.7), which is the
   only mixing integration that survived structurally.
4. **Single-configuration rule:** the plan runs one config; every §2/§3 item is an ablation
   candidate by nature. The honest routes are: (a) adopt a small label-exact set (2.1–2.3)
   from step 0 as part of the single config, with the §4.2 audit done first; or (b) route the
   review's data-side candidates to v2.1.1 and ship V2's aug table exactly as written
   (model-side items — the §4.5 PatchDropout flag — stay V2-side per §4.5). Splitting the
   difference mid-run is the one option the plan's own confound objection forbids.

---

## 6. Dead ends (judge-confirmed; do not re-litigate)

All rejected with citations verified; one-line reasons here, full analyses in the workflow
transcript (session scratchpad).

| Candidate | Fatal reason |
|---|---|
| MixUp (soft λ targets) | Fractional targets contaminate both ASL branches (four cross-terms/tag); ghost composites OOD for line art; hue/counting corruption; zero upside at zero train/val gap |
| CutMix (area-λ labels) | Manufactures the dominant noise pathology by construction; SpliceMix (IEEE TMM 2025) itself rejects patch-paste for MLIC |
| Saliency/attention-guided mixing (SaliencyMix, PuzzleMix, Co-Mixup, TransMix, TokenMix) | Pays extra compute to make the counting-tag violation *systematic*; fractional targets; single-label only |
| SpliceMix | Best-credentialed MLIC mixing, still loses: counting-tag corruption of head tags + small-evidence destruction at 224-effective; also composite layouts visually instantiate `multiple_views`/`comic`/`2koma`/`border` labeled 0 |
| Cutout at Q2L parameters (factor 0.5) | 80 large-object classes → 19K tags with tens-of-pixels evidence does not transfer; whole tag families sit inside the footprint |
| GridMask | Guaranteed spatial coverage = manufactured-wrong-positive maximizer for tiny-evidence tags; CNN-era gains never reproduced in ViT recipes |
| Hide-and-Seek | Localization-completeness benefit has no consumer here; the multi-label objective already forces meticulousness |
| Saliency-guided erasing (KeepAugment lineage) | One-salient-region premise false in dense multi-label anime images; heavy engineering for nothing |
| Standard RandomResizedCrop | Manufactures wrong-positives on clipped labeled tags (full-weight positive branch) + missing-positives on newly-true tags; multi-label SOTA (ML-Decoder) already omits aggressive RRC |
| Shear ≤8° | Near-zero increment over live ±2–5° rotation; proportion/pose-tag risk without evidence of gain |
| Elastic/grid distortion (generic) | No evidence outside handwriting/sketch/medical; GPU cost; mask engineering; style-tag adjacency |
| Aspect-ratio jitter ±5% | Safe magnitudes are sub-threshold; effective magnitudes manufacture adjacent-proportion noise; NaViT favors AR preservation |
| Gaussian/sensor noise | Deploy images are digital-native; real corruptions are JPEG+resize (covered); Geirhos narrow-transfer says target those directly |
| Posterize/solarize/invert/equalize/autocontrast/hue/color ops | Each is a label flip or a no-op on flat-color illustration at safe parameters (op-by-op audit) |
| Chromatic aberration aug | No lens at deploy; CA in the wild is deliberate — and tagged (`chromatic_aberration`, 17,354) |
| Frequency-aware per-tag aug strength | Ill-defined per-image strength under co-occurring head+tail tags; curriculum statistic unestimable at this sparsity |
| AugMix + JS consistency | Buys corruption robustness/calibration the objective doesn't price, at 3× compute + novel loss next to a locked ASL |
| Soft Augmentation (magnitude-conditioned target softening) | Binary-target precondition + ill-posed per-tag softening; value is as citable justification for keeping aug mild enough that labels stay exactly true |
| MixStyle/pAdaIN feature-space style mixing | ~Zero within-domain by MixStyle's own ablation (train and eval are both booru); unauditable color-tag corruption risk |
| Screentone/halftone overlay | `screentones` (2,136) is a live tag — manufactured evidence on negatives; fatal by the §4.2 rule |
| PatchDropout-as-mask-extension | Killed as specified (no compute saved; masking ≠ removal) — but see §4.5 for the fixable re-formulation |

---

## 7. Suggested reading order for review

1. §1 (mechanism correction) and §4.2 (mimicry rule) — these two reframe everything.
2. §5 (plan conflicts) — decides whether anything ships in V2 at all or the whole review
   routes to v2.1.1.
3. §2 (the three adopt-grade candidates) against §4.4's stacking caveats and §4.6's
   resume-compat requirement.
4. §4.3 — the degraded-EVAL harness is the cheapest no-regret first action regardless of
   every other decision.
5. §4.1 — the live-jitter luminance-pair audit is the second no-regret action.
6. §3 at leisure; §6 only to confirm you agree the dead ends are dead.

---

*Provenance: 13-agent workflow (6 researchers + 6 adversarial judges + completeness critic),
run 2026-07-31, ~900K tokens, 308 tool calls. Judges verified citations by web search, tag
claims against `vocabulary.json` (19,294 tags), and loss claims against `loss_functions.py`.
Full per-agent transcripts: session workflow directory (`wf_843f0541-785`).*
