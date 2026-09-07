# V2 Augmentation Review — Round 2 (2026-08-07)

> **Reading status:** supporting research. Start with the
> [consolidated augmentation decision record](../v2-augmentation.md) for adopted policy,
> pending proposals, and the latest cross-round disposition. Historical recommendations
> below do not override [the production plan](../v2-plan.md).

> **Status: research deliverable, awaiting user review. Nothing here is adopted; nothing
> self-executes.** Four items need a user decision (table below). Everything else is either a
> no-regret cleanup or a recorded rejection.

**Scope.** A second judged research sweep for image-augmentation regularization, scoped strictly
to what the round-1 review (2026-07-31) did **not** cover, plus a full audit of the live and
planned stack. Five researcher families (spectral; uncovered photometric; true token removal;
uncovered geometric; 2023–2026 recent literature), each adversarially judged; two stack
auditors; one final judge over everything. Scholarly sources only; every citation
judge-verified; tag claims verified against `vocabulary.json`; code claims verified against the
repo. Method and provenance: see the Appendix.

**How to read this document**

- **R1 §x** = a section of the round-1 doc, `todos/reviews/v2-augmentation-review.md`.
  Plain **§x** = this document.
- Action tags: **[CODE]** repo change · **[PLAN]** v2-plan amendment (the plan is
  decisions-only) · **[DOC]** comment/doc correction, zero behavior change ·
  **[MONITORING]** per-tag-group AP rider (all land via §5.7).
- Fast review path: decisions table → headline table → §8 action order. Sections 2–7 hold the
  evidence behind each line.

## Decisions required (step-0, decide once before launch)

| # | Decision | Recommendation | Detail |
|---|---|---|---|
| **D1** | Add the 4 chirality tags to `ignore_indices` (mapped by name, never frozen index) | **ADOPT** — cost ~zero, mechanism exists, strictly better than training on corrupted labels | §5.1 |
| **D2** | Sharpness / unsharp-mask jitter | Conditional behind gates G1–G5; honest expectation is G1(b) fails — that is the gate working | §2.1 |
| **D3** | True random token removal | **DECLINE for V2**; record as the pre-adjudicated v3 occlusion route | §2.2 |
| **D4** | Reopen the phase-1 aug-intensity direction question? | Only as an explicit v2-plan §7 direction question, if §7 is reopened deliberately | §3.2 |

## Headline results

| Question | Answer |
|---|---|
| New candidates evaluated | 26 → **0 adopt-grade, 2 conditional, 3 parked, 21 killed** |
| Conditional survivors | sharpness/unsharp jitter (§2.1); true token removal (§2.2) |
| Live stack (6 ops) | **all KEEP their values**; no removals |
| hflip "losslessness" | **falsified** — 4 chirality tags (~6.2K occurrences); fix is `ignore_indices` by name (D1) |
| Rotation widening to ±7–8° (R1 §2.3) | **DO_NOT_ADD** (both audits, convergent grounds) |
| Round-1 conditional tier | Random Erasing (R1 §3.2), gamma (R1 §3.4), TrivialAugment (R1 §3.5), WSD anneal (R1 §3.6) → **rejected-for-V2**; JPEG (R1 §3.1) stays conditional with a tightened gate; down-up (R1 §3.3) unchanged |
| v2.1.1-routed jitters | routing stands (user decision); **do not bundle** into v2.1.1's measured config |

Zero adopt-grade is the judged conclusion, not a failure: V1's measured zero train/val gap
(micro-F1 0.70044 seen vs 0.70088 held-out) prices the regularization value of nearly
everything at ~0, and the mimicry rule (R1 §4.2) kills nearly everything else in a 19K-tag
vocabulary that names its own artifacts.

---

## 1. Binding rules (carried from round 1, applied throughout)

1. **Two-channel ASL scoring** (R1 §1). Positive branch: evidence removed under a kept label is
   a full-weight ungated hallucination pull (γ_pos=0, `loss_functions.py:317–319`). Negative
   branch: an artifact that makes an unlabeled tag visually true is punished by (p−0.05)₊^7
   (`:327–329`) exactly when the model perceives it correctly.
2. **Vocabulary-mimicry rule** (R1 §4.2). Before adopting any augmentation, grep the vocabulary
   for tags that describe the augmentation's own artifact.
3. **Composed-stack severity budget** (R1 §4.4). Detail-destroying ops share one axis; at most
   one per sample.
4. **Resume-compat** (R1 §4.6). Any adopted aug key needs a warning-tier, phase-gated compat
   entry.
5. **Single-config rule** (R1 §5 item 4). Step-0 plan amendment or v2.1.1 routing — nothing
   mid-run. The v2.1.1 route covers data-side items only; model-side work is step-0-or-nothing.

---

## 2. New candidates — CONDITIONAL

Both are step-0 plan amendments or nothing. Neither is recommended for adoption as-is; the
gates define what adoption would require.

### 2.1 Sharpness / unsharp-mask jitter → decision D2

**Verdict: conditional — the only photometric survivor.** It earned it: no in-vocab tag names
the oversharpening artifact (verified: `sharp*` vocabulary hits are anatomy only); flat-region
color drift is provably zero (the operator is zero-mean off-edge); both protected pair types
(hue-adjacent, luminance-coded) are untouched.

Gates — all mandatory:

- **G1 (precondition, two-part).** (a) Name the consumer that actually presents sharpened
  inputs — only the HF Spaces public path qualifies; the janitor and v2.1.1 consume the same
  Pillow q95 corpus files the model trains on. Then (b) measure the current model's penalty on
  a sharpened copy of the sealed EVAL slice (R1 §4.3 harness). Negligible penalty → do not
  ship.
- **G2.** Step-0 adoption only, as a v2-plan §7 amendment; warning-tier phase-gated aug-key
  compat entry (R1 §4.6).
- **G3.** Joins the detail-destruction axis (R1 §4.4) with **mutual exclusion vs Gaussian
  blur** per sample; factor s ~ U[0.7, 1.5], p = 0.15, cap s ≤ 1.3 at the 320 phase.
- **G4.** Per-tag AP gates on `jpeg_artifacts` (33,995), `aliasing` (931), `scan_artifacts`,
  `outline` (37,120), `white_outline` (21,999), `blurry` (179,644).
- **G5.** The config comment must answer R1 §3.5's "name sharpness excluded" note (that was
  pool-hygiene, not an adjudication) and cite AutoAugment honestly (sharpness appears in 1 of
  24 ImageNet sub-policies, paired with Invert).

Priority: below JPEG (R1 §3.1; see §6). Honest expectation: G1(b) likely fails — and that is
the gate working.

### 2.2 True random token removal → decision D3

**Verdict: conditional; final-judge recommendation is decline-for-V2.** With both value legs
weak (PatchDropout random-init +0.16%; zero V1 train/val gap; ~2% compute), decline at step 0
and **record this as the pre-adjudicated v3 occlusion route** if a train/val gap ever emerges.
With Random Erasing reclassified rejected-for-V2 (§6), this is the only occupant of the
occlusion slot if one is wanted at all. User decision; the gates below are what adoption
requires.

What it is: gather-to-shorter-sequence token dropping (PatchDropout, WACV 2023; FLIP,
CVPR 2023), fixed K per phase. This is the sanctioned R1 §4.5 re-evaluation of the round-1
kill, not a re-proposal (R1 §6's own kill row defers to §4.5). Unique properties: the
negative/mimicry channel is **structurally zero** (nothing is rendered — no fill pixels, no
censor-bar hazard), color safety is exact, and compute cost is negative.

Judge-corrected facts (binding):

- The compute leg is **~2% aggregate**, not "10–15% wall-clock" (ρ=0.9 confined to phase 2
  ≈ 17% of ~400M samples; attention ≈ 16% of FLOPs at 448 per the plan's own estimate). The
  pad-waste-recovery framing belongs to the parked pad-token variant (§3.1), not to uniform
  sampling.
- Single-token (≤16px) evidence exposure at ρ=0.9 is **10%/epoch ≈ 10× the rate accepted for
  Random Erasing** → the small-evidence tag-group monitoring from R1 §3.2 is mandatory; ρ=0.9
  is a hard floor; phase 1 runs ρ=1.0 (or ≥0.95); phase 3 runs ρ=1.0 (FLIP-style
  density-consolidation tail).
- Preconditions: V2 loss config applied first (binding rule 1 — any A/B before that measures
  the wrong loss); rotation-mask fix committed; warning-tier aug key (R1 §4.6); sampling
  **uniform over all tokens** — the pad-first hybrid is banned (aspect-correlated
  regularization strength; see §3.1's design trap).
- Engineering is real: the repo has token-ignore threshold logic
  (`model_architecture.py:627`) but no gather/shorten machinery; fixed-K batching and the
  flex_attention BlockMask path both need work.

---

## 3. New candidates — PARKED

### 3.1 Deterministic pad-token removal

Mathematically exact under the double-sided mask (`model_architecture.py:532–536`; head reads
`x[:,0]` at `:703`) — but it is a throughput optimization, not an augmentation, and the savings
are batch-max-dependent (~0 in mixed-aspect batches without aspect-bucketed sampling / NaViT
packing; serve side needs export work — `ONNX_Export.py:82` pads square).

**Revisit triggers:** §2.2 lands (free rider), or aspect bucketing is built for other reasons.

**Recorded design trap:** pad-first drop order makes content-drop strength anti-correlated
with pad fraction → aspect-correlated → composition-tag-correlated. Sampling must stay
uniform.

### 3.2 Phase-1 aug-intensity direction question → decision D4

The one researcher/judge disagreement of the round; resolved in the judge's favor (final judge
re-verified `v2-plan.md:420–428`). As proposed it was fatal: derived against the yaml phase-2
column as if it were the all-phase baseline, silently reversing the plan's recorded hot-first
decision ("ship this table as written," re-affirmed 2026-08-01) while deleting the plan's
strongest aug cell (P1 brightness 0.30→0.11 etc.) — an intensity cut confounded with schedule
shape, incoherent with EfficientTrain's own ramp-to-full design.

**The legitimate residue that is parked:** EfficientTrain/EfficientTrain++ (ICCV 2023 /
TPAMI 2024, verified, previously uncited) plus R1 §4.4's own 320-severity critique (severity
floors negotiated at 448 are ~1.6× harsher at 320) put real evidence pressure on the hot-first
phase-1 choice. **Reopen only** as an explicit v2-plan §7 direction question, with parameters
re-derived against the actual P1 column and total intensity controlled.

### 3.3 Anime-domain negative-space survey

The 2023–2025 scholarly anime literature contains no technique the live stack lacks — the
negative-space confirmation of R1 §2.3 extends.

**Mandatory correction before citing:** the "dissertation p.19" quote is Chen & Zwicker
**WACV 2022** (arXiv 2108.01819 §4.2) — the same lineage already cited in R1 §2.3, so the
"three works converge" count is really two lineages plus Yi (IJCNN 2023), and Yi is silent on
hue. The ±15° rotation datapoint there is a smaller-vocab precedent, not a license (this
vocabulary has `dutch_angle` at 87,987).

---

## 4. New candidates — KILLED

Append these to R1 §6; do not re-litigate.

**Spectral family — one systemic kill.** On flat-color cel art every spectral method
decomposes as: low band = color op (the DC bin IS per-channel mean color), high band =
line-art degradation op, mid band = no-op. There is no fourth thing. This decomposition kills
the family and all descendants.

| Candidate | Fatal reason |
|---|---|
| FDA / FACT amplitude swap | DC bin is per-channel mean color: every β>0 is a hue op on the exact pixels carrying aqua-vs-green evidence; FACT also needs a declined co-teacher loss |
| APR amplitude-phase recombination | Its stated objective (amplitude invariance) is color-blindness on a vocabulary where amplitude IS the label (`white_hair` 503,877 vs `grey_hair` 490,380) |
| AFA Fourier-basis gratings | Stripe/lattice mimicry (`striped` 45,932, `polka_dot` 47,939, `plaid` 12,938) + per-channel chromatic gratings (code-verified) + declined dual-batch auxiliary-loss structure |
| RandConv | 1×1 case is an unbounded random hue rotation; edge-map draws mimic `sketch` 103,028 / `monochrome` 426,265; published gains need declined KL consistency |
| PRIME | Three-primitive ensemble whose every primitive maps onto an existing kill (color-curve ops / elastic / saturated R1 §4.4 detail axis); corruption target unpriced on a digital-native corpus |
| Band-pass / high-pass / low-pass filtering | High-pass destroys all color evidence under kept labels (APR-class); low-pass beyond σ1.0 walks into `blurry` 179,644; the one safe residue duplicates a saturated R1 §4.4 axis |

**Photometric**

| Candidate | Fatal reason |
|---|---|
| Random grayscale | Output IS the definition of two top-102 tags (`monochrome` 426,265, `greyscale` 348,367) with 348K genuine positives coexisting → contradictory supervision; erases every color label's evidence at target=1 |
| White-balance / color-temperature jitter | The neutral axis kills it: any effective shift d tints white/grey hair (S ≈ 2d/(1+d)) and manufactures `sepia`/`*_theme`; safe d is a no-op |
| CLAHE | Absolute-luminance→local-rank remap attacks the vocabulary's largest luminance-coded pair (white/grey −16…−18% separation, judge-reproduced); strictly-worse variant of dead equalize |
| Fancy PCA | At published strength an 18×-weaker duplicate of live brightness jitter; at effective strength it becomes white-balance jitter |
| Film grain / ISO noise | The operator's name is a live tag (`film_grain` 9,114); the round-1 Gaussian-noise kill in structured clothing |
| Palette quantization / dithering | Identity on cel flats; banding in gradients = `limited_palette` 12,217 / `dithering` / `gradient_*` mimicry (>200K gradient-labeled instances flattened at target=1); posterize dichotomy with a smarter clusterer |
| Sepia | Named after the tag it manufactures (5,557); annihilates hue evidence for the largest label mass; mild blend = white-balance jitter, full strength = worse than grayscale |

**Token / patch**

| Candidate | Fatal reason |
|---|---|
| Block-wise token removal | Box-geometry full-cover of small evidence without Random Erasing's area cap + awakens the absence channel (whole-figure excision); MAE's own ablation says random > block |
| PatchShuffle | Reproduces mosaic censoring's literal generative process (`mosaic_censoring`, 139,769 occurrences) on an NSFW corpus; strongest R1 §4.2 violation on record |

**Geometric**

| Candidate | Fatal reason |
|---|---|
| Vertical flip | Output satisfies `upside-down` (18,332) by definition; inverts gravity evidence for ~1.97M pose occurrences; no sub-threshold band exists for a flip |
| 90° rotation | Trains `standing` 647,379 / `sitting` 716,450 from horizontal figures; anime is maximally orientation-canonical (RotNet) |
| Perspective warp | No viewpoint process at deploy; perceptible δ manufactures `bad_perspective`/`dutch_angle`, sub-perceptible is value-free; absent from every tuned-policy pool |
| Fisheye / barrel distortion | Artifact IS the `fisheye` tag (3,779); no lens at deploy; worst mask-engineering bill (curved letterbox boundary increases the R1 §4.2 boundary-token residual) |
| Motion / zoom blur | Nine-plus-tag ~510K-occurrence mimicry field topped by `blurry` 179,644; the domain draws motion (`motion_lines`/`speed_lines` are ink) so the invariance conflicts with evidence the tagger must keep sharp |
| Subpixel translation | Dominated: integer jitter over ≥16px exhausts all residues mod patch 16; LANCZOS fit + bicubic rotation already randomize kernel phase; boolean pmask makes fractional shift ill-posed (reintroduces the pixel/mask-disagreement class the rotation fix removed) |
| Radial centered zoom | Non-distinct by exhaustive four-branch decomposition (= routed scale jitter \| dead upscale/RRC \| composition of two routed ops \| fisheye); decomposition recorded to close the zoom family permanently |

**Post-round adjudication (2026-08-07, adversarial debate — advocate/critic/judge): the
label-MODIFYING vertical-flip variant.** Proposal: rot180 (lossless double flip, NOT vertical
mirror) at small p with `upside-down` programmatically set to 1 — the rescue of the vflip kill
above via label edit; not scored by either round. **Verdict: KILL-for-V2.** The mimicry prong
genuinely dissolves: judged against the fetched Danbooru wiki, the minted label is TRUE
("Inverted; the wrong way up"; no canvas/subject sense split exists at 180°, unlike
`sideways`/`from_side` at 90°; `rotated` (563 posts, absent from this vocab) is defined as the
*correcting* third-party edit). Killed on the remaining grounds: (1) benefit is
sub-instrument-floor on every **built** surface — sealed EVAL/CALIB and both corpus consumers
are 100% canonical-orientation, and EVAL's ~450 natural-mode positives all have upright frames,
so the installed frame-rotation cue has zero measurable support (the huge-effect surface,
rot180(EVAL), is the still-unbuilt R1 §4.3 harness); (2) the sole real consumer — pixel-baked
inverted HF Spaces uploads, prevalence unmeasured — is strictly dominated by EXIF auto-orient +
inference-time 4-way canonicalization, which corrects all 19,288 outputs at zero training risk
(even a maximally favorable gate outcome selects that remedy, so no gate can save the aug);
(3) the kill row's gravity prong transfers at reduced scale — ~1,970 truthful-label
inverted-evidence pose events/epoch at the defended p=0.001 (0.64× the D1 chirality channel
this round paid to eliminate) — with no per-sample-per-class masking rung available if
monitoring ever trips. Judged NOT destructive at p≤0.1% (wrong-positive edge case bounded by
the fitted ρ̂≈1.4% → ~260 unlabeled genuinely-inverted images → ~18 corrupt presentations per
run; synthetic fraction 24% of the tag); destructive territory begins near the tag's own base
rate p≈0.31%, where synthetics become the majority sense and hidden-stratification risk lands
on the natural mode the instrument measures. **Recorded for v3:** a separate S4L-style
orientation head outside the semantic vocabulary — the architecture used by the entire
published rotation-prediction pedigree (RotNet/S4L/AugSelf) — delivers upload canonicalization
compositionally; the 180° semantics finding is reusable if a mode-split target is ever added.
**Flip condition:** measured non-zero inverted fraction in Spaces uploads (a week of 4-way
self-vote telemetry after EXIF auto-orient ships) flips KILL→PARK for the
canonicalization-vs-v3-head choice only — no measurement revives in-vocab injection.

**Recent literature**

| Candidate | Fatal reason |
|---|---|
| DeiT III 3-Augment grayscale/solarize | Grayscale is the photometric kill above with a recipe pedigree; closes the "just use DeiT III" argument line-by-line |
| Per-class aug policy (CUDA / DODA / Kirichenko intervention) | Re-proposal of R1 §6's frequency-aware row at 2024–25 SOTA; "the class's augmentation" is undefined when head and tail tags share pixels. Keep the Kirichenko citation for the monitoring rider |
| Learnable semantic aug (ISDA, TIP 2024) | The method IS its loss (closed-form expected-CE surrogate) next to a locked ASL; unauditable feature-space color mimicry |
| FlexiViT patch-size randomization | Cleanest two-channel profile ever scored and still dead: locked architecture (fixed conv stem, `model_architecture.py:379–383`), single deploy point, its own paper caps gain at ~zero at native patch size |

---

## 5. Live-stack audit — all six ops KEEP their values

| # | Op (live values) | Disposition | Follow-ups |
|---|---|---|---|
| 1 | hflip p=0.5 | KEEP | falsified-claim fixes + chirality `ignore_indices` (§5.1, D1) |
| 2 | brightness 0.22 @ p=0.5 | KEEP | luminance-pair monitoring + pre-registered fallback (§5.2) |
| 3 | contrast 0.15 @ p=0.5 | KEEP | same as brightness (§5.2) |
| 4 | saturation 0.06 @ p=0.5 | KEEP (ballast) | comment rewrite only (§5.3) |
| 5 | rotation ±[2°,5°] @ p=0.30 | KEEP | commit mask fix; `dutch_angle` AP monitoring (§5.4) |
| 6 | Gaussian blur k3 p=0.15 σ≤1.0 | KEEP | blur-family AP monitoring; stale comment fix (§5.5) |

### 5.1 hflip — KEEP; losslessness record falsified → decision D1

Both audits independently found the same four chirality tags (final-judge verified counts):
`left-handed` 3,489, `left-to-right_manga` 1,526, `right-over-left_kimono` 615,
`right-to-left_comic` 589 — ≈6.2K occurrences, ~0.1% of label mass, 50% evidence-label
decorrelation under flip, plus negative-branch pressure over the `kimono` family (~300K) and
`holding_weapon` (196,644) images.

Changes:

- **[CODE]** Rewrite the false "Every tag in the vocabulary is orientation-agnostic" comment
  at `dataset_loader.py:1820–1822`.
- **[PLAN]** Amend `v2-plan.md:424`: delete the nonexistent "orientation-aware tag swap" (no
  swap exists in code), replace "lossless" with "label-exact except 4 documented chirality
  tail tags."
- **[PLAN — decision D1]** Add the four tags to `ignore_indices` — **mapped by name at load
  time, never by frozen index** (4863/8690/16527/17026 are current-vocab-build indices). This
  resolved the audits' one substantive conflict (one said optional, one recommended); the
  final judge adjudicated **adopt**: cost ~zero, mechanism exists
  (`loss_functions.py:46/:129–134/:260–261`), and the tags are unlearnable under flip anyway —
  ignoring converts 4 corrupted tags into 4 untrained tags, strictly better. Excluded from all
  per-tag AP claims either way.
- **[MONITORING]** Script-text group (`english/chinese/korean_text`, `signature`, `watermark`,
  `clothes_writing`) added to the per-tag-group slice. Presence tags stay true under mirror
  (no `mirrored_text` tag exists — verified absent), so this is feature-quality monitoring
  only.

### 5.2 Brightness 0.22 / contrast 0.15 — KEEP + monitoring

The R1 §4.1 exposure is real: by the audit's analytic model, white/grey boundary crossings
occur in ~13% of `white_hair` samples/epoch, and compound brightness×contrast full-category
shifts in ~0.66% of samples. Mitigations that ground the KEEP: the never-jittered grey pad
anchor (jitter runs pre-letterbox, `dataset_loader.py:2383–2386` — see §7.1), epoch
decorrelation, and V1's 33 epochs at wider values without observed collapse (weak evidence —
see §7.5).

Changes:

- **[MONITORING]** Luminance-pair slices {white/grey/silver hair}, {pale_skin/tan/dark_skin},
  {light_blue/blue hair} + **hue-coded control pair {`aqua_hair` 84,472, `green_hair`
  292,099}** as the differential diagnostic (a luminance-pair drop with a flat hue-pair
  isolates the luminance axis).
- **[PLAN]** Pre-register the fallback in v2-plan §7: **0.22→0.15 / 0.15→0.10 at a phase
  boundary only, never mid-phase**, so a tripped slice has a defined consequence.
- **[DOC]** One-sentence yaml:149 addendum: contrast f=0.85 is ~15% chroma compression — 2.5×
  the saturation op's entire range.

### 5.3 Saturation 0.06 — KEEP as ballast

Exactly hue-invariant: saturation scaling uniformly scales channel differences and cannot
rotate hue — aqua/green is unreachable at any factor.

- **[DOC]** Rewrite yaml:146–151: the "attacks color-named tag identity" warning belongs to
  hue ops, and the "¼ from BYOL asymmetry" derivation is a double misreading (BYOL's own ratio
  is ½ — 0.2 vs 0.4/0.4 — and its asymmetry is per-view blur/solarize probabilities, not
  saturation).

### 5.4 Rotation ±[2°,5°] @ p=0.30 — KEEP; widening to ±7–8° DO_NOT_ADD

KEEP precondition re-flagged: **commit the rotating-mask fix + `test_rotation_mask.py`**
(still uncommitted/untracked; multiple round-2 verdicts assume the fixed semantics).

- **[MONITORING]** `dutch_angle` per-tag AP.

**R1 §2.3's ±7–8° extension: DO_NOT_ADD** — both audits agree on convergent grounds:

- The review's own adoption condition — engage the in-repo 5-vs-8 rationale (yaml:204–208) —
  was engaged and **lost**: spatial precision matters more at 448/512.
- P1 already delivers ±8° for the majority-epoch phase.
- "Below dutch-angle perceptibility" was asserted, never measured; an 8° letterbox tilt is
  plainly perceptible.
- Expected gain is sub-instrument-floor by the same standard that refuted the label-exact
  jitters.

**[PLAN + DOC]** Record the rejection in v2-plan §7 and strike §2.3 from the round-1 adopt
tier so it is not re-litigated.

### 5.5 Gaussian blur — KEEP + monitoring

- **[MONITORING]** Blur-family per-tag AP: `blurry`, `blurry_background`,
  `blurry_foreground`, `motion_blur`, `depth_of_field`.
- **[DOC]** Replace the stale "Phase 1 only — drop for Phase 2" sentence at yaml:221 (the plan
  table and the live keys already agree blur runs in P2/P3 at p=0.15 σ≤1.0; only the comment
  lies).

### 5.6 Config-default drift — FIX

Behavior-preserving under the current yaml; protective if a key is ever lost.

- **[CODE]** `Configuration_System.py`: rotation dataclass defaults 5.0/10.0° → **2.0/5.0°**
  (verified at `:994–995`; the current default range sits at the edge of the killed
  dutch-angle band and validation only enforces ≤45° at `:1117–1118`); saturation 0.1 → 0.06
  (`:980`); blur sigma_max 1.5 → 1.0 (`:1002`).
- **[CODE]** Mirror in `dataset_loader.py` constructor defaults and fallbacks; add explicit
  `gaussian_blur_enabled=False` to the val-dataset constructor (currently blur-free only by
  default value).

**Principle adopted: no code default may exceed the reviewed-safe live value on any axis.**

### 5.7 Monitoring landing site — ADD

The v2-plan L458–460 monitoring riders have no operational home.

- **[DOC]** Add a "Per-tag-group AP slices (v2-plan §7 riders)" section to
  `todos/v2-monitoring.md`: ~20 tags — luminance pairs, hue control pair, `dutch_angle`, blur
  family, letterboxed/border group, script-text group; streaming per-batch,
  threshold-independent AP, direction-only vs the V2 baseline epoch.
- **[DOC]** Add a carve-out in "Things explicitly not to track" (per-tag F1 stays excluded;
  these AP slices are the sanctioned exception). Cross-reference the §5.2 pre-registered
  fallback.

---

## 6. Round-1 conditional tier — reclassifications

| Round-1 item | New status | Core reason |
|---|---|---|
| R1 §2.1/§2.2 translation + scale jitter (v2.1.1-routed) | routing stands; **DO_NOT_BUNDLE** | attribution: v2.1.1's yardstick can't absorb a confound |
| R1 §2.3 rotation widening ±7–8° | **DO_NOT_ADD** | see §5.4 |
| R1 §3.1 JPEG recompression | CONDITIONAL (gate tightened) | consumer-existence gate added |
| R1 §3.2 Random Erasing | **REJECTED-for-V2** | monitor-and-abort is unhostable under single-config |
| R1 §3.3 down-up cycle | CONDITIONAL (unchanged) | dormant behind honest gates |
| R1 §3.4 gamma jitter | **REJECTED-for-V2** | smallest benefit on the most fragile axis |
| R1 §3.5 restricted TrivialAugment | **REJECTED-for-V2** | one-op safety property voided when composed |
| R1 §3.6 WSD aug anneal | **DEAD-for-V2 by structure** | ablation-arm-only in a plan with no ablation arms |

Details:

- **Translation/scale jitter (v2.1.1).** The routing is a user decision and stands untouched.
  But v2.1.1's yardstick is an off-diagonal metric on a small sealed clean slice plus a
  no-regression gate; bundling an aug change destroys attribution in both directions, and the
  effect size is sub-floor even on 146K EVAL. **[DOC]** Record in
  `v2.1.1-gold-training-pipeline.md` that both jitters are PARKED there, adoptable only in a
  dedicated aug-ablation arm (this also closes the doc-sync gap — the v2.1.1 doc never
  absorbed the routing). If ever un-parked, translation goes first (no mimicry surface, no
  phase gating).
- **JPEG (R1 §3.1).** The R1 §4.3 harness gate has a hole — it measures model penalty on
  recompressed inputs, not whether any consumer presents them, and this project's primary
  consumers (janitor, v2.1.1) run on the same Pillow q95/4:4:4 files the model trains on. New
  two-part gate: (1) name a consumer that actually presents recompressed inputs (the HF Spaces
  path is the only plausible one), then (2) run the harness at that consumer's quality
  distribution. The harness itself stays the cheapest no-regret build regardless.
- **Random Erasing (R1 §3.2).** Its mandatory safety conditions are monitor-and-**abort**, and
  under the single-config rule an abort is a mid-run knob change — the op is unhostable by
  V2's own discipline. The mitigation ladder is blocked at every rung (no (B,C) masking; noise
  fill unusable pre-normalize), and the value prices a phantom at zero train/val gap. Preserve
  the censor-mimicry finding as family precedent. Revisit only if per-sample-per-class masking
  is built for other reasons. (This reclassification also mooted the token judge's "coherence
  with RE" argument for §2.2 — noted there.)
- **Gamma jitter (R1 §3.4).** Its own round-1 entry concedes smallest expected benefit,
  purchased on the R1 §4.1-fragile luminance axis, before the live jitter's own audit has run;
  the mutual-exclusion condition concedes the tone axis is at budget. Revisit only if the
  luminance-pair audit both clears the live jitter and shows a measured tone deficit.
- **Restricted TrivialAugment (R1 §3.5).** R1 §4.4 voids the one-op safety property when
  layered on always-on jitters; honoring it means replacing a table the plan just re-affirmed;
  the curated pool contains an R1 §6 dead end (shear); the required intensity-controlled A/B
  is unhostable under single-config. Keep the RandAugment per-op-ablation citation as support
  for the existing exclusions.
- **WSD aug anneal (R1 §3.6).** Ablation-arm-only in a plan with no ablation arms; perturbs
  stop_conditions' instruments at the decision-critical window; the clean-calibration
  rationale is already satisfied offline by v2-plan §8.3. Preserve the
  fork-relative-schedule-state caution as a general resume note.

---

## 7. Cross-cutting findings (new invariants and required corrections)

### 7.1 The pad-anchor invariant (promote from accident to rule)

Neither round ever scored pad-color/padding-value jitter — and round 2 quietly made its
absence *load-bearing*: the §5.2 defense of brightness jitter relies on the never-jittered
grey pad as a constant absolute-luminance anchor (jitter runs pre-letterbox). **Record as a
rule: the pad is never photometrically perturbed.** This simultaneously pre-kills any future
pad-randomization proposal.

### 7.2 The severity budget (R1 §4.4) still has no implementation spec

Two conditionals now share the detail-destruction axis with a mandated mutual exclusion (live
blur, proposed sharpness), and JPEG/down-up would join it — but no code, config shape, or
sampling scheme for "at most one detail op per sample from a shared budget" exists anywhere.
If any conditional ever passes its gates, this becomes the blocking engineering item.

### 7.3 Bookkeeping corrections (do not propagate)

- **Rank vs count.** Some round-2 sub-reports quoted vocabulary *indices* as frequencies
  (e.g. "~252/~302/~116" for censor tags). True counts verified: `mosaic_censoring` has
  **139,769 occurrences** (rank ~254 — the mimicry is *worse* than the erroneous figures
  implied). No verdict changes.
- **Vocab size.** `vocabulary.json` `tag_frequencies` has **19,288** entries vs the round-1
  doc's "19,294 tags." Reconcile once (special tokens) and stop propagating both numbers.
- **Attribution.** The anime-domain survey's "dissertation p.19" quote is Chen & Zwicker
  WACV 2022 (arXiv 2108.01819 §4.2) — see §3.3.

### 7.4 Trivially-dead families to record by precedent (one line each, in R1 §6)

Never researched because obviously dead, but undocumented — worth one row each so the next
sweep doesn't spend a researcher on them: weather/particle overlays (rain/snow/sparkle — all
in-vocab tags, dead by R1 §4.2), RGB channel permutation (dead by the RandConv 1×1 hue
argument), inpainting-fill erasing (generative fill belongs to the v2.1.1 boundary).

### 7.5 Evidence-quality caveats (carry with the verdicts)

- The §5.2 white/grey boundary model (~V0.85–0.90) and the 13% / 0.66% exposure figures are
  **analytic swatch modeling, not corpus measurement**. Acceptable for KEEP+monitor (the
  monitoring is the measurement); never cite as measured facts.
- "V1 ran wider values without observed pair collapse" is **absence-of-looking, not
  absence-of-harm** — the per-pair audit was never run on V1. Weak supporting evidence only.
- Un-reverified micro-claims, plausible and low-stakes, flagged for the record: the flip's
  per-(image_id, epoch) CRC32 decorrelation (`dataset_loader.py:2184–2229`) and the "47
  `single_*` tags are all count-semantics" enumeration — each asserted by one audit, checked
  by no judge.

---

## 8. Recommended action order (cheapest no-regret first)

- [ ] **1. Commit the rotation-mask fix + `test_rotation_mask.py`** (already v2-plan §12;
  named as a precondition by four separate round-2 verdicts; the test file is still
  untracked).
- [ ] **2. Apply the V2 loss config** (clip 0.05 / detach — v2-plan §12 item 1). Every A/B and
  every two-channel argument in this review assumes it; nothing is measurable before it lands.
- [ ] **3. Comment/doc corrections, zero behavior change:** `dataset_loader.py:1820` hflip
  comment; `v2-plan.md:424` tag-swap wording; yaml:221 stale blur sentence; yaml:146–151 BYOL
  rewrite; yaml:149 contrast-chroma sentence; the §7.1 pad-anchor invariant.
- [ ] **4. Config-default hardening** (§5.6): rotation 2.0/5.0, saturation 0.06, blur σmax
  1.0, mirrors, explicit val-blur off. Protective, invisible under the current yaml.
- [ ] **5. Monitoring landing site in `v2-monitoring.md`** (§5.7): ~20-tag AP slice section +
  carve-out + pre-registered brightness/contrast phase-boundary fallback cross-reference.
- [ ] **6. Review-doc + plan bookkeeping in one pass:** append the §4 kill tables to R1 §6
  (with the spectral and zoom decompositions as standing entries, and pattern-lattice tags as
  an R1 §4.2 entry); apply the §6 reclassifications; amend the JPEG precondition (R1 §3.1);
  record the §5.4 rejection; add the v2.1.1 parking note to the v2.1.1 doc. Every plan-text
  change is an amendment; amend in place per the doc-set rule.
- [ ] **7. Build the degraded-EVAL harness (R1 §4.3)** (JPEG-recompressed + sharpened +
  down-up copies of sealed EVAL, one script). Named "the cheapest no-regret action" in round
  1, re-named as a precondition by two round-2 verdicts, and it **still does not exist** — the
  review process is now two rounds deep in verdicts gated on an instrument nobody has built.
- [ ] **8. Warning-tier aug-key compat check (R1 §4.6)** (already v2-plan §12) — the systemic
  fix for which action 4 is only the bound.
- [ ] **9. Step-0 user decisions before launch** (plan amendments, decide once): **D1–D4** in
  the decisions table at the top of this doc.

*Nothing in this report self-executes. Deliverables that change behavior: actions 1–2
(already-planned commits), 4 (default hardening), 7 (new harness), and any step-0 adoption
under 9. Everything else is documentation that makes the decision record match the code and
the evidence.*

---

## Appendix — final-judge verification record

The final judge independently re-read R1 §1/§2–3/§4.1–4.6/§5/§6 and `v2-plan.md` §7
(lines 418–465), overturned no judge verdict, and resolved the single researcher/judge
disagreement (§3.2 above) in the judge's favor with independent evidence. Spot-checks run
before adjudicating:

- **Vocab greps (~30 tags re-verified exact against `vocabulary.json` `tag_frequencies`):**
  `upside-down` 18,332; `dutch_angle` 87,987; `greyscale` 348,367; `monochrome` 426,265;
  `film_grain` 9,114; `sepia` 5,557; `polka_dot` 47,939; `fisheye` 3,779; `motion_blur`
  17,801; the four chirality tags 3,489 / 1,526 / 615 / 589; `gradient_background` 108,066;
  `gradient_hair` 102,696. Absences confirmed: `scanlines`, `moire`, `rotated`, `color_cast`,
  `wrong_color`, `posterization`, `mirrored_text`.
- **Code claims:** `dataset_loader.py:1820–1824` contains the false orientation-agnostic
  comment verbatim (with the abandoned force_flip design note); `Configuration_System.py`
  default drift confirmed at `:994–995` (rotation 5.0/10.0), `:980` (saturation 0.1), `:1002`
  (blur σmax 1.5), validation ≤45° only at `:1117–1118`; `loss_functions.py:46/:129–134/
  :260–261` — `ignore_indices` column-filters logits AND targets, so the chirality fix
  mechanism exists exactly as claimed.
- **Plan claims:** `v2-plan.md:420` "Phase 1 at full strength" with P1 = 0.30/0.20/0.08,
  rot ±[2°,8°]@0.50, blur p=0.30 σ≤1.5 (confirming the §3.2 wrong-baseline correction);
  line 424's stale "orientation-aware tag swap"; lines 435–436/456–458 translation-jitter
  rejection, "ship this table as written," BY-NAME v2.1.1 routing; lines 458–460 monitoring
  riders with no landing site in `v2-monitoring.md`.

*Provenance: 13-agent workflow (5 family researchers + 5 adversarial judges + 2 stack auditors
+ 1 final judge), run 2026-08-07, ~1.18M tokens, 280 tool calls, run id `wf_8a29455f-427`.
Judges verified citations by web search, tag claims against `vocabulary.json`, loss claims
against `loss_functions.py`, code claims against the repo. Full per-agent transcripts: session
workflow directory.*
