This is a synthesis-and-recommendation task, not a code task. I have the full EVIDENCE (five method dossiers) and the adversarial CRITIQUE. The user wants a 7-part deliverable for an expert ML engineer. I have everything I need to write it directly — no tool calls required. Let me deliver concretely, respecting every non-negotiable.

---

# M4 Hair-Length GT Cleaning: The Non-VLM Plan

## 1. THE CORE METHOD

The winning mechanism is a **gold-anchored, direction-aware fusion of four orthogonal-error signals**, with the **DINOv2-frozen-backbone + CORN/SORD ordinal specialist (M-D)** as the corpus-scale workhorse and **entity-canonical directional-upper consensus (M-B)** as the near-free first pass, both calibrated against a **rubric-first, boundary-and-medium-oversampled human GOLD set** that is the single point where bias-independent truth enters. This breaks the shared convention bias because the bias lives in the *labels* (annotators reflexively tag `long_hair`, skip `medium`, omit `very_long`) — so every Danbooru-derived model (v1, WD-ViT/ML-Danbooru/Camie) learns `image→convention` and its errors are *correlated with the convention by construction* (Xia NeurIPS'20 / Yao NeurIPS'21: instance-dependent + systematic noise is unidentifiable from the biased corpus alone — an external-process anchor is mathematically required). DINOv2 features come from a *different process* (self-supervised, zero tags) and the head is trained on rubric truth, so its residual errors are image-content failures (occlusion, pose, stylization) that are *random w.r.t. whether an annotator typed `long_hair`* — that is exactly the error-orthogonality the VLM had but squandered on poor anime accuracy. M-B's orthogonality is narrower but real: the unidirectional-DOWN bias *structurally cannot fabricate* `very_long`, so the existence of `very_long` on a reliable minority of a character's images is a signal the convention cannot produce — read with a noisy-OR "at-least-one" aggregator (Hoffmann ACL'11), not the mode (the mode launders the bias — diversity-prediction theorem). The **community-delta** (snapshot→current re-pull via `id_index.json`) and **geometry (M-A)** are confirmers, not load-bearing; **noise-robust training (M-C)** is the corpus-wide v2 *training* fix that *consumes* the gold the cleaning track produces. No signal is individually sufficient — all five are *transducers that spend the external gold anchor across the corpus*; M-D and M-E (human routing) dominate, M-B accelerates, M-C/M-A finish.

---

## 2. THE PIPELINE (end-to-end, direction-aware, gold-calibrated)

### 2.0 The GOLD set (the load-bearing anchor — build FIRST)
- **Rubric-first labeling**: reviewers see reference images + the written boundary rubric, **never the existing tags** (tag-anchoring is silent orthogonality collapse). Rubric defines `very_long` = "hair endpoint past hips/waist" and the `short`/`medium`/`long` base-triple cutoffs explicitly enough to kill the coin-flip.
- **Sampling**: **deliberately oversample** the long↔very_long boundary AND the collapsed `medium` (NEVER proportional — proportional inherits the 5.8% medium starvation and is the single failure that invalidates everything; §5).
- **Held-out Goodhart slice**: a second gold partition the *entire pipeline never sees* — sole defense against calibrating clean-looking metrics on contaminated gold.
- **Reviewer QC**: honeypot/MCC gating (existing DataCleaning `screening_*` backbone) before any gold is trusted.
- Size: ~1.5–3K boundary/medium-oversampled images (~15–25 human-hrs). This gold is **shared** across M-B threshold calibration, M-D head training, M-C transition-cell estimation, and M-E fusion calibration — one anchor, four consumers.

### 2.1 Signal A — Entity-canonical directional-upper (M-B) — long→very_long, character head
1. **Character identification**: attach Danbooru categories (category 4 = character) via the tag-category table / `selected_tags.csv` keyed on tag string. The `name_(copyright)` suffix is a *fallback only* (misses bare names `hatsune_miku`, `reimu_hakurei`).
2. **Build bags**: group solo images by single character tag. **EXCLUDE** before estimation: `alternate_hairstyle` / `alternate_hair_length` / `cosplay` / `wig` / `alternate_costume` (the AU set — each rare, ~0.3–1.8%, explicitly tagged → cheap high-precision exclusion). Route multi-char (2girls+, ~19–32%) to a separate low-trust path (length unattributable).
3. **Aggregate UPPER, not mode**: a character is canonically ≥`very_long` iff its non-AU `very_long(+absurdly)` rate clears a **gold-calibrated** threshold (corpus base ~16%; canonical-long chars measured 23–86%). Require bag ≥10–15 instances; **abstain** below. Detect **bimodal bags** (variable-hair/story-haircut chars) → abstain, never propagate.
4. **Anchor head characters** once with community re-pull / wiki / a single human confirm on top-K canonical images (a few thousand chars, paid once).
5. **Emit** per long-only image in a canonical-very_long bag: vote = ADD `very_long`, KEEP `long`. Intermediate bag rate → soft-ordinal (SORD) vote, not hard.
- **Catches**: long↔very_long (upgrade direction only). **Cannot catch medium** (medium is rarely canonical; fence M-B to upgrade-only — never let it strip medium).

### 2.2 Signal B — Community-delta (near-free, temporal-orthogonal)
- Re-pull current Danbooru tags via `post_id == filename-stem` (`id_index.json`, 5.92M). Compute **DELTA** = current minus snapshot length tags. Where the community *since added* `very_long` to a snapshot-long-only image, that is a human upgrade in the M4 direction.
- **Use the DELTA, not the static current tags** (static current is same-site, same-convention = semi-correlated; only the *change over time* is editor-attention-driven and orthogonal). Weight by edit-recency + distinct-editor-count.
- Canonicalize via Danbooru `tag_aliases` before comparison; enforce `tag_implications` closure (absurdly⇒very_long⇒long) on every write.
- **Catches**: long↔very_long confirmation. Feeds both M-B anchoring and the fusion as one vote.

### 2.3 Signal C — DINOv2 specialist ordinal head (M-D) — the workhorse, MEDIUM recovery
1. **Frozen DINOv2 ViT-L/14** (registers) over each triaged image → CLS + mean-pooled patch embedding, cached once. Never trained on tags = orthogonality anchor.
2. **CORN head** (Shi-Cao-Raschka 2023): K−1=5 rank-monotone conditional outputs over `very_short<short<medium<long<very_long<absurdly`. Rank-monotonicity enforces the implication chain *by construction*. Train with **SORD soft targets** (Diaz & Marathe CVPR'19), width from gold confusion spread → fuzzy boundary supervised as honest-soft, not coin-flip.
3. **Grow gold via active learning**: BADGE / core-set (k-center on DINOv2 space) deliberately mines **boundary + medium** cases — re-fit head in seconds (frozen backbone). This is how medium gets enough examples to be learnable.
4. **Decode** to multi-label ops: detect "true rank ≥ very_long while GT long-only" (upgrade) and "true rank = medium while GT short-only or long-only" (medium recovery). Base triple exclusive; modifiers ride along.
5. **(Optional) scale to 6M** via FixMatch / Noisy-Student — **gated**: high-confidence retention only, re-measure held-out off-diagonal each round, **STOP when it stops shrinking** (semi-sup is a laundering engine if the seed head skews — this gate is non-negotiable).
- **Catches**: **short↔medium and medium↔long (THE ONLY signal that can)**, plus long↔very_long (soft). DINOv2-for-base-triple is *documented* (DACoN 2025: 57.5% zero-shot anime line-art part-matching > CLIP 36.7% — so **use DINOv2, not CLIP**); DINOv2-for-the-fine-boundary is treated as soft/abstain.

### 2.4 Signal D — Geometry (M-A) — narrow optional confirmer
- Run **only** on the cleanly-measurable slice: solo + longish + (`full_body` OR `cowboy_shot`) + not(tied/back/seated/chibi) — ~9.4% of the longish-solo pool. Pose (Chen & Zwicker WACV'22 COCO-17) + ISNet char-seg (Qin ECCV'22) + hair parser → R = landmark-normalized hair-extent ratio → rubric cutoffs.
- **ABSTAIN** aggressively: hip off-frame, tied hair (35.7%), back/side/seated/chibi, low pose-confidence. Per-direction confusion matrix (it has a *downward* artifact on occluded hair — opposite the Danbooru up-bias — so it brackets rather than launders; weight asymmetrically).
- **High-value use**: when M-A *agrees* with another orthogonal signal on the clean slice → cross-family evidence strong enough to auto-upgrade long→very_long. Build LAST or skip if budget-tight (multi-week build, ~10% coverage, abstention anti-correlates with the very_long target).

### 2.5 FUSION (direction-aware IBCC/DBCC, gold-calibrated)
- Each signal enters as an **annotator with its own gold-calibrated, PER-DIRECTION confusion matrix** (IBCC/DBCC + ordinal-adjacency + asymmetric priors). Direction-awareness is essential: M-A skews down, M-B/community-delta skew up — symmetric weighting cancels real upgrades.
- Posterior is computed from the **orthogonal fusion** (M-B-upper + M-D + community-delta + M-A), **NEVER from tagger consensus** — if the posterior were tagger-fed, M4 mislabels sit on the diagonal (everyone agrees "long"), look low-noise, and never get queued (the silent killer; §5 second weakest link).

### 2.6 ROUTING (auto-apply / human / abstain)
- **Selector** = Bernhardt (Nature Comms 2022): CrossEntropy(fusion-posterior, observed-label) − self-entropy(fusion-posterior). High label-noise + LOW intrinsic ambiguity → human first; high intrinsic ambiguity (genuine 50/50 boundary) → **abstain** (don't burn human minutes on coin-flips). Use GLAD/CrowdTruth item-difficulty to split "ambiguous-because-rubric-undefined" (fix rubric, then resolvable) from "ambiguous-because-genuinely-50/50" (abstain).
- **Rare-class protection** (ALR, Bhattacharya MICCAI'24): explicitly inject/protect `medium`/`very_long` so the noisiness ranker doesn't flag true-medium as suspicious and let a reviewer strip it toward long.
- **Auto-apply** only when ≥2 orthogonal signals agree in-direction AND the item is low-ambiguity. **ADD `very_long`, KEEP `long`** (never downgrade — downgrades are human-only by construction). **Never force color groups exclusive.** Apply via `apply_corrections_fast.py` (atomic remove+add, encoding-safe), implication-closure LAST.
- **Confusion-cell ownership summary**:
  | Cell | Primary | Confirmer | Direction policy |
  |---|---|---|---|
  | long↔very_long | M-B-upper + M-D | community-delta, M-A(clean slice) | ADD very_long, KEEP long; auto only on ≥2-signal agree |
  | medium↔long | **M-D only** | human (M-E) | recover medium; protect existing medium from removal |
  | short↔medium | **M-D only** | human (M-E) | recover medium; AL must mine these |
  | very_long↔absurdly | human/fusion | — | never auto-write fine call |

### 2.7 How MEDIUM specifically gets recovered (the hardest cell, 5.8%, eaten both sides)
Medium is the cell four of five methods fail. **Only M-D + human gold can recover it**, and only if gold oversamples it:
1. Gold set **oversamples medium** (not proportional) → M-D head sees enough medium to learn short/medium/long as a part-extent distinction from non-Danbooru features.
2. BADGE/core-set AL **deliberately mines medium-boundary** cases each round.
3. M-D fires `medium` where GT omitted it (short-only or long-only images whose true rank = medium); SORD makes short↔medium↔long adjacency cheap so the head isn't punished for honest near-boundary uncertainty.
4. Human (M-E) adjudicates the M-D medium candidates the selector surfaces; **ALR protects** existing `medium_hair` from removal (the non-negotiable).
5. **Defer aggressive medium recovery on Danbooru-only signals** (M-B, community-delta, taggers all under-tag medium — per the non-negotiable). Medium recovery is **gold-and-specialist-bound or it does not happen.**

---

## 3. WHY THIS BEATS THE VLM PATH (directly)

- **The VLM is anime-OOD and spatial-blind**; this path never asks a model to *perceive* hair length per-image where that breaks. DINOv2 features are *documented* to carry part-level anime line-art semantics (DACoN 57.5%); the human makes the relative-spatial call once per cluster/character against the rubric (BLINK human ceiling ~96% vs VLM ~50% on relative-spatial); M-B never looks at a pixel (pure GT co-occurrence, immune to all perception failure).
- **The VLM had its OWN systematic bias** (the 112K pilot: 85% of its "long" calls were really "medium" — overshoots upward). This path's orthogonal signals have *uncorrelated* error sources, and direction-aware fusion handles each signal's directional skew explicitly rather than trusting a single biased oracle.
- **Coverage**: the VLM pilot reached ~6.83% after real effort + had an NSFW-moderation coverage hole. M-B covers the character-tagged majority (~62%+) at near-zero per-image cost; M-D scales to 6M as an offline batch job with no moderation hole; community-delta covers 5.92M post-ids for free.
- **Cost**: orders of magnitude cheaper per image (no API/rate-limit), and the gold anchor is reused four ways instead of paying per-image VLM inference.
- **Control**: the VLM's correlated-error channel (its training distribution) is *uncontrollable*; M-D's only correlated-error channel (the gold set) is the *one thing you fully control* — making it the strongest non-circular anchor available.

---

## 4. PRIORITIZED PLAN (cheapest/highest-leverage signal first → v2 retrain)

| # | Step | Auto/Human | Yields | Trap it avoids |
|---|---|---|---|---|
| **0** | **Build rubric-first GOLD** (boundary+medium-oversampled) + held-out Goodhart slice + honeypot/MCC reviewer gating | **Human** (~15–25 hr) | The shared orthogonality anchor for all 4 consumers | **Convention-contaminated gold** — the single failure that invalidates everything (§5). NO proportional sampling; NO tag-anchored labeling. |
| **1** | **Community-delta re-pull** via `id_index.json`; canonicalize via `tag_aliases`; enforce `tag_implications` | **Auto** (I/O hours) | Free temporal-orthogonal upgrade votes + provenance | Treating static current tags as a clean independent oracle (semi-correlated — use the DELTA only). |
| **2** | **M-B entity-canonical directional-upper** (categories → bags → exclude AU → upper-aggregate → anchor head chars) | **Auto** + **Human** confirm on ~few-K head chars (~50–170 hr) | Long→very_long upgrade on the character majority, near-zero/image | **Entity-MODE launders the bias** — use directional-upper; abstain on bimodal/small bags; fence to upgrade-only. |
| **3** | **DINOv2 embeddings** (cache once, triaged subset) + **M-E clustering** (TypiClust on DINOv2) for the characterless/OC residual | **Auto** embed; **Human** labels 1 typical/cluster (~20–170 hr) | Cluster-level labels for OCs (~13%) + rare chars | CLIP substrate (downgrade, 36.7%); propagating low-purity clusters (route those to per-image). |
| **4** | **M-D specialist**: CORN/SORD head on gold; grow via BADGE/core-set mining boundary+medium | **Auto** train; **Human** AL labels | **MEDIUM recovery** + corpus-scale ordinal votes | Gold contamination (silent collapse); over-trusting argmax on fine boundary (keep soft). |
| **5** | **Fusion + active-cleaning routing** (IBCC/DBCC direction-aware; Bernhardt selector on ORTHOGONAL posterior) | **Auto** rank; **Human** adjudicate top queue (~20–40 hr) | Auto-applied high-confidence corrections + human-resolved tail | **Tagger-fed posterior** → defect on diagonal, never queued (§5 #2). Self-entropy abstaining the whole defect (fix rubric, then resolve). |
| **6** | **M-A geometry** on the ~9% clean slice as cross-family confirmer (OPTIONAL) | **Auto** | Auto-upgrade where geometry agrees with another orthogonal signal | Symmetric weighting of its downward artifact; running it broadly (~80% abstain). Build last or skip. |
| **7** | **Apply corrections** via `apply_corrections_fast.py` (atomic, closure last); spot-audit each propagated batch on held-out gold before commit | **Auto** + **Human** audit | Cleaned GT (head of corpus) | Propagation amplifying one bad label across thousands — require ≥2-signal corroboration + batch spot-audit. |
| **8** | **v2 retrain (M-C)**: fix `loss_functions.py:269` → invert gamma_neg 7→0–2 on fine-end length tags (SPLC/Hill) → SORD ordinal sub-head → gold-anchored single-cell forward-correction (Patrini) + T-Revision | **Auto** | v2 emits the true distribution corpus-wide; covers the silent majority cleaning can't reach | Self-estimated T (launders bias with false consistency proof — gold-anchor the off-diagonal ONLY); soft labels silently re-binarized if L269 not fixed first. |

Long tail (rare/OC not reached by head cleaning) is **deferred to the v2 soft-ordinal retrain**, capping human cost at the head (~150–400 hr total vs >25 person-years naive).

---

## 5. ROLE OF NOISE-ROBUST TRAINING (M-C): complement, not substitute

**It is a COMPLEMENT, and it is a category mismatch with the user's stated goal if sold as a substitute.** The user's goal is explicitly to **clean the poison GT on disk** and use the model as a *suggestion engine*. M-C does the opposite: it leaves the 6M poisoned sidecars untouched and trains a v2 whose *predictions* are de-biased.

Reconciliation:
- **What M-C does well (the "better suggestions" half)**: the cheap loss-geometry fixes are *do-this-first* and domain-agnostic. `gamma_neg=7.0` literally rewards reproducing the under-tag (it makes the model maximally confident an absent-but-true `very_long` is a true negative) — inverting it toward 0–2 / applying SPLC (treat absent fine-end tags as *unobserved* not *negative* when `long` is present) stops the v2 amplifying the bias. SORD makes the model honestly-soft at the fuzzy boundary instead of confidently-wrong. **Fix `loss_functions.py:269` first** (`targets_for_focal = targets` re-aliases before smoothing; soft targets leak into the focal gating masks at lines 314–315 and are silently clamped to hard — every soft scheme is void without this).
- **What M-C does NOT do (the "clean the GT" half)**: the on-disk sidecars stay poisoned; any future retrain/export/downstream consumer re-inherits the bias. **M-C is not a clean dataset; it is a clean-ish model over a dirty dataset.**
- **Why T-correction can't stand alone**: Patrini's consistency holds only under class-conditional noise P(noisy|true,x)=P(noisy|true). M4 violates this twice — instance-dependent (the boundary is a feature-dependent perceptual judgment) AND systematically directional. Xia'20/Yao'21 *prove* instance-dependent T is unidentifiable without external structure → a corpus-self-estimated T (Dual-T, VolMinNet, anchor-point, Li 2023) **deconvolves to the convention** and trains the bias in *with a false proof of correctness*. The only admissible T estimates **one dominant off-diagonal cell** (rho10 for `very_long`, secondarily the two medium cells) **from the external gold set** (GALC-SLR-style), conditions the flip on `long`-present (reduces instance-dependence to a sub-population), and accepts it corrects the **aggregate rate, not per-image** — acceptable only because the user wants suggestions.

**Net**: run M-C *last*, as the corpus-wide v2 training fix that **consumes the gold + high-confidence corrections** the cleaning track (steps 0–7) produces. Cleaning and robust-training are **multiplicative**: cleaning supplies the mathematically-required external anchor; robust-training spends it across all 6M including the silent majority cleaning never reaches. The GT cleaning (steps 0–7) is what satisfies the user's literal goal; M-C makes the suggestion engine itself stop recommending `long`-only.

---

## 6. SUCCESS METRICS & STOPPING

**Primary (on the held-out Goodhart gold slice the pipeline never saw):**
1. **Off-diagonal shrinkage** of the per-direction confusion matrix — specifically `P(predict long | true very_long)` and `P(predict short/long | true medium)` must fall toward gold rates. **Stop when the off-diagonal stops shrinking** between iterations (and that is the semi-sup gate in step 4 and the M-C T-Revision stopping rule).
2. **Asymmetry check on the off-diagonal**: if the held-out confusion is *symmetric*, the feature/label path is clean; if it's *directionally skewed toward long*, the bias is leaking through (gold contamination or feature confound) — investigate before trusting any auto-apply.

**Medium recovery:**
3. **Medium share** rising from ~5.8% toward the rubric-true rate on gold (measured on the held-out slice, not the training gold). Track short→medium and medium→long *separately* (two cells).
4. **Medium false-removal floor = 0**: the non-negotiable "protect existing medium" — audit that no auto-apply ever removed a `medium_hair` tag.

**Per-source orthogonality checks:**
5. **Pairwise error-correlation** between signals on the held-out gold: M-B-upper, M-D, community-delta, M-A errors should be *uncorrelated* (low off-diagonal agreement on *wrong* answers). If two signals are highly correlated in their errors, one is laundering the bias — demote it in fusion.
6. **"All-signals-said-keep" false-negative probe**: a held-out subset where every automated signal voted "no change" — measure the true M4 miss rate to bound the silent floor (the defect the triage structurally can't see).
7. **Implication-closure violations** stay ≈0% post-write (absurdly⇒very_long⇒long); **color-group exclusivity never forced** (heterochromia/multi-color preserved).

**Goodhart guard (overarching):** every threshold/weight is calibrated on the *training* gold; *all* success is measured on the *held-out* slice. If training-gold metrics improve while held-out stalls → you're fitting the gold, not the truth — stop and re-sample gold.

---

## 7. TOP RISKS & OPEN DECISIONS FOR THE USER

**Risks (ranked):**
1. **Convention-contaminated gold = total silent collapse** (the §5 single failure). If gold is tag-anchored, proportionally sampled, or rubric-unstable, M-B/M-D/M-C/M-E *all fail invisibly and simultaneously* and you ship a v2 that confidently reproduces the bias over a corpus you believe you cleaned. **Mitigations are non-negotiable, not best-effort**: rubric-first + boundary/medium-oversampled + held-out Goodhart slice + all-signals-said-keep probe + honeypot/MCC gating.
2. **Tagger-fed selector posterior**: if the Bernhardt score uses tagger consensus, M4 mislabels sit on the diagonal and never queue — the triage structurally can't see its own target. Posterior MUST come from the orthogonal fusion.
3. **Semi-supervised (FixMatch/Noisy-Student) laundering**: if M-D's seed head skews, high-confidence retention propagates "long" across 6M. Gate on held-out off-diagonal shrinkage; stop when it stalls. Consider skipping the 6M semi-sup stage entirely and relying on M-C for corpus-scale.
4. **Propagation amplification**: one bad cluster/character-canonical label multiplies across thousands. Require ≥2-signal corroboration before bulk write + spot-audit each batch on held-out gold.
5. **M-A engineering cost vs yield**: no off-the-shelf anime hair-class segmenter (multi-week build) for ~10% coverage with target-anti-correlated abstention.

**Open decisions for the user:**
- **A. Char-identification source**: do you have/can you pull the Danbooru tag-category table (category 4)? The `name_(copyright)` suffix heuristic alone silently drops bare-name characters (`hatsune_miku`, `reimu_hakurei`) — a real M-B coverage hole. **(Blocks step 2.)**
- **B. DINOv2 backbone choice**: stock natural-image DINOv2 ViT-L (documented via DACoN) vs a LoRA light-domain-adapt on unlabeled Danbooru if base-triple boundary accuracy is marginal. Start stock; adapt only if held-out gold says so.
- **C. Build M-A geometry at all?** It's a multi-week build for ~10% coverage. Recommendation: **defer/skip** unless you specifically want the cross-family auto-upgrade on the clean full-body slice.
- **D. 6M semi-sup scale (step 4 optional) vs lean entirely on M-C (step 8)** for corpus coverage — both spend the gold; M-C is cheaper and you're retraining v2 anyway. Recommendation: **lean on M-C**, treat semi-sup as optional.
- **E. Medium recovery aggressiveness**: the non-negotiable says *defer* medium recovery on Danbooru-only signals. Confirm medium recovery is **gated to M-D + human gold only** this iteration, with M-B/community-delta fenced to the long→very_long upgrade.
- **F. Gold budget split**: how many of the ~150–400 head-cleaning hours go to gold-set construction (step 0) vs head-character confirmation (step 2) vs queue adjudication (step 5)? Gold is the load-bearing dependency — recommend front-loading it.

**Non-negotiables honored throughout**: ADD `very_long` / KEEP `long` (never remove long); only base triple exclusive; never force color groups exclusive; protect existing `medium_hair` from removal; defer medium recovery on Danbooru-only signals; do not trust v1/flip-TTA to find its own bias; re-binarize loss `L269` before any soft labels.