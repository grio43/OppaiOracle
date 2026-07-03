# Near-Duplicate Detection: Pipeline Simplification Review

Companion to [near_duplicate_detection_thresholds.md](near_duplicate_detection_thresholds.md). That file is a threshold reference — facts only. This file synthesizes peer-reviewed evidence on the meta-question: **for a 5.4M-anime-image multi-label ViT trained from scratch, is a multi-stage pHash → SSCD cascade actually warranted, or is a simpler approach better-supported by the literature?**

Scholarly sources only (NeurIPS, CVPR, ICLR, TMLR, ECCV, ICML, JMLR, peer-reviewed journals, arxiv preprints from those communities). No homebrew anime tagger projects.

---

## 0. TL;DR for planning

Three findings the literature actually supports, and one it doesn't:

1. **For supervised multi-label classification, dedup's strongest justification is train/val leakage, not training-set quality.** Generative-model and LM-side dedup motivations (memorization, regurgitation, privacy) do not transfer cleanly to discriminative classifiers. Direct classifier-side ablations are sparse and mostly show duplicates are *redundant* (compute waste), not toxic.
2. **Modern dataset-curation papers do not use pHash as a prefilter.** DINOv2, SemDeDup, DataComp, Webster's LAION-2B audit, Somepalli's diffusion-replication papers, MetaCLIP — all single-stage learned-embedding ANN. The pHash → SSCD cascade is a compute optimization, not an accuracy step, and no peer-reviewed paper publishes its incremental recall@P over SSCD-only.
3. **Horizontal flip is a documented blind spot of every benchmarked detector** — pHash/dHash/PDQ/NeuralHash (McKeown 2023) and also SSCD (DISC21 augmentation taxonomy does not include flip; SSCD's own training augmentations do not learn flip equivariance). Anime has very high mirror-repost rates relative to natural photos. **Pair-querying x and flip(x) is non-negotiable regardless of detector choice.**
4. **Anime/illustration-domain copy detection is not benchmarked anywhere peer-reviewed.** SSCD, DINOv2, Yokoo, DISC21 are all natural-photo-trained on YFCC100M-derivatives. Cross-domain transfer to flat-shaded line art is an open empirical question. All thresholds from §2 of the threshold reference are uncalibrated for this domain.

The implication for planning is conditional, not directive: if the dominant goal is honest evaluation, invest in **split-boundary** dedup; if the goal is whole-corpus pruning, the literature reads as "single-stage embedding + flip-aware ANN, calibrated on a held-out anime sample," with pHash relegated to a speed knob.

---

## 1. Does training-set near-duplicate contamination materially hurt supervised classifier training?

The peer-reviewed evidence is weaker than dedup folklore suggests. Most cited dedup motivations target generative models, language models, or evaluation honesty — not classifier training-set quality.

### 1.1 Test-set / cross-split contamination — the strongest evidence

This is where dedup is unambiguously justified by peer-reviewed literature.

- **Barz & Denzler 2020** (J. Imaging, arXiv:1902.00423). 3.3% of CIFAR-10 and 10% of CIFAR-100 test images have near-duplicates in train. On their duplicate-free *ciFAIR* test set, accuracy drops 9–14% relative. Direct empirical motivation for split-boundary dedup. Methodology is **manual adjudication**, not threshold-based.
- **Recht et al. 2018** (*Do CIFAR-10 Classifiers Generalize to CIFAR-10?*, arXiv:1806.00451) and **Recht et al. 2019** (*Do ImageNet Classifiers Generalize to ImageNet?*, arXiv:1902.10811). Built CIFAR-10.1 with explicit cross-split near-duplicate filtering using ℓ2 on Tiny-Image features + manual top-10 review. Found 4–10% accuracy drop on truly unseen images.
- **Beyer et al. 2020** (*Are we done with ImageNet?*, arXiv:2006.07159). Focuses on ImageNet label noise & multi-label reality. Notes duplicate/synonym class pairs, but the headline finding is *labeling procedure*, not duplicate contamination — less directly relevant.
- **Vasudevan et al. 2022** (NeurIPS 2022, arXiv:2205.04596). Re-examines residual ImageNet errors; reclassifies ~half as not-actually-mistakes. Contribution is multi-label re-evaluation, not duplicate accounting — weaker dedup evidence than the title might suggest.

### 1.2 Direct classifier-side dedup ablations — sparse

- **Liu & Nikzad-Khasmakhi 2025** (arXiv:2504.00638, preprint, not yet at Tier-1 venue). Only direct "dedup vs no dedup" classifier ablation found (CIFAR-10). Finding: standard supervised training **tolerates uniform duplication well**; non-uniform (class-skewed) duplication and adversarial training are where duplicates hurt accuracy.
- **Birodkar, Mobahi, Bengio 2019** (arXiv:1901.11409, *Semantic Redundancies in Image-Classification Datasets: The 10% You Don't Need*). At least 10% of ImageNet/CIFAR-10 can be removed via embedding clustering with **no test-accuracy loss**. Implies semantic duplicates are mostly redundant rather than toxic.
- **Abbas et al. 2023** (NeurIPS 2023 R0-FoMo workshop, arXiv:2303.09540, *SemDeDup*). 50% of LAION can be removed with minimal performance loss; framed as **efficiency**, with OOD generalization sometimes *improving*. CLIP/contrastive setting, but the principle (dedup ≈ compute win) carries.
- **Toneva et al. 2019** (ICLR 2019, arXiv:1812.05159, *An Empirical Study of Example Forgetting*). A large fraction of training examples are "unforgettable" and removable with no test-accuracy hit. Complements Birodkar.

### 1.3 Memorization / long-tail caveat — argues *against* aggressive pruning

- **Feldman 2020** (STOC 2020, arXiv:1906.05271, *Does Learning Require Memorization?*). Proves that for long-tailed natural distributions, memorizing rare/atypical examples is *necessary* for near-optimal generalization.
- **Feldman & Zhang 2020** (NeurIPS 2020, arXiv:2008.03703). Empirical confirmation: memorized training points are atypical and high-marginal-utility for visually similar test points. **Implication for an aggressive embedding-distance dedup pass on a 5.4M anime corpus: rare-tag canonical exemplars (e.g., a single high-quality reference for a low-frequency tag) can look like "near-duplicates" of each other, and over-pruning kills tail-tag mAP.**

### 1.4 Generative / LM dedup motivations — analogy, not direct evidence

- **Lee et al. 2022** (ACL 2022, arXiv:2107.06499). Dedup cuts memorized regurgitation 10×; >4% train/val overlap in standard LM datasets.
- **Kandpal et al. 2022** (ICML 2022, arXiv:2202.06539). 10× duplication ≈ 1000× extraction-attack success (super-linear).
- **Carlini et al. 2023** (ICLR 2023, arXiv:2202.07646). Log-linear memorization-vs-duplication scaling.

These are all LM / generative; the privacy/regurgitation framing **does not transfer to discriminative multi-label classifiers** — there is no "extraction attack" analogue against an ASL-trained tagger.

### 1.5 Documented gap

No peer-reviewed paper at a Tier-1 venue ablates train-only dedup on a >1M multi-label image dataset with mAP/macro-F1 reported. Liu & Nikzad-Khasmakhi 2025 (CIFAR-10, single-label, preprint) is the closest. OpenImages-scale or LVIS-scale "dedup vs. no dedup, hold test fixed" is missing from the literature.

---

## 2. Is the pHash → SSCD cascade actually used in modern curation papers?

No. Recent (2022+) peer-reviewed dataset-curation papers go single-stage learned-embedding.

| Paper | Venue | Image-side dedup method | pHash prefilter? |
|---|---|---|---|
| Pizzi et al. 2022 (SSCD) | CVPR 2022 | RN50/RN101 GeM + entropy reg + FAISS | No — proposed as single-stage |
| Yokoo 2021 (ISC2021 winner) | arXiv:2112.04323 | EfficientNetV2-M + score normalization + pairwise re-rank | No |
| Papakipos et al. 2022 (DISC21 results) | PMLR 176 | All winning entries: end-to-end learned embeddings | No pHash among winning components |
| Somepalli et al. 2023a (Diffusion replication) | CVPR 2023 | SSCD chosen after benchmarking SSL/retrieval features | No |
| Somepalli et al. 2023b | NeurIPS 2023 | SSCD | No |
| Webster 2023 (LAION-2B audit) | arXiv:2303.12733 | SNIP-compressed CLIP-L/H features + IVF-PQ | No |
| Abbas et al. 2023 (SemDeDup) | NeurIPS 2023 R0-FoMo | k-means + cosine on CLIP/OPT embeddings | No |
| Gadre et al. 2023 (DataComp) | NeurIPS 2023 D&B | Yokoo (256-dim, ISC21) embeddings; CLIP for eval-set leakage | No |
| Oquab et al. 2024 (DINOv2) | TMLR 2024 | PCA-hash + Faiss k-NN on learned embeddings, plus SSCD | No |
| Xu et al. 2024 (MetaCLIP) | ICLR 2024 | Metadata-string-based + balancing; embedding source unspecified | No image pHash |

The **only** stage-comparison study located is:

- **Yang et al. 2025**, *Comparative Evaluation of Perceptual Hashing and Deep Embedding Methods for Robust and Efficient Image Deduplication*, Electronics 15(7):1493 (peer-reviewed journal, not a top-tier ML venue but methodologically clean). On UKBench / ABO, all four pHash variants (a/d/p/wHash) show sharply dropping precision as recall increases. CNN embeddings sustain high precision across the recall curve under geometric transforms. Closest published head-to-head; concludes pHash is a speed optimization, not an accuracy contribution.
- **Vasilev et al. 2023** (arXiv:2304.02296, *Efficient Deduplication and Leakage Detection in Large Scale Image Datasets*). Adopts learned embeddings as primary; treats pHash as legacy.

**Documented gap.** No peer-reviewed paper isolates the **incremental recall@P of pHash as a prefilter on top of an SSCD or DINOv2 embedding stage**. The cascade is treated by industrial practitioners as a cheap recall-1 prefilter, but its added recall over single-stage SSCD is undocumented in scholarly venues. Whether the cascade is needed at 5.4M is therefore **a wall-clock cost question**, not an accuracy question, in the literature's framing.

Rough scale check (paper-derived numbers, not project measurements): SSCD inference is dominated by RN50/RN101 forward + L2-normalize. At 5.4M images, this is a one-shot pass — comparable in cost to a single training epoch on a small ResNet, far less than a single ViT epoch. Embedding storage at 512 dim × float16 ≈ 5.5 GB. ANN with FAISS IVF-PQ on 5.4M × 512 is well-trodden territory in DINOv2's pipeline (LVD-142M used Faiss IVF-PQ at this exact scale).

---

## 3. The anime-domain calibration gap

**No peer-reviewed copy-detection benchmark exists for anime / illustration / manga.**

Detectors and thresholds in the threshold reference are all calibrated on natural photos:

- SSCD: trained on YFCC100M, benchmarked on DISC21 (YFCC100M + DFDC).
- Yokoo / DISC21: built on YFCC100M.
- DataComp's 0.604169 cosine threshold: ISC2021 benchmark (natural photos).
- pHash (Zauner 2010, McKeown 2023): natural photos / Flickr-1M.
- DINOv2's SSCD application: web-photo distribution.

Cross-domain transfer evidence that natural-image features under-serve anime:

- **Li et al. 2022** (CVPRW 2022, arXiv:2204.14034, *A Challenging Benchmark of Anime Style Recognition*). ImageNet-pretrained features transfer poorly to anime style discrimination. Not a dedup study, but the closest peer-reviewed evidence that anime style is far from natural-photo statistics in feature space.
- **Saito & Matsui 2015** (SIGGRAPH Asia Tech Briefs, *Illustration2Vec*). The only widely cited peer-reviewed illustration embedding. Pre-deep-SSL era; nothing modern (DINOv2/SSCD-class) has been published on Danbooru / Pixiv-derived data at NeurIPS / CVPR / ICLR / ECCV / ICML / TMLR / JMLR.
- **Matsui et al. 2017** (Manga109, MTAP 2017, arXiv:1510.04389). Manga retrieval task is sketch-query → manga-page, not photometric near-duplicate detection. Different problem.

Perceptual-hash failure modes specifically applicable to flat-shaded illustration:

- **McKeown & Buchanan 2023** (FSI:DI, DFRWS-EU 2023, arXiv:2212.08035). Flags fractal/patterned images as a pHash weakness and solid-color backgrounds / smooth gradients as a Blockhash weakness. Both regimes are common in anime (sky/wall backgrounds, gradient cel shading, monochrome panels).
- **McKeown et al. 2024** (FSI:DI 48, *PHASER*). Reproducible framework confirming the same low-entropy degradation pattern.

**Practical implication for planning.** Any cosine threshold ported from §2 of the threshold reference is a starting point only. The honest path forward is to construct a small calibration set from the actual corpus (held-out anime images with known transformations: re-encode, rescale, mirror, re-color, censorship variants) and re-derive a precision-recall curve in-domain. This is what Pizzi et al., Yokoo, and Gadre et al. all do on their own benchmarks. Skipping calibration imports a domain assumption the literature does not support.

---

## 4. Horizontal flip as a structural blind spot

Mirror is the single transformation broken by every benchmarked detector at the bit/embedding level:

- **pHash 64-bit**: intra-flip BER ≈ 0.497 (Zauner 2010 Tab. 6.6); mean intra is indistinguishable from inter mean ≈ 0.50 (random pairs). McKeown 2023 Tab. 5: mirror BER 0.4904 — same as random.
- **dHash, aHash, PDQ, NeuralHash, Blockhash, Wavehash**: McKeown 2023 reports mirror as a degenerate failure across all hash families.
- **SSCD**: DISC21's transformation taxonomy (Papakipos et al. 2022, App. A) does **not** include horizontal flip in the standard 31-op set. SSCD's training augmentations (Pizzi et al. 2022 §3) likewise do not include flip equivariance. **The model's published recall numbers therefore do not certify mirror invariance.** SSCD-mixup is reportedly more robust (per their model card / threshold reference §4) but this is not benchmarked in the CVPR paper.
- **DINOv2**: uses SSCD for dedup; inherits the same blind spot.
- **CLIP**: anecdotally flip-invariant in semantic space, but at 0.9 precision on ISC has only 0.02 recall (DataComp App. F) — too coarse for copy detection regardless of flip.

**Why anime amplifies this.** Mirror-reposts are common in illustration distribution channels (re-uploads, "fixed pose" variants, board-flipped reposts). Natural-photo benchmarks under-represent this transformation; anime distribution over-represents it. No peer-reviewed paper quantifies the rate for illustration content, but the qualitative direction is unambiguous.

**Implication for planning.** Querying x and flip(x) in the same ANN index is structurally required, not optional, regardless of which detector wins. This doubles query cost but not index size; it is a 2× constant, not a complexity-class change. McKeown 2023 §6 discusses canonicalization (e.g., always picking the lexicographically-smaller hash of x and flip(x)) as a hash-side mitigation, but it has not been ablated for SSCD/DINOv2 embeddings in any peer-reviewed work.

---

## 5. What the literature actually does NOT answer (gaps worth recording)

1. **No Tier-1-venue ablation of train-only dedup on a large multi-label image classifier.** Liu & Nikzad-Khasmakhi 2025 is CIFAR-10 single-label preprint. OpenImages / LVIS-scale evidence is absent.
2. **No published incremental-recall measurement of pHash-prefilter on top of SSCD.** Cascade is justified by compute folklore, not by ablation.
3. **No peer-reviewed copy-detection benchmark on anime / illustration / manga photometric near-duplicates.** Li et al. 2022 is style recognition, not dedup. Manga109 is sketch-to-page retrieval, not dedup.
4. **No peer-reviewed flip-invariance ablation for SSCD or DINOv2 dedup.** All benchmarks omit horizontal flip from the transformation set.
5. **No peer-reviewed paper publishes a "pHash → SSCD" cascade with named thresholds for both stages.**
6. **No peer-reviewed calibration of cosine thresholds for stylized / low-entropy content.** McKeown 2023 flags the failure mode but does not derive an alternative cutoff.
7. **No peer-reviewed study of dedup's effect on long-tail tag mAP** in multi-label classification. Feldman 2020 / Feldman & Zhang 2020 frame the theoretical risk; nobody has quantified it for tag-rich datasets.

These gaps are not project blockers — they are reasons to **internally calibrate rather than blindly apply published thresholds**, and to keep the dedup decision reversible (e.g., store the embedding index and the duplicate-cluster IDs, don't physically delete the candidate-duplicate images).

---

## 6. Synthesis: what "less complex" looks like by the literature

Three options the peer-reviewed evidence supports, ordered by simplicity:

**Option A — Split-boundary only.** Run dedup only between train and val/test (and any held-out probes). Skip whole-corpus pruning. Justified by §1.1: this is the use case where peer-reviewed evidence of accuracy harm is unambiguous. Operationally: SSCD or DINOv2 embeddings on val/test only (small set), then cosine ANN against train, cluster matches, exclude from training. Single stage, single embedding model, no pHash. Mirror handled by flip-paired query.

**Option B — Single-stage embedding dedup, calibrated in-domain.** Compute embeddings (SSCD or DINOv2 — both used by peer-reviewed corpora at this scale) for the entire 5.4M, FAISS IVF-PQ ANN, threshold derived from a held-out calibration set of anime pairs with known transformations. No pHash stage. Aligns with what DINOv2, SemDeDup, DataComp, Webster-LAION, and Somepalli all do. Does not over-prune: keep cluster-IDs, treat duplicate-cluster sampling as a *training-loop* concern (e.g., one-per-cluster per epoch) rather than physical deletion — preserves Feldman's long-tail exemplars.

**Option C — pHash → SSCD cascade.** Justified only as a wall-clock optimization. No peer-reviewed paper publishes its incremental recall@P benefit. If embedding inference at 5.4M is not a budget bottleneck (SSCD on a single A100 finishes in hours), the cascade is gratuitous.

The case for the cascade in the literature is weak; the case for **calibration over threshold-importing** is strong; the case for **flip-paired queries** is structurally required.

---

## 7. Primary sources cited above

Section 1 (does dedup help classifier training):
- [Barz & Denzler 2020 (J. Imaging) — arXiv:1902.00423](https://arxiv.org/abs/1902.00423)
- [Recht et al. 2018 — arXiv:1806.00451](https://arxiv.org/abs/1806.00451)
- [Recht et al. 2019 — arXiv:1902.10811](https://arxiv.org/abs/1902.10811)
- [Beyer et al. 2020 — arXiv:2006.07159](https://arxiv.org/abs/2006.07159)
- [Vasudevan et al. NeurIPS 2022 — arXiv:2205.04596](https://arxiv.org/abs/2205.04596)
- [Liu & Nikzad-Khasmakhi 2025 — arXiv:2504.00638](https://arxiv.org/abs/2504.00638)
- [Birodkar et al. 2019 — arXiv:1901.11409](https://arxiv.org/abs/1901.11409)
- [Toneva et al. ICLR 2019 — arXiv:1812.05159](https://arxiv.org/abs/1812.05159)
- [Feldman STOC 2020 — arXiv:1906.05271](https://arxiv.org/abs/1906.05271)
- [Feldman & Zhang NeurIPS 2020 — arXiv:2008.03703](https://arxiv.org/abs/2008.03703)
- [Lee et al. ACL 2022 — arXiv:2107.06499](https://arxiv.org/abs/2107.06499)
- [Kandpal et al. ICML 2022 — arXiv:2202.06539](https://arxiv.org/abs/2202.06539)
- [Carlini et al. ICLR 2023 — arXiv:2202.07646](https://arxiv.org/abs/2202.07646)

Section 2 (cascade vs single-stage):
- [Pizzi et al. (SSCD) CVPR 2022 — arXiv:2202.10261](https://arxiv.org/abs/2202.10261)
- [Yokoo 2021 — arXiv:2112.04323](https://arxiv.org/abs/2112.04323)
- [Papakipos et al. 2022 — arXiv:2202.04007](https://arxiv.org/abs/2202.04007)
- [Somepalli et al. CVPR 2023 — arXiv:2212.03860](https://arxiv.org/abs/2212.03860)
- [Somepalli et al. NeurIPS 2023 — arXiv:2305.20086](https://arxiv.org/abs/2305.20086)
- [Webster 2023 — arXiv:2303.12733](https://arxiv.org/abs/2303.12733)
- [Abbas et al. 2023 (SemDeDup) — arXiv:2303.09540](https://arxiv.org/abs/2303.09540)
- [Gadre et al. NeurIPS 2023 (DataComp) — arXiv:2304.14108](https://arxiv.org/abs/2304.14108)
- [Oquab et al. TMLR 2024 (DINOv2) — arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
- [Xu et al. ICLR 2024 (MetaCLIP) — arXiv:2309.16671](https://arxiv.org/abs/2309.16671)
- [Yang et al. Electronics 2025, 15(7):1493](https://www.mdpi.com/2079-9292/15/7/1493)
- [Vasilev et al. 2023 — arXiv:2304.02296](https://arxiv.org/abs/2304.02296)

Section 3 (anime / illustration domain):
- [Li et al. CVPRW 2022 — arXiv:2204.14034](https://arxiv.org/abs/2204.14034)
- [Saito & Matsui SIGGRAPH Asia 2015 (Illustration2Vec)](https://dl.acm.org/doi/10.1145/2820903.2820907)
- [Matsui et al. (Manga109) MTAP 2017 — arXiv:1510.04389](https://arxiv.org/abs/1510.04389)
- [McKeown & Buchanan FSI:DI 2023 — arXiv:2212.08035](https://arxiv.org/abs/2212.08035)
- [McKeown et al. FSI:DI 2024 (PHASER)](https://www.sciencedirect.com/science/article/pii/S2666281723001993)

Section 4 (horizontal flip):
- [Zauner 2010 (TU Hagenberg thesis)](https://www.phash.org/docs/pubs/thesis_zauner.pdf)
- [Pizzi et al. CVPR 2022 — arXiv:2202.10261](https://arxiv.org/abs/2202.10261)
- [Papakipos et al. 2022 — arXiv:2202.04007](https://arxiv.org/abs/2202.04007)
- [McKeown & Buchanan 2023 — arXiv:2212.08035](https://arxiv.org/abs/2212.08035)
