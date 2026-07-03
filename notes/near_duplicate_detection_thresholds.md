# Near-Duplicate Detection: PDQ + Flip-Aware Matching

Notes for the dedup stage on the anime / illustration corpus (~5.4M images). Calibration anchored to peer-reviewed sources; numbers are starting points, not domain-validated.

---

## 1. Upstream Preprocessing (Affects This Stage)

All corpus images are resized so the **longest side = 520 px**, original aspect ratio preserved. No padding pixels are added at this stage — images keep their variable shorter side. Sizing is *calculated as if* a uniform 520×520 canvas were the target (the open ~30–40% dead-space tolerance question), but that is a downstream training-input concern; the bytes that go into the hash are unpadded.

Implications for dedup:

- Every image goes through the same resize pipeline, so within-corpus comparisons are between hashes of similarly-conditioned images. The "scale" / "thumbnail" intra-BER numbers from the literature (which measure hash drift under arbitrary resize) no longer describe the within-corpus regime — they only describe what happens if the same source image was resized differently *before* entering the corpus.
- Hash is computed on rectangular images of varying shorter side. PDQ internally downsamples to its own fixed grid before applying its DCT/Haar-like transform, so aspect-ratio differences mostly wash out, but extreme aspect-ratio mismatches between near-duplicates of the same source can still survive into the hash.

---

## 2. Approach

Per-image: store two perceptual hashes — `pdq(image)` and `pdq(hflip(image))`. A query is a near-duplicate if its hash is within the Hamming threshold of *either* stored column.

Rationale: perceptual hashes in this family (PDQ, pHash) are provably blind to horizontal flip — flipped hashes land at intra-BER ≈ 0.49, indistinguishable from random (McKeown 2023). Storing the flipped hash converts a continuous failure into a deterministic exact-flip-or-not decision. Index size doubles, query cost is one extra lookup. No scholarly reference for this construction; the literature documents the failure but not the remediation.

Horizontal flip only. Vertical flip and rotations >5° are out of scope and accepted as residual noise.

### Why PDQ, not embeddings

- No anime / illustration calibration exists in the peer-reviewed literature for SSCD, CLIP, PDQ, or pHash. SSCD is trained on YFCC100M; DataComp's Yokoo threshold is ISC-calibrated. Both are natural-photo distributions.
- AnyPattern (Wang 2024, arXiv:2404.13788) shows ICD SOTA models lose ~60% accuracy on novel patterns; SSCD μAP drops to 14.22% on novel patterns, zero-shot CLIP <10% μAP. Anime is a domain shift relative to YFCC100M.
- SemDeDup (cosine ≥ 0.93 on OpenCLIP) catches alt-color variants and same-character re-shoots as duplicates. For a tagger these are distinct training samples. FairDeDup (Slyman 2024, arXiv:2404.16123) further documents minority-representation harm from semantic dedup.

PDQ's blindness to recolor and re-draw is a feature in this domain. Its failure modes (flip, heavy crop, rotation >5°) are discrete and known.

---

## 3. Threshold Target

For 256-bit PDQ (pHash equivalents in parentheses for cross-reference):

| Threshold | PDQ Hamming | (pHash Hamming) | BER | Catches | False positives |
|---|---|---|---|---|---|
| ≥ 95.3% | ≤ 12 | (≤ 3) | ≤ 0.047 | JPEG, scale, thumbnail | Near-zero |
| **≥ 93.75%** | **≤ 16** | **(≤ 4)** | **≤ 0.063** | **+ most watermarks** | **Very low** |
| ≥ 90.6% | ≤ 24 | (≤ 6) | ≤ 0.094 | + light crops with borders | Low–moderate |

Default operating point: **similarity ≥ 93.75% (PDQ Hamming ≤ 16/256)**. Below McKeown's empirical "mostly safe" line of BER 0.094, with headroom for the anime-entropy effect (§5).

Refine after the boundary audit. Tighten to ≥ 95.3% if 93.75% audit is noisy; loosen to ≥ 90.6% if 93.75% boundary is clean and more recall is wanted. PDQ's stronger inter-image separation (§6) leaves room to push tighter than pHash would tolerate.

---

## 4. Manual Boundary Adjudication

Procedure from Barz & Denzler 2020 (arXiv:1902.00423), the only scholarly precedent for dedup on a corpus without a pre-calibrated threshold:

- Sort candidate pairs by increasing Hamming distance.
- Adjudicate ~200 pairs in the band around the threshold (PDQ Hamming 12–24 for the default operating point) into Exact / Near-Duplicate / Very Similar / Different.
- Stopping rule: stop after 20 consecutive "Different" pairs.
- Tighten or loosen based on where FPs first appear.

---

## 5. Anime-Specific Calibration Risk

McKeown 2023's inter-image stdev rankings (PDQ 0.0321, pHash 0.0649) are from Flickr-1M natural photos. Low-entropy content — flat backgrounds, near-monochrome panels, large solid-color regions — shrinks the inter-image stdev below the photo baseline, raising the false-positive rate at any fixed threshold. The anime-specific magnitude is undocumented; PDQ's 2× separation advantage over pHash gives more headroom but does not eliminate the risk.

The 93.75% target sits below McKeown's 90.6% boundary specifically to absorb this drift. Any quoted threshold is a starting point inferred from natural-photo calibration — the boundary audit (§4) is non-optional.

---

## 6. Hash Choice: PDQ

Default is 256-bit PDQ. McKeown 2023 measured PDQ inter-image stdev at 0.0321 — half of pHash's 0.0649; strictly the best discriminator of any hash benchmarked. Storage is 4× per hash (32 B vs 8 B), trivial at this corpus size (~350 MB raw). The flip-aware construction (§2) and threshold percentages in §3 apply identically; integer Hamming cutoffs scale by 4. Tooling (Meta's `pdqhash` Python bindings) is less ubiquitous than `imagehash` but stable. Fallback to 64-bit pHash if `pdqhash` integration proves problematic — no algorithmic blocker, just a 2× loss of inter-image separation.

---

## 7. Out of Scope

- Heavy crops (>~30%), rotations >5°, perspective warps, vertical flips: PDQ cannot see these. Separate stage (embedding-based or manual) if they matter, otherwise accepted as residual noise.
- Recolors, alt outfits, redrawn scenes: *intentionally* not addressed — distinct training samples for a tagger.

---

## 8. Primary Sources

- Zauner 2010 — *Implementation and Benchmarking of Perceptual Image Hash Functions* (TU Hagenberg MSc thesis): https://www.phash.org/docs/pubs/thesis_zauner.pdf
- McKeown & Buchanan 2023 — *Hamming Distributions of Popular Perceptual Hashing Techniques* (arXiv:2212.08035): https://arxiv.org/abs/2212.08035
- Barz & Denzler 2020 — *Do We Train on Test Data? Purging CIFAR of Near-Duplicates* (arXiv:1902.00423): https://arxiv.org/abs/1902.00423
- Wang et al. 2024 — *AnyPattern* (arXiv:2404.13788): https://arxiv.org/abs/2404.13788
- Slyman et al. 2024 — *FairDeDup* (arXiv:2404.16123): https://arxiv.org/abs/2404.16123
- Pizzi et al. 2022 — *SSCD* (arXiv:2202.10261): https://arxiv.org/abs/2202.10261

---

## 9. Pipeline & Storage Plan

Multi-stage pipeline backed by SQLite. Stage 2 caches pair discovery at the loosest threshold; stages 3–4 are cheap re-runs over the cached pair list, so threshold choice is interactive.

### 9.1 Definitions

- **Dead space** — area lost to padding when an image is fitted into the square training canvas downstream. Function of stored width/height only: `dead_space = 1 - min(w,h)/max(w,h)`. No image inspection at dedup time.
- **Quality** — no-reference proxy for "this copy is worth keeping." Composite of blur and compression signals (§9.3).

### 9.2 Stages

1. **Hash**: scan corpus → compute `pdq`, `pdq_hflip`, read `w`, `h` from JPEG header → upsert into `images` table. Idempotent (`path TEXT UNIQUE` is the natural key; corpus root `L:\Dab\Dab` is stable for the lifetime of the dataset, so paths don't move). Resumable.
2. **Pair Discovery**: load all hashes into RAM, build BK-tree, enumerate all pairs at the *loosest* threshold the review will ever consider (PDQ Hamming ≤ 24, §3). Write to `pairs` table with the Hamming distance stored. Run `pdq`-vs-`pdq` and `pdq`-vs-`pdq_hflip`; the other two flip combinations are redundant by symmetry. Exclude self-pairs from the cross-orientation query — bilaterally symmetric images (`pdq == pdq_hflip`) would otherwise match themselves at distance 0. One-time cost.
3. **Cluster & Resolve**: filter `pairs` to chosen threshold → complete-linkage clustering (§9.4) → apply dead-space tiebreaker (§9.3) → mark `keep` on `images` or write `survivors` table.
4. **Bucket Counts** (review UX): run stage 3 in dry-run mode at thresholds {95.3%, 93.75%, 90.6%, …} → report resulting dataset size per bucket. Pure SQL/Python over cached `pairs` table; seconds, not hours.

### 9.3 Keeper Selection and Disposition

**Disposition.** The keeper retains its image and JSON sidecar unchanged. Each discarded image is removed along with its sidecar — sidecar removal is triggered by image removal, never independently. Tags are *not* merged across the cluster: discarded sidecars may describe a different crop, watermark variant, or carry tagger-specific noise, and merging adds risk for marginal recall gain.

**Tiebreaker.** Resize to longest=520 happens *upstream* of dedup and pre-resize dimensions are not preserved. Per-image quality metrics (Laplacian variance, JPEG quality factor, BRISQUE) are out of scope — not worth the per-image compute or the schema overhead at this stage.

The only signal available is the stored width/height of the resized image, which yields the dead-space metric directly.

Tiebreaker order within a cluster:

1. Least dead space — `1 - min(w,h)/max(w,h)`, smaller is better.
2. If tied, random.

Trade-off accepted: when two near-duplicates have identical aspect ratios, the keeper is chosen blind to blur and compression quality.

### 9.4 Cluster Collapse Rule

**Complete linkage.** Drop members of a candidate component only if every pairwise Hamming distance within the component is ≤ threshold. Conservative against the over-collapse risk in §5 (chains of A~B~C where A≁C silently merging at looser thresholds on low-entropy art). Required because per-cluster human review is not feasible at corpus scale — §4's boundary audit operates on a fixed-size sample of pairs near the threshold and stays tractable, but does not catch chain-collapse inside accepted clusters.

Cost is O(k²) per candidate component, bounded because components stay small at the operating thresholds.

### 9.5 Storage

SQLite. Sketch (subject to revision):

- `images` — `id INTEGER PRIMARY KEY` (rowid surrogate), `path TEXT UNIQUE NOT NULL`, `pdq BLOB(32)`, `pdq_hflip BLOB(32)`, `w INT`, `h INT`, `keep BOOL` (populated by stage 3). All-JPEG corpus, no `format` column needed. Surrogate INT PK keeps `pairs` rows compact; `path` carries the natural key.
- `pairs` — `image_a_id INT`, `image_b_id INT` (both FK → `images.id`), `hamming_distance INT`, `match_type` ENUM(`same_orientation`, `flipped`). Populated once at stage 2 at the loosest threshold.
- `clusters` (optional) — cluster_id, image_id, threshold. Re-derivable from `pairs`; materialize if review UX needs it.

10.8 M hashes (5.4 M × 2) at 32 B = ~350 MB raw, ~3 GB with BK-tree node overhead. Fits comfortably in RAM. End-state SQLite file is in the low single-digit GB after stages 1–2; well within SQLite's effective limits.

### 9.6 Open Questions

- **Threshold calibration.** §5's anime-entropy drift is undocumented. §4's boundary audit covers the threshold neighborhood at fixed cost and is feasible at scale, but does not quantify the global FP rate or catch chain-collapse inside accepted clusters. Mitigation is to start tight (95.3% or 93.75%), use the bucket counts (stage 4), and add a small fixed-size random spot-check to gain confidence before committing.
- **Stage 2 throughput unmeasured.** BK-tree pair enumeration at Hamming ≤ 6 over ~10.8 M hashes has not been benchmarked. Run a 100K-image sample first; if runtime is prohibitive, switch to LSH or sort-and-bucket before committing to the full corpus.
