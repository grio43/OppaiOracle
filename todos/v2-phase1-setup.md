# V2 Phase 1 setup

The active recipe is [v2-plan.md](v2-plan.md), implemented by
[unified_config.yaml](../configs/unified_config.yaml). This setup covers Phase 1
only; Phase 2/3 launch automation and serving/export verification remain separate.

The dataset container is `E:\Dataset`. It currently contains
`Danbooru\Danbooru\shard_*`; add the second dataset as another top-level folder.
Keep all five tag categories (`gen:`, `char:`, `copyright:`, `artist:`, `meta:`)
except the names in [Tags_ignore.txt](../Tags_ignore.txt). Environments, updater
state files, and unpaired JSON files are excluded from sidecar discovery.

## After the second subset is ready

From the repository root:

```powershell
L:\Dab\payton_env\Scripts\python.exe -B tools/prepare_v2_dataset.py --config configs/unified_config.yaml
```

Preparation scans the complete combined dataset, validates tags/ratings and image
filenames, counts each tag once per image, builds `vocabulary/v2/vocabulary.json`,
and freezes exactly 30,000 validation images with seed 42. Everything else trains.
It rejects MD5 duplicates inside validation or crossing the train/validation
boundary. It never edits dataset images or sidecars. A failed preparation leaves
no valid readiness stamp.

`logs/v2_phase1/dataset_manifest.json` records membership and label hashes, the
vocabulary and rejected-list hashes, counts, and source-sidecar SHA256 snapshots.
Startup checks the prepared artifacts and top-level subset list. These checks
are not a full rescan of every source sidecar; freeze the dataset after preparation
and rerun preparation after editing labels or adding images. A new top-level
subset is detected automatically and requires preparation again. Rebuilding is
refused once that experiment has checkpoints, to protect indices and the holdout.

Then launch:

```powershell
.\Start_AI_Training.ps1
```

No production vocabulary or training run was created during setup, as requested.
V1 checkpoint/prediction-cache tensors were removed (10 files, 42.83 GiB), along
with the old train/validation path-list caches. Released exports and source
images remain available. New checkpoints use `experiments/v2_phase1_vit`.

## Bad-image reporting

Bad images encountered in prepared V2 runs are skipped for that batch and recorded
once per full path in `logs/v2_phase1/bad_images.txt`. Each tab-separated row contains
the path, image ID, and error. This is a report, not a blacklist: repaired images
can load on a later pass, and prepared sample positions remain unchanged on resume.
The old `cache_exclusions.txt` is ignored by prepared V2 loaders.

The trainer submits only failures already returned by its DataLoader. A background
thread at the lowest ordinary Windows thread priority writes batches every 30
seconds and flushes at clean shutdown. Deduplication uses paths, not content hashes.
There is no extra image scan or worker disk write. A forced process kill can lose
the most recent buffered report entries; reporting errors do not stop training.

Validation continues with readable images and logs evaluated/skipped counts. Scores
describe those readable images, so changing failures can affect comparability.
If no validation image loads, progress is saved without a validation score or best
checkpoint selection. `validation.ap_update_chunk_size: 8` slices the metric update
while preserving the model inference batch size and the same 200 AP thresholds.

## Training behavior

Phase 1 uses 320px, width 896, 18 layers, 14 heads, MLP 2320, QK normalization,
LayerScale 0.1, drop-path 0.20, and zero plain dropout. ASL remains detached
7/0/.05 with mean reduction and ignored PAD/UNK. AdamW8bit uses fp32 state for
the tag head and position embeddings; master parameters stay fp32 under bf16 AMP.

The provisional microbatch is 64 with 16-step accumulation (effective batch 1024).
Profile it on the final vocabulary before a long run. The base LR is 2.7e-4,
sqrt-scaled to 5.4e-4. WSD warms up for 10,000 updates, holds constant, then uses
a 1-sqrt cooldown of 15% of elapsed pre-cooldown updates. The **60-epoch ceiling
includes cooldown**. Budget exhaustion triggers cooldown automatically; reaching
the ceiling is not evidence of convergence.

`training.wsd_plateau_enabled` starts false. Review roughly ten stable-phase
validation events and freeze the jitter/window decision before enabling it.
Once armed, a plateau starts cooldown instead of immediately stopping. Other
statistical stop signals remain advice under WSD. NaN/Inf loss or gradients and
sampled attention-logit divergence abort; a soft stop pauses at an optimizer
boundary and preserves scheduler state. `best_model.pt` is selected from finite
validation measurements during cooldown once those measurements exist.

The attention diagnostic computes exact absolute maxima on a two-image sample
at the first batch of each epoch and logging updates; it is not an exhaustive
maximum across every training image. Per-decile mAP, binned oracle F1 (explicitly
reporting-only), and exact-versus-binned mAP diagnostics do not control the loss.

## Rating behavior before and after migration

| Rule | Before | V2 |
|---|---|---|
| Sidecar field | Full names, `g/q/e`, `safe`, or integers 0–3 accepted | Also accepts `s` as **sensitive**; `g/s/q/e` all work |
| Output tag names | `rating:general`, `rating:sensitive`, `rating:questionable`, `rating:explicit` | Same canonical names |
| Model/loss | Four entries in the shared sigmoid tag head and ASL loss; no separate rating head or softmax constraint | Same |
| Vocabulary | Four rating entries appended even below the frequency cutoff; the separate field was not counted by the main counter | Mandatory entries retained; rating-field positives now counted once per image |
| Unknown rating | Samples rejected, including valid Danbooru `s` samples that the old mapper missed | Image retained; the four rating labels are masked from loss, metrics and calibration; `s` samples accepted |
| Monitoring | Ratings excluded from ASL score/separation telemetry, included in headline tag metrics | Headline rating metrics use only rated images; sibling content groups also resolve `gen:` names |

The old spelling `safe` remains an alias for **general**. The new code `s` means
**sensitive**, never safe. The sidecar rating field is authoritative when building
the V2 vocabulary and training targets. Missing or unrecognized ratings never become four negative labels. Preparation reports rated/unrated counts; rating bias priors use the rated-image count. Ordinary tag thresholding still applies at inference; this
migration does not add a rule forcing exactly one rating prediction.

Rating regression coverage includes missing fields, nulls, blanks, unrecognized codes,
malformed field types, every accepted alias and conflicting inline rating tags. Both
uncached and Arrow worker paths retain every otherwise valid image. An entirely
unrated corpus still trains content tags; its rating loss and rating-logit gradients
are zero. As with any shared model, rating predictions can change through backbone
updates or optimizer momentum/weight decay; the guarantee is no rating supervision
from an unrated image. Validation and calibration also exclude its unknown ratings.
A calibration bucket with no positive evidence keeps the default threshold.

Run `python -B test_rating_system.py` for the focused audit and
`python -B test_v2_pipeline.py` for mixed/all-unrated training and resume coverage.
