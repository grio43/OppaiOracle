# V2 training readiness review — 2026-09-06

Scope: the current working tree, normal `Start_AI_Training.ps1` startup, and
`configs/unified_config.yaml`. This review covers Phase 1 training, preparation,
validation, pause/resume, and the installed hardware. Alternate CLI modes,
export/inference, and unapproved augmentation changes are outside scope.

**Initial review: two reproduced resume failures; no unconditional P0/critical
failure established.** R1 is deferred at the user's request because cache rebuilds
are rare. R2 and AP chunking are implemented below. The unfinished
dataset is an expected preparation gate, not a bug: the production configuration
and recipe assertions pass, and readiness correctly refuses the absent manifest.

The repository already had extensive modified/untracked implementation files.
The initial review did not change production code, YAML, vocabulary, dataset, or
checkpoint. The authorized follow-up changes code/YAML for R2 and AP chunking;
production data, vocabulary, and checkpoints remain untouched. Tests use temporary synthetic datasets/checkpoints; performance probes
use disposable model weights. Only a three-sidecar schema sample was inspected
from the real corpus. It contains only the `Danbooru` top-level subset currently.
The user confirmed that all data is deduplicated before import. Upstream
deduplication is the data contract; this review does not request a new hash policy
or an additional deduplication gate.

## Findings

### R1 — P1: rebuilding the Arrow cache changes what a saved resume offset means

**Status: deferred by user; cache ordering is unchanged.**

Locations: `utils/metadata_cache.py:357–360`, `dataset_loader.py:2002–2007`,
`train_direct.py:2044–2075`.

The cache builder appends parse chunks in `as_completed` order. The dataset then
selects rows in Arrow order, rather than the frozen split-file order. The sampler
permutes integer row indices, and mid-epoch resume restores a numeric sample
offset. Neither the manifest check nor the sampler's size check binds those
indices to an ordered list of image IDs.

**Trigger:** pause mid-epoch and rebuild an otherwise identical Arrow cache before
normal startup. Rebuilds can follow a missing/invalid cache or its metadata,
staleness detection, or the YAML rebuild setting. Dataset labels and membership
need not change. Warm-cache resume does not require such a rebuild and is not
claimed to fail.

**Reproduction:** prepared 2,050 synthetic images, reserved 50, and used the real
loader/sampler. Built the cache once with chunks completing in submission order,
then again in reverse order. Only the completion iterator was controlled; actual
parsing, cache writing, dataset selection, and sampler logic ran. Joining the
first 1,000 samples before the rebuild to the resumed suffix afterward repeated
**509 training images and omitted 509**. Both loaders had 2,000 training rows and
`verify_preparation` passed. These are demonstration counts, not a production
failure-rate estimate.

**Fix direction:** give each prepared dataset a deterministic ordered identity
independent of physical Arrow order, for example by mapping frozen split paths
to Arrow rows in split order. Restore sample offsets against that same sequence
on resume; this does not require a content-hash or deduplication policy.
Making parsing output deterministic alone should also account for the sequential
fallback, which uses a different source ordering today.

**Closure:** preserve the exact sample sequence across a forced cache rebuild and
resume; also exercise cache-to-fallback transitions. Counts and disjoint TRAIN/EVAL
sets are insufficient to verify this property.

### R2 — P1: automatic bad-image exclusions make the next normal restart fail

**Status: implemented with the user's report-only policy.** Prepared rows retain
their positions; failed images are skipped and logged once per path by a buffered
background writer in `<log_dir>/bad_images.txt`. No preparation decode scan or
blacklist is used. Legacy exclusions are ignored for prepared rows. Validation
reports its readable/skipped counts and continues; an all-failed validation pass
saves progress without selecting a best checkpoint. The original reproduction
below describes the pre-fix behavior.

Locations: `utils/v2_preparation.py:89–91`, `dataset_loader.py:1985–1993` and
`2492–2506`, `train_direct.py:602–606`.

Preparation checks image existence but does not decode images. During training,
the loader handles a bad image as a skipped sample and persists its exclusion
after repeated failures. On the next startup, dataset construction removes the
excluded row. The exact prepared-count check now aborts before checkpoint resume.
Re-preparation is refused once this experiment has checkpoints.

**Reproduction:** a corrupt image was present **before** preparation of a
20-image synthetic corpus. Preparation succeeded with 18 training and two
validation members. Fetching that training sample twice persisted the exclusion.
Recreating the normal loaders produced **17 training rows versus 18 in the
manifest**, while the manifest itself still verified. This reaches the trainer's
explicit count-mismatch exception. No edits after preparation were needed.
The restart failure requires a persisted exclusion; a single failed fetch does
not automatically satisfy that trigger. Worker-local retry tracking also affects
how soon a particular bad training image is persisted.

**Impact:** an error that training tolerates becomes an unexpected inability to
resume the same experiment. With a large corpus and long training, this is a
material recovery defect. A corrupt validation image has a related earlier
problem: preparation admits it, then validation deliberately aborts after the
first training epoch because EVAL lost a sample.

**Adopted fix:** preserve membership and report failures without stopping for bad
images. File writes run on a low-priority thread every 30 seconds and at clean
shutdown; normal sample reads perform no reporting I/O. The report is never used
to suppress a repaired image. Partial validation scores describe the readable
subset, which is visible in evaluated/skipped counts.

**Closure evidence:** `test_training_readiness.py` exercises cache/fallback worker
resume positions, retries after repair, report deduplication across restarts,
background-only file I/O, and real CUDA WSD pause/resume with bad training images
and partially/entirely unreadable validation sets.

## Hardware and verification basis

| Component | Observed configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32,607 MiB reported VRAM, compute capability 12.0 |
| CPU | Ryzen 9 7950X3D, 16 cores / 32 logical processors |
| RAM | 255.11 GiB physical; approximately 174 GiB available at initial inspection |
| Dataset drive E: | Predator GM7000 4 TB NVMe SSD |
| Repository drive L: | Approximately 16.38 TB Storage Spaces virtual disk |
| Python | 3.11.9 in `L:\Dab\payton_env` |
| CUDA software | Torch 2.11.0+cu130, torchvision 0.26.0+cu130, driver 610.62 |
| Other dependencies | triton-windows 3.6.0.post26, bitsandbytes 0.49.0, TorchMetrics 1.8.2, PyArrow 22.0.0 |

About 11.5 GiB of GPU memory was occupied by other work before model probes.
The initial compiled-model probes ran under that condition. The user then
authorized closing ComfyUI; stopping its identified Python backend freed about
7.1 GiB, leaving approximately 27 GiB reported free before subsequent probes.
Other desktop applications remained running. These are workstation measurements,
not an idle-device throughput certification.

## Optimizations for this hardware

### First priority: bound the GPU mAP update independently of inference batch size

`train_direct.py:3131` passes all 48 validation rows to the binned AP update.
The installed TorchMetrics implementation materializes large intermediate tensors
over batch, labels, and all 200 thresholds. This scales with the final vocabulary,
which is not known yet. It is a memory risk to profile, not an unconditional
out-of-memory finding at today's vocabulary size.

Measured on the RTX 5090 with 19,292 scored labels, 48 rows, 200 thresholds, and
unknown rating targets included:

| Rows per metric update | Additional peak allocated GPU memory | Warm time for the same 48 rows | Confusion counts |
|---|---:|---:|---|
| 48, current path | 8.46 GiB | 39.1–39.9 ms | Reference |
| 8 | 1.18 GiB | 44.3–44.6 ms | Exactly identical |
| 4 | 0.71 GiB | 49.2–50.4 ms | Exactly identical |

**Implemented:** validation model inference keeps its chosen batch size;
`validation.ap_update_chunk_size: 8` feeds predictions/targets into AP in eight-row
slices in both training and the matching epoch-validation tool. This cut the
measured transient allocation by **86%**, at about 5 ms extra per 48 images.
Identical integer confusion counts preserve this binned AP calculation; changing
the threshold count would change the metric instead. The measurement excludes
the model and other metrics, so it is not total validation VRAM. Recheck memory
with the final vocabulary and retain a smaller slice option if needed.

### Keep batch size and checkpointing provisional until the final vocabulary exists

After closing ComfyUI, the full 18-layer model completed three optimizer updates
at **batch 64 × accumulation 16**, with fp32 image/target buffers, bf16 autocast,
the configured optimizer, and 19,294 outputs. Losses and gradient norms remained
finite. Peak allocated VRAM was **13.70 GiB**; peak reserved VRAM was **14.31 GiB**.
Warm optimizer updates took **3.20 seconds** each. The first update took 172
seconds including compilation with earlier compiler caches available.

This supports retaining 64 × 16 as the provisional configuration. It does not
certify the final output shape, a complete validation pass, or real-data
throughput. The probe reuses a synthetic batch and excludes loading,
augmentation, transfers, telemetry, checkpoint writes, and scheduling overhead.

The configured `checkpoint_every_n_layers: 4` checkpoints layers 0, 4, 8, 12, and
16: **five of 18 blocks**, not groups of four (`model_architecture.py:723–728`).
If final-shape profiling needs more training memory, testing interval 2 or 1 is
a targeted option. More checkpointing trades saved activations for recomputation;
choose using measured update time and peak memory. [PyTorch's checkpointing
explanation](https://pytorch.org/blog/activation-checkpointing-techniques/)
describes this tradeoff.

Retain bf16 autocast with fp32 parameters and the configured fp32 head optimizer
state. The checkpoint probe verifies that this mixed optimizer-state policy
survives resume. An 8-bit optimizer saves parameter-dependent state; it does not
solve activation or metric allocations. [bitsandbytes optimizer
documentation](https://huggingface.co/docs/bitsandbytes/optimizers)
explains that distinction.

### Establish an end-to-end baseline before changing CPU or loader settings

The dataset is already on an NVMe SSD, and the configuration uses six persistent
workers, pinned memory, and prefetch factor three. Nothing measured here supports
replacing hardware or increasing all worker/thread counts automatically. A
30,000-row, 256-label exact-frequency-metric CPU probe took about 1.16 seconds at
one or 16 threads, and 1.01 seconds at four; that narrow test is insufficient to
choose a global thread setting.

Once preparation is possible, measure data wait, optimizer-update time, and the
whole validation pass together. Use the final vocabulary, real augmentations,
and the configured accumulation. Synthetic model timings omit those costs.

## Validation completed

- Follow-up fixes: six readiness regressions plus the 29 existing audit/V2/rating
  tests pass. Flip-worker checks and canonical YAML validation also pass. The
  readiness tests exercise the actual CPU/CUDA AP helper and real CUDA WSD
  pause/resume, including an all-failed validation pass. Logs:
  [readiness](../../logs/fix_2026_09_06_readiness.log),
  [regressions](../../logs/fix_2026_09_06_regressions.log),
  [flip](../../logs/fix_2026_09_06_flip.log),
  [config](../../logs/fix_2026_09_06_config.log).
- Configuration loading and the Phase 1 recipe checks pass. The missing prepared
  manifest correctly blocks production startup while the additional subset is
  pending.
- All **29** tests in `test_training_audit`, `test_v2_pipeline`, and
  `test_rating_system` pass, including their CUDA cases. All **39** stop-condition
  tests pass. The flip and rotation regression scripts also pass.
- A separate CUDA AdamW8bit/WSD checkpoint round trip produced **zero parameter
  difference** after the next update compared with uninterrupted execution, and
  identical scheduler state. Head optimizer state remained fp32 and the tested
  backbone state remained uint8.
- A two-layer model with the configured width, head geometry, 320px resolution,
  Flex Attention, and 19,294 outputs compiled and completed three finite training
  updates. Evaluation also succeeded at batch sizes two and one. This verifies
  the installed Windows compiler path on a bounded model, not full-run capacity.

The initial full-size probe also completed three finite updates with 151,218,174
parameters, 19,294 outputs, and batch 64. Its first update took 823 seconds
including compilation/autotuning; subsequent updates took 0.219 and 0.213 seconds.
Peak allocated GPU memory was 13.05 GiB, with peak reserved memory 13.84 GiB.
That initial probe used bf16 input buffers and one microbatch per optimizer
update, so it is a preliminary kernel/resource check, not the configured
16-microbatch training measurement. Both model probes use four CPU threads and
synthetic inputs; the full-size one ran while ComfyUI was still open.

These checks cover the recent loss, rating, validation precision, checkpoint,
and WSD work. They do not establish convergence or final-corpus throughput, and
they do not cover R1/R2 without the additional reproductions above.

## Reproduction artifacts

Local, ignored probe scripts and logs are retained for this working-tree review:

- [Data reproductions](../../.code-review/review_20260906_data.py) and
  [results](../../logs/review_2026_09_06_data.log): R1–R2.
- [Model probes](../../.code-review/review_20260906_probes.py),
  [small compiled model log](../../logs/review_2026_09_06_compile.log), and
  [initial full-shape log](../../logs/review_2026_09_06_fullshape.log), plus
  [configured-accumulation results](../../logs/review_2026_09_06_accumulation.log).
- [mAP probe](../../.code-review/review_20260906_metrics.py) and
  [mAP results](../../logs/review_2026_09_06_metrics.log);
  [optimizer resume probe](../../.code-review/review_20260906_resume.py), with
  [resume results](../../logs/review_2026_09_06_resume.log).
- [CPU metric probe](../../.code-review/review_20260906_cpu_metrics.py) and
  [results](../../logs/review_2026_09_06_cpu_metrics.log).
- [Existing regression results](../../logs/review_2026_09_06_regressions.log),
  [stop results](../../logs/review_2026_09_06_stop.log),
  [flip results](../../logs/review_2026_09_06_flip.log), and
  [rotation results](../../logs/review_2026_09_06_rotation.log).

These logs and probes are local artifacts, not dependencies of production
training and not files to add to a release.
