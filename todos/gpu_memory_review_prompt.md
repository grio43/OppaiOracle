# Investigation Prompt: GPU Memory Layout Review (ViT Training)

## Framing

This is a **review**, not a bug hunt. Training is making forward progress at the
expected throughput. Nothing is broken. The goal is to build an accurate
mental model of what currently lives in **dedicated GPU memory (VRAM)** and
what lives in **Windows "Shared GPU memory"**, decide whether the current
layout is healthy, and surface any optimizations worth considering for the
next training run.

If everything is working as designed, the verdict should say so plainly.

## Objective Facts

**Project root:** `L:\Dab\OppaiOracle`
**Active training log:** `L:\Dab\OppaiOracle\logs\training.log`
**Config:** `L:\Dab\OppaiOracle\configs\unified_config.yaml`
**Main entry point:** `L:\Dab\OppaiOracle\train_direct.py`

### Current observed memory state (Windows Task Manager / nvidia-smi)
- **Dedicated GPU memory (VRAM):** ~31 / 31.5 GB used
- **Shared GPU memory (system RAM via WDDM):** ~14 GB used
- **GPU utilization:** 100%
- **Per-step throughput:** unchanged from earlier epochs — i.e., kernels are
  NOT being paged across PCIe per step. Whatever is in shared GPU memory is
  cold (not touched on the hot path).
- The user's normal training VRAM in prior runs was reported as ~29 GB.

### Relevant config values (carry-forward from prior investigation)
- `data.batch_size: 102` (Phase 1, 320px, 18 layers); grad-accum = 9 → effective batch 918
- `data.num_workers: 6`, `pin_memory: true`, `prefetch_factor: 3`, `persistent_workers: true`
- Validation loader: `batch_size: 58`, `num_workers: 6`, `pin_memory: true`, `prefetch_factor: 3`, `persistent_workers: true`
- `compile_dynamic: true`, `compile_mode: "max-autotune-no-cudagraphs"` (`unified_config.yaml:340-343`)
- `logging_steps: 10000` (`unified_config.yaml:318`)
- Optimizer: verify which is selected (Adan vs AdamW8bit vs AdamW vs other —
  state footprint per parameter differs significantly).

### Findings carried forward from prior investigation (do not re-investigate)
- The skip-validation branch (`train_direct.py:2240-2256`) is inert — pure
  Python attribute reads and a log emit. No GPU work, no `is_best` save (gated
  by strict `>` at `train_direct.py:2468` / `2504`).
- The per-epoch `gc.collect()` + `torch.cuda.empty_cache()` cleanup at
  `train_direct.py:2356-2358` lives **only inside the validation-ran `else:`
  branch**. On skip-validation epochs no cleanup runs. This is a known
  asymmetry; whether it matters depends on allocator behavior.
- `set_epoch()` on the dataset/sampler is trivial — just sets an integer
  (`dataset_loader.py:1329-1340`, `dataset_loader.py:2227-2244`).
- No `cudaMallocManaged` / unified-memory usage in the codebase (verified via grep).

## Three Review Areas (parallelizable)

Dispatch three independent agents. Each works from source code only and
returns file:line citations and concrete numbers (estimated bytes per
allocation type) where possible.

### Review 1: VRAM accounting — what's in the ~31 GB?
Build a line-item budget of dedicated GPU memory. Don't speculate from
filenames; trace the actual code that allocates VRAM.

- **Model weights**: which architecture, how many parameters, dtype. Estimate bytes.
- **Optimizer state**: which optimizer is actually constructed, what state per
  parameter (e.g., Adan keeps 3 moments → 3× param bytes; AdamW8bit quantizes
  to 1 byte; AdamW fp32 keeps 2 moments at 4 bytes each). Estimate bytes.
- **Gradients**: dtype (fp32/fp16/bf16), kept in same dtype as params? Estimate bytes.
- **Activations**: batch_size 102 × 320×320×3 input → ViT activation memory.
  With grad-accum=9, only one micro-batch of activations is live at a time.
  Are activations checkpointed? (search for `gradient_checkpointing` /
  `grad_checkpointing` / `checkpoint_sequential` / `utils.checkpoint`)
- **torch.compile workspace**: with `compile_mode="max-autotune-no-cudagraphs"`
  and `compile_dynamic=True`, Inductor allocates persistent workspace tensors
  per cached graph. Look for compile config and any explicit workspace allocations.
- **Loss scaler / mixed precision** state.
- **EMA model copy** (search for EMA / `ExponentialMovingAverage` /
  `swa_utils.AveragedModel`).
- **Pre-allocated metric / accumulator buffers** on device (`cat_probs`,
  `cat_targs` — search around `train_direct.py:2266-2434`).
- **Allocator caching headroom**: PyTorch caching allocator typically holds
  ~5-15% extra reserved-but-unused. Note `PYTORCH_CUDA_ALLOC_CONF` if set.

Output a rough table: component → estimated MB → file:line where allocated.
Then sum and compare to the observed ~31 GB. A residual of a few GB is
expected (allocator fragmentation + caching); a residual of >5 GB is worth
flagging as "unaccounted."

### Review 2: Shared GPU memory accounting — what's in the ~14 GB?
On Windows 11 + WDDM 2.x, "Shared GPU memory" can include several distinct
things. Identify which apply here:

- **Page-locked (pinned) host memory from DataLoader workers**. With
  `pin_memory=True`, each prefetched batch is pinned. Compute the worst-case
  pinned footprint:
  - Train: workers × prefetch_factor × batch_size × per-sample bytes (image +
    label tensor)
  - Val: same formula
  - Per-sample bytes: 320×320×3 = 307,200 pixels. If uint8, ~300 KB; if
    fp16/fp32, 600 KB / 1.2 MB. Plus label tensor (number of tags × dtype).
  - Verify in code which dtype is actually pinned (the collate / dataset path).
- **Driver-evicted VRAM blocks** that were promoted out of dedicated VRAM into
  system RAM by WDDM when VRAM pressure rose. These are managed by the
  Windows display driver, not by PyTorch. If they aren't touched per step,
  throughput is unaffected (matches our observation).
- **Triton / Inductor kernel binary cache** loaded into pinned memory for
  fast upload to GPU.
- **Any other host-side shadow buffers** — e.g., 8-bit optimizer block-wise
  quantization state, EMA on CPU, optimizer-offload to CPU, host-side gradient
  reductions, etc.

Compute the expected pinned-memory footprint from config and compare to ~14
GB. If the math accounts for it, this is benign and expected. If not, find
the residual.

### Review 3: Is the current layout healthy and stable?
Higher-level review of memory dynamics:

- **Will VRAM grow further during training?** Identify any per-step or
  per-epoch allocations that aren't reused (lists that append tensors, caches
  that don't evict, monotonically-growing buffers). For each, decide whether
  it's bounded.
- **The skip-validation cleanup gap** (`train_direct.py:2356-2358` only
  running on validate-ran epochs): in steady state, does this matter? If the
  allocator's free-list reaches a stable size after ~1 epoch, missed
  `empty_cache()` calls are no-ops. Validate or refute.
- **torch.compile autotune behavior**: with `dynamic=True`, how many distinct
  shape buckets exist (multi-resolution sampler buckets)? Each new shape
  triggers an autotune sweep that allocates persistent workspace. After all
  buckets have been seen once, no more autotune fires. Has training reached
  shape-stability?
- **Health check**: given throughput is unchanged, is the current layout
  acceptable? Any change you'd recommend for the next run? (e.g., lower
  `prefetch_factor` to free pinned memory, raise it to push throughput,
  enable activation checkpointing, etc.)

## Required Output Per Agent

1. **Verdict** for that review area: Healthy / Healthy with notes / Concern
2. **Numerical breakdown** with file:line citations for each component
3. **Sum vs observed** — does the math close?
4. **Optional recommendations** — only if there's a clear win. "No change
   needed" is a valid answer.

## Decision Output Required

After agents return, synthesize:
- A one-paragraph "what's in your GPU memory right now" explainer
- Whether the current layout is healthy (yes / no / yes-with-caveats)
- Any config tweaks worth considering for the next run, with expected impact
  in MB and on throughput
- Anything genuinely unaccounted for that warrants a follow-up investigation

## Process Hygiene

- Do not modify any code. File reads only.
- The training process is still running.
- Use `Grep` and `Read` directly. Spawn sub-agents only for the three review
  areas above, in parallel.
- Verify which optimizer is actually selected before estimating its state size
  — `unified_config.yaml` may declare multiple options.
