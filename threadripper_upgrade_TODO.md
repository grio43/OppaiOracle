Threadripper Upgrade — CPU Utilization & Throughput TODO

Purpose: Prepare the training stack to leverage a high‑core Threadripper so the GPU stays fed and decoding/augmentation runs efficiently on CPU. No code changes required; focus on config, environment, and OS.

Goals
- Keep GPU at high utilization with minimal input stalls.
- Shift and parallelize more input work onto CPU (decode, letterbox, normalize).
- Reduce decode/copy latency via caching, pinning, and prefetching.

DataLoader & Prefetch
- data.num_workers: Increase substantially on TR. Start with 24–32 for 32C/64T; try 48 if CPU headroom remains. Tune by watching GPU idle time.
- data.prefetch_factor: Raise to 4–8 (total outstanding = workers × prefetch).
- data.pin_memory: true (keeps non_blocking H2D copies effective).
- data.persistent_workers: true (avoid per‑epoch fork cost).
- validation.dataloader: Mirror a smaller but higher‑than‑default setting (e.g., 16 workers, prefetch 4) so eval doesn’t bottleneck.
- data.preload_files: Increase modestly (e.g., 8–32) to warm caches at startup without freezing the UI.

Caching (RAM/NVMe)
- L1 (per‑worker RAM): l1_per_worker_mb scales with workers. Choose 256–512 MB each when running many workers. Example: 32 workers × 512 MB ≈ 16 GB.
- L2 (LMDB on fast NVMe): Ensure l2_cache_enabled: true, set l2_cache_path to a local NVMe, and provision a large l2_max_size_gb (hundreds of GB or more on TR workstation SSDs).
- l2_max_readers: Increase if you scale workers high (e.g., 8192) to avoid reader contention.
- Writer queue depth: Set env L2_WRITER_QUEUE_MAX=4096 so the async writer doesn’t drop writes under heavy load.

Faster CPU Decode
- Install Pillow‑SIMD with libjpeg‑turbo for faster JPEG decode:
  - pip uninstall pillow
  - pip install --upgrade pillow-simd
  - On the OS, install libjpeg‑turbo. Verify: python -c "from PIL import features; print(features.check_feature('libjpeg_turbo'))"
- Keep images local (avoid NAS latency for hot shards); reserve NAS for cold tiers.

Environment & OS Tuning
- Avoid BLAS oversubscription in workers (critical with many DataLoader processes):
  - OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
- File descriptor limits (LMDB, many files): ulimit -n 65535
- CPU governor: Set performance and ensure turbo is enabled.
- NUMA: If multiple memory controllers, launch with numactl --interleave=all for balanced memory access.
- Disk scheduler: Use mq-deadline/none for NVMe; mount with noatime where appropriate.

Storage Layout
- Place l2_cache_path on local NVMe SSD.
- Keep active shards on local SSD; background/cold shards can stay on NAS but will slow workers.

Monitoring & Validation
- Watch logs for “L2 writer queue full” or “L2 cache decode failed” to catch pressure/size issues; increase queue size, map size, or readers accordingly.
- Observe GPU utilization alongside CPU utilization and dataloader wait times in your monitoring. Increase workers/prefetch only while GPU idle time decreases.
- Keep pin_memory true and confirm non_blocking=True is active (already wired in train loop).

Suggested YAML Tweaks (start point)
- configs/unified_config.yaml → data:
  - num_workers: 32
  - prefetch_factor: 6
  - pin_memory: true
  - persistent_workers: true
  - l1_per_worker_mb: 512
  - l2_max_readers: 8192 (add if missing)
- validation.dataloader:
  - num_workers: 16
  - prefetch_factor: 4

Example Launch (env knobs)
- OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_MAX_THREADS=1 L2_WRITER_QUEUE_MAX=4096 \
  python train_direct.py --config configs/unified_config.yaml

Rollout Plan
- Step 1: Install Pillow‑SIMD + libjpeg‑turbo; keep current workers; verify throughput (+10–30% typical on JPEG‑heavy sets).
- Step 2: Raise num_workers (16 → 24 → 32) and prefetch_factor (2 → 4 → 6), measure images/sec and GPU idle.
- Step 3: Tune l1_per_worker_mb to fit RAM; validate no swapping.
- Step 4: Move/resize L2 on NVMe; set l2_max_readers and writer queue env.
- Step 5: Validate evaluation loader settings so validation doesn’t bottleneck.

Optional (later, if desired; requires code changes)
- Add torchvision v2 joint_transforms (color jitter, random erasing) to use CPU more without desyncing masks.
- Move validation metrics to CPU to shift eval work off GPU.

Risks & Watchouts
- Too many workers can thrash the page cache and hurt end‑to‑end latency; watch GPU idle metrics.
- L1 memory is per‑worker; multiplying by workers can exceed system RAM—monitor and adjust.
- NAS backing will limit benefits; prioritize local NVMe for hot paths and LMDB.

