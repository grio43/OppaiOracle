# TODO: Multi-Drive L2 Caches (14 TiB each)

- Goal: Configure additional LMDB L2 caches on separate drives, each sized to 14 TiB, to distribute IO and capacity.
- Path: For each run, set `data.l2_cache_path` to a unique directory on the target drive (e.g., `/mnt/driveB/l2_cache_b`). One environment (directory) per run.
- Size: Use `data.l2_max_size_gb: 14336.0` (14 TiB) or `data.l2_map_size_bytes: 15393162788864` in the run’s config. No code changes required.
- Filesystem: Verify the target filesystem supports ≥14 TiB single files and sparse allocation (XFS preferred; ext4 may work depending on features). Check free space and quotas.
- Concurrency: Ensure a single writer per environment. Do not point two writers at the same `l2_cache_path`. Any number of read-only processes may open the same environment.
- Process model: Our code uses one environment per process/run. Running many huge environments simultaneously in a single process is not supported; use separate processes.
- Warm-up: First pass populates L2; subsequent epochs/runs see hits. With manifest mode, L1 may mask some L2 reads. Monitor `du -sh` on the env dir to confirm growth.
- Logging: Confirm startup log “L2 cache enabled at … (map_size_bytes=…)” after changing the config. Expect no lock contention: readers don’t block writers.
- Git hygiene: `l2_cache/` is gitignored. If using different paths, ensure those directories are also ignored to avoid pushing huge sparse files.
- Optional symlink: If keeping a stable relative path, make `./l2_cache` a symlink to the target drive’s directory per run.
- Apply changes: Stop running jobs before changing `l2_cache_path`. To resize an existing env immediately, stop processes and remove the old env directory; otherwise, writer will grow on `MapFull` but readers may need reopen.
- Verify: After launch, `ls -lh …/data.mdb` should show ~14T size (sparse). `du -sh` reflects actual usage. Confirm no errors in logs regarding `MapFull` or `MDB_MAP_RESIZED`.

