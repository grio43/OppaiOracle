# Performance Analysis Report: Downsampling GPU Accelerated (Live Test)
**Test Date:** 2025-11-14
**Test Shard:** L:\Dab\OppaiOracle\test_downsample\shard_00000
**Target Size:** 512px
**Workers:** 8 CPU cores

---

## Executive Summary

**Critical Finding:** Write operations are the primary bottleneck, consuming 64% of total processing time (331.3s out of 514.3s total).

---

## Test Results Comparison

### Dry Run (Estimation Mode)
```
Total Time:    170.0s (2.8 min)
├─ Load:       165.4s (97%) ← Bottleneck in dry run
├─ Process:      4.6s (3%)
└─ Write:        0.0s (dry run, no actual writes)

Speed: 43.1 images/sec
```

### Live Test (Actual Execution with --direct-write)
```
Total Time:    514.3s (8.6 min)
├─ Load:       164.8s (32%)
├─ Process:     18.0s (4%)
└─ Write:      331.3s (64%) ← CRITICAL BOTTLENECK

Speed: 14.3 images/sec
Processing: 2,338 images modified
Total images: 7,340 images scanned
```

---

## Performance Breakdown

### 1. Loading Phase (PHASE 1)
- **Time:** 164.8s
- **Data loaded:** 4,429.6 MB
- **Speed:** 26.9 MB/s
- **Images loaded:** 7,340
- **Load speed:** 44.5 images/sec
- **Status:** ✅ Acceptable

**Analysis:**
Loading performance is reasonable. The variance in load speed (35-64 img/s) indicates disk performance variability, likely due to:
- Disk seek times
- Competing I/O from other processes

### 2. Processing Phase (PHASE 2)
- **Time:** 18.0s (vs 4.6s dry run)
- **Images processed:** 7,340
- **Processing speed:** 407.8 images/sec
- **CPU utilization:** 8 workers
- **Status:** ✅ Good - CPU efficiently handles parallel processing

**Analysis:**
- **Dry run:** 4.6s (1,596 img/s) - Just size calculations, no actual encoding
- **Live run:** 18.0s (407.8 img/s) - Full JPEG encoding on CPU
- The 4x slowdown from dry run is expected since actual JPEG encoding is compute-intensive
- Still excellent considering JPEG encoding, resizing, and color conversion happening in parallel

**Processing breakdown:**
- 2,338 images needed processing (32% of total)
- 5,002 images skipped (already optimized)

### 3. Writing Phase (PHASE 3) ⚠️ CRITICAL ISSUE
- **Time:** 331.3s
- **Images written:** 2,338
- **Write speed:** 7.1 images/sec
- **Data written:** ~660 MB
- **Write throughput:** 2.0 MB/s
- **Status:** ❌ CRITICAL BOTTLENECK

**Analysis:**
Write performance is **13x slower than read performance** (2.0 MB/s vs 26.9 MB/s). This is the primary bottleneck.

**Possible causes:**
1. **Multiple small writes** - Writing many small files with overhead
2. **File deletion overhead** - PNG files being deleted after JPG written (2,338 deletions)
3. **Directory metadata updates** - 2,338 files being modified requires directory table updates
4. **No write buffering** - Direct write mode may be causing sync-on-write behavior

---

## Data Savings

| Metric | Value |
|--------|-------|
| Original size | 4.33 GB |
| Final size | 0.66 GB |
| **Saved** | **3.67 GB (84.7% compression)** |
| Images processed | 2,338 (32% of total) |

**Compression effectiveness:** Excellent. Converting PNG → JPEG and downsampling achieved 84.7% size reduction.

---

## Identified Issues & Bugs

### 1. ⚠️ CRITICAL: Write Bottleneck (64% of total time)
**Impact:** High - Makes processing 3x slower than it should be

**Root cause:**
- Writing 2,338 images takes 331.3s (7.1 img/s)
- Read speed is 44.5 img/s, write is only 7.1 img/s
- This is a 6.3x slowdown

**Recommendations:**
1. **Batch writes** - Write multiple images in larger sequential blocks
2. **Delayed deletes** - Don't delete PNGs immediately; batch delete at end
3. **Async writes** - Use buffered/async I/O instead of direct writes
4. **Larger write buffers** - Enable write buffering for better throughput

### 2. ⚠️ Processing Time Discrepancy
**Issue:** Processing takes 18.0s in live mode vs 4.6s in dry run

**Analysis:**
This is expected behavior, not a bug:
- Dry run: Only calculates estimated sizes (no encoding)
- Live run: Full PIL image decode, resize, JPEG encode, color conversion

**Status:** ✓ Working as intended

### 3. ⚠️ Load Speed Variance
**Issue:** Load speed varies wildly (35-64 img/s)

**Analysis:**
Disk seek time variability. Could be improved with:
- Better file sorting (by disk location, not name)
- Larger read buffer sizes
- Prefetching next shard during processing

**Impact:** Low - Load is only 32% of total time

### 4. ✅ No Error Handling Observed
The script completed successfully with no errors in either dry run or live run. Good stability.

---

## Code Review Findings

### Potential Bug: Inconsistent Write Logic

**Location:** [downsample_gpu_accelerated.py:389-412](downsample_gpu_accelerated.py#L389-L412)

```python
if DIRECT_WRITE:
    # Direct write mode - faster but less safe
    # Write directly to final location FIRST
    with open(result.output_path, 'wb') as f:
        f.write(result.image_data)

    # Only delete original AFTER successful write
    if result.should_delete_original and result.path != result.output_path:
        if result.path.exists():
            result.path.unlink()
else:
    # Safe mode: Write to temporary file first (atomic operation)
    temp_path = result.output_path.with_suffix('.tmp')

    with open(temp_path, 'wb') as f:
        f.write(result.image_data)

    # Atomic rename (NOTE: May not be truly atomic on Windows SMB)
    temp_path.replace(result.output_path)

    # Delete original AFTER successful write
    if result.should_delete_original and result.path != result.output_path:
        if result.path.exists():
            result.path.unlink()
```

**Issues:**
1. **Sequential writes** - Writing one file at a time with file open/close overhead
2. **Immediate deletes** - Delete happens immediately after each write (expensive metadata operation)
3. **No buffering** - Each `write()` call may flush to disk
4. **Network drives** - Atomic rename may not be atomic on network drives

**Recommendation:**
Refactor to batch writes and deletes:
```python
# Phase 3A: Write all files (buffered)
for result in results_to_write:
    with open(result.output_path, 'wb', buffering=8192*16) as f:  # Large buffer
        f.write(result.image_data)

# Phase 3B: Batch delete originals (after all writes complete)
to_delete = [r.path for r in results_to_write if r.should_delete_original]
for path in to_delete:
    if path.exists():
        path.unlink()
```

---

## Performance Optimization Recommendations

### Priority 1: Fix Write Bottleneck
**Expected improvement:** 3-4x speedup (from 8.6 min to 2-3 min)

1. **Increase write buffer size:**
   ```python
   with open(result.output_path, 'wb', buffering=8192*16) as f:  # 128KB buffer
       f.write(result.image_data)
   ```

2. **Batch deletes:**
   - Separate write and delete phases
   - Delete all PNGs in one batch at the end

3. **Use async I/O:**
   ```python
   import asyncio
   import aiofiles

   async def write_async(path, data):
       async with aiofiles.open(path, 'wb') as f:
           await f.write(data)
   ```

4. **Parallel writes (if safe):**
   - Write multiple files concurrently
   - Use ThreadPoolExecutor for I/O-bound writes

### Priority 2: Optimize Memory Usage
**Current:** Loads entire shard (4.4 GB) into RAM

**Recommendation:** Stream processing for very large shards:
- Load in batches of 1000 images
- Process batch
- Write batch
- Repeat

### Priority 3: Improve Load Performance
**Expected improvement:** 10-20% speedup

1. Sort files by disk block location (if API available)
2. Increase read buffer size
3. Prefetch next batch during processing

---

## Benchmark Summary

| Phase | Dry Run | Live Test | Difference |
|-------|---------|-----------|------------|
| **Load** | 165.4s (97%) | 164.8s (32%) | -0.6s |
| **Process** | 4.6s (3%) | 18.0s (4%) | +13.4s |
| **Write** | 0.0s (0%) | 331.3s (64%) | +331.3s |
| **TOTAL** | 170.0s | 514.3s | +344.3s (3x slower) |

**Overall throughput:** 14.3 images/sec (live test with writes)

---

## Conclusion

The downsampling script works correctly and achieves excellent compression (84.7%). However, **write performance is the critical bottleneck**, consuming 64% of total processing time.

**Action items:**
1. ✅ **Immediate:** Use the identified optimizations above (buffering, batch deletes)
2. ⚠️ **Short-term:** Test async writes and parallel I/O

**Expected outcome with fixes:** Total time could drop from 514s to ~150-200s (2.5-3.3 min), a 2.5-3x improvement.
