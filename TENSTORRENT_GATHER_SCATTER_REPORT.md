# TensTorrent Gather-Scatter Implementation Report
## Questions and Findings for TensTorrent Team

### Executive Summary
We have successfully implemented the gather_scatter kernel for the TensTorrent backend in the Spatter benchmark suite. The implementation works correctly for single-core execution and for multi-core execution with non-overlapping access patterns. However, we've identified a race condition issue when multiple cores write to overlapping memory regions, which requires atomic operations to resolve.

---

## 1. Implementation Overview

### What We Built
- **Kernel**: [`gather_scatter_kernel.cpp`](https://github.com/bubblepipe/tt-spatter/blob/main/src/Spatter/kernels/gather_scatter_kernel.cpp) - Implements sparse-to-sparse copy operation
- **Operation**: `sparse_scatter[pattern_scatter[j] + delta_scatter * i] = sparse_gather[pattern_gather[j] + delta_gather * i]`
- **Architecture**: Multi-core support using `split_work_to_cores()`
- **Memory**: Uses 4 L1 buffers (pattern_gather, pattern_scatter, sparse_gather, sparse_scatter)

### Performance Results
- **Single-core**: ~127 MB/s (50K elements with complex pattern)
- **Validation**: Passes all tests with non-overlapping patterns

---

## 2. The Race Condition Issue

### Problem Description
When multiple cores process the gather_scatter operation in parallel, they may write to overlapping regions of the sparse_scatter buffer, causing race conditions.

### Example
```bash
# This test fails with 4 cores but passes with 1 core
./spatter --backend tenstorrent --tt-cores 4 -k gs \
          -g "UNIFORM:64:4" -u "UNIFORM:64:8" -l 50000

# Result: 185 mismatches in validation
```

### Root Cause Analysis
1. Each core processes a range of iterations (e.g., core 0: iterations 0-799, core 1: iterations 800-1599)
2. With pattern_scatter containing values up to 504 and delta_scatter=8:
   - Core 0 might write to index: 504 + 8*799 = 6896
   - Core 1 might write to index: 0 + 8*800 = 6400
   - These cores could write to the same tile (tiles are 1024 elements)
3. The kernel performs read-modify-write on tiles:
   ```cpp
   // Read tile
   noc_async_read_tile(dst_tile_idx, sparse_scatter_accessor, sparse_scatter_l1_addr);
   // Modify in L1
   sparse_scatter_data[dst_elem_offset] = src_value;
   // Write back
   noc_async_write_tile(dst_tile_idx, sparse_scatter_accessor, sparse_scatter_l1_addr);
   ```
4. Without synchronization, the last core to write back wins, losing other cores' updates

---

## 3. Comparison with Other Backends

### Serial (CPU) Implementation
```cpp
// From src/Spatter/Configuration.cc:477-492
void Configuration<Spatter::Serial>::gather_scatter(
    bool timed, unsigned long run_id) {
  assert(pattern_scatter.size() == pattern_gather.size());
  size_t pattern_length = pattern_scatter.size();

  if (timed)
    timer.start();

  for (size_t i = 0; i < count; ++i)
    for (size_t j = 0; j < pattern_length; ++j)
      sparse_scatter[pattern_scatter[j] + delta_scatter * i] =
          sparse_gather[pattern_gather[j] + delta_gather * i];

  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
}
```
- **Sequential execution** - No race conditions possible
- **Nested loops** - Outer loop over iterations, inner over pattern

### OpenMP Implementation
```cpp
// From src/Spatter/Configuration.cc:657-678
void Configuration<Spatter::OpenMP>::gather_scatter(
    bool timed, unsigned long run_id) {
  assert(pattern_scatter.size() == pattern_gather.size());
  size_t pattern_length = pattern_scatter.size();

  if (timed)
    timer.start();

#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    double *tl = sparse_scatter.data() + delta_scatter * i;
    double *sl = sparse_gather.data() + delta_gather * i;

#pragma omp simd
    for (size_t j = 0; j < pattern_length; ++j) {
      tl[pattern_scatter[j]] = sl[pattern_gather[j]];
    }
  }

  if (atomic_fence)
    std::atomic_thread_fence(std::memory_order_release);

  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
}
```
- **Parallel outer loop** - Each thread handles complete iterations
- **SIMD inner loop** - Vectorization for pattern processing
- **Race condition potential** - Multiple threads may write to same location
- **Optional atomic fence** - Memory ordering guarantee

### CUDA Implementation
CUDA provides two versions:

**Non-atomic version (has race conditions):**
```cuda
// From src/Spatter/CudaBackend.cu:136-150
__global__ void cuda_gather_scatter(const size_t *pattern_scatter,
    double *sparse_scatter, const size_t *pattern_gather,
    const double *sparse_gather, const size_t pattern_length,
    const size_t delta_scatter, const size_t delta_gather, const size_t wrap,
    const size_t count) {
  size_t total_id =
      (size_t)((size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x);
  size_t j = total_id % pattern_length; // pat_idx
  size_t i = total_id / pattern_length; // count_idx

  if (i < count)
    sparse_scatter[pattern_scatter[j] + delta_scatter * i] =
        sparse_gather[pattern_gather[j] + delta_gather * i];
}
```

**Atomic version (handles race conditions correctly):**
```cuda
// From src/Spatter/CudaBackend.cu:152-168
__global__ void cuda_gather_scatter_atomic(const size_t *pattern_scatter,
    double *sparse_scatter, const size_t *pattern_gather,
    const double *sparse_gather, const size_t pattern_length,
    const size_t delta_scatter, const size_t delta_gather, const size_t wrap,
    const size_t count) {
  size_t total_id =
      (size_t)((size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x);
  size_t j = total_id % pattern_length; // pat_idx
  size_t i = total_id / pattern_length; // count_idx

  if (i < count)
    atomicExch((unsigned long long int *)&sparse_scatter[pattern_scatter[j] +
                   delta_scatter * i],
        __double_as_longlong(
            sparse_gather[pattern_gather[j] + delta_gather * i]));
}
```

**CUDA wrapper selection:**
```cpp
// From src/Spatter/Configuration.cc:877-884
if (atomic)
  time_ms = cuda_gather_scatter_atomic_wrapper(dev_pattern_scatter,
      dev_sparse_scatter, dev_pattern_gather, dev_sparse_gather,
      pattern_length, delta_scatter, delta_gather, wrap, count);
else
  time_ms = cuda_gather_scatter_wrapper(dev_pattern_scatter,
      dev_sparse_scatter, dev_pattern_gather, dev_sparse_gather,
      pattern_length, delta_scatter, delta_gather, wrap, count);
```

### Key Differences
| Backend | Parallelization | Race Condition Handling |
|---------|----------------|------------------------|
| **Serial** | None (sequential) | N/A - No races possible |
| **OpenMP** | Thread-level (outer loop) | Optional atomic_fence, but no atomic writes |
| **CUDA (non-atomic)** | Thread per element | No handling - has races |
| **CUDA (atomic)** | Thread per element | atomicExch prevents races |
| **TensTorrent** | Core-level (work split) | No handling - has races |

### Our TensTorrent Implementation
- Currently behaves like **non-atomic CUDA version**
- Race conditions occur with overlapping writes
- Need atomic operations to match CUDA's atomic version

---

## 4. Investigation of TT-Metal Atomic Support

### Available Atomic Operations (found in `tt-metal/tt_metal/hw/inc/blackhole/noc/noc.h`)
```cpp
// Atomic increment only
void noc_atomic_increment(uint32_t noc_coordinate, uint64_t addr, 
                          uint32_t incr, uint32_t wrap, bool linked);

void noc_atomic_read_and_increment(...);  // Fetch-and-add
void noc_multicast_atomic_increment(...); // Multicast atomic increment
```

### What's Missing for Scatter Operations
- ❌ **Atomic exchange** (atomicExch equivalent)
- ❌ **Atomic write** (store with guarantee)
- ❌ **Compare-and-swap** (CAS operations)
