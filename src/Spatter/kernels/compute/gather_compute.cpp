// SPDX-FileCopyrightText: © 2025 Spatter Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary.h"
#include "compute_kernel_api.h"

using std::uint32_t;

/*
 * Compute Kernel for Gather Operation (Multi-core version)
 * 
 * Performs the core gather logic: dense[j] = sparse[pattern[j] + delta * i]
 * Reads from circular buffers populated by reader, outputs to circular buffer
 * for writer. Follows the official TT-Metal 3-kernel architecture.
 * 
 * Runtime Args:
 * - arg0: num_tiles - Number of tiles to process
 * - arg1: delta - Stride parameter for iterations
 * - arg2: elements_per_tile - Number of elements per tile (1024 for BFloat16)
 */

namespace NAMESPACE {

void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);    // number of tiles to process
    uint32_t delta = get_arg_val<uint32_t>(1);        // stride parameter
    uint32_t elements_per_tile = get_arg_val<uint32_t>(2); // elements per tile

    // Circular buffer indices matching reader/writer
    constexpr tt::CBIndex cb_sparse = tt::CBIndex::c_0;   // sparse data from reader
    constexpr tt::CBIndex cb_pattern = tt::CBIndex::c_1;  // pattern indices from reader
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;     // output to writer

    // Initialize compute kernel for copy operations
    // Since gather is primarily a data movement operation, we use copy operations
    copy_tile_init();
    
    // Process each tile
    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        // Wait for input tiles to be available from reader
        cb_wait_front(cb_sparse, 1);
        cb_wait_front(cb_pattern, 1);
        
        // Acquire destination registers for output tile
        acquire_dst();
        
        // For the gather operation, we need to implement the pattern-based indexing
        // dense[j] = sparse[pattern[j] + delta * i]
        // 
        // In the tile-based system, this means:
        // 1. Read the pattern tile to understand indices
        // 2. Use those indices to gather from sparse tile
        // 3. Write result to output tile
        //
        // Since TT-Metal compute kernels work primarily with full tiles,
        // the detailed element-wise pattern indexing is challenging to implement
        // efficiently in the compute kernel. For now, we do a simple tile copy
        // and let the reader handle the complex indexing logic.
        //
        // TODO: Implement true pattern-based gather using compute APIs
        // This might require using unpack/pack operations to access individual elements
        
        // Simple tile copy for now (sparse -> output)
        // This preserves the data flow but doesn't implement true gather logic yet
        copy_tile(cb_sparse, 0, 0);
        
        // Reserve space in output circular buffer
        cb_reserve_back(cb_out, 1);
        
        // Pack the result tile to output
        pack_tile(0, cb_out);
        
        // Signal that output tile is ready
        cb_push_back(cb_out, 1);
        
        // Release destination registers
        release_dst();
        
        // Mark input tiles as consumed
        cb_pop_front(cb_sparse, 1);
        cb_pop_front(cb_pattern, 1);
    }
}

} // namespace NAMESPACE