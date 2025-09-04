// SPDX-FileCopyrightText: © 2025 Spatter Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

/*
 * TEMPORARY: Loopback test kernel to isolate host vs kernel issues
 * 
 * This is a simplified version that just copies from sparse to dense buffer
 * to test if the basic buffer operations work correctly.
 * 
 * Based on tt-metal loopback example from:
 * /home/bubblepipe/tt/tt-metal/tt_metal/programming_examples/loopback/kernels/loopback_dram_copy.cpp
 * 
 * Runtime Args:
 * - arg0: l1_buffer_addr - L1 temporary buffer address
 * - arg1: num_elements - Number of elements to process
 * - arg2: delta - (ignored in loopback test)
 * - arg3: pattern_length - (ignored in loopback test)
 * - arg4: sparse_addr - Source buffer address
 * - arg5: dense_addr - Destination buffer address
 * - arg6: pattern_addr - (ignored in loopback test)
 */

void kernel_main() {
    // Read parameters from kernel arguments
    uint32_t l1_buffer_addr = get_arg_val<uint32_t>(0);
    uint32_t num_elements = get_arg_val<uint32_t>(1);
    // Skip delta and pattern_length (args 2,3) - not needed for loopback
    uint32_t sparse_addr = get_arg_val<uint32_t>(4);
    uint32_t dense_addr = get_arg_val<uint32_t>(5);
    // Skip pattern_addr (arg 6) - not needed for loopback

    // Each tile is 32x32 elements of bfloat16, which is 2 bytes per element.
    // So the tile size in bytes is 32 * 32 * 2 = 2048 bytes.
    const uint32_t tile_size_bytes = 32 * 32 * 2;
    const uint32_t elements_per_tile = 32 * 32;  // 1024 elements
    
    // Calculate number of tiles to copy
    uint32_t num_tiles = (num_elements + elements_per_tile - 1) / elements_per_tile;

    // Create TensorAccessors for sparse (source) and dense (destination)
    constexpr auto sparse_args = TensorAccessorArgs<0>();
    const auto sparse_accessor = TensorAccessor(sparse_args, sparse_addr, tile_size_bytes);

    constexpr auto dense_args = TensorAccessorArgs<sparse_args.next_compile_time_args_offset()>();
    const auto dense_accessor = TensorAccessor(dense_args, dense_addr, tile_size_bytes);

    // Simple loopback copy - read from sparse, write to dense
    for (uint32_t i = 0; i < num_tiles; i++) {
        // Read tile from source (sparse buffer) to L1
        noc_async_read_tile(i, sparse_accessor, l1_buffer_addr);
        noc_async_read_barrier();
        
        // Write tile from L1 to destination (dense buffer)
        noc_async_write_tile(i, dense_accessor, l1_buffer_addr);
        noc_async_write_barrier();
    }
    
    // Ensure all writes are flushed to DRAM
    noc_async_writes_flushed();
}