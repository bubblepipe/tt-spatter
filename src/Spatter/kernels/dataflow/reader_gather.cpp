// SPDX-FileCopyrightText: © 2025 Spatter Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"

/*
 * Reader Kernel for Gather Operation (Multi-core version)
 * 
 * Reads sparse data and pattern tiles from DRAM and feeds them to compute kernel
 * via circular buffers. Follows the official TT-Metal 3-kernel architecture.
 * 
 * Runtime Args:
 * - arg0: src_buffer_addr - Source (sparse) buffer address in DRAM
 * - arg1: pattern_buffer_addr - Pattern array buffer address in DRAM
 * - arg2: work_offset - Starting offset for this core's work
 * - arg3: work_per_core - Number of elements for this core to process
 * - arg4: delta - Stride parameter for iterations
 * - arg5: sparse_size - Total size of sparse buffer (for bounds checking)
 */

void kernel_main() {
    // Get runtime arguments
    uint32_t src_buffer_addr = get_arg_val<uint32_t>(0);
    uint32_t pattern_buffer_addr = get_arg_val<uint32_t>(1);
    uint32_t work_offset = get_arg_val<uint32_t>(2);
    uint32_t work_per_core = get_arg_val<uint32_t>(3);
    uint32_t delta = get_arg_val<uint32_t>(4);
    uint32_t sparse_size = get_arg_val<uint32_t>(5);

    // Circular buffer indices matching matmul pattern
    constexpr uint32_t cb_id_sparse = tt::CBIndex::c_0;   // sparse data
    constexpr uint32_t cb_id_pattern = tt::CBIndex::c_1;  // pattern indices

    // Get tile sizes from circular buffers
    const uint32_t sparse_tile_bytes = get_tile_size(cb_id_sparse);
    const uint32_t pattern_tile_bytes = get_tile_size(cb_id_pattern);

    // Create TensorAccessors for DRAM buffers
    constexpr auto sparse_args = TensorAccessorArgs<0>();
    const auto sparse_accessor = TensorAccessor(sparse_args, src_buffer_addr, sparse_tile_bytes);

    constexpr auto pattern_args = TensorAccessorArgs<sparse_args.next_compile_time_args_offset()>();
    const auto pattern_accessor = TensorAccessor(pattern_args, pattern_buffer_addr, pattern_tile_bytes);

    // Calculate tile dimensions
    constexpr uint32_t elements_per_tile = 32 * 32;  // 1024 elements per BFloat16 tile
    
    // Calculate work range for this core
    uint32_t start_element = work_offset;
    uint32_t end_element = work_offset + work_per_core;
    
    // Calculate tile range
    uint32_t start_tile = start_element / elements_per_tile;
    uint32_t end_tile = (end_element + elements_per_tile - 1) / elements_per_tile;

    // Process each output tile this core is responsible for
    for (uint32_t tile_idx = start_tile; tile_idx < end_tile; tile_idx++) {
        // Calculate element range within this tile for this core
        uint32_t tile_start_elem = tile_idx * elements_per_tile;
        uint32_t tile_end_elem = (tile_idx + 1) * elements_per_tile;
        
        // Intersect tile range with this core's work range
        uint32_t work_start_in_tile = (start_element > tile_start_elem) ? (start_element - tile_start_elem) : 0;
        uint32_t work_end_in_tile = (end_element < tile_end_elem) ? (end_element - tile_start_elem) : elements_per_tile;
        
        if (work_start_in_tile >= work_end_in_tile) continue; // No work in this tile

        // Read pattern tile for this output chunk
        {
            cb_reserve_back(cb_id_pattern, 1);
            uint32_t l1_write_addr_pattern = get_write_ptr(cb_id_pattern);
            noc_async_read_tile(tile_idx, pattern_accessor, l1_write_addr_pattern);
            noc_async_read_barrier();
            cb_push_back(cb_id_pattern, 1);
        }

        // For gather operation, we need to read sparse tiles based on pattern indices
        // This is more complex than matmul since we need indirect addressing
        // We'll read multiple sparse tiles that might be needed for this output tile
        
        // For now, read the sparse tiles in a straightforward manner
        // The compute kernel will handle the gather logic with pattern indirection
        {
            cb_reserve_back(cb_id_sparse, 1);
            uint32_t l1_write_addr_sparse = get_write_ptr(cb_id_sparse);
            
            // For gather, we need to determine which sparse tiles are needed
            // This is a simplified approach - read consecutive tiles
            // The compute kernel will handle the complex pattern-based access
            noc_async_read_tile(tile_idx, sparse_accessor, l1_write_addr_sparse);
            noc_async_read_barrier();
            cb_push_back(cb_id_sparse, 1);
        }
    }
}