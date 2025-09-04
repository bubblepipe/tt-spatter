// SPDX-FileCopyrightText: © 2025 Spatter Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

/*
 * Gather Kernel for Spatter TensTorrent Backend - HARDCODED PATTERN VERSION
 * 
 * This version uses hardcoded patterns to isolate gather logic from pattern buffer issues.
 * 
 * Implements: dense[i] = sparse[pattern[i % pattern_length] + delta * (i / pattern_length)]
 * 
 * Runtime Args:
 * - arg0: l1_buffer_addr - L1 temporary buffer address
 * - arg1: num_elements - Total number of elements to gather
 * - arg2: delta - Stride between pattern iterations
 * - arg3: pattern_length - Length of the pattern array (used to select hardcoded pattern)
 * - arg4: sparse_addr - Source buffer address (DRAM)
 * - arg5: dense_addr - Destination buffer address (DRAM)
 * - arg6: pattern_addr - (IGNORED - using hardcoded patterns)
 */

void kernel_main() {
    // Read runtime arguments
    uint32_t l1_buffer_addr = get_arg_val<uint32_t>(0);
    uint32_t num_elements = get_arg_val<uint32_t>(1);
    uint32_t delta = get_arg_val<uint32_t>(2);
    uint32_t pattern_length = get_arg_val<uint32_t>(3);
    uint32_t sparse_addr = get_arg_val<uint32_t>(4);
    uint32_t dense_addr = get_arg_val<uint32_t>(5);
    // arg6 (pattern_addr) is ignored - we use hardcoded patterns

    // Tile constants
    const uint32_t tile_size_bytes = 32 * 32 * 2;  // 2048 bytes per tile
    const uint32_t elements_per_tile = 32 * 32;    // 1024 elements per tile

    // Create TensorAccessors for sparse and dense buffers only
    constexpr auto sparse_args = TensorAccessorArgs<0>();
    const auto sparse_accessor = TensorAccessor(sparse_args, sparse_addr, tile_size_bytes);
    
    constexpr auto dense_args = TensorAccessorArgs<sparse_args.next_compile_time_args_offset()>();
    const auto dense_accessor = TensorAccessor(dense_args, dense_addr, tile_size_bytes);


    // HARDCODED PATTERNS for testing
    // UNIFORM:8:1 pattern = [0,1,2,3,4,5,6,7]
    uint32_t pattern_uniform_8_1[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    
    // UNIFORM:16:3 pattern = [0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45]
    uint32_t pattern_uniform_16_3[16] = {0, 3, 6, 9, 12, 15, 18, 21, 
                                          24, 27, 30, 33, 36, 39, 42, 45};
    
    // // Select pattern based on pattern_length
    // uint32_t* pattern_data;
    // if (pattern_length == 8) {
    //     pattern_data = pattern_uniform_8_1;
    // } else if (pattern_length == 16) {
    //     pattern_data = pattern_uniform_16_3;
    // } else {
    //     // For any other pattern length, default to sequential [0,1,2,3,...]
    //     // This is limited to 64 elements for stack allocation
    //     uint32_t default_pattern[64];
    //     for (uint32_t i = 0; i < pattern_length && i < 64; i++) {
    //         default_pattern[i] = i;
    //     }
    //     pattern_data = default_pattern;
    // }

    uint32_t* pattern_data = pattern_uniform_8_1;

    // Process output in tiles
    uint32_t num_output_tiles = (num_elements + elements_per_tile - 1) / elements_per_tile;
    
    // Track the last loaded sparse tile to avoid redundant loads
    uint32_t last_sparse_tile = UINT32_MAX;
    
    for (uint32_t out_tile_idx = 0; out_tile_idx < num_output_tiles; out_tile_idx++) {
        // Calculate the range of elements for this output tile
        uint32_t tile_start = out_tile_idx * elements_per_tile;
        uint32_t tile_end = tile_start + elements_per_tile;
        if (tile_end > num_elements) {
            tile_end = num_elements;
        }
        
        // Clear the dense output tile first
        uint16_t* dense_data = reinterpret_cast<uint16_t*>(dense_addr);
        for (uint32_t i = 0; i < elements_per_tile; i++) {
            dense_data[i] = 0;
        }
        
        // Gather elements for this output tile
        for (uint32_t elem_idx = tile_start; elem_idx < tile_end; elem_idx++) {
            // Calculate pattern index and iteration
            uint32_t pattern_idx = elem_idx % pattern_length;
            uint32_t iteration = elem_idx / pattern_length;
            
            // Get the base index from pattern and add delta offset
            uint32_t src_index = pattern_data[pattern_idx] + (delta * iteration);
            
            // Determine which sparse tile contains this element
            uint32_t src_tile_idx = src_index / elements_per_tile;
            uint32_t src_elem_offset = src_index % elements_per_tile;
            
            // Load the sparse tile if not already loaded
            if (src_tile_idx != last_sparse_tile) {
                noc_async_read_tile(src_tile_idx, sparse_accessor, sparse_addr);
                noc_async_read_barrier();
                last_sparse_tile = src_tile_idx;
            }
            
            // Copy the element from sparse to dense
            uint16_t* sparse_data = reinterpret_cast<uint16_t*>(sparse_addr);
            uint32_t dense_offset = elem_idx - tile_start;
            dense_data[dense_offset] = sparse_data[src_elem_offset];
        }
        
        // Write the completed output tile back to DRAM
        noc_async_write_tile(out_tile_idx, dense_accessor, dense_addr);
        noc_async_write_barrier();
    }
    

    //  for (uint32_t i = 0; i < num_output_tiles; i++) {
    //     // Issue a read to the NoC and write to the L1 buffer. This operation is asynchronous.
    //     // thus a barrier is needed to ensure that the read is complete before the write.
    //     noc_async_read_tile(i, in0, l1_buffer_addr);
    //     noc_async_read_barrier();
    //     // Write back the tile to the destination DRAM buffer.
    //     // Again, this is an asynchronous operation, so we need a barrier to ensure the write
    //     // is complete before the next iteration.
    //     noc_async_write_tile(i, out0, l1_buffer_addr);
    //     noc_async_write_barrier();
    // }


    // Ensure all writes are complete
    noc_async_writes_flushed();
}