// SPDX-FileCopyrightText: © 2025 Spatter Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

/*
 * Gather Kernel for Spatter TensTorrent Backend
 * 
 * Implements: dense[i] = sparse[pattern[i % pattern_length] + delta * (i / pattern_length)]
 * 
 * Runtime Args:
 * - arg0: pattern_l1_addr - Pattern L1 buffer address
 * - arg1: sparse_l1_addr - Sparse L1 buffer address
 * - arg2: dense_l1_addr - Dense L1 buffer address
 * - arg3: num_elements - Total number of elements to gather
 * - arg4: delta - Stride between pattern iterations
 * - arg5: pattern_length - Length of the pattern array
 * - arg6: sparse_addr - Source buffer address (DRAM)
 * - arg7: dense_addr - Destination buffer address (DRAM)
 * - arg8: pattern_addr - Pattern buffer address (DRAM)
 */

void kernel_main() {
    // Read runtime arguments - three separate L1 buffers
    uint32_t pattern_l1_addr = get_arg_val<uint32_t>(0);
    uint32_t sparse_l1_addr = get_arg_val<uint32_t>(1);
    uint32_t dense_l1_addr = get_arg_val<uint32_t>(2);
    uint32_t num_elements = get_arg_val<uint32_t>(3);
    uint32_t delta = get_arg_val<uint32_t>(4);
    uint32_t pattern_length = get_arg_val<uint32_t>(5);
    uint32_t sparse_addr = get_arg_val<uint32_t>(6);
    uint32_t dense_addr = get_arg_val<uint32_t>(7);
    uint32_t pattern_addr = get_arg_val<uint32_t>(8);

    // Tile constants
    const uint32_t tile_size_bytes = 32 * 32 * 2;  // 2048 bytes per tile
    const uint32_t elements_per_tile = 32 * 32;    // 1024 elements per tile

    // Create TensorAccessors for all three buffers
    constexpr auto sparse_args = TensorAccessorArgs<0>();
    const auto sparse_accessor = TensorAccessor(sparse_args, sparse_addr, tile_size_bytes);
    
    constexpr auto dense_args = TensorAccessorArgs<sparse_args.next_compile_time_args_offset()>();
    const auto dense_accessor = TensorAccessor(dense_args, dense_addr, tile_size_bytes);
    
    constexpr auto pattern_args = TensorAccessorArgs<dense_args.next_compile_time_args_offset()>();
    const auto pattern_accessor = TensorAccessor(pattern_args, pattern_addr, tile_size_bytes);
    
    // Load pattern tile once (it will be reused)
    // Pattern is stored as uint32_t values
    noc_async_read_tile(0, pattern_accessor, pattern_l1_addr);
    noc_async_read_barrier();
    
    // Get pointer to pattern data in L1
    uint32_t* pattern_data = reinterpret_cast<uint32_t*>(pattern_l1_addr);

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
        
        // Clear the dense output tile in L1
        uint16_t* dense_data = reinterpret_cast<uint16_t*>(dense_l1_addr);
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
                noc_async_read_tile(src_tile_idx, sparse_accessor, sparse_l1_addr);
                noc_async_read_barrier();
                last_sparse_tile = src_tile_idx;
            }
            
            // Copy the element from sparse L1 to dense L1
            uint16_t* sparse_data = reinterpret_cast<uint16_t*>(sparse_l1_addr);
            uint32_t dense_offset = elem_idx - tile_start;
            dense_data[dense_offset] = sparse_data[src_elem_offset];
        }
        
        // Write the completed output tile from L1 back to DRAM
        noc_async_write_tile(out_tile_idx, dense_accessor, dense_l1_addr);
        noc_async_write_barrier();
    }
    
    noc_async_writes_flushed();
}