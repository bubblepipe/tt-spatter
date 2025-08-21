// SPDX-FileCopyrightText: © 2025 Spatter Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

/*
 * Scatter Kernel for Spatter TensTorrent Backend
 * 
 * Implements: sparse[pattern[j] + delta * i] = dense[j]
 * 
 * Runtime Args:
 * - arg0: src_buffer_addr - Source (dense) buffer address in DRAM
 * - arg1: dst_buffer_addr - Destination (sparse) buffer address in DRAM  
 * - arg2: pattern_buffer_addr - Pattern array buffer address in DRAM
 * - arg3: num_elements - Number of elements to scatter
 * - arg4: delta - Stride parameter for iterations
 */

void kernel_main() {
    // Get kernel arguments
    uint32_t src_buffer_addr = get_arg_val<uint32_t>(0);    // dense buffer
    uint32_t dst_buffer_addr = get_arg_val<uint32_t>(1);    // sparse buffer
    uint32_t pattern_buffer_addr = get_arg_val<uint32_t>(2);
    uint32_t num_elements = get_arg_val<uint32_t>(3);
    uint32_t delta = get_arg_val<uint32_t>(4);

    constexpr uint32_t tile_size_bytes = 32 * 32 * 2; // BFloat16 tile size
    constexpr uint32_t elements_per_tile = 32 * 32;
    constexpr uint32_t l1_buffer_base = 0x10000; // L1 staging area
    
    // Set up tensor accessors using proper TT-Metal pattern
    constexpr auto src_args = TensorAccessorArgs<0>();
    const auto src_accessor = TensorAccessor(src_args, src_buffer_addr, tile_size_bytes);
    
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const auto dst_accessor = TensorAccessor(dst_args, dst_buffer_addr, tile_size_bytes);
    
    constexpr auto pattern_args = TensorAccessorArgs<dst_args.next_compile_time_args_offset()>();
    const auto pattern_accessor = TensorAccessor(pattern_args, pattern_buffer_addr, tile_size_bytes);
    
    // L1 buffer layout:
    // 0x10000: pattern data tile
    // 0x10800: source data tile (dense)
    // 0x11000: destination data tile (sparse)
    uint32_t pattern_l1_addr = l1_buffer_base;
    uint32_t src_l1_addr = l1_buffer_base + tile_size_bytes;
    uint32_t dst_l1_addr = l1_buffer_base + 2 * tile_size_bytes;
    
    // Process elements by tiles for better NOC efficiency
    uint32_t num_tiles = (num_elements + elements_per_tile - 1) / elements_per_tile;
    
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        uint32_t elements_in_tile = (tile_idx == num_tiles - 1) ? 
            (num_elements - tile_idx * elements_per_tile) : elements_per_tile;
        
        // Read pattern tile for this chunk
        noc_async_read_page(tile_idx, pattern_accessor, pattern_l1_addr);
        noc_async_read_barrier();
        
        // Read source tile (dense data to scatter)
        noc_async_read_page(tile_idx, src_accessor, src_l1_addr);
        noc_async_read_barrier();
        
        // Cast L1 buffers to appropriate types
        uint32_t* pattern_data = reinterpret_cast<uint32_t*>(pattern_l1_addr);
        uint16_t* src_data = reinterpret_cast<uint16_t*>(src_l1_addr); // BFloat16
        
        // Process each element in the tile
        for (uint32_t elem_idx = 0; elem_idx < elements_in_tile; elem_idx++) {
            uint32_t global_elem_idx = tile_idx * elements_per_tile + elem_idx;
            uint32_t pattern_index = pattern_data[elem_idx];
            uint32_t dst_index = pattern_index + delta * (global_elem_idx / elements_per_tile);
            
            // Calculate which destination tile we need to update
            uint32_t dst_tile_idx = dst_index / elements_per_tile;
            uint32_t dst_elem_offset = dst_index % elements_per_tile;
            
            // Read destination tile (sparse buffer)
            noc_async_read_page(dst_tile_idx, dst_accessor, dst_l1_addr);
            noc_async_read_barrier();
            
            uint16_t* dst_data = reinterpret_cast<uint16_t*>(dst_l1_addr);
            
            // Perform scatter operation: sparse[pattern[j] + delta * i] = dense[j]
            dst_data[dst_elem_offset] = src_data[elem_idx];
            
            // Write back the updated destination tile
            noc_async_write_page(dst_tile_idx, dst_accessor, dst_l1_addr);
            noc_async_write_barrier();
        }
    }
    
    // Invalidate L1 cache for Blackhole architecture
    invalidate_l1_cache();
}