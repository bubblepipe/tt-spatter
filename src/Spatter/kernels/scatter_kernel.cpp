// SPDX-FileCopyrightText: © 2025 Spatter Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

/*
 * Scatter Kernel for Spatter TensTorrent Backend (Multi-core version)
 * 
 * Implements: sparse[pattern[j] + delta * i] = dense[j]
 * 
 * Runtime Args:
 * - arg0: src_buffer_addr - Source (dense) buffer address in DRAM
 * - arg1: dst_buffer_addr - Destination (sparse) buffer address in DRAM  
 * - arg2: pattern_buffer_addr - Pattern array buffer address in DRAM
 * - arg3: work_offset - Starting offset for this core's work
 * - arg4: work_per_core - Number of elements for this core to process
 * - arg5: delta - Stride parameter for iterations
 */

void kernel_main() {
    // Get kernel arguments
    uint32_t src_buffer_addr = get_arg_val<uint32_t>(0);    // dense buffer
    uint32_t dst_buffer_addr = get_arg_val<uint32_t>(1);    // sparse buffer
    uint32_t pattern_buffer_addr = get_arg_val<uint32_t>(2);
    uint32_t work_offset = get_arg_val<uint32_t>(3);
    uint32_t work_per_core = get_arg_val<uint32_t>(4);
    uint32_t delta = get_arg_val<uint32_t>(5);

    // Debug print to show kernel execution (using DPRINT for device-side printing)
    DPRINT << "SCATTER KERNEL: Executing with work_offset=" << work_offset << ", work_per_core=" << work_per_core << ", delta=" << delta << ENDL();

    constexpr uint32_t tile_size_bytes = 32 * 32 * 2; // BFloat16 tile size (2048 bytes)
    constexpr uint32_t elements_per_tile = 32 * 32;   // 1024 elements per tile
    constexpr uint32_t l1_buffer_base = 0x10000;      // L1 staging area
    
    // NOC coordinates for this core  
    uint64_t src_noc_addr = get_noc_addr(src_buffer_addr);
    uint64_t dst_noc_addr = get_noc_addr(dst_buffer_addr);
    uint64_t pattern_noc_addr = get_noc_addr(pattern_buffer_addr);
    
    // L1 buffer layout:
    // 0x10000: pattern data tile (2048 bytes)
    // 0x10800: source data tile (dense) (2048 bytes)
    // 0x11000: destination data tile (sparse) (2048 bytes)
    uint32_t pattern_l1_addr = l1_buffer_base;
    uint32_t src_l1_addr = l1_buffer_base + tile_size_bytes;
    uint32_t dst_l1_addr = l1_buffer_base + 2 * tile_size_bytes;
    
    // Process this core's portion of the work
    // Each core processes work_per_core elements starting from work_offset
    uint32_t start_element = work_offset;
    uint32_t end_element = work_offset + work_per_core;
    
    // Calculate tile range for this core's work
    uint32_t start_tile = start_element / elements_per_tile;
    uint32_t end_tile = (end_element + elements_per_tile - 1) / elements_per_tile;
    
    for (uint32_t tile_idx = start_tile; tile_idx < end_tile; tile_idx++) {
        // Calculate element range within this tile for this core
        uint32_t tile_start_elem = tile_idx * elements_per_tile;
        uint32_t tile_end_elem = (tile_idx + 1) * elements_per_tile;
        
        // Intersect tile range with this core's work range
        uint32_t work_start_in_tile = (start_element > tile_start_elem) ? (start_element - tile_start_elem) : 0;
        uint32_t work_end_in_tile = (end_element < tile_end_elem) ? (end_element - tile_start_elem) : elements_per_tile;
        
        if (work_start_in_tile >= work_end_in_tile) continue; // No work in this tile
        
        // Read pattern tile for this chunk
        uint64_t pattern_src_addr = pattern_noc_addr + tile_idx * tile_size_bytes;
        noc_async_read(pattern_src_addr, pattern_l1_addr, tile_size_bytes);
        noc_async_read_barrier();
        
        // Read source tile (dense data to scatter)
        uint64_t src_tile_addr = src_noc_addr + tile_idx * tile_size_bytes;
        noc_async_read(src_tile_addr, src_l1_addr, tile_size_bytes);
        noc_async_read_barrier();
        
        // Cast L1 buffers to appropriate types
        uint32_t* pattern_data = reinterpret_cast<uint32_t*>(pattern_l1_addr);
        uint16_t* src_data = reinterpret_cast<uint16_t*>(src_l1_addr); // BFloat16
        
        // Process each element in this core's work range within the tile
        for (uint32_t elem_idx = work_start_in_tile; elem_idx < work_end_in_tile; elem_idx++) {
            uint32_t global_elem_idx = tile_start_elem + elem_idx;
            uint32_t pattern_index = pattern_data[elem_idx];
            uint32_t dst_index = pattern_index + delta * (global_elem_idx / elements_per_tile);
            
            // Calculate which destination tile we need to update
            uint32_t dst_tile_idx = dst_index / elements_per_tile;
            uint32_t dst_elem_offset = dst_index % elements_per_tile;
            
            // Read destination tile (sparse buffer)
            uint64_t dst_tile_addr = dst_noc_addr + dst_tile_idx * tile_size_bytes;
            noc_async_read(dst_tile_addr, dst_l1_addr, tile_size_bytes);
            noc_async_read_barrier();
            
            uint16_t* dst_data = reinterpret_cast<uint16_t*>(dst_l1_addr);
            
            // Perform scatter operation: sparse[pattern[j] + delta * i] = dense[j]
            dst_data[dst_elem_offset] = src_data[elem_idx];
            
            // Write back the updated destination tile
            noc_async_write(dst_tile_addr, dst_l1_addr, tile_size_bytes);
            noc_async_write_barrier();
        }
    }
    
    // Debug print to show kernel completion
    DPRINT << "SCATTER KERNEL: Completed processing" << ENDL();
    
    // Invalidate L1 cache for Blackhole architecture
    invalidate_l1_cache();
}