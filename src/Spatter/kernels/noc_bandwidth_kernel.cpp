// SPDX-FileCopyrightText: © 2025 Spatter Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

/*
 * NOC Bandwidth Saturation Kernel
 * 
 * This kernel saturates NOC bandwidth by continuously transferring tiles
 * between neighboring Tensix cores to measure peak NOC performance.
 * 
 * Runtime Args:
 * - arg0: source_addr - Source buffer address in L1 
 * - arg1: dest_addr - Destination buffer address on neighbor core
 * - arg2: num_tiles - Number of tiles to transfer
 * - arg3: neighbor_noc_x - Neighbor core NOC X coordinate
 * - arg4: neighbor_noc_y - Neighbor core NOC Y coordinate
 */

void kernel_main() {
    // Get runtime arguments
    uint32_t source_addr = get_arg_val<uint32_t>(0);
    uint32_t dest_addr = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t neighbor_noc_x = get_arg_val<uint32_t>(3);
    uint32_t neighbor_noc_y = get_arg_val<uint32_t>(4);
    
    constexpr uint32_t tile_size_bytes = 32 * 32 * 2; // 32x32 BFloat16 = 2048 bytes
    constexpr uint32_t l1_buffer_addr = 0x10000; // L1 staging area
    
    // Calculate neighbor NOC coordinates
    uint64_t neighbor_noc_addr = get_noc_addr(neighbor_noc_x, neighbor_noc_y, dest_addr);
    
    // Bandwidth saturation loop - stream tiles as fast as possible
    for (uint32_t tile = 0; tile < num_tiles; tile++) {
        uint32_t current_src_addr = source_addr + (tile * tile_size_bytes);
        uint32_t current_dest_addr = dest_addr + (tile * tile_size_bytes);
        
        // Use low-level NOC read/write functions
        // Read tile from local L1 to staging buffer
        noc_async_read(current_src_addr, l1_buffer_addr, tile_size_bytes);
        noc_async_read_barrier();
        
        // Write tile directly to neighbor's L1 using NOC coordinates
        uint64_t neighbor_dest_noc_addr = get_noc_addr(neighbor_noc_x, neighbor_noc_y, current_dest_addr);
        noc_async_write(l1_buffer_addr, neighbor_dest_noc_addr, tile_size_bytes);
        noc_async_write_barrier();
    }
    
    // Alternative: Bulk transfer approach for maximum bandwidth
    // Transfer multiple tiles without intermediate barriers (pipeline)
    /*
    constexpr uint32_t pipeline_depth = 4;
    
    // Fill pipeline
    for (uint32_t i = 0; i < pipeline_depth && i < num_tiles; i++) {
        uint32_t src_addr = source_addr + (i * tile_size_bytes);
        uint32_t staging_addr = l1_buffer_addr + (i % pipeline_depth) * tile_size_bytes;
        noc_async_read_tile(src_addr, staging_addr);
    }
    
    // Process tiles in pipeline
    for (uint32_t tile = 0; tile < num_tiles; tile++) {
        uint32_t buffer_idx = tile % pipeline_depth;
        uint32_t staging_addr = l1_buffer_addr + buffer_idx * tile_size_bytes;
        uint32_t dest_offset = dest_addr + (tile * tile_size_bytes);
        
        // Wait for read to complete
        noc_async_read_barrier();
        
        // Start next read (if more tiles)
        if (tile + pipeline_depth < num_tiles) {
            uint32_t next_src = source_addr + ((tile + pipeline_depth) * tile_size_bytes);
            noc_async_read_tile(next_src, staging_addr);
        }
        
        // Write current tile to neighbor
        uint64_t neighbor_dest = get_noc_addr(neighbor_noc_x, neighbor_noc_y, dest_offset);
        noc_async_write_tile(staging_addr, neighbor_dest);
        noc_async_write_barrier();
    }
    */
    
    // Invalidate L1 cache for Blackhole
    invalidate_l1_cache();
}