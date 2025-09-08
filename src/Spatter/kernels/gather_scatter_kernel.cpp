// SPDX-FileCopyrightText: © 2025 Spatter Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"  // required in all kernels using DPRINT

/*
 * Gather-Scatter Kernel for Spatter TensTorrent Backend (Multi-Core)
 * 
 * Implements: sparse_scatter[pattern_scatter[j] + delta_scatter * i] = 
 *             sparse_gather[pattern_gather[j] + delta_gather * i]
 * 
 * Runtime Args:
 * - arg0: pattern_gather_l1_addr - Pattern gather L1 buffer address
 * - arg1: pattern_scatter_l1_addr - Pattern scatter L1 buffer address
 * - arg2: sparse_gather_l1_addr - Sparse gather L1 buffer address (source)
 * - arg3: sparse_scatter_l1_addr - Sparse scatter L1 buffer address (destination)
 * - arg4: start_element - Starting element index for this core
 * - arg5: num_elements_per_core - Number of elements this core should process
 * - arg6: delta_gather - Stride for gather pattern iterations
 * - arg7: delta_scatter - Stride for scatter pattern iterations
 * - arg8: pattern_length - Length of the pattern arrays
 * - arg9: sparse_gather_addr - Source buffer address (DRAM)
 * - arg10: sparse_scatter_addr - Destination buffer address (DRAM)
 * - arg11: pattern_gather_addr - Pattern gather buffer address (DRAM)
 * - arg12: pattern_scatter_addr - Pattern scatter buffer address (DRAM)
 */

void kernel_main() {
    // Read runtime arguments - four separate L1 buffers
    uint32_t pattern_gather_l1_addr = get_arg_val<uint32_t>(0);
    uint32_t pattern_scatter_l1_addr = get_arg_val<uint32_t>(1);
    uint32_t sparse_gather_l1_addr = get_arg_val<uint32_t>(2);
    uint32_t sparse_scatter_l1_addr = get_arg_val<uint32_t>(3);
    uint32_t start_element = get_arg_val<uint32_t>(4);
    uint32_t num_elements_per_core = get_arg_val<uint32_t>(5);
    uint32_t delta_gather = get_arg_val<uint32_t>(6);
    uint32_t delta_scatter = get_arg_val<uint32_t>(7);
    uint32_t pattern_length = get_arg_val<uint32_t>(8);
    uint32_t sparse_gather_addr = get_arg_val<uint32_t>(9);
    uint32_t sparse_scatter_addr = get_arg_val<uint32_t>(10);
    uint32_t pattern_gather_addr = get_arg_val<uint32_t>(11);
    uint32_t pattern_scatter_addr = get_arg_val<uint32_t>(12);

    DPRINT_MATH(DPRINT << "Gather-Scatter kernel: MATH core" << ENDL());
    DPRINT_UNPACK(DPRINT << "Gather-Scatter kernel: UNPACK core" << ENDL());
    DPRINT_PACK(DPRINT << "Gather-Scatter kernel: PACK core" << ENDL());
    
    // Tile constants
    const uint32_t tile_size_bytes = 32 * 32 * 2;  // 2048 bytes per tile
    const uint32_t elements_per_tile = 32 * 32;    // 1024 elements per tile

    // Create TensorAccessors for all four buffers
    constexpr auto sparse_gather_args = TensorAccessorArgs<0>();
    const auto sparse_gather_accessor = TensorAccessor(sparse_gather_args, sparse_gather_addr, tile_size_bytes);
    
    constexpr auto sparse_scatter_args = TensorAccessorArgs<sparse_gather_args.next_compile_time_args_offset()>();
    const auto sparse_scatter_accessor = TensorAccessor(sparse_scatter_args, sparse_scatter_addr, tile_size_bytes);
    
    constexpr auto pattern_gather_args = TensorAccessorArgs<sparse_scatter_args.next_compile_time_args_offset()>();
    const auto pattern_gather_accessor = TensorAccessor(pattern_gather_args, pattern_gather_addr, tile_size_bytes);
    
    constexpr auto pattern_scatter_args = TensorAccessorArgs<pattern_gather_args.next_compile_time_args_offset()>();
    const auto pattern_scatter_accessor = TensorAccessor(pattern_scatter_args, pattern_scatter_addr, tile_size_bytes);
    
    // Load pattern tiles once (they will be reused)
    // Patterns are stored as uint32_t values
    noc_async_read_tile(0, pattern_gather_accessor, pattern_gather_l1_addr);
    noc_async_read_tile(0, pattern_scatter_accessor, pattern_scatter_l1_addr);
    noc_async_read_barrier();
    
    // Get pointers to pattern data in L1
    uint32_t* pattern_gather_data = reinterpret_cast<uint32_t*>(pattern_gather_l1_addr);
    uint32_t* pattern_scatter_data = reinterpret_cast<uint32_t*>(pattern_scatter_l1_addr);

    // Calculate the end element for this core
    uint32_t end_element = start_element + num_elements_per_core;
    
    // Track the last loaded tiles to avoid redundant loads
    uint32_t last_gather_tile = UINT32_MAX;
    uint32_t last_scatter_tile = UINT32_MAX;
    
    // Process elements assigned to this core
    for (uint32_t elem_idx = start_element; elem_idx < end_element; elem_idx++) {
        // Calculate pattern index and iteration
        uint32_t pattern_idx = elem_idx % pattern_length;
        uint32_t iteration = elem_idx / pattern_length;
        
        // Get the source index from pattern_gather and add delta_gather offset
        uint32_t src_index = pattern_gather_data[pattern_idx] + (delta_gather * iteration);
        
        // Get the destination index from pattern_scatter and add delta_scatter offset
        uint32_t dst_index = pattern_scatter_data[pattern_idx] + (delta_scatter * iteration);
        
        // Determine which sparse_gather tile contains the source element
        uint32_t src_tile_idx = src_index / elements_per_tile;
        uint32_t src_elem_offset = src_index % elements_per_tile;
        
        // Load the sparse_gather tile if not already loaded
        if (src_tile_idx != last_gather_tile) {
            noc_async_read_tile(src_tile_idx, sparse_gather_accessor, sparse_gather_l1_addr);
            noc_async_read_barrier();
            last_gather_tile = src_tile_idx;
        }
        
        // Get the source value
        uint16_t* sparse_gather_data = reinterpret_cast<uint16_t*>(sparse_gather_l1_addr);
        uint16_t src_value = sparse_gather_data[src_elem_offset];
        
        // Determine which sparse_scatter tile contains the destination
        uint32_t dst_tile_idx = dst_index / elements_per_tile;
        uint32_t dst_elem_offset = dst_index % elements_per_tile;
        
        // Load the sparse_scatter tile if not already loaded (for read-modify-write)
        if (dst_tile_idx != last_scatter_tile) {
            // If we have a previously modified scatter tile, write it back
            if (last_scatter_tile != UINT32_MAX) {
                noc_async_write_tile(last_scatter_tile, sparse_scatter_accessor, sparse_scatter_l1_addr);
                noc_async_write_barrier();
            }
            
            // Load the new sparse_scatter tile for read-modify-write
            noc_async_read_tile(dst_tile_idx, sparse_scatter_accessor, sparse_scatter_l1_addr);
            noc_async_read_barrier();
            last_scatter_tile = dst_tile_idx;
        }
        
        // Copy the element from sparse_gather to sparse_scatter
        uint16_t* sparse_scatter_data = reinterpret_cast<uint16_t*>(sparse_scatter_l1_addr);
        sparse_scatter_data[dst_elem_offset] = src_value;
    }
    
    // Write back the last modified scatter tile
    if (last_scatter_tile != UINT32_MAX) {
        noc_async_write_tile(last_scatter_tile, sparse_scatter_accessor, sparse_scatter_l1_addr);
        noc_async_write_barrier();
    }
    
    noc_async_writes_flushed();
}