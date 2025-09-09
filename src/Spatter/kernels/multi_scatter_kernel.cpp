// SPDX-FileCopyrightText: 2025 The Spatter Authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"

/*
  * Multi-Scatter Kernel for Spatter TensTorrent Backend (Multi-Core)
 * 
 * Implements: sparse[pattern[pattern_scatter[j]] + delta * i] = dense[j + pattern_length * (i % wrap)]
 * 
 * This is a double indirection scatter:
 * 1. pattern_scatter[j] gives an index into the pattern array
 * 2. pattern[pattern_scatter[j]] gives the actual sparse array index
 * 
 * Runtime Args:
 * - arg0: pattern_l1_addr - Pattern L1 buffer address
 * - arg1: pattern_scatter_l1_addr - Pattern scatter L1 buffer address (first indirection)
 * - arg2: sparse_l1_addr - Sparse L1 buffer address (destination)
 * - arg3: dense_l1_addr - Dense L1 buffer address (source)
 * - arg4: start_element - Starting element index for this core
 * - arg5: end_element - Ending element index for this core
 * - arg6: pattern_length - Length of the pattern arrays
 * - arg7: delta - Stride for sparse access
 * - arg8: count - Number of pattern iterations
 * - arg9: wrap - Wrap parameter for dense buffer indexing
 * - arg10: sparse_size_elements - Total size of sparse buffer in elements
 * - arg11: pattern_addr - Pattern buffer address (DRAM)
 * - arg12: pattern_scatter_addr - Pattern scatter buffer address (DRAM)
 * - arg13: sparse_addr - Sparse buffer address (DRAM)
 * - arg14: dense_addr - Dense buffer address (DRAM)
 */

void kernel_main() {
    // Read runtime arguments - four separate L1 buffers
    uint32_t pattern_l1_addr = get_arg_val<uint32_t>(0);
    uint32_t pattern_scatter_l1_addr = get_arg_val<uint32_t>(1);
    uint32_t sparse_l1_addr = get_arg_val<uint32_t>(2);
    uint32_t dense_l1_addr = get_arg_val<uint32_t>(3);
    uint32_t start_element = get_arg_val<uint32_t>(4);
    uint32_t end_element = get_arg_val<uint32_t>(5);
    uint32_t pattern_length = get_arg_val<uint32_t>(6);
    uint32_t delta = get_arg_val<uint32_t>(7);
    uint32_t count = get_arg_val<uint32_t>(8);
    uint32_t wrap = get_arg_val<uint32_t>(9);
    uint32_t sparse_size_elements = get_arg_val<uint32_t>(10);
    uint32_t pattern_addr = get_arg_val<uint32_t>(11);
    uint32_t pattern_scatter_addr = get_arg_val<uint32_t>(12);
    uint32_t sparse_addr = get_arg_val<uint32_t>(13);
    uint32_t dense_addr = get_arg_val<uint32_t>(14);

    // Early exit if no work to do
    if (start_element >= end_element) {
        return;
    }

    // Tile constants
    const uint32_t tile_size_bytes = 32 * 32 * 2;  // 2048 bytes per tile
    const uint32_t elements_per_tile = 32 * 32;    // 1024 elements per tile

    // Create TensorAccessors for all four buffers
    constexpr auto pattern_args = TensorAccessorArgs<0>();
    const auto pattern_accessor = TensorAccessor(pattern_args, pattern_addr, tile_size_bytes);
    
    constexpr auto pattern_scatter_args = TensorAccessorArgs<pattern_args.next_compile_time_args_offset()>();
    const auto pattern_scatter_accessor = TensorAccessor(pattern_scatter_args, pattern_scatter_addr, tile_size_bytes);
    
    constexpr auto sparse_args = TensorAccessorArgs<pattern_scatter_args.next_compile_time_args_offset()>();
    const auto sparse_accessor = TensorAccessor(sparse_args, sparse_addr, tile_size_bytes);
    
    constexpr auto dense_args = TensorAccessorArgs<sparse_args.next_compile_time_args_offset()>();
    const auto dense_accessor = TensorAccessor(dense_args, dense_addr, tile_size_bytes);

    // Track cached tiles to avoid redundant loads
    uint32_t cached_pattern_scatter_tile = UINT32_MAX;
    uint32_t cached_pattern_tile = UINT32_MAX;
    uint32_t cached_sparse_tile = UINT32_MAX;
    uint32_t cached_dense_tile = UINT32_MAX;

    // Process elements assigned to this core
    for (uint32_t elem_idx = start_element; elem_idx < end_element; elem_idx++) {
        // Calculate j and i for the current element
        uint32_t j = elem_idx % count;
        uint32_t i = elem_idx / count;
        
        // Calculate pattern index with wrap
        uint32_t pattern_idx = j % pattern_length;
        
        // Step 1: Calculate dense index with wrap and load dense tile if needed
        uint32_t dense_idx = j + pattern_length * (i % wrap);
        uint32_t dense_tile_id = dense_idx / elements_per_tile;
        
        if (dense_tile_id != cached_dense_tile) {
            noc_async_read_tile(dense_tile_id, dense_accessor, dense_l1_addr);
            noc_async_read_barrier();
            cached_dense_tile = dense_tile_id;
        }
        
        // Step 2: Read value from dense buffer
        uint16_t* dense_data = reinterpret_cast<uint16_t*>(dense_l1_addr);
        uint32_t dense_elem_offset = dense_idx % elements_per_tile;
        uint16_t value = dense_data[dense_elem_offset];
        
        // Step 3: Load pattern_scatter tile if needed
        uint32_t pattern_scatter_tile_id = pattern_idx / elements_per_tile;
        if (pattern_scatter_tile_id != cached_pattern_scatter_tile) {
            noc_async_read_tile(pattern_scatter_tile_id, pattern_scatter_accessor, pattern_scatter_l1_addr);
            noc_async_read_barrier();
            cached_pattern_scatter_tile = pattern_scatter_tile_id;
        }
        
        // Step 4: Get first indirection - pattern_scatter[j] gives index into pattern array
        uint32_t* pattern_scatter_data = reinterpret_cast<uint32_t*>(pattern_scatter_l1_addr);
        uint32_t pattern_scatter_elem_offset = pattern_idx % elements_per_tile;
        uint32_t first_indirection_idx = pattern_scatter_data[pattern_scatter_elem_offset];
        
        // Bounds check for first indirection
        first_indirection_idx = first_indirection_idx % pattern_length;
        
        // Step 5: Load pattern tile containing pattern[first_indirection_idx]
        uint32_t pattern_tile_id = first_indirection_idx / elements_per_tile;
        if (pattern_tile_id != cached_pattern_tile) {
            noc_async_read_tile(pattern_tile_id, pattern_accessor, pattern_l1_addr);
            noc_async_read_barrier();
            cached_pattern_tile = pattern_tile_id;
        }
        
        // Step 6: Get second indirection - pattern[pattern_scatter[j]]
        uint32_t* pattern_data = reinterpret_cast<uint32_t*>(pattern_l1_addr);
        uint32_t pattern_elem_offset = first_indirection_idx % elements_per_tile;
        uint32_t sparse_base_idx = pattern_data[pattern_elem_offset];
        
        // Step 7: Calculate final sparse index with delta
        uint32_t sparse_idx = (sparse_base_idx + delta * i) % sparse_size_elements;
        
        // Step 8: Load sparse tile if needed (for read-modify-write)
        uint32_t sparse_tile_id = sparse_idx / elements_per_tile;
        if (sparse_tile_id != cached_sparse_tile) {
            // Write back previous sparse tile if modified
            if (cached_sparse_tile != UINT32_MAX) {
                noc_async_write_tile(cached_sparse_tile, sparse_accessor, sparse_l1_addr);
                noc_async_write_barrier();
            }
            
            // Load new sparse tile for read-modify-write
            noc_async_read_tile(sparse_tile_id, sparse_accessor, sparse_l1_addr);
            noc_async_read_barrier();
            cached_sparse_tile = sparse_tile_id;
        }
        
        // Step 9: Write value to sparse buffer
        uint16_t* sparse_data = reinterpret_cast<uint16_t*>(sparse_l1_addr);
        uint32_t sparse_elem_offset = sparse_idx % elements_per_tile;
        sparse_data[sparse_elem_offset] = value;
    }
    
    // Write back the last modified sparse tile
    if (cached_sparse_tile != UINT32_MAX) {
        noc_async_write_tile(cached_sparse_tile, sparse_accessor, sparse_l1_addr);
        noc_async_write_barrier();
    }
    
    noc_async_writes_flushed();
}