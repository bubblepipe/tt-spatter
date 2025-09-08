// SPDX-FileCopyrightText: 2025 The Spatter Authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    // Runtime arguments
    uint32_t sparse_addr = get_arg_val<uint32_t>(0);
    uint32_t dense_addr = get_arg_val<uint32_t>(1);
    uint32_t pattern_addr = get_arg_val<uint32_t>(2);
    uint32_t pattern_scatter_addr = get_arg_val<uint32_t>(3);
    uint32_t start_element = get_arg_val<uint32_t>(4);
    uint32_t end_element = get_arg_val<uint32_t>(5);
    uint32_t pattern_length = get_arg_val<uint32_t>(6);
    uint32_t delta = get_arg_val<uint32_t>(7);
    uint32_t count = get_arg_val<uint32_t>(8);
    uint32_t wrap = get_arg_val<uint32_t>(9);
    uint32_t sparse_size_elements = get_arg_val<uint32_t>(10);
    
    // Compile-time buffer indices
    constexpr uint32_t cb_pattern = get_compile_time_arg_val(0);
    constexpr uint32_t cb_pattern_scatter = get_compile_time_arg_val(1);
    constexpr uint32_t cb_sparse = get_compile_time_arg_val(2);
    constexpr uint32_t cb_dense = get_compile_time_arg_val(3);
    
    constexpr uint32_t tile_size_bytes = 2048;  // 32x32 BFloat16 elements
    constexpr uint32_t elements_per_tile = 1024;  // 32x32
    
    // L1 buffer addresses for caching
    uint32_t pattern_l1_addr = get_write_ptr(cb_pattern);
    uint32_t pattern_scatter_l1_addr = get_write_ptr(cb_pattern_scatter);
    uint32_t sparse_l1_addr = get_write_ptr(cb_sparse);
    uint32_t dense_l1_addr = get_write_ptr(cb_dense);
    
    // Cache for tiles to minimize DRAM reads
    uint32_t cached_pattern_tile_id = UINT32_MAX;
    uint32_t cached_pattern_scatter_tile_id = UINT32_MAX;
    uint32_t cached_sparse_tile_id = UINT32_MAX;
    uint32_t cached_dense_tile_id = UINT32_MAX;
    
    DPRINT << "Multi-scatter kernel: processing elements " << start_element << " to " << end_element << ENDL();
    DPRINT << "Pattern length: " << pattern_length << ", delta: " << delta << ", count: " << count << ", wrap: " << wrap << ENDL();
    
    // Process elements assigned to this core
    for (uint32_t elem_idx = start_element; elem_idx < end_element; elem_idx++) {
        uint32_t j = elem_idx % pattern_length;
        uint32_t i = elem_idx / pattern_length;
        
        // Load pattern_scatter tile if needed
        uint32_t pattern_scatter_tile_id = j / elements_per_tile;
        if (pattern_scatter_tile_id != cached_pattern_scatter_tile_id) {
            uint32_t pattern_scatter_tile_addr = pattern_scatter_addr + pattern_scatter_tile_id * tile_size_bytes;
            noc_async_read(get_noc_addr(pattern_scatter_tile_addr), pattern_scatter_l1_addr, tile_size_bytes);
            noc_async_read_barrier();
            cached_pattern_scatter_tile_id = pattern_scatter_tile_id;
        }
        
        // Get pattern_scatter[j] value
        uint32_t* pattern_scatter_data = reinterpret_cast<uint32_t*>(pattern_scatter_l1_addr);
        uint32_t pattern_scatter_idx = pattern_scatter_data[j % elements_per_tile];
        
        // Load pattern tile if needed
        uint32_t pattern_tile_id = pattern_scatter_idx / elements_per_tile;
        if (pattern_tile_id != cached_pattern_tile_id) {
            uint32_t pattern_tile_addr = pattern_addr + pattern_tile_id * tile_size_bytes;
            noc_async_read(get_noc_addr(pattern_tile_addr), pattern_l1_addr, tile_size_bytes);
            noc_async_read_barrier();
            cached_pattern_tile_id = pattern_tile_id;
        }
        
        // Get pattern[pattern_scatter[j]] value - double indirection
        uint32_t* pattern_data = reinterpret_cast<uint32_t*>(pattern_l1_addr);
        uint32_t pattern_idx = pattern_data[pattern_scatter_idx % elements_per_tile];
        
        // Calculate destination index in sparse array
        uint32_t dst_index = pattern_idx + (delta * i);
        if (dst_index >= sparse_size_elements) {
            dst_index = dst_index % sparse_size_elements;
        }
        
        // Calculate source index in dense array
        uint32_t src_index = j + pattern_length * (i % wrap);
        
        // Load dense tile if needed
        uint32_t dense_tile_id = src_index / elements_per_tile;
        if (dense_tile_id != cached_dense_tile_id) {
            uint32_t dense_tile_addr = dense_addr + dense_tile_id * tile_size_bytes;
            noc_async_read(get_noc_addr(dense_tile_addr), dense_l1_addr, tile_size_bytes);
            noc_async_read_barrier();
            cached_dense_tile_id = dense_tile_id;
        }
        
        // Read value from dense array (using uint16_t for BFloat16)
        uint16_t* dense_data = reinterpret_cast<uint16_t*>(dense_l1_addr);
        uint16_t src_value = dense_data[src_index % elements_per_tile];
        
        // Load sparse tile if different from cached
        uint32_t sparse_tile_id = dst_index / elements_per_tile;
        if (sparse_tile_id != cached_sparse_tile_id) {
            // Write back previous sparse tile if we had one
            if (cached_sparse_tile_id != UINT32_MAX) {
                uint32_t prev_sparse_tile_addr = sparse_addr + cached_sparse_tile_id * tile_size_bytes;
                noc_async_write(sparse_l1_addr, get_noc_addr(prev_sparse_tile_addr), tile_size_bytes);
                noc_async_write_barrier();
            }
            
            uint32_t sparse_tile_addr = sparse_addr + sparse_tile_id * tile_size_bytes;
            noc_async_read(get_noc_addr(sparse_tile_addr), sparse_l1_addr, tile_size_bytes);
            noc_async_read_barrier();
            cached_sparse_tile_id = sparse_tile_id;
        }
        
        // Write value to sparse array (using uint16_t for BFloat16)
        uint16_t* sparse_data = reinterpret_cast<uint16_t*>(sparse_l1_addr);
        sparse_data[dst_index % elements_per_tile] = src_value;
    }
    
    // Write back the last sparse tile if we modified any
    if (cached_sparse_tile_id != UINT32_MAX) {
        uint32_t sparse_tile_addr = sparse_addr + cached_sparse_tile_id * tile_size_bytes;
        noc_async_write(sparse_l1_addr, get_noc_addr(sparse_tile_addr), tile_size_bytes);
        noc_async_write_barrier();
    }
    
    DPRINT << "Multi-scatter kernel complete for core" << ENDL();
}