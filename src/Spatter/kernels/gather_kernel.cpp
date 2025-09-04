// SPDX-FileCopyrightText: © 2025 Spatter Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

/*
 * Single Gather Kernel for Spatter TensTorrent Backend
 * 
 * Implements: dense[j] = sparse[pattern[j] + delta * i]
 * 
 * This is a pure data movement kernel following the loopback example pattern.
 * No computation kernels or circular buffers needed - just direct DRAM operations.
 * 
 * Runtime Args:
 * - arg0: l1_buffer_addr - L1 temporary buffer address
 * - arg1: num_elements - Number of elements to process (pattern_length * count)
 * - arg2: delta - Stride parameter for iterations
 * - arg3: pattern_length - Length of pattern array (for reuse)
 * 
 * Compile-time Args (via TensorAccessorArgs):
 * - Sparse buffer configuration
 * - Dense buffer configuration
 * - Pattern buffer configuration
 */

void kernel_main() {
    // Read parameters from kernel arguments
    uint32_t l1_buffer_addr = get_arg_val<uint32_t>(0);
    uint32_t num_elements = get_arg_val<uint32_t>(1);
    uint32_t delta = get_arg_val<uint32_t>(2);
    uint32_t pattern_length = get_arg_val<uint32_t>(3);
    
    uint32_t sparse_addr = get_arg_val<uint32_t>(4);
    uint32_t dense_addr = get_arg_val<uint32_t>(5);
    uint32_t pattern_addr = get_arg_val<uint32_t>(6);

    // Each tile is 32x32 elements of bfloat16, which is 2 bytes per element.
    // So the tile size in bytes is 32 * 32 * 2 = 2048 bytes.
    const uint32_t tile_size_bytes = 32 * 32 * 2;
    const uint32_t elements_per_tile = 32 * 32;

    // Create TensorAccessors using compile-time TensorAccessorArgs + runtime addresses
    // Following the exact pattern from eltwise_binary/kernels/dataflow/read_tiles.cpp
    constexpr auto sparse_args = TensorAccessorArgs<0>();
    const auto sparse_accessor = TensorAccessor(sparse_args, sparse_addr, tile_size_bytes);

    constexpr auto dense_args = TensorAccessorArgs<sparse_args.next_compile_time_args_offset()>();
    const auto dense_accessor = TensorAccessor(dense_args, dense_addr, tile_size_bytes);

    constexpr auto pattern_args = TensorAccessorArgs<dense_args.next_compile_time_args_offset()>();
    const auto pattern_accessor = TensorAccessor(pattern_args, pattern_addr, tile_size_bytes);

    // Calculate number of tiles to process
    uint32_t num_tiles = (num_elements + elements_per_tile - 1) / elements_per_tile;

    // L1 buffer layout (like loopback example):
    // l1_buffer_addr: pattern data tile
    // l1_buffer_addr + tile_size_bytes: sparse data tile
    // l1_buffer_addr + 2*tile_size_bytes: dense data tile (output)
    uint32_t pattern_l1_addr = l1_buffer_addr;
    uint32_t sparse_l1_addr = l1_buffer_addr + tile_size_bytes;
    uint32_t dense_l1_addr = l1_buffer_addr + 2 * tile_size_bytes;

    invalidate_l1_cache();
    noc_async_read_tile(0, pattern_accessor, pattern_l1_addr);
    noc_async_read_barrier();
    
    // Since dense buffer is smaller (pattern_length elements), we only need one tile for it
    // Read the single dense tile
    noc_async_read_tile(0, dense_accessor, dense_l1_addr);
    noc_async_read_barrier();
    
    uint32_t* pattern_data = reinterpret_cast<uint32_t*>(pattern_l1_addr);
    uint16_t* dense_data = reinterpret_cast<uint16_t*>(dense_l1_addr);
    
    // Process all elements, writing sequentially to dense buffer
    for (uint32_t i = 0; i < num_elements; i++) {
        // Calculate which pattern element to use (pattern reuse)
        uint32_t pattern_idx = i % pattern_length;
        uint32_t pattern_value = pattern_data[pattern_idx];
        
        // Calculate iteration for delta stride
        uint32_t iteration = i / pattern_length;
        uint32_t src_index = pattern_value + delta * iteration;
        
        // Calculate source tile and offset
        uint32_t src_tile = src_index / elements_per_tile;
        uint32_t src_offset = src_index % elements_per_tile;
        
        invalidate_l1_cache();
        noc_async_read_tile(src_tile, sparse_accessor, sparse_l1_addr);
        noc_async_read_barrier();
        
        uint16_t* sparse_data = reinterpret_cast<uint16_t*>(sparse_l1_addr);
        // FIX: Write to sequential position i, not pattern_idx
        dense_data[i] = sparse_data[src_offset];
    }
    
    // Write the single dense tile back to DRAM
    noc_async_write_tile(0, dense_accessor, dense_l1_addr);
    noc_async_write_barrier();
    
    // Ensure all writes are flushed to DRAM
    // This is critical for proper synchronization with host reads
    noc_async_writes_flushed();
}