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
 * - arg0: l1_buffer_addr - L1 temporary buffaer address
 * - arg1: sparse_buffer_addr - Source (sparse) buffer address in DRAM
 * - arg2: dense_buffer_addr - Destination (dense) buffer address in DRAM
 * - arg3: pattern_buffer_addr - Pattern array buffer address in DRAM
 * - arg4: num_elements - Number of coelements to process (pattern_length * count)
 * - arg5: delta - Stride parameter for iterations
 * - arg6: pattern_length - Length of pattern array (for reuse)
 */

void kernel_main() {
    // Read parameters from kernel arguments (following loopback pattern)
    uint32_t l1_buffer_addr = get_arg_val<uint32_t>(0);
    uint32_t sparse_buffer_addr = get_arg_val<uint32_t>(1);
    uint32_t dense_buffer_addr = get_arg_val<uint32_t>(2);
    uint32_t pattern_buffer_addr = get_arg_val<uint32_t>(3);
    uint32_t num_elements = get_arg_val<uint32_t>(4);
    uint32_t delta = get_arg_val<uint32_t>(5);
    uint32_t pattern_length = get_arg_val<uint32_t>(6);

    // Each tile is 32x32 elements of bfloat16, which is 2 bytes per element.
    // So the tile size in bytes is 32 * 32 * 2 = 2048 bytes.
    const uint32_t tile_size_bytes = 32 * 32 * 2;
    const uint32_t elements_per_tile = 32 * 32;

    // Create TensorAccessors for all buffers (following loopback pattern exactly)
    constexpr auto sparse_args = TensorAccessorArgs<0>();
    const auto sparse_accessor = TensorAccessor(sparse_args, sparse_buffer_addr, tile_size_bytes);

    constexpr auto dense_args = TensorAccessorArgs<sparse_args.next_compile_time_args_offset()>();
    const auto dense_accessor = TensorAccessor(dense_args, dense_buffer_addr, tile_size_bytes);

    constexpr auto pattern_args = TensorAccessorArgs<dense_args.next_compile_time_args_offset()>();
    const auto pattern_accessor = TensorAccessor(pattern_args, pattern_buffer_addr, tile_size_bytes); // Pattern buffer is also tile-aligned

    // Calculate number of tiles to process
    uint32_t num_tiles = (num_elements + elements_per_tile - 1) / elements_per_tile;

    // L1 buffer layout (like loopback example):
    // l1_buffer_addr: pattern data tile
    // l1_buffer_addr + tile_size_bytes: sparse data tile
    // l1_buffer_addr + 2*tile_size_bytes: dense data tile (output)
    uint32_t pattern_l1_addr = l1_buffer_addr;
    uint32_t sparse_l1_addr = l1_buffer_addr + tile_size_bytes;
    uint32_t dense_l1_addr = l1_buffer_addr + 2 * tile_size_bytes;

    // Read pattern tile once (pattern reuse means we only need the first tile)
    // Pattern should fit in one tile for typical Spatter patterns
    noc_async_read_tile(0, pattern_accessor, pattern_l1_addr);
    noc_async_read_barrier();
    
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        // Read current dense tile (for partial updates)
        noc_async_read_tile(tile_idx, dense_accessor, dense_l1_addr);
        noc_async_read_barrier();

        // Cast L1 buffers to appropriate types
        uint32_t* pattern_data = reinterpret_cast<uint32_t*>(pattern_l1_addr);
        uint16_t* dense_data = reinterpret_cast<uint16_t*>(dense_l1_addr); // BFloat16

        // Calculate element range for this tile
        uint32_t tile_start_elem = tile_idx * elements_per_tile;
        uint32_t tile_end_elem = (tile_idx + 1) * elements_per_tile;
        if (tile_end_elem > num_elements) {
            tile_end_elem = num_elements;
        }

        // Process each element in the tile
        for (uint32_t elem_idx = 0; elem_idx < (tile_end_elem - tile_start_elem); elem_idx++) {
            uint32_t global_elem_idx = tile_start_elem + elem_idx;
            
            // Implement pattern reuse: pattern[j % pattern_length]
            // This matches Spatter's gather logic: dense[j] = sparse[pattern[j % pattern_length] + delta * (j / pattern_length)]
            uint32_t pattern_offset = global_elem_idx % pattern_length;
            uint32_t pattern_index = pattern_data[pattern_offset];
            
            // Calculate iteration number for delta stride
            uint32_t iteration = global_elem_idx / pattern_length;
            uint32_t src_index = pattern_index + delta * iteration;
            
            // Calculate which sparse tile we need to read
            uint32_t src_tile_idx = src_index / elements_per_tile;
            uint32_t src_elem_offset = src_index % elements_per_tile;
            
            // Read the source sparse tile (following loopback async pattern)
            noc_async_read_tile(src_tile_idx, sparse_accessor, sparse_l1_addr);
            noc_async_read_barrier();
            
            uint16_t* sparse_data = reinterpret_cast<uint16_t*>(sparse_l1_addr);
            
            // Perform gather operation: dense[j] = sparse[pattern[j] + delta * i]
            dense_data[elem_idx] = sparse_data[src_elem_offset];
        }

        // Write back the updated dense tile to DRAM (following loopback async pattern)
        noc_async_write_tile(tile_idx, dense_accessor, dense_l1_addr);
        noc_async_write_barrier();
    }
}