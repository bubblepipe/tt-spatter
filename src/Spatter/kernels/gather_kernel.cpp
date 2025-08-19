// SPDX-FileCopyrightText: © 2025 Spatter Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    // Get kernel arguments
    std::uint32_t src_buffer_addr = get_arg_val<uint32_t>(0);
    std::uint32_t dst_buffer_addr = get_arg_val<uint32_t>(1);
    std::uint32_t pattern_buffer_addr = get_arg_val<uint32_t>(2);
    std::uint32_t num_elements = get_arg_val<uint32_t>(3);
    std::uint32_t l1_buffer_addr = get_arg_val<uint32_t>(4);

    const uint32_t element_size_bytes = 4; // sizeof(float) for BFloat16 -> float conversion
    const uint32_t tile_size_bytes = 32 * 32 * 2; // BFloat16 tile size
    
    // Set up tensor accessors for source and destination buffers
    constexpr auto src_args = TensorAccessorArgs<0>();
    const auto src_accessor = TensorAccessor(src_args, src_buffer_addr, element_size_bytes);
    
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const auto dst_accessor = TensorAccessor(dst_args, dst_buffer_addr, element_size_bytes);
    
    constexpr auto pattern_args = TensorAccessorArgs<dst_args.next_compile_time_args_offset()>();
    const auto pattern_accessor = TensorAccessor(pattern_args, pattern_buffer_addr, sizeof(uint32_t));
    
    // Process elements in chunks suitable for tile-based processing
    for (uint32_t i = 0; i < num_elements; i++) {
        // Read the index from the pattern buffer
        uint32_t index_tile = i / (32 * 32); // Which tile the index is in
        uint32_t index_offset = i % (32 * 32); // Offset within the tile
        
        // Read the pattern index
        noc_async_read_tile(index_tile, pattern_accessor, l1_buffer_addr);
        noc_async_read_barrier();
        
        // Extract the actual index from L1 buffer
        uint32_t* pattern_data = reinterpret_cast<uint32_t*>(l1_buffer_addr);
        uint32_t src_index = pattern_data[index_offset];
        
        // Calculate which tile the source data is in
        uint32_t src_tile = src_index / (32 * 32);
        uint32_t src_offset = src_index % (32 * 32);
        
        // Read source data tile
        noc_async_read_tile(src_tile, src_accessor, l1_buffer_addr + tile_size_bytes);
        noc_async_read_barrier();
        
        // Calculate destination tile
        uint32_t dst_tile = i / (32 * 32);
        
        // For simplicity, copy the element from source to destination position
        // In a real implementation, you would batch these operations
        float* src_data = reinterpret_cast<float*>(l1_buffer_addr + tile_size_bytes);
        float* dst_data = reinterpret_cast<float*>(l1_buffer_addr + 2 * tile_size_bytes);
        
        // Read destination tile if it exists
        if (i % (32 * 32) == 0) {
            noc_async_read_tile(dst_tile, dst_accessor, l1_buffer_addr + 2 * tile_size_bytes);
            noc_async_read_barrier();
        }
        
        // Perform the gather operation
        dst_data[i % (32 * 32)] = src_data[src_offset];
        
        // Write back destination tile when it's full or at the end
        if ((i + 1) % (32 * 32) == 0 || i == num_elements - 1) {
            noc_async_write_tile(dst_tile, dst_accessor, l1_buffer_addr + 2 * tile_size_bytes);
            noc_async_write_barrier();
        }
    }
    
    // Invalidate L1 cache for Blackhole
    invalidate_l1_cache();
}