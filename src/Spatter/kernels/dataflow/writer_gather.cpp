// SPDX-FileCopyrightText: © 2025 Spatter Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

/*
 * Writer Kernel for Gather Operation (Multi-core version)
 * 
 * Reads computed gather results from circular buffer and writes them back to 
 * dense buffer in DRAM. Follows the official TT-Metal 3-kernel architecture.
 * 
 * Runtime Args:
 * - arg0: dst_buffer_addr - Destination (dense) buffer address in DRAM
 * - arg1: num_tiles - Number of tiles to write
 * - arg2: work_offset - Starting tile offset for this core's work
 */

void kernel_main() {
    // Runtime arguments for writing gathered data back to output buffer
    uint32_t dst_buffer_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);       // number of output tiles to write
    uint32_t work_offset = get_arg_val<uint32_t>(2);     // starting tile offset for this core

    // Circular buffer for output data from compute kernel
    constexpr uint32_t cb_id_out = tt::CBIndex::c_16;

    // Create TensorAccessor for the output DRAM buffer
    // This matches the pattern used in the official matmul writer
    constexpr uint32_t onetile = 1;  // single-tile operations
    const uint32_t tile_bytes = get_tile_size(cb_id_out);

    constexpr auto dst_args = TensorAccessorArgs<0>();
    const auto dst_accessor = TensorAccessor(dst_args, dst_buffer_addr, tile_bytes);

    // Calculate start and end tile indices for this core
    uint32_t start_tile_id = work_offset / (32 * 32);  // Convert element offset to tile offset
    uint32_t end_tile_id = start_tile_id + num_tiles;

    // Write each tile back to DRAM in order
    for (uint32_t tile_id = start_tile_id; tile_id < end_tile_id; ++tile_id) {
        // Wait for the compute kernel to produce an output tile
        cb_wait_front(cb_id_out, onetile);
        
        // Get the L1 address where the tile data is stored
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);
        
        // Write the output tile to DRAM at the correct position
        noc_async_write_tile(tile_id, dst_accessor, l1_read_addr);
        
        // Ensure the write completes before proceeding
        noc_async_write_barrier();
        
        // Mark the tile as consumed from the circular buffer
        cb_pop_front(cb_id_out, onetile);
    }
}