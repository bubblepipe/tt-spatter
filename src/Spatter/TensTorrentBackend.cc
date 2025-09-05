/*!
  \file TensTorrentBackend.cc
*/

#include "TensTorrentBackend.hh"

#ifdef USE_TENSTORRENT

#include <iostream>
#include <stdexcept>
#include <set>
#include "AlignedAllocator.hh"
#include <algorithm>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::tt_metal;

namespace Spatter {

TensTorrentDevice::TensTorrentDevice(int device_id, int num_cores) 
    : device_id_(device_id), initialized_(false), device_(nullptr), command_queue_(nullptr), num_cores_(num_cores) {
}

TensTorrentDevice::~TensTorrentDevice() {
    cleanup();
}

void TensTorrentDevice::discover_cores() {
    // Get the actual compute grid size from the device
    compute_grid_size_ = device_->compute_with_storage_grid_size();
    
    
    // Calculate effective grid size based on user's request
    uint32_t total_device_cores = compute_grid_size_.x * compute_grid_size_.y;
    uint32_t effective_cores;
    
    if (num_cores_ <= 0) {
        // Use all cores if user didn't specify or specified 0/negative
        effective_cores = total_device_cores;
        effective_grid_size_ = compute_grid_size_;
    } else {
        // Limit to user's requested number
        effective_cores = std::min(static_cast<uint32_t>(num_cores_), total_device_cores);
        
        // Calculate effective grid dimensions
        // For simplicity, we'll use a rectangular grid that fits the requested cores
        // Start with same aspect ratio as device, then adjust
        uint32_t effective_x = std::min(static_cast<uint32_t>(compute_grid_size_.x), effective_cores);
        uint32_t effective_y = (effective_cores + effective_x - 1) / effective_x; // Ceiling division
        effective_y = std::min(effective_y, static_cast<uint32_t>(compute_grid_size_.y));
        
        // Recalculate to ensure we don't exceed device limits
        effective_cores = std::min(effective_x * effective_y, total_device_cores);
        effective_grid_size_ = CoreCoord{effective_x, effective_y};
        
    }
    
    
    // For now, we'll use the split_work_to_cores approach directly in the kernel execution
    // The active_cores_ vector will be populated by split_work_to_cores when we execute kernels
    // This allows for optimal work distribution based on the actual workload size
    
}

bool TensTorrentDevice::initialize() {
    try {
        // Initialize the TensTorrent device
        device_ = CreateDevice(device_id_);
        if (!device_) {
            std::cerr << "Failed to create TensTorrent device " << device_id_ << std::endl;
            return false;
        }
        
        // Get command queue for this device
        command_queue_ = &device_->command_queue();
        
        // Query available compute cores
        discover_cores();
        
        // Compile kernels
        compile_kernels();
        
        initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "TensTorrent device initialization failed: " << e.what() << std::endl;
        return false;
    }
}

void TensTorrentDevice::cleanup() {
    if (device_) {
        CloseDevice(device_);
        device_ = nullptr;
        command_queue_ = nullptr;
    }
    initialized_ = false;
}

std::shared_ptr<tt::tt_metal::Buffer> TensTorrentDevice::allocate_buffer(size_t size_bytes, 
                                                                         tt::tt_metal::BufferType type) {
    if (!initialized_) {
        throw std::runtime_error("TensTorrent device not initialized");
    }
    
    // Debug: Print original size
    
    // Align size to proper boundaries
    size_t aligned_size = align_to_tile_size(size_bytes);
    
    if (type == tt::tt_metal::BufferType::DRAM) {
        // Blackhole requires 64B alignment for DRAM
        size_t before_dram = aligned_size;
        aligned_size = ((aligned_size + DRAM_ALIGNMENT - 1) / DRAM_ALIGNMENT) * DRAM_ALIGNMENT;
    }
    
    // Check if this is a potentially problematic size
    bool is_power_of_2 = (aligned_size & (aligned_size - 1)) == 0;
    
    InterleavedBufferConfig config{
        .device = device_,
        .size = aligned_size,
        .page_size = TILE_SIZE_BYTES,
        .buffer_type = type
    };
    
    
    try {
        auto buffer = CreateBuffer(config);
        return buffer;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: CreateBuffer failed with size " << aligned_size 
                  << " bytes: " << e.what() << std::endl;
        throw;
    }
}

// Method removed - use allocate_buffer directly

void TensTorrentDevice::write_buffer(std::shared_ptr<tt::tt_metal::Buffer> buffer, 
                                     const std::vector<double>& data, bool blocking) {
    if (!initialized_) {
        throw std::runtime_error("TensTorrent device not initialized");
    }
    
    if (!buffer) {
        throw std::runtime_error("Invalid buffer pointer in write_buffer");
    }
    
    if (data.empty()) {
        return;
    }
    
    buffer_sizes_[buffer] = data.size();
    
    size_t tile_elements = TILE_SIZE_BYTES / sizeof(bfloat16);
    size_t aligned_size = ((data.size() + tile_elements - 1) / tile_elements) * tile_elements;
    
    std::vector<bfloat16> tt_data;
    tt_data.reserve(aligned_size);
    
    for (size_t i = 0; i < data.size(); ++i) {
        tt_data.push_back(bfloat16(static_cast<float>(data[i])));
    }
    
    for (size_t i = data.size(); i < aligned_size; ++i) {
        tt_data.push_back(bfloat16(0.0f));
    }
    
    if (tt_data.empty() || !tt_data.data()) {
        throw std::runtime_error("Invalid converted data for TT-Metal buffer write");
    }
    
    try {
        EnqueueWriteBuffer(*command_queue_, buffer, tt_data, blocking);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: EnqueueWriteBuffer failed: " << e.what() << std::endl;
        throw;
    }
}

void TensTorrentDevice::read_buffer(std::shared_ptr<tt::tt_metal::Buffer> buffer, 
                                    std::vector<double>& data, bool blocking) {
    if (!initialized_) {
        throw std::runtime_error("TensTorrent device not initialized");
    }
    
    std::vector<bfloat16> tt_data;
    EnqueueReadBuffer(*command_queue_, buffer, tt_data, blocking);
    
    size_t original_size = tt_data.size();
    if (buffer_sizes_.find(buffer) != buffer_sizes_.end()) {
        original_size = buffer_sizes_[buffer];
    }
    
    data.clear();
    data.reserve(original_size);
    
    for (size_t i = 0; i < std::min(original_size, tt_data.size()); ++i) {
        data.push_back(static_cast<double>(tt_data[i].to_float()));
    }
}

// Method removed - use read_buffer directly

// Method removed - use write_buffer directly

void TensTorrentDevice::writeBuffer(std::shared_ptr<tt::tt_metal::Buffer> buffer, 
                                    const std::vector<uint32_t>& data, bool blocking) {
    if (!initialized_) {
        throw std::runtime_error("TensTorrent device not initialized");
    }
    
    // Store the original size for later reads
    buffer_sizes_[buffer] = data.size();
    
    // Align data to tile boundary with zero padding
    constexpr size_t elements_per_tile = 32 * 32; // 1024 elements per tile
    size_t aligned_size = ((data.size() + elements_per_tile - 1) / elements_per_tile) * elements_per_tile;
    
    std::vector<uint32_t> aligned_data = data;
    aligned_data.resize(aligned_size, 0); // Pad with zeros
    
    EnqueueWriteBuffer(*command_queue_, buffer, aligned_data, blocking);
}

void TensTorrentDevice::writeBuffer(std::shared_ptr<tt::tt_metal::Buffer> buffer, 
                                    const aligned_vector<double>& data, bool blocking) {
    // Convert aligned_vector<double> to std::vector<double> and use existing method
    std::vector<double> std_data(data.begin(), data.end());
    write_buffer(buffer, std_data, blocking);
}

void TensTorrentDevice::writeBuffer(std::shared_ptr<tt::tt_metal::Buffer> buffer, 
                                    const aligned_vector<size_t>& data, bool blocking) {
    // Convert aligned_vector<size_t> to std::vector<uint32_t> for TT-Metal
    std::vector<uint32_t> uint32_data;
    uint32_data.reserve(data.size());
    for (size_t val : data) {
        uint32_data.push_back(static_cast<uint32_t>(val));
    }
    writeBuffer(buffer, uint32_data, blocking);
}

void TensTorrentDevice::readBuffer(std::shared_ptr<tt::tt_metal::Buffer> buffer, 
                                   aligned_vector<double>& data, bool blocking) {
    // Use existing read_buffer and convert result to aligned_vector
    std::vector<double> std_data;
    read_buffer(buffer, std_data, blocking);
    
    // Copy to aligned_vector
    data.clear();
    data.reserve(std_data.size());
    for (double val : std_data) {
        data.push_back(val);
    }
}

// Legacy method removed - use executeGatherKernel instead

bool TensTorrentDevice::executeGatherKernel(
    std::shared_ptr<tt::tt_metal::Buffer> src_buffer,
    std::shared_ptr<tt::tt_metal::Buffer> dst_buffer,
    std::shared_ptr<tt::tt_metal::Buffer> pattern_buffer,
    uint32_t num_elements,
    uint32_t delta,
    uint32_t pattern_length) {
    
    if (!initialized_) {
        return false;
    }
    
    try {
        constexpr CoreCoord core = {0, 0};
        Program gather_program = CreateProgram();
        
        // Tile size constants
        constexpr uint32_t tile_size_bytes = 32 * 32 * 2;  // 2048 bytes per tile
        
        // Create three separate L1 buffers for pattern, sparse, and dense tiles
        // Following the pattern from loopback example
        InterleavedBufferConfig l1_pattern_config{
            .device = device_,
            .size = tile_size_bytes,
            .page_size = tile_size_bytes,
            .buffer_type = tt::tt_metal::BufferType::L1
        };
        
        InterleavedBufferConfig l1_sparse_config{
            .device = device_,
            .size = tile_size_bytes,
            .page_size = tile_size_bytes,
            .buffer_type = tt::tt_metal::BufferType::L1
        };
        
        InterleavedBufferConfig l1_dense_config{
            .device = device_,
            .size = tile_size_bytes,
            .page_size = tile_size_bytes,
            .buffer_type = tt::tt_metal::BufferType::L1
        };
        
        // Allocate the L1 buffers
        auto l1_pattern_buffer = CreateBuffer(l1_pattern_config);
        auto l1_sparse_buffer = CreateBuffer(l1_sparse_config);
        auto l1_dense_buffer = CreateBuffer(l1_dense_config);
        
        std::vector<uint32_t> compile_time_args;
        TensorAccessorArgs(*src_buffer).append_to(compile_time_args);
        TensorAccessorArgs(*dst_buffer).append_to(compile_time_args);
        TensorAccessorArgs(*pattern_buffer).append_to(compile_time_args);
        
        KernelHandle gather_kernel_id = CreateKernel(
            gather_program,
            "/storage/tt/tt-spatter/src/Spatter/kernels/gather_kernel.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = compile_time_args
            }
        );
        
        const std::vector<uint32_t> runtime_args = {
            l1_pattern_buffer->address(),    // arg0: pattern L1 buffer
            l1_sparse_buffer->address(),     // arg1: sparse L1 buffer
            l1_dense_buffer->address(),      // arg2: dense L1 buffer
            num_elements,                    // arg3: number of elements
            delta,                           // arg4: delta
            pattern_length,                  // arg5: pattern length
            src_buffer->address(),           // arg6: sparse DRAM buffer
            dst_buffer->address(),           // arg7: dense DRAM buffer
            pattern_buffer->address()        // arg8: pattern DRAM buffer
        };
        
        SetRuntimeArgs(gather_program, gather_kernel_id, core, runtime_args);
        EnqueueProgram(*command_queue_, gather_program, false);
        Finish(*command_queue_);
        
        return true;
        
    } catch (const std::exception& e) {
        return false;
    }
}

bool TensTorrentDevice::executeScatterKernel(
    std::shared_ptr<tt::tt_metal::Buffer> src_buffer,
    std::shared_ptr<tt::tt_metal::Buffer> dst_buffer,
    std::shared_ptr<tt::tt_metal::Buffer> pattern_buffer,
    uint32_t num_elements,
    uint32_t delta,
    uint32_t pattern_length) {
    
    if (!initialized_ || !src_buffer || !dst_buffer || !pattern_buffer) {
        return false;
    }
    
    try {
        constexpr CoreCoord core = {0, 0};
        Program scatter_program = CreateProgram();
        
        // Tile size constants
        constexpr uint32_t tile_size_bytes = 32 * 32 * 2;  // 2048 bytes per tile
        
        // Create three separate L1 buffers for pattern, dense, and sparse tiles
        // Following the same pattern as gather kernel
        InterleavedBufferConfig l1_pattern_config{
            .device = device_,
            .size = tile_size_bytes,
            .page_size = tile_size_bytes,
            .buffer_type = tt::tt_metal::BufferType::L1
        };
        
        InterleavedBufferConfig l1_dense_config{
            .device = device_,
            .size = tile_size_bytes,
            .page_size = tile_size_bytes,
            .buffer_type = tt::tt_metal::BufferType::L1
        };
        
        InterleavedBufferConfig l1_sparse_config{
            .device = device_,
            .size = tile_size_bytes,
            .page_size = tile_size_bytes,
            .buffer_type = tt::tt_metal::BufferType::L1
        };
        
        // Allocate the L1 buffers
        auto l1_pattern_buffer = CreateBuffer(l1_pattern_config);
        auto l1_dense_buffer = CreateBuffer(l1_dense_config);
        auto l1_sparse_buffer = CreateBuffer(l1_sparse_config);
        
        std::vector<uint32_t> compile_time_args;
        TensorAccessorArgs(*src_buffer).append_to(compile_time_args);
        TensorAccessorArgs(*dst_buffer).append_to(compile_time_args);
        TensorAccessorArgs(*pattern_buffer).append_to(compile_time_args);
        
        KernelHandle scatter_kernel_id = CreateKernel(
            scatter_program,
            "/storage/tt/tt-spatter/src/Spatter/kernels/scatter_kernel.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = compile_time_args
            }
        );
        
        // Runtime arguments matching gather kernel structure (9 args)
        const std::vector<uint32_t> runtime_args = {
            l1_pattern_buffer->address(),   // arg0: pattern L1 buffer
            l1_dense_buffer->address(),     // arg1: dense L1 buffer (source)
            l1_sparse_buffer->address(),    // arg2: sparse L1 buffer (destination)
            num_elements,                   // arg3: number of elements
            delta,                          // arg4: delta
            pattern_length,                 // arg5: pattern length
            src_buffer->address(),          // arg6: dense DRAM buffer (source)
            dst_buffer->address(),          // arg7: sparse DRAM buffer (destination)
            pattern_buffer->address()       // arg8: pattern DRAM buffer
        };
        
        SetRuntimeArgs(scatter_program, scatter_kernel_id, core, runtime_args);
        EnqueueProgram(*command_queue_, scatter_program, false);
        Finish(*command_queue_);
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "TensTorrent scatter kernel execution failed: " << e.what() << std::endl;
        return false;
    }
}

// Legacy method removed - use executeScatterKernel instead

// Method removed - NOC bandwidth kernel no longer available

std::string TensTorrentDevice::get_device_info() const {
    if (!initialized_) {
        return "TensTorrent device not initialized";
    }
    
    return "TensTorrent Blackhole Device " + std::to_string(device_id_);
}

size_t TensTorrentDevice::get_max_memory() const {
    // Blackhole has ~4GB DRAM per device
    return 4ULL * 1024 * 1024 * 1024;
}

void TensTorrentDevice::compile_kernels() {
    // Single-kernel architecture - kernels compiled on-demand
}

// Method removed - using single-kernel approach following loopback pattern

// Method removed - duplicated functionality with initializeGatherProgram

size_t TensTorrentDevice::align_to_tile_size(size_t size) const {
    return ((size + TILE_SIZE_BYTES - 1) / TILE_SIZE_BYTES) * TILE_SIZE_BYTES;
}

// Conversion methods removed - conversion handled directly in write_buffer/read_buffer

// Helper functions
void check_tt_error(const std::string& operation) {
    // TT-Metalium uses exceptions for error handling
    // This function is a placeholder for any additional error checking
}

size_t calculate_buffer_size(size_t num_elements) {
    const size_t element_size = sizeof(bfloat16);
    const size_t tile_size = 32 * 32 * element_size;
    size_t total_size = num_elements * element_size;
    
    // Round up to tile boundary
    return ((total_size + tile_size - 1) / tile_size) * tile_size;
}

} // namespace Spatter

#endif // USE_TENSTORRENT