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
#include <tt-metalium/work_split.hpp>

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

// Helper function to pretty-print buffer configuration
void print_buffer_config(const InterleavedBufferConfig& config) {
    uint32_t num_tiles = config.size / (32 * 32 * 2);  // 2KB per tile
    
    std::cout << "buffer config:"<< std::endl; 
    std::cout << "  Buffer size: " << config.size << " bytes (" 
              << config.size / 1024 << " KB)" << std::endl;
    std::cout << "  Page size: " << config.page_size << " bytes" << std::endl;
    std::cout << "  Number of tiles: " << num_tiles << std::endl;
    std::cout << "  Buffer type: " << (config.buffer_type == BufferType::DRAM ? "DRAM" : "L1") << std::endl;
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
    
    // Always use TILE_SIZE_BYTES as page_size for DRAM buffers
    // This matches the TT-Metal examples (loopback, vecadd_multi_core)
    size_t page_size = TILE_SIZE_BYTES;
    
    InterleavedBufferConfig config{
        .device = device_,
        .size = aligned_size,
        .page_size = page_size,
        .buffer_type = type
    };
    
    
    print_buffer_config(config);

    auto buffer = CreateBuffer(config);
    return buffer;

}

void TensTorrentDevice::write_buffer(std::shared_ptr<tt::tt_metal::Buffer> buffer, 
                                     const std::vector<double>& data, bool blocking) {
    std::cout << "[DEBUG] write_buffer(double) called with " << data.size() << " elements" << std::endl;
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
    
    // Create vector with exact aligned size (no extra padding)
    // This matches TT-Metal examples which don't add extra padding
    std::vector<bfloat16> tt_data(aligned_size);
    
    // Copy actual data
    for (size_t i = 0; i < data.size(); ++i) {
        tt_data[i] = bfloat16(static_cast<float>(data[i]));
    }
    
    // Zero out padding
    for (size_t i = data.size(); i < aligned_size; ++i) {
        tt_data[i] = bfloat16(0.0f);
    }
    
    if (tt_data.empty() || !tt_data.data()) {
        throw std::runtime_error("Invalid converted data for TT-Metal buffer write");
    }
    
    std::cout << "[DEBUG] About to call EnqueueWriteBuffer(BFloat16) with " << tt_data.size() << " elements" << std::endl;
    std::cout << "[DEBUG] Buffer address: 0x" << std::hex << buffer->address() << std::dec 
              << ", Buffer size: " << buffer->size() << " bytes" << std::endl;
    std::cout << "[DEBUG] Data vector address: " << std::hex << (void*)tt_data.data() << std::dec 
              << ", Data size in bytes: " << (tt_data.size() * sizeof(bfloat16)) << std::endl;
    
    // Verify data size doesn't exceed buffer size
    size_t data_bytes = tt_data.size() * sizeof(bfloat16);
    if (data_bytes > buffer->size()) {
        std::cerr << "ERROR: Data size (" << data_bytes << " bytes) exceeds buffer size (" 
                  << buffer->size() << " bytes)" << std::endl;
        throw std::runtime_error("Data size exceeds buffer size");
    }
    
    try {
        // Always use blocking write to ensure data is copied before vector goes out of scope
        EnqueueWriteBuffer(*command_queue_, buffer, tt_data, true);
        // Add explicit finish to ensure write completes
        Finish(*command_queue_);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: EnqueueWriteBuffer failed: " << e.what() << std::endl;
        std::cerr << "  Buffer type: " << (buffer->buffer_type() == BufferType::DRAM ? "DRAM" : "L1") << std::endl;
        std::cerr << "  Buffer size: " << buffer->size() << " bytes" << std::endl;
        std::cerr << "  Data size: " << data_bytes << " bytes" << std::endl;
        throw;
    }
    std::cout << "[DEBUG] EnqueueWriteBuffer(BFloat16) execution successful" << std::endl ;
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
    
    std::cout << "[DEBUG] About to call EnqueueWriteBuffer(uint32_t) with " << aligned_data.size() << " elements" << std::endl;
    std::cout << "[DEBUG] Buffer address: 0x" << std::hex << buffer->address() << std::dec 
              << ", Buffer size: " << buffer->size() << " bytes" << std::endl;
    EnqueueWriteBuffer(*command_queue_, buffer, aligned_data, blocking);
    std::cout << "[DEBUG] EnqueueWriteBuffer(uint32_t) execution successful" << std::endl ;
}

void TensTorrentDevice::writeBuffer(std::shared_ptr<tt::tt_metal::Buffer> buffer, 
                                    const aligned_vector<double>& data, bool blocking) {
    std::cout << "[DEBUG] writeBuffer(aligned_vector<double>) called with " << data.size() << " elements" << std::endl;
    // Convert aligned_vector<double> to std::vector<double> and use existing method
    std::vector<double> std_data(data.begin(), data.end());
    write_buffer(buffer, std_data, blocking);
}

void TensTorrentDevice::writeBuffer(std::shared_ptr<tt::tt_metal::Buffer> buffer, 
                                    const aligned_vector<size_t>& data, bool blocking) {
    std::cout << "[DEBUG] writeBuffer(aligned_vector<size_t>) called with " << data.size() << " elements" << std::endl;
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
        Program gather_program = CreateProgram();
        
        // Tile size constants
        constexpr uint32_t tile_size_bytes = 32 * 32 * 2;  // 2048 bytes per tile
        
        // Use the effective grid size based on --tt-cores parameter
        auto core_grid = effective_grid_size_;
        
        // Split work across cores
        constexpr bool row_major = true;
        auto [num_cores, all_cores, core_group_1, core_group_2, 
              elements_per_core_group_1, elements_per_core_group_2] = 
            split_work_to_cores(core_grid, num_elements, row_major);
        
        // Debug output for multi-core analysis
        std::cout << "[TensTorrent Gather] Multi-core debug info:" << std::endl;
        std::cout << "  - Requested cores (--tt-cores): " << num_cores_ << std::endl;
        std::cout << "  - Device grid size: " << compute_grid_size_.x << "x" << compute_grid_size_.y 
                  << " = " << (compute_grid_size_.x * compute_grid_size_.y) << " cores" << std::endl;
        std::cout << "  - Effective grid size: " << effective_grid_size_.x << "x" << effective_grid_size_.y 
                  << " = " << (effective_grid_size_.x * effective_grid_size_.y) << " cores" << std::endl;
        std::cout << "  - Core grid passed to split_work: " << core_grid.x << "x" << core_grid.y 
                  << " = " << (core_grid.x * core_grid.y) << " cores" << std::endl;
        std::cout << "  - Cores actually used: " << num_cores << std::endl;
        std::cout << "  - Elements to process: " << num_elements << std::endl;
        std::cout << "  - Elements per core (group 1): " << elements_per_core_group_1 << std::endl;
        std::cout << "  - Elements per core (group 2): " << elements_per_core_group_2 << std::endl;
        std::cout << "  - Number of cores in group 1: " << core_group_1.num_cores() << std::endl;
        std::cout << "  - Number of cores in group 2: " << core_group_2.num_cores() << std::endl;
        
        // Additional debug for crash investigation
        bool is_power_of_2 = (num_elements & (num_elements - 1)) == 0;
        if (is_power_of_2) {
            std::cout << "  ⚠ WARNING: Size is exact power of 2!" << std::endl;
        }
        
        // Check buffer sizes
        size_t pattern_bytes = num_elements * sizeof(uint32_t);
        size_t sparse_bytes = num_elements * sizeof(bfloat16);
        size_t dense_bytes = num_elements * sizeof(bfloat16);
        std::cout << "  - Buffer sizes:" << std::endl;
        std::cout << "    - Pattern: " << pattern_bytes << " bytes (" << pattern_bytes/1024 << " KB)" << std::endl;
        std::cout << "    - Sparse: " << sparse_bytes << " bytes (" << sparse_bytes/1024 << " KB)" << std::endl;
        std::cout << "    - Dense: " << dense_bytes << " bytes (" << dense_bytes/1024 << " KB)" << std::endl;
        std::cout << "    - Total DRAM: " << (pattern_bytes + sparse_bytes + dense_bytes) << " bytes (" 
                  << (pattern_bytes + sparse_bytes + dense_bytes)/1024 << " KB)" << std::endl;
        
        // Create L1 buffers on all cores
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
        
        // Allocate the L1 buffers (shared across all cores)
        auto l1_pattern_buffer = CreateBuffer(l1_pattern_config);
        auto l1_sparse_buffer = CreateBuffer(l1_sparse_config);
        auto l1_dense_buffer = CreateBuffer(l1_dense_config);
        
        // Compile-time arguments
        std::vector<uint32_t> compile_time_args;
        TensorAccessorArgs(*src_buffer).append_to(compile_time_args);
        TensorAccessorArgs(*dst_buffer).append_to(compile_time_args);
        TensorAccessorArgs(*pattern_buffer).append_to(compile_time_args);
        
        // Create kernel on all cores
        KernelHandle gather_kernel_id = CreateKernel(
            gather_program,
            "/storage/tt/tt-spatter/src/Spatter/kernels/gather_kernel.cpp",
            all_cores,  // Run on all cores instead of single core
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = compile_time_args
            }
        );
        
        // Set runtime arguments per core
        uint32_t start_element = 0;
        auto work_groups = {
            std::make_pair(core_group_1, elements_per_core_group_1),
            std::make_pair(core_group_2, elements_per_core_group_2)
        };
        
        for (const auto& [group, elements_per_core] : work_groups) {
            for (const auto& range : group.ranges()) {
                for (const auto& core : range) {
                    const std::vector<uint32_t> runtime_args = {
                        l1_pattern_buffer->address(),    // arg0: pattern L1 buffer
                        l1_sparse_buffer->address(),     // arg1: sparse L1 buffer  
                        l1_dense_buffer->address(),      // arg2: dense L1 buffer
                        start_element,                   // arg3: start element for this core
                        elements_per_core,               // arg4: number of elements for this core
                        delta,                           // arg5: delta
                        pattern_length,                  // arg6: pattern length
                        src_buffer->address(),           // arg7: sparse DRAM buffer
                        dst_buffer->address(),           // arg8: dense DRAM buffer
                        pattern_buffer->address()        // arg9: pattern DRAM buffer
                    };
                    
                    SetRuntimeArgs(gather_program, gather_kernel_id, core, runtime_args);
                    start_element += elements_per_core;
                }
            }
        }
        
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
        Program scatter_program = CreateProgram();
        
        // Tile size constants
        constexpr uint32_t tile_size_bytes = 32 * 32 * 2;  // 2048 bytes per tile
        
        // Use the effective grid size based on --tt-cores parameter
        auto core_grid = effective_grid_size_;
        
        // Split work across cores
        constexpr bool row_major = true;
        auto [num_cores, all_cores, core_group_1, core_group_2, 
              elements_per_core_group_1, elements_per_core_group_2] = 
            split_work_to_cores(core_grid, num_elements, row_major);
        
        // Debug output for multi-core analysis
        std::cout << "[TensTorrent Scatter] Multi-core debug info:" << std::endl;
        std::cout << "  - Requested cores (--tt-cores): " << num_cores_ << std::endl;
        std::cout << "  - Device grid size: " << compute_grid_size_.x << "x" << compute_grid_size_.y 
                  << " = " << (compute_grid_size_.x * compute_grid_size_.y) << " cores" << std::endl;
        std::cout << "  - Effective grid size: " << effective_grid_size_.x << "x" << effective_grid_size_.y 
                  << " = " << (effective_grid_size_.x * effective_grid_size_.y) << " cores" << std::endl;
        std::cout << "  - Core grid passed to split_work: " << core_grid.x << "x" << core_grid.y 
                  << " = " << (core_grid.x * core_grid.y) << " cores" << std::endl;
        std::cout << "  - Cores actually used: " << num_cores << std::endl;
        std::cout << "  - Elements to process: " << num_elements << std::endl;
        std::cout << "  - Elements per core (group 1): " << elements_per_core_group_1 << std::endl;
        std::cout << "  - Elements per core (group 2): " << elements_per_core_group_2 << std::endl;
        std::cout << "  - Number of cores in group 1: " << core_group_1.num_cores() << std::endl;
        std::cout << "  - Number of cores in group 2: " << core_group_2.num_cores() << std::endl;
        
        // Create L1 buffers on all cores
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
        
        // Allocate the L1 buffers (shared across all cores)
        auto l1_pattern_buffer = CreateBuffer(l1_pattern_config);
        auto l1_dense_buffer = CreateBuffer(l1_dense_config);
        auto l1_sparse_buffer = CreateBuffer(l1_sparse_config);
        
        // Compile-time arguments
        std::vector<uint32_t> compile_time_args;
        TensorAccessorArgs(*src_buffer).append_to(compile_time_args);
        TensorAccessorArgs(*dst_buffer).append_to(compile_time_args);
        TensorAccessorArgs(*pattern_buffer).append_to(compile_time_args);
        
        // Create kernel on all cores
        KernelHandle scatter_kernel_id = CreateKernel(
            scatter_program,
            "/storage/tt/tt-spatter/src/Spatter/kernels/scatter_kernel.cpp",
            all_cores,  // Run on all cores instead of single core
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = compile_time_args
            }
        );
        
        // Set runtime arguments per core
        uint32_t start_element = 0;
        auto work_groups = {
            std::make_pair(core_group_1, elements_per_core_group_1),
            std::make_pair(core_group_2, elements_per_core_group_2)
        };
        
        for (const auto& [group, elements_per_core] : work_groups) {
            for (const auto& range : group.ranges()) {
                for (const auto& core : range) {
                    const std::vector<uint32_t> runtime_args = {
                        l1_pattern_buffer->address(),   // arg0: pattern L1 buffer
                        l1_dense_buffer->address(),     // arg1: dense L1 buffer (source)
                        l1_sparse_buffer->address(),    // arg2: sparse L1 buffer (destination)
                        start_element,                  // arg3: start element for this core
                        elements_per_core,              // arg4: number of elements for this core
                        delta,                          // arg5: delta
                        pattern_length,                 // arg6: pattern length
                        src_buffer->address(),          // arg7: dense DRAM buffer (source)
                        dst_buffer->address(),          // arg8: sparse DRAM buffer (destination)
                        pattern_buffer->address()       // arg9: pattern DRAM buffer
                    };
                    
                    SetRuntimeArgs(scatter_program, scatter_kernel_id, core, runtime_args);
                    start_element += elements_per_core;
                }
            }
        }
        
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