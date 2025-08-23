/*!
  \file TensTorrentBackend.cc
*/

#include "TensTorrentBackend.hh"

#ifdef USE_TENSTORRENT

#include <iostream>
#include <stdexcept>
#include "AlignedAllocator.hh"
#include <algorithm>
#include <tt-metalium/bfloat16.hpp>

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
    
    std::cout << "=== TensTorrent Core Configuration Debug ===" << std::endl;
    std::cout << "Device compute grid size: " << compute_grid_size_.x << "x" << compute_grid_size_.y 
              << " (" << (compute_grid_size_.x * compute_grid_size_.y) << " total cores)" << std::endl;
    std::cout << "User requested cores (--tt-cores): " << num_cores_ << std::endl;
    
    // Calculate effective grid size based on user's request
    uint32_t total_device_cores = compute_grid_size_.x * compute_grid_size_.y;
    uint32_t effective_cores;
    
    if (num_cores_ <= 0) {
        // Use all cores if user didn't specify or specified 0/negative
        effective_cores = total_device_cores;
        effective_grid_size_ = compute_grid_size_;
        std::cout << "Using all available cores (default behavior)" << std::endl;
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
        
        std::cout << "Limiting to " << effective_cores << " cores as requested by user" << std::endl;
    }
    
    std::cout << "Effective grid size: " << effective_grid_size_.x << "x" << effective_grid_size_.y 
              << " (" << (effective_grid_size_.x * effective_grid_size_.y) << " cores)" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    // For now, we'll use the split_work_to_cores approach directly in the kernel execution
    // The active_cores_ vector will be populated by split_work_to_cores when we execute kernels
    // This allows for optimal work distribution based on the actual workload size
    
    std::cout << "Multi-core discovery completed. Cores will be allocated dynamically based on workload." << std::endl;
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
    
    // Align size to proper boundaries
    size_t aligned_size = align_to_tile_size(size_bytes);
    if (type == tt::tt_metal::BufferType::DRAM) {
        // Blackhole requires 64B alignment for DRAM
        aligned_size = ((aligned_size + DRAM_ALIGNMENT - 1) / DRAM_ALIGNMENT) * DRAM_ALIGNMENT;
    }
    
    InterleavedBufferConfig config{
        .device = device_,
        .size = aligned_size,
        .page_size = TILE_SIZE_BYTES,
        .buffer_type = type
    };
    
    return CreateBuffer(config);
}

std::shared_ptr<tt::tt_metal::Buffer> TensTorrentDevice::createBuffer(size_t size_bytes, 
                                                                      tt::tt_metal::BufferType type) {
    // Wrapper method for consistency with Configuration naming
    return allocate_buffer(size_bytes, type);
}

void TensTorrentDevice::write_buffer(std::shared_ptr<tt::tt_metal::Buffer> buffer, 
                                     const std::vector<double>& data, bool blocking) {
    if (!initialized_) {
        throw std::runtime_error("TensTorrent device not initialized");
    }
    
    // Store the original size for later reads
    buffer_sizes_[buffer] = data.size();
    std::cout << "DEBUG: write_buffer storing size " << data.size() << " for buffer" << std::endl;
    
    // Convert double to bfloat16 format and pad to tile alignment if needed
    size_t tile_elements = TILE_SIZE_BYTES / sizeof(bfloat16); // 1024 elements per tile
    size_t aligned_size = ((data.size() + tile_elements - 1) / tile_elements) * tile_elements;
    
    std::vector<bfloat16> tt_data;
    tt_data.reserve(aligned_size);
    
    // Copy actual data
    for (size_t i = 0; i < data.size(); ++i) {
        tt_data.push_back(bfloat16(static_cast<float>(data[i])));
    }
    
    // Pad with zeros to tile alignment
    for (size_t i = data.size(); i < aligned_size; ++i) {
        tt_data.push_back(bfloat16(0.0f));
    }
    
    EnqueueWriteBuffer(*command_queue_, buffer, tt_data, blocking);
}

void TensTorrentDevice::read_buffer(std::shared_ptr<tt::tt_metal::Buffer> buffer, 
                                    std::vector<double>& data, bool blocking) {
    if (!initialized_) {
        throw std::runtime_error("TensTorrent device not initialized");
    }
    
    std::vector<bfloat16> tt_data;
    EnqueueReadBuffer(*command_queue_, buffer, tt_data, blocking);
    
    // Get the original size that was written
    size_t original_size = tt_data.size();
    std::cout << "DEBUG: read_buffer tt_data.size() = " << tt_data.size() << std::endl;
    if (buffer_sizes_.find(buffer) != buffer_sizes_.end()) {
        original_size = buffer_sizes_[buffer];
        std::cout << "DEBUG: read_buffer found stored size " << original_size << std::endl;
    } else {
        std::cout << "DEBUG: read_buffer buffer not found in map, using tt_data.size()" << std::endl;
    }
    
    // Convert bfloat16 back to double, only for the original size
    data.clear();
    data.reserve(original_size);
    for (size_t i = 0; i < std::min(original_size, tt_data.size()); ++i) {
        data.push_back(static_cast<double>(tt_data[i].to_float()));
    }
}

void TensTorrentDevice::readBuffer(std::shared_ptr<tt::tt_metal::Buffer> buffer, 
                                   std::vector<double>& data, bool blocking) {
    // Wrapper method for consistency with Configuration naming
    read_buffer(buffer, data, blocking);
}

void TensTorrentDevice::writeBuffer(std::shared_ptr<tt::tt_metal::Buffer> buffer, 
                                    const std::vector<double>& data, bool blocking) {
    // Wrapper method for consistency with Configuration naming
    write_buffer(buffer, data, blocking);
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

void TensTorrentDevice::execute_gather_kernel(
    std::shared_ptr<tt::tt_metal::Buffer> src_buffer,
    std::shared_ptr<tt::tt_metal::Buffer> dst_buffer,
    std::shared_ptr<tt::tt_metal::Buffer> pattern_buffer,
    size_t num_elements) {
    
    if (!initialized_) {
        throw std::runtime_error("TensTorrent device not initialized");
    }
    
    // Allocate L1 buffer for temporary storage
    auto l1_buffer = allocate_buffer(3 * TILE_SIZE_BYTES, tt::tt_metal::BufferType::L1);
    
    // Set runtime arguments for the gather kernel
    const std::vector<uint32_t> runtime_args = {
        src_buffer->address(),
        dst_buffer->address(), 
        pattern_buffer->address(),
        static_cast<uint32_t>(num_elements),
        l1_buffer->address()
    };
    
    // Use the first active core for legacy gather function
    if (!active_cores_.empty()) {
        SetRuntimeArgs(gather_program_, gather_reader_kernel_handle_, active_cores_[0], runtime_args);
    }
    
    // Execute the program
    EnqueueProgram(*command_queue_, gather_program_, false);
    Finish(*command_queue_);
}

bool TensTorrentDevice::executeGatherKernel(
    std::shared_ptr<tt::tt_metal::Buffer> src_buffer,
    std::shared_ptr<tt::tt_metal::Buffer> dst_buffer,
    std::shared_ptr<tt::tt_metal::Buffer> pattern_buffer,
    uint32_t num_elements,
    uint32_t delta) {
    
    std::cout << "DEBUG: executeGatherKernel called!" << std::endl;
    
    if (!initialized_) {
        std::cout << "DEBUG: executeGatherKernel - device not initialized!" << std::endl;
        return false;
    }
    
    try {
        std::cout << "\n=== Gather Kernel Execution Debug ===" << std::endl;
        std::cout << "Total elements to process: " << num_elements << std::endl;
        std::cout << "Delta (stride): " << delta << std::endl;
        std::cout << "Effective grid size for work distribution: " << effective_grid_size_.x << "x" << effective_grid_size_.y << std::endl;
        
        // Use split_work_to_cores utility to distribute work properly using effective grid size
        auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core1, work_per_core2] = 
            split_work_to_cores(effective_grid_size_, num_elements);
            
        std::cout << "split_work_to_cores result:" << std::endl;
        std::cout << "  - Cores that will be used: " << num_cores << std::endl;
        std::cout << "  - Core group 1 work per core: " << work_per_core1 << std::endl;
        std::cout << "  - Core group 2 work per core: " << work_per_core2 << std::endl;
        
        // Set runtime arguments for each core following TT-Metal multi-core pattern
        uint32_t work_offset = 0;
        uint32_t cores_configured = 0;
        auto work_groups = {std::make_pair(core_group_1, work_per_core1), 
                           std::make_pair(core_group_2, work_per_core2)};
        
        std::cout << "Configuring runtime arguments for cores:" << std::endl;
        
        // Iterate through each work group and assign work to cores
        for (const auto& [ranges, work_per_core] : work_groups) {
            if (work_per_core == 0) {
                std::cout << "  Skipping work group with 0 work per core" << std::endl;
                continue;
            }
            
            for (const auto& range : ranges.ranges()) {
                for (const auto& core : range) {
                    cores_configured++;
                    std::cout << "  Core (" << core.x << "," << core.y << "): offset=" << work_offset 
                              << ", work=" << work_per_core << std::endl;
                    
                    // Set runtime arguments for 3-kernel architecture
                    
                    // Reader kernel arguments
                    std::vector<uint32_t> reader_runtime_args = {
                        src_buffer->address(),      // arg0: src_buffer_addr
                        pattern_buffer->address(),  // arg1: pattern_buffer_addr  
                        work_offset,                // arg2: work_offset for this core
                        work_per_core,              // arg3: work_per_core for this core
                        delta,                      // arg4: delta stride
                        num_elements                // arg5: sparse_size for bounds checking
                    };
                    SetRuntimeArgs(gather_program_, gather_reader_kernel_handle_, core, reader_runtime_args);
                    
                    // Compute kernel arguments
                    uint32_t num_tiles = (work_per_core + (32*32-1)) / (32*32); // Convert elements to tiles
                    std::vector<uint32_t> compute_runtime_args = {
                        num_tiles,                  // arg0: num_tiles to process
                        delta,                      // arg1: delta stride
                        32 * 32                     // arg2: elements_per_tile
                    };
                    SetRuntimeArgs(gather_program_, gather_compute_kernel_handle_, core, compute_runtime_args);
                    
                    // Writer kernel arguments  
                    std::vector<uint32_t> writer_runtime_args = {
                        dst_buffer->address(),      // arg0: dst_buffer_addr
                        num_tiles,                  // arg1: num_tiles to write
                        work_offset                 // arg2: work_offset for this core
                    };
                    SetRuntimeArgs(gather_program_, gather_writer_kernel_handle_, core, writer_runtime_args);
                    work_offset += work_per_core;
                }
            }
        }
        
        std::cout << "Total cores configured: " << cores_configured << std::endl;
        std::cout << "Total work distributed: " << work_offset << " elements" << std::endl;
        
        // Execute the program on all cores with a single EnqueueProgram call
        EnqueueProgram(*command_queue_, gather_program_, false);
        Finish(*command_queue_);
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "TensTorrent gather kernel execution failed: " << e.what() << std::endl;
        return false;
    }
}

bool TensTorrentDevice::executeScatterKernel(
    std::shared_ptr<tt::tt_metal::Buffer> src_buffer,
    std::shared_ptr<tt::tt_metal::Buffer> dst_buffer,
    std::shared_ptr<tt::tt_metal::Buffer> pattern_buffer,
    uint32_t num_elements,
    uint32_t delta) {
    
    std::cout << "DEBUG: executeScatterKernel called!" << std::endl;
    
    if (!initialized_ || !src_buffer || !dst_buffer || !pattern_buffer) {
        std::cout << "DEBUG: executeScatterKernel - initialization check failed!" << std::endl;
        return false;
    }
    
    try {
        std::cout << "\n=== Scatter Kernel Execution Debug ===" << std::endl;
        std::cout << "Total elements to process: " << num_elements << std::endl;
        std::cout << "Delta (stride): " << delta << std::endl;
        std::cout << "Effective grid size for work distribution: " << effective_grid_size_.x << "x" << effective_grid_size_.y << std::endl;
        
        // Use split_work_to_cores utility to distribute work properly using effective grid size
        auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core1, work_per_core2] = 
            split_work_to_cores(effective_grid_size_, num_elements);
            
        std::cout << "split_work_to_cores result:" << std::endl;
        std::cout << "  - Cores that will be used: " << num_cores << std::endl;
        std::cout << "  - Core group 1 work per core: " << work_per_core1 << std::endl;
        std::cout << "  - Core group 2 work per core: " << work_per_core2 << std::endl;
        
        // Set runtime arguments for each core following TT-Metal multi-core pattern
        uint32_t work_offset = 0;
        uint32_t cores_configured = 0;
        auto work_groups = {std::make_pair(core_group_1, work_per_core1), 
                           std::make_pair(core_group_2, work_per_core2)};
        
        std::cout << "Configuring runtime arguments for cores:" << std::endl;
        
        // Iterate through each work group and assign work to cores
        for (const auto& [ranges, work_per_core] : work_groups) {
            if (work_per_core == 0) {
                std::cout << "  Skipping work group with 0 work per core" << std::endl;
                continue;
            }
            
            for (const auto& range : ranges.ranges()) {
                for (const auto& core : range) {
                    cores_configured++;
                    std::cout << "  Core (" << core.x << "," << core.y << "): offset=" << work_offset 
                              << ", work=" << work_per_core << std::endl;
                    
                    // Set runtime arguments for this core
                    std::vector<uint32_t> core_runtime_args = {
                        src_buffer->address(),      // arg0: src_buffer_addr (dense)
                        dst_buffer->address(),      // arg1: dst_buffer_addr (sparse)
                        pattern_buffer->address(),  // arg2: pattern_buffer_addr
                        work_offset,                // arg3: work_offset for this core
                        work_per_core,              // arg4: work_per_core for this core
                        delta                       // arg5: delta stride
                    };
                    
                    SetRuntimeArgs(scatter_program_, scatter_reader_kernel_handle_, core, core_runtime_args);
                    work_offset += work_per_core;
                }
            }
        }
        
        std::cout << "Total cores configured: " << cores_configured << std::endl;
        std::cout << "Total work distributed: " << work_offset << " elements" << std::endl;
        
        // Execute the program on all cores with a single EnqueueProgram call
        EnqueueProgram(*command_queue_, scatter_program_, false);
        Finish(*command_queue_);
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "TensTorrent scatter kernel execution failed: " << e.what() << std::endl;
        return false;
    }
}

void TensTorrentDevice::execute_scatter_kernel(
    std::shared_ptr<tt::tt_metal::Buffer> src_buffer,
    std::shared_ptr<tt::tt_metal::Buffer> dst_buffer,
    std::shared_ptr<tt::tt_metal::Buffer> pattern_buffer,
    size_t num_elements) {
    
    if (!initialized_) {
        throw std::runtime_error("TensTorrent device not initialized");
    }
    
    // Similar implementation to gather but with scatter logic
    // For now, throw as not implemented
    throw std::runtime_error("Scatter kernel not yet implemented");
}

void TensTorrentDevice::execute_noc_bandwidth_kernel(
    std::shared_ptr<tt::tt_metal::Buffer> src_buffer,
    std::shared_ptr<tt::tt_metal::Buffer> dst_buffer,
    size_t num_tiles,
    uint32_t neighbor_x,
    uint32_t neighbor_y) {
    
    if (!initialized_) {
        throw std::runtime_error("TensTorrent device not initialized");
    }
    
    // Set runtime arguments for NOC bandwidth kernel
    std::vector<uint32_t> runtime_args = {
        static_cast<uint32_t>(src_buffer->address()),  // Source buffer address
        static_cast<uint32_t>(dst_buffer->address()),  // Destination buffer address  
        static_cast<uint32_t>(num_tiles),              // Number of tiles to transfer
        neighbor_x,                                    // Neighbor NOC X coordinate
        neighbor_y                                     // Neighbor NOC Y coordinate
    };
    
    // Use the first active core for NOC bandwidth test
    if (!active_cores_.empty()) {
        SetRuntimeArgs(noc_bandwidth_program_, noc_bandwidth_kernel_handle_, active_cores_[0], runtime_args);
    }
    
    // Execute the NOC bandwidth kernel
    EnqueueProgram(*command_queue_, noc_bandwidth_program_, false);
    Finish(*command_queue_);
}

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
    // Get the compute core range from the device using effective grid size
    // This creates a CoreRangeSet covering the effective compute cores
    CoreRangeSet all_cores = CoreRangeSet(CoreRange(CoreCoord{0, 0}, CoreCoord{effective_grid_size_.x - 1, effective_grid_size_.y - 1}));
    
    std::cout << "=== 3-Kernel Architecture Compilation Debug ===" << std::endl;
    std::cout << "Compiling kernels on effective grid: " << effective_grid_size_.x << "x" << effective_grid_size_.y << " cores" << std::endl;
    std::cout << "Kernel compilation range: (0,0) to (" << (effective_grid_size_.x - 1) << "," << (effective_grid_size_.y - 1) << ")" << std::endl;
    
    // Constants for circular buffers
    constexpr uint32_t single_tile_size = 32 * 32 * 2; // BFloat16 tile size = 2048 bytes
    constexpr uint32_t num_tiles_per_cb = 2; // Double buffering
    const auto cb_data_format = tt::DataFormat::Float16_b;
    
    // ===============================
    // GATHER PROGRAM SETUP
    // ===============================
    gather_program_ = CreateProgram();
    
    // Create circular buffers for gather operation (following matmul pattern)
    // CB_c_0: Sparse data (from reader to compute)
    tt::tt_metal::CreateCircularBuffer(
        gather_program_,
        all_cores,
        CircularBufferConfig(num_tiles_per_cb * single_tile_size, {{tt::CBIndex::c_0, cb_data_format}})
            .set_page_size(tt::CBIndex::c_0, single_tile_size)
    );
    
    // CB_c_1: Pattern data (from reader to compute)
    tt::tt_metal::CreateCircularBuffer(
        gather_program_,
        all_cores,
        CircularBufferConfig(num_tiles_per_cb * single_tile_size, {{tt::CBIndex::c_1, cb_data_format}})
            .set_page_size(tt::CBIndex::c_1, single_tile_size)
    );
    
    // CB_c_16: Output data (from compute to writer)
    tt::tt_metal::CreateCircularBuffer(
        gather_program_,
        all_cores,
        CircularBufferConfig(num_tiles_per_cb * single_tile_size, {{tt::CBIndex::c_16, cb_data_format}})
            .set_page_size(tt::CBIndex::c_16, single_tile_size)
    );
    
    // Create gather kernels (3-kernel architecture)
    std::vector<uint32_t> reader_compile_args;
    // TODO: Add TensorAccessor compile args when buffers are known
    
    // Reader kernel (RISCV_1, NOC_1)
    gather_reader_kernel_handle_ = CreateKernel(
        gather_program_,
        "/storage/tt/tt-spatter/src/Spatter/kernels/dataflow/reader_gather.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_args
        }
    );
    
    // Compute kernel (Math cores)
    gather_compute_kernel_handle_ = CreateKernel(
        gather_program_,
        "/storage/tt/tt-spatter/src/Spatter/kernels/compute/gather_compute.cpp",
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .compile_args = {}
        }
    );
    
    // Writer kernel (RISCV_0, NOC_0)
    std::vector<uint32_t> writer_compile_args;
    gather_writer_kernel_handle_ = CreateKernel(
        gather_program_,
        "/storage/tt/tt-spatter/src/Spatter/kernels/dataflow/writer_gather.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_args
        }
    );
    
    // ===============================
    // SCATTER PROGRAM SETUP (TODO)
    // ===============================
    scatter_program_ = CreateProgram();
    // TODO: Implement scatter 3-kernel architecture
    
    // ===============================
    // NOC BANDWIDTH TEST PROGRAM
    // ===============================
    noc_bandwidth_program_ = CreateProgram();
    
    // For NOC bandwidth test, we only need one core
    CoreCoord single_core = CoreCoord{0, 0};
    std::vector<uint32_t> noc_compile_args;
    noc_bandwidth_kernel_handle_ = CreateKernel(
        noc_bandwidth_program_,
        "/storage/tt/tt-spatter/src/Spatter/kernels/noc_bandwidth_kernel.cpp",
        single_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = noc_compile_args
        }
    );
    
    std::cout << "3-Kernel architecture compilation completed on all compute cores" << std::endl;
}

size_t TensTorrentDevice::align_to_tile_size(size_t size) const {
    return ((size + TILE_SIZE_BYTES - 1) / TILE_SIZE_BYTES) * TILE_SIZE_BYTES;
}

std::vector<float> TensTorrentDevice::convert_double_to_bfloat16(const std::vector<double>& input) const {
    std::vector<float> output;
    output.reserve(input.size());
    for (double val : input) {
        output.push_back(static_cast<float>(val));
    }
    return output;
}

std::vector<double> TensTorrentDevice::convert_bfloat16_to_double(const std::vector<float>& input) const {
    std::vector<double> output;
    output.reserve(input.size());
    for (float val : input) {
        output.push_back(static_cast<double>(val));
    }
    return output;
}

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