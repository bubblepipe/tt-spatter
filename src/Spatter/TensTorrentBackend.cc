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

TensTorrentDevice::TensTorrentDevice(int device_id) 
    : device_id_(device_id), initialized_(false), device_(nullptr), command_queue_(nullptr) {
}

TensTorrentDevice::~TensTorrentDevice() {
    cleanup();
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
    
    CoreCoord core = get_default_core();
    SetRuntimeArgs(gather_program_, gather_kernel_handle_, core, runtime_args);
    
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
    
    if (!initialized_) {
        return false;
    }
    
    try {
        // Set up runtime arguments for gather kernel
        std::vector<uint32_t> runtime_args = {
            src_buffer->address(),      // arg0: src_buffer_addr
            dst_buffer->address(),      // arg1: dst_buffer_addr  
            pattern_buffer->address(),  // arg2: pattern_buffer_addr
            num_elements,               // arg3: num_elements
            delta                       // arg4: delta stride
        };
        
        // Set runtime arguments
        SetRuntimeArgs(gather_program_, gather_kernel_handle_, get_default_core(), runtime_args);
        
        // Execute the program
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
    
    if (!initialized_ || !src_buffer || !dst_buffer || !pattern_buffer) {
        return false;
    }
    
    try {
        // Set up runtime arguments for scatter kernel
        std::vector<uint32_t> runtime_args = {
            src_buffer->address(),      // arg0: src_buffer_addr (dense)
            dst_buffer->address(),      // arg1: dst_buffer_addr (sparse)
            pattern_buffer->address(),  // arg2: pattern_buffer_addr
            num_elements,               // arg3: num_elements
            delta                       // arg4: delta stride
        };
        
        // Set runtime arguments for scatter program
        SetRuntimeArgs(scatter_program_, scatter_kernel_handle_, get_default_core(), runtime_args);
        
        // Execute the program
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
    
    auto core = get_default_core();
    SetRuntimeArgs(noc_bandwidth_program_, noc_bandwidth_kernel_handle_, core, runtime_args);
    
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
    // Create gather program
    gather_program_ = CreateProgram();
    
    CoreCoord core = get_default_core();
    
    // Create gather kernel
    std::vector<uint32_t> compile_args;
    // Add any compile-time arguments here
    
    gather_kernel_handle_ = CreateKernel(
        gather_program_,
        "/storage/tt/tt-spatter/src/Spatter/kernels/gather_kernel.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = compile_args
        }
    );
    
    // Create scatter program
    scatter_program_ = CreateProgram();
    
    scatter_kernel_handle_ = CreateKernel(
        scatter_program_,
        "/storage/tt/tt-spatter/src/Spatter/kernels/scatter_kernel.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = compile_args
        }
    );
    
    // Create NOC bandwidth program  
    noc_bandwidth_program_ = CreateProgram();
    
    noc_bandwidth_kernel_handle_ = CreateKernel(
        noc_bandwidth_program_,
        "/storage/tt/tt-spatter/src/Spatter/kernels/noc_bandwidth_kernel.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = compile_args
        }
    );
    
    // TODO: Compile gather-scatter and multi kernels
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

CoreCoord get_default_core() {
    return {0, 0};  // Use core (0,0) as default
}

} // namespace Spatter

#endif // USE_TENSTORRENT