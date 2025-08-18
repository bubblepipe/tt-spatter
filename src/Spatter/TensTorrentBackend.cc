/*!
  \file TensTorrentBackend.cc
*/

#include "TensTorrentBackend.hh"

#ifdef USE_TENSTORRENT

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <tt-metalium/bfloat16.hpp>

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
        device_ = tt::tt_metal::CreateDevice(device_id_);
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
        tt::tt_metal::CloseDevice(device_);
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
    
    tt::tt_metal::InterleavedBufferConfig config{
        .device = device_,
        .size = aligned_size,
        .page_size = TILE_SIZE_BYTES,
        .buffer_type = type
    };
    
    return tt::tt_metal::CreateBuffer(config);
}

void TensTorrentDevice::write_buffer(std::shared_ptr<tt::tt_metal::Buffer> buffer, 
                                     const std::vector<double>& data, bool blocking) {
    if (!initialized_) {
        throw std::runtime_error("TensTorrent device not initialized");
    }
    
    // Convert double to bfloat16 format
    std::vector<tt::tt_metal::bfloat16> tt_data;
    tt_data.reserve(data.size());
    for (double val : data) {
        tt_data.push_back(tt::tt_metal::bfloat16(static_cast<float>(val)));
    }
    
    tt::tt_metal::EnqueueWriteBuffer(*command_queue_, buffer, tt_data, blocking);
}

void TensTorrentDevice::read_buffer(std::shared_ptr<tt::tt_metal::Buffer> buffer, 
                                    std::vector<double>& data, bool blocking) {
    if (!initialized_) {
        throw std::runtime_error("TensTorrent device not initialized");
    }
    
    std::vector<tt::tt_metal::bfloat16> tt_data;
    tt::tt_metal::EnqueueReadBuffer(*command_queue_, buffer, tt_data, blocking);
    
    // Convert bfloat16 back to double
    data.clear();
    data.reserve(tt_data.size());
    for (const auto& val : tt_data) {
        data.push_back(static_cast<double>(static_cast<float>(val)));
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
    
    tt::tt_metal::CoreCoord core = get_default_core();
    tt::tt_metal::SetRuntimeArgs(gather_program_, gather_kernel_handle_, core, runtime_args);
    
    // Execute the program
    tt::tt_metal::EnqueueProgram(*command_queue_, gather_program_, false);
    tt::tt_metal::Finish(*command_queue_);
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
    gather_program_ = tt::tt_metal::CreateProgram();
    
    tt::tt_metal::CoreCoord core = get_default_core();
    
    // Create gather kernel
    std::vector<uint32_t> compile_args;
    // Add any compile-time arguments here
    
    gather_kernel_handle_ = tt::tt_metal::CreateKernel(
        gather_program_,
        "src/Spatter/kernels/gather_kernel.cpp",
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_args
        }
    );
    
    // TODO: Compile scatter and other kernels
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
    const size_t element_size = sizeof(tt::tt_metal::bfloat16);
    const size_t tile_size = 32 * 32 * element_size;
    size_t total_size = num_elements * element_size;
    
    // Round up to tile boundary
    return ((total_size + tile_size - 1) / tile_size) * tile_size;
}

tt::tt_metal::CoreCoord get_default_core() {
    return {0, 0};  // Use core (0,0) as default
}

} // namespace Spatter

#endif // USE_TENSTORRENT