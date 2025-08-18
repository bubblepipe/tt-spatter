/*!
  \file TensTorrentBackend.hh
*/

#ifndef SPATTER_TENSTORRENT_BACKEND_HH
#define SPATTER_TENSTORRENT_BACKEND_HH

#ifdef USE_TENSTORRENT

#include <memory>
#include <vector>
#include <string>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>

namespace Spatter {

class TensTorrentDevice {
public:
    TensTorrentDevice(int device_id = 0);
    ~TensTorrentDevice();
    
    // Device management
    bool initialize();
    void cleanup();
    bool is_initialized() const { return initialized_; }
    
    // Memory management
    std::shared_ptr<tt::tt_metal::Buffer> allocate_buffer(size_t size_bytes, 
                                                          tt::tt_metal::BufferType type = tt::tt_metal::BufferType::DRAM);
    
    // Data transfer
    void write_buffer(std::shared_ptr<tt::tt_metal::Buffer> buffer, 
                      const std::vector<double>& data, bool blocking = true);
    void read_buffer(std::shared_ptr<tt::tt_metal::Buffer> buffer, 
                     std::vector<double>& data, bool blocking = true);
    
    // Kernel execution
    void execute_gather_kernel(
        std::shared_ptr<tt::tt_metal::Buffer> src_buffer,
        std::shared_ptr<tt::tt_metal::Buffer> dst_buffer,
        std::shared_ptr<tt::tt_metal::Buffer> pattern_buffer,
        size_t num_elements);
        
    void execute_scatter_kernel(
        std::shared_ptr<tt::tt_metal::Buffer> src_buffer,
        std::shared_ptr<tt::tt_metal::Buffer> dst_buffer,
        std::shared_ptr<tt::tt_metal::Buffer> pattern_buffer,
        size_t num_elements);
    
    // Device information
    std::string get_device_info() const;
    size_t get_max_memory() const;
    
private:
    int device_id_;
    bool initialized_;
    tt::tt_metal::IDevice* device_;
    tt::tt_metal::CommandQueue* command_queue_;
    
    // Kernel programs
    tt::tt_metal::Program gather_program_;
    tt::tt_metal::Program scatter_program_;
    tt::tt_metal::KernelHandle gather_kernel_handle_;
    tt::tt_metal::KernelHandle scatter_kernel_handle_;
    
    // Helper methods
    void compile_kernels();
    size_t align_to_tile_size(size_t size) const;
    std::vector<float> convert_double_to_bfloat16(const std::vector<double>& input) const;
    std::vector<double> convert_bfloat16_to_double(const std::vector<float>& input) const;
    
    // Constants
    static constexpr size_t TILE_WIDTH = 32;
    static constexpr size_t TILE_HEIGHT = 32;
    static constexpr size_t TILE_SIZE_BYTES = TILE_WIDTH * TILE_HEIGHT * sizeof(bfloat16);
    static constexpr size_t DRAM_ALIGNMENT = 64; // Blackhole requires 64B alignment
};

// Helper functions for TensTorrent backend
void check_tt_error(const std::string& operation);
size_t calculate_buffer_size(size_t num_elements);
tt::tt_metal::CoreCoord get_default_core();

} // namespace Spatter

#endif // USE_TENSTORRENT

#endif // SPATTER_TENSTORRENT_BACKEND_HH