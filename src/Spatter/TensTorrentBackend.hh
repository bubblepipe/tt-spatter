/*!
  \file TensTorrentBackend.hh
*/

#ifndef SPATTER_TENSTORRENT_BACKEND_HH
#define SPATTER_TENSTORRENT_BACKEND_HH

#ifdef USE_TENSTORRENT

#include <memory>
#include <vector>
#include <string>
#include <map>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/bfloat16.hpp>

// Forward declaration for aligned_vector - definition is in Configuration.hh
template <typename T, size_t Alignment>
class aligned_allocator;

template <typename T>
using aligned_vector = std::vector<T, aligned_allocator<T, 64>>;

namespace Spatter {

class TensTorrentDevice {
public:
    TensTorrentDevice(int device_id = 0, int num_cores = 0);
    ~TensTorrentDevice();
    
    // Device management
    bool initialize();
    void cleanup();
    bool is_initialized() const { return initialized_; }
    
    // Core management
    void discover_cores();
    std::vector<CoreCoord> get_active_cores() const { return active_cores_; }
    
    // Memory management
    std::shared_ptr<tt::tt_metal::Buffer> allocate_buffer(size_t size_bytes, 
                                                          tt::tt_metal::BufferType type = tt::tt_metal::BufferType::DRAM);
    // Method removed - use allocate_buffer directly
    
    // Data transfer
    void write_buffer(std::shared_ptr<tt::tt_metal::Buffer> buffer, 
                      const std::vector<double>& data, bool blocking = true);
    void read_buffer(std::shared_ptr<tt::tt_metal::Buffer> buffer, 
                     std::vector<double>& data, bool blocking = true);
                     
    // Only keep essential aligned_vector methods needed by Configuration
    void readBuffer(std::shared_ptr<tt::tt_metal::Buffer> buffer, 
                    aligned_vector<double>& data, bool blocking = true);
    void writeBuffer(std::shared_ptr<tt::tt_metal::Buffer> buffer, 
                     const std::vector<uint32_t>& data, bool blocking = true);
    void writeBuffer(std::shared_ptr<tt::tt_metal::Buffer> buffer, 
                     const aligned_vector<double>& data, bool blocking = true);
    void writeBuffer(std::shared_ptr<tt::tt_metal::Buffer> buffer, 
                     const aligned_vector<size_t>& data, bool blocking = true);
    
    // Kernel execution - only keep working methods
    bool executeGatherKernel(
        std::shared_ptr<tt::tt_metal::Buffer> src_buffer,
        std::shared_ptr<tt::tt_metal::Buffer> dst_buffer,
        std::shared_ptr<tt::tt_metal::Buffer> pattern_buffer,
        uint32_t num_elements,
        uint32_t delta);
        
    bool executeScatterKernel(
        std::shared_ptr<tt::tt_metal::Buffer> src_buffer,
        std::shared_ptr<tt::tt_metal::Buffer> dst_buffer,
        std::shared_ptr<tt::tt_metal::Buffer> pattern_buffer,
        uint32_t num_elements,
        uint32_t delta);
    
    // Device information
    std::string get_device_info() const;
    size_t get_max_memory() const;
    
private:
    int device_id_;
    bool initialized_;
    int num_cores_;
    tt::tt_metal::IDevice* device_;
    tt::tt_metal::CommandQueue* command_queue_;
    std::vector<CoreCoord> active_cores_;
    CoreCoord compute_grid_size_;
    CoreCoord effective_grid_size_;  // Limited by user's --tt-cores parameter
    
    // Single-kernel programs (created on-demand like loopback example)
    
    // Single-kernel handles (created on-demand like loopback example)
    
    // Removed legacy kernel handles
    
    // Single-kernel approach - no lazy initialization needed
    
    // Buffer size tracking for reads
    std::map<std::shared_ptr<tt::tt_metal::Buffer>, size_t> buffer_sizes_;
    
    // Helper methods
    void compile_kernels();
    // Single-kernel approach - no complex initialization needed
        
    size_t align_to_tile_size(size_t size) const;
    
    // Constants
    static constexpr size_t TILE_WIDTH = 32;
    static constexpr size_t TILE_HEIGHT = 32;
    static constexpr size_t TILE_SIZE_BYTES = TILE_WIDTH * TILE_HEIGHT * sizeof(bfloat16);
    static constexpr size_t DRAM_ALIGNMENT = 64; // Blackhole requires 64B alignment
};

// Helper functions for TensTorrent backend
void check_tt_error(const std::string& operation);
size_t calculate_buffer_size(size_t num_elements);

} // namespace Spatter

#endif // USE_TENSTORRENT

#endif // SPATTER_TENSTORRENT_BACKEND_HH