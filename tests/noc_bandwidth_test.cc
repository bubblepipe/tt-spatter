/*!
  \file noc_bandwidth_test.cc
  NOC bandwidth saturation test between neighboring Tensix cores
*/

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

#ifdef USE_TENSTORRENT
#include "Spatter/Configuration.hh"
#include "Spatter/TensTorrentBackend.hh"

class NOCBandwidthTest {
private:
    Spatter::TensTorrentDevice device_;
    static constexpr size_t TILE_SIZE_BYTES = 32 * 32 * 2; // 2048 bytes per tile
    static constexpr size_t L1_SIZE = 1464 * 1024; // ~1.4MB L1 SRAM
    static constexpr size_t MAX_TILES_PER_CORE = L1_SIZE / TILE_SIZE_BYTES / 2; // Leave space for staging

public:
    NOCBandwidthTest() : device_(0) {}

    bool initialize() {
        if (!device_.initialize()) {
            std::cerr << "Failed to initialize TensTorrent device" << std::endl;
            return false;
        }
        std::cout << "TensTorrent device initialized successfully" << std::endl;
        return true;
    }

    void run_bandwidth_test(size_t num_tiles = 100) {
        std::cout << "\n=== NOC Bandwidth Saturation Test ===" << std::endl;
        std::cout << "Tiles to transfer: " << num_tiles << std::endl;
        std::cout << "Bytes per tile: " << TILE_SIZE_BYTES << std::endl;
        std::cout << "Total data: " << (num_tiles * TILE_SIZE_BYTES) / 1024 << " KB" << std::endl;

        try {
            // Test between adjacent cores (0,0) -> (0,1)
            test_core_pair(0, 0, 0, 1, num_tiles);
            
            // Test between cores with distance 2: (0,0) -> (0,2)
            test_core_pair(0, 0, 0, 2, num_tiles);
            
            // Test diagonal: (0,0) -> (1,1)
            test_core_pair(0, 0, 1, 1, num_tiles);
            
        } catch (const std::exception& e) {
            std::cerr << "Test failed: " << e.what() << std::endl;
        }
    }

private:
    void test_core_pair(uint32_t src_x, uint32_t src_y, uint32_t dest_x, uint32_t dest_y, size_t num_tiles) {
        std::cout << "\n--- Testing: Core(" << src_x << "," << src_y << ") -> Core(" 
                  << dest_x << "," << dest_y << ") ---" << std::endl;

        // Calculate NOC distance
        uint32_t noc_hops = std::abs((int)dest_x - (int)src_x) + std::abs((int)dest_y - (int)src_y);
        std::cout << "NOC hops: " << noc_hops << std::endl;

        // Allocate buffers
        size_t buffer_size = num_tiles * TILE_SIZE_BYTES;
        auto src_buffer = device_.allocate_buffer(buffer_size);
        auto dest_buffer = device_.allocate_buffer(buffer_size);

        if (!src_buffer || !dest_buffer) {
            std::cerr << "Failed to allocate buffers" << std::endl;
            return;
        }

        // Initialize source data
        std::vector<double> test_data(num_tiles * TILE_SIZE_BYTES / sizeof(double));
        for (size_t i = 0; i < test_data.size(); ++i) {
            test_data[i] = static_cast<double>(i % 256); // Repeating pattern
        }

        device_.write_buffer(src_buffer, test_data, true);

        // Compile and run NOC bandwidth kernel
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Execute NOC bandwidth kernel
        device_.execute_noc_bandwidth_kernel(src_buffer, dest_buffer, num_tiles, dest_x, dest_y);

        auto end_time = std::chrono::high_resolution_clock::now();
        
        // Calculate performance metrics
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double time_seconds = duration.count() / 1e6;
        double bytes_transferred = buffer_size;
        double bandwidth_mbps = bytes_transferred / time_seconds / 1e6; // MB/s
        double bandwidth_gbps = bandwidth_mbps / 1000.0; // GB/s

        // Theoretical NOC bandwidth (estimated)
        double theoretical_noc_bw_gbps = 50.0; // Estimated NOC bandwidth per direction
        double efficiency = (bandwidth_gbps / theoretical_noc_bw_gbps) * 100.0;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Transfer time: " << time_seconds * 1000 << " ms" << std::endl;
        std::cout << "Bandwidth: " << bandwidth_mbps << " MB/s (" << bandwidth_gbps << " GB/s)" << std::endl;
        std::cout << "NOC efficiency: " << efficiency << "%" << std::endl;
        std::cout << "Latency per tile: " << (time_seconds * 1e6) / num_tiles << " μs" << std::endl;

        // Verify data integrity
        std::vector<double> read_data;
        device_.read_buffer(dest_buffer, read_data, true);
        
        bool data_correct = true;
        size_t mismatches = 0;
        for (size_t i = 0; i < std::min(test_data.size(), read_data.size()); ++i) {
            if (std::abs(test_data[i] - read_data[i]) > 0.01) {
                mismatches++;
                if (mismatches <= 5) { // Only report first few mismatches
                    std::cout << "Data mismatch at index " << i 
                              << ": expected " << test_data[i] 
                              << ", got " << read_data[i] << std::endl;
                }
                data_correct = false;
            }
        }

        if (data_correct) {
            std::cout << "✓ Data integrity verified" << std::endl;
        } else {
            std::cout << "✗ Data integrity failed (" << mismatches << " mismatches)" << std::endl;
        }
    }
};

int main() {
    try {
        std::cout << "TensTorrent NOC Bandwidth Saturation Test" << std::endl;
        
        NOCBandwidthTest test;
        if (!test.initialize()) {
            return 1;
        }

        // Run tests with different data sizes
        std::vector<size_t> test_sizes = {10, 50, 100, 200, 500}; // Number of tiles
        
        for (size_t tiles : test_sizes) {
            test.run_bandwidth_test(tiles);
            std::cout << "\n" << std::string(60, '=') << std::endl;
        }

        std::cout << "\nNOC Bandwidth Test Complete!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}

#else
int main() {
    std::cout << "TensTorrent support not compiled in, skipping NOC bandwidth test" << std::endl;
    return 0;
}
#endif