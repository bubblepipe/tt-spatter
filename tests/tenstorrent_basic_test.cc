/*!
  \file tenstorrent_basic_test.cc
  Basic test for TensTorrent backend functionality
*/

#include <iostream>
#include <vector>
#include <memory>

#ifdef USE_TENSTORRENT
#include "Spatter/Configuration.hh"
#include "Spatter/TensTorrentBackend.hh"

int main() {
    try {
        std::cout << "Testing TensTorrent backend..." << std::endl;
        
        // Test device initialization
        Spatter::TensTorrentDevice device(0);
        if (!device.initialize()) {
            std::cerr << "Failed to initialize TensTorrent device" << std::endl;
            return 1;
        }
        
        std::cout << "Device info: " << device.get_device_info() << std::endl;
        std::cout << "Max memory: " << device.get_max_memory() / (1024*1024) << " MB" << std::endl;
        
        // Test buffer allocation
        const size_t test_size = 1024 * sizeof(float);
        auto test_buffer = device.allocate_buffer(test_size);
        if (!test_buffer) {
            std::cerr << "Failed to allocate test buffer" << std::endl;
            return 1;
        }
        
        std::cout << "Successfully allocated buffer of size " << test_size << " bytes" << std::endl;
        
        // Test data transfer
        std::vector<double> test_data(256);
        for (size_t i = 0; i < test_data.size(); ++i) {
            test_data[i] = static_cast<double>(i);
        }
        
        device.write_buffer(test_buffer, test_data, true);
        std::cout << "Successfully wrote test data to buffer" << std::endl;
        
        std::vector<double> read_data;
        device.read_buffer(test_buffer, read_data, true);
        std::cout << "Successfully read test data from buffer" << std::endl;
        
        // Verify data integrity
        if (read_data.size() != test_data.size()) {
            std::cerr << "Data size mismatch: expected " << test_data.size() 
                      << ", got " << read_data.size() << std::endl;
            return 1;
        }
        
        bool data_matches = true;
        for (size_t i = 0; i < test_data.size(); ++i) {
            // Allow for small floating point differences due to BFloat16 conversion
            if (std::abs(read_data[i] - test_data[i]) > 0.01) {
                std::cerr << "Data mismatch at index " << i 
                          << ": expected " << test_data[i] 
                          << ", got " << read_data[i] << std::endl;
                data_matches = false;
                break;
            }
        }
        
        if (!data_matches) {
            std::cerr << "Data verification failed" << std::endl;
            return 1;
        }
        
        std::cout << "Data verification passed" << std::endl;
        
        // Test helper functions
        size_t buffer_size = Spatter::calculate_buffer_size(1000);
        std::cout << "Calculated buffer size for 1000 elements: " << buffer_size << " bytes" << std::endl;
        
        auto core = Spatter::get_default_core();
        std::cout << "Default core coordinates: (" << core.x << ", " << core.y << ")" << std::endl;
        
        std::cout << "All TensTorrent backend tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}

#else
int main() {
    std::cout << "TensTorrent support not compiled in, skipping test" << std::endl;
    return 0;
}
#endif