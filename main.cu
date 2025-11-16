#include "macro/sha256.cuh"
#include <chrono>
#include <random>
#include <iomanip>
#include <cstdint>
#include <cstring>

class Benchmark {
public:
    static void run_performance_test() {
        std::cout << "=== SHA256 GPU Extreme Performance Benchmark ===\n";
        
        const size_t num_tests = 1000000; 
        const size_t message_len = 64;    
        
        std::vector<uint8_t> test_data(num_tests * message_len);
        std::vector<uint32_t> hashes(num_tests * 8);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 255);
        
        for (auto& byte : test_data) {
            byte = static_cast<uint8_t>(dis(gen));
        }
        
        // Warmup GPU
        SHA256GPU hasher(num_tests);
        hasher.compute_batch(test_data.data(), message_len, 1000, hashes.data());
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        
        hasher.compute_batch(test_data.data(), message_len, num_tests, hashes.data());
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double seconds = duration.count() / 1000000.0;
        double hashes_per_second = num_tests / seconds;
        
        std::cout << "Processed " << num_tests << " hashes in " 
                  << seconds << " seconds\n";
        std::cout << "Performance: " << std::fixed << std::setprecision(2) 
                  << (hashes_per_second / 1000000.0) << " MH/s\n";
        std::cout << "Average time per hash: " << (duration.count() * 1000.0 / num_tests) 
                  << " ns\n";
    }
};

int main() {
    Benchmark::run_performance_test();
    return 0;
}