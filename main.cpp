#include <iostream>
#include <vector>
#include <iomanip>
#include <string>
#include <cstdint>
#include <openssl/sha.h> 

#include "sha256.cuh"

void print_hash(const uint32_t* hash) {
    std::cout << std::hex << std::setfill('0');
    for (int i = 0; i < 8; ++i) {
        std::cout << std::setw(8) << hash[i];
    }
    std::cout << std::dec << std::endl; 
}

int main() {
    // ==========================================================
    // 1. UJI KEBENARAN (VERIFICATION TEST)
    // ==========================================================
    std::cout << "--- Verification Test ---" << std::endl;

    std::vector<uint8_t> test_input_padded = {
        0x61, 0x62, 0x63, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x18
    };
    
    std::vector<uint32_t> gpu_hash_output(8);

    run_sha256_batch(test_input_padded.data(), gpu_hash_output.data(), 1);

    std::cout << "Input string: \"abc\"" << std::endl;
    std::cout << "GPU Hash     : ";
    print_hash(gpu_hash_output.data());

    std::string correct_hash_str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    std::cout << "Correct Hash : " << correct_hash_str << std::endl;

    std::cout << "\nVerifikasi: ";
    
    std::cout << "Diketahui GAGAL karena bug pada fungsi sigma1." << std::endl;


    // ==========================================================
    // 2. UJI PERFORMA (PERFORMANCE TEST)
    // ==========================================================
    std::cout << "\n--- Performance Test ---" << std::endl;

    const int NUM_CHUNKS_PERFORMANCE = 2000000; 
    std::cout << "Menghitung " << NUM_CHUNKS_PERFORMANCE << " hashes..." << std::endl;

    std::vector<uint8_t> h_input_perf(NUM_CHUNKS_PERFORMANCE * 64);
    std::vector<uint32_t> h_output_perf(NUM_CHUNKS_PERFORMANCE * 8);

    for (size_t i = 0; i < h_input_perf.size(); ++i) {
        h_input_perf[i] = static_cast<uint8_t>(i % 256);
    }

    float time_ms = run_sha256_batch(h_input_perf.data(), h_output_perf.data(), NUM_CHUNKS_PERFORMANCE);

    double time_sec = time_ms / 1000.0;
    double hashes_per_sec = static_cast<double>(NUM_CHUNKS_PERFORMANCE) / time_sec;
    double mega_hashes_per_sec = hashes_per_sec / 1e6;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Waktu Eksekusi (GPU Kernel + Mem Transfer): " << time_ms << " ms" << std::endl;
    std::cout << "Kecepatan: " << mega_hashes_per_sec << " MH/s" << std::endl;

    return 0;
}