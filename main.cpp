#include <iostream>
#include <vector>
#include <cstdint>
#include <iomanip>
#include <random>
#include <string>
#include <cstring>

#include "macro/sha256.cuh" 

void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(result));
        exit(1);
    }
}

void pad_message(const std::string& message, std::vector<uint8_t>& padded_chunk) {
    padded_chunk.resize(64, 0);
    size_t len = message.length();
    memcpy(padded_chunk.data(), message.c_str(), len);
    padded_chunk[len] = 0x80;
    uint64_t message_len_bits = len * 8;
    for (int i = 0; i < 8; ++i) {
        padded_chunk[56 + i] = (message_len_bits >> (56 - 8 * i)) & 0xFF;
    }
}

int main() {
    // --- Batch Data ---
    const int NUM_HASHES = 4000000; 
    std::cout << "Mempersiapkan " << NUM_HASHES << " pesan..." << std::endl;
    std::vector<uint8_t> h_input(NUM_HASHES * 64);
    std::vector<uint32_t> h_output(NUM_HASHES * 8);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 55);

    for (int i = 0; i < NUM_HASHES; ++i) {
        std::string msg(dis(gen), 'a');
        std::vector<uint8_t> padded;
        pad_message(msg, padded);
        memcpy(&h_input[i * 64], padded.data(), 64);
    }
    std::cout << "Persiapan data selesai." << std::endl;

    std::cout << "Memulai perhitungan..." << std::endl;
    float elapsed_ms = run_sha256_batch(h_input.data(), h_output.data(), NUM_HASHES);

    double seconds = elapsed_ms / 1000.0;
    std::cout << "Perhitungan selesai." << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Total waktu untuk " << NUM_HASHES << " hash: " << seconds << " detik" << std::endl;
    std::cout << "Throughput: " << (NUM_HASHES / seconds) / 1000000.0 << " Juta Hash/detik" << std::endl;

    std::string test_msg = "abc";
    std::vector<uint8_t> test_padded;
    pad_message(test_msg, test_padded);
    std::vector<uint32_t> test_hash(8);
    
    bool found = false;
    for(int i=0; i<NUM_HASHES; ++i) {
        if(memcmp(&h_input[i*64], test_padded.data(), 64) == 0) {
            memcpy(test_hash.data(), &h_output[i*8], 8 * sizeof(uint32_t));
            found = true;
            break;
        }
    }

    if (found) {
        std::cout << "SHA-256 Hash: ";
        for (int i = 0; i < 8; ++i) {
            std::cout << std::hex << std::setw(8) << std::setfill('0') << test_hash[i];
        }
        std::cout << std::dec << std::endl;
    } else {
        std::vector<uint8_t> abc_input(64);
        std::vector<uint32_t> abc_output(8);
        pad_message("abc", abc_input);
        run_sha256_batch(abc_input.data(), abc_output.data(), 1);
        std::cout << "SHA-256 Hash: ";
        for (int i = 0; i < 8; ++i) {
            std::cout << std::hex << std::setw(8) << std::setfill('0') << abc_output[i];
        }
        std::cout << std::dec << std::endl;
    }

    return 0;
}