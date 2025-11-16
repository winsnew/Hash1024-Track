#include "macro/sha256.cuh"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>

void print_hash(const uint32_t* hash) {
    for (int i = 0; i < 8; i++) {
        std::cout << std::hex << std::setw(8) << std::setfill('0') << hash[i];
    }
    std::cout << std::dec << std::endl;
}

void test_single_message() {
    std::cout << "=== SHA256 GPU Single Message Test ===" << std::endl;
    
    const char* message = "abc";
    size_t message_len = 3;
    
    uint32_t hash[8];
    
    auto start = std::chrono::high_resolution_clock::now();
    sha256_gpu(reinterpret_cast<const uint8_t*>(message), message_len, hash);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Pesan: \"" << message << "\"" << std::endl;
    std::cout << "SHA256: ";
    print_hash(hash);
    std::cout << "Waktu: " << duration.count() << " mikrodetik" << std::endl;
    std::cout << std::endl;
}

void test_batch_performance() {
    std::cout << "=== SHA256 GPU Batch Performance Test ===" << std::endl;
    
    const size_t num_messages = 1000000; 
    const size_t message_len = 3;
    const char* base_message = "abc";
    
    std::vector<uint8_t> messages(num_messages * message_len);
    std::vector<uint32_t> hashes(num_messages * 8);
    
    for (size_t i = 0; i < num_messages; i++) {
        memcpy(&messages[i * message_len], base_message, message_len);
    }
    
    SHA256GPU sha256_gpu_handler(num_messages);
    
    auto start = std::chrono::high_resolution_clock::now();
    sha256_gpu_handler.compute_batch(messages.data(), message_len, num_messages, hashes.data());
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double seconds = duration.count() / 1000000.0;
    
    std::cout << "Contoh hash pertama: ";
    print_hash(hashes.data());
    
    double hashes_per_second = num_messages / seconds;
    double mhashes_per_second = hashes_per_second / 1000000.0;
    
    std::cout << "Jumlah pesan: " << num_messages << std::endl;
    std::cout << "Waktu total: " << seconds << " detik" << std::endl;
    std::cout << "Kecepatan: " << hashes_per_second << " H/s" << std::endl;
    std::cout << "Kecepatan: " << std::fixed << std::setprecision(2) << mhashes_per_second << " MH/s" << std::endl;
    std::cout << std::endl;
}

void test_different_message_lengths() {
    std::cout << "=== Test Panjang Pesan Berbeda ===" << std::endl;
    
    const char* test_messages[] = {
        "a",
        "ab",
        "abc",
        "abcd",
        "abcde",
        "Hello World!",
        "CUDA SHA256 Implementation"
    };
    
    const int num_tests = sizeof(test_messages) / sizeof(test_messages[0]);
    
    for (int i = 0; i < num_tests; i++) {
        const char* msg = test_messages[i];
        size_t len = strlen(msg);
        uint32_t hash[8];
        
        sha256_gpu(reinterpret_cast<const uint8_t*>(msg), len, hash);
        
        std::cout << "Pesan: \"" << msg << "\"" << std::endl;
        std::cout << "Hash:  ";
        print_hash(hash);
    }
}

int main() {
    try {
        std::cout << "SHA256 GPU Implementation Test" << std::endl;
        std::cout << "=================================" << std::endl << std::endl;
        
        test_single_message();
        
        test_batch_performance();
        
        // test_different_message_lengths();
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}