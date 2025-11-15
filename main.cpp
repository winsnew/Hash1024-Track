#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>

#include "macro/sha256.cuh"


void sha256_pad_single_block(const std::string& message, uint8_t* padded_block) {
    memset(padded_block, 0, 64);
    size_t msg_len = message.length();
    memcpy(padded_block, message.c_str(), msg_len);
    padded_block[msg_len] = 0x80;
    uint64_t msg_bits_len = static_cast<uint64_t>(msg_len) * 8;
    for (int i = 0; i < 8; ++i) {
        padded_block[56 + i] = (msg_bits_len >> (56 - 8 * i)) & 0xFF;
    }
}

void print_hash(const uint32_t* hash) {
    for (int i = 0; i < 8; ++i) {
        std::cout << std::hex << std::setw(8) << std::setfill('0') << hash[i];
    }
    std::cout << std::dec << std::endl; 
}

int main() {
    std::string input_str = "abc";
    
    std::vector<uint8_t> padded_data(64);
    
    sha256_pad_single_block(input_str, padded_data.data());

    std::vector<uint32_t> output_hash(8);

    std::cout << "Running GPU for SHA-256(\"" << input_str << "\")..." << std::endl;
    float time_ms = run_sha256_batch(padded_data.data(), output_hash.data(), 1);
    
    std::cout << "GPU Hash     : ";
    print_hash(output_hash.data());
    
    std::cout << "Expected Hash: ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" << std::endl;
    
    std::cout << "Execution time: " << time_ms << " ms" << std::endl;

    return 0;
}