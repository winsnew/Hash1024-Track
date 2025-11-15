#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>

#include "macro/sha256.cuh"


std::vector<uint8_t> sha256_pad_message(const std::string& message) {
    std::vector<uint8_t> padded_data;
    uint64_t msg_len_bits = message.length() * 8;
    padded_data.insert(padded_data.end(), message.begin(), message.end());
    padded_data.push_back(0x80);

    while (padded_data.size() % 64 != 56) {
        padded_data.push_back(0x00);
    }

    for (int i = 7; i >= 0; --i) {
        padded_data.push_back((msg_len_bits >> (i * 8)) & 0xFF);
    }

    return padded_data;
}

int main() {
    std::string input_str = "bf7413e8df4d039a6800aaf48383c0d75d596879";
    std::vector<uint8_t> padded_input = sha256_pad_message(input_str);
    int num_chunks = padded_input.size() / 64;
    std::vector<uint32_t> output_hashes(num_chunks * 8);

    std::cout << "Running GPU for SHA-256(\"" << input_str << "\")..." << std::endl;
    float time = run_sha256_batch(padded_input.data(), output_hashes.data(), num_chunks);

    std::cout << "GPU Hash     : ";
    for (int i = 0; i < 8; ++i) {
        printf("%08x", output_hashes[i]);
    }
    std::cout << std::endl;
    
    std::cout << "Expected Hash: ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" << std::endl;
    std::cout << "Time: " << time << " ms" << std::endl;

    return 0;
}