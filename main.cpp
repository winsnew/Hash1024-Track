#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>

#include "macro/sha256.cuh"


void sha256_pad_single_block(const uint8_t* message, size_t message_len, uint8_t* padded_block) {
    memcpy(padded_block, message, message_len);
    padded_block[message_len] = 0x80;
    memset(padded_block + message_len + 1, 0, 64 - message_len - 1 - 8);

    uint64_t message_len_bits = (uint64_t)message_len * 8;
    for (int i = 0; i < 8; ++i) {
        padded_block[63 - i] = (message_len_bits >> (i * 8)) & 0xFF;
    }
}

int main() {
    const char* message_str = "0390f24331aec7341074cd4058dc69c686fb638cd041e0aa479f78f4fed8058700";
    size_t message_len = strlen(message_str);

    std::vector<uint8_t> input_padded(64);
    
    sha256_pad_single_block(reinterpret_cast<const uint8_t*>(message_str), message_len, input_padded.data());
    std::vector<uint32_t> output_hash(8);

    float time = run_sha256_batch(input_padded.data(), output_hash.data(), 1);

    std::cout << "Running GPU for SHA-256(\"" << message_str << "\")..." << std::endl;
    std::cout << "GPU Hash     : ";
    for (int i = 0; i < 8; ++i) {
        printf("%08x", output_hash[i]);
    }
    std::cout << std::endl;
    
    const char* expected_hash = "17969b7d1bdd29f9413adefeaac697eb219817c328afd94037bf2faf96e90f4c";
    std::cout << "Expected Hash: " << expected_hash << std::endl;
    std::cout << "Time: " << time << " ms" << std::endl;

    return 0;
}