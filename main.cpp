#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include "sha256.cuh"

void print_hash(const uint32_t* hash) {
    std::cout << std::hex << std::setfill('0');
    for (int i = 0; i < 8; ++i) {
        std::cout << std::setw(8) << hash[i];
    }
    std::cout << std::dec << std::endl;
}

int main() {
    std::vector<std::string> test_messages = {
        "", 
        "abc",
        "message digest",
        "The quick brown fox jumps over the lazy dog",
        "hello world"
    };

    std::vector<std::string> expected_hashes = {
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "f96b697d7cb7938d525a2f31aaf161d0b2e65f2583120b2105c8d0c8a3b545a7",
        "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb7620f65c7b6f1b76b1",
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    };

    std::cout << "Testing SHA-256 GPU implementation..." << std::endl;
    std::cout << "======================================" << std::endl;

    for (size_t i = 0; i < test_messages.size(); ++i) {
        const uint8_t* msg_ptr = reinterpret_cast<const uint8_t*>(test_messages[i].c_str());
        size_t msg_len = test_messages[i].length();
        uint32_t computed_hash[8];

        sha256_gpu(msg_ptr, msg_len, computed_hash);

        std::cout << "Input: \"" << test_messages[i] << "\"" << std::endl;
        std::cout << "Computed: ";
        print_hash(computed_hash);
        std::cout << "Expected: " << expected_hashes[i] << std::endl;
        
        std::stringstream ss;
        ss << std::hex << std::setfill('0');
        for(int j=0; j<8; ++j) ss << std::setw(8) << computed_hash[j];
        
        if (ss.str() == expected_hashes[i]) {
            std::cout << "Status: PASSED" << std::endl;
        } else {
            std::cout << "Status: FAILED" << std::endl;
        }
        std::cout << "--------------------------------------" << std::endl;
    }

    return 0;
}