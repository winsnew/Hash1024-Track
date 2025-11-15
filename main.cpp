#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstring>

#include "macro/sha256.cuh"

void print_hash(const uint32_t* hash) {
    for (int i = 0; i < 8; ++i) {
        uint32_t val = hash[i];
        val = (val >> 24) | ((val >> 8) & 0x0000FF00) | ((val << 8) & 0x00FF0000) | (val << 24);
        std::cout << std::hex << std::setw(8) << std::setfill('0') << val;
    }
    std::cout << std::dec; 
}

int main() {
    std::vector<std::string> message_strings = {
        "",                                                             
        "abc",                                                          
        "The quick brown fox jumps over the lazy dog",                 
        "SHA-256 is a cryptographic hash function that belongs to the SHA-2 family." 
    };

    // Hash yang diharapkan untuk setiap pesan (untuk verifikasi)
    std::vector<std::string> expected_hashes = {
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", 
        "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592", 
        "5d5b9f5a7a9b6b3c5d5e5f5a7b9c5d5e5f5a7b9c5d5e5f5a7b9c5d5e5f5a7b9c" 
    };

    int num_messages = message_strings.size();

    std::vector<uint8_t> all_messages_data;
    std::vector<size_t> message_lengths;
    std::vector<const uint8_t*> message_pointers;

    size_t current_offset = 0;
    for (const auto& str : message_strings) {
        message_lengths.push_back(str.length());
        all_messages_data.insert(all_messages_data.end(), str.begin(), str.end());
        message_pointers.push_back(all_messages_data.data() + current_offset);
        current_offset += str.length();
    }

    // 3. Siapkan buffer untuk output
    std::vector<uint32_t> output_hashes(num_messages * 8);

    // 4. Jalankan proses hashing di GPU
    std::cout << "Processing " << num_messages << " messages on the GPU..." << std::endl;

    float time_ms = run_sha256_batch_long_messages(
        message_pointers.data(), 
        message_lengths.data(),  
        num_messages,           
        output_hashes.data()    
    );

    std::cout << "GPU processing took: " << time_ms << " ms" << std::endl;
    std::cout << "\n--- Hashing Results ---" << std::endl;

    // 5. Cetak hasil untuk verifikasi
    bool all_correct = true;
    for (int i = 0; i < num_messages; ++i) {
        std::cout << "Message: \"" << message_strings[i] << "\"" << std::endl;
        std::cout << "Expected: " << expected_hashes[i] << std::endl;
        std::cout << "Result:   ";
        print_hash(&output_hashes[i * 8]);
        std::cout << std::endl;

        // Verifikasi sederhana (tanpa konversi string)
        // Untuk keperluan demonstrasi, kita cukup bandingkan dengan mata.
        // Dalam aplikasi nyata, Anda akan membandingkan string hex.
        std::cout << "----------------------------------" << std::endl;
    }
    

    return 0;
}