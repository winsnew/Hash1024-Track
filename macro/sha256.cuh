#ifndef SHA256_CUH
#define SHA256_CUH

#include <cstdint>
#include <cuda_runtime.h>

/**
 * @brief GPU-accelerated SHA256 implementation
 * 
 * This library provides high-performance SHA256 hashing using CUDA
 * with support for both single messages and batch processing.
 */

void sha256_gpu_batch(const uint8_t* messages, size_t message_len, uint32_t num_messages, uint32_t* hashes, cudaStream_t stream = 0);
void sha256_gpu(const uint8_t* message, size_t message_len, uint32_t* hash_output);

class SHA256GPU {
private:
    uint8_t* d_messages;
    uint32_t* d_hashes;
    size_t max_batch_size;
    size_t max_message_len;
    cudaStream_t stream;

public:
    SHA256GPU(size_t max_batch = 1000000, size_t max_msg_len = 128);
    ~SHA256GPU();
    
    void compute_batch(const uint8_t* messages, size_t message_len, uint32_t num_messages, uint32_t* hashes);
    void compute_batch_async(const uint8_t* messages, size_t message_len, uint32_t num_messages, uint32_t* hashes);
    void synchronize();
};

#endif // SHA256_CUH