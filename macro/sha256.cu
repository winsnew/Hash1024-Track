#include "./sha256.cuh"
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstring> 


#define CUDA_CHECK(err) { \
    cudaError_t error = err; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(error), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define ROTR(a,b) (((a) >> (b)) | ((a) << (32-(b))))

__constant__ uint32_t k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__constant__ uint32_t initial_h[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

__device__ uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
__device__ uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
__device__ uint32_t Sigma0(uint32_t x) { return ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22); }
__device__ uint32_t Sigma1(uint32_t x) { return ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25); }
__device__ uint32_t sigma0(uint32_t x) { return ROTR(x, 7) ^ ROTR(x, 18) ^ (x >> 3); }
__device__ uint32_t sigma1(uint32_t x) { return ROTR(x, 17) ^ ROTR(x, 19) ^ (x >> 10); }

/**
 * @brief 
 * 
 * 
 * 
 * 
 * @param data_chunk 
 * @param current_hash_state 
 */
__global__ void sha256_gpu_kernel_compression(const uint8_t* __restrict__ data_chunk, uint32_t* __restrict__ current_hash_state) { 
    uint32_t w[64];
    
    #pragma unroll
    for (int t = 0; t < 16; ++t) {
        
        w[t] = (uint32_t)data_chunk[t * 4] << 24 |
               (uint32_t)data_chunk[t * 4 + 1] << 16 |
               (uint32_t)data_chunk[t * 4 + 2] << 8 |
               (uint32_t)data_chunk[t * 4 + 3];
    }
    
    #pragma unroll
    for (int t = 16; t < 64; ++t) {
        w[t] = sigma1(w[t - 2]) + w[t - 7] + sigma0(w[t - 15]) + w[t - 16];
    }

    
    uint32_t a = current_hash_state[0], b = current_hash_state[1], c = current_hash_state[2], d = current_hash_state[3];
    uint32_t e = current_hash_state[4], f = current_hash_state[5], g = current_hash_state[6], h_val = current_hash_state[7];

    #pragma unroll
    for (int t = 0; t < 64; ++t) {
        uint32_t t1 = h_val + Sigma1(e) + Ch(e, f, g) + k[t] + w[t];
        uint32_t t2 = Sigma0(a) + Maj(a, b, c);
        h_val = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    
    current_hash_state[0] += a; current_hash_state[1] += b;
    current_hash_state[2] += c; current_hash_state[3] += d;
    current_hash_state[4] += e; current_hash_state[5] += f;
    current_hash_state[6] += g; current_hash_state[7] += h_val;
}

void sha256_gpu(const uint8_t* message, size_t message_len, uint32_t* hash_output) {
    if (message == nullptr || hash_output == nullptr) {
        throw std::invalid_argument("Input pointers cannot be null.");
    }

    
    size_t num_blocks = (message_len + 8 + 63) / 64; 
    std::vector<uint8_t> padded_message(num_blocks * 64, 0);

    memcpy(padded_message.data(), message, message_len);

    padded_message[message_len] = 0x80;

    uint64_t message_len_bits = (uint64_t)message_len * 8;
    for (int i = 0; i < 8; ++i) {
        padded_message[num_blocks * 64 - 8 + i] = (message_len_bits >> (56 - 8 * i)) & 0xFF;
    }

    uint8_t* d_chunk;
    uint32_t* d_hash_state;
    CUDA_CHECK(cudaMalloc(&d_chunk, 64 * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_hash_state, 8 * sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(d_hash_state, initial_h, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    for (size_t i = 0; i < num_blocks; ++i) {
        
        CUDA_CHECK(cudaMemcpy(d_chunk, &padded_message[i * 64], 64 * sizeof(uint8_t), cudaMemcpyHostToDevice));
        
        sha256_gpu_kernel_compression<<<1, 1>>>(d_chunk, d_hash_state);
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaMemcpy(hash_output, d_hash_state, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_chunk));
    CUDA_CHECK(cudaFree(d_hash_state));
}