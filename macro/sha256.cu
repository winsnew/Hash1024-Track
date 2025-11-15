#include "./sha256.cuh"
#include <iostream>
#include <cstdio>   
#include <cstdlib>  

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

__global__ void sha256_gpu_kernel_optimized(const uint8_t* __restrict__ input_chunks, uint32_t* __restrict__ output_hashes, int num_chunks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_chunks) return;

    const uint8_t* global_chunk_ptr = &input_chunks[idx * 64];
    uint32_t w[64];

    #pragma unroll
    for (int t = 0; t < 16; ++t) {
        w[t] = ( (uint32_t)global_chunk_ptr[t*4] << 24 ) |
               ( (uint32_t)global_chunk_ptr[t*4+1] << 16 ) |
               ( (uint32_t)global_chunk_ptr[t*4+2] << 8 ) |
               ( (uint32_t)global_chunk_ptr[t*4+3] );
    }
    
    #pragma unroll
    for (int t = 16; t < 64; ++t) w[t] = sigma1(w[t - 2]) + w[t - 7] + sigma0(w[t - 15]) + w[t - 16];

    uint32_t a = initial_h[0], b = initial_h[1], c = initial_h[2], d = initial_h[3];
    uint32_t e = initial_h[4], f = initial_h[5], g = initial_h[6], h_val = initial_h[7];

    #pragma unroll
    for (int t = 0; t < 64; ++t) {
        uint32_t t1 = h_val + Sigma1(e) + Ch(e, f, g) + k[t] + w[t];
        uint32_t t2 = Sigma0(a) + Maj(a, b, c);
        h_val = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    uint32_t* out_ptr = &output_hashes[idx * 8];
    out_ptr[0] = a + initial_h[0]; out_ptr[1] = b + initial_h[1];
    out_ptr[2] = c + initial_h[2]; out_ptr[3] = d + initial_h[3];
    out_ptr[4] = e + initial_h[4]; out_ptr[5] = f + initial_h[5];
    out_ptr[6] = g + initial_h[6]; out_ptr[7] = h_val + initial_h[7];
}

float run_sha256_batch(const uint8_t* input_chunks, uint32_t* output_hashes, int num_chunks) {
    const int NUM_STREAMS = 4;
    const int BATCH_SIZE = (num_chunks + NUM_STREAMS - 1) / NUM_STREAMS;

    uint8_t* d_input;
    uint32_t* d_output;
    size_t input_size = num_chunks * 64 * sizeof(uint8_t);
    size_t output_size = num_chunks * 8 * sizeof(uint32_t);
    
    CUDA_CHECK(cudaMalloc(&d_input, input_size));
    CUDA_CHECK(cudaMalloc(&d_output, output_size));

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) CUDA_CHECK(cudaStreamCreate(&streams[i]));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < NUM_STREAMS; ++i) {
        int offset = i * BATCH_SIZE;
        if (offset >= num_chunks) break;
        
        int current_batch_size = BATCH_SIZE;
        if (offset + BATCH_SIZE > num_chunks) {
            current_batch_size = num_chunks - offset;
        }

        size_t batch_input_bytes = current_batch_size * 64 * sizeof(uint8_t);
        size_t batch_output_bytes = current_batch_size * 8 * sizeof(uint32_t);

        CUDA_CHECK(cudaMemcpyAsync(&d_input[offset * 64], &input_chunks[offset * 64], batch_input_bytes, cudaMemcpyHostToDevice, streams[i]));
        
        int threads_per_block = 256;
        int blocks_per_grid = (current_batch_size + threads_per_block - 1) / threads_per_block;
        sha256_gpu_kernel_optimized<<<blocks_per_grid, threads_per_block, 0, streams[i]>>>
            (&d_input[offset * 64], &d_output[offset * 8], current_batch_size);
        CUDA_CHECK(cudaGetLastError()); 

        CUDA_CHECK(cudaMemcpyAsync(&output_hashes[offset * 8], &d_output[offset * 8], batch_output_bytes, cudaMemcpyDeviceToHost, streams[i]));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    for (int i = 0; i < NUM_STREAMS; ++i) CUDA_CHECK(cudaStreamDestroy(streams[i]);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return milliseconds;
}