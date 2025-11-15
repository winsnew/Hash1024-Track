#include "./sha256.cuh"
#include <iostream>
#include <vector>
#include <cstring> 

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
__device__ uint32_t sigma0(uint32_t x) { return ROTR(x, 7) ^ ROTR(x, 18) & (x >> 3); }
__device__ uint32_t sigma1(uint32_t x) { return ROTR(x, 17) ^ ROTR(x, 19) ^ (x >> 10); }


__global__ void sha256_gpu_kernel_long_messages_shared(
    const uint8_t* __restrict__ padded_chunks,
    const uint32_t* __restrict__ chunk_offsets,
    const uint32_t* __restrict__ chunks_per_message,
    uint32_t* __restrict__ output_hashes,
    int num_messages)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_messages) return;

    
    __shared__ uint32_t shared_chunk[16];

    uint32_t offset = chunk_offsets[idx];
    uint32_t num_my_chunks = chunks_per_message[idx];
    const uint8_t* my_chunk_ptr = &padded_chunks[offset * 64];

    uint32_t h[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        h[i] = initial_h[i];
    }

    for (uint32_t c = 0; c < num_my_chunks; ++c) {        
        if (threadIdx.x < 16) {
            const uint8_t* chunk_data_ptr = my_chunk_ptr + c * 64;
            shared_chunk[threadIdx.x] = (uint32_t)chunk_data_ptr[threadIdx.x * 4] << 24 |
                                        (uint32_t)chunk_data_ptr[threadIdx.x * 4 + 1] << 16 |
                                        (uint32_t)chunk_data_ptr[threadIdx.x * 4 + 2] << 8 |
                                        (uint32_t)chunk_data_ptr[threadIdx.x * 4 + 3];
        }
        
        __syncthreads();


        uint32_t w[64];
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            w[t] = shared_chunk[t];
        }

        #pragma unroll
        for (int t = 16; t < 64; ++t) {
            w[t] = sigma1(w[t - 2]) + w[t - 7] + sigma0(w[t - 15]) + w[t - 16];
        }

        uint32_t a = h[0], b = h[1], c_var = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], h_val = h[7];

        #pragma unroll
        for (int t = 0; t < 64; ++t) {
            uint32_t t1 = h_val + Sigma1(e) + Ch(e, f, g) + k[t] + w[t];
            uint32_t t2 = Sigma0(a) + Maj(a, b, c_var);
            h_val = g; g = f; f = e; e = d + t1;
            d = c_var; c_var = b; b = a; a = t1 + t2;
        }

        h[0] += a; h[1] += b; h[2] += c_var; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += h_val;

        __syncthreads();
    }

    uint32_t* out_ptr = &output_hashes[idx * 8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out_ptr[i] = h[i];
    }
}


std::vector<uint8_t> pad_message(const uint8_t* message, size_t len) {
    size_t new_len = len + 1 + 8; 
    size_t num_chunks = (new_len + 63) / 64; 
    size_t padded_len = num_chunks * 64;

    std::vector<uint8_t> padded(padded_len, 0);
    if (len > 0) {
        memcpy(padded.data(), message, len);
    }

    padded[len] = 0x80; 

    uint64_t bit_len = len * 8;
    for (int i = 0; i < 8; ++i) {
        padded[padded_len - 8 + i] = (bit_len >> (56 - i * 8)) & 0xFF;
    }

    return padded;
}


float run_sha256_batch_long_messages(
    const uint8_t* messages,
    const size_t* message_lengths,
    int num_messages,
    uint32_t* output_hashes)
{
    std::vector<uint32_t> chunks_per_message(num_messages);
    std::vector<uint32_t> chunk_offsets(num_messages);
    std::vector<uint8_t> all_padded_chunks;
    size_t total_chunks = 0;

    for (int i = 0; i < num_messages; ++i) {
        std::vector<uint8_t> padded = pad_message(messages + (i > 0 ? message_lengths[i-1] : 0), message_lengths[i]);
        chunks_per_message[i] = padded.size() / 64;
        chunk_offsets[i] = total_chunks;
        total_chunks += chunks_per_message[i];
        all_padded_chunks.insert(all_padded_chunks.end(), padded.begin(), padded.end());
    }

    uint8_t* d_padded_chunks;
    uint32_t* d_chunks_per_message;
    uint32_t* d_chunk_offsets;
    uint32_t* d_output_hashes;

    size_t padded_data_size = total_chunks * 64 * sizeof(uint8_t);
    size_t chunk_meta_size = num_messages * sizeof(uint32_t);
    size_t output_size = num_messages * 8 * sizeof(uint32_t);

    cudaMalloc(&d_padded_chunks, padded_data_size);
    cudaMalloc(&d_chunks_per_message, chunk_meta_size);
    cudaMalloc(&d_chunk_offsets, chunk_meta_size);
    cudaMalloc(&d_output_hashes, output_size);

    cudaMemcpy(d_padded_chunks, all_padded_chunks.data(), padded_data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_chunks_per_message, chunks_per_message.data(), chunk_meta_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_chunk_offsets, chunk_offsets.data(), chunk_meta_size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads_per_block = 256; 
    int blocks_per_grid = (num_messages + threads_per_block - 1) / threads_per_block;

    cudaEventRecord(start);
    sha256_gpu_kernel_long_messages_shared<<<blocks_per_grid, threads_per_block>>>(
        d_padded_chunks, d_chunk_offsets, d_chunks_per_message, d_output_hashes, num_messages);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(output_hashes, d_output_hashes, output_size, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_padded_chunks);
    cudaFree(d_chunks_per_message);
    cudaFree(d_chunk_offsets);
    cudaFree(d_output_hashes);

    return milliseconds;
}