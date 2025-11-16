#include "./sha256.cuh"
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CUDA_CHECK(err) { \
    cudaError_t error = err; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(error), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

__forceinline__ __device__ uint32_t ROTR(uint32_t a, uint32_t b) {
    return __funnelshift_r(a, a, b);
}

__forceinline__ __device__ uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__forceinline__ __device__ uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__forceinline__ __device__ uint32_t Sigma0(uint32_t x) {
    return ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22);
}

__forceinline__ __device__ uint32_t Sigma1(uint32_t x) {
    return ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25);
}

__forceinline__ __device__ uint32_t sigma0(uint32_t x) {
    return ROTR(x, 7) ^ ROTR(x, 18) ^ (x >> 3);
}

__forceinline__ __device__ uint32_t sigma1(uint32_t x) {
    return ROTR(x, 17) ^ ROTR(x, 19) ^ (x >> 10);
}

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

/**
 * @brief Warp-based SHA256 kernel for variable length messages
 * 
 * This kernel processes one message per warp (32 threads) with
 * optimized shared memory usage and warp-level operations.
 */
__global__ void sha256_extreme_batch_kernel(
    const uint8_t* __restrict__ messages, 
    size_t message_len,
    uint32_t num_messages, 
    uint32_t* __restrict__ hashes) {
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane_id = threadIdx.x % 32;
    
    // Enhanced bounds checking
    if (warp_id >= num_messages) return;
    
    extern __shared__ uint8_t shared_memory[];
    uint32_t* shared_w = (uint32_t*)shared_memory;
    uint32_t* shared_state = (uint32_t*)&shared_memory[64 * 32 * sizeof(uint32_t)];
    
    const int warps_per_block = blockDim.x / 32;
    const int warp_id_in_block = threadIdx.x / 32;
    
    if (warp_id_in_block >= warps_per_block) return;
    
    uint32_t* warp_w = &shared_w[warp_id_in_block * 64];
    uint32_t* warp_state = &shared_state[warp_id_in_block * 8];
    
    const uint8_t* my_message = messages + (warp_id * message_len);
    
    if (lane_id < 8) {
        warp_state[lane_id] = initial_h[lane_id];
    }
    
   
    size_t num_blocks = (message_len + 8 + 63) / 64; 
    
    for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint8_t* block_data = my_message + (block_idx * 64);
        size_t bytes_in_block = (block_idx == num_blocks - 1) ? 
            (message_len - block_idx * 64) : 64;
        
        #pragma unroll
        for (int t = lane_id; t < 16; t += 32) {
            if (t * 4 + 3 < bytes_in_block) {
                warp_w[t] = (uint32_t)block_data[t * 4] << 24 |
                           (uint32_t)block_data[t * 4 + 1] << 16 |
                           (uint32_t)block_data[t * 4 + 2] << 8 |
                           (uint32_t)block_data[t * 4 + 3];
            } else if (t * 4 < bytes_in_block) {
                uint32_t word = 0;
                for (int i = 0; i < 4 && (t * 4 + i) < bytes_in_block; i++) {
                    uint8_t byte_val = block_data[t * 4 + i];
                    word |= (uint32_t)byte_val << (24 - i * 8);
                }
                warp_w[t] = word;
            } else {
                warp_w[t] = 0;
            }
        }
        
        if (block_idx == num_blocks - 1) {
            if (lane_id == 0) {
                size_t padding_pos = bytes_in_block;
                if (padding_pos < 64) {
                    if (padding_pos < 56) {
                       
                        warp_w[padding_pos / 4] |= 0x80 << (24 - (padding_pos % 4) * 8);
                    } else {
                        // Tidak ada space, butuh block tambahan
                        // Handle di block berikutnya (tapi ini block terakhir)
                    }
                }
            }
            
            
            if (bytes_in_block <= 56) {
                if (lane_id == 14) {
                    uint64_t bit_len = message_len * 8;
                    warp_w[14] = (bit_len >> 32) & 0xFFFFFFFF;
                }
                if (lane_id == 15) {
                    uint64_t bit_len = message_len * 8;
                    warp_w[15] = bit_len & 0xFFFFFFFF;
                }
            }
        }
        
        
        #pragma unroll
        for (int t = 16 + lane_id; t < 64; t += 32) {
            warp_w[t] = sigma1(warp_w[t-2]) + warp_w[t-7] + 
                       sigma0(warp_w[t-15]) + warp_w[t-16];
        }
        
        block.sync();
        
        uint32_t a, b, c, d, e, f, g, h_val;
        
        if (lane_id == 0) {
            a = warp_state[0]; b = warp_state[1]; c = warp_state[2]; d = warp_state[3];
            e = warp_state[4]; f = warp_state[5]; g = warp_state[6]; h_val = warp_state[7];
        }
        
        a = warp.shfl(a, 0);
        b = warp.shfl(b, 0);
        c = warp.shfl(c, 0);
        d = warp.shfl(d, 0);
        e = warp.shfl(e, 0);
        f = warp.shfl(f, 0);
        g = warp.shfl(g, 0);
        h_val = warp.shfl(h_val, 0);
        
        int rounds_per_thread = (64 + 31) / 32;
        #pragma unroll
        for (int round_base = lane_id * rounds_per_thread; 
             round_base < min(64, (lane_id + 1) * rounds_per_thread); 
             round_base++) {
            uint32_t t1 = h_val + Sigma1(e) + Ch(e, f, g) + k[round_base] + warp_w[round_base];
            uint32_t t2 = Sigma0(a) + Maj(a, b, c);
            h_val = g; 
            g = f; 
            f = e; 
            e = d + t1;
            d = c; 
            c = b; 
            b = a; 
            a = t1 + t2;
        }
        
        for (int offset = 16; offset > 0; offset /= 2) {
            uint32_t a_temp = warp.shfl_down(a, offset);
            uint32_t b_temp = warp.shfl_down(b, offset);
            uint32_t c_temp = warp.shfl_down(c, offset);
            uint32_t d_temp = warp.shfl_down(d, offset);
            uint32_t e_temp = warp.shfl_down(e, offset);
            uint32_t f_temp = warp.shfl_down(f, offset);
            uint32_t g_temp = warp.shfl_down(g, offset);
            uint32_t h_temp = warp.shfl_down(h_val, offset);
            
            if (lane_id < offset) {
                a += a_temp; b += b_temp; c += c_temp; d += d_temp;
                e += e_temp; f += f_temp; g += g_temp; h_val += h_temp;
            }
        }
        
        if (lane_id == 0) {
            warp_state[0] += a; 
            warp_state[1] += b;
            warp_state[2] += c; 
            warp_state[3] += d;
            warp_state[4] += e; 
            warp_state[5] += f;
            warp_state[6] += g; 
            warp_state[7] += h_val;
        }
        
        warp.sync();
    }
    
    if (lane_id < 8 && warp_id < num_messages) {
        hashes[warp_id * 8 + lane_id] = warp_state[lane_id];
    }
}

/**
 * @brief Template kernel for fixed-length messages
 * 
 * Optimized for specific message lengths with compile-time unrolling.
 */
template<int MESSAGE_LENGTH>
__global__ void sha256_fixed_length_kernel(
    const uint8_t* __restrict__ messages,
    uint32_t num_messages, 
    uint32_t* __restrict__ hashes) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_messages) return;
    
    constexpr size_t num_blocks = (MESSAGE_LENGTH + 8 + 63) / 64;
    constexpr size_t last_block_bytes = MESSAGE_LENGTH % 64;
    
    uint32_t h0 = initial_h[0], h1 = initial_h[1], h2 = initial_h[2], h3 = initial_h[3];
    uint32_t h4 = initial_h[4], h5 = initial_h[5], h6 = initial_h[6], h7 = initial_h[7];
    
    const uint8_t* my_message = messages + (tid * MESSAGE_LENGTH);
    
    for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        uint32_t w[64];
        const uint8_t* block_data = my_message + (block_idx * 64);
        
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            size_t byte_pos = block_idx * 64 + i * 4;
            
            if (byte_pos + 3 < MESSAGE_LENGTH) {
                w[i] = (uint32_t)block_data[i * 4] << 24 |
                       (uint32_t)block_data[i * 4 + 1] << 16 |
                       (uint32_t)block_data[i * 4 + 2] << 8 |
                       (uint32_t)block_data[i * 4 + 3];
            } else if (byte_pos < MESSAGE_LENGTH) {
                uint32_t word = 0;
                for (int j = 0; j < 4; j++) {
                    size_t current_byte = byte_pos + j;
                    if (current_byte < MESSAGE_LENGTH) {
                        word |= (uint32_t)block_data[i * 4 + j] << (24 - j * 8);
                    } else if (current_byte == MESSAGE_LENGTH && block_idx == num_blocks - 1) {
                        word |= 0x80 << (24 - j * 8);
                    }
                }
                w[i] = word;
            } else {
                w[i] = 0;
            }
        }
        
        if (block_idx == num_blocks - 1) {
            uint64_t bit_len = MESSAGE_LENGTH * 8;
            w[14] = (bit_len >> 32) & 0xFFFFFFFF;
            w[15] = bit_len & 0xFFFFFFFF;
        }
        
        #pragma unroll
        for (int i = 16; i < 64; i++) {
            w[i] = sigma1(w[i-2]) + w[i-7] + sigma0(w[i-15]) + w[i-16];
        }
        
        uint32_t a = h0, b = h1, c = h2, d = h3;
        uint32_t e = h4, f = h5, g = h6, h_val = h7;
        
        #pragma unroll
        for (int i = 0; i < 64; i++) {
            uint32_t t1 = h_val + Sigma1(e) + Ch(e, f, g) + k[i] + w[i];
            uint32_t t2 = Sigma0(a) + Maj(a, b, c);
            h_val = g; 
            g = f; 
            f = e; 
            e = d + t1;
            d = c; 
            c = b; 
            b = a; 
            a = t1 + t2;
        }
        
        h0 += a; h1 += b; h2 += c; h3 += d;
        h4 += e; h5 += f; h6 += g; h7 += h_val;
    }
    
    uint32_t* my_hash = hashes + (tid * 8);
    my_hash[0] = h0; my_hash[1] = h1; my_hash[2] = h2; my_hash[3] = h3;
    my_hash[4] = h4; my_hash[5] = h5; my_hash[6] = h6; my_hash[7] = h7;
}

/**
 * @brief 
 */
__global__ void sha256_64byte_kernel(
    const uint8_t* __restrict__ messages,
    uint32_t num_messages, 
    uint32_t* __restrict__ hashes) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_messages) return;
    
    uint32_t h0 = initial_h[0], h1 = initial_h[1], h2 = initial_h[2], h3 = initial_h[3];
    uint32_t h4 = initial_h[4], h5 = initial_h[5], h6 = initial_h[6], h7 = initial_h[7];
    
    const uint8_t* my_message = messages + (tid * 64);
    
    uint32_t w[64];
    
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        w[i] = (uint32_t)my_message[i * 4] << 24 |
               (uint32_t)my_message[i * 4 + 1] << 16 |
               (uint32_t)my_message[i * 4 + 2] << 8 |
               (uint32_t)my_message[i * 4 + 3];
    }
    
    w[15] = 0x00000200; 
    
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        w[i] = sigma1(w[i-2]) + w[i-7] + sigma0(w[i-15]) + w[i-16];
    }
    
    uint32_t a = h0, b = h1, c = h2, d = h3;
    uint32_t e = h4, f = h5, g = h6, h_val = h7;
    
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h_val + Sigma1(e) + Ch(e, f, g) + k[i] + w[i];
        uint32_t t2 = Sigma0(a) + Maj(a, b, c);
        h_val = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    
    uint32_t* my_hash = hashes + (tid * 8);
    my_hash[0] = h0 + a; my_hash[1] = h1 + b; my_hash[2] = h2 + c; my_hash[3] = h3 + d;
    my_hash[4] = h4 + e; my_hash[5] = h5 + f; my_hash[6] = h6 + g; my_hash[7] = h7 + h_val;
}

/**
 * @brief Main batch processing function with automatic kernel selection
 */
void sha256_gpu_batch(const uint8_t* messages, size_t message_len, uint32_t num_messages, uint32_t* hashes, cudaStream_t stream) {
    if (messages == nullptr || hashes == nullptr || num_messages == 0) {
        throw std::invalid_argument("Invalid input parameters");
    }
    
    // PERBAIKAN: Validasi message length
    if (message_len == 0 || message_len > 128) {
        throw std::invalid_argument("Message length must be between 1 and 128 bytes");
    }
    
    if (message_len == 64) {
        const int threads_per_block = 256;
        const int blocks_per_grid = (num_messages + threads_per_block - 1) / threads_per_block;
        sha256_64byte_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(messages, num_messages, hashes);
    }
    else if (message_len <= 64) {
        const int threads_per_block = 256;
        const int blocks_per_grid = (num_messages + threads_per_block - 1) / threads_per_block;
        
        switch (message_len) {
            case 56: sha256_fixed_length_kernel<56><<<blocks_per_grid, threads_per_block, 0, stream>>>(messages, num_messages, hashes); break;
            case 40: sha256_fixed_length_kernel<40><<<blocks_per_grid, threads_per_block, 0, stream>>>(messages, num_messages, hashes); break;
            case 32: sha256_fixed_length_kernel<32><<<blocks_per_grid, threads_per_block, 0, stream>>>(messages, num_messages, hashes); break;
            default: 
                // PERBAIKAN: Gunakan dynamic shared memory untuk extreme batch kernel
                const int warp_threads = 256;
                const int warps_per_block = warp_threads / 32;
                const int blocks = (num_messages + warps_per_block - 1) / warps_per_block;
                size_t shared_mem_size = (64 * 8 + 8 * 8) * warps_per_block * sizeof(uint32_t);
                sha256_extreme_batch_kernel<<<blocks, warp_threads, shared_mem_size, stream>>>(
                    messages, message_len, num_messages, hashes);
        }
    } else {
        const int warp_threads = 256;
        const int warps_per_block = warp_threads / 32;
        const int blocks = (num_messages + warps_per_block - 1) / warps_per_block;
        size_t shared_mem_size = (64 * 8 + 8 * 8) * warps_per_block * sizeof(uint32_t);
        sha256_extreme_batch_kernel<<<blocks, warp_threads, shared_mem_size, stream>>>(
            messages, message_len, num_messages, hashes);
    }
    
    CUDA_CHECK(cudaGetLastError());
    if (stream == 0) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void sha256_gpu(const uint8_t* message, size_t message_len, uint32_t* hash_output) {
    sha256_gpu_batch(message, message_len, 1, hash_output, 0);
}

SHA256GPU::SHA256GPU(size_t max_batch, size_t max_msg_len) : 
    max_batch_size(max_batch), max_message_len(max_msg_len) {
    CUDA_CHECK(cudaMalloc(&d_messages, max_batch_size * max_message_len));
    CUDA_CHECK(cudaMalloc(&d_hashes, max_batch_size * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaStreamCreate(&stream));
}

SHA256GPU::~SHA256GPU() {
    if (d_messages) cudaFree(d_messages);
    if (d_hashes) cudaFree(d_hashes);
    if (stream) cudaStreamDestroy(stream);
}

void SHA256GPU::compute_batch(const uint8_t* messages, size_t message_len, uint32_t num_messages, uint32_t* hashes) {
    if (num_messages > max_batch_size) {
        throw std::runtime_error("Batch size exceeds maximum capacity");
    }
    
    if (message_len > max_message_len) {
        throw std::runtime_error("Message length exceeds maximum supported length");
    }
    
    CUDA_CHECK(cudaMemcpyAsync(d_messages, messages, num_messages * message_len, cudaMemcpyHostToDevice, stream));
    sha256_gpu_batch(d_messages, message_len, num_messages, d_hashes, stream);
    CUDA_CHECK(cudaMemcpyAsync(hashes, d_hashes, num_messages * 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void SHA256GPU::compute_batch_async(const uint8_t* messages, size_t message_len, uint32_t num_messages, uint32_t* hashes) {
    if (num_messages > max_batch_size) {
        throw std::runtime_error("Batch size exceeds maximum capacity");
    }
    
    if (message_len > max_message_len) {
        throw std::runtime_error("Message length exceeds maximum supported length");
    }
    
    CUDA_CHECK(cudaMemcpyAsync(d_messages, messages, num_messages * message_len, cudaMemcpyHostToDevice, stream));
    sha256_gpu_batch(d_messages, message_len, num_messages, d_hashes, stream);
    CUDA_CHECK(cudaMemcpyAsync(hashes, d_hashes, num_messages * 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
}

void SHA256GPU::synchronize() {
    CUDA_CHECK(cudaStreamSynchronize(stream));
}