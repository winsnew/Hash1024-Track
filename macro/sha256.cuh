#ifndef SHA256_CUH
#define SHA256_CUH

#include <cstdint>
#include <cuda_runtime.h>

/**
 * @brief 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 */
void sha256_gpu(const uint8_t* message, size_t message_len, uint32_t* hash_output);

#endif // SHA256_CUH