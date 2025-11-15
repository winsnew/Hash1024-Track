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
 * @param input_chunks 
 * @param output_hashes 
 * @param num_chunks 
 * @return 
 */
float run_sha256_batch(const uint8_t* input_chunks, uint32_t* output_hashes, int num_chunks);

#endif // SHA256_CUH