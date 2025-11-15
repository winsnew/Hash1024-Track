// sha256.cuh
#ifndef SHA256_CUH
#define SHA256_CUH

#include <cstdint>
#include <cuda_runtime.h>

/**
 * @brief Computes the SHA-256 hash for a batch of variable-length messages on the GPU.
 *
 * This function handles messages of arbitrary length. It performs the necessary
 * padding on the host, transfers the padded data to the device, and launches a
 * kernel where each thread processes one entire message (which may consist of
 * multiple 64-byte chunks).
 *
 * @param messages A pointer to a contiguous buffer in host memory containing all messages.
 * @param message_lengths An array where message_lengths[i] is the length of the i-th message in bytes.
 * @param num_messages The total number of messages to hash.
 * @param output_hashes A pointer to a host buffer to store the resulting 32-byte hashes.
 *                      The buffer must be large enough to hold num_messages * 32 bytes.
 * @return The time in milliseconds for the GPU computation and data transfers.
 */
float run_sha256_batch_long_messages(
    const uint8_t* messages,
    const size_t* message_lengths,
    int num_messages,
    uint32_t* output_hashes
);

#endif // SHA256_CUH