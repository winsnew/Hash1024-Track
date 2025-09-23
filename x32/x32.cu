#include <stdio.h>
#include <stdint.h>

__global__ void kernel32(uint32_t *a, uint32_t *b, uint32_t *add_result, 
                        uint32_t *sub_result, uint32_t *mul_result, 
                        uint32_t *div_result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        add_result[idx] = a[idx] + b[idx];
        sub_result[idx] = a[idx] - b[idx];
        mul_result[idx] = a[idx] * b[idx];
        div_result[idx] = a[idx] / b[idx];
    }
}
