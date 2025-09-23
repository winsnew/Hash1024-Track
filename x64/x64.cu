#include <stdio.h>
#include <stdint.h>

typedef struct {
    uint32_t low;
    uint32_t high;
} uint64;

__global__ void kernel64(uint64 *a, uint64 *b, uint64 *add_result, 
                        uint64 *sub_result, uint64 *mul_result, 
                        uint64 *div_result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        uint64 add;
        add.low = a[idx].low + b[idx].low;
        add.high = a[idx].high + b[idx].high + (add.low < a[idx].low);
        add_result[idx] = add;
        
        uint64 sub;
        sub.low = a[idx].low - b[idx].low;
        sub.high = a[idx].high - b[idx].high - (a[idx].low < b[idx].low);
        sub_result[idx] = sub;
        uint64 mul = {0, 0};
        uint64 temp = a[idx];
        uint64 multiplier = b[idx];
        
        for (int i = 0; i < 64; i++) {
            if (multiplier.low & 1) {
                uint64 temp_add;
                temp_add.low = mul.low + temp.low;
                temp_add.high = mul.high + temp.high + (temp_add.low < mul.low);
                mul = temp_add;
            }
            uint32_t carry = (temp.low >> 31) & 1;
            temp.low = (temp.low << 1) | (temp.high << 1);
            temp.high = (temp.high << 1) | carry;
            multiplier.low >>= 1;
        }
        
        mul_result[idx] = mul;
        
        uint64 div = {0, 0};
        uint64 remainder = {0, 0};
        uint64 dividend = a[idx];
        uint64 divisor = b[idx];
        if (divisor.low == 0 && divisor.high == 0) {
            div.low = 0xFFFFFFFF; // Error Value Check
            div.high = 0xFFFFFFFF;
        } else {
            for (int i = 0; i < 64; i++) {
                uint32_t carry = (remainder.low >> 31) & 1;
                remainder.low = (remainder.low << 1) | (remainder.high << 1);
                remainder.high = (remainder.high << 1) | (dividend.high >> 31);
                carry = (dividend.low >> 31) & 1;
                dividend.low = (dividend.low << 1) | (dividend.high << 1);
                dividend.high = (dividend.high << 1) | carry;
                
                if (remainder.high > divisor.high || 
                    (remainder.high == divisor.high && remainder.low >= divisor.low)) {
                    uint64 temp_sub;
                    temp_sub.low = remainder.low - divisor.low;
                    temp_sub.high = remainder.high - divisor.high - (remainder.low < divisor.low);
                    remainder = temp_sub;
                
                    if (i < 32) {
                        div.low |= (1 << (31 - i));
                    } else {
                        div.high |= (1 << (63 - i));
                    }
                }
            }
        }
        div_result[idx] = div;
    }
}

