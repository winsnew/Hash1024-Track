#include <stdio.h>
#include <stdint.h>

typedef struct {
    uint64_t low;
    uint64_t high;
} uint128;

uint128 add128(uint128 a, uint128 b) {
    uint128 result;
    result.low = a.low + b.low;
    result.high = a.high + b.high + (result.low < a.low); 
    return result;
}
uint128 sub128(uint128 a, uint128 b) {
    uint128 result;
    result.low = a.low - b.low;
    result.high = a.high - b.high - (a.low < b.low); 
    return result;
}
uint128 mul128(uint128 a, uint128 b) {
    uint128 result = {0, 0};
    uint128 temp = a;
    for (int i = 0; i < 128; i++) {
        if (b.low & 1) {
            result = add128(result, temp);
        }
        uint64 carry = (temp.low >> 63) & 1;
        temp.low = (temp.low << 1) | (temp.high << 1);
        temp.high = (temp.high << 1) | carry;
        b.low >>= 1;
    }
    return result;
}
