#include <stdio.h>
#include <stdint.h>

typedef struct {
    uint32 low;
    uint32 high;
} uint64;

uint64 add64(uint64 a, uint64 b) {
    uint64 result;
    result.low = a.low + b.low;
    result.high = a.high + b.high + (result.low < a.low);
    return result;
}
uint64 sub64(uint64 a, uint64 b) {
    uint64 result;
    result.low = a.low - b.low;
    result.high = a.high - b.high - (a.low < b.low); 
    return result;
}
uint64 mul64(uint64 a, uint64 b) {
    uint64 result = {0, 0};
    uint64 temp = a;
    
    for (int i = 0; i < 64; i++) {
        if (b.low & 1) {
            result = add64(result, temp);
        }
        temp.high = (temp.high << 1) | (temp.low >> 31);
        temp.low <<= 1;
        b.low >>= 1;
    }
    return result;
}
