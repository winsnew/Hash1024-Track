#include <stdio.h>
#include <stdint.h>

typedef struct {
    uint64_t low;
    uint64_t mid_low;
    uint64_t mid_high;
    uint64_t high;
} uint256;

uint256 add256(uint256 a, uint256 b) {
    uint256 result;
    uint64_t carry = 0;
    
    result.low = a.low + b.low;
    carry = (result.low < a.low);
    
    result.mid_low = a.mid_low + b.mid_low + carry;
    carry = (result.mid_low < a.mid_low) || (carry && (result.mid_low == a.mid_low));
    
    result.mid_high = a.mid_high + b.mid_high + carry;
    carry = (result.mid_high < a.mid_high) || (carry && (result.mid_high == a.mid_high));
    
    result.high = a.high + b.high + carry;
    return result;
}

uint256 sub256(uint256 a, uint256 b) {
    uint256 result;
    uint64_t borrow = 0;
    
    result.low = a.low - b.low;
    borrow = (a.low < b.low);
    
    result.mid_low = a.mid_low - b.mid_low - borrow;
    borrow = (a.mid_low < b.mid_low) || (borrow && (a.mid_low == b.mid_low));
    
    result.mid_high = a.mid_high - b.mid_high - borrow;
    borrow = (a.mid_high < b.mid_high) || (borrow && (a.mid_high == b.mid_high));
    
    result.high = a.high - b.high - borrow;
    return result;
}

uint256 mul256(uint256 a, uint256 b) {
    uint256 result = {0, 0, 0, 0};
    uint256 temp = a;
    
    for (int i = 0; i < 256; i++) {
        if (b.low & 1) {
            result = add256(result, temp);
        }
        uint64_t carry = (temp.low >> 63) & 1;
        temp.low = (temp.low << 1) | (temp.mid_low << 1);
        temp.mid_low = (temp.mid_low << 1) | (temp.mid_high << 1);
        temp.mid_high = (temp.mid_high << 1) | (temp.high << 1);
        temp.high = (temp.high << 1) | carry;
        b.low >>= 1;
    }
    return result;
}

