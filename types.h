#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

enum gemm_status {
    gemm_success = 0,
    gemm_error = 1,
    gemm_invalid_argument = 2,
    gemm_unsupported_system = 3,
    gemm_invalid_state = 4,
};

#define gemm_max(a, b) ((a) > (b) ? (a) : (b))
#define gemm_min(a, b) ((a) < (b) ? (a) : (b))

inline static size_t math_ceil_div(size_t numer, size_t denom) {
    return (numer + denom - 1) / denom;
}

inline static  uint32_t rng_random(uint64_t offset, uint64_t seed) {
    const uint64_t y = offset * seed;
    const uint64_t z = y + seed;

    /* Round 1 */
    uint64_t x = y * y + y;
    x = (x >> 32) | (x << 32);

    /* Round 2 */
    x = x * x + z;
    x = (x >> 32) | (x << 32);

    /* Round 3 */
    x = x * x + y;
    x = (x >> 32) | (x << 32);

    /* Round 4 */
    x = x * x + z;
    return (uint32_t) (x >> 32);
}

