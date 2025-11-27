#pragma once
#if !defined(__METAL_VERSION__)
#include <stdint.h>
#endif

typedef struct matmul_v1_args {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    float alpha;
    float beta;
} matmul_v1_args_t;

typedef matmul_v1_args_t matmul_v2_args_t; // alias for v1

#define BLOCK_SIZE 32 //cache block size (32 x 32)
typedef matmul_v1_args_t matmul_v3_args_t;

#define BM_SIZE 64   //big block size (64 x 64)
#define BN_SIZE 64
#define BK_SIZE 8
#define TM_SIZE 8
typedef matmul_v1_args_t matmul_v4_args_t;

#define BM 64
#define BN 64
#define BK 8
#define TM 8
#define TN 4
typedef matmul_v1_args_t matmul_v5_args_t;

typedef matmul_v1_args_t matmul_v6_args_t;

#define NUM_THREADS 128
#define WBM 64
#define WBN 64
#define WBK 8
#define WM 32
#define WN 32
#define WTN 2
#define WTM 4

#define WARPSIZE 32 //simdgroup size
#define WNITER 2
#define WMITER ((WM * WN) / (WARPSIZE * WTM * WTN * WNITER))
#define WSUBM (WM / WMITER)
#define WSUBN (WN / WNITER)

typedef matmul_v1_args_t matmul_v7_args_t;
