#pragma once

#include "metal.h"

void bench_mark_cpu_matmul(int m, int n, int k);

void bench_mark_mps_gemm(int m, int n, int k);

void bench_mark_metal_matmul_v1(int m, int n, int k);

void bench_mark_metal_matmul_v2(int m, int n, int k);

void bench_mark_metal_matmul_v3(int m, int n, int k);

void bench_mark_metal_matmul_v4(int m, int n, int k);

void bench_mark_metal_matmul_v5(int m, int n, int k);

void bench_mark_metal_matmul_v6(int m, int n, int k);

void bench_mark_metal_matmul_v7(int m, int n, int k);