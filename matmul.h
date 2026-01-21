#pragma once

#include <stdint.h>
#include <stddef.h>

void cpu_gemm_v1(float* a, float* b, float* c, int m, int n, int k);
void cpu_gemm_v2(float* a, float* b, float* c, int m, int n, int k, int tile_size);

void cpu_softmax(const float* in, float* out, int n);
void cpu_online_softmax(const float* in, float* out, int n);
