#include <omp.h>
#include "types.h"
#include "matmul.h"

void cpu_gemm_v1(float* a, float* b, float* c, int m, int n, int k) {
    int i;
    #pragma omp parallel for private(i)
    for(i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            for(int l = 0; l < k; l++) {
                c[i * n + j] += a[i * k + l] * b[l * n + j];
            }
        }
    }
}

void cpu_gemm_v2(float* a, float* b, float* c, int m, int n, int k, int tile_size) {
    #pragma omp parallel for shared(c, a, b) collapse(2)
    for(int rt = 0; rt < m; rt += 256) {
        for (int ct = 0; ct < n; ct += 256) {
            for (int it = 0; it < k; it += tile_size) {
                for (int i = rt; i < rt + 256; i++) {
                    int end = gemm_min(it + tile_size, k);
                    for (int l = it; l < end; l++) {
                        for (int j = ct; j < ct + 256; j++) {
                            c[i * n + j] += a[i * k + l] * b[l * n + j];
                        }
                    }
                }
            }
        }
    }
}