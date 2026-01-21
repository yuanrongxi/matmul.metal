#include <omp.h>
#include <math.h>
#include "types.h"
#include "matmul.h"

void cpu_gemm_v1(float* a, float* b, float* c, int m, int n, int k) {
    int i;
    #pragma omp parallel for private(i)
    for(i = 0; i < m; i++) {
        for(int l = 0; l < k; l++) {
            for(int j = 0; j < n; j++) {
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

void cpu_softmax(const float* in, float* out, int n) {
    float maxval = in[0];
    for(int i = 1; i < n; i++) {
        if(maxval < in[i]) {
            maxval = in[i];
        }
    }

    double sum = 0.0f;
    for(int i = 0; i < n; i++) {
        out[i] = expf(in[i] - maxval);
        sum += out[i];
    }

    float norm = 1.0f / sum;
    for(int i = 0; i < n; i++) {
        out[i] *= norm;
    }
}

void cpu_online_softmax(const float* in, float* out, int n) {
    float maxval = -INFINITY;
    double sum = 0.0f;
    for(int i = 0; i < n; i++) {
        if(maxval < in[i]) {
            float prev_maxval = maxval;
            maxval = in[i];
            sum = sum * expf(prev_maxval - maxval) + expf(in[i] - maxval);
        } else {
            sum += expf(in[i] - maxval);
        }
    }

    for(int i = 0; i < n; i++) {
        out[i] = expf(in[i] - maxval) / sum;
    }
}