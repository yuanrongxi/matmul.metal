#include "kernel.h"
#include <omp.h>
#include <stdio.h>
#include <getopt.h>

static void run() {
    bench_mark_cpu_matmul(1024, 1024, 1024);
    bench_mark_mps_gemm(1024, 1024, 1024);
    bench_mark_metal_matmul_v1(1024, 1024, 1024);
    bench_mark_metal_matmul_v2(1024, 1024, 1024);
    bench_mark_metal_matmul_v3(1024, 1024, 1024);
    bench_mark_metal_matmul_v4(1024, 1024, 1024);
    bench_mark_metal_matmul_v5(1024, 1024, 1024);
    bench_mark_metal_matmul_v6(1024, 1024, 1024);
    bench_mark_metal_matmul_v7(1024, 1024, 1024);
}

int main(int argc, char** argv) {
    omp_set_num_threads(omp_get_max_threads());
    run();
    return 0;
}
