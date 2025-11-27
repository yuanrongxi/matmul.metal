#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "kernel.h"
#include "args.h"
#include "metal.h"
#include "matmul.h"

static const uint64_t seed = UINT64_C(1019827666124465389);
static const float scale = 2.0f * 0x1.0p-32f;
static const float bias = 0.0f * 0.5f;

typedef enum gemm_status (*metal_matmul_func_t)(metal_command_buffer_t* command_buffer, metal_function_t* function, 
    int m, int n, int k, metal_buffer_t* A, metal_buffer_t* B, metal_buffer_t* C);

static inline void generate_random_f32(float* d, size_t size){
    for(size_t i = 0; i < size; i++) {
        int val = (int)(rng_random(i, seed));
        d[i] = val * scale + bias;
    }
}

uint64_t get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

static void init_buffer(metal_device_t* device, 
    metal_buffer_t* A, metal_buffer_t* B, 
    metal_buffer_t* C, metal_buffer_t* C_cpu, 
    int m, int n, int k) {
    metal_buffer_create(device, m * k * sizeof(float), NULL, A);
    metal_buffer_create(device, k * n * sizeof(float), NULL, B);
    metal_buffer_create(device, m * n * sizeof(float), NULL, C);  
    metal_buffer_create(device, m * n * sizeof(float), NULL, C_cpu);

    float* a = (float*) A->ptr;
    float* b = (float*) B->ptr;
    float* c = (float*) C->ptr;
    float* c_cpu = (float*) C_cpu->ptr;

    generate_random_f32(a, m * k);
    generate_random_f32(b, k * n);
    for(size_t i = 0; i < m * n; i++) {
        c[i] = 0.0f;
        c_cpu[i] = 0.0f;
    }
}

static void validate_gemm_result(float* c, float* c_cpu, int m, int n) {
    for(size_t i = 0; i < m; i++) {
        for(size_t j = 0; j < n; j++) {
            if (fabs(c[i * n + j] - c_cpu[i * n + j]) >1e-3f) {
                printf("error at %zu, %zu: %f != %f\n", i, j, c[i * n + j], c_cpu[i * n + j]);
                return;
            }
        }
    }
}

static void release_buffer(metal_buffer_t* A, metal_buffer_t* B, metal_buffer_t* C, metal_buffer_t* C_cpu) {
    metal_buffer_release(A);
    metal_buffer_release(B);
    metal_buffer_release(C);
    metal_buffer_release(C_cpu);
}

#define BENCH_MARK_COUNT 100

void bench_mark_cpu_matmul(int m, int n, int k) {
    metal_buffer_t A;
    metal_buffer_t B;
    metal_buffer_t C;
    metal_buffer_t C_cpu;

    metal_device_t device;
    if(metal_device_create(&device) != gemm_success) {
        printf("failed to create Metal device\n");
        return;
    }

    init_buffer(&device, &A, &B, &C, &C_cpu, m, n, k);

    double cost_sec = 0;
    uint64_t start = get_time_us();
    cpu_gemm_v1((float*)A.ptr, (float*)B.ptr, (float*)C_cpu.ptr, m, n, k);
    uint64_t end = get_time_us();
    cost_sec = (double)(end - start);

    double avg_time = cost_sec;
    double gflops = (2.0 * m * n * k) * 1000000.0 / (avg_time * 1024 * 1024 * 1024);
    printf("CPU Native matmul: %.3f us (%.3f GFLOPS)\n", avg_time, gflops);

    start = get_time_us();
    cpu_gemm_v2((float*)A.ptr, (float*)B.ptr, (float*)C_cpu.ptr, m, n, k, 16);
    end = get_time_us();
    cost_sec = (double)(end - start);
    avg_time = cost_sec;
    gflops = (2.0 * m * n * k) * 1000000.0 / (avg_time * 1024 * 1024 * 1024);
    printf("CPU Tiling matmul: %.3f us (%.3f GFLOPS)\n", avg_time, gflops);

    release_buffer(&A, &B, &C, &C_cpu);
    metal_device_release(&device);
}
void bench_mark_mps_gemm(int m, int n, int k) {
    metal_buffer_t A;
    metal_buffer_t B;
    metal_buffer_t C;
    metal_buffer_t C_cpu;

    metal_device_t device;
    if(metal_device_create(&device) != gemm_success) {
        printf("failed to create Metal device\n");
        return;
    }

    metal_command_queue_t command_queue;
    if(metal_command_queue_create(&device, &command_queue) != gemm_success) {
        printf("failed to create Metal command queue\n");
        metal_device_release(&device);
        return;
    }

    init_buffer(&device, &A, &B, &C, &C_cpu, m, n, k);

    metal_command_buffer_t command_buffer;
    if(metal_command_buffer_create(&command_queue, &command_buffer) != gemm_success) {
        printf("failed to create Metal command buffer\n");
        metal_device_release(&device);
        metal_command_queue_release(&command_queue);
        release_buffer(&A, &B, &C, &C_cpu);
        return;
    }

    metal_mps_matrix_t A_mps;
    metal_mps_matrix_t B_mps;
    metal_mps_matrix_t C_mps;
    metal_mps_matrix_create(&device, &A, m, k, &A_mps);
    metal_mps_matrix_create(&device, &B, k, n, &B_mps);
    metal_mps_matrix_create(&device, &C, m, n, &C_mps);

    metal_mps_gemm_func_t gemm_func;
    if(metal_mps_gemm_func_create(&device, m, n, k, 1.0f, 0.0f, &gemm_func) != gemm_success) {
        printf("failed to create Metal MPS gemm func\n");
        metal_device_release(&device);
        metal_command_queue_release(&command_queue);
        release_buffer(&A, &B, &C, &C_cpu);
        return;
    }

    double cost_sec = 0;
     for(int i = 0; i < BENCH_MARK_COUNT; i++) {
       meta_mps_matmul(&device, &command_buffer, &gemm_func, &A_mps, &B_mps, &C_mps);
    }
    metal_command_buffer_commit(&command_buffer);
    metal_command_buffer_wait_completion(&command_buffer, &cost_sec);
    metal_command_buffer_release(&command_buffer);
    
    double avg_time = cost_sec / (BENCH_MARK_COUNT);
    double gflops = (2.0 * m * n * k) / (avg_time * 1024 * 1024 * 1024);
     printf("MPS matmul: %.3f us (%.3f GFLOPS)\n", avg_time * 1000000.0, gflops);

    cpu_gemm_v2((float*)A.ptr, (float*)B.ptr, (float*)C_cpu.ptr, m, n, k, 16);

    validate_gemm_result((float*)C.ptr, (float*)C_cpu.ptr, m, n);

    metal_mps_matrix_release(&A_mps);
    metal_mps_matrix_release(&B_mps);
    metal_mps_matrix_release(&C_mps);
    metal_mps_gemm_func_release(&gemm_func);

    release_buffer(&A, &B, &C, &C_cpu);
    metal_command_queue_release(&command_queue);
    metal_device_release(&device);
}

static enum gemm_status create_metal(metal_device_t* device, metal_library_t* library, 
                                        metal_function_t* function, metal_command_queue_t* command_queue, const char* name) {
                                                    
    enum gemm_status ret = metal_device_create(device);
    if(ret != gemm_success) {
        printf("failed to create Metal device\n");
        return ret;
    }

    ret = metal_library_create(device, library);
    if(ret != gemm_success) {
        printf("failed to create Metal library\n");
        return ret;
    }

    ret = metal_function_create(library, name, function);
    if(ret != gemm_success) {
        printf("failed to create Metal function\n");
        return ret;
    }

    ret = metal_command_queue_create(device, command_queue);
    if(ret != gemm_success) {
        printf("failed to create Metal command queue\n");
        return ret;
    }

    return gemm_success;
}

static void release_metal(metal_device_t* device, metal_library_t* library, 
    metal_function_t* function, metal_command_queue_t* command_queue) {
    metal_device_release(device);
    metal_library_release(library);
    metal_function_release(function);
    metal_command_queue_release(command_queue);
}

static void run_metal_matmul(metal_matmul_func_t func, const char* name, int m, int n, int k) {
    metal_buffer_t A;
    metal_buffer_t B;
    metal_buffer_t C;
    metal_buffer_t C_cpu;

    metal_device_t device;
    metal_library_t library;
    metal_function_t function;
    metal_command_queue_t command_queue;
    metal_command_buffer_t command_buffer;

    if(create_metal(&device, &library, &function, &command_queue, name) != gemm_success) {
        release_metal(&device, &library, &function, &command_queue);
        return;
    }
    
    if(metal_command_buffer_create(&command_queue, &command_buffer) != gemm_success) {
        release_metal(&device, &library, &function, &command_queue);
        printf("failed to create Metal command buffer\n");
        return;
    }

    init_buffer(&device, &A, &B, &C, &C_cpu, m, n, k);

    double cost_sec = 0;
    for(int i = 0; i < BENCH_MARK_COUNT; i++) {
        func(&command_buffer, &function, m, n, k, &A, &B, &C);
    }
    metal_command_buffer_commit(&command_buffer);
    metal_command_buffer_wait_completion(&command_buffer, &cost_sec);

    metal_command_buffer_release(&command_buffer);

    double avg_time = cost_sec / (BENCH_MARK_COUNT);
    double gflops = (2.0 * m * n * k) / (avg_time * 1024 * 1024 * 1024);
    printf(" %s: %.3f us (%.3f GFLOPS)\n", name, avg_time * 1000000.0, gflops);
    
    cpu_gemm_v2((float*)A.ptr, (float*)B.ptr, (float*)C_cpu.ptr, m, n, k, 16);
    validate_gemm_result((float*)C.ptr, (float*)C_cpu.ptr, m, n);

    release_buffer(&A, &B, &C, &C_cpu);
    release_metal(&device, &library, &function, &command_queue);
} 

static enum gemm_status metal_command_buffer_matmul_v1(metal_command_buffer_t* command_buffer, metal_function_t* function,
                int m, int n, int k, 
                metal_buffer_t* A, metal_buffer_t* B, metal_buffer_t* C) {
    enum gemm_status ret = gemm_success;
    if (command_buffer->object == NULL || function->pipeline_state_object == NULL) {
        return gemm_invalid_state;
    }

    matmul_v1_args_t args = {
        .m = m,
        .n = n,
        .k = k,
        .alpha = 1.0f,
        .beta = 0.0f,
    };

    size_t threadgroup_size = gemm_min(1024, function->max_threadgroup_threads);
    size_t num_threadgroups = math_ceil_div(m * n, threadgroup_size);

    return metal_command_buffer_encode_launch_kernel(command_buffer, function, 
        threadgroup_size, 1, 1,
        num_threadgroups, 1, 1, 
        sizeof(args), &args, 3, 
        (const metal_buffer_t*[]){A, B, C}, 
        (const size_t[]){0, 0, 0});
}

void bench_mark_metal_matmul_v1(int m, int n, int k) {
    run_metal_matmul(metal_command_buffer_matmul_v1, "metal_matmul_v1", m, n, k);
}

static enum gemm_status metal_command_buffer_matmul_v2(metal_command_buffer_t* command_buffer, metal_function_t* function,
                int m, int n, int k, 
                metal_buffer_t* A, metal_buffer_t* B, metal_buffer_t* C) {
    enum gemm_status ret = gemm_success;
    if (command_buffer->object == NULL || function->pipeline_state_object == NULL) {
        return gemm_invalid_state;
    }

    matmul_v2_args_t args = {
        .m = m,
        .n = n,
        .k = k,
        .alpha = 1.0f,
        .beta = 0.0f,
    };

    size_t simdgroup_size = function->simdgroup_threads;
    size_t threadgroup_size = gemm_min(n, function->max_threadgroup_threads);
    size_t num_threadgroups = math_ceil_div(m * n * simdgroup_size, threadgroup_size);

    return metal_command_buffer_encode_launch_kernel(command_buffer, function, 
        threadgroup_size, 1, 1,
        num_threadgroups, 1, 1, 
        sizeof(args), &args, 3, 
        (const metal_buffer_t*[]){A, B, C}, 
        (const size_t[]){0, 0, 0});
}

void bench_mark_metal_matmul_v2(int m, int n, int k) {
    run_metal_matmul(metal_command_buffer_matmul_v2, "metal_matmul_v2", m, n, k);
}

static enum gemm_status metal_command_buffer_matmul_v3(metal_command_buffer_t* command_buffer, metal_function_t* function,
                int m, int n, int k, 
                metal_buffer_t* A, metal_buffer_t* B, metal_buffer_t* C) {
    enum gemm_status ret = gemm_success;
    if (command_buffer->object == NULL || function->pipeline_state_object == NULL) {
        return gemm_invalid_state;
    }
    
    matmul_v3_args_t args = {
        .m = m,
        .n = n,
        .k = k,
        .alpha = 1.0f,
        .beta = 0.0f,
    };
    
    size_t threadgroup_size = gemm_min(BLOCK_SIZE * BLOCK_SIZE, function->max_threadgroup_threads);

    size_t x_num_threadgroups = math_ceil_div(m, BLOCK_SIZE);
    size_t y_num_threadgroups = math_ceil_div(n, BLOCK_SIZE);

    return metal_command_buffer_encode_launch_kernel(command_buffer, function, 
        threadgroup_size, 1, 1,
        x_num_threadgroups, y_num_threadgroups, 1, 
        sizeof(args), &args, 3, 
        (const metal_buffer_t*[]){A, B, C}, 
        (const size_t[]){0, 0, 0});
}

void bench_mark_metal_matmul_v3(int m, int n, int k) {
    run_metal_matmul(metal_command_buffer_matmul_v3, "metal_matmul_v3", m, n, k);

}

static enum gemm_status metal_command_buffer_matmul_v4(metal_command_buffer_t* command_buffer, metal_function_t* function,
                int m, int n, int k, 
                metal_buffer_t* A, metal_buffer_t* B, metal_buffer_t* C) {
    enum gemm_status ret = gemm_success;
    if (command_buffer->object == NULL || function->pipeline_state_object == NULL) {
        return gemm_invalid_state;
    }
    
    matmul_v4_args_t args = {
        .m = m,
        .n = n,
        .k = k,
        .alpha = 1.0f,
        .beta = 0.0f,
    };
    
    size_t threadgroup_size = gemm_min(BM_SIZE * BN_SIZE / TM_SIZE, function->max_threadgroup_threads);
    size_t x_num_threadgroups = math_ceil_div(n, BN_SIZE);
    size_t y_num_threadgroups = math_ceil_div(m, BM_SIZE);
    
    return metal_command_buffer_encode_launch_kernel(command_buffer, function, 
        threadgroup_size, 1, 1,
        x_num_threadgroups, y_num_threadgroups, 1,  
        sizeof(args), &args, 3, 
        (const metal_buffer_t*[]){A, B, C}, 
        (const size_t[]){0, 0, 0});
}

void bench_mark_metal_matmul_v4(int m, int n, int k) {
    run_metal_matmul(metal_command_buffer_matmul_v4, "metal_matmul_v4", m, n, k);
}

static enum gemm_status metal_command_buffer_matmul_v5(metal_command_buffer_t* command_buffer, metal_function_t* function,
                int m, int n, int k, 
                metal_buffer_t* A, metal_buffer_t* B, metal_buffer_t* C) {
    enum gemm_status ret = gemm_success;
    if (command_buffer->object == NULL || function->pipeline_state_object == NULL) {
        return gemm_invalid_state;
    }
    
    matmul_v5_args_t args = {
        .m = m,
        .n = n,
        .k = k,
        .alpha = 1.0f,
        .beta = 0.0f,
    };
    
    size_t threadgroup_size = gemm_min((BM * BN) / (TM * TN), function->max_threadgroup_threads);
    size_t x_num_threadgroups = math_ceil_div(n, BN);
    size_t y_num_threadgroups = math_ceil_div(m, BM);
    
    return metal_command_buffer_encode_launch_kernel(command_buffer, function, 
        threadgroup_size, 1, 1,
        x_num_threadgroups, y_num_threadgroups, 1,  
        sizeof(args), &args, 3, 
        (const metal_buffer_t*[]){A, B, C}, 
        (const size_t[]){0, 0, 0});
}

void bench_mark_metal_matmul_v5(int m, int n, int k) {
    run_metal_matmul(metal_command_buffer_matmul_v5, "metal_matmul_v5", m, n, k);
}

static enum gemm_status metal_command_buffer_matmul_v6(metal_command_buffer_t* command_buffer, metal_function_t* function,
                int m, int n, int k, 
                metal_buffer_t* A, metal_buffer_t* B, metal_buffer_t* C) {
    enum gemm_status ret = gemm_success;
    if (command_buffer->object == NULL || function->pipeline_state_object == NULL) {
        return gemm_invalid_state;
    }
    
    matmul_v6_args_t args = {
        .m = m,
        .n = n,
        .k = k,
        .alpha = 1.0f,
        .beta = 0.0f,
    };
    
    size_t threadgroup_size = gemm_min((BM * BN) / (TM * TN), function->max_threadgroup_threads);
    size_t x_num_threadgroups = math_ceil_div(n, BN);
    size_t y_num_threadgroups = math_ceil_div(m, BM);
    
    return metal_command_buffer_encode_launch_kernel(command_buffer, function, 
        threadgroup_size, 1, 1,
        x_num_threadgroups, y_num_threadgroups, 1,  
        sizeof(args), &args, 3, 
        (const metal_buffer_t*[]){A, B, C}, 
        (const size_t[]){0, 0, 0});
}

void bench_mark_metal_matmul_v6(int m, int n, int k) {
    run_metal_matmul(metal_command_buffer_matmul_v6, "metal_matmul_v6", m, n, k);
}

static enum gemm_status metal_command_buffer_matmul_v7(metal_command_buffer_t* command_buffer, metal_function_t* function,
                int m, int n, int k, 
                metal_buffer_t* A, metal_buffer_t* B, metal_buffer_t* C) {
    enum gemm_status ret = gemm_success;
    if (command_buffer->object == NULL || function->pipeline_state_object == NULL) {
        return gemm_invalid_state;
    }
    
    matmul_v7_args_t args = {
        .m = m,
        .n = n,
        .k = k,
        .alpha = 1.0f,
        .beta = 0.0f,
    };

    assert((WBM * WBN) % (WARPSIZE * WTM * WTN * WNITER) == 0);
    assert((WBM % WMITER == 0) && (WBN % WNITER == 0));
    assert((NUM_THREADS * 4) % WBK == 0);
    assert((NUM_THREADS * 4) % WBN == 0);
    assert(WBN % (16 * WTN) == 0);
    assert(WBM % (16 * WTM) == 0);
    assert((WBM * WBK) % (4 * NUM_THREADS) == 0);
    assert((WBN * WBK) % (4 * NUM_THREADS) == 0);
    
    size_t threadgroup_size = gemm_min(NUM_THREADS, function->max_threadgroup_threads);
    size_t x_num_threadgroups = math_ceil_div(n, WBN);
    size_t y_num_threadgroups = math_ceil_div(m, WBM);
    
    return metal_command_buffer_encode_launch_kernel(command_buffer, function, 
        threadgroup_size, 1, 1,
        x_num_threadgroups, y_num_threadgroups, 1,  
        sizeof(args), &args, 3, 
        (const metal_buffer_t*[]){A, B, C}, 
        (const size_t[]){0, 0, 0});
}

void bench_mark_metal_matmul_v7(int m, int n, int k) {
    run_metal_matmul(metal_command_buffer_matmul_v7, "metal_matmul_v7", m, n, k);
}