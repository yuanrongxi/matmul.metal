#include "common.h"

static const uint64_t seed = UINT64_C(1019827666124465389);
static const float scale = 2.0f * 0x1.0p-32f;
static const float bias = 0.0f * 0.5f;

void generate_random_f32(float* d, size_t size){
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

void validate_gemm_result(float* c, float* c_cpu, int n) {
    for(size_t i = 0; i < n; i++) {
        if (fabs(c[i] - c_cpu[i]) > 1e-3f) {
            printf("error at %zu: %f != %f\n", i, c[i], c_cpu[i]);
            return;
        }
    }
}

enum gemm_status create_metal(metal_device_t* device, metal_library_t* library, 
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

void release_metal(metal_device_t* device, metal_library_t* library, 
    metal_function_t* function, metal_command_queue_t* command_queue) {
    metal_device_release(device);
    metal_library_release(library);
    metal_function_release(function);
    metal_command_queue_release(command_queue);
}

void print_f32(float* ptr, int n) {
    for(int i = 0; i < n; i++){
        printf("%f ", ptr[i]);
    }
    printf("\n");
}
