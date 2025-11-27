#pragma once
#include <stddef.h>

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif


//device
typedef struct metal_device {
    void* object;
    size_t num_cores;
    size_t max_buffer_size;
    size_t max_threadgroup_memory;
    size_t max_threadgroup_threads_x;
    size_t max_threadgroup_threads_y;
    size_t max_threadgroup_threads_z;
}metal_device_t;

enum gemm_status metal_device_create(metal_device_t* device);
enum gemm_status metal_device_release(metal_device_t* device);


//library
typedef struct metal_library {
    void* object;
}metal_library_t;

enum gemm_status metal_library_create(
    const metal_device_t* device,
    metal_library_t* library_out);

enum gemm_status metal_library_release(metal_library_t* library);

//function
typedef struct metal_function {
    void* function_object;          // id<MTLFunction>
    void* pipeline_state_object;    // id<MTLComputePipelineState>
    size_t max_threadgroup_threads;
    size_t simdgroup_threads;
    size_t static_threadgroup_memory;
}metal_function_t;


enum gemm_status metal_function_create(
    const metal_library_t* library,
    const char* name,
    metal_function_t* function_out);

enum gemm_status metal_function_release(
    metal_function_t* function);


//buffer
typedef struct metal_buffer {
    void* object; // id<MTLBuffer>
    size_t size;
    void* ptr;
}metal_buffer_t;

enum gemm_status metal_buffer_create(
    const metal_device_t* device,
    size_t size,
    const void* data,
    metal_buffer_t* buffer_out);

enum gemm_status metal_buffer_wrap(
    const metal_device_t* device,
    size_t size,
    const void* data,
    metal_buffer_t* buffer_out);

enum gemm_status metal_buffer_release(
    metal_buffer_t* buffer);
 
    
//command_queue
typedef struct metal_command_queue {
    void* object; // id<MTLCommandQueue>
}metal_command_queue_t;

enum gemm_status metal_command_queue_create(
    const metal_device_t* device,
    metal_command_queue_t* command_queue_out);

enum gemm_status metal_command_queue_release(
    metal_command_queue_t* command_queue);

typedef struct metal_command_buffer {
        void* object; // id<MTLCommandBuffer>
}metal_command_buffer_t;

enum gemm_status metal_command_buffer_create(
    const metal_command_queue_t* command_queue,
    metal_command_buffer_t* command_buffer_out);

enum gemm_status metal_command_buffer_release(
    metal_command_buffer_t* command_buffer);

enum gemm_status metal_command_queue_encode_fill_buffer(
    const metal_command_buffer_t* command_buffer,
    const metal_buffer_t* buffer,
    size_t offset,
    size_t size,
    uint8_t fill_value);

enum gemm_status metal_command_queue_encode_copy_buffer(
    const metal_command_buffer_t* command_buffer,
    const metal_buffer_t* input_buffer,
    size_t input_offset,
    const metal_buffer_t* output_buffer,
    size_t output_offset,
    size_t size);

enum gemm_status metal_command_buffer_encode_launch_kernel(
    const metal_command_buffer_t* command_buffer,
    const metal_function_t* function,
    size_t threadgroup_size_x,
    size_t threadgroup_size_y,
    size_t threadgroup_size_z,
    size_t num_threadgroups_x,
    size_t num_threadgroups_y,
    size_t num_threadgroups_z,
    size_t params_size,
    const void* params,
    size_t num_buffers,
    const metal_buffer_t** buffers,
    const size_t* buffer_offsets);
    
enum gemm_status metal_command_buffer_commit(
    const metal_command_buffer_t* command_buffer);
    
enum gemm_status metal_command_buffer_wait_completion(
    const metal_command_buffer_t* command_buffer,
    double* elapsed_seconds);

    //buffer
typedef struct metal_mps_matrix {
    void* object; // id<MTLBuffer>
    void* descriptor; // MPSMatrixDescriptor
}metal_mps_matrix_t;
enum gemm_status metal_mps_matrix_create(const metal_device_t* device, metal_buffer_t* buff, int m, int n, metal_mps_matrix_t* matrix);
enum gemm_status metal_mps_matrix_release(metal_mps_matrix_t* matrix);

typedef struct metal_mps_gemm_func {
    void* object; // MPSMatrixMultiplication
}metal_mps_gemm_func_t;
enum gemm_status metal_mps_gemm_func_create(const metal_device_t* device, int m, int n, int k, float alpha, float beta, metal_mps_gemm_func_t* func);
enum gemm_status metal_mps_gemm_func_release(metal_mps_gemm_func_t* func);



enum gemm_status metal_gemm(const metal_device_t* device, const metal_command_buffer_t* command_buffer,
    int m, int n, int k,
    float alpha, metal_buffer_t* A, metal_buffer_t* B,
    float beta, metal_buffer_t* C,
    double* elapsed_seconds);

enum gemm_status meta_mps_matmul(const metal_device_t* device, const metal_command_buffer_t* command_buffer,
    metal_mps_gemm_func_t* func,
    metal_mps_matrix_t* A, metal_mps_matrix_t* B,
    metal_mps_matrix_t* C);


#ifdef __cplusplus
}
#endif