#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <dispatch/dispatch.h>
#include <mach-o/getsect.h>

#include "types.h"
#include "metal.h"

static size_t metal_device_get_core_count(id<MTLDevice> device) {
    if (!device) {
        return 0;
    }

    const uint64_t target_registry_id = [device registryID];

    io_iterator_t it = IO_OBJECT_NULL;
    const kern_return_t kr = IOServiceGetMatchingServices(kIOMainPortDefault,
        IOServiceMatching("IOAccelerator"), &it);
    if (kr != KERN_SUCCESS) {
        printf("failed to find IOAccelerator objects: error %d\n", kr);
        return 0;
    }

    size_t result = 0;
    for (io_object_t obj = IOIteratorNext(it); obj != IO_OBJECT_NULL; obj = IOIteratorNext(it)) {
        uint64_t registry_id = 0;
        if (IORegistryEntryGetRegistryEntryID(obj, &registry_id) == KERN_SUCCESS &&
            registry_id == target_registry_id)
        {
            // Read "gpu-core-count" from this accelerator node
            const CFTypeRef value = IORegistryEntryCreateCFProperty(obj, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
            if (value != NULL) {
                if (CFGetTypeID(value) == CFNumberGetTypeID()) {
                    int32_t n = -1;
                    if (CFNumberGetValue((CFNumberRef) value, kCFNumberSInt32Type, &n) && n > 0) {
                        result = (size_t) n;
                    }
                }
                CFRelease(value);
            }
            IOObjectRelease(obj);
            break;
        }
        IOObjectRelease(obj);
    }

    IOObjectRelease(it);
    return result;    
}

enum gemm_status metal_device_create(metal_device_t* device_out) {
    id<MTLDevice> device_obj = MTLCreateSystemDefaultDevice();
    if (device_obj == nil) {
        printf("failed to create Metal device\n");
        return gemm_unsupported_system;
    }

    device_out->object = (void*) device_obj;
    device_out->num_cores = metal_device_get_core_count(device_obj);
    device_out->max_buffer_size = (size_t) [device_obj maxBufferLength];
    device_out->max_threadgroup_memory = (size_t) [device_obj maxThreadgroupMemoryLength];
    const MTLSize max_threadgroup_threads = [device_obj maxThreadsPerThreadgroup];
    device_out->max_threadgroup_threads_x = (size_t) max_threadgroup_threads.width;
    device_out->max_threadgroup_threads_y = (size_t) max_threadgroup_threads.height;
    device_out->max_threadgroup_threads_z = (size_t) max_threadgroup_threads.depth;
 
    return gemm_success;
}

enum gemm_status metal_device_release(metal_device_t* device) {
    if (device->object != NULL) {
        id<MTLDevice> device_obj = (id<MTLDevice>) device->object;
        [device_obj release];
    }
    memset(device, 0, sizeof(metal_device_t));
    return gemm_success;
}

extern const struct mach_header_64 __dso_handle;

enum gemm_status metal_library_create(const metal_device_t* device, metal_library_t* library_out) {
    enum gemm_status status = gemm_success;
    id<MTLDevice> device_obj = (id<MTLDevice>) device->object;
    id<MTLLibrary> library_obj = nil;
    NSError* error_obj = nil;
    NSString* error_string_obj = nil;
    dispatch_data_t library_blob = NULL;

    unsigned long library_size = 0;
    uint8_t* library_data = getsectiondata(&__dso_handle, "__METAL", "__shaders", &library_size);
    if (library_data != NULL) {
        library_blob = dispatch_data_create(library_data, library_size, NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        library_obj = [device_obj newLibraryWithData:library_blob error:&error_obj];
        if (library_obj == nil) {
            error_string_obj = [error_obj localizedDescription];
            printf("failed to create Metal library: %s\n", [error_string_obj UTF8String]);
            status = gemm_unsupported_system;
            goto cleanup;
        }
    } else {
        // Fall-back to loading from the bundle
        library_obj = [device_obj newDefaultLibrary];
        if (library_obj == nil) {
            printf("failed to create Metal default library\n");
            status = gemm_unsupported_system;
            goto cleanup;
        }
    }

    *library_out = (metal_library_t) {
        .object = (void*) library_obj,
    };

cleanup:
    if (library_blob != NULL) {
        dispatch_release(library_blob);
    }
    if (error_string_obj != nil) {
        [error_string_obj release];
    }
    if (error_obj != nil) {
        [error_obj release];
    }
    return status;
}

enum gemm_status metal_library_release(metal_library_t* library) {
    if (library->object != NULL) {
        id<MTLLibrary> library_obj = (id<MTLLibrary>) library->object;
        [library_obj release];
    }
    memset(library, 0, sizeof(metal_library_t));
    return gemm_success;
}

enum gemm_status metal_function_create(const metal_library_t* library, const char* name, metal_function_t* function_out) {
    NSString* name_obj = nil;
    NSError* error_obj = nil;
    NSString* error_string_obj = nil;
    id<MTLFunction> function_obj = nil;
    enum gemm_status status = gemm_success;

    id<MTLLibrary> library_obj = (id<MTLLibrary>) library->object;
    name_obj = [NSString stringWithUTF8String:name];
    function_obj = [library_obj newFunctionWithName:name_obj];
    if (function_obj == nil) {
        printf("failed to create Metal function %s\n", name);
        status = gemm_unsupported_system;
        goto cleanup;
    }

    id<MTLDevice> device_obj = [library_obj device];
    id<MTLComputePipelineState> pipeline_state_obj = [device_obj newComputePipelineStateWithFunction:function_obj error:&error_obj];
    if (pipeline_state_obj == nil) {
        error_string_obj = [error_obj localizedDescription];
        printf("failed to create Metal compute pipeline state for function %s: %s\n",
            name, [error_string_obj UTF8String]);
        status = gemm_unsupported_system;
        goto cleanup;
    }

    // Commit
    function_out->function_object = function_obj;
    function_out->pipeline_state_object = pipeline_state_obj;
    function_out->max_threadgroup_threads = (size_t) [pipeline_state_obj maxTotalThreadsPerThreadgroup];
    function_out->simdgroup_threads = (size_t) [pipeline_state_obj threadExecutionWidth];
    function_out->static_threadgroup_memory = (size_t) [pipeline_state_obj staticThreadgroupMemoryLength];

    function_obj = nil;
    pipeline_state_obj = nil;

cleanup:
    if (name_obj != nil) {
        [name_obj release];
    }
    if (function_obj != nil) {
        [function_obj release];
    }
    if (error_string_obj != nil) {
        [error_string_obj release];
    }
    if (error_obj != nil) {
        [error_obj release];
    }
    return status;
}

enum gemm_status metal_function_release(metal_function_t* function) {
    if (function->pipeline_state_object != NULL) {
        id<MTLComputePipelineState> pipeline_state_obj = (id<MTLComputePipelineState>) function->pipeline_state_object;
        [pipeline_state_obj release];
    }
    if (function->function_object != NULL) {
        id<MTLFunction> function_obj = (id<MTLFunction>) function->function_object;
        [function_obj release];
    }
    memset(function, 0, sizeof(metal_function_t));
    return gemm_success;
}

enum gemm_status metal_buffer_create(const metal_device_t* device, size_t size, const void* data, metal_buffer_t* buffer_out) {
    id<MTLDevice> device_obj = (id<MTLDevice>) device->object;
    id<MTLBuffer> buffer_obj = nil;
    if (data != NULL) {
        buffer_obj = [device_obj newBufferWithBytes:data length:size options:MTLResourceStorageModeShared];
    } else {
        buffer_obj = [device_obj newBufferWithLength:size options:MTLResourceStorageModeShared];
    }
    if (buffer_obj == nil) {
        printf("failed to create Metal buffer of size %zu\n", size);
        return gemm_unsupported_system;
    }
    buffer_out->object = (void*) buffer_obj;
    buffer_out->size = size;
    buffer_out->ptr = [buffer_obj contents];
    return gemm_success;
}

enum gemm_status metal_buffer_wrap(const metal_device_t* device,size_t size,const void* data, metal_buffer_t* buffer_out) {
    id<MTLDevice> device_obj = (id<MTLDevice>) device->object;
    id<MTLBuffer> buffer_obj = [device_obj newBufferWithBytesNoCopy:(void*) data length:size options:MTLResourceStorageModeShared deallocator:nil];
    if (buffer_obj == nil) {
        printf("failed to wrap Metal buffer of size %zu\n", size);
        return gemm_unsupported_system;
    }
    buffer_out->object = (void*) buffer_obj;
    buffer_out->size = size;
    buffer_out->ptr = (void*) data;
    return gemm_success;
}

enum gemm_status metal_buffer_release(metal_buffer_t* buffer) {
    if (buffer->object != NULL) {
        id<MTLBuffer> buffer_obj = (id<MTLBuffer>) buffer->object;
        [buffer_obj release];
    }
    memset(buffer, 0, sizeof(metal_buffer_t));
    return gemm_success;
}

enum gemm_status metal_command_queue_create(const metal_device_t* device, metal_command_queue_t* command_queue_out){
    id<MTLDevice> device_obj = (id<MTLDevice>) device->object;
    id<MTLCommandQueue> command_queue_obj = [device_obj newCommandQueue];
    if (command_queue_obj == nil) {
        printf("failed to create Metal command queue\n");
        return gemm_unsupported_system;
    }
    command_queue_out->object = (void*) command_queue_obj;
    return gemm_success;
}

enum gemm_status metal_command_queue_release(metal_command_queue_t* command_queue) {
    if (command_queue->object != NULL) {
        id<MTLCommandQueue> command_queue_obj = (id<MTLCommandQueue>) command_queue->object;
        [command_queue_obj release];
    }
    memset(command_queue, 0, sizeof(metal_command_queue_t));
    return gemm_success;
}

enum gemm_status metal_command_buffer_create(const metal_command_queue_t* command_queue, metal_command_buffer_t* command_buffer_out) {
    id<MTLCommandQueue> command_queue_obj = (id<MTLCommandQueue>) command_queue->object;
    id<MTLCommandBuffer> command_buffer_obj = [command_queue_obj commandBuffer];
    if (command_buffer_obj == nil) {
        printf("failed to create Metal command buffer\n");
        return gemm_unsupported_system;
    }
    [command_buffer_obj retain];
    command_buffer_out->object = (void*) command_buffer_obj;
    return gemm_success;
}

enum gemm_status metal_command_buffer_encode_fill_buffer(
    const metal_command_buffer_t* command_buffer,
    const metal_buffer_t* buffer,
    size_t offset,
    size_t size,
    uint8_t fill_value) {
    if (command_buffer->object == NULL) {
        return gemm_invalid_state;
    }
    if (buffer->object == NULL) {
        return gemm_invalid_argument;
    }

    id<MTLCommandBuffer> command_buffer_obj = (id<MTLCommandBuffer>) command_buffer->object;
    id<MTLBuffer> buffer_obj = (id<MTLBuffer>) buffer->object;

    id<MTLBlitCommandEncoder> command_encoder_obj = [command_buffer_obj blitCommandEncoder];

    const NSRange range = NSMakeRange((NSUInteger) offset, (NSUInteger) size);
    [command_encoder_obj fillBuffer:buffer_obj range:range value:fill_value];
    [command_encoder_obj endEncoding];

    return gemm_success;
}

enum gemm_status metal_command_buffer_encode_copy_buffer(
    const metal_command_buffer_t* command_buffer,
    const metal_buffer_t* input_buffer,
    size_t input_offset,
    const metal_buffer_t* output_buffer,
    size_t output_offset,
    size_t size) {
    if (command_buffer->object == NULL) {
        return gemm_invalid_state;
    }
    if (input_buffer->object == NULL) {
        return gemm_invalid_argument;
    }
    if (output_buffer->object == NULL) {
        return gemm_invalid_argument;
    }

    id<MTLCommandBuffer> command_buffer_obj = (id<MTLCommandBuffer>) command_buffer->object;
    id<MTLBuffer> input_buffer_obj = (id<MTLBuffer>) input_buffer->object;
    id<MTLBuffer> output_buffer_obj = (id<MTLBuffer>) output_buffer->object;

    id<MTLBlitCommandEncoder> command_encoder_obj = [command_buffer_obj blitCommandEncoder];

    [command_encoder_obj copyFromBuffer:input_buffer_obj sourceOffset:(NSUInteger) input_offset
                         toBuffer:output_buffer_obj destinationOffset:(NSUInteger) output_offset
                         size:(NSUInteger) size];
    [command_encoder_obj endEncoding];

    return gemm_success;
}

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
    const size_t* buffer_offsets) {
    if (command_buffer->object == NULL || function->pipeline_state_object == NULL) {
        return gemm_invalid_state;
    }

    id<MTLCommandBuffer> command_buffer_obj = (id<MTLCommandBuffer>) command_buffer->object;
    id<MTLComputePipelineState> pipeline_state_obj = (id<MTLComputePipelineState>) function->pipeline_state_object;

    id<MTLComputeCommandEncoder> command_encoder_obj = [command_buffer_obj computeCommandEncoder];

    // Set kernel arguments
    [command_encoder_obj setComputePipelineState:pipeline_state_obj];
    [command_encoder_obj setBytes:params length:params_size atIndex:0];
    for (size_t i = 0; i < num_buffers; ++i) {
        id<MTLBuffer> buffer_obj = (id<MTLBuffer>) buffers[i]->object;
        const NSUInteger offset = buffer_offsets == NULL ? 0 : (NSUInteger) buffer_offsets[i];
        [command_encoder_obj setBuffer:buffer_obj offset:offset atIndex:i + 1];
    }

    // Dispatch kernel
    const MTLSize threadgroup_size = MTLSizeMake(threadgroup_size_x, threadgroup_size_y, threadgroup_size_z);
    const MTLSize num_threadgroups = MTLSizeMake(num_threadgroups_x, num_threadgroups_y, num_threadgroups_z);
    [command_encoder_obj dispatchThreadgroups:num_threadgroups threadsPerThreadgroup:threadgroup_size];
    [command_encoder_obj endEncoding];

    return gemm_success;
}

enum gemm_status metal_command_buffer_commit(const metal_command_buffer_t* command_buffer) {
    if (command_buffer->object == NULL) {
        return gemm_invalid_state;
    }

    id<MTLCommandBuffer> command_buffer_obj = (id<MTLCommandBuffer>) command_buffer->object;
    [command_buffer_obj commit];

    return gemm_success;
}

enum gemm_status metal_command_buffer_wait_completion(const metal_command_buffer_t* command_buffer, double* elapsed_seconds) {
    if (command_buffer->object == NULL) {
        return gemm_invalid_state;
    }

    id<MTLCommandBuffer> command_buffer_obj = (id<MTLCommandBuffer>) command_buffer->object;
    [command_buffer_obj waitUntilCompleted];
    if (elapsed_seconds != NULL) {
        const CFTimeInterval start_time = [command_buffer_obj GPUStartTime];
        const CFTimeInterval end_time = [command_buffer_obj GPUEndTime];
        *elapsed_seconds = (double) end_time - (double) start_time;
    }
    return gemm_success;
}

enum gemm_status metal_command_buffer_release(metal_command_buffer_t* command_buffer) {
    if (command_buffer->object != NULL) {
        id<MTLCommandBuffer> command_buffer_obj = (id<MTLCommandBuffer>) command_buffer->object;
        [command_buffer_obj release];
    }
    memset(command_buffer, 0, sizeof(metal_command_buffer_t));
    return gemm_success;
}



enum gemm_status metal_gemm(const metal_device_t* device, const metal_command_buffer_t* command_buffer,
    int m, int n, int k,
    float alpha, metal_buffer_t* A, metal_buffer_t* B,
    float beta, metal_buffer_t* C,
    double* elapsed_seconds) {
    
    if (command_buffer->object == NULL || device->object == NULL) {
        return gemm_invalid_state;
    }

    id<MTLDevice> device_obj = (id<MTLDevice>) device->object;
    id<MTLCommandBuffer> command_buffer_obj = (id<MTLCommandBuffer>) command_buffer->object;

    id<MTLBuffer> bufferA = (id<MTLBuffer>) A->object;
    id<MTLBuffer> bufferB = (id<MTLBuffer>) B->object;
    id<MTLBuffer> bufferC = (id<MTLBuffer>) C->object;

    MPSMatrixDescriptor *dA = [MPSMatrixDescriptor matrixDescriptorWithRows:m
                                                                    columns:k
                                                                    rowBytes:k * sizeof(float)
                                                                    dataType:MPSDataTypeFloat32];

    MPSMatrixDescriptor *dB = [MPSMatrixDescriptor matrixDescriptorWithRows:k
                                                                    columns:n
                                                                    rowBytes:n * sizeof(float)
                                                                    dataType:MPSDataTypeFloat32];

    MPSMatrixDescriptor *dC = [MPSMatrixDescriptor matrixDescriptorWithRows:m
                                                                    columns:n
                                                                    rowBytes:n * sizeof(float)
                                                                    dataType:MPSDataTypeFloat32];

    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:dA];
    MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:dB];
    MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:dC];

    MPSMatrixMultiplication *gemm =[[MPSMatrixMultiplication alloc] initWithDevice:device_obj
                                                transposeLeft:NO
                                                transposeRight:NO
                                                resultRows:m
                                                resultColumns:n
                                                interiorColumns:k
                                                alpha:alpha
                                                beta:beta];  
     
    [gemm encodeToCommandBuffer:command_buffer_obj leftMatrix:matA rightMatrix:matB resultMatrix:matC];

    [command_buffer_obj commit];
    [command_buffer_obj waitUntilCompleted];
    if (elapsed_seconds != NULL) {
        *elapsed_seconds = (double) [command_buffer_obj GPUEndTime] - (double) [command_buffer_obj GPUStartTime];
    }

    [dB release];
    [dC release];
    [dA release];

    [matA release];
    [matB release];
    [matC release];
    [gemm release];

    return gemm_success;
}

enum gemm_status metal_mps_matrix_create(const metal_device_t* device, metal_buffer_t* buff, int m, int n, metal_mps_matrix_t* matrix) {
    if (device->object == NULL || buff->object == NULL) {
        return gemm_invalid_state;
    }

    id<MTLDevice> device_obj = (id<MTLDevice>) device->object;
    id<MTLBuffer> buffer_obj = (id<MTLBuffer>) buff->object;
    MPSMatrixDescriptor *descriptor = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n rowBytes:n * sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrix *matrix_obj = [[MPSMatrix alloc] initWithBuffer:buffer_obj descriptor:descriptor];
    matrix->object = (void*) matrix_obj;
    matrix->descriptor = (void*) descriptor;
    return gemm_success;
}

enum gemm_status metal_mps_matrix_release(metal_mps_matrix_t* matrix) {
    if (matrix->object != NULL) {
        MPSMatrix* matrix_obj = (MPSMatrix*) matrix->object;
        [matrix_obj release];
    }
    matrix->object = NULL;

    if (matrix->descriptor != NULL) {
        MPSMatrixDescriptor* descriptor = (MPSMatrixDescriptor*) matrix->descriptor;
        [descriptor release];
    }
    matrix->descriptor = NULL;
    return gemm_success;
}

enum gemm_status metal_mps_gemm_func_create(const metal_device_t* device, int m, int n, int k, float alpha, float beta, metal_mps_gemm_func_t* func) {
    if (device->object == NULL) {
        return gemm_invalid_state;
    }

    id<MTLDevice> device_obj = (id<MTLDevice>) device->object;
    MPSMatrixMultiplication *gemm = [[MPSMatrixMultiplication alloc] initWithDevice:device_obj transposeLeft:NO transposeRight:NO resultRows:m resultColumns:n interiorColumns:k alpha:alpha beta:beta];
    func->object = (void*) gemm;
    return gemm_success;
}

enum gemm_status metal_mps_gemm_func_release(metal_mps_gemm_func_t* func) {     
    if (func->object != NULL) {
        MPSMatrixMultiplication* gemm = (MPSMatrixMultiplication*) func->object;
        [gemm release];
    }
    func->object = NULL;
    return gemm_success;
}

enum gemm_status meta_mps_matmul(const metal_device_t* device, const metal_command_buffer_t* command_buffer, metal_mps_gemm_func_t* func,
    metal_mps_matrix_t* A, metal_mps_matrix_t* B,
    metal_mps_matrix_t* C) {
    
    if (device->object == NULL || command_buffer->object == NULL || func->object == NULL
    || A->object == NULL || B->object == NULL || C->object == NULL
    || A->descriptor == NULL || B->descriptor == NULL || C->descriptor == NULL) {
        return gemm_invalid_state;
    }

    id<MTLDevice> device_obj = (id<MTLDevice>) device->object;
    id<MTLCommandBuffer> command_buffer_obj = (id<MTLCommandBuffer>) command_buffer->object;
    MPSMatrixMultiplication* gemm = (MPSMatrixMultiplication*) func->object;

    [gemm encodeToCommandBuffer:command_buffer_obj leftMatrix:(MPSMatrix*) A->object rightMatrix:(MPSMatrix*) B->object resultMatrix:(MPSMatrix*) C->object];
    return gemm_success;
}