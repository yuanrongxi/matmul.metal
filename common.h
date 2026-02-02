#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "metal.h"

// Generates random float32 numbers for a buffer
void generate_random_f32(float* d, size_t size);

// Returns current time in microseconds
uint64_t get_time_us();

// Validates GEMM result by comparing with CPU result
void validate_gemm_result(float* c, float* c_cpu, int n);

// Creates Metal context (device, library, function, command queue)
enum gemm_status create_metal(metal_device_t* device, metal_library_t* library, 
                                        metal_function_t* function, metal_command_queue_t* command_queue, const char* name);

// Releases Metal context
void release_metal(metal_device_t* device, metal_library_t* library, 
    metal_function_t* function, metal_command_queue_t* command_queue);

// Prints float buffer
void print_f32(float* ptr, int n);

#endif
