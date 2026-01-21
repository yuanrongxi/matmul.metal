#include <metal_atomic>
#include <metal_compute>
#include <metal_integer>
#include <metal_math>
#include <metal_simdgroup>

#include <args.h>

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

//SIMD Native
kernel void metal_softmax_v1(
    constant                    softmax_v1_args_t& args [[ buffer(0) ]],
    const device float*         in [[ buffer(1) ]],
    device float*               out [[ buffer(2) ]],
    uint                        tid [[ thread_position_in_threadgroup ]],
    uint                        threadgroup_size [[ threads_per_threadgroup ]])    
{
        //share memory
        threadgroup float buff[SOFTMAX_BLOCK_SIZE];

        //max reduction
        float max_val= -INFINITY;
        for(uint i = tid; i < args.n; i += SOFTMAX_BLOCK_SIZE) {
            max_val = metal::max(max_val, in[i]);
        }
        buff[tid] = max_val;

        for(uint i = SOFTMAX_BLOCK_SIZE / 2; i >= 1; i /= 2) {
            metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if(tid < i) {
                buff[tid] = metal::max(buff[tid], buff[tid + i]);
            }
        }
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        max_val = buff[0];

        //sum reduction
        float sum_val = 0.0f;
        for(uint i = tid; i < args.n; i += SOFTMAX_BLOCK_SIZE) {
            out[i] = metal::exp(in[i] - max_val);
            sum_val += out[i];
        }
        buff[tid] = sum_val;

        for(uint i = SOFTMAX_BLOCK_SIZE / 2; i >= 1; i /= 2) {
            metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if(tid < i) {
                buff[tid] += buff[tid + i];
            }
        }
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        sum_val = buff[0];

        //softmax normalization
        for(uint i = tid; i < args.n; i += SOFTMAX_BLOCK_SIZE) {
            out[i] /= sum_val;
        }
}

kernel void metal_softmax_v2(
    constant                    softmax_v2_args_t& args [[ buffer(0) ]],
    const device float*         in [[ buffer(1) ]],
    device float*               out [[ buffer(2) ]],
    uint                        sid [[ thread_index_in_simdgroup ]])
    {
        //max reduction
        float max_val= -INFINITY;
        for(uint i = sid; i < args.n; i += WARPSIZE) {
            max_val = metal::max(max_val, in[i]);
        }
        max_val = metal::simd_max(max_val);
        
        //sum reduction
        float sum_val = 0.0f;
        for(uint i = sid; i < args.n; i += WARPSIZE) {
            out[i] = metal::exp(in[i] - max_val);
            sum_val += out[i];
        }
        sum_val = metal::simd_sum(sum_val);

        for(uint i = sid; i < args.n; i += WARPSIZE) {
            out[i] /= sum_val;
        }
}

kernel void metal_softmax_v3(
    constant                    softmax_v3_args_t& args [[ buffer(0) ]],
    const device float*         in [[ buffer(1) ]],
    device float*               out [[ buffer(2) ]],
    uint                        tid [[ thread_position_in_threadgroup ]],
    uint                        sid [[ thread_index_in_simdgroup ]],
    uint                        swap_id [[ simdgroup_index_in_threadgroup ]]) 
{
        threadgroup float buff[V3_BUFF_LEN];

        //max reduction
        float max_val = -INFINITY;
        for(uint i = sid; i < args.n; i += SOFTMAX_V3_BLOCK_SIZE) {
            max_val = metal::max(max_val, in[i]);
        }
        max_val = metal::simd_max(max_val);
        if(sid == 0) buff[sid] = max_val;
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        if(tid == 0) {
            float v = buff[0];
            for(uint i = 1; i < V3_BUFF_LEN; i++) {
                v = metal::max(v, buff[i]);
            }
            buff[0] = v;
        }
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        float offset = buff[0];

        //sum reduction
        float sum_val = 0.0f;
        for(uint i = sid; i < args.n; i += SOFTMAX_V3_BLOCK_SIZE) {
            out[i] = metal::exp(in[i] - offset);
            sum_val += out[i];
        }
        sum_val = metal::simd_sum(sum_val);
        if(sid == 0) buff[sid] = sum_val;
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        if(tid == 0) {
            float s = buff[0];
            for(uint i = 1; i < V3_BUFF_LEN; i++) {
                s += buff[i];
            }
            buff[0] = s;
        }
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        float sum = buff[0];
        
        //softmax normalization
        for(uint i = sid; i < args.n; i += SOFTMAX_V3_BLOCK_SIZE) {
            out[i] /= sum;
        }
}

kernel void metal_softmax_online(
    constant                    softmax_v4_args_t& args [[ buffer(0) ]],
    const device float*         in [[ buffer(1) ]],
    device float*               out [[ buffer(2) ]],
    uint                        tid [[ thread_position_in_threadgroup ]],
    uint                        sid [[ thread_index_in_simdgroup ]],
    uint                        swap_id [[ simdgroup_index_in_threadgroup ]]) 
{
        float max_val = -INFINITY;
        float sum_val = 0.0f;
        float big_val;
        for(uint i = sid; i < args.n; i += WARPSIZE) {
            big_val = metal::max(max_val, in[i]);
            sum_val = sum_val * metal::exp(max_val - big_val) + metal::exp(in[i] - big_val);
            max_val = big_val;
        }
        big_val = metal::simd_max(max_val);
        sum_val = sum_val * metal::exp(max_val - big_val);
        sum_val = metal::simd_sum(sum_val);
        
        for(uint i = sid; i < args.n; i += WARPSIZE) {
            out[i] = metal::exp(in[i] - big_val) / sum_val;
        }
}

kernel void metal_softmax_online_v2(
    constant                    softmax_v4_args_t& args [[ buffer(0) ]],
    const device float*         in [[ buffer(1) ]],
    device float*               out [[ buffer(2) ]],
    uint                        tid [[ thread_position_in_threadgroup ]],
    uint                        sid [[ thread_index_in_simdgroup ]],
    uint                        swap_id [[ simdgroup_index_in_threadgroup ]]) 
{
        threadgroup float max_buff[V3_BUFF_LEN];
        threadgroup float sum_buff[V3_BUFF_LEN];

        //swap reduction
        float max_val = -INFINITY;
        float sum_val = 0.0f;
        float big_val;
        for(uint i = sid; i < args.n; i += SOFTMAX_V3_BLOCK_SIZE) {
            big_val = metal::max(max_val, in[i]);
            sum_val = sum_val * metal::exp(max_val - big_val) + metal::exp(in[i] - big_val);
            max_val = big_val;
        }
        big_val = metal::simd_max(max_val);
        sum_val = sum_val * metal::exp(max_val - big_val);
        sum_val = metal::simd_sum(sum_val);
        if(sid == 0) {
            max_buff[swap_id] = big_val;
            sum_buff[swap_id] = sum_val;
        }
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        //threadgroup reduction
        if(tid == 0) {
            big_val = max_buff[0]; 
            sum_val = sum_buff[0];
            for(uint i = 1; i < V3_BUFF_LEN; i++) {
                big_val = metal::max(big_val, max_buff[i]);
                sum_val = sum_val * metal::exp(max_buff[i] - big_val) + sum_buff[i];
            }

            max_buff[0] = big_val;
            sum_buff[0] = sum_val;
        }
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        //softmax normalization
        max_val = max_buff[0];
        sum_val = sum_buff[0];
        for(uint i = sid; i < args.n; i += SOFTMAX_V3_BLOCK_SIZE) {
            out[i] = metal::exp(in[i] - max_val) / sum_val;
        }
}



