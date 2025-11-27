#include <metal_atomic>
#include <metal_compute>
#include <metal_integer>
#include <metal_math>
#include <metal_simdgroup>

#include <args.h>

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

//Native 
kernel void metal_matmul_v1(
    constant                    matmul_v1_args_t& args [[ buffer(0) ]],
    const device float*         a [[ buffer(1) ]],
    const device float*         b [[ buffer(2) ]],
    device float*               c [[ buffer(3) ]],
    uint                        i [[ threadgroup_position_in_grid ]],
    uint                        j [[ thread_position_in_threadgroup ]],
    uint                        gid [[ thread_position_in_grid ]])
{
    if(gid >= args.m * args.n) return;

    float sum = 0.0f;
    for(uint l = 0; l < args.k; l++) {
        float v1 = a[i * args.k + l];
        float v2 = b[l * args.n + j];
        sum = metal::fma(v1, v2, sum);
    }

    c[gid] = args.alpha * sum + args.beta * c[gid];
}

//SIMD Native
kernel void metal_matmul_v2(
    constant                    matmul_v2_args_t& args [[ buffer(0) ]],
    const device float*         a [[ buffer(1) ]],
    const device float*         b [[ buffer(2) ]],
    device float*               c [[ buffer(3) ]], 
    uint                        threadgroup_id [[ threadgroup_position_in_grid ]],
    uint                        simd_gid [[ simdgroup_index_in_threadgroup ]],
    uint                        simd_tid [[ thread_index_in_simdgroup ]],
    uint                        num_simdgroups [[ simdgroups_per_threadgroup ]])
{
    const uint simdgroup_size = 32;

    uint simd_index = threadgroup_id * num_simdgroups + simd_gid;
    uint i = simd_index / args.n;
    uint j = simd_index % args.n;

    uint count = (args.k + simdgroup_size - 1 - simd_tid) / simdgroup_size;
    a += i * args.k + simd_tid;
    b += simd_tid * args.n + j;

    float sum = 0.0f;
    for(uint i = 0; i < count; i++) {
        float v1 = *a;
        float v2 = *b;
        sum  = metal::fma(v1, v2, sum);

        a += simdgroup_size;
        b += simdgroup_size * args.n;
    }
    sum = metal::simd_sum(sum);

    if(metal::simd_is_first()){
        c[simd_index] = args.alpha * sum + args.beta * c[simd_index];
    }
}

//Block share memory cache
kernel void metal_matmul_v3(
    constant                    matmul_v3_args_t& args [[ buffer(0) ]],
    const device float*         a [[ buffer(1) ]],
    const device float*         b [[ buffer(2) ]],
    device float*               c [[ buffer(3) ]],
    uint2                       group_id [[ threadgroup_position_in_grid ]],
    uint2                       thread_id [[ thread_position_in_threadgroup ]])
{
    threadgroup float as[BLOCK_SIZE * BLOCK_SIZE];
    threadgroup float bs[BLOCK_SIZE * BLOCK_SIZE]; 

    //block position
    uint row = group_id.y;
    uint col = group_id.x;

    //thread position, block‘s x，y
    uint thread_row = thread_id.x / BLOCK_SIZE;
    uint thread_col = thread_id.x % BLOCK_SIZE;
    
    //set position (a, b, c) 
    a += row * args.k * BLOCK_SIZE;
    b += col * BLOCK_SIZE;
    c += row * args.n *BLOCK_SIZE + col * BLOCK_SIZE;

    float sum = 0.0f;
    for(uint k = 0; k < args.k; k += BLOCK_SIZE) {
        //load block data from global memory  to share memeory
        as[thread_row * BLOCK_SIZE + thread_col] = a[thread_row * args.k + thread_col];
        bs[thread_row * BLOCK_SIZE + thread_col] = b[thread_row * args.n + thread_col];

        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        a += BLOCK_SIZE;
        b += BLOCK_SIZE * args.n;

        //dot
        for(uint i = 0; i < BLOCK_SIZE; i++) {
            sum = metal::fma(as[thread_row * BLOCK_SIZE + i], bs[i * BLOCK_SIZE + thread_col], sum);
        }

        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    c[thread_row * args.n + thread_col] = metal::fma(sum, args.alpha, c[thread_row * args.n + thread_col] * args.beta);
}

//1D Block tiling
kernel void metal_matmul_v4(
    constant                    matmul_v4_args_t& args [[ buffer(0) ]],
    const device float*         a [[ buffer(1) ]],
    const device float*         b [[ buffer(2) ]],
    device float*               c [[ buffer(3) ]],
    uint2                       group_id [[ threadgroup_position_in_grid ]],
    uint2                       thread_id [[ thread_position_in_threadgroup ]])
{
    //block position
    const uint row = group_id.y;
    const uint col = group_id.x;

    // each warp will calculate 32*TM elements, with 32 being the columnar dim.
    const uint thread_row = thread_id.x / BN_SIZE;
    const uint thread_col = thread_id.x % BN_SIZE;

    threadgroup float as[BM_SIZE * BK_SIZE];
    threadgroup float bs[BK_SIZE * BN_SIZE]; 

    a += row * BM_SIZE * args.k;
    b += col * BN_SIZE;
    c += row * BM_SIZE * args.n + col * BN_SIZE;

    uint a_row = thread_id.x / BK_SIZE;
    uint a_col = thread_id.x % BK_SIZE;
    uint b_row = thread_id.x / BN_SIZE;
    uint b_col = thread_id.x % BN_SIZE;

    float thread_result[TM_SIZE] = {0.0};
    for(uint k = 0; k < args.k; k += BK_SIZE) {
        //load block data from global memory  to share memeory
        as[a_row * BK_SIZE + a_col] = a[a_row * args.k + a_col];
        bs[b_row * BN_SIZE + b_col] = b[b_row * args.n + b_col];

        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        a += BK_SIZE;
        b += BK_SIZE * args.n;

        for (uint i = 0; i < BK_SIZE; i++) {
            float b_val = bs[i * BN_SIZE + thread_col];

            for(uint idx = 0; idx < TM_SIZE; idx ++) {
                float a_val = as[(thread_row * TM_SIZE + idx) * BK_SIZE + i];
                thread_result[idx] = metal::fma(a_val, b_val, thread_result[idx]); 
            }
        }

        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    for(uint idx = 0; idx < TM_SIZE; idx++){
        c[(thread_row * TM_SIZE + idx) * args.n + thread_col] =
        metal::fma(thread_result[idx], args.alpha, c[(thread_row * TM_SIZE + idx) * args.n + thread_col] * args.beta);
    }
}

//2D Block tiling
kernel void metal_matmul_v5(
    constant                    matmul_v5_args_t& args [[ buffer(0) ]],
    const device float*         a [[ buffer(1) ]],
    const device float*         b [[ buffer(2) ]],
    device float*               c [[ buffer(3) ]],
    uint2                       group_id [[ threadgroup_position_in_grid ]],
    uint2                       thread_id [[ thread_position_in_threadgroup ]])
{
    //block position
    const uint row = group_id.y;
    const uint col = group_id.x;

    //group elements
    const uint block_tile = BM * BN;
    //A thread is responsible for calculating TM*TN elements in the blocktile
    const uint thread_tile = block_tile / (TM * TN);

    // each warp will calculate 32*TM elements, with 32 being the columnar dim.
    const uint thread_row = thread_id.x / (BN/TN);
    const uint thread_col = thread_id.x % (BN/TN);

    threadgroup float as[BM * BK];
    threadgroup float bs[BK * BN]; 

    a += row * BM * args.k;
    b += col * BN;
    c += row * BM * args.n + col * BN;

    uint a_row = thread_id.x / BK;
    uint a_col = thread_id.x % BK;
    uint a_stride = thread_tile / BK;

    uint b_row = thread_id.x / BN;
    uint b_col = thread_id.x % BN;
    uint b_stride = thread_tile / BN;

    float thread_result[TM * TN] = {0.0f};
    float reg_m[TM] = {0.0f};
    float reg_n[TN] = {0.0f};

    for(uint k = 0; k < args.k; k+= BK) {
        for(uint i = 0; i < BM; i += a_stride){
            as[(a_row + i) * BK + a_col] = a[(a_row + i) * args.k + a_col];
        }

        for(uint i = 0; i < BK; i += b_stride){
            bs[(b_row + i) * BN + b_col] = b[(b_row + i) * args.n + b_col];
        }

        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        a += BK;
        b += BK * args.n;

        for(uint di = 0; di < BK; di ++) {
            for (uint i = 0; i < TM; i++) {
                reg_m[i] = as[(thread_row * TM + i) * BK + di];
            }

            for(uint i = 0; i < TN; i++) {
                reg_n[i] = bs[di * BN + thread_col * TN + i];
            }

            for(uint i = 0; i < TM; i ++) {
                for (uint j = 0; j < TN; j++) {
                    thread_result[i * TN + j] = metal::fma(reg_m[i], reg_n[j], thread_result[i * TN + j]);
                }
            }
        }

        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    for(uint i = 0; i < TM; i++) {
        for(uint j = 0; j < TN; j++) {
            uint idx = (thread_row * TM + i) * args.n + thread_col * TN  + j;
            c[idx] = metal::fma(thread_result[i * TN + j], args.alpha, c[idx] * args.beta);
        }
    }
}

//vectorize
kernel void metal_matmul_v6(
    constant                    matmul_v6_args_t& args [[ buffer(0) ]],
    const device float*         a [[ buffer(1) ]],
    const device float*         b [[ buffer(2) ]],
    device float*               c [[ buffer(3) ]],
    uint2                       group_id [[ threadgroup_position_in_grid ]],
    uint2                       thread_id [[ thread_position_in_threadgroup ]])
{
    //block position
    const uint row = group_id.y;
    const uint col = group_id.x;

    // each warp will calculate 32*TM elements, with 32 being the columnar dim.
    const uint thread_row = thread_id.x / (BN/TN);
    const uint thread_col = thread_id.x % (BN/TN);

    threadgroup float as[BM * BK];
    threadgroup float bs[BK * BN]; 

    a += row * BM * args.k;
    b += col * BN;
    c += row * BM * args.n + col * BN;

    uint a_row = thread_id.x / (BK / 4);
    uint a_col = thread_id.x % (BK / 4);

    uint b_row = thread_id.x / (BN / 4);
    uint b_col = thread_id.x % (BN / 4);

    float result[TM * TN] = {0.0f};
    float reg_m[TM] = {0.0f};
    float reg_n[TN] = {0.0f};

    for(uint k = 0; k < args.k; k += BK) {
        float4 tmp = reinterpret_cast<device const float4*>(&a[a_row * args.k + a_col * 4])[0];
        as[(a_col * 4 + 0) * BM + a_row] = tmp.x;
        as[(a_col * 4 + 1) * BM + a_row] = tmp.y;
        as[(a_col * 4 + 2) * BM + a_row] = tmp.z;
        as[(a_col * 4 + 3) * BM + a_row] = tmp.w;

        reinterpret_cast<threadgroup float4*>(&bs[b_row * BN + b_col * 4])[0] = 
                        reinterpret_cast<device const float4*>(&b[b_row * args.n + b_col * 4])[0];

        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        a += BK;
        b += BK * args.n;

        for(uint di = 0; di < BK; di++) {
            for(uint i = 0; i < TM; i++){
                reg_m[i] = as[di * BM + thread_row * TM + i];
            }

            for(uint i = 0; i < TN; i ++) {
                reg_n[i] = bs[di * BN + thread_col * TN + i];
            }

            for(uint i = 0; i < TM; i ++) {
                for (uint j = 0; j < TN; j++) {
                    result[i * TN + j] = metal::fma(reg_m[i], reg_n[j], result[i * TN + j]);
                }
            } 
        }

        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    for(uint i = 0; i < TM; i++) {
        for(uint j = 0; j < TN; j+=4) {
            uint idx = (thread_row * TM + i) * args.n + thread_col * TN  + j;
                
            float4 res = reinterpret_cast<device float4*>(&c[idx])[0];
            res.x = metal::fma(result[i * TN + j], args.alpha, res.x * args.beta);
            res.y = metal::fma(result[i * TN + j + 1], args.alpha, res.y * args.beta);
            res.z = metal::fma(result[i * TN + j + 2], args.alpha, res.z * args.beta);
            res.w = metal::fma(result[i * TN + j + 3], args.alpha, res.w * args.beta);

            reinterpret_cast<device float4*>(&c[idx])[0] = res;
        }
    }
}

inline void load_from_gmem(const device float* A, const device float* B, 
    uint M, uint K, uint N,
    threadgroup float* as, threadgroup float* bs, 
    uint a_row, uint a_col, uint b_row, uint b_col,
    uint stride_a, uint stride_b) {
    //load A
    for(uint os = 0; os < WBM; os += stride_a) {
        float4 val = reinterpret_cast<const device float4*>(&A[(a_row + os) * K + a_col * 4])[0];
        as[(a_col * 4 + 0) * WBM + a_row + os] = val.x;
        as[(a_col * 4 + 1) * WBM + a_row + os] = val.y;
        as[(a_col * 4 + 2) * WBM + a_row + os] = val.z;
        as[(a_col * 4 + 3) * WBM + a_row + os] = val.w;
    }

    //load B
    for(uint os = 0; os < WBK; os += stride_b) {
        reinterpret_cast<threadgroup float4*>(&bs[(b_row + os) * WBN + b_col * 4])[0] = 
                reinterpret_cast<const device float4*>(&B[(b_row + os) * N + b_col * 4])[0];
    }
}

inline void process_smem(threadgroup float* as, threadgroup float* bs, thread float* reg_m, thread float* reg_n, thread float* result,
    uint warp_row, uint warp_col, const uint thread_row_in_warp, const uint thread_col_in_warp) {
    for(uint di = 0; di < WBK; di++) {
        for(uint row_idx = 0; row_idx < WMITER; row_idx++) {
            for(uint i = 0; i < WTM; i++) {
                reg_m[row_idx * WTM + i] = 
                        as[di * WBM + warp_row * WM + row_idx * WSUBM + thread_row_in_warp * WTM + i];
            }
        }

        for(uint col_idx = 0; col_idx < WNITER; col_idx++) {
            for(uint i = 0; i < WTN; i++) {
                reg_n[col_idx * WTN + i] = 
                        bs[di * WBN + warp_col * WN + col_idx * WSUBN + thread_col_in_warp * WTN + i];
            }
        }

        //dot
        for(uint row_idx = 0; row_idx < WMITER; row_idx++) {
            for(uint col_idx = 0; col_idx < WNITER; col_idx++) {
                for(uint i = 0; i < WTM; i++) {
                    for(uint j = 0; j < WTN; j++) {
                        uint idx = (row_idx * WTM + i) * (WTN * WNITER) + col_idx * WTN + j;
                        result[idx] = metal::fma(reg_m[row_idx * WTM + i], reg_n[col_idx * WTN + j], result[idx]);
                    }
                }
            }
        }
    }
}

kernel void metal_matmul_v7(
    constant                    matmul_v7_args_t& args [[ buffer(0) ]],
    const device float*         a [[ buffer(1) ]],
    const device float*         b [[ buffer(2) ]],
    device float*               c [[ buffer(3) ]],
    uint2                       group_id [[ threadgroup_position_in_grid ]],
    uint2                       thread_id [[ thread_position_in_threadgroup ]],
    uint                        simd_id [[ thread_index_in_simdgroup ]], 
    uint                        simdgroup_idx [[ simdgroup_index_in_threadgroup ]])
{
    //block position
    const uint row = group_id.y;
    const uint col = group_id.x;

    const uint warp_row = simdgroup_idx / (WBN / WN);
    const uint warp_col = simdgroup_idx % (WBN / WN);

    const uint thread_row_in_warp = simd_id / (WSUBN / WTN);
    const uint thread_col_in_warp = simd_id % (WSUBN / WTN);

    threadgroup float as[WBM * WBK];
    threadgroup float bs[WBK * WBN];
    
    a += row * WBM * args.k;
    b += col * WBN;
    c += (row * WBM + warp_row * WM) * args.n + col * WBN + warp_col * WN;

    uint a_row = thread_id.x / (WBK /4);
    uint a_col = thread_id.x % (WBK /4);
    uint a_stride = NUM_THREADS * 4 / WBK;

    uint b_row = thread_id.x / (WBN /4);
    uint b_col = thread_id.x % (WBN /4);
    uint b_stride = NUM_THREADS * 4 / WBN;

    float result[WTM * WTN * WNITER * WMITER] = {0.0f};
    float reg_m[WTM * WMITER] = {0.0f};
    float reg_n[WTN * WNITER] = {0.0f};
    
    for(uint k = 0; k < args.k; k += WBK) {
        load_from_gmem(a, b, args.m, args.k, args.n, 
                        as, bs, a_row, a_col, b_row, b_col, 
                        a_stride, b_stride);

        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        process_smem(as, bs, reg_m, reg_n, result, 
                        warp_row, warp_col, thread_row_in_warp, thread_col_in_warp);
    
        a += WBK;
        b += WBK * args.n;
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    for(uint row_idx = 0; row_idx < WMITER; row_idx ++) {
        for(uint col_idx = 0; col_idx < WNITER; col_idx ++) {
            device float* ic = c + (row_idx * WSUBM) * args.n + col_idx * WSUBN; 

            for(uint i = 0; i < WTM; i++) {
                uint idx = (thread_row_in_warp * WTM + i) *args.n + thread_col_in_warp * WTN;
                uint res_idx = (row_idx * WTM + i) * (WNITER * WTN) + col_idx * WTN;

                for(uint j = 0; j < WTN; j ++) {
                    ic[idx + j] = metal::fma(result[res_idx + j], args.alpha, ic[idx + j] * args.beta);
                }
            }
        }
    }
}


