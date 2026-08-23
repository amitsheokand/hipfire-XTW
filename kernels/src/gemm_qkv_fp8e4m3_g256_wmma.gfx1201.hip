// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 amitsheokand
// hipfire — see LICENSE and NOTICE in the project root.
//
// gfx1201 fused FA QKV sister of gemm_qkvza_fp8e4m3_g256_wmma.gfx1201.hip.
// Three FA projections (wq / wk / wv) share one packed-X read. Same WMMA
// loop and G256 E4M3 layout. Not a tok/s claim.
//
// Grid: [ceil((q_m+k_m+v_m)/16), ceil(N/16)]. Block: [32].

#include <hip/hip_runtime.h>

typedef float __attribute__((ext_vector_type(8))) v8f_t;
typedef int   __attribute__((ext_vector_type(2))) v2i_t;

#define FP8_GEMM_GROUP 256

__device__ __forceinline__ void fp8_qkv_pick(
    int row, int q_m, int k_m,
    const char* A_q, const char* A_k, const char* A_v,
    const char** A, int* local
) {
    const int k0 = q_m;
    const int v0 = q_m + k_m;
    if (row < k0) {
        *A = A_q;
        *local = row;
    } else if (row < v0) {
        *A = A_k;
        *local = row - k0;
    } else {
        *A = A_v;
        *local = row - v0;
    }
}

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_qkv_fp8e4m3_g256_wmma_gfx1201(
    const char* __restrict__ A_q,
    const char* __restrict__ A_k,
    const char* __restrict__ A_v,
    const unsigned char* __restrict__ X_fp8,
    float* __restrict__ Y_q,
    float* __restrict__ Y_k,
    float* __restrict__ Y_v,
    int q_m,
    int k_m,
    int v_m,
    int K,
    int N
) {
    const int total_m = q_m + k_m + v_m;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid = threadIdx.x;

    if (row_start >= total_m || batch_start >= N) return;

    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const int my_row = row_start + m_lane;
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1);
    const char* A;
    int local_row;
    fp8_qkv_pick(safe_row, q_m, k_m, A_q, A_k, A_v, &A, &local_row);

    const int n_blocks = K / FP8_GEMM_GROUP;
    const int scale_padded = ((n_blocks + 15) >> 4) << 4;
    const long long row_total =
        16LL + (long long)scale_padded * 2 + (long long)n_blocks * FP8_GEMM_GROUP;
    const long long row_off = (long long)local_row * row_total;

    const char* hdr = A + row_off;
    const unsigned short rs_bits = *(const unsigned short*)(hdr + 0);
    const _Float16 row_scale_h = __builtin_bit_cast(_Float16, rs_bits);
    const float row_scale = (float)row_scale_h;

    const char* scale_base = hdr + 16;
    const char* leaf_base  = hdr + 16 + (long long)scale_padded * 2;

    float out_row_scales[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        int src_lane = 8 * k_grp + j;
        out_row_scales[j] = __shfl(row_scale, src_lane);
    }

    const int safe_batch = (batch_start + m_lane < N) ? (batch_start + m_lane) : 0;

    v8f_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < n_blocks; g++) {
        const unsigned short gs_bits =
            *(const unsigned short*)(scale_base + g * 2);
        const _Float16 gscale_h = __builtin_bit_cast(_Float16, gs_bits);
        const float gscale_self = (float)gscale_h;

        float gscale[8];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            int src_lane = 8 * k_grp + j;
            gscale[j] = out_row_scales[j] * __shfl(gscale_self, src_lane);
        }

        const char* group_leaf = leaf_base + (long long)g * FP8_GEMM_GROUP;
        const unsigned char* x_group =
            X_fp8 + (long long)safe_batch * K + g * FP8_GEMM_GROUP;

#pragma unroll
        for (int kt = 0; kt < 16; kt += 4) {
            const int k_off_a = (kt + 0) * 16 + k_grp * 8;
            const int k_off_b = (kt + 1) * 16 + k_grp * 8;
            const int k_off_c = (kt + 2) * 16 + k_grp * 8;
            const int k_off_d = (kt + 3) * 16 + k_grp * 8;

            v2i_t w_a = *(const v2i_t*)(group_leaf + k_off_a);
            v2i_t w_b = *(const v2i_t*)(group_leaf + k_off_b);
            v2i_t w_c = *(const v2i_t*)(group_leaf + k_off_c);
            v2i_t w_d = *(const v2i_t*)(group_leaf + k_off_d);

            v2i_t x_a = *(const v2i_t*)(x_group + k_off_a);
            v2i_t x_b = *(const v2i_t*)(x_group + k_off_b);
            v2i_t x_c = *(const v2i_t*)(x_group + k_off_c);
            v2i_t x_d = *(const v2i_t*)(x_group + k_off_d);

            v8f_t part_ab = {0, 0, 0, 0, 0, 0, 0, 0};
            part_ab = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(w_a, x_a, part_ab);
            part_ab = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(w_b, x_b, part_ab);
#pragma unroll
            for (int j = 0; j < 8; j++) acc[j] += part_ab[j] * gscale[j];

            v8f_t part_cd = {0, 0, 0, 0, 0, 0, 0, 0};
            part_cd = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(w_c, x_c, part_cd);
            part_cd = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(w_d, x_d, part_cd);
#pragma unroll
            for (int j = 0; j < 8; j++) acc[j] += part_cd[j] * gscale[j];
        }
    }

    const int k0 = q_m;
    const int v0 = q_m + k_m;
    const int out_col = batch_start + m_lane;
    if (out_col < N) {
        const int out_row_base = row_start + 8 * k_grp;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = out_row_base + j;
            if (out_row >= total_m) continue;
            if (out_row < k0) {
                Y_q[(long long)out_col * q_m + out_row] = acc[j];
            } else if (out_row < v0) {
                Y_k[(long long)out_col * k_m + (out_row - k0)] = acc[j];
            } else {
                Y_v[(long long)out_col * v_m + (out_row - v0)] = acc[j];
            }
        }
    }
}
