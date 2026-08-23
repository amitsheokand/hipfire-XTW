// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 amitsheokand
// hipfire — see LICENSE and NOTICE in the project root.
//
// gfx1201 fused QKVZA sister of gemm_gate_up_fp8e4m3_g256_wmma.gfx1201.hip.
// Four LA projections (wqkv / wz / w_beta / w_alpha) share one packed-X
// read. Same WMMA loop and G256 E4M3 layout. Not a tok/s claim.
//
// Grid: [ceil((qkv_m+z_m+beta_m+alpha_m)/16), ceil(N/16)]. Block: [32].

#include <hip/hip_runtime.h>

typedef float __attribute__((ext_vector_type(8))) v8f_t;
typedef int   __attribute__((ext_vector_type(2))) v2i_t;

#define FP8_GEMM_GROUP 256

__device__ __forceinline__ void fp8_qkvza_pick(
    int row, int qkv_m, int z_m, int beta_m,
    const char* A_qkv, const char* A_z, const char* A_beta, const char* A_alpha,
    const char** A, int* local
) {
    const int z0 = qkv_m;
    const int b0 = qkv_m + z_m;
    const int a0 = qkv_m + z_m + beta_m;
    if (row < z0) {
        *A = A_qkv;
        *local = row;
    } else if (row < b0) {
        *A = A_z;
        *local = row - z0;
    } else if (row < a0) {
        *A = A_beta;
        *local = row - b0;
    } else {
        *A = A_alpha;
        *local = row - a0;
    }
}

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_qkvza_fp8e4m3_g256_wmma_gfx1201(
    const char* __restrict__ A_qkv,
    const char* __restrict__ A_z,
    const char* __restrict__ A_beta,
    const char* __restrict__ A_alpha,
    const unsigned char* __restrict__ X_fp8,
    float* __restrict__ Y_qkv,
    float* __restrict__ Y_z,
    float* __restrict__ Y_beta,
    float* __restrict__ Y_alpha,
    int qkv_m,
    int z_m,
    int beta_m,
    int alpha_m,
    int K,
    int N
) {
    const int total_m = qkv_m + z_m + beta_m + alpha_m;
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
    fp8_qkvza_pick(safe_row, qkv_m, z_m, beta_m, A_qkv, A_z, A_beta, A_alpha, &A, &local_row);

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

    const int z0 = qkv_m;
    const int b0 = qkv_m + z_m;
    const int a0 = qkv_m + z_m + beta_m;
    const int out_col = batch_start + m_lane;
    if (out_col < N) {
        const int out_row_base = row_start + 8 * k_grp;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = out_row_base + j;
            if (out_row >= total_m) continue;
            if (out_row < z0) {
                Y_qkv[(long long)out_col * qkv_m + out_row] = acc[j];
            } else if (out_row < b0) {
                Y_z[(long long)out_col * z_m + (out_row - z0)] = acc[j];
            } else if (out_row < a0) {
                Y_beta[(long long)out_col * beta_m + (out_row - b0)] = acc[j];
            } else {
                Y_alpha[(long long)out_col * alpha_m + (out_row - a0)] = acc[j];
            }
        }
    }
}
