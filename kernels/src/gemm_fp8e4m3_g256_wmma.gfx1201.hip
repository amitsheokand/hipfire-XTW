// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 amitsheokand
// hipfire — see LICENSE and NOTICE in the project root.
//
// gfx1201 (RDNA4) FP8 E4M3 block-scaled GEMM (prefill) using the native
// gfx12 WMMA FP8 instruction:
//
//     __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12
//
// Computes Y[N x M] = X[N x K] @ A[M x K]^T where A is FP8E4M3G256
// (block-scaled E4M3: [16 B hdr][n_blocks x 2 B fp16 scale, 16-aligned]
// [n_blocks x 256 B E4M3 leaves]) and X is E4M3 packed by
// pack_f32_to_fp8_gfx12 (in-kernel cvt was the ALU tax; LDS staging
// doubled VGPRs — see fp8-gemm-lds-xpanel-vgpr-cliff).
//
// Fragment layout mirrors gemm_qkv_hfp4g32_wmma_fp8.gfx12.hip (proven on
// this card): m_lane = tid & 15, k_grp = tid >> 4, 16x16x16 WMMA tiles,
// per-group fp16 scale applied post-WMMA on the F32 accumulator.
//
// Grid: [ceil(M/16), ceil(N/16)]. Block: [32] (one wave32).

#include <hip/hip_runtime.h>

typedef float __attribute__((ext_vector_type(8))) v8f_t;
typedef int   __attribute__((ext_vector_type(2))) v2i_t;

#define FP8_GEMM_GROUP 256

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_fp8e4m3_g256_wmma_gfx1201(
    const char* __restrict__ A,                 // [M] rows of FP8E4M3G256
    const unsigned char* __restrict__ X_fp8,    // [N x K] E4M3
    float* __restrict__ Y,                      // [N x M] output
    int M,
    int K,
    int N
) {
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid = threadIdx.x;

    if (row_start >= M || batch_start >= N) return;

    // gfx12 wave32 WMMA lane decomposition.
    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const int my_row = row_start + m_lane;
    const int safe_row = (my_row < M) ? my_row : (M - 1);

    const int n_blocks = K / FP8_GEMM_GROUP;
    const int scale_padded = ((n_blocks + 15) >> 4) << 4;
    const long long row_total =
        16LL + (long long)scale_padded * 2 + (long long)n_blocks * FP8_GEMM_GROUP;
    const long long row_off = (long long)safe_row * row_total;

    const char* hdr = A + row_off;
    const unsigned short rs_bits = *(const unsigned short*)(hdr + 0);
    const _Float16 row_scale_h = __builtin_bit_cast(_Float16, rs_bits);
    const float row_scale = (float)row_scale_h;

    const char* scale_base = hdr + 16;
    const char* leaf_base  = hdr + 16 + (long long)scale_padded * 2;

    // Gather the 8 per-output-row scales for this lane's output mapping
    // (rows 8*k_grp .. 8*k_grp+7 under gfx12 C-mapping). Source lanes live
    // in lane group 0 (lanes 0..15).
    float out_row_scales[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        int src_lane = 8 * k_grp + j;
        out_row_scales[j] = __shfl(row_scale, src_lane);
    }

    const int safe_batch = (batch_start + m_lane < N) ? (batch_start + m_lane) : 0;

    v8f_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < n_blocks; g++) {
        // Per-group fp16 scale for this lane's A-row.
        const unsigned short gs_bits =
            *(const unsigned short*)(scale_base + g * 2);
        const _Float16 gscale_h = __builtin_bit_cast(_Float16, gs_bits);
        const float gscale_self = (float)gscale_h;

        // Gather per-output-row scales x group scale.
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
            // 4 WMMA tiles of K=16 each; lane k_grp carries K[8*k_grp..8*k_grp+7].
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

            // Scale must apply to THIS group's contribution only — never
            // chain through acc, or group g+1 inherits scale_g (compounding
            // error). Fresh partial per tile, scaled, then folded into acc
            // (same structure as gemm_qkv_hfp4g32_wmma_fp8.gfx12.hip).
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

    // Write the 16x16 output tile (gfx12 wave32 WMMA C-mapping:
    // acc[j] = C[8*(tid>>4) + j][tid & 15]).
    // Y is [N][M] row-major (N = batch rows, M = output cols).
    const int out_col = batch_start + m_lane;
    if (out_col < N) {
        const int out_row_base = row_start + 8 * k_grp;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = out_row_base + j;
            if (out_row >= M) continue;
            Y[(long long)out_col * M + out_row] = acc[j];
        }
    }
}
