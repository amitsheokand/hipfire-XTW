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
// [n_blocks x 256 B E4M3 leaves]) and X is F32 activations converted to
// E4M3 inline.
//
// Fragment layout mirrors gemm_qkv_hfp4g32_wmma_fp8.gfx12.hip (proven on
// this card): m_lane = tid & 15, k_grp = tid >> 4, 16x16x16 WMMA tiles,
// per-group fp16 scale applied post-WMMA on the F32 accumulator.
//
// Grid: [ceil(M/16), ceil(N/16)]. Block: [32] (one wave32).

#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>

typedef float __attribute__((ext_vector_type(8))) v8f_t;
typedef int   __attribute__((ext_vector_type(2))) v2i_t;

#define FP8_GEMM_GROUP 256

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_fp8e4m3_g256_wmma_gfx1201(
    const char* __restrict__ A,        // [M] rows of FP8E4M3G256
    const float* __restrict__ X,       // [N x K] F32 activations
    float* __restrict__ Y,             // [N x M] output
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
        const float* x_group = X + (long long)safe_batch * K + g * FP8_GEMM_GROUP;

#pragma unroll
        for (int kt = 0; kt < 16; kt += 4) {
            // 4 WMMA tiles of K=16 each; lane k_grp carries K[8*k_grp..8*k_grp+7].
            const int k_off_a = (kt + 0) * 16 + k_grp * 8;
            const int k_off_b = (kt + 1) * 16 + k_grp * 8;
            const int k_off_c = (kt + 2) * 16 + k_grp * 8;
            const int k_off_d = (kt + 3) * 16 + k_grp * 8;

            // 8 weight E4M3 bytes per tile per lane.
            v2i_t w_a = *(const v2i_t*)(group_leaf + k_off_a);
            v2i_t w_b = *(const v2i_t*)(group_leaf + k_off_b);
            v2i_t w_c = *(const v2i_t*)(group_leaf + k_off_c);
            v2i_t w_d = *(const v2i_t*)(group_leaf + k_off_d);

            // F32 activations -> E4M3 bytes inline, mirroring
            // mq_rotate_x_dual.gfx12.hip exactly: 4 cvt_pk_fp8_f32 calls
            // pack 8 floats into v2i_t {lo (v0..v3), hi (v4..v7)}.
            // Each lane covers 8 K-elements per tile.
            const float* xa = x_group + k_off_a;
            const float* xb = x_group + k_off_b;
            const float* xc = x_group + k_off_c;
            const float* xd = x_group + k_off_d;
            unsigned int xa_lo = 0u, xa_hi = 0u;
            xa_lo = __builtin_amdgcn_cvt_pk_fp8_f32(xa[0], xa[1], xa_lo, false);
            xa_lo = __builtin_amdgcn_cvt_pk_fp8_f32(xa[2], xa[3], xa_lo, true);
            xa_hi = __builtin_amdgcn_cvt_pk_fp8_f32(xa[4], xa[5], xa_hi, false);
            xa_hi = __builtin_amdgcn_cvt_pk_fp8_f32(xa[6], xa[7], xa_hi, true);
            const v2i_t x_a = { (int)xa_lo, (int)xa_hi };
            unsigned int xb_lo = 0u, xb_hi = 0u;
            xb_lo = __builtin_amdgcn_cvt_pk_fp8_f32(xb[0], xb[1], xb_lo, false);
            xb_lo = __builtin_amdgcn_cvt_pk_fp8_f32(xb[2], xb[3], xb_lo, true);
            xb_hi = __builtin_amdgcn_cvt_pk_fp8_f32(xb[4], xb[5], xb_hi, false);
            xb_hi = __builtin_amdgcn_cvt_pk_fp8_f32(xb[6], xb[7], xb_hi, true);
            const v2i_t x_b = { (int)xb_lo, (int)xb_hi };
            unsigned int xc_lo = 0u, xc_hi = 0u;
            xc_lo = __builtin_amdgcn_cvt_pk_fp8_f32(xc[0], xc[1], xc_lo, false);
            xc_lo = __builtin_amdgcn_cvt_pk_fp8_f32(xc[2], xc[3], xc_lo, true);
            xc_hi = __builtin_amdgcn_cvt_pk_fp8_f32(xc[4], xc[5], xc_hi, false);
            xc_hi = __builtin_amdgcn_cvt_pk_fp8_f32(xc[6], xc[7], xc_hi, true);
            const v2i_t x_c = { (int)xc_lo, (int)xc_hi };
            unsigned int xd_lo = 0u, xd_hi = 0u;
            xd_lo = __builtin_amdgcn_cvt_pk_fp8_f32(xd[0], xd[1], xd_lo, false);
            xd_lo = __builtin_amdgcn_cvt_pk_fp8_f32(xd[2], xd[3], xd_lo, true);
            xd_hi = __builtin_amdgcn_cvt_pk_fp8_f32(xd[4], xd[5], xd_hi, false);
            xd_hi = __builtin_amdgcn_cvt_pk_fp8_f32(xd[6], xd[7], xd_hi, true);
            const v2i_t x_d = { (int)xd_lo, (int)xd_hi };

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
