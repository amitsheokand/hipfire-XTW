// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kevin Read
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx12 small-N sister of gemm_f16_wmma_mb8.gfx12.hip.
//
// MB8 register-blocks NB=8 (16×128 N panel). That reuses the W A-tile across
// up to 128 batch rows, which is the right trade for vision-scale N. DFlash
// decode is N=B≤8 (sometimes a first window ~24): seven of the eight WMMA
// along N multiply zeros, and grid_n stays 1, so the K-loop is ~8× too much
// compute for the same weight traffic. NB=1 is one 16-col N tile — enough
// for B≤16, two tiles for B=17..32 — with the same gfx12 WMMA layout:
//   acc[j] = C[8 * (tid >> 4) + j][tid & 15]
//
// Dispatch: gemm_f16_wmma_mb8() on has_wmma_w32_gfx12 && N<=32.
// Contract identical to MB8: Y[n][m] = sum_k W_f16[m][k] * X_f32[n][k].

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

#define NB 1

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_f16_wmma_mb1_gfx12(
    const _Float16* __restrict__ W,   // [M, K] row-major F16 (weights)
    const float*    __restrict__ X,   // [N, K] row-major F32 (input)
    float*          __restrict__ Y,   // [N, M] row-major F32 (transposed output)
    int M, int K, int N
) {
    const int row_start = blockIdx.x * 16;
    const int col_start = blockIdx.y * (16 * NB);
    const int tid = threadIdx.x;
    const int lane_m = tid & 15;
    const int k_grp = tid >> 4;

    if (row_start >= M || col_start >= N) return;

    const int my_a_row = row_start + lane_m;
    const bool a_in_bounds = my_a_row < M;

    float8_t acc[NB];
    #pragma unroll
    for (int nb = 0; nb < NB; nb++) {
        acc[nb] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }

    for (int k0 = 0; k0 < K; k0 += 16) {
        half8_t a_reg;
        if (a_in_bounds) {
            const _Float16* src = W + (long long)my_a_row * K + k0;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const int kk = k0 + k_grp * 8 + j;
                a_reg[j] = kk < K ? src[k_grp * 8 + j] : (_Float16)0.0f;
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 8; j++) a_reg[j] = (_Float16)0.0f;
        }

        #pragma unroll
        for (int nb = 0; nb < NB; nb++) {
            const int my_b_row = col_start + nb * 16 + lane_m;
            half8_t b_reg;
            if (my_b_row < N) {
                const float* src = X + (long long)my_b_row * K + k0;
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    const int kk = k0 + k_grp * 8 + j;
                    b_reg[j] = kk < K ? (_Float16)src[k_grp * 8 + j] : (_Float16)0.0f;
                }
            } else {
                #pragma unroll
                for (int j = 0; j < 8; j++) b_reg[j] = (_Float16)0.0f;
            }
            acc[nb] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_reg, acc[nb]);
        }
    }

    #pragma unroll
    for (int nb = 0; nb < NB; nb++) {
        const int out_col = col_start + nb * 16 + lane_m;
        if (out_col < N) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const int out_row = row_start + 8 * k_grp + j;
                if (out_row < M) {
                    Y[(long long)out_col * M + out_row] = acc[nb][j];
                }
            }
        }
    }
}
