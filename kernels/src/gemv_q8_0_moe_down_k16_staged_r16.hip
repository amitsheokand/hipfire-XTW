// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Whittle-shaped Q8 MoE down (decode T=1): stage the 16×192 silu activation
// once, then 8 warps each own two top-k ranks and RPB output rows.
//
// Replaces gemv_q8_0_moe_down_residual_scaled_k8_indexed at K=192, k_top=16,
// which launches M×16 = 81,920 thirty-two-thread CTAs (72 µs, ~220 GiB/s on
// gfx1201 vs ~435 GiB/s on the sibling MQ4 gate_up). Same Q8_0 block math
// (34 B / 32 elements, f16 scale + int8) and the same
// topk_weights * per_expert_scale * acc fold; the fold is a single owner per
// row instead of atomicAdd (associative, not bit-identical with the racy
// atomics).
//
// Not a larger dense DMMV workgroup — 32-thread DMMV stays put. This is the
// ninepath pattern: one 256-thread CTA, LDS-staged act, multi-row.
//
// Requirements: K == 192, k_top == 16, M % RPB == 0, blockDim == 256,
// batch == 1 (grid.z unused). Opt out with HIPFIRE_WHITTLE_Q8_STAGED=0.

__launch_bounds__(256, 4)
extern "C" __global__ void gemv_q8_0_moe_down_k16_staged_r16(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ topk_weights,
    const float* __restrict__ per_expert_scale,
    const float* __restrict__ hidden_batch,
    float* __restrict__ x_residual,
    int M, int K
) {
    constexpr int RPB = 16;
    constexpr int N_RANKS = 16;
    constexpr int KK = 192;
    constexpr int BLOCKS = KK / 32;

    __shared__ float act_s[N_RANKS][KK];
    __shared__ float part[N_RANKS][RPB];

    const int tid = threadIdx.x;
    for (int i = tid; i < N_RANKS * KK; i += 256)
        (&act_s[0][0])[i] = hidden_batch[i];
    __syncthreads();

    (void)M;
    (void)K;

    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int r0 = (int)blockIdx.x * RPB;

    #pragma unroll
    for (int pass = 0; pass < 2; pass++) {
        const int krank = warp + pass * 8;
        const int expert_id = topk_indices[krank];
        const unsigned char* __restrict__ A =
            reinterpret_cast<const unsigned char*>(expert_ptrs[expert_id]);
        const float* __restrict__ xa = act_s[krank];
        const float scale = topk_weights[krank] * per_expert_scale[expert_id];

        for (int r = 0; r < RPB; r++) {
            const int row = r0 + r;
            const unsigned char* row_data =
                A + (size_t)row * (size_t)BLOCKS * 34u;
            float sum = 0.0f;
            #pragma unroll
            for (int bi = 0; bi < BLOCKS; bi++) {
                const unsigned char* block = row_data + bi * 34;
                float d = (float)*((const _Float16*)block);
                signed char qval = (signed char)block[2 + lane];
                sum += d * (float)qval * xa[bi * 32 + lane];
            }
            for (int offset = 16; offset > 0; offset >>= 1)
                sum += __shfl_down(sum, offset);
            if (lane == 0)
                part[krank][r] = scale * sum;
        }
    }
    __syncthreads();

    if (warp == 0 && lane < RPB) {
        float a = 0.0f;
        #pragma unroll
        for (int w = 0; w < N_RANKS; w++)
            a += part[w][lane];
        x_residual[r0 + lane] += a;
    }
}
