// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE in the project root.

#include <hip/hip_runtime.h>

// Fused k=2 MoE router: Q8 GEMV (8 rows) + sqrtsoftplus gate + top-2 + renorm.
//
// Replaces three launches per MoE layer on the Fuse-family k=2 indexed path:
//   gemv_q8_0(router) -> moe_router_sqrtsoftplus -> moe_topk_renorm_k2
//
// Single block of 256 threads (8 warps, warp w owns router row w).
//
// BIT-EXACTNESS contract:
//   Phase 1 transcribes gemv_q8_0's per-row loop verbatim (block-major oi,
//   sub-minor unroll, same FMA order, same shfl_down reduction tree), so
//   each row sum matches the standalone kernel bit for bit. All 8 rows run
//   as warps in one block instead of 8 blocks — the per-row math is
//   untouched, only the block index source changes.
//   Phase 2 transcribes moe_router_sqrtsoftplus (logf(1+exp) form, NOT
//   log1p) and moe_topk_renorm_k2 (strict-> tournament => lowest index wins
//   ties, direct-division renorm, NaN/OOB clamp to expert 0 / weight 0).
//
// Admission (checked by dispatch, re-asserted cheaply here): m == n_exp <= 8
// rows, K a multiple of 32 (the remainder loop is kept for generality but
// the fused path only fires when run_auto would have picked gemv_q8_0).
// Grid is exactly one block; warps w >= m exit after phase 1.

#define K1_WARPS 8

__device__ __forceinline__ float k1_sqrtsoftplus(float logit) {
    const float v = logit - 2.0f;
    const float sp =
        (v > 0.0f) ? (v + logf(1.0f + expf(-v))) : logf(1.0f + expf(v));
    return sqrtf(sp);
}

extern "C" __launch_bounds__(256, 2)
__global__ void moe_router_q8_sqrtsoftplus_topk2(
    const unsigned char* __restrict__ A_q8, // Q8_0 router [m, k]
    const float* __restrict__ x,            // [k] f32 activation
    float* __restrict__ logits_out,         // [m] f32 (debug/dump parity)
    int* __restrict__ topk_idx,             // [2] i32
    float* __restrict__ topk_w,             // [2] f32
    int m,                                  // == n_exp, <= 8
    int k,                                  // e.g. 2560
    int norm_topk
) {
    __shared__ float row_sum[K1_WARPS];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // ── Phase 1: Q8 GEMV row (verbatim gemv_q8_0) ────────────────────────
    float sum = 0.0f;
    if (warp < m) {
        const int blocks_per_row = k / 32;
        const unsigned char* row_data = A_q8 + (size_t)warp * blocks_per_row * 34;
        const int outer_iters = blocks_per_row / 8;
        for (int oi = 0; oi < outer_iters; oi++) {
            const unsigned char* base = row_data + oi * 8 * 34;
            const float* xb = x + oi * 256;
            #pragma unroll
            for (int sub = 0; sub < 8; sub++) {
                const unsigned char* block = base + sub * 34;
                float d = (float)*((const _Float16*)block);
                signed char qval = (signed char)block[2 + lane];
                sum += d * (float)qval * xb[sub * 32 + lane];
            }
        }
        for (int bi = outer_iters * 8; bi < blocks_per_row; bi++) {
            const unsigned char* block = row_data + bi * 34;
            float d = (float)*((const _Float16*)block);
            signed char qval = (signed char)block[2 + lane];
            sum += d * (float)qval * x[bi * 32 + lane];
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down(sum, offset);
        if (lane == 0) row_sum[warp] = sum;
    }
    __syncthreads();

    // ── Phase 2: gate + top-2 + renorm (warp 0) ─────────────────────────
    // All 32 lanes of warp 0 participate in the shuffle tree (idle lanes
    // hold -INFINITY) so the reduction is well-defined for any m <= 8;
    // only lanes 0..m-1 read scores and lane 0 writes outputs.
    float score = -INFINITY;
    if (warp == 0 && lane < m) score = k1_sqrtsoftplus(row_sum[lane]);
    if (warp == 0) {
        if (lane < m) logits_out[lane] = score;

        int picked_idx[2];
        float picked_w[2];
        for (int r = 0; r < 2; r++) {
            float v = (lane < K1_WARPS) ? score : -INFINITY;
            int idx = lane;
            if (r == 1 && lane == picked_idx[0]) v = -INFINITY;
            // Strict-> tournament over lanes 0..7: lowest index wins ties,
            // matching moe_topk_renorm_k2's rescan order.
            for (int off = 4; off > 0; off >>= 1) {
                float ov = __shfl_down(v, off);
                int oi = __shfl_down(idx, off);
                if (ov > v) { v = ov; idx = oi; }
            }
            // Lanes 8..31 hold -INFINITY and never win; broadcast winner.
            v = __shfl(v, 0);
            idx = __shfl(idx, 0);
            picked_idx[r] = idx;
            picked_w[r] = v;
        }

        if (lane == 0) {
            float s = picked_w[0] + picked_w[1];
            if (norm_topk && s > 0.0f) {
                picked_w[0] /= s;
                picked_w[1] /= s;
            }
            for (int r = 0; r < 2; r++) {
                int safe_idx = picked_idx[r];
                float safe_w = picked_w[r];
                if (safe_idx < 0 || safe_idx >= m) {
                    safe_idx = 0;
                    safe_w = 0.0f;
                }
                topk_idx[r] = safe_idx;
                topk_w[r] = safe_w;
            }
        }
    }
}
