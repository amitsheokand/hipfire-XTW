// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// MoE router top-2 + (optional) renormalize, given PRE-GATED scores.
// k=2 companion to moe_topk_renorm_k8: the dispatch site runs the gate
// kernel first (softmax_f32 for softmax models, moe_router_sqrtsoftplus
// for Fuse-family sqrtsoftplus), then this kernel for top-K + renorm —
// same split as the k=8 decode path, so routing math matches the CPU
// fallback path's gated-buffer inputs exactly.
//
// Reads [n_exp] gated scores, writes [2] i32 indices and [2] f32 weights
// to device buffers. One workgroup, 256 threads. Supports n_exp up to
// 1024 (Fuse: 8). Single general rescan path — no exact-256 fast path:
// at n_exp=8 the loop degenerates to one slot per thread and the second
// pick round reuses the same scan.
//
// Renormalize uses direct division (matches CPU fallback `*w /= sum`
// exactly; mul-by-reciprocal would add 1 ULP per element which the
// hyper-quantized MoE stack amplifies into a structural attractor —
// same hazard documented on moe_topk_renorm_k8).

#define BLOCK_SIZE 256
#define MAX_NEXP   1024
#define K_TOP      2

__device__ __forceinline__ void k2_topk_pair_reduce(float& v, int& idx) {
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down(v, off);
        int oi = __shfl_down(idx, off);
        if (ov > v) {
            v = ov;
            idx = oi;
        }
    }
}

extern "C" __launch_bounds__(BLOCK_SIZE, 2)
__global__ void moe_topk_renorm_k2(
    const float* __restrict__ scores_in, // [n_exp] pre-gated scores
    int*         __restrict__ topk_idx,  // [2]
    float*       __restrict__ topk_w,    // [2]
    int n_exp,
    int norm_topk
) {
    __shared__ int   picked_idx[K_TOP];
    __shared__ float picked_w[K_TOP];
    __shared__ float warp_v[8];
    __shared__ int   warp_i[8];

    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;

    for (int k = 0; k < K_TOP; k++) {
        float best_v = -INFINITY;
        int   best_i = -1;
        for (int i = tid; i < n_exp; i += BLOCK_SIZE) {
            bool already_picked = false;
            for (int j = 0; j < k; j++) {
                if (picked_idx[j] == i) already_picked = true;
            }
            float v = already_picked ? -INFINITY : scores_in[i];
            if (v > best_v) { best_v = v; best_i = i; }
        }
        k2_topk_pair_reduce(best_v, best_i);
        if (lane_id == 0) {
            warp_v[warp_id] = best_v;
            warp_i[warp_id] = best_i;
        }
        __syncthreads();

        float final_v = (warp_id == 0 && lane_id < 8) ? warp_v[lane_id] : -INFINITY;
        int final_i = (warp_id == 0 && lane_id < 8) ? warp_i[lane_id] : -1;
        if (warp_id == 0) {
            k2_topk_pair_reduce(final_v, final_i);
            if (lane_id == 0) {
                picked_idx[k] = final_i;
                picked_w[k]   = final_v;
            }
        }
        __syncthreads();
    }

    // Renormalize using direct division (matches CPU fallback path
    // *w /= sum exactly — see header note).
    if (tid == 0) {
        float s = 0.0f;
        for (int k = 0; k < K_TOP; k++) s += picked_w[k];
        if (norm_topk && s > 0.0f) {
            for (int k = 0; k < K_TOP; k++) picked_w[k] /= s;
        }
    }
    __syncthreads();

    // Write output. Clamp invalid winner (NaN/OOB from corrupted scores)
    // to expert 0 with weight 0 — same defense as moe_topk_renorm_k8.
    if (tid < K_TOP) {
        int   safe_idx = picked_idx[tid];
        float safe_w   = picked_w[tid];
        if (safe_idx < 0 || safe_idx >= n_exp) {
            safe_idx = 0;
            safe_w   = 0.0f;
        }
        topk_idx[tid] = safe_idx;
        topk_w[tid]   = safe_w;
    }
}
