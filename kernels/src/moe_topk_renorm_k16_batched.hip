// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched MoE top-16 + renorm given pre-softmaxed probs. Companion to
// moe_topk_renorm_k16 for the prefill path (Whittle k=16).
//
// One workgroup per token. blockIdx.x selects the token row in
// [N × n_exp] probs. Same algorithm as moe_topk_renorm_k8_batched.

#define BLOCK_SIZE 256
#define MAX_NEXP   1024
#define K_TOP      16

extern "C" __launch_bounds__(BLOCK_SIZE, 2)
__global__ void moe_topk_renorm_k16_batched(
    const float* __restrict__ probs_in,    // [N × n_exp] pre-softmaxed
    int*         __restrict__ topk_idx,    // [N × 16]
    float*       __restrict__ topk_w,      // [N × 16]
    int n_exp,
    int norm_topk
) {
    __shared__ float smem_v[MAX_NEXP];
    __shared__ int   smem_i[MAX_NEXP];
    __shared__ float reduce_v[BLOCK_SIZE];
    __shared__ int   reduce_i[BLOCK_SIZE];
    __shared__ int   picked_idx[K_TOP];
    __shared__ float picked_w[K_TOP];

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const float* __restrict__ row_probs = probs_in + (long long)bid * n_exp;
    int*   __restrict__ row_idx = topk_idx + bid * K_TOP;
    float* __restrict__ row_w   = topk_w   + bid * K_TOP;

    for (int i = tid; i < MAX_NEXP; i += BLOCK_SIZE) {
        smem_v[i] = (i < n_exp) ? row_probs[i] : -INFINITY;
        smem_i[i] = i;
    }
    __syncthreads();

    for (int k = 0; k < K_TOP; k++) {
        float best_v = -INFINITY;
        int   best_i = -1;
        for (int i = tid; i < n_exp; i += BLOCK_SIZE) {
            float v = smem_v[i];
            if (v > best_v) { best_v = v; best_i = smem_i[i]; }
        }
        reduce_v[tid] = best_v;
        reduce_i[tid] = best_i;
        __syncthreads();

        for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
            if (tid < s) {
                float va = reduce_v[tid];
                float vb = reduce_v[tid + s];
                if (vb > va) {
                    reduce_v[tid] = vb;
                    reduce_i[tid] = reduce_i[tid + s];
                }
            }
            __syncthreads();
        }
        const int   winner = reduce_i[0];
        const float winner_w = reduce_v[0];
        if (tid == 0) {
            picked_idx[k] = winner;
            picked_w[k]   = winner_w;
        }
        if (winner >= 0 && winner < n_exp) {
            if ((winner & (BLOCK_SIZE - 1)) == tid) {
                smem_v[winner] = -INFINITY;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        float s = 0.0f;
        for (int k = 0; k < K_TOP; k++) s += picked_w[k];
        if (norm_topk && s > 0.0f) {
            for (int k = 0; k < K_TOP; k++) picked_w[k] /= s;
        }
    }
    __syncthreads();

    if (tid < K_TOP) {
        int   safe_idx = picked_idx[tid];
        float safe_w   = picked_w[tid];
        if (safe_idx < 0 || safe_idx >= n_exp) {
            safe_idx = 0;
            safe_w   = 0.0f;
        }
        row_idx[tid] = safe_idx;
        row_w[tid]   = safe_w;
    }
}
