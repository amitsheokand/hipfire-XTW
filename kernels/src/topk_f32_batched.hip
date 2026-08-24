// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched top-k over F32 rows. Grid.x is the batch dim (one block per row).
// Each thread keeps a local top-k of its stride, then thread 0 merges.
// Tie-break matches host `dflash2_topk_rows`: higher value wins, equal
// values keep the lower index.
//
// DFlash 2's greedy selector needs top-16, not argmax. Downloading full
// `[B, vocab]` logits was ~8 MB D2H per cycle on Qwen3.8-27B.
#define TOPK_F32_KMAX 16

__device__ inline void topk_f32_insert(
    float* vals, int* ids, int k, float v, int idx
) {
    int slot = k;
    for (int j = 0; j < k; j++) {
        if (v > vals[j] || (v == vals[j] && idx < ids[j])) {
            slot = j;
            break;
        }
    }
    if (slot < k) {
        for (int j = k - 1; j > slot; j--) {
            vals[j] = vals[j - 1];
            ids[j] = ids[j - 1];
        }
        vals[slot] = v;
        ids[slot] = idx;
    }
}

extern "C" __global__ void topk_f32_batched(
    const float* __restrict__ data,  // [batch, n] row-major
    int* __restrict__ out_ids,       // [batch, k]
    float* __restrict__ out_vals,    // [batch, k]
    int n,
    int k
) {
    if (k < 1) {
        return;
    }
    if (k > TOPK_F32_KMAX) {
        k = TOPK_F32_KMAX;
    }
    extern __shared__ float smem[];
    float* svals = smem;
    int* sids = (int*)(svals + blockDim.x * TOPK_F32_KMAX);
    const int batch_idx = blockIdx.x;
    const float* row = data + (size_t)batch_idx * n;

    float lv[TOPK_F32_KMAX];
    int li[TOPK_F32_KMAX];
    for (int j = 0; j < TOPK_F32_KMAX; j++) {
        lv[j] = -1e30f;
        li[j] = 0x7fffffff;
    }
    for (int i = (int)threadIdx.x; i < n; i += (int)blockDim.x) {
        topk_f32_insert(lv, li, k, row[i], i);
    }
    for (int j = 0; j < k; j++) {
        svals[threadIdx.x * TOPK_F32_KMAX + j] = lv[j];
        sids[threadIdx.x * TOPK_F32_KMAX + j] = li[j];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float fv[TOPK_F32_KMAX];
        int fi[TOPK_F32_KMAX];
        for (int j = 0; j < TOPK_F32_KMAX; j++) {
            fv[j] = -1e30f;
            fi[j] = 0x7fffffff;
        }
        for (int t = 0; t < (int)blockDim.x; t++) {
            for (int j = 0; j < k; j++) {
                topk_f32_insert(
                    fv,
                    fi,
                    k,
                    svals[t * TOPK_F32_KMAX + j],
                    sids[t * TOPK_F32_KMAX + j]
                );
            }
        }
        int* ids_row = out_ids + batch_idx * k;
        float* vals_row = out_vals + batch_idx * k;
        for (int j = 0; j < k; j++) {
            ids_row[j] = fi[j];
            vals_row[j] = fv[j];
        }
    }
}
