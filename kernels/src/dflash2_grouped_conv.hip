// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// DFlash 2 grouped two-tap conv (vLLM `_grouped_conv` / `DFlashGroupedConv`).
//
// x:      [B, H]
// coeff:  [B, 2, taps, num_groups]   row-major, produced by kernel_projection
// base:   [taps, H]                  one side of base_kernel
// out:    [B, H]
//
// Per element:
//   out[b, h] = (base[0,h] + delta[b,0,g]) * x[b,h]
//             + 1_{pos>=1 && b>=1} * (base[1,h] + delta[b,1,g]) * x[b-1,h]
//   g = h / group_size,  pos = b % block_size
//
// Tap-1 at query row 0 is zero: hipfire's draft block is [seed, mask, ...]
// so row 0 is the last verified token and has no in-query predecessor.
// `side` selects coeff[:, side, :, :] (0 = prepare, 1 = finish).

extern "C" __global__ void dflash2_grouped_conv_f32(
    const float* __restrict__ x,
    const float* __restrict__ coeff,
    const float* __restrict__ base,
    float* __restrict__ out,
    int B,
    int H,
    int taps,
    int group_size,
    int block_size,
    int side
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int n = B * H;
    if (i >= n) {
        return;
    }
    int b = i / H;
    int h = i - b * H;
    int g = h / group_size;
    int num_groups = H / group_size;
    int taps_ng = taps * num_groups;
    int coeff_row = b * 2 * taps_ng + side * taps_ng;

    int pos = (block_size > 0) ? (b % block_size) : b;
    float d0 = coeff[coeff_row + g];
    float acc = (base[h] + d0) * x[b * H + h];
    for (int tap = 1; tap < taps; tap++) {
        if (pos >= tap && b >= tap) {
            float dt = coeff[coeff_row + tap * num_groups + g];
            acc += (base[tap * H + h] + dt) * x[(b - tap) * H + h];
        }
    }
    out[i] = acc;
}
