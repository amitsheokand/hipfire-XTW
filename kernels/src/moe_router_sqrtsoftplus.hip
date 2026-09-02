// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Fuse4 router gate: in-place sqrt(softplus(x - 2)) over [n_exp] logits.
// Companion to moe_topk_renorm_k2 for k=2 Fuse-family routing; the softmax
// path instead runs the regular softmax_f32 kernel first (same split as the
// k=8 decode path: gate kernel, then top-K + renorm).
//
// Op order matches the CPU reference in hipfire-dispatch
// (run_moe_decode_cpu_fallback) EXACTLY — logf(1+exp) form, NOT log1p — so
// GPU-routed weights are bit-identical to host-routed ones:
//   x = p - 2.0
//   sp = x > 0 ? x + ln(1 + exp(-x)) : ln(1 + exp(x))
//   out = sqrt(sp)

extern "C" __global__ void moe_router_sqrtsoftplus(
    float* __restrict__ x, // [n_exp] f32 logits, replaced by gate scores
    int n_exp
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_exp) {
        return;
    }
    const float v = x[i] - 2.0f;
    const float sp =
        (v > 0.0f) ? (v + logf(1.0f + expf(-v))) : logf(1.0f + expf(v));
    x[i] = sqrtf(sp);
}
