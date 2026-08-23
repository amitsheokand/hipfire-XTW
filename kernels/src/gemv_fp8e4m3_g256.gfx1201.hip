// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 amitsheokand
// hipfire — see LICENSE and NOTICE in the project root.
//
// gfx1201 (RDNA4) FP8 E4M3 block-scaled decode GEMV.
//
// Weight layout (group = 256 elements, matches QuantType::FP8E4M3G256):
//   [16 B row header: fp16 row_scale + fp16 row_bias]  (row_bias reserved = 0)
//   [n_blocks x 2 B fp16 per-block scale]               n_blocks = K / 256
//   [n_blocks x 256 B E4M3 leaves]                      1 byte per weight
//
// One wave32 block computes one output row y[row] = A[row,:] . x[:] using
// the native gfx1201 `v_dot4_f32_fp8_fp8` instruction. Each thread owns
// 8 consecutive K-elements per group (32 lanes x 8 = 256), reads them as two
// uint32 (4 fp8 each), and warp-reduces with __shfl_down at the end.
//
// Structure mirrors the validated gemv_hfp4g32_fp8.gfx12.hip (same launch
// bounds, same uint32 dot4 operands, same shuffle reduction) so the
// optimizer cannot strip the compute as it did with the __shared__ +
// __syncthreads reduction variant.

#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>

#define FP8_GEMV_GROUP 256
#define FP8_GEMV_LANES 32

__launch_bounds__(FP8_GEMV_LANES, 16)
extern "C" __global__ void gemv_fp8e4m3_g256_gfx1201(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int n_blocks = K / FP8_GEMV_GROUP;
    if (K % FP8_GEMV_GROUP != 0) {
        if (tid == 0) y[row] = 0.0f;
        return;
    }

    const int scale_padded = ((n_blocks + 15) >> 4) << 4; // 16 B align
    const long long row_total =
        16LL + (long long)scale_padded * 2 + (long long)n_blocks * FP8_GEMV_GROUP;
    const long long row_off = (long long)row * row_total;

    const char* hdr = A + row_off;
    const unsigned short rs_bits = *(const unsigned short*)(hdr + 0);
    const _Float16 row_scale_h = __builtin_bit_cast(_Float16, rs_bits);
    const float row_scale = (float)row_scale_h;

    const char* scale_base = hdr + 16;
    const char* leaf_base  = hdr + 16 + (long long)scale_padded * 2;

    float acc = 0.0f;

    for (int g = 0; g < n_blocks; ++g) {
        const unsigned short gs_bits =
            *(const unsigned short*)(scale_base + g * 2);
        const _Float16 gscale_h = __builtin_bit_cast(_Float16, gs_bits);
        const float gscale = (float)gscale_h;

        // Each thread owns 8 consecutive E4M3 leaves (32 lanes x 8 = 256).
        const int x_base = g * FP8_GEMV_GROUP + tid * 8;
        const long long w_off =
            (long long)g * FP8_GEMV_GROUP + (long long)tid * 8;

        // 8 weight bytes + 8 activation floats -> 2 uint32 + 8 floats.
        // Weights are 1 byte per element (E4M3), activations are f32.
        const unsigned int w_lo =
            *(const unsigned int*)(leaf_base + w_off);
        const unsigned int w_hi =
            *(const unsigned int*)(leaf_base + w_off + 4);

        // Activations are F32; convert to FP8 bytes inline per 4.
        // (v_dot4_f32_fp8_fp8 consumes both operands as 4x fp8 bytes.)
        const float* xp = x + x_base;
        const float2 xa = { xp[0], xp[1] };
        const float2 xb = { xp[2], xp[3] };
        const float2 xc = { xp[4], xp[5] };
        const float2 xd = { xp[6], xp[7] };
        const unsigned int x_lo =
            __hip_cvt_float2_to_fp8x2(xa, __HIP_SATFINITE, __HIP_E4M3) |
            ((unsigned int)__hip_cvt_float2_to_fp8x2(xb, __HIP_SATFINITE, __HIP_E4M3) << 16);
        const unsigned int x_hi =
            __hip_cvt_float2_to_fp8x2(xc, __HIP_SATFINITE, __HIP_E4M3) |
            ((unsigned int)__hip_cvt_float2_to_fp8x2(xd, __HIP_SATFINITE, __HIP_E4M3) << 16);

        float partial = 0.0f;
        partial = __builtin_amdgcn_dot4_f32_fp8_fp8(w_lo, x_lo, partial);
        partial = __builtin_amdgcn_dot4_f32_fp8_fp8(w_hi, x_hi, partial);

        // Apply per-group fp16 scale and accumulate.
        acc = __builtin_fmaf(partial, gscale, acc);
    }

    // Warp reduce + write (no shared memory, no syncthreads).
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset);
    }
    if (tid == 0) y[row] = acc * row_scale;
}
