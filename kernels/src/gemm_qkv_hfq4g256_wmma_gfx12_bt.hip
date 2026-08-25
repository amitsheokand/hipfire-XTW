// SPDX-License-Identifier: MIT OR Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Batch-tiled gfx12 HFQ4-G256 3-way fused QKV GEMM (Qwen FullAttention
// Q/K/V projections). Sister of gemm_qkvza_hfq4g256_wmma_gfx12_bt.hip.
// B independent acc chains over the batch (N) dim hide WMMA latency;
// weights+dequant loaded once per K-tile, reused across the B N-tiles.
// Byte-exact vs gemm_qkv_hfq4g256_wmma_gfx12. Grid: [ceil(total_m/16),
// ceil(N/(16*B))]. Instantiations bt2/4/8/12 — Muse already falsified
// bt16 (VGPR spill). B=2 is the #611 gfx1100 weight-reuse band (N≥32).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;
#define HFQ4_DQ8(pk, sc, zp) \
    a[0]=sc*(_Float16)(float)(((pk)>>0)&0xFu)+zp; a[1]=sc*(_Float16)(float)(((pk)>>4)&0xFu)+zp; \
    a[2]=sc*(_Float16)(float)(((pk)>>8)&0xFu)+zp; a[3]=sc*(_Float16)(float)(((pk)>>12)&0xFu)+zp; \
    a[4]=sc*(_Float16)(float)(((pk)>>16)&0xFu)+zp; a[5]=sc*(_Float16)(float)(((pk)>>20)&0xFu)+zp; \
    a[6]=sc*(_Float16)(float)(((pk)>>24)&0xFu)+zp; a[7]=sc*(_Float16)(float)(((pk)>>28)&0xFu)+zp

#define GEN_QKV_BT(BV) \
extern "C" __launch_bounds__(32, 2) __global__ void gemm_qkv_hfq4g256_wmma_gfx12_bt##BV( \
    const char* __restrict__ A_q, const char* __restrict__ A_k, const char* __restrict__ A_v, \
    const _Float16* __restrict__ X, \
    float* __restrict__ Y_q, float* __restrict__ Y_k, float* __restrict__ Y_v, \
    int q_m, int k_m, int v_m, int K, int N) { \
    const int total_m = q_m + k_m + v_m, rs = blockIdx.x * 16, bs = blockIdx.y * (16 * BV), tid = threadIdx.x; \
    if (rs >= total_m || bs >= N) return; \
    const int ml = tid & 15, kg = tid >> 4; \
    const int sr = (rs + ml < total_m) ? (rs + ml) : (total_m - 1); \
    const char* A; int lrow; \
    if (sr < q_m) { A = A_q; lrow = sr; } \
    else if (sr < q_m + k_m) { A = A_k; lrow = sr - q_m; } \
    else { A = A_v; lrow = sr - q_m - k_m; } \
    const int gpr = K / 256; const char* rb = A + (long long)lrow * gpr * 136; \
    int sb[BV]; for (int b = 0; b < BV; b++) { int t = bs + b * 16 + ml; sb[b] = (t < N) ? t : 0; } \
    float8_t acc[BV]; for (int b = 0; b < BV; b++) acc[b] = float8_t{0,0,0,0,0,0,0,0}; \
    for (int g = 0; g < gpr; g++) { \
        const char* gp = rb + g * 136; \
        _Float16 sc = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp)); \
        _Float16 zp = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gp + 4)); \
        const _Float16* xg[BV]; for (int b = 0; b < BV; b++) xg[b] = X + (long long)sb[b] * K + g * 256; \
        for (int kt = 0; kt < 16; kt += 4) { \
            unsigned int p[4] = { *(const unsigned int*)(gp+8+(kt+0)*8+kg*4), *(const unsigned int*)(gp+8+(kt+1)*8+kg*4), \
                                  *(const unsigned int*)(gp+8+(kt+2)*8+kg*4), *(const unsigned int*)(gp+8+(kt+3)*8+kg*4) }; \
            _Pragma("unroll") \
            for (int ki = 0; ki < 4; ki++) { half8_t a; HFQ4_DQ8(p[ki], sc, zp); \
                _Pragma("unroll") \
                for (int b = 0; b < BV; b++) { half8_t bb = *(const half8_t*)(xg[b] + (kt+ki)*16 + kg*8); \
                    acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, acc[b]); } } } \
    } \
    for (int b = 0; b < BV; b++) { const int oc = bs + b * 16 + ml; if (oc < N) for (int j = 0; j < 8; j++) { \
        int orow = rs + 8 * kg + j; if (orow < total_m) { float* Y; int pr, os; \
            if (orow < q_m) { Y = Y_q; pr = orow; os = q_m; } \
            else if (orow < q_m + k_m) { Y = Y_k; pr = orow - q_m; os = k_m; } \
            else { Y = Y_v; pr = orow - q_m - k_m; os = v_m; } \
            Y[(long long)oc * os + pr] = acc[b][j]; } } } \
}

GEN_QKV_BT(2)
GEN_QKV_BT(4)
GEN_QKV_BT(8)
GEN_QKV_BT(12)
