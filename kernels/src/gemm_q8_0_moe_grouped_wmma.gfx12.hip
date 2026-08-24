// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx12 (RDNA4) grouped-WMMA Q8_0 MoE GEMM. Same scatter contract as
// `gemm_hfq4g256_moe_grouped_wmma_gfx12`; Q8_0 block math from
// `gemm_q8_0_wmma_gfx12` (34 B / 32 elements, f16 scale + int8).
//
// Whittle routed down is K=192 (not a G256 multiple), so the HFQ4 grouped
// sister cannot run it. Path 1 stays on the staged indexed GEMV; this is
// the Path 2 arm. K must be a multiple of 32. gfx11 is not wired.
//
// Per-block contract:
//   - Tile (blockIdx.x, blockIdx.y) owns a 16-row × 16-slot output tile.
//   - All 16 slots share the same expert (scatter pads to BLOCK_M = 16).
//   - X is gathered via sorted_slot_index[slot_start + m_lane]; -1
//     padding lanes substitute a zero B row.
//   - `x_row_div` matches the HFQ4 grouped sister (down: 1).
//
// Grid: [ceil(M/16), m_total/16]. Block: [32]. LDS: 0.

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_q8_0_moe_grouped_wmma_gfx12(
    const unsigned long long* __restrict__ expert_weight_ptrs,
    const int*                __restrict__ expert_tile_ids,
    const int*                __restrict__ sorted_slot_index,
    const _Float16*           __restrict__ X_src,
    float*                    __restrict__ Y_grouped,
    int M, int K, int x_row_div, int m_total
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * 16;

    if (row_start >= M || slot_start >= m_total) return;

    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    const int slot_idx = slot_start + m_lane;
    int x_row = -1;
    if (slot_idx < m_total) {
        const int flat = sorted_slot_index[slot_idx];
        if (flat >= 0) {
            x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
    }
    const _Float16* x_base =
        (x_row >= 0) ? (X_src + (long long)x_row * K) : (const _Float16*)nullptr;

    const int safe_row = (row_start + m_lane < M) ? (row_start + m_lane) : (M - 1);
    const int blocks_per_row = K / 32;
    const long long row_stride = (long long)blocks_per_row * 34;
    const char* row_base = A + (long long)safe_row * row_stride;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const half8_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0};

    #define DQ8_Q(pk, sc) \
        a_reg[0] = (sc) * (_Float16)(float)(signed char)(((pk) >>  0) & 0xFFu); \
        a_reg[1] = (sc) * (_Float16)(float)(signed char)(((pk) >>  8) & 0xFFu); \
        a_reg[2] = (sc) * (_Float16)(float)(signed char)(((pk) >> 16) & 0xFFu); \
        a_reg[3] = (sc) * (_Float16)(float)(signed char)(((pk) >> 24) & 0xFFu); \
        a_reg[4] = (sc) * (_Float16)(float)(signed char)(((pk) >> 32) & 0xFFu); \
        a_reg[5] = (sc) * (_Float16)(float)(signed char)(((pk) >> 40) & 0xFFu); \
        a_reg[6] = (sc) * (_Float16)(float)(signed char)(((pk) >> 48) & 0xFFu); \
        a_reg[7] = (sc) * (_Float16)(float)(signed char)(((pk) >> 56) & 0xFFu)

    int bi = 0;
    for (; bi + 1 < blocks_per_row; bi += 2) {
        const char* bp_a = row_base + bi * 34;
        const char* bp_b = row_base + (bi + 1) * 34;

        _Float16 sc_a, sc_b;
        {
            unsigned short s_bits;
            __builtin_memcpy(&s_bits, bp_a, 2); __builtin_memcpy(&sc_a, &s_bits, 2);
            __builtin_memcpy(&s_bits, bp_b, 2); __builtin_memcpy(&sc_b, &s_bits, 2);
        }

        unsigned long long pk0a, pk1a, pk0b, pk1b;
        __builtin_memcpy(&pk0a, bp_a + 2 +  0 + k_grp * 8, 8);
        __builtin_memcpy(&pk1a, bp_a + 2 + 16 + k_grp * 8, 8);
        __builtin_memcpy(&pk0b, bp_b + 2 +  0 + k_grp * 8, 8);
        __builtin_memcpy(&pk1b, bp_b + 2 + 16 + k_grp * 8, 8);

        half8_t b0a, b1a, b0b, b1b;
        if (x_base != nullptr) {
            const _Float16* x_a = x_base + bi * 32;
            const _Float16* x_b = x_base + (bi + 1) * 32;
            b0a = *(const half8_t*)(x_a +  0 + k_grp * 8);
            b1a = *(const half8_t*)(x_a + 16 + k_grp * 8);
            b0b = *(const half8_t*)(x_b +  0 + k_grp * 8);
            b1b = *(const half8_t*)(x_b + 16 + k_grp * 8);
        } else {
            b0a = zero_b; b1a = zero_b; b0b = zero_b; b1b = zero_b;
        }

        half8_t a_reg;
        DQ8_Q(pk0a, sc_a);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b0a, acc);
        DQ8_Q(pk1a, sc_a);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b1a, acc);
        DQ8_Q(pk0b, sc_b);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b0b, acc);
        DQ8_Q(pk1b, sc_b);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b1b, acc);
    }

    if (bi < blocks_per_row) {
        const char* bp = row_base + bi * 34;
        _Float16 sc;
        {
            unsigned short s_bits;
            __builtin_memcpy(&s_bits, bp, 2);
            __builtin_memcpy(&sc, &s_bits, 2);
        }

        unsigned long long pk0, pk1;
        __builtin_memcpy(&pk0, bp + 2 +  0 + k_grp * 8, 8);
        __builtin_memcpy(&pk1, bp + 2 + 16 + k_grp * 8, 8);

        half8_t b0, b1;
        if (x_base != nullptr) {
            const _Float16* xb = x_base + bi * 32;
            b0 = *(const half8_t*)(xb +  0 + k_grp * 8);
            b1 = *(const half8_t*)(xb + 16 + k_grp * 8);
        } else {
            b0 = zero_b; b1 = zero_b;
        }

        half8_t a_reg;
        DQ8_Q(pk0, sc);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b0, acc);
        DQ8_Q(pk1, sc);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b1, acc);
    }
    #undef DQ8_Q

    const int out_col = slot_start + m_lane;
    if (out_col < m_total) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 8 * k_grp + j;
            if (out_row < M) {
                Y_grouped[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}
