---
title: gfx1201 FP8 GEMM LDS X-panel rejected — 76→155 VGPR occupancy cliff
date: 2026-08-23
tags: [gfx1201, fp8, wmma, lds, occupancy, negative-result]
---

After GEMM ALL PASS (76 VGPR / 20 SGPR / 0 spill, 50 global loads),
tried one tiling lever: cooperative pack of the 16×256 F32 X panel into
4 KB LDS E4M3, WMMA B from LDS. QUICK shapes stayed NRMSE-identical
(2.906 / 2.787 / 2.749e-2). Radiowave: VGPR 76→154 (unrolled fill) then
155 with `#pragma unroll 1`; global loads 50→20; waits 130→81; inst
533→398. Occupancy halves (granule-16: ~19 waves → ~9). Small-tile µs
moved ~5–11% (launch-dominated; not a claim). Do not keep LDS convert
in the kernel. Pack pre-pass **landed** (GEMM-only −42–51% on large-N
prefill; ffn1 N=128 +8.3%). See [[fp8-gemm-pack-prepass]]. Do not retry
in-kernel LDS convert. Next is encoder + `fp8_wmma` dispatch, default-off.
Related: [[r9700-fp8-branch-status]].
