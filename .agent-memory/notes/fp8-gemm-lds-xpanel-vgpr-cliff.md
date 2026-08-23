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
in the kernel. Next lever is a **separate** `pack_f32_to_fp8` pre-pass
(HFP4G32 FP8 sister already measured in-kernel cvt as the tax; pre-pass
recovered ~10pp) — not another in-kernel shared staging of F32→E4M3.
Related: [[r9700-fp8-branch-status]].
