---
title: RDNA4 ISA names V_WMMA_F32_16X16X16_FP8_FP8; F8_Mode for WMMA is unspecified
date: 2026-08-23
tags: [rdna4, isa, wmma, fp8, gfx1201]
---

Public ISA (doc 70651, 7 Apr 2025) is family-level — no gfx1200 vs
gfx1201 split. Wave32 FP8 WMMA: A/B = 2 VGPR (8 FP8), C = 8 F32, kRepeat=1.
C-map: acc[j] = C[8*(tid>>4)+j][tid&15] (worked example; ignore the
conflicting “first half of each row” sentence). Matches
gemm_fp8e4m3_g256_wmma.gfx1201.hip. F8_Mode 0 = OCP E4M3fn bias 7 max 448;
mode 1 = bias 8 max 240. ISA does not say which WMMA / V_CVT_PK_FP8_F32
use — prove with a max/NaN oracle before trusting the encoder. Viewer:
docs.amd.com/v/u/en-US/rdna4-instruction-set-architecture. Do not commit
the PDF. Related: [[r9700-fp8-branch-status]].
