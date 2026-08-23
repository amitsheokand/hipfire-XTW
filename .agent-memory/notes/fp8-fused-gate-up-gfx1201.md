---
title: gfx1201 FP8 fused gate+up GEMM (qt=40)
date: 2026-08-24
tags: [gfx1201, r9700, fp8, prefill, gemm, fused]
---

One lever after 4B qt=40 text was readable: fused gate+up for
`FP8E4M3G256` prefill. Same WMMA loop as overwrite
`gemm_fp8e4m3_g256_wmma.gfx1201`; grid.x covers `gate_m+up_m` so one
launch reads packed X once. QKV / QKVZA / residual stay overwrite.
`fp8_wmma` stays default-off.

Lab (`HIPFIRE_FP8_GEMM_QUICK=1`, HIP_VISIBLE_DEVICES=0, HIP 7.2): fused
vs two overwrite GEMMs **max|Δ|=0** on N=32, gate=up=64, K=256. Existing
three small GEMM tiles still ALL PASS.

4B thinking-off serve still language (Paris / Seine / Eiffel). Not a
tok/s or admission claim. `finish=length` at 48 tokens is the cap.

Related: [[fp8-batched-prefill-overwrite-gemm]], [[fp8-e2e-qwen35-4b]],
[[r9700-fp8-branch-status]].
