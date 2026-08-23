---
title: gfx1201 FP8 fused QKVZA GEMM (qt=40)
date: 2026-08-24
tags: [gfx1201, r9700, fp8, prefill, gemm, fused, qkvza]
---

One lever after 27B qt=40 text was readable: fused QKVZA for
`FP8E4M3G256` LA prefill (wqkv / wz / w_beta / w_alpha). Same WMMA
loop as fused gate+up; grid.x covers the four M banks so one launch
reads packed X once. FA QKV (3-way) and residual stay overwrite.
`fp8_wmma` stays default-off.

Lab (`HIPFIRE_FP8_GEMM_QUICK=1`, HIP_VISIBLE_DEVICES=0, HIP 7.2): fused
vs four overwrite GEMMs **max|Δ|=0** on N=32, qkv_m=48, z_m=32,
beta_m=16, alpha_m=32, K=256 (uneven M so a 16-row tile straddles
banks). Gate+up dual-GEMM check and the three small GEMM tiles still
ALL PASS.

27B thinking-off serve still language (Paris / Seine / Eiffel +
merge_sort). Attractor=0. `finish=length` at 64 is the cap. Not a
tok/s or admission claim.

Related: [[fp8-fused-gate-up-gfx1201]], [[fp8-e2e-qwen38-27b]],
[[r9700-fp8-branch-status]].
