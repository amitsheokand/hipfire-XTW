---
title: gfx1201 FP8 fused FA QKV GEMM (qt=40)
date: 2026-08-24
tags: [gfx1201, r9700, fp8, prefill, gemm, fused, qkv]
---

One lever after fused QKVZA: fused FA QKV for `FP8E4M3G256` prefill
(wq / wk / wv). Same WMMA loop as gate+up / QKVZA; grid.x covers the
three M banks so one launch reads packed X once. Residual wo/down stay
overwrite. `fp8_wmma` stays default-off.

Lab (`HIPFIRE_FP8_GEMM_QUICK=1`, HIP_VISIBLE_DEVICES=0, HIP 7.2): fused
vs three overwrite GEMMs **max|Δ|=0** on N=32, q_m=48, k_m=32, v_m=16,
K=256 (uneven M so a 16-row tile straddles banks). Gate+up, QKVZA, and
the three small GEMM tiles still ALL PASS.

Wired on Qwen3.5 dense/MoE FA and llama FA batch.

27B thinking-off serve still language (Paris / Seine / Eiffel +
merge_sort). Attractor=0. `finish=length` at 64 is the cap. Not a
tok/s or admission claim.

Related: [[fp8-fused-qkvza-gfx1201]], [[fp8-fused-gate-up-gfx1201]],
[[fp8-e2e-qwen38-27b]], [[r9700-fp8-branch-status]].
