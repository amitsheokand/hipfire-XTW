---
title: r9700-fp8-adaptive has GPU-validated GEMV; WMMA GEMM is uncommitted
date: 2026-08-23
tags: [gfx1201, r9700, fp8, wmma, gemv, gemm]
---

Branch `r9700-fp8-adaptive` (fork hipfire-XTW) sits two commits ahead of
`master` `80a572c8`. `3b9c6b70` is Phase 0–2 (docs/R9700.md, experimental
config keys, A/B + PC-sample scripts, QuantType 40/41). `7e345c4c` is the
native E4M3 G256 GEMV (`v_dot4_f32_fp8_fp8`) with lab example ALL PASS
(NRMSE 2.5–2.7%, 5% rel tol, 46–58% peak BW). Working tree (committed with this note) adds
`gemm_fp8e4m3_g256_wmma.gfx1201.hip` + `Gfx1201Device::fp8_gemm_e4m3_g256`
+ `test_gemm_fp8e4m3_g256` — not yet GPU-proven. HIP launch must pass
`&mut` addresses; `__shared__`+`__syncthreads` reduce compiled to a
24-instruction stub under `--genco`. Next: GEMM NRMSE on R9700 before any
tiling or daemon wire-up. Related: [[fp8-g256-not-llamacpp-34b]],
[[rdna4-isa-fp8-wmma-confirmed]].
