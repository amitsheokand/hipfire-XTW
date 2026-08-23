---
title: r9700-fp8-adaptive GEMV and WMMA GEMM both GPU-validated
date: 2026-08-23
tags: [gfx1201, r9700, fp8, wmma, gemv, gemm]
---

Branch `r9700-fp8-adaptive` (fork hipfire-XTW) on `master` `80a572c8`.

- `3b9c6b70` Phase 0–2 (docs/R9700.md, experimental keys, A/B + PC-sample, qt 40/41)
- `7e345c4c` GEMV `v_dot4_f32_fp8_fp8` ALL PASS (NRMSE 2.5–2.7%, 46–58% peak BW)
- `8191effe` WMMA GEMM kernel + dispatch + lab harness
- GEMM lab `test_gemm_fp8e4m3_g256 --release` **ALL PASS** 2026-08-23 on
  R9700 HIP 7.2, `HIP_VISIBLE_DEVICES=0`: NRMSE 2.50–2.91e-2 across 8
  shapes (oracle 16×16×256 through w_down 4096×2048×11008). Matches the
  F32→E4M3 activation band; CPU ref is OCP E4M3fn bias 7, so F8_Mode 0
  is empirically confirmed.
- Radiowave inspect: 76 VGPR / 20 SGPR / 0 spill, 533 inst, 50 global
  loads, 130 waits, no LDS. `__launch_bounds__(32, 2)`. LDS X-panel
  lever **rejected** (76→155 VGPR); next is a pack_f32_to_fp8 pre-pass.
  See [[fp8-gemm-lds-xpanel-vgpr-cliff]].

HIP launch must pass `&mut` addresses. Related:
[[fp8-g256-not-llamacpp-34b]], [[rdna4-isa-fp8-wmma-confirmed]].
