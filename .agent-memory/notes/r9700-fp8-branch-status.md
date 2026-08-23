---
title: r9700-fp8-adaptive GEMV + packed-X WMMA GEMM GPU-validated
date: 2026-08-23
tags: [gfx1201, r9700, fp8, wmma, gemv, gemm, pack]
---

Branch `r9700-fp8-adaptive` (fork hipfire-XTW) on `master` `80a572c8`.

- `3b9c6b70` Phase 0–2 (docs/R9700.md, experimental keys, A/B + PC-sample, qt 40/41)
- `7e345c4c` GEMV `v_dot4_f32_fp8_fp8` ALL PASS (NRMSE 2.5–2.7%, 46–58% peak BW)
- `8191effe` WMMA GEMM kernel + dispatch + lab harness
- `258e29d2` GEMM ALL PASS recorded; qt=40 comment; `HIPFIRE_FP8_GEMM_QUICK=1`
- `fff94abd` LDS X-panel **rejected** (76→155 VGPR)
- pack pre-pass: GEMM consumes `pack_f32_to_fp8_gfx12` via `ensure_fp8_x`
- CPU encoder `--format fp8e4m3` (qt=40). Runtime **loads and dispatches**
  via `DType::FP8E4M3G256` (not `is_batchable_la`). See
  [[fp8-runtime-dtype-dispatch]], [[fp8-encoder-cpu]].

Packed-X Radiowave: **68 VGPR / 20 SGPR / 0 spill**, 375 inst, 34 gld,
94 waits. NRMSE unchanged vs in-kernel cvt. Large-N prefill GEMM-only
−42% to −51% (gate_up / qkv / attn); ffn1 N=128 **+8.3%**. Occupancy
unchanged (granule-16 both round to 80 VGPR). See
[[fp8-gemm-pack-prepass]]. Encoder: [[fp8-encoder-cpu]]. `fp8_wmma` stays
default-off.

HIP launch must pass `&mut` addresses. Related:
[[fp8-g256-not-llamacpp-34b]], [[rdna4-isa-fp8-wmma-confirmed]],
[[fp8-gemm-lds-xpanel-vgpr-cliff]].
