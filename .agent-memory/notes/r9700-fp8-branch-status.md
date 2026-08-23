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
- CPU encoder `--format fp8e4m3` (qt=40)
- `6ff97bce` runtime DType + GEMV/GEMM dispatch + encoder→GPU cosine oracle
- batched prefill: gfx1201-only `is_batchable_la` + overwrite GEMM arms
  (llama + qwen35 dense/MoE-attn). See [[fp8-batched-prefill-overwrite-gemm]].
  `fp8_wmma` stays default-off. No MoE-3D encoder.
- E2E: Qwen3.5-0.8B `--format fp8e4m3` → load → short serve on gfx1201.
  Encoder gates restored (3D expert + F16 fallback). Flattened-X GEMM
  numel assert. 0.8B output attractor is known-incoherent, not FP8.
  See [[fp8-e2e-qwen35-08b]].
- E2E: Qwen3.5-4B qt=40 HFQ (4.1 GB, 248× FP8E4M3G256) thinking-off
  serve on gfx1201 produced readable Paris/Seine + merge_sort text.
  Attractor=0. Serve JIT must run inside `nix develop`. Not a tok/s
  or admission claim. See [[fp8-e2e-qwen35-4b]].
- Fused gate+up FP8 GEMM on gfx1201 (same WMMA as overwrite GEMM). Lab
  fused vs dual GEMM max|Δ|=0. QKV still overwrite. See
  [[fp8-fused-gate-up-gfx1201]].

Packed-X Radiowave: **68 VGPR / 20 SGPR / 0 spill**, 375 inst, 34 gld,
94 waits. NRMSE unchanged vs in-kernel cvt. Large-N prefill GEMM-only
−42% to −51% (gate_up / qkv / attn); ffn1 N=128 **+8.3%**. Occupancy
unchanged (granule-16 both round to 80 VGPR). See
[[fp8-gemm-pack-prepass]]. Encoder: [[fp8-encoder-cpu]]. Runtime:
[[fp8-runtime-dtype-dispatch]].

HIP launch must pass `&mut` addresses. Related:
[[fp8-g256-not-llamacpp-34b]], [[rdna4-isa-fp8-wmma-confirmed]],
[[fp8-gemm-lds-xpanel-vgpr-cliff]].
