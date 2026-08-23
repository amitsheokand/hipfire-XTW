---
title: gfx1201 FP8 GEMM pack_f32_to_fp8 pre-pass — large-N prefill win
date: 2026-08-23
tags: [gfx1201, fp8, wmma, pack, gemm]
---

Replaced in-kernel `cvt_pk_fp8_f32` on F32 X with existing
`pack_f32_to_fp8_gfx12` via `Gpu::ensure_fp8_x` (same cache-by-ptr as
HFP4G32 FP8 GEMM). Lab times are **GEMM-only after pack cache warm**.
Not a tok/s or admission claim. `fp8_wmma` stays default-off.

Radiowave (packed X): **vgpr 68, sgpr 20, spill 0, inst 375, gld 34,
wait 94** vs in-kernel cvt 76 / 20 / 0 / 533 / 50 / 130. Granule-16 both
round to 80 VGPR — occupancy unchanged. Win is dropped convert ALU +
4× less X traffic.

Prefill `HIPFIRE_FP8_GEMM_QUICK=prefill` ALL PASS 2026-08-23, HIP 7.2,
`HIP_VISIBLE_DEVICES=0`. NRMSE **bit-identical** to in-kernel cvt table
(2.906 / 2.787 / 2.749 / 2.673 / 2.663 / 2.641 / 2.664e-2).

| Shape | in-kernel cvt µs | packed GEMM µs | Δ |
|---|---:|---:|---:|
| oracle 16×16×256 | 10.27 | 6.71 | launch-dominated |
| tile 32×64×256 | 8.15 | 5.80 | launch-dominated |
| small 128×128×512 | 11.63 | 9.07 | launch-dominated |
| gate_up 2048×4096×2048 | 1586.54 | 772.58 | **−51%** |
| qkv 512×2048×2048 | 217.45 | 118.08 | **−46%** |
| ffn1 128×11008×2048 | 237.25 | 256.98 | **+8.3%** |
| attn 2048×2048×2048 | 726.97 | 420.94 | **−42%** |

ffn1 (thin N=128, fat M) is the exception: A traffic dominates, X convert
was hidden. Δ≥5% noted; one process, 20 trials after 3 warmup — not a
fresh-process protocol claim. Do not reverse the pack on this shape
alone. Product path still pays pack once per unique X pointer (not in
these µs).

Related: [[fp8-gemm-lds-xpanel-vgpr-cliff]], [[r9700-fp8-branch-status]].
Next: encoder + `fp8_wmma` dispatch into one prefill GEMM (phase 5),
still default-off.
