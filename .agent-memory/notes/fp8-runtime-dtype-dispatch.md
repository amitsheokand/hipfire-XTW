---
title: qt=40 FP8E4M3G256 loads and dispatches on gfx1201
date: 2026-08-23
tags: [gfx1201, r9700, fp8, dtype, dispatch]
---

`DType::FP8E4M3G256` (qt=40) is a RAW_CODECS passthrough. GEMV resolves
to `KernelKey::GemvFp8E4m3G256` under `ArchPredicate::HasWmmaGfx12`.
`weight_gemm` calls `Gpu::fp8_gemm_e4m3_g256`. Native opt-in; **not**
gated on `fp8_wmma` (that flag is HFP4's optional FP8 activation path).

**Not** in `is_batchable_la` (llama or qwen35). The QKV `else` is
`gemm_qkv_hfq4g256` and would treat E4M3 bytes as HFQ4. Prefill falls
back to per-token `weight_gemv` (slow, correct). No MoE-3D encoder path.

Encoder→GPU cosine oracle (R9700, HIP 7.2, `HIP_VISIBLE_DEVICES=0`,
N=M=16 K=256, 2026-08-23): encoder NRMSE vs F32 W@X **2.533e-3**; GEMV
NRMSE vs dequant **1.156e-3**, cosine vs W_f32 **0.999998**; GEMM NRMSE
vs dequant **2.518e-3**, cosine **0.999995**. Synth tiles still ALL PASS
(`HIPFIRE_FP8_GEMM_QUICK=1`). Not a tok/s or admission claim.

Related: [[fp8-encoder-cpu]], [[r9700-fp8-branch-status]],
[[fp8-gemm-pack-prepass]].
