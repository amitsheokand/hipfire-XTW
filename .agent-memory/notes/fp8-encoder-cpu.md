---
title: FP8E4M3G256 CPU encoder emits kernel layout (qt=40)
date: 2026-08-23
tags: [gfx1201, fp8, quantize, encoder]
---

`hipfire-quantize` `--format fp8e4m3` (aliases `fp8e4m3g256`, `fp8-e4m3`)
writes `QuantType::FP8E4M3G256` = 40. Layout matches the gfx1201 GEMV/GEMM
kernels: `[16 B hdr][n_blocks × 2 B fp16 scale, 16-aligned][n_blocks × 256 B
E4M3]`. Per-group scale is `amax/448` (OCP E4M3fn max finite). Embeddings
stay Q8F16. Ragged K falls back to HFQ4G128. No MoE-3D expert path yet.

CPU tests (`cargo test -p hipfire-quantize --lib -- fp8e4m3`): max code
0x7E = 448, G256 row size, Gaussian NRMSE < 5%. Runtime **cannot load**
qt=40 until `DType::FP8E4M3G256` + RAW_CODECS + one prefill GEMM dispatch
land. `fp8_wmma` stays default-off.

Related: [[fp8-gemm-pack-prepass]], [[r9700-fp8-branch-status]],
[[fp8-g256-not-llamacpp-34b]].
