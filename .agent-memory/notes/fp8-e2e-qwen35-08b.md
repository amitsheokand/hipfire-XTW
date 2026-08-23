---
title: Qwen3.5-0.8B qt=40 encode/load/serve on gfx1201
date: 2026-08-23
tags: [gfx1201, r9700, fp8, encoder, serve, e2e]
---

ai-mac had no ingestible BF16/F16 Qwen3.5 (MLX 4-bit + one Q8 GGUF). Downloaded
`Qwen/Qwen3.5-0.8B` BF16 to `~/.hipfire/hf-cache/Qwen3.5-0.8B/` (1.75 GB
safetensors). Encoded with `hipfire-quantize --format fp8e4m3 --uniform`.

Artifact: `~/.hipfire/models/qwen35-0.8b-fp8e4m3.hfq` (801.9 MB). 186×
FP8E4M3G256 (all 2D K%256==0), 19× Q8F16 (embed + 18 conv1d), 115× F16
(norms / A_log / dt_bias). Vision+MTP skipped. No HFQ4G128 ragged fallback.

Two encoder extract bugs from `d1d172e9` blocked the first encode/load:

1. `handle_moe_expert_3d` was called on every tensor (`is_moe_expert_3d`
   computed but unused) → panic `inner_shape[1]` on rank-2 dense weights.
2. `handle_main_quant` dropped the F16 else (norms/biases) → load panic
   `tensor not found: norm.weight`.

Prefill `fp8_gemm_e4m3_g256` asserted 2D `X[N,K]`; serve activations are
flattened 1D. Relaxed to numel (`X>=N*K`, `Y>=N*M`); GEMV likewise.

R9700 (`HIP_VISIBLE_DEVICES=0`, HIP 7.2) `serve_harness.py` one turn,
`--thinking off --speculation off --max-tokens 48`: load 24/24 layers,
prefill 16.8 ms / 1428 tok/s, decode 148 tok/s, finish=stop, decoded `5`.
Harness flagged attractor — **expected**: 0.8B is incoherent on hipfire at
every precision (`docs/plans/kld-measurements-master.md`). Not a quality,
tok/s, or admission claim. Proves qt=40 load + gfx1201 overwrite GEMM +
GEMV decode ran. Coherence-valid next encode is 4B.

Related: [[fp8-encoder-cpu]], [[fp8-batched-prefill-overwrite-gemm]],
[[r9700-fp8-branch-status]].
