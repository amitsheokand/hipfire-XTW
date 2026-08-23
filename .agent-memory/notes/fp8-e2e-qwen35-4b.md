---
title: Qwen3.5-4B qt=40 encode/load/serve on gfx1201
date: 2026-08-24
tags: [gfx1201, r9700, fp8, encoder, serve, e2e, qwen35-4b]
---

`Qwen/Qwen3.5-4B` BF16 (9.32 GB, two shards) encoded with
`hipfire-quantize --format fp8e4m3 --uniform` (vision/MTP off).

Artifact: `~/.hipfire/models/qwen35-4b-fp8e4m3.hfq` (4.1 GB / 4319.6 MB).
248× FP8E4M3G256 (all 2D K%256==0), 25× Q8F16 (embed + 24 conv1d), 153×
F16 (norms / A_log / dt_bias). 454M vision+MTP params skipped. No
HFQ4G128 ragged fallback. Hidden 2560, intermediate 9216, 32 layers,
LA/FA interval 4.

R9700 (`HIP_VISIBLE_DEVICES=0`, HIP 7.2) thinking-off / speculation-off
`serve_harness.py` two turns, `--max-tokens 64` (finish=length is the
cap, not a loop). Load 32/32 layers. First serve outside `nix develop`
JIT-failed (`fused_qk_l2_norm_scale_interleave_f32_batched` missing
ROCm device lib); rerun inside `nix develop` succeeded. Attractor=0.

Decoded text is language, not collapse:

- factual: Paris, Seine, Eiffel Tower, historical de-facto capital
- code: recursive `merge_sort` + `merge` in a python fence

Not a tok/s, quality-vs-MQ4, or admission claim. Prefill/decode numbers
existed (cold JIT then ~2k tok/s prefill / ~100 tok/s decode) — ignore
for promotion. `fp8_wmma` still default-off (overwrite GEMM prefill +
GEMV decode).

Related: [[fp8-e2e-qwen35-08b]], [[fp8-encoder-cpu]],
[[fp8-batched-prefill-overwrite-gemm]], [[r9700-fp8-branch-status]].
