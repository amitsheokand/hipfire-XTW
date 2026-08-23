---
title: Qwen3.8-27B qt=40 encode/load/serve on gfx1201
date: 2026-08-24
tags: [gfx1201, r9700, fp8, encoder, serve, e2e, qwen38-27b]
---

`Qwen/Qwen3.8-27B` BF16 (~53 GB, 18 shards) encoded with
`hipfire-quantize --format fp8e4m3 --uniform --threads 10` (vision/MTP
off). Arch `qwen3_5` (id=5). Hidden 5120, intermediate 17408, 64
layers, LA/FA interval 4, GQA 24/4, head_dim 256.

Artifact: `~/.hipfire/models/qwen38-27b-fp8e4m3.hfq` (26 GB /
27354.3 MB written). Encoder LUT + row-parallel Rayon: see
[[fp8-encoder-rayon-lut]].

R9700 (`HIP_VISIBLE_DEVICES=0`, HIP 7.2, HIP reports 34.2 GB VRAM)
thinking-off / speculation-off `serve_harness.py` two turns inside
`nix develop`, `--max-tokens 64 --max-seq 2048 --kv q8`. Load 64/64
layers. Attractor=0. `finish=length` is the token cap, not a loop.
`!RUNAWAY` is the harness label for that cap.

Decoded text is language, not collapse:

- factual: Paris, Seine, Eiffel Tower, historical importance (cut at 64)
- code: recursive `merge_sort` + start of `merge` in a python fence

Weights ~26 GB on a 32 GB card is tight; first smoke used seq 2048 /
q8 KV. Not a tok/s, quality-vs-MQ4, or admission claim. `fp8_wmma`
still default-off (fused gate_up + overwrite QKV GEMM prefill + GEMV
decode). No MoE-3D encoder.

Related: [[fp8-e2e-qwen35-4b]], [[fp8-e2e-qwen35-08b]],
[[fp8-fused-gate-up-gfx1201]], [[r9700-fp8-branch-status]].
