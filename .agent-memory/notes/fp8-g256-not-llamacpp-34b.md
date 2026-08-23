---
title: hipfire FP8E4M3G256 is 256-elem groups; llama.cpp F8E4M3 is 32-elem/34 B
date: 2026-08-23
tags: [fp8, layout, gfx1201, the-rock8, quantize]
---

The-Monk llama.cpp `roc8` GGUF F8E4M3 is 32 elements / 34 B (leaf + fp16
scale), 8.5 bpw. hipfire `QuantType::FP8E4M3G256` (qt=40) kernels use
`[16 B hdr][n_blocks × 2 B fp16 scale, 16-aligned][n_blocks × 256 B E4M3]`.
Same builtin `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`, not
drop-in. `hfq.rs` currently comments both “group-256” and “34 B/group
8.5 bpw” — the 34 B line is a llama.cpp copy and must be fixed when the
encoder lands. Steal The-Rock8 tiling (two WMMA/K32, LDS double-buffer,
geom sweep), not the GGUF packing.
