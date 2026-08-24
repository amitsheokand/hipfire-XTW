---
title: Whittle serve battery vs dense; 32k ctx eats the decode win
date: 2026-08-24
tags: [r9700, gfx1201, whittle, serve-harness, kv]
---

Greedy battery, thinking off, spec off, kv q8, max=64. Attractor=0 both
SKUs. Length-runaway is the 64-token cap. Whittle factual `finish=stop`.

Whittle avg decode **29.6** at harness default max_seq=32768 vs dense
**36.7**. Same Whittle at max_seq=2048 is **43.5** (matches protocol-lite).
Dense at 32k stayed in the 36.6 bench band. The MoE win is short-ctx;
32k KV tax is not expert-pager-shaped. Pin max_seq on serve comparisons.
Checkpoint:
docs/perf-checkpoints/2026-08-24-whittle-vs-dense-serve-battery-r9700.md.
