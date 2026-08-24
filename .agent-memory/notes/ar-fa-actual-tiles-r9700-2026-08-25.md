---
title: AR hipGraph Q8 flash actual_tiles recovers Whittle 32k decode
date: 2026-08-25
tags: [r9700, gfx1201, whittle, hipgraph, flash-attn, serve-harness]
---

AR hipGraph used to bake Q8 flash Y-grid = ceil(max_seq/tile). At 32768/128
that is 256 mostly-empty tiles on every replay. Recapture-on-growth + live
actual_tiles (Qwen35 `ar_fa_grow_recapture`). Redline and verify graphs still
keep max_tiles.

Same serve battery as 2026-08-24: greedy, thinking off, spec off, kv q8,
max=64, max_seq=32768. Whittle **43.6** (was 29.6). Dense **36.8** (was 36.7).
Attractor=0. The 32k tax was graph grid, not KV and not idle experts — do
not start the pager. Checkpoint:
docs/perf-checkpoints/2026-08-25-whittle-ar-fa-actual-tiles-r9700.md.
