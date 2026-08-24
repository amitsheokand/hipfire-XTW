---
title: R9700 bandwidth campaign: dense MQ4 then Whittle-MoE then pager
date: 2026-08-24
tags: [gfx1201, r9700, whittle, moe, dflash2, freetoken, disk]
---

Dense Qwen3.8-27B MQ4 is the local floor (fits; HBM-bound ~631 GB/s).
Do not page that trunk. Next: DFlash 2 on the parent
(z-lab/Qwen3.8-27B-DFlash2), then logic65/Qwen3.8-Whittle-MoE-27B-A17.8B
v2.1 (64×192 + shared 5120, top-16, arch_id 6). First encode risk: K=192
not a multiple of 256. Pager / q* only after bytes/token vs dense and
WeightPager P0 traces. Steal FreeToken algorithms; do not vendor CUDA.
Keep: qwen38-27b.mq4 + .mtp + Qwen3.8-27B BF16 cache. Reclaimed 2026-08-24:
0.8B/4B FP8 encodes + caches, losing 27B FP8 trunk. Plan:
docs/plans/2026-08-24-r9700-whittle-bandwidth.md.
