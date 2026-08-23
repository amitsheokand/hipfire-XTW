---
title: FreeToken (arXiv 2608.16157) is NVIDIA edge-MoE serving, not DFlash
date: 2026-08-23
tags: [freetoken, moe, weight-pager, serving, parked]
---

Paper + FlashML-org/FreeToken: host-resident expert pool, GPU LRU of
complete (layer, expert) slots, q* = m*B_P/B_H miss split (PCIe fill vs
CPU in-place, exact merge), semantic-anchor recurrent checkpoints at
think/tool/turn tokens. CUDA/Python, NVIDIA-only. Does not help gfx1201
FP8 GEMM or spec-decode. Overlap is experimental.moe_expert_cache_mb and
WeightPager v0.1 (no eviction yet). hipfire pager spec cites SpecMD that
LRU can lose to staleness-aware eviction — do not copy “always LRU”.
Park until a MoE pool does not fit 32 GB. LCP dies on harness *edits*;
anchors are the piece LCP does not cover. Measure on the daemon with
byte-identical prompts if this is ever tried. Briefing:
docs/investigations/2026-08-23-r9700-fp8-external-refs.md.
