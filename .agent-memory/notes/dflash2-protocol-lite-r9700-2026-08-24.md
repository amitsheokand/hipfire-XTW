---
title: DFlash 2 protocol-lite vs MTP vs AR on Qwen3.8-27B MQ4 R9700
date: 2026-08-24
tags: [dflash2, mtp, r9700, gfx1201, protocol-lite]
---

Protocol-lite (3×128, noslots, q8, default relativity prompt md5
`d94d3115a3001f08a654d91461d6bdc4`), hipfire `d8a4677d…`, daemon `51ef5fe2…`.

AR 36.6 tok/s. `--spec mtp` alone is silent AR (no τ). `HIPFIRE_QWEN_MTP=1`
+ `--spec mtp` → 47.7 tok/s τ=2.40. DFlash 2 pinned draft → 37.2 tok/s
τ=2.43, prefill 401→260, VRAM free 16292→10480. DFlash 2 fired (not DFlash 1).
τ matches MTP; tok/s does not on this prose prompt. Serve code second-turn
93.5/τ=6.88 is a different fixture. Checkpoint:
docs/perf-checkpoints/2026-08-24-qwen38-27b-dflash2-vs-mtp-ar-r9700.md.
Do not assume DFlash 2 on Whittle. Pager still not indicated.
