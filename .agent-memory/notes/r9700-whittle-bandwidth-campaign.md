---
title: R9700 bandwidth campaign: dense MQ4 then Whittle-MoE then pager
date: 2026-08-24
tags: [gfx1201, r9700, whittle, moe, dflash2, freetoken, disk]
---

Dense Qwen3.8-27B MQ4 is the local floor (fits; HBM-bound ~631 GB/s).
Do not page that trunk. Steps 1–4 measured: dense AR 36.6; Whittle AR
decode 43.5 / prefill 407.6 after Path 2 Q8 grouped WMMA; DFlash 2
protocol-lite on the parent is 37.2 tok/s τ=2.43 vs MTP 47.7 τ=2.40
(`HIPFIRE_QWEN_MTP=1` required on daemon `51ef5fe2…`; `--spec mtp` alone
is MTP after `661c230a…`) vs AR 36.6
on the default relativity prompt. Code prompt (merge_sort md5 `51a6c736…`):
DFlash 2 84.7 τ=6.94 vs MTP 75.7 τ=3.74 vs AR 36.6. Genre-conditional.
Do not assume transfer onto Whittle.
Pager / q* only if traces say miss-bound and DDR idle — not indicated
on this 32 GB box. Plan:
docs/plans/2026-08-24-r9700-whittle-bandwidth.md.
