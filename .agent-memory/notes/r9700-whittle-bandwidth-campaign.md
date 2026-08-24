---
title: R9700 bandwidth campaign: dense MQ4 then Whittle-MoE then pager
date: 2026-08-24
tags: [gfx1201, r9700, whittle, moe, dflash2, freetoken, disk]
---

Dense Qwen3.8-27B MQ4 is the local floor (fits; HBM-bound ~631 GB/s).
Do not page that trunk. Steps 1–4 measured: dense AR 36.6; Whittle AR
decode 43.5 / prefill 407.6 after Path 2 Q8 grouped WMMA.

DFlash 2 protocol-lite on the parent (graphs off, GPU top-k, gfx12 F16
MB1): prose **47.1** τ=2.43 vs MTP **47.8** τ=2.40 vs AR 36.6 (relativity
md5 `d94d3115…`). MB1 is +4.9% vs GPU-topk 44.9 and does not clear 5%
or beat MTP. Fused F16 gate+up rejected (41.5, −11.9%). DFlash 2 step is
genre-conditional: stop grinding draft FFN microkernels. History:
graphs-on 37.2; nograph host-logits 40.9; GPU
top-k 44.9. `--spec mtp` is live after `661c230a…` (`mtp_mode=on`).
Code merge_sort md5 `51a6c736…`: DFlash 2 GPU-topk **101.8** τ=6.94 vs
MTP 75.7 τ=3.74 vs AR 36.6. Genre-conditional. Do not put DFlash 2 on
Whittle. Adaptive-B 8→4 rejected. Do not shrink B=8.

Pager / q* only if traces say miss-bound and DDR idle — not indicated
on this 32 GB box. Plan:
docs/plans/2026-08-24-r9700-whittle-bandwidth.md.
