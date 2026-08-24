---
title: DFlash 2 F16 fused gate+up rejected on R9700
date: 2026-08-24
tags: [dflash2, r9700, gfx1201, wmma, negative]
---

Tried gfx12 `gemm_f16_gate_up_wmma_mb1` (one WG, gate+up, shared X) after
MB1. Channel vs sequential MB1 was bit-exact. Protocol-lite prose nograph
**41.5 vs 47.1 (−11.9%)**, τ=2.43 unchanged, 37 windows. Dump: the two
`17408×5120` calls vanished. Rejected; dispatch not kept. Same fusion
class as the 2026-04-21 HFQ4 draft FFN no-op. Do not retry fused F16
QKV/KV here — extra accumulators plus 2× weight in the K-loop, and
skinny K/V fusion would not add waves (still 64 WGs at M=1024).
Checkpoint:
docs/perf-checkpoints/2026-08-24-qwen38-27b-dflash2-f16-gate-up-fuse-r9700.md.
DFlash 2 step is genre-conditional: prose ~MTP, code already wins. Next
is not another draft FFN microkernel.
