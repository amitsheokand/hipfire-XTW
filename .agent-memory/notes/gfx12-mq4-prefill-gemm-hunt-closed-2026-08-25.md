---
title: gfx12 dense MQ4 prefill GEMM tile hunt closed 2026-08-25
date: 2026-08-25
tags: [gfx1201, r9700, mq4, prefill, wmma, parked]
---

Dense Qwen3.8-27B MQ4 pp2520 vs llama.cpp stew675 `rdna-boosts` Q8_0 (~1108).
Do not vendor GGUF / Vulkan / BF16 KV. Hunt closed; leftover is not another
hipfire tile/occupancy lever on this family.

| Lever | Result |
|---|---|
| ubatch 2048 (`PREFILL_MAX_BATCH=2048`) | **−20%** reject |
| FA off / scalar FA | −10%; WMMA FA already default, keep |
| gfx12 16×16 i8 HFQ4 MMQ | **−51%** keep `HIPFIRE_MMQ=1` opt-in only |
| Force BT B=12 / B=4 / 1-acc at N=256 | −4.6 / −11 / −25%; **B=8 stays** |
| `PREFILL_MAX_BATCH=192` (exact B=12) | **+1.1%** wash; qkvza/qkv bt12 **spills** |
| Muse residual B=16 | VGPR spill; do not re-add |
| FA QKV BT8 | **+2.3%** keep (under 5% claim bar) |
| gate_up BT `gcn-iterative-ilp` | **+0.1%** revert |

Production BT8: 139 VGPR, 0 spill, ~11 waves. Occupancy is not the binder.
pp2520 after QKV BT: **732**. Short-prompt protocol-lite ~400 is a different row.

Parked: llama.cpp-scale prefill wants a different GEMM mechanism (larger LDS
tile / fused i8 MMQ), not B-width, chunk, FA, or this scheduler.
