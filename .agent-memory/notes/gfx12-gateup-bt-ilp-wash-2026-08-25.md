---
title: gfx12 gate_up BT gcn-iterative-ilp is a wash on pp2520
date: 2026-08-25
tags: [gfx1201, r9700, mq4, wmma, occupancy, scheduler, prefill]
---

Production HFQ4 BT8 on gfx1201 (radiowave inspect, 1536 VGPR/SIMD):

| kernel | VGPR | spill | scratch | waitcnt | delay_alu |
|---|---:|---:|---:|---:|---:|
| gate_up bt8 | 139 | 0 | 0 | 865 | 414 |
| residual bt8 | 139 | 0 | 0 | 511 | 230 |
| qkvza bt8 | 139 | 0 | 0 | 307 | 118 |
| qkv bt8 | 139 | 0 | 0 | 294 | 117 |
| qkvza/qkv **bt12** | 256 | **21** | **480** | — | — |

Occupancy ~11 waves. Not VGPR-bound. Muse affine residual already lost vs
shared bt8 at N=256. qkvza/qkv bt12 spill is extra evidence chunk 192 cannot
win (those layers take the spilled symbol).

**Lever:** `// HIPFIRE_COMPILER_FLAGS: -mllvm -misched=gcn-iterative-ilp` on
`gemm_gate_up_hfq4g256_wmma_gfx12_bt.hip` only.

**pp2520** (Qwen3.8-27B MQ4 `129909ad`, n=3, kv q8, noslots, QKV BT already on):

| Arm | hipfire md5 | pp2520 | tg16@128 |
|---|---|---:|---:|
| default sched | `8cfacff8` | **732.1** | 36.50 |
| iterative-ilp | `439483ac` | 732.8 | 36.56 |

ISA after: vgpr 140, waits 839, delay_alu 393. **+0.1%**. Reverted. Do not
ship the flag. Remaining llama.cpp pp2520 ~1108 gap is not occupancy or this
scheduler on gate_up BT.
