---
title: gfx12 PREFILL_MAX_BATCH=192 is a +1% wash on Qwen3.8-27B pp2520
date: 2026-08-25
tags: [gfx1201, r9700, mq4, prefill, batch-tile]
---

Picked over a new B=16 residual (Muse already spilled 53 VGPR at bt16) and
ahead of FA QKV BT because chunk 192 is the only way to make adaptive
`HIPFIRE_GATE_UP_BT` land **exact B=12** on every fused HFQ4 WMMA (gate_up +
residual + qkvza), not just one kernel.

**pp2520** (Qwen3.8-27B MQ4 md5 `129909ad`, n=3, kv q8, noslots,
`HIPFIRE_PREFILL_REUSE_PBS=1`, bins `3fe504cd` / `4986e3a5`):

| HIPFIRE_PREFILL_MAX_BATCH | adaptive B | pp2520 median | tg16@128 |
|---|---:|---:|---:|
| 256 (default) | 8 exact | **715.9** | 36.50 |
| 192 | 12 exact | 723.6 | 36.41 |

**+1.1%**. Inside the warm within-session band. Do **not** flip the gfx12
default chunk to 192. Muse's "192 beats 256 e2e" did not transfer to this
3.8-27B pp2520 fixture.

Keep `PREFILL_MAX_BATCH=256`. Do not re-add B=16. FA QKV BT was ported next
(`gfx12-hfq4-qkv-bt-2026-08-25`): +2.3% pp2520, under the 5% claim bar.
