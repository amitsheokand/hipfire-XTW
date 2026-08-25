---
title: llama.cpp rdna-boosts prefill steal on R9700 — ubatch 2048 rejected, WMMA FA already on
date: 2026-08-25
tags: [r9700, gfx1201, prefill, flash-attn, gdn, llama.cpp]
---

Compared stew675 `rdna-boosts` Qwen3.8-27B Q8_0 pp2520 (~1108 tok/s tweaked ROCm,
KV BF16, non-MTP) to hipfire dense MQ4 q8 KV on the same R9700. Do not vendor
GGUF, Vulkan, or BF16 KV.

**Matched shape:** `hipfire bench --matrix --pp 2520 --ctx 128 --tg 16 --spec off
--kv-mode q8 --backend noslots`. Synthetic `bench_prefill`. Bins
`19a9d2c0` / `8dc28aed`, model md5 `129909ad`, commit `0c039d83` plus local FA
timer. Protocol-lite short-prompt ~400 prefill tok/s is **not** this row.

| Arm | pp2520 median |
|---|---:|
| default (`PREFILL_MAX_BATCH=256`, query16 WMMA FA) | **721** |
| `HIPFIRE_PREFILL_MAX_BATCH=2048` (llama.cpp `-ub 2048`) | 573 (−20%) |
| `HIPFIRE_FLASH_PREFILL=0` | 651 (−10%) |
| `HIPFIRE_FLASH_PREFILL_KERNEL=scalar` | 649 |

R9700.md item 3 (GDN ubatch ≥2048) does **not** transfer. Keep 256. GDN
`batch_seq` is sequential per token inside the kernel; larger chunks slow
GEMM/FA occupancy instead of helping.

WMMA flash prefill is already the gfx1201 default inside the query16 envelope
and is a real +11% vs legacy LDS/tiled. It was invisible to `HIPFIRE_PROFILE`
until `attention_q8_0_flash_prefill_wmma` got `begin_timer` (same omission class
as gfx12 Q8 GEMM in `docs/lessons_learned/gfx12_prefill_wmma_2026_05_19.md`).
Timed pp2520: FA **6.9%**, GDN 4.8%, gate_up+residual+qkvza WMMA **~80%**.
Closing the remaining gap vs llama.cpp pp is GEMM tile/occupancy, not FA and
not ubatch. Do not flip FA default off.
