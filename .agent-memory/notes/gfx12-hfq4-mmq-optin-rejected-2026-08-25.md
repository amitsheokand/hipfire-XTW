---
title: gfx12 HFQ4 16x16 i8 MMQ rejected for dense MQ4 prefill
date: 2026-08-25
tags: [gfx1201, r9700, mq4, mmq, prefill, negative-result]
---

Dense Qwen3.8-27B is **MQ4G256 uniform** (md5 `129909ad`), not Lloyd. Prefill
GEMMs are fused HFQ4 fp16 WMMA (`gemm_gate_up_hfq4g256_wmma_gfx12` K4, 16×16,
LDS 0). `ArchCaps::has_mmq` is gfx906 ∪ RDNA3 only, so `should_use_mmq` never
fired on gfx1201 despite `gemm_hfq4g256_residual_mmq.gfx12.hip` existing.

**Hypothesis:** HIPFIRE_MMQ=1 should close the llama.cpp pp2520 gap (721 vs
~1108) by using i8 WMMA + Q8_1 X, the gfx11 MMQ route.

**Measurement** (2026-08-25, `hipfire bench --matrix --pp 2520 --ctx 128
--tg 16 --spec off --kv-mode q8 --backend noslots`, n=3/warmups=3,
`PREFILL_MAX_BATCH=256`, bins `31a5f6c3` / `483d7f82`, commit `0c039d83`
+ local `should_use_mmq` opt-in):

| Arm | pp2520 median | tg16@128 |
|---|---:|---:|
| default fused fp16 WMMA | **717.6** (718.7/717.6/716.1) | 36.51 |
| `HIPFIRE_MMQ=1` | **351.5** (352.4/351.5/351.2) | 36.42 |

**−51% prefill.** Decode held (GEMV). Correctness:
`test_hfq4g256_mmq_portable` ALL PASS (NRMSE 1.8–2.4e-3).

**Why:** gfx12 MMQ is still a 16×16 tile with LDS=0, and gate_up becomes two
`mmq_set_prequant` launches (fusion lost). gfx11's MMQ win was **128×128 LDS
tiles**, not i8-vs-fp16 at the same 16×16 grid. Do not set
`has_mmq |= is_rdna4`. Keep HIPFIRE_MMQ=1 as the measured-dead opt-in.

**Next lever:** fused HFQ4 gfx12 **mb4** (16×64 batch fanout, keep one
gate+up launch) — Phase D on Lloyd gfx11 was 1.4–2.2× at large M. Not a
larger ubatch (already rejected, see `llama-cpp-rdna-boosts-prefill-steal`).
