---
title: DFlash 2 protocol-lite vs MTP vs AR on Qwen3.8-27B MQ4 R9700
date: 2026-08-24
tags: [dflash2, mtp, r9700, gfx1201, protocol-lite]
---

Protocol-lite (3×128, noslots, q8, default relativity prompt md5
`d94d3115a3001f08a654d91461d6bdc4`), hipfire `d8a4677d…`, daemon `51ef5fe2…`.

AR 36.6 tok/s. `--spec mtp` alone is silent AR on daemon `51ef5fe2…`.
After `mtp_mode=on` generate opt-in (daemon `661c230a…`) `--spec mtp` is
47.8 τ=2.40 without `HIPFIRE_QWEN_MTP`. DFlash 2 graphs-on 37.2 τ=2.43;
`HIPFIRE_VERIFY_GRAPH=0` 40.9 τ=2.43. Code prompt still DFlash 2 84.7
τ=6.94. After GPU selector top-k (daemon `ff448beb…`): prose nograph
**44.9** τ=2.43; code **101.8** τ=6.94. Adaptive-B 8→4 on prose
**rejected** (38.5 τ=1.89). gfx12 F16 GEMM MB1 (N≤32): prose nograph
**47.1** τ=2.43 (+4.9% vs 44.9, under the 5% claim bar; MTP still 47.8).
Checkpoints: prose, code, mtp/nograph, gpu-topk, adaptive-B, and
docs/perf-checkpoints/2026-08-24-qwen38-27b-dflash2-f16-mb1-r9700.md.
