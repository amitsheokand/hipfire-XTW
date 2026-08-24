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
τ=6.94. Checkpoints: prose, code, and
docs/perf-checkpoints/2026-08-24-qwen38-mtp-spec-and-dflash-nograph-r9700.md.
