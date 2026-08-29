---
title: Qwen3.8-27B MIX policy spends Q8 on GDN in_proj_qkv, not lm_head
date: 2026-08-29
tags: [qwen38,27b,astrea,mix,imatrix,quant]
---

Calibration MIX on local `~/.hipfire/models/qwen38-27b.mq4` (md5 `129909ad…`, 14 980 361 216 B, qt=13 MQ4G256, embed Q8 / lm_head MQ4 — XT-sized) using bartowski `Qwen3.8-27B-imatrix.gguf` (14 MB at `~/.hipfire/calib/`). Join: 496/496 body tensors, 0 unmatched, lm_head unscored.

Astrea `policy` base=mq4 promote=q8 ranked **Gated-DeltaNet `in_proj_qkv`**, not the product-ladder lifts (`lm_head`, `ssm_out`). Old policy only pulled `in_proj_a/b` when those ranked; **47/48 layers left `in_proj_z` MQ4**. Runtime now treats that as mixed QKVZA (unfused). Astrea also now pulls z/a/b whenever qkv is selected. MLP never selected.

| budget | selected extra | who gets Q8 |
|---|---:|---|
| 682 MiB (xt→base size) | 675 MB | 24 late-ish GDN qkv layers |
| 1.38 GiB (xt→pro size) | 1 483 MB | **all 48** GDN qkv + FA q/k/v on layers 3,7,59 |
| 1.70 GiB (mq4 class ceiling) | ~1.70 GB | all GDN qkv + 12/16 FA layers |

Policies: `~/.hipfire/calib/qwen38-27b-mix/policy-*.json`.

**Candidate written 2026-08-29** (policy is still not quality evidence):
- parent: `Qwen/Qwen3.8-27B` safetensors, 18/18 shards, 55 563 006 776 B, `~/.hipfire/hf-cache/qwen3.8-27b`
- `astrea promote` 1.38 GiB policy → `~/.hipfire/models/qwen38-27b.mix-qkv.hfq`
- bytes `16463519700` md5 `321fedb522243ed6d61a6942945f6e5c` bpw **4.897** (mq4 class)
- dtypes F16=305 MQ4G256=343 Q8F16=203 (was 49 Q8; +154 promotions)
- extra 1 483 161 600 B; −662 572 B vs published mq4-pro size

Equal-byte test that matters: this MIX (GDN `in_proj_qkv` Q8) vs published `mq4 pro` (`ssm_out` Q8). Next: KLD vs XT file and vs mq4-pro; do not swap serve until then.

**Runtime 2026-08-29 page-fault (R9700 @ 18b53f8b):** MIX loaded then GPU faulted. Cause: release `debug_assert` skipped; fused Q8 WMMA read MQ4 `in_proj_z`.

**Runtime 2026-08-29 after mixed-QKVZA fix (daemon md5 `e020bd5b…`, hipfire `f6dde368…`):** same MIX file generates. `serve_harness` factual, thinking off, greedy, max=64, kv=q8, spec off, prompt md5 `1d32df5f…`: Paris / Seine / Eiffel, `finish=length`, prefill 31.1 tok/s, decode 20.4 tok/s (single run, not a claim). JIT of `gemm_q8_0_wmma_gfx12` needs `nix develop` (ROCm device lib). `runaway=1` is the 64-token cap, same class as XT. KLD still not run. Serve units stay masked. Optional: re-promote with qkv→z/a/b bundle for uniform fused Q8.
