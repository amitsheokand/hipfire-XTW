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

**Runtime 2026-08-29 after mixed-QKVZA fix (daemon md5 `e020bd5b…`, hipfire `f6dde368…`):** same MIX file generates. JIT of `gemm_q8_0_wmma_gfx12` needs `nix develop` (ROCm device lib). `serve_harness` thinking off, greedy, max=64, kv=q8, spec off:

- cold factual (prompt md5 `1d32df5f…`): Paris / Seine / Eiffel, `finish=length`, prefill 31.1 tok/s, decode 20.4 tok/s
- warm battery (same home): factual `finish=length` prefill 354.6 tok/s decode 24.9 tok/s; code (prompt md5 `02daccb4…`) `def reverse_string(s: str) -> str: return s[::-1]`, `finish=stop`, gen=23, decode 25.0 tok/s, empty=0 attractor=0

Single-session numbers, not a quality or perf claim. `runaway=1` is the 64-token factual cap, same class as XT. KLD still not run (no Qwen3.8-27B kldref; `fetch-eval-refs.sh` only has 3.5-9B / 3.6-27B). Local models dir has MIX + XT mq4, not published mq4-pro. Serve units stay masked.

`in_proj_z` is **not** tiny: shape `[6144, 5120]`, extra **16.7 MiB** each vs `a`/`b` at 128 KiB. Bundling all 48 z with qkv is ~802 MiB extra — it does **not** fit the xt→pro 1.38 GiB budget (a rebuild kept 33 complete GDN groups and dropped FA). Mixed QKVZA (Q8 qkv/a/b + MQ4 z) is the intended MIX analog; do not re-promote z just to hit fused Q8.
