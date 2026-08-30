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

Single-session numbers, not a quality or perf claim. `runaway=1` is the 64-token factual cap, same class as XT. Serve units stay masked.

**KLD 2026-08-30 (local llama-b10488 WT2 teacher, not Hub-comparable):** dumped `~/.hipfire/kldref/qwen3.8-27b.ref_wt2.native-llama-b10488.bin` (50 675 552 B, md5 `3c0268e9…`). llama.cpp vulkan b10488 / `9d77fa172` on Vulkan1 R9700; `--chunks 24` `--n-ctx 2048` top-k 256. Teacher PPL **6.2391** (hiptrx BF16 native was 6.2385). GGUF `qwen3.8-27b-bf16.gguf` md5 `4f958917…` deleted after dump.

`eval_hipfire` git `36e4a282` fingerprint `47041ba6…` gfx1201, `--kv-mode q8 --kv-v q8 --scoring-mode prefill --max-chunks 24`, `HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_GRAPH=0`. All 24 chunks finite, no KLD=0.

| variant | bytes | body codec | extra Q8 | WT2 KLD | PPL |
|---|---:|---|---|---:|---:|
| XT `qwen38-27b.mq4` | 14 980 361 216 | MQ4G256 V1 | embed+conv1d | **0.066394** | 6.5684 |
| MIX `qwen38-27b.mix-qkv.hfq` | 16 463 519 700 | MQ4G256 V1 | + GDN `in_proj_qkv` (+ FA q/k/v 3/7/59) | **0.056936** | 6.5228 |
| pro `qwen3.8-27b.mq4-pro` | 16 464 182 272 | MQ4V2 | lm_head+embed+conv1d+`ssm_out` | **0.032715** | 6.3311 |

MIX beats its V1 floor (−14.2 % KLD vs XT). MIX does **not** beat equal-byte pro (codec confound: V1+qkv-Q8 vs V2+ssm_out-Q8). Artifacts: `~/.hipfire/calib/qwen38-27b-mix/wt2-kld-summary.json` + `kldseq/`.

**Hy4 residual / hybrid 2026-08-30 (allocation isolation on V1 XT):** Astrea `--role-prior residual` + pack-rank (ssm_out + lm_head before QKVZA bundles) + streaming promote (index-only prefix, chunked Q8, no 15 GB `bytearray` — the in-RAM writer OOM-killed at ~47 GB RSS). Keep MIX as qkv control. Disk kept all scored files.

| variant | bytes | extra Q8 | WT2 KLD | PPL |
|---|---:|---|---:|---:|
| residual-v1 `qwen38-27b.mix-hy4-residual-v1.hfq` | 16 457 949 140 | 48× `out_proj` + lm_head | **0.050907** | 6.5053 |
| hybrid-v1 `qwen38-27b.mix-hy4-hybrid-v1.hfq` | 16 458 819 540 | last-8 GDN `out_proj` + 30 qkv groups | **0.058232** | 6.5049 |

Residual-V1 is **between** MIX 0.0569 and pro 0.0327: allocation helps vs MIX, codec is still the larger gap vs pro. Hybrid lost to MIX. Policies: `policy-hy4-residual-v1.json` / `policy-hy4-hybrid-v1.json`; tensortypes export next to them. Serve stays masked. No Hub-number citation.

`in_proj_z` is **not** tiny: shape `[6144, 5120]`, extra **16.7 MiB** each vs `a`/`b` at 128 KiB. Bundling all 48 z with qkv is ~802 MiB extra — it does **not** fit the xt→pro 1.38 GiB budget (a rebuild kept 33 complete GDN groups and dropped FA). Mixed QKVZA (Q8 qkv/a/b + MQ4 z) is the intended MIX analog; do not re-promote z just to hit fused Q8.

**Product axes once architecture is in place (MQ4V2 + ladder xt/base/pro):** mix-bit at 4.9 bpw is mostly done. Residual-v1 0.0509 vs pro 0.0327 is the codec gap, not leftover allocation.

- **Disk / weight VRAM:** equal-byte ~15.33 GiB. No smaller SKU from another knapsack at `--max-extra-bytes 1483821056`. Smaller = mq3/mq2 + GSQ encoder (later L3) or serve `xt`.
- **Serve speed:** not measured on residual/hybrid. Ladder AR decode ~ xt 35 → base 33 → pro 32 tok/s; Q8 lm_head costs BW. Remaining lever is DFlash τ / kernels, not 1.38 GiB Q8 placement.
- **Bandwidth:** KV + spec, not Q8 role at fixed size.
- **34 GB context:** weights leave ~19 GB for KV + DeltaNet/conv. Long-ctx (windowed DFlash, CASK, KV mode, rolling buffer) is the product lever that still moves.

**N4/N5b (2026-08-30):** published mq4-xt sha256 `9f91556f…`. `mix-qkv-v2.hfq` 16 459 603 021 B, md5 `f9b55d7b…`, 33 fused QKVZA groups.

| variant | WT2 KLD | PPL |
|---|---:|---:|
| V1 XT | 0.066394 | 6.568 |
| MIX (V1+qkv) | 0.056936 | 6.523 |
| v2-xt | 0.057414 | 6.416 |
| mix-qkv-v2 | **0.050561** | 6.392 |
| residual-v1 | 0.050907 | 6.505 |
| pro | **0.032715** | 6.331 |

V2 floor ≈ MIX. V2+qkv ≈ V1+residual. Pro (V2+residual) still −0.018 KLD ahead. On V2, residual extras beat qkv extras.

Astrea `metrics` (product baseline = `mq4-pro`): **`no_quality_gain`** (KLD +0.0178, PPL +0.061). Report: *quality evidence does not justify promotion yet*. vs MIX: `quality_improved` (codec). residual-v1 vs MIX: `quality_improved` (allocation). Do not Atlas. Do not unmask serve. Product remains published mq4-pro.

Artifacts: `~/.hipfire/calib/qwen38-27b-mix/metrics-mix-qkv-v2-vs-pro.json`, `report-mix-qkv-v2-vs-pro.json`. L1–L4 not this round.

**Astrea MQ V2 type table (continue):** qt 44 `MQ4G256V2` → mq4 was N3. Same class for the rest of the V2 family so inspect/policy of published mq3-pro is not `UNKNOWN_49`: 45 `MQ4CG256`→mq4, 47 `MQ6G256V2`→mq6, 48 `MQ5G256V2`→mq5, 49 `MQ3G256V2`→mq3, 50 `MQ2G256V2`→mq2. Aliases `mq3v2`/`mq2v2`/`mq5v2`/`mq6v2`. `file_summary` skips md5 above 256 MiB (`md5_skipped`) so `metrics --candidate-model` does not hash 16G HFQs. Tests: `uv run --with numpy python3 scripts/test_astrea.py` **45/45**. Promote of an mq3 body is **not** N6.

**N6 (2026-08-30, same local llama-b10488 teacher, 24 chunks, all finite, no KLD=0):** published `qwen3.8:27b-mq3-pro` sha256 `394c50966bf4f68172df8eb34cd7ded8f9d0576c9ef24ea6ba639a88c184f795`, 13 184 433 152 B. Inspect: `MQ3G256V2` 448 + `MQ6G256V2` 48 (`ssm_out`) + `Q8F16` 50 (lm_head+embed+48×conv1d) + F16 753. `hipfire_base_format=mq3v2` `hipfire_product_tier=pro`. Prefill 154 tok/s (not a serve claim).

| variant | bytes | body | extra | WT2 KLD | PPL |
|---|---:|---|---|---:|---:|
| mq4-pro | 16 464 182 272 | MQ4V2 | lm_head+embed+conv1d+`ssm_out` Q8 | **0.032715** | 6.331 |
| v2-xt | 14 980 361 216 | MQ4V2 | embed+conv1d | 0.057414 | 6.416 |
| mq3-pro | 13 184 433 152 | MQ3V2 | lm_head+embed+conv1d Q8, `ssm_out` MQ6 | **0.130304** | 6.775 |

Astrea `metrics` vs `mq4-pro`: **`no_quality_gain`** (KLD +0.0976, PPL +0.444). Report: *quality evidence does not justify promotion yet*. This is the smaller SKU, not another 4.9 bpw knapsack. Local 0.1303 is the number that belongs in this table — do not import Hub WT2 0.130 as a second protocol. Serve stays masked. Product remains published **mq4-pro**. Closing the mq3 gap is L3 (GSQ encoder), not mix-bit at 4.9 bpw. Artifacts: `kldseq/mq3-pro__gfx1201__prefill.kldseq`, `inspect-mq3-pro.json`, `metrics-mq3-pro-vs-mq4-pro.json`, `report-mq3-pro-vs-mq4-pro.json`.

**L3 slice 1 (2026-08-30):** MQ3V2 default encoder is now least-squares + reassign on the same qt=49 wire (`HIPFIRE_MQ3V2_FIT=minmax` reproduces published). Unit tests: pack/unpack, degenerate match, Gaussian MSE never-regresses and is a strict win. Candidate `qwen38-27b.mq3v2.pro.ls.hfq` same 13 184 433 152 B / same census as published mq3-pro; `hipfire_mq3v2_fit=ls`. Same local teacher, 24/24 finite.

| variant | WT2 KLD | PPL |
|---|---:|---:|
| mq3-pro (minmax) | 0.130304 | 6.775 |
| mq3-pro-ls | **0.109235** | **6.610** |
| mq4-pro | 0.032715 | 6.331 |

Astrea vs mq3-pro: **`quality_improved`** (KLD −0.0211 / −16.2%, recovered 21.6% of the gap to mq4-pro). vs mq4-pro: **`no_quality_gain`**. p99 KLD 2.38→2.53 (mean better, tail slightly worse). Do not Atlas. Do not unmask serve. Do not replace published mq3-pro yet. Gumbel (slice 2) skipped — KLD was not flat. Next L3 lever is imatrix-weighted LS (slice 3). L1/L2/L6/production stay blocked. Artifacts: `kldseq/mq3-pro-ls__gfx1201__prefill.kldseq`, `inspect-mq3-pro-ls.json`, `metrics-mq3-pro-ls-vs-mq3-pro.json`, `metrics-mq3-pro-ls-vs-mq4-pro.json`, charter `docs/plans/2026-08-30-l3-gsq-mq3v2.md`.

**L3 slice 3 (2026-08-30):** imatrix-weighted LS (`hipfire_mq3v2_fit=ls-w`) on the same recipe. `qwen38-27b.mq3v2.pro.ls-w.hfq` 13 184 433 152 B. WT2 KLD **0.108887** PPL **6.599** vs ls 0.109235 / 6.610 — **tie** (inside noise). Column-weighting the post-FWHT grid does not move KLD.

**L3 slice 2 (2026-08-30):** Gumbel/Concrete softmax on qt=49 (`HIPFIRE_MQ3V2_FIT=gumbel`). Proto MSE **tied** unweighted LS (Gaussian 268.64, heavy-tail 2194.14, delta 0). Never-regress keeps the LS blob. **No 27B encode.** L3 encoder work stops. Do not L1/L2/L6. Product remains mq4-pro. Optional later: refresh published mq3-pro SKU to unweighted LS without overwriting Hub-pinned `qwen3.8-27b.mq3-pro` sha256 `394c5096…`.

**Production slice 1 (2026-08-30, mq4-pro serve_harness, systemd still masked):** one-shot loopback battery on published `qwen3.8-27b.mq4-pro`, tag `qwen3.8:27b-mq4-pro`, `--speculation off --kv q8 --max-seq 8192 --sampling greedy`, isolated `HIPFIRE_HOME`. hipfire `f6dde368…` daemon `e020bd5b…`. Prompts: `bare_factual.txt` md5 `1d32df5f…`, `humaneval_3_below_zero.txt` md5 `37c5aad9…`.

- `--thinking off` (TOML `thinking_budget`) is **ignored** on qwen_jinja effort-native: both turns opened `<think>`, hit max=64, validation `open think span`, empty visible (`empty=2`). Artifact: `harness-mq4-pro-prod-battery.json`.
- Retry with `--max-think-tokens 1`: **empty=0 attractor=0 think=0**. Factual: Paris / Seine / Eiffel. Code: starts the `below_zero` stub. `runaway=2` is the 64-token cap. Decode 23.0 / 29.4 tok/s is one-session, not a claim (first prefill 28.8 tok/s is JIT; second 546). Artifact: `harness-mq4-pro-prod-battery-nothink.json`. Do not write `admissions.yml`. Do not unmask. Local hardlink `qwen3.8-27b.mq3-pro-ls` → the LS file (inode shared; published mq3-pro untouched).
