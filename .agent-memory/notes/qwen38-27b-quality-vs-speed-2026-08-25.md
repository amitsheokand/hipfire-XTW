---
title: Qwen3.8-27B MQ4 quality AR vs speed MTP on current bins
date: 2026-08-25
tags: [gfx1201, r9700, mq4, mtp, dflash2, protocol-lite, protocol-510, quality]
---

Quality AR vs speed MTP vs DFlash 2 on dense MQ4 think-off. Protocol-lite
n=3/warmups=3, `--max-tokens 128 --backend noslots --workload stateless
--kv-mode q8`, `HIPFIRE_DPM_WARMUP_SECS=10`, `HIP_VISIBLE_DEVICES=0`,
`nix develop`. Verify graphs left on. Draft pinned via
`HIPFIRE_DFLASH_DRAFT` (filename does not auto-match).

## Identity

| field | value |
|-------|-------|
| commit | `5f4f6bef` |
| hipfire md5 | `d66acfe8e351bf923fb44d20bd28d157` |
| daemon md5 | `0148617792c7626f2f513e1754a5a60d` |
| model | `~/.hipfire/models/qwen38-27b.mq4` md5 `129909ad0fed21dcf72b5b9225e85604` |
| DFlash 2 draft | `~/.hipfire/models/qwen38-27b-dflash2.hfq` md5 `cdf5dd280cb8a33455579b7e82ef0a58` |
| GPU | gfx1201 R9700, HIP 7.2. AR/MTP vram_free 16292 MB; DFlash 2 10480 MB |

## Quality arm — AR

`--spec off` **and** `HIPFIRE_QWEN_MTP=0`. `--spec off` alone is **not**
enough if `HIPFIRE_QWEN_MTP=1` is leaked in the shell (first pass of this
session: JSON said `speculation=off` but τ=2.4 / 47.7 tok/s — MTP).

| metric | median | samples |
|--------|--------|---------|
| decode tok/s | **36.6** | `[36.6, 36.6, 36.6]` |
| prefill tok/s | 399.3 | `[398.8, 399.6, 399.3]` |
| wall tok/s | 36.0 | |
| ttft ms | 60.1 | |
| τ | none | `[]` |

JSON: `/tmp/r9700-quality-speed/ar-real.json`. Sidecar still loads; it is
not used. Matches Aug-24 5/10 AR **36.7** within 0.3%.

Serve quality (historical, same model): factual + merge_sort Pass, attractor=0.
See [[qwen38-27b-r9700-selection-2026-08-24]].

## Speed arm — MTP (general prompt)

`--spec mtp`. Confirmed: 128 tok, 53 windows, `ar_windows=0`, τ=2.40.

| metric | median | samples |
|--------|--------|---------|
| decode tok/s | **47.7** | `[47.7, 47.7, 47.7]` |
| prefill tok/s | 330.1 | `[330.1, 330.1, 330.5]` |
| wall tok/s | 46.5 | |
| ttft ms | 72.0 | |
| τ | **2.40** | `[2.4, 2.4, 2.4]` |

JSON: `/tmp/r9700-quality-speed/mtp.json`. vs AR decode **1.30×**. Matches
Aug-24 5/10 MTP **47.8**. Serve quality (historical): MTP greedy Pass.

## DFlash 2 — default prompt (relativity md5 `d94d3115`)

`--spec dflash`, `HIPFIRE_DFLASH_DRAFT` pinned, `HIPFIRE_QWEN_MTP=0`.
Confirmed DFlash 2 (grouped conv + selector), not DFlash 1. 128 tok,
37 windows, τ=2.43.

| metric | median | samples |
|--------|--------|---------|
| decode tok/s | **47.2** | `[47.2, 47.2, 47.3]` |
| prefill tok/s | 262.3 | `[261.9, 262.3, 263.8]` |
| wall tok/s | 45.6 | |
| ttft ms | 91.5 | |
| τ | **2.43** | `[2.43, 2.43, 2.43]` |

JSON: `/tmp/r9700-quality-speed/dflash-prose.json`. vs MTP 47.7: **−1.0%**
(wash). vs AR: **1.29×**. TTFT 91.5 vs MTP 72.0. Draft costs ~5.8 GB
(10480 vs 16292 free).

## Code prompt (`merge_sort`, md5 `51a6c7360e00a7853521a6575975b8e3`)

Same protocol-lite, graphs on. First DFlash-code attempt reused the
relativity prompt (`$@` did not pass through `bash -lc`); discarded.
Rerun below is the real code cell.

| arm | decode | τ | prefill | TTFT | windows | samples decode |
|-----|--------|---|---------|------|---------|----------------|
| MTP | **75.5** | 3.74 | 352.7 | 104.0 | 34 MTP | `[75.6, 75.4, 75.5]` |
| DFlash 2 | **107.5** | **6.94** | 321.1 | 115.2 | 16 | `[107.3, 107.5, 107.5]` |

JSON: `/tmp/r9700-quality-speed/mtp-code.json`,
`/tmp/r9700-quality-speed/dflash-code.json`. DFlash 2 vs MTP **1.42×**,
vs AR 36.6 **2.94×**. τ=6.94 matches Aug-24; tok/s is above the 101.8
GPU-topk row (current bins + graphs on).

## Product 5/10 (2026-08-25)

`--runs 5 --warmups 10`, same flags otherwise. hipfire `d66acfe8…`
daemon `01486177…`. AR/MTP: default prompt md5 `d94d3115`. DFlash 2:
code prompt md5 `51a6c736`. Graphs on.

| arm | prompt | decode | τ | prefill | TTFT | samples decode |
|-----|--------|--------|---|---------|------|----------------|
| AR | relativity | **36.7** | — | 399.2 | 60.1 | `[36.7, 36.7, 36.7, 36.6, 36.6]` |
| MTP | relativity | **47.6** | 2.40 | 330.1 | 72.0 | `[47.7, 47.6, 47.6, 47.5, 47.6]` |
| DFlash 2 | merge_sort | **107.4** | **6.94** | 319.9 | 115.7 | `[106.6, 107.4, 107.4, 107.4, 107.1]` |

JSON: `/tmp/r9700-quality-speed/ar-510.json`, `mtp-510.json`,
`dflash-code-510.json`. MTP 53 windows / `ar_windows=0`. DFlash 2
detected, 16 windows, vram_free 10480. Matches protocol-lite within
0.3%. MTP vs AR **1.30×**. DFlash 2 vs AR (different prompt; AR code
was 36.6 on this trunk) **2.93×**.

## Picks on these bins

| goal | pick | 5/10 decode |
|------|------|-------------|
| quality (no spec distortion) | AR | 36.7 |
| speed, general/prose | MTP | 47.6 (DFlash 2 prose lite 47.2 is a wash, more VRAM) |
| speed, code | DFlash 2 | 107.4 (MTP code lite 75.5) |

## Not in this ranking

- Think×MTP 54.7 is max=256, not comparable to these 128-tok cells.
- Whittle AR ~43.6 is faster than dense AR but research-preview quality.
- FP8 AR 20.6 / MTP 38.7 is slower than both MQ4 arms.

## Serve config applied 2026-08-25

Persisted on this machine (`~/.hipfire/models.toml` + `config.toml`), not
in-repo. `qwen3.8:27b` path points at local `qwen38-27b.mq4`.

| key | value |
|-----|--------|
| `memory.max_seq` | **262144** (VMM; 32k occupy proven) |
| `memory.kv_cache` | q8 |
| `memory.kv_backend` | **vmm** (load-checked) |
| `reasoning.mode` | on |
| `reasoning.effort` / `budget` | **xhigh** (24576 think tokens) |
| `generation.max_tokens` | **32768** |
| `speculation.mtp` | on |
| `speculation.dflash` | off (opt in with `--spec dflash`) |
| `developer.dflash_ctx_cap` | **32768** (64k now backoffs to 32k instead of silent AR) |
| `developer.dflash_draft` | `qwen38-27b-dflash2.hfq` |

VMM check (1×32 tok, not a perf row): MTP `Q8 vmm` `max_seq=65536`
`mapped_prefix=1927`, vram_free **17220** MiB. DFlash 2 `65536 -> 16384`
cap, vram_free **9264** MiB. Both RC=0.

## Context ladder (2026-08-25)

Idle load (16 tok), VMM Q8, gfx1201 32624 MiB.

| cell | max_seq | DFlash cap | free MiB | result |
|------|--------:|----------:|---------:|--------|
| MTP 64k (prior) | 65536 | 16384 | 17220 | OK |
| MTP 128k | 131072 | 16384 | 17026 | OK |
| MTP 262k | 262144 | 16384 | 16640 | OK |
| DFlash 16k cap (prior) | 65536 | 16384 | 9264 | OK |
| DFlash 32k cap | 65536 | 32768 | 4976 | OK, drafter=dflash |
| DFlash 64k cap | 65536 | 65536 | 450 | **draft hipMalloc OOM → AR** (pre-fix) |

Occupy (matrix `--runs 1 --warmups 1`, logical 262k): **pp8192 657.5 tok/s** RC=0;
**pp32768 429.7 tok/s**, tg16@32k **30.9 tok/s** RC=0. pp65536 fill started,
no `pp65536` line after ~12 min — killed, occupy-64k **unproven**.

Persisted: `max_seq=262144`, `developer.dflash_ctx_cap=32768`.

## DFlash 64k OOM fix (2026-08-25)

Root: Legacy `HIPFIRE_DFLASH_CTX_CAP=65536` allocated `HiddenStateRingBuffer`
at 64k rows (5 extract × 65536 × 5120 f32 ≈ 6.25 GiB) after draft scratch
already fit. `GpuTensor` has no Drop; a mid-loop `hipMalloc` failure leaked
partial rings, `or_free` only returned the rest to the GpuPool (no hipFree),
and the loader silently AR'd with ~450 MiB free.

Fix on `r9700-fp8-adaptive`:

1. Unwind partial ring allocs with `release_tensor_immediate`.
2. On Legacy VRAM OOM, drain the pool and retry at half cap down to 8192
   (65536 → 32768, which previously loaded with ~4.9 GiB free).
3. Drain the pool on any remaining DFlash load error so AR fallback is not
   sitting on pooled draft VRAM.

64k physical rings still do not fit this 32 GB / F16-draft / 27B MQ4
footprint. The product behaviour is keep DFlash at 32k, not silent AR.

GPU smoke after the fix (`HIPFIRE_DFLASH_WINDOW=0`,
`HIPFIRE_DFLASH_CTX_CAP=65536`, 16 tok, vmm q8):

```
DFlash draft OOM at 65536 rows (HiddenStateRingBuffer::new: HipError(2)...);
retrying Legacy cap 32768 rows
DFlash draft loaded: ...qwen38-27b-dflash2.hfq (layers=5, hidden=5120, block=8)
drafter=dflash tau=2.75 decode=49.2  vram_free_mb=4396
```

Bins: hipfire `3b3da1388b255192165c174d6e45aad9`, daemon
`47011dd3f0f274c0c743ac4f0f84dddb`.

Do not `hipfire config qwen38-27b set` while `qwen3.8:27b` already owns
that file path — duplicate catalog path. Edit the `qwen38-27b` overlay by
hand if needed.

## Caveat

Greedy MTP only. Quality serve not re-run on these bins (same model md5
as the Aug-24 Pass). DFlash 2 quality on code not re-smoked this session.
DFlash 2 prose and MTP code were protocol-lite only, not 5/10.
