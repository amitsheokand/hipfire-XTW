# 2026-08-24 — DFlash 2 GPU selector top-k (R9700)

**Lifecycle: `historical`.** Evidence under the exact fixture and method below.
Not a current default, not an automatic baseline, not an admission decision.

## Why this run exists

DFlash 2's greedy selector downloaded full `[B-1, vocab]` logits (~7.6 MB
D2H per cycle on Qwen3.8-27B). DFlash 1 already avoided that with
`argmax_f32_batched`. This row is GPU `topk_f32_batched` (k≤16) plus a
small `[B, k]` D2H, then the same host lattice walk.

Compared to
[`2026-08-24-qwen38-mtp-spec-and-dflash-nograph-r9700.md`](2026-08-24-qwen38-mtp-spec-and-dflash-nograph-r9700.md)
(prose, `HIPFIRE_VERIFY_GRAPH=0`, 40.9 tok/s, τ=2.43) and
[`2026-08-24-qwen38-27b-dflash2-code-r9700.md`](2026-08-24-qwen38-27b-dflash2-code-r9700.md)
(code, graphs on, 84.7 tok/s, τ=6.94).

## Fixture

Same host / prompts / harness as those rows: Radeon AI PRO R9700, gfx1201,
HIP 7.2, `HIP_VISIBLE_DEVICES=0`, `nix develop`, `HIPFIRE_DPM_WARMUP_SECS=10`,
GPU lock, `--runs 3 --warmups 3 --max-tokens 128 --backend noslots
--workload stateless --kv-mode q8 -j`. Trunk `qwen38-27b.mq4` md5
`129909ad0fed21dcf72b5b9225e85604`. Draft `qwen38-27b-dflash2.hfq` md5
`cdf5dd280cb8a33455579b7e82ef0a58`, pinned via `HIPFIRE_DFLASH_DRAFT`.

| | |
|---|---|
| hipfire md5 | `0a0bde09a1144eb6c0c9fdb32efa3495` |
| daemon md5 | `ff448beb840ec868369bedfcf419ca23` |
| prose prompt | default relativity md5 `d94d3115a3001f08a654d91461d6bdc4` |
| code prompt | merge_sort md5 `51a6c7360e00a7853521a6575975b8e3` |

Prose arm: `HIPFIRE_VERIFY_GRAPH=0` (matched to the 40.9 row). Code arm:
default verify-graph on (matched to the 84.7 row).

## Result

| arm | decode | prefill | wall | TTFT | τ | windows |
|---|---:|---:|---:|---:|---:|---:|
| prose nograph **before** (host logits D2H) | 40.9 | 261.4 | 39.7 | 91.8 ms | 2.43 | 37 |
| prose nograph **GPU top-k** | **44.9** | 261.6 | 43.5 | 91.7 ms | **2.43** | 37 |
| code graphs-on **before** | 84.7 | 320.8 | 78.6 | 115.3 ms | 6.94 | 16 |
| code graphs-on **GPU top-k** | **101.8** | 319.9 | 93.3 | 115.7 ms | **6.94** | 16 |

JSON: [`…-prose-nograph-r9700.json`](2026-08-24-qwen38-27b-dflash2-gpu-topk-prose-nograph-r9700.json),
[`…-code-r9700.json`](2026-08-24-qwen38-27b-dflash2-gpu-topk-code-r9700.json).

Samples: prose decode 44.9 / 44.9 / 44.8. Code 102.1 / 101.8 / 101.8.
Log: `DFlash 2 draft detected (grouped conv + candidate selector); not running as DFlash 1`.
First prose warmup 14.0 tok/s then 45.8 — new kernel JIT, then the measure.

Vs matched prior cell: prose **+9.8%** decode; code **+20.2%**. τ and window
counts unchanged → GPU top-k matches host `dflash2_topk_rows` on these
greedy cells. Prefill/TTFT unchanged (selector is decode-path). VRAM free
after load still 10480 / 32624 MB.

Vs AR 36.6 on the same protocol: prose DFlash 2 is now 44.9 (still under
MTP 47.8 τ=2.40). Code DFlash 2 101.8 vs MTP 75.7.

## Reading

The missing D2H was real and bigger on the high-τ code cell (fewer cycles
but the same ~7.6 MB/cycle, and more of the cycle was wasted on PCIe).
Prose still does not convert τ=2.43 into a win over native MTP; the
leftover is draft+verify work, not logit download. GEMV lm_head fallback
still uses host `dflash2_greedy_select`. Not pager. Not Whittle.
