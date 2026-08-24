# 2026-08-25 — AR hipGraph Q8 flash actual_tiles (R9700)

**Lifecycle: `historical`.** Evidence under the exact fixture and method below.
Not a current default, not an automatic baseline, not an admission decision.
Whittle remains a research-preview SKU.

## Why this run exists

[`2026-08-24-whittle-vs-dense-serve-battery-r9700.md`](2026-08-24-whittle-vs-dense-serve-battery-r9700.md)
showed Whittle decode **29.6** at harness `max_seq=32768` vs **43.5** at 2048.
That row guessed KV footprint. The launch grid was the tax: AR hipGraph
captured Q8 flash at `ceil(max_seq / tile)` (256 dummy tiles at 32k / 128)
and replayed that Y-grid forever. Dummy tiles exit immediately but still
occupy the CP; they are a larger fraction of Whittle's cheaper FFN cycle
than of dense.

This row is the same serve battery after AR hipGraph capture uses the live
tile count and recaptures when it grows. Redline recording and per-B
verify/tape graphs still bake `max_tiles`.

## Fixture

Radeon AI PRO R9700, gfx1201, HIP 7.2, `HIP_VISIBLE_DEVICES=0`, `nix develop`,
GPU lock. Greedy, thinking off, speculation off, kv q8, max_tokens 64,
`--max-seq 32768`. Built-in genre battery (code / reason / factual / prose /
instruct). hipfire/daemon md5 `bbb45afe…` / `2abb3435…`.

| artifact | md5 |
|---|---|
| `qwen38-whittle-27b-a17.8.mq4` | `bb8dd08b909b3e5249db1d3f9e477953` |
| `qwen38-27b.mq4` | `129909ad0fed21dcf72b5b9225e85604` |

## Result

`!RUNAWAY` is `finish=length` at the 64-token cap. Attractor=0 and empty=0
on every turn for both SKUs.

| arm | max_seq | avg decode | avg prefill | attractor | factual finish |
|---|---:|---:|---:|---:|---|
| Whittle (this row) | 32768 | **43.6** | 507 | 0 | stop @ 50 tok |
| Dense parent (this row) | 32768 | **36.8** | 464 | 0 | length @ 64 |
| Whittle (2026-08-24) | 32768 | 29.6 | 412 | 0 | stop @ 50 tok |
| Whittle (2026-08-24) | 2048 | 43.5 | 504 | 0 | stop @ 50 tok |
| Dense (2026-08-24) | 32768 | 36.7 | 466 | 0 | length @ 64 |

JSON: [`…-whittle-seq32768.json`](2026-08-25-whittle-ar-fa-actual-tiles-r9700-whittle-seq32768.json),
[`…-dense-seq32768.json`](2026-08-25-whittle-ar-fa-actual-tiles-r9700-dense-seq32768.json).

Whittle 32k decode matches the 2048 / protocol-lite 43.5 band (**+47%** vs
the 29.6 cell). Dense 32k stayed in the 36.6–36.8 band (no regression).

Eyeball: same quality shape as 2026-08-24 — coherent code/reason/prose/instruct;
Whittle factual still closes on axis tilt; dense factual still expanding at
the cap.

## Reading

The 32k serve tax was AR hipGraph FA tile-grid over-launch, not idle expert
weights and not KV bytes/token. Do not start the expert pager on this 32 GB
box from either this row or the 2026-08-24 battery. Step 4 of
[`docs/plans/2026-08-24-r9700-whittle-bandwidth.md`](../plans/2026-08-24-r9700-whittle-bandwidth.md)
now shows the FFN sparsity win on the harness default 32k window. Not DFlash 2.
Not admission.
