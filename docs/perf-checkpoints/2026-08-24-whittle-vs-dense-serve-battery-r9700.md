# 2026-08-24 — Whittle vs dense parent serve battery (R9700)

**Lifecycle: `historical`.** Evidence under the exact fixture and method below.
Not a current default, not an automatic baseline, not an admission decision.
Whittle remains a research-preview SKU.

## Why this run exists

Plan step 4 asks for serve quality vs the dense parent on the same prompts
before treating Whittle decode 43.5 as a product-path win.
[`2026-08-24-whittle-q8-grouped-wmma-r9700.md`](2026-08-24-whittle-q8-grouped-wmma-r9700.md)
is `hipfire bench` protocol-lite. This row is `serve_harness.py --mode battery`
(user-facing spawn).

## Fixture

Radeon AI PRO R9700, gfx1201, HIP 7.2, `HIP_VISIBLE_DEVICES=0`, `nix develop`,
GPU lock. Greedy, thinking off, speculation off, kv q8, max_tokens 64.
Built-in genre battery (code / reason / factual / prose / instruct).
hipfire/daemon md5 `b8e5ac0b…` / `fe35036e…` (MB1 tree; this path is AR, not
DFlash 2).

| artifact | md5 |
|---|---|
| `qwen38-whittle-27b-a17.8.mq4` | `bb8dd08b909b3e5249db1d3f9e477953` |
| `qwen38-27b.mq4` | `129909ad0fed21dcf72b5b9225e85604` |

## Result

`!RUNAWAY` here is `finish=length` at the 64-token cap, not an attractor
loop. Attractor=0 and empty=0 on every turn for both SKUs.

| arm | max_seq | avg decode | avg prefill | attractor | factual finish |
|---|---:|---:|---:|---:|---|
| Whittle | 32768 (harness default) | **29.6** | 412 | 0 | stop @ 50 tok |
| Dense parent | 32768 | **36.7** | 466 | 0 | length @ 64 |
| Whittle | **2048** | **43.5** | 504 | 0 | stop @ 50 tok |

JSON: [`…-whittle-seq32768.json`](2026-08-24-whittle-vs-dense-serve-battery-r9700-whittle-seq32768.json),
[`…-dense-seq32768.json`](2026-08-24-whittle-vs-dense-serve-battery-r9700-dense-seq32768.json),
[`…-whittle-seq2048.json`](2026-08-24-whittle-vs-dense-serve-battery-r9700-whittle-seq2048.json).

Whittle seq2048 decode matches protocol-lite 43.5. seq32768 loses **32%**
vs that cell. Dense seq32768 stayed at the protocol-lite 36.6 band.

Eyeball: both SKUs open coherent code/reason/prose/instruct; Whittle
factual closed in three axis-tilt sentences; dense factual still expanding
at the cap. Not a lobotomy.

## Reading

The FFN sparsity win is real on the bench / short-ctx serve path and
**does not show up** on the harness default 32k context. That tax is KV
footprint, not idle expert weights — do not start the expert pager on this
32 GB box from this row. Pin `--max-seq` (or config) when comparing Whittle
to dense on serve. Not DFlash 2. Not admission.
