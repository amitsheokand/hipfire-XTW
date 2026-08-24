# 2026-08-24 — DFlash 2 adaptive-B trip-wire (R9700)

**Lifecycle: `historical`.** Evidence under the exact fixture and method below.
Not a current default, not an automatic baseline, not an admission decision.

## Why this run exists

After GPU selector top-k
([`2026-08-24-qwen38-27b-dflash2-gpu-topk-r9700.md`](2026-08-24-qwen38-27b-dflash2-gpu-topk-r9700.md)),
prose DFlash 2 was still under MTP (44.9 vs 47.8) at τ=2.43 / B=8. The
daemon had accepted `dflash_adaptive_b` since 0.1.7-alpha and discarded
it (`let _adaptive_b`). This row honors the τ-window trip-wire (mean
accept < 2.5 → B = trained/2) on the serve path.

## Fixture

Same host / prompts / harness as the GPU top-k row. hipfire
`4dcaa50692db7076636041631ce42d9b`, daemon
`75388b52ebb701a52c6ba558281aba6a`. Adaptive-B **on** (then the schema
default; since recorded as a loss, default is now **off**). Prose:
`HIPFIRE_VERIFY_GRAPH=0`. Code: graphs on.

## Result

| arm | decode | τ | windows | note |
|---|---:|---:|---:|---|
| prose nograph, B=8 (prior cell) | 44.9 | 2.43 | 37 | GPU top-k |
| prose nograph, trip-wire **8→4** | **38.5** | **1.89** | 44 | `[dflash] adaptive-B 8 → 4 (mean accept 2.17)` |
| code graphs-on, prior | 101.8 | 6.94 | 16 | GPU top-k |
| code graphs-on, adaptive on | 101.5 | 6.94 | 16 | did not shrink |

JSON: [`…-prose-nograph-r9700.json`](2026-08-24-qwen38-27b-dflash2-adaptive-b-prose-nograph-r9700.json),
[`…-code-r9700.json`](2026-08-24-qwen38-27b-dflash2-adaptive-b-code-r9700.json).

Prose samples decode 38.6 / 38.5 / 38.5. Code 102.2 / 101.5 / 101.3.

## Reading

**Rejected for DFlash 2.** Shrinking the trained B=8 window to 4 cuts
acceptance, not just per-cycle work. τ 2.43→1.89, decode −14% vs the
matched nograph cell. Code is unchanged because mean accept stayed
above 2.5. The leftover vs MTP on prose is still draft+verify at B=8,
not an oversized block. `speculation.dflash_adaptive_b` defaults
**false**; `HIPFIRE_DFLASH_ADAPTIVE_B=1` remains an opt-in for DFlash 1
(trained 16). Not pager. Not Whittle.
