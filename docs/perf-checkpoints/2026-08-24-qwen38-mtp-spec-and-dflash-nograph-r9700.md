# 2026-08-24 — `--spec mtp` without env, and DFlash 2 verify-graph off (R9700)

**Lifecycle: `historical`.** Evidence under the exact fixture and method below.
Not a current default, not an automatic baseline, not an admission decision.

## Why this run exists

Two follow-ups after
[`2026-08-24-qwen38-27b-dflash2-vs-mtp-ar-r9700.md`](2026-08-24-qwen38-27b-dflash2-vs-mtp-ar-r9700.md):

1. `--spec mtp` used to load the sidecar and still route AR unless
   `HIPFIRE_QWEN_MTP=1`. Generate now treats `mtp_mode=on` as the opt-in.
2. DFlash 2 prose was τ=2.43 at 37.2 tok/s with default verify-graph. Cheap
   attribution: same cell with `HIPFIRE_VERIFY_GRAPH=0`.

## Fixture

Same host / prompt / harness as the prose matrix (default relativity md5
`d94d3115a3001f08a654d91461d6bdc4`, 3×128, noslots, kv q8). hipfire still
`d8a4677d382aa36cb8def0ffbe169061`. daemon **this row**
`661c230ab8132114b4b155f0b8b22f43` (MTP route fix).

## Result

| arm | decode | τ | note |
|---|---:|---:|---|
| `--spec mtp` **no** `HIPFIRE_QWEN_MTP` | **47.8** | **2.40** | `mtp_windows=53`; matches the earlier env-opt-in 47.7 |
| DFlash 2, `HIPFIRE_VERIFY_GRAPH=0` | 40.9 | 2.43 | vs 37.2 with graphs on; no `[verify-graph]` lines |

JSON: [`…-mtp-spec-no-env-r9700.json`](2026-08-24-qwen38-27b-mtp-spec-no-env-r9700.json),
[`…-dflash2-verify-graph-off-r9700.json`](2026-08-24-qwen38-27b-dflash2-verify-graph-off-r9700.json).

## Reading

`--spec mtp` is now an MTP measurement. Verify-graph was a real but small
slice on this prose cell (+10% decode, still far from MTP and far from
τ×AR). Remaining DFlash 2 prose overhead is draft+verify work, not graph
capture. `HIPFIRE_QWEN_MTP=0` still opts out.
