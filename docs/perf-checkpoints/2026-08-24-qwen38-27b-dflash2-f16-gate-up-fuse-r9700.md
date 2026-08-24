# 2026-08-24 — DFlash 2 F16 fused gate+up REJECTED (R9700)

**Lifecycle: `historical`.** Evidence under the exact fixture and method below.
Not a current default, not an automatic baseline, not an admission decision.
**Rejected** — do not re-land as a DFlash 2 default.

## Why this run exists

After [`2026-08-24-qwen38-27b-dflash2-f16-mb1-r9700.md`](2026-08-24-qwen38-27b-dflash2-f16-mb1-r9700.md)
(prose nograph 47.1 tok/s), `HIPFIRE_DRAFT_GEMM_DUMP` still named two F16
SwiGLU GEMMs (`M=17408 K=5120`, 517 µs, 347 GB/s) as ~30% of dumped draft
GEMM time. Hypothesis: one gfx12 NB=1 WG computes gate and up, sharing the
X load (`gemm_f16_gate_up_wmma_mb1_gfx12`). Channel vs two sequential MB1
calls was bit-exact (max_abs 0).

This is the same fusion class that was already a no-op on HFQ4 draft FFN
(comment in `dflash.rs` draft_forward, 2026-04-21). Measured here on F16
DFlash 2 / gfx1201 anyway so the miss is dated.

## Fixture

Matched to the MB1 row: Radeon AI PRO R9700, gfx1201, HIP 7.2,
`HIP_VISIBLE_DEVICES=0`, `nix develop`, `HIPFIRE_DPM_WARMUP_SECS=10`, GPU
lock, `--runs 3 --warmups 3 --max-tokens 128 --backend noslots --workload
stateless --kv-mode q8 -j`, `HIPFIRE_VERIFY_GRAPH=0`,
`HIPFIRE_DFLASH_ADAPTIVE_B=0`. Trunk md5 `129909ad0fed21dcf72b5b9225e85604`.
Draft md5 `cdf5dd280cb8a33455579b7e82ef0a58`. Relativity prompt md5
`d94d3115a3001f08a654d91461d6bdc4`.

| | |
|---|---|
| hipfire md5 | `b8e5ac0b9204a7aa1b98840edae784f3` |
| daemon md5 | `fe35036e70c1738a6b7b823e3fbeb2aa` |

## Result

| arm | decode | τ | windows |
|---|---:|---:|---:|
| MB1 sequential gate/up | 47.1 | 2.43 | 37 |
| **fused gate+up (this row)** | **41.5** | **2.43** | 37 |

JSON: [`…-f16-gate-up-fuse-r9700.json`](2026-08-24-qwen38-27b-dflash2-f16-gate-up-fuse-r9700.json).

Samples: decode 41.6 / 41.5 / 41.5. τ and window count unchanged (not a
selector/numerics bug). Dump confirmed the two `17408×5120` calls disappeared.

Vs MB1: **−11.9%** decode. Reject. Dispatch reverted; kernel not kept.

## Reading

Sharing X does not pay when each projection is already near half-roofline and
the fused K-loop streams **both** 178 MB weights. Extra accumulators likely
cut occupancy / issue rate enough to lose more than one launch. Do not retry
fused F16 QKV/KV on this path for the same reason. Skinny K/V (57 GB/s) is
occupancy-on-64-waves, which fusion also does not increase. Next campaign
step is not another DFlash 2 FFN microkernel. Not pager. Not Whittle.
