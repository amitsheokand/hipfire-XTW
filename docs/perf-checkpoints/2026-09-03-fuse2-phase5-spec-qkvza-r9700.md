# Phase 5: speculation verdict + qkvza roofline (Fuse-2, gfx1201)

**Date:** 2026-09-03. Baseline: 113.8 tok/s decode (spec off, greedy).

## Speculation (all modes measured, winner = OFF)

| Mode | Decode | Notes |
|---|---|---|
| off | **113.8 tok/s**, sd 0.1 | stable baseline |
| ngram | 97.3 tok/s, sd 30 | tau 0.0–2.75; draft overhead + low acceptance = net loss, unstable |
| dflash (auto) | 113.4 tok/s | no-op: no compatible draft on disk, runs target-only |
| dflash + qwen38 draft | AR fallback | clean reject: draft `target_layer_ids` contains 33 ≥ 32 Fuse layers |
| mtp | clean load error | no `.mtp` head/sidecar for fuse-2-moe |

On-disk draft assets: `qwen38-27b-dflash-mq4.hfq` (1.2 GB, qwen38-only),
`ornith-1.5-35b-a3b.mtp`, `qwen3.8-27b.mtp`. Nothing Fuse-compatible.
`models.toml` keeps `dflash/mtp/mode = off` for fuse-2-moe (no change).
Future lever is model work (train/bake a Fuse draft or MTP head), not kernels.

## qkvza: DRAM weight-bound, no economic attack

Fuse DeltaNet dims (from mq4 tensor index): in_proj_qkv [8192×2560],
in_proj_z [4096×2560], in_proj_a/b [32×2560] each → 12 352 rows, K=2560
(10 groups). Per call: 16.8 MB weights + 126 MB x re-reads (L1-absorbed:
10 KB x fits L0V 32 KB after first touch). Measured 34.5 µs/call →
weights alone account for ~29.5 µs (85%) at ~487 GB/s effective.

The kernel (`fused_qkvza_mq4g256v2`, already 4-way fused + float4
x-hoist + gfx12 s_prefetch) is at its byte roofline. LDS-staging x would
save only the ~3–5 µs L1-resident portion (~10–15% of 0.72 ms/step ≈
+1% decode best case) for a risky rewrite of a 520-line tuned kernel:
below the bar, not pursued. Same verdict as K2a/K2b.

## Standing conclusion (W0 → K1 → Phase 5)

71.65 → 113.8 tok/s (+59%) via waste removal (W0) + router fusion (K1,
incl. fallthrough fix). All GEMV families at 80–100% of peak GB/s; the
vocab head (1.15 ms, 13%) is exactly at roofline. Remaining gap to A3B
is active-bytes-per-token (Fuse ~51 MB/layer vs A3B ~14 MB/layer), not
kernel efficiency. Next economic levers: a Fuse draft/MTP head
(speculation), or attention-side structural work — both out of scope
for the MoE fusion line.
