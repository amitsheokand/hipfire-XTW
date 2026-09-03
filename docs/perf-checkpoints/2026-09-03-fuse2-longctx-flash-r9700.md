# Long-ctx decode profile + flash verdict (Fuse-2, gfx1201)

**Date:** 2026-09-03. **Method:** rocprofv3 A/B (5 vs 12 steps) on the
example path at ctx 8192 (primed via `HIPFIRE_PREFILL_MAX_BATCH=384`
chunked prefill; unchunked 8k prefill fails JIT on the bt12 tile —
separate production path chunks, so example-only gap).

## Growth ctx133 → ctx8192 (per decode step)

| kernel | @133 | @8k | Δ | note |
|---|---|---|---|---|
| attention_flash_q8_0_tile | 128 µs | 524 µs | **+396** | 8 calls, 65.5 µs each |
| attention_flash_q8_0_reduce | 19 µs | 177 µs | **+158** | 8 calls, 21.8 µs each |
| everything else (MoE/attn/head) | — | — | ~flat | GEMVs are ctx-independent |
| **step total** | 8.19 ms | 9.36 ms | **+1.17 ms** | flash = ~half the growth |

Grids (rocprof reports threads; true blocks): tile [16 heads × 65 tiles],
one wave32/block, LDS 1536 B (tile 128, head_dim 256). All 64 tiles live
at 8k (graph-safe max_tiles grid).

## Flash analysis (why, and why not attacked)

- Tile pass: ~17.8 MB unique Q8 K/V per call in 65 µs (~260 GB/s eff).
  Memory pattern is sound (coalesced Q8 blocks, GQA 4× L2 sharing); the
  gap to peak is L2-latency/occupancy (1024 single-wave blocks), not
  traffic shape. Exactness-constrained softmax (see reduce header).
- Reduce: 21.8 µs/call for ~1 MB combine — partials [heads, tiles, 258]
  get evicted from L2 by the 66 MB tile streaming, then re-read from DRAM
  with 1032 B-stride row chaos (~50 GB/s eff). Structural to the
  split tile/reduce design; a fused persistent design risks spin-wait
  deadlocks under hipGraph capture. Best case ~+1%.
- `HIPFIRE_Q8_FLASH_TILE` sweep (64/128/256): **no effect** under stable
  conditions at either ctx133 (bench 113.4–113.8, neutral) or ctx8k.
  Bigger tiles halve partials but slow the tile pass by as much; net zero.

## Methodology warning (please read before profiling here)

Example-path wall time is ENVIRONMENTALLY UNSTABLE (9.5 → 16.1 ms/step
across back-to-back identical runs — thermal/power throttle suspected;
first-cold-run-fast pattern). Sub-10% A/B on wall is meaningless here.
The bench path (daemon, stdev ~0.1) and rocprofv3 device timestamps are
 trustworthy; example wall is not. All tok/s claims in this line use bench.

## Verdict

Flash is the only ctx-growing component and is correctly shaped;
attacking it is high-risk for +1–3% combined. Not pursued. At 32k ctx
the flash term roughly quadruples again (~+2 ms) — revisit only with a
long-ctx production need, via the reduce layout, not the tile pass.
