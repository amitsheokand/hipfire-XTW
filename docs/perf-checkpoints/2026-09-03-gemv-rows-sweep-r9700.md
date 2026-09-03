# GEMV R-sweep + scalar-load analysis (gfx1201, Fuse-2 shared MQ4V2)

**Date:** 2026-09-03. Decode tok/s, `hipfire bench` standard, spec off.

## R-sweep (`HIPFIRE_GEMV_ROWS`, shared gate/up/down MQ4V2)

| R | Decode | Note |
|---|---|---|
| 2 (default) | **113.8 tok/s** | optimal; keep |
| 1 | 23.0 tok/s | 5× collapse |
| 4 | 17.4 tok/s | occupancy collapse |

- R=1 collapse cause: the scalar kernel issues 8 scalar x-loads per
  group per thread (2560 loads/row, address-calc heavy) vs multirow's
  once-per-quad hoist — instruction-overhead bound, not traffic. The R=1
  `use_wide` branch was additionally bit-rotted (referenced nonexistent
  symbol `gemv_mq4g256v2_wide` → hip500); fixed to fall through to the
  scalar reference (commit `fix(gemv)`).
- R=4 collapse cause: VGPR/occupancy (matches gfx1010 file notes).
- gfx1201 inherits `_ => 2` default — measurement confirms it is already
  optimal here. No default change.

## Scalar-load promotion (SGPR) — analyzed out

- MQ4V2 headers (8 B/group) are wave-uniform but L1-resident; promoting
  to scalar saves ~3 VGPRs: multirow_r2 104 → ~100, still over the 96
  full-occupancy line. ~0% expected. Not pursued.
- Fused shared gate/up (one x-pass for both projections): x is
  L1-absorbed across sequential waves on a CU (10 KB fits L0V; DRAM
  x-traffic ≈ 1–3 MB/GEMV, not 46 MB). Saves ~+0.2–0.3%. Not pursued.
- Packed-nibble loads are already 128 B-coalesced across the wave (32×4 B
  contiguous). Nothing to widen.

Item closed: the GEMV line is at its practical ceiling on gfx1201.
