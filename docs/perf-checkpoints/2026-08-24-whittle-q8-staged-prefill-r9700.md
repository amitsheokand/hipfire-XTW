# 2026-08-24 — Whittle Q8 down staged-r16 prefill batch on R9700 (protocol-lite)

**Lifecycle: `historical`.** Evidence under the exact fixture and method below.
Not a current default, not an automatic baseline, not an admission decision.
Whittle remains a research-preview SKU.

## Why this run exists

Decode staging in [`2026-08-24-whittle-q8-staged-r16-r9700.md`](2026-08-24-whittle-q8-staged-r16-r9700.md)
gated `batch==1`, so Path 1 prefill stayed on the indexed 32-thread Q8 down.
`HIPFIRE_PROFILE=1` `bench_qwen35_mq4 --prefill 128` named that kernel as
**63.4% of GPU** (8203 µs/call, 247.8 GiB/s). One lever: `grid.z = N` on the
same staged CTA (still not grouped-WMMA Q8, still not a larger DMMV).

## Fixture

| | |
|---|---|
| host | Radeon AI PRO R9700, gfx1201, HIP 7.2, `HIP_VISIBLE_DEVICES=0` |
| hipfire md5 | `09591b799b3c30711d81c65ec10f41cf` |
| daemon md5 | `07159020cef5352b46da2d358a2b2deb` |
| prompt | default `hipfire bench` string `Explain the theory of general relativity in simple terms.` (no trailing newline) |
| prompt md5 | `d94d3115a3001f08a654d91461d6bdc4` |
| harness | `hipfire bench --runs 3 --warmups 3 --max-tokens 128 --spec off --backend noslots --workload stateless --kv-mode q8 -j` |
| env | `nix develop`, `HIPFIRE_DPM_WARMUP_SECS=10`, GPU lock, `HIPFIRE_DFLASH_DRAFT` unset |
| prior row | staged-r16 decode-only, daemon `0e7581d1…`: decode 43.3, prefill 132.8, TTFT 180.8 ms |

### Artifact

`~/.hipfire/models/qwen38-whittle-27b-a17.8.mq4` md5 `bb8dd08b909b3e5249db1d3f9e477953`.

## Result

JSON: [`2026-08-24-whittle-q8-staged-prefill-r9700.json`](2026-08-24-whittle-q8-staged-prefill-r9700.json).

| metric (median) | decode-only staged | + prefill `grid.z=N` | delta |
|---|---:|---:|---:|
| decode tok/s | 43.3 | 43.4 | ~0 |
| **prefill tok/s** | **132.8** | **203.9** | **+53.5 %** |
| wall tok/s | 40.8 | 41.8 | +2.5 % |
| **TTFT ms** | **180.8** | **117.7** | **−34.9 %** |

Samples (this row): decode 43.5 / 43.4 / 43.4. Prefill 203.6 / 204.4 / 203.9.

N=128 internal profile after the change: Q8 down 525 → 172 ms, 248 → 758 GiB/s
(analytical bytes count per-token weights; cache reuse inflates GiB/s).
Serialized GPU 828 → 493 ms. Kernel prefill 154.5 → 259.7 tok/s.

Serve smoke (greedy, thinking off, spec off, kv q8, max=64, same two prompts):
attractor=0, prefill 222.8 / 220.8 tok/s, decode 43.3 / 43.6. Landmark still
"Louie Tower".

## Reading

Prefill Path 1's leftover was the same tiny-K Q8 down, not missing WMMA on
gate_up (already ~1100 GiB/s batched). Dense MQ4 protocol-lite prefill is
still ~401 tok/s — Whittle is **~2×** behind, not 3×. Next Path 1 slices are
batched MQ4 gate_up (~25%) and shared sigmoid GEMV (~20%). Grouped-WMMA Q8
is a different lever. Not product. Not pager evidence.
