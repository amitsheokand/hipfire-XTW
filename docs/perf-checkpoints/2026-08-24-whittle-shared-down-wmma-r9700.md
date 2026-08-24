# 2026-08-24 — Whittle shared-expert MQ4 down residual WMMA on R9700 (protocol-lite)

**Lifecycle: `historical`.** Evidence under the exact fixture and method below.
Not a current default, not an automatic baseline, not an admission decision.
Whittle remains a research-preview SKU.

## Why this run exists

After [`2026-08-24-whittle-q8-staged-prefill-r9700.md`](2026-08-24-whittle-q8-staged-prefill-r9700.md),
N=128 Path 1 still spent **~20% GPU** on
`gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched` (1537 µs/call, ~1085 GiB/s)
while sibling `gemm_hfq4g256_residual_wmma_gfx12_bt8` was 17 ms. Shared
**gate_up** already uses fused WMMA. Shared **down** is M=K=5120, N=prefill —
a WMMA shape. One lever: the Q8 shared-down pattern (overwrite-shaped residual
GEMM into the free `down_expanded` prefix, then `sigmoid_scaled_residual_add`).
`n==1` keeps the fused GEMV. Not grouped-WMMA Q8, not a larger DMMV.

## Fixture

| | |
|---|---|
| host | Radeon AI PRO R9700, gfx1201, HIP 7.2, `HIP_VISIBLE_DEVICES=0` |
| hipfire md5 | `a0c3ea8fc8085f06f9b4c53b3c904b4b` |
| daemon md5 | `ca0089115ef118956f46f03c4dc53d1c` |
| prompt | default `hipfire bench` string `Explain the theory of general relativity in simple terms.` (no trailing newline) |
| prompt md5 | `d94d3115a3001f08a654d91461d6bdc4` |
| harness | `hipfire bench --runs 3 --warmups 3 --max-tokens 128 --spec off --backend noslots --workload stateless --kv-mode q8 -j` |
| env | `nix develop`, `HIPFIRE_DPM_WARMUP_SECS=10`, GPU lock, `HIPFIRE_DFLASH_DRAFT` unset |
| prior row | staged Q8 `grid.z=N`, daemon `07159020…`: decode 43.4, prefill 203.9, TTFT 117.7 ms |

### Artifact

`~/.hipfire/models/qwen38-whittle-27b-a17.8.mq4` md5 `bb8dd08b909b3e5249db1d3f9e477953`.

## Result

JSON: [`2026-08-24-whittle-shared-down-wmma-r9700.json`](2026-08-24-whittle-shared-down-wmma-r9700.json).

| metric (median) | Q8 `grid.z=N` | + shared-down WMMA | delta |
|---|---:|---:|---:|
| decode tok/s | 43.4 | 43.5 | ~0 |
| **prefill tok/s** | **203.9** | **241.4** | **+18.4 %** |
| wall tok/s | 41.8 | 42.0 | +0.5 % |
| **TTFT ms** | **117.7** | **99.4** | **−15.5 %** |

Samples (this row): decode 43.5 / 43.5 / 43.4. Prefill 241.5 / 241.4 / 241.3.

N=128 internal profile (`HIPFIRE_PROFILE=1` `bench_qwen35_mq4`, graphs off)
after the change: fused sigmoid GEMV gone. Residual WMMA 64→128 launches
(17.4 → 34.5 ms). Serialized GPU 493 → 398 ms. Kernel prefill 259.7 → 321.4
tok/s. Remaining top slices: staged Q8 down 42.1% (776 GiB/s), indexed MQ4
gate_up 29.7% (1104 GiB/s).

Serve smoke (greedy, thinking off, spec off, kv q8, max=64, same two prompts):
attractor=0, prefill 268.5 / 264.4 tok/s, decode 43.3 / 43.6. Landmark still
"Louie Tower".

## Reading

Shared down was a WMMA-shaped leftover, not HBM-saturated GEMV. Dense MQ4
protocol-lite prefill is still ~401 tok/s — Whittle is **~1.7×** behind, not
2×. Indexed gate_up is already ~1100 GiB/s; the leftover with headroom is
routed Q8 down (~42% GPU, 776 GiB/s). Grouped-WMMA Q8 is a different lever.
Not product. Not pager evidence.
