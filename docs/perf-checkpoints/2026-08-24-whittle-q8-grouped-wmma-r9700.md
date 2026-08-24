# 2026-08-24 — Whittle Path 2 Q8 grouped WMMA on R9700 (protocol-lite)

**Lifecycle: `historical`.** Evidence under the exact fixture and method below.
Not a current default, not an automatic baseline, not an admission decision.
Whittle remains a research-preview SKU.

## Why this run exists

After [`2026-08-24-whittle-shared-down-wmma-r9700.md`](2026-08-24-whittle-shared-down-wmma-r9700.md),
N=128 Path 1 leftover was routed Q8 down (`gemv_q8_0_moe_down_k16_staged_r16`,
42% GPU, 776 GiB/s analytical). K=192 is not G256, so HFQ4 grouped-WMMA cannot
run it; `q8_no_grouped` forced the whole Path 2 scatter pipeline off, which
also left MQ4 gate_up on the indexed GEMV (~30% GPU). One lever: gfx12
`gemm_q8_0_moe_grouped_wmma_gfx12` (same scatter contract as the HFQ4 grouped
sister; Q8_0 block math from `gemm_q8_0_wmma_gfx12`). gfx11 stays Path 1.
Q8 gate_up still has no grouped arm. Opt out: `HIPFIRE_MOE_GROUPED_GEMM=0`.

## Fixture

| | |
|---|---|
| host | Radeon AI PRO R9700, gfx1201, HIP 7.2, `HIP_VISIBLE_DEVICES=0` |
| hipfire md5 | `669785021db4ec4c45d470f0c896b4ab` |
| daemon md5 | `51ef5fe234c0ae52c2f0c26d4b3d4127` |
| prompt | default `hipfire bench` string `Explain the theory of general relativity in simple terms.` (no trailing newline) |
| prompt md5 | `d94d3115a3001f08a654d91461d6bdc4` |
| harness | `hipfire bench --runs 3 --warmups 3 --max-tokens 128 --spec off --backend noslots --workload stateless --kv-mode q8 -j` |
| env | `nix develop`, `HIPFIRE_DPM_WARMUP_SECS=10`, GPU lock, `HIPFIRE_DFLASH_DRAFT` unset |
| prior row | shared-down WMMA, daemon `ca008911…`: decode 43.5, prefill 241.4, TTFT 99.4 ms |

### Artifact

`~/.hipfire/models/qwen38-whittle-27b-a17.8.mq4` md5 `bb8dd08b909b3e5249db1d3f9e477953`.

## Result

JSON: [`2026-08-24-whittle-q8-grouped-wmma-r9700.json`](2026-08-24-whittle-q8-grouped-wmma-r9700.json).

| metric (median) | shared-down WMMA | + Path 2 Q8 grouped | delta |
|---|---:|---:|---:|
| decode tok/s | 43.5 | 43.5 | 0 |
| **prefill tok/s** | **241.4** | **407.6** | **+68.8 %** |
| wall tok/s | 42.0 | 42.6 | +1.4 % |
| **TTFT ms** | **99.4** | **58.9** | **−40.7 %** |

Samples (this row): decode 43.5 / 43.5 / 43.5. Prefill 406.4 / 407.9 / 407.6.

Dense MQ4 protocol-lite prefill on this box was **401 tok/s / 60 ms TTFT**.
Whittle prefill now sits in that band.

N=128 internal profile (`HIPFIRE_PROFILE=1` `bench_qwen35_mq4`, graphs off)
after the change: Path 2 `path2=true`. Staged Q8 GEMV gone. Q8 grouped WMMA
64× 263 µs (16.9 ms, was 167.7 ms). MQ4 gate_up grouped 64× 229 µs (14.6 ms,
was indexed 118 ms). Serialized GPU 398 → 141 ms. Kernel prefill 321 → 910
tok/s.

Serve smoke (greedy, thinking off, spec off, kv q8, max=64, same two prompts):
attractor=0, prefill 523.7 / 510.0 tok/s, decode 43.3 / 43.7. Landmark still
"Louie Tower". Greedy tokens are not byte-identical with Path 1 (WMMA ULP);
Seine still appears in the factual turn.

## Reading

The leftover was small-K compute, not HBM and not pager-shaped. Lifting
`q8_no_grouped` on gfx12 also moved MQ4 gate_up onto the existing grouped
WMMA sister — that was blocked by the same flag, not a second kernel.
Decode is unchanged (Path 1 indexed + staged Q8). gfx11 / Q8 gate_up stay
Path 1. Not product. Not pager evidence.
