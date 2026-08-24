# 2026-08-24 — Whittle Q8 down staged-r16 vs indexed on R9700 (protocol-lite)

**Lifecycle: `historical`.** Evidence under the exact fixture and method below.
Not a current default, not an automatic baseline, not an admission decision.
Whittle remains a research-preview SKU.

## Why this run exists

Campaign leftover after [`2026-08-24-whittle-vs-dense-mq4-r9700.md`](2026-08-24-whittle-vs-dense-mq4-r9700.md):
decode only +9.6% vs dense despite ~47% FFN width. `HIPFIRE_PROFILE_DECODE=1`
on `bench_qwen35_mq4` (graphs off, kv q8, 32 gen) named the routed Q8 down as
the MoE kernel that was **not** HBM-saturated:

| kernel | launches | µs/call | % GPU | GiB/s |
|---|---:|---:|---:|---:|
| `fused_qkvza_hfq4g256` | 3584 | 72 | 26.9 | 454.9 |
| `gemv_q8_0_moe_down_residual_scaled_k8_indexed` | 2048 | 72 | 15.4 | 219.8 |
| `gemv_hfq4g256_moe_gate_up_k8_indexed` | 2048 | 37 | 7.8 | 434.6 |

One lever: ninepath-style stage of the 16×192 silu activation, 256-thread CTA,
16 rows × 16 ranks, atomic-free fold. Not a larger dense DMMV workgroup
(llama.cpp R9700 thread: that was slower and reverted). Shape-gated
`k==192 && n_ranks==16 && batch==1 && m%16==0`. Opt out:
`HIPFIRE_WHITTLE_Q8_STAGED=0`. Prefill `batch>1` stays on the indexed kernel.

## Fixture

| | |
|---|---|
| host | Radeon AI PRO R9700, gfx1201, HIP 7.2, `HIP_VISIBLE_DEVICES=0` |
| hipfire md5 | `9c79e50b1d52bce4ad6e64458cc75d6a` |
| daemon md5 | `0e7581d1d0aecdf39e5b2ec4839a64d5` |
| prompt | default `hipfire bench` string `Explain the theory of general relativity in simple terms.` (no trailing newline) |
| prompt md5 | `d94d3115a3001f08a654d91461d6bdc4` |
| harness | `hipfire bench --runs 3 --warmups 3 --max-tokens 128 --spec off --backend noslots --workload stateless --kv-mode q8 -j` |
| env | `nix develop`, `HIPFIRE_DPM_WARMUP_SECS=10`, GPU lock, `HIPFIRE_DFLASH_DRAFT` unset |
| prior row | Whittle median **40.1** decode in the sibling checkpoint (daemon `2f54c9fb…`) |

### Artifact

`~/.hipfire/models/qwen38-whittle-27b-a17.8.mq4` md5 `bb8dd08b909b3e5249db1d3f9e477953`.

## Result

JSON: [`2026-08-24-whittle-q8-staged-r16-r9700.json`](2026-08-24-whittle-q8-staged-r16-r9700.json).

| metric (median) | indexed Q8 down | staged-r16 | delta |
|---|---:|---:|---:|
| **decode tok/s** | **40.1** | **43.3** | **+8.0 %** |
| prefill tok/s | 132.4 | 132.8 | ~0 |
| wall tok/s | 38.0 | 40.8 | +7.4 % |
| TTFT ms | 181.2 | 180.8 | ~0 |

Samples (staged): decode 43.4 / 43.3 / 43.3. Prefill 133.1 / 132.8 / 132.8.

Kernel timer on the same `bench_qwen35_mq4` fixture after the change:
`gemv_q8_0_moe_down_k16_staged_r16` 2048× 46 µs/call, **343.2 GiB/s**, 10.3% of
GPU (was 72 µs, 219.8 GiB/s, 15.4%). Serialized GPU 963 → 916 ms. Example
wall tok/s stayed 19.8 — that harness D2H-downloads logits every token.

Serve smoke (greedy, thinking off, spec off, kv q8, max=64, same two prompts
as the k=16 smoke): attractor=0, decode 43.3 / 43.5 tok/s. Landmark text is
still "Louie Tower" (model, not this kernel).

## Reading

The 16×192 re-read was real and was not pager-shaped. Staging it recovered
about as much decode as the original Whittle vs dense gap (+8% on top of
+9.6%). Q8 down is still below sibling MQ4 GEMVs (~435 GiB/s) and below
QKVZA (~450). `fused_qkvza_hfq4g256` remains the largest GPU slice (~29%)
and is already bandwidth-saturated — parent attention, not routed FFN.

Prefill unchanged (Path 1, batched indexed Q8). Not product. Not pager
evidence: HBM is not the leftover on the routed down after this lever.
