# 2026-08-24 — Whittle MoE vs dense MQ4 AR on R9700 (protocol-lite)

**Lifecycle: `historical`.** Evidence under the exact fixture and method below.
Not a current default, not an automatic baseline, not an admission decision.
Whittle remains a research-preview SKU.

## Why this run exists

Campaign step 4 of [`docs/plans/2026-08-24-r9700-whittle-bandwidth.md`](../plans/2026-08-24-r9700-whittle-bandwidth.md):
after k=16 GPU decode and batched MQ4/Q8 prefill (`506681c4`), measure AR
bytes/token proxy (tok/s) against the dense MQ4 floor on the **same binary,
same default bench prompt, same protocol**. Serve smokes already showed
language + attractor 0; this row is speed only.

## Fixture

| | |
|---|---|
| host | Radeon AI PRO R9700, gfx1201, HIP 7.2, `HIP_VISIBLE_DEVICES=0` |
| commit | `506681c4` `feat: GPU k=16 MoE decode and batched MQ4/Q8 prefill` |
| hipfire md5 | `7e14f9c307d519ba7b235ec66daacaf3` |
| daemon md5 | `2f54c9fb112ebc73e0cb01f46390495e` |
| prompt | default `hipfire bench` string `Explain the theory of general relativity in simple terms.` (no trailing newline) |
| prompt md5 | `d94d3115a3001f08a654d91461d6bdc4` |
| harness | `hipfire bench --runs 3 --warmups 3 --max-tokens 128 --spec off --backend noslots --workload stateless --kv-mode q8 -j` |
| env | `nix develop`, `HIPFIRE_DPM_WARMUP_SECS=10`, GPU lock, fresh process per model, `HIPFIRE_DFLASH_DRAFT` unset |
| isolation | dense then Whittle, sequential, one lock hold |

### Artifacts

| arm | path | md5 | loaded arch | VRAM free after load |
|---|---|---|---|---|
| dense | `~/.hipfire/models/qwen38-27b.mq4` | `129909ad0fed21dcf72b5b9225e85604` | `qwen3_5` | 16292 / 32624 MB |
| Whittle | `~/.hipfire/models/qwen38-whittle-27b-a17.8.mq4` | `bb8dd08b909b3e5249db1d3f9e477953` | `qwen3_5_moe` | 14456 / 32624 MB |

Whittle shape: 64 experts / top-16 / moe_intermediate 192 / shared 5120. Routed
`down` is Q8 (K=192); gate_up / attn / shared / router MQ4.

## Result

JSON: `/tmp/qwen38-27b-mq4-matched-ar.json`, `/tmp/qwen38-whittle-mq4-matched-ar.json`.

| metric (median) | dense MQ4 | Whittle MQ4 | delta |
|---|---:|---:|---:|
| **decode tok/s** | **36.6** | **40.1** | **+9.6 %** |
| prefill tok/s | 401.0 | 132.4 | −67.0 % |
| wall tok/s | 36.0 | 38.0 | +5.6 % |
| TTFT ms | 59.9 | 181.2 | +3.02× |

Samples:

| | dense decode | Whittle decode | dense prefill | Whittle prefill |
|---|---|---|---|---|
| r1 | 36.7 | 40.1 | 401.0 | 132.8 |
| r2 | 36.6 | 40.1 | 399.9 | 131.6 |
| r3 | 36.5 | 40.1 | 401.6 | 132.4 |
| stdev | 0.082 | 0.0 | 0.70 | 0.50 |

Whittle decode stdev 0.0 at 0.1 tok/s reporting precision is the same 40.1
seen on the serve smoke (40.2–40.5), not a spec-decode attractor.

## Reading

Theoretical FFN sparsity is 8192/17408 ≈ 47% of dense width (16×192 + shared
5120). Decode only moved **+9.6%**. Indexed kernels already launch
`grid.y = k_top` (one launch per projection, not 16 host GEMVs). Remaining
cost is 16 ranks × tiny K=192 re-reading `x`, plus a full shared expert 5120
every token. A3B ninepath (`hidden<=2048`, `mi==512`, `k==8`) does not admit
this shape.

Prefill Path 1 (indexed batched GEMV, Q8 down has no grouped-WMMA arm) is
**~3×** behind dense MQ4 WMMA (132 vs 401 tok/s). Serve battery on 37–42 token
prompts was 135–140 tok/s once JIT was warm — same band.

Not product. Not pager evidence. Next measured lever is decode activation
re-read (stage-x-once / grouped expert GEMV) **after** a `HIPFIRE_PROFILE`
name of the hot kernel — not pager traces, and not a second speculative
kernel.
