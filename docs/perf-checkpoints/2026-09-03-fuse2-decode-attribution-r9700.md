# Fuse-2 MoE decode attribution (rocprofv3, gfx1201)

**Date:** 2026-09-03. **Method:** `rocprofv3 --kernel-trace` on
`profile_qwen35_mq4` (example, no AR-graph capture), two runs
(prefill 32 + warmup 2 + 5 vs 12 profile steps); per-step = (B−A)/7,
prefill = A − 7·step. 32 layers, 24 live-routed. Sums to 8.19 ms
kernel-time/step vs 8.79 ms wall (113.8 tok/s) — 93% closure.

## Per decode step (µs, n calls)

| µs/step | n/step | kernel | note |
|---|---|---|---|
| 1808 | 74 | gemv_mq4g256v2_multirow_r2 | shared gate/up/down (MQ4V2 layers) |
| 1155 | 1 | gemv_q8_0 | vocab head 248320×2560 (~660 MB @ ~575 GB/s — roofline) |
| 908 | 22 | gemv_hfq6g256 | shared expert (MQ6 layers) |
| 724 | 21 | fused_qkvza_mq4g256v2 | DeltaNet attention |
| 551 | 24 | gemv_mq4g256v2_moe_gate_up_k8_indexed_batched | routed gate_up (~460 GB/s) |
| 369 | 28 | gemv_mq4g256v2_residual | shared down (MQ4V2 layers) |
| 334 | 152 | mq_rotate_x | FWHT micro-launches, 2.2 µs each |
| 288 | 24 | gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded | routed down |
| 283 | 49 | rmsnorm_f32 | 5.5 µs each |
| 234 | 7 | fused_qkv_mq4g256v2 | full-attention layers (every 4th) |
| 174 | 24 | gated_delta_net_q8_compact2_b2 | SSM state |
| 171 | 32 | fused_rmsnorm_mq_rotate | |
| 128 | 8 | attention_flash_q8_0_tile | full-attn softmax tiles |
| 113 | 24 | moe_router_q8_sqrtsoftplus_topk2 (K1) | 4.6 µs each |
| 113 | 64 | add_f32 | 1.8 µs each |
| 99 | 4 | gemv_hfq6g256_residual | |
| 44 | 24 | moe_down_combine_k8_batched | 1.8 µs each |
| 44 | 32 | silu_mul_f32 | |
| 42 | 24 | fused_silu_mul_mq_rotate | routed silu+rot |
| 41 | 24 | fused_sigmoid_alpha_gate_f32 | |
| 29 | 24 | scale_f32 | moe_norm_scale |
| — | 0 | moe_topk_renorm_k2 (non-batched) | ABSENT on decode — K1 fallthrough fix verified |

## Budget by subsystem (% of 8.79 ms wall)

- Shared expert (smi=9216: multirow + hfq6 + residual + hfq6_res + silu share): ~3.2 ms (**~37%**)
- Vocab head (single Q8 GEMV): 1.15 ms (**13%**, at roofline — nothing to gain)
- Attention (qkvza + delta + f_qkv + flash + kv/rope/conv/norms): ~1.5 ms (**17%**)
- Routed experts (gate_up + down + K1 + combine + silu_rot + scale): ~1.07 ms (**12%**)
- Micro-launches (rotate_x + adds + rmsnorm + misc ≤6 µs): ~0.7 ms (8%; collapses under AR-graph capture in production — do NOT chase from this trace)

## Conclusions

1. K1 verified in trace (24 calls/step, 4.6 µs) and the old non-batched
   topk is fully gone from decode after the fallthrough fix (+1.4% measured).
2. The remaining A3B gap (113.8 vs 174 tok/s) is **model structure, not
   kernels**: Fuse-2's active footprint per layer (~35 MB shared + ~16 MB
   routed weights) is ~3× A3B's; every GEMV family here runs at 80–100% of
   measured peak GB/s (~460–575). Fusion cannot beat the bytes.
3. K2a is structurally blocked (FWHT needs full-vector sync across the
   row-parallel gate_up grid); K2b (self-combining down) is worth ~+0.5%
   (44 µs combine + 20 KB traffic) — below the bar. Not pursued.
4. Economic next levers are outside MoE fusion: speculation (skip tokens;
   currently off for fuse-2-moe), or attention-side work (qkvza 0.72 ms).
   The in-process `profile::{start,stop}` example path is dormant for
   forward (thread-local never arms in launchers — pre-existing tooling
   quirk); rocprofv3 A/B differencing is the working method (see above).
