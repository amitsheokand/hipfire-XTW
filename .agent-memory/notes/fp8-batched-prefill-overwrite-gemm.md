---
title: gfx1201 FP8E4M3G256 batched prefill via overwrite GEMM
date: 2026-08-23
tags: [gfx1201, r9700, fp8, prefill, gemm]
---

`is_batchable_la(FP8E4M3G256)` is true **only** on `arch == "gfx1201"`
(llama + qwen35 copies). Kernel sources are `.gfx1201.hip`; gfx1200
stays false. Not gated on `fp8_wmma` (HFP4 activation flag).

No fused QKV/QKVZA/gate_up/residual FP8 kernel. Prefill uses
`gpu.fp8_gemm_e4m3_g256` overwrite (three/four times for QKV/QKVZA,
two for gate+up). Residual is GEMM-into-scratch + `add_inplace_f32`
(same pattern as non-WMMA Q8). Pack cache hits after the first GEMM
on the same X pointer.

Every HFQ4 `else` that would have fired has an explicit FP8 arm:
llama QKV/wo/gate_up/down; qwen35 dense LA/FA QKVZA/QKV/wo/gate_up/down;
MoE attention QKVZA/QKV/wo (unrotated input, like Q8 — not MQ rotate).
`batched_gemm_single_weight` covers mixed-format FA QKV.

**Not** in `moe_ffn_batched_admissible` / shared-expert match (no MoE-3D
encoder). Uniform FP8 MoE stays per-token. Mixed FP8-attn + MQ4-MoE FFN
is eligible on gfx1201.

Slot-aware `forward_slots.rs` still admits only uniform Q8_0/MQ4G256;
FP8 errors out rather than taking HFQ4 keys.

Unit tests: `is_batchable_la_fp8_gfx1201_only` (runtime),
`qwen35_is_batchable_la_fp8_gfx1201_only`, dispatch-tests llama.
Not a tok/s or admission claim; no E2E `.hfq` serve yet.

Related: [[fp8-runtime-dtype-dispatch]], [[r9700-fp8-branch-status]],
[[fp8-encoder-cpu]].
