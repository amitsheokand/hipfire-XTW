---
title: DFlash 2 engine port landed on r9700-fp8-adaptive
date: 2026-08-24
tags: [dflash2, r9700, whittle]
---

DFlash 2 is no longer a silent DFlash 1 load. `DflashConfig::from_hfq` tags `DFlash2DraftModel` / conv tensors; `DflashWeights::load` requires per-layer `attention_conv`/`mlp_conv` plus `candidate_selector` or panics. HIP grouped two-tap conv (`kernels/src/dflash2_grouped_conv.hip`) wraps attn and MLP in `draft_forward`. Greedy T=0 selector walks top-16 lattice from the seed token (host codebook + top-k). Do **not** set `HIPFIRE_DFLASH_DRAFT` until a serve_harness pass on dense MQ4.

Packed draft: `~/.hipfire/models/qwen38-27b-dflash2.hfq` (F16, 81 tensors, block 8). Existing packed file has no `variant` key; detection uses `config.architectures`.

Whittle v2+v2.1 download finished 2026-08-24 (~53 GB) at `~/.hipfire/hf-cache/Qwen3.8-Whittle-MoE-27B-A17.8B`. v2.1 is a PEFT LoRA (`r=128`, `alpha=256`) plus `modules_to_save` = 64 routers and late-layer shared experts — merge before encode. Shape: 64 experts / top-16 / moe_intermediate 192 / shared 5120. First encode risk: routed `down` K=192 vs group-256.
