---
title: DFlash 2 engine port landed on r9700-fp8-adaptive
date: 2026-08-24
tags: [dflash2, r9700, whittle]
---

DFlash 2 is no longer a silent DFlash 1 load. `DflashConfig::from_hfq` tags `DFlash2DraftModel` / conv tensors; `DflashWeights::load` requires per-layer `attention_conv`/`mlp_conv` plus `candidate_selector` or panics. HIP grouped two-tap conv (`kernels/src/dflash2_grouped_conv.hip`) wraps attn and MLP in `draft_forward`. Greedy T=0 selector walks top-16 lattice from the seed token (host codebook + top-k). Do **not** set `HIPFIRE_DFLASH_DRAFT` until a serve_harness pass on dense MQ4.

Packed draft: `~/.hipfire/models/qwen38-27b-dflash2.hfq` (F16, 81 tensors, block 8). Existing packed file has no `variant` key; detection uses `config.architectures`.

2026-08-24 serve_harness on dense `qwen38-27b.mq4` (thinking off, greedy, kv q8, max=64, same Paris/`merge_sort` prompts): DFlash 2 fired (not silent DFlash 1). attractor=0. τ=2.94 factual / 6.88 code. First-turn decode 16.5 tok/s is JIT; second 93.5. Not a bench claim.

Protocol-lite on the default relativity prompt (md5 `d94d3115a3001f08a654d91461d6bdc4`):
AR 36.6; MTP live (`HIPFIRE_QWEN_MTP=1`) 47.7 τ=2.40; DFlash 2 37.2 τ=2.43.
`--spec mtp` without the env is silent AR on daemon `51ef5fe2…`. After
`mtp_mode=on` generate opt-in (daemon `661c230a…`), `--spec mtp` alone is
47.8 τ=2.40 on the same prompt. DFlash 2 `HIPFIRE_VERIFY_GRAPH=0` is 40.9
vs 37.2 graphs-on; τ stays 2.43. Checkpoint
`docs/perf-checkpoints/2026-08-24-qwen38-mtp-spec-and-dflash-nograph-r9700.md`.

Code prompt (md5 `51a6c7360e00a7853521a6575975b8e3`, same protocol): AR 36.6;
MTP 75.7 τ=3.74; DFlash 2 84.7 τ=6.94. Genre-conditional: DFlash 2 converts
τ on code, not on the relativity cell. Checkpoint
`docs/perf-checkpoints/2026-08-24-qwen38-27b-dflash2-code-r9700.md`.

Whittle v2+v2.1 download finished 2026-08-24 (~53 GB) at `~/.hipfire/hf-cache/Qwen3.8-Whittle-MoE-27B-A17.8B`. v2.1 is a PEFT LoRA (`r=128`, `alpha=256`) plus `modules_to_save` = 64 routers and late-layer shared experts — merge before encode. Shape: 64 experts / top-16 / moe_intermediate 192 / shared 5120. First encode risk: routed `down` K=192 vs group-256.
