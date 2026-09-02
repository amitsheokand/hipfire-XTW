# Fuse-2 MoE (R9700) bring-up baseline — gfx1201

**Date:** 2026-09-03 UTC
**Lifecycle:** `historical`
**Authority:** Dated, fixture-bound **Measured** evidence only.
**Disposition:** Phase 1–3 completion record for the Fuse-2 MoE bring-up task
list (cli-controls, moe-config, gpu-topk, prefill-pbs, gemv-k2). Phase 4
validation evidence. Not a product baseline or admission decision.

## Fixture

| field | value |
|---|---|
| GPU | gfx1201 (34.2 GB VRAM, HIP 7.2) |
| Branch | `R9700` |
| Artifact | `/tmp/fuse2.mq4` (arch 6 `qwen35moe`, retranscoded 2026-09-03 from `/tmp/Fuse-2-MoE-BF16.gguf` with Fuse headers: `router_activation=sqrtsoftplus`, `moe_norm_scale=0.018421`, `moe_norm_skip_rmsnorm=true`) |
| Model | 2560d 32L, 8 experts, k=2, uniform MQ4V2 routed experts, Q8 router, F16 scalar gate, MQ4V2/MQ6 shared expert, q8 KV |
| Commits | `74a63cc3` (raw controls), `79391ad7` (moe-config), `f38011c3` + `2ae7ee56` (compiler device-lib + gpu-topk), `19f2d689` (prefill-pbs), `a22edc65` (gemv-k2) |
| Env | `HIPFIRE_ROCM_DEVICE_LIB_PATH=<nix rocm-device-libs>/amdgcn/bitcode` (JIT needs it on split installs); no `HIPFIRE_FUSE_*` behavior vars set |

## Phase deltas (same artifact family, greedy)

| metric | before (per-token prefill, CPU top-k) | after (PBS + indexed k=2) |
|---|---|---|
| prefill, ~200-token prompt | ~50–60 tok/s (33 s / ~1800 tok) | ~500–700 tok/s (2.7 s / ~1800 tok) |
| prefill, bench matrix | — | pp128 2777 / pp512 3105 / pp2048 3058 tok/s |
| decode, bench standard | — | median 71.65 tok/s |
| decode, bench matrix tg64 | — | 76.6 / 76.8 / 75.8 tok/s @ ctx 128/512/2048 (flat — KV working) |
| decode, `run --json` | 66.1 tok/s (CPU fallback) | 73.6 tok/s (indexed) |
| D2H syncs / decode token | 24 (one [n_exp] download per live layer) | 0 on the indexed path |
| ttft (short prompt) | — | 19.2 ms |

## Correctness evidence

- `hipfire run --raw "2 + 2 ="` → ` 4`; docstring completions (`is_even` → `return n % 2`) correct, no env vars.
- GPU top-2 routing bit-identical to host path (`HIPFIRE_MOE_K2_PARITY=1`: zero mismatches; identical indices/weights).
- PBS-vs-per-token prefill: layer outputs track to 0.027 max-abs at depth 32 (smooth growth, no explosions); short answers identical; 50-token shared prefix on a 64-token code probe.
- Indexed-vs-fallback decode MoE outputs agree to 1 ULP (kernel-order noise); greedy flips only inside degenerate repetition loops.
- `hipfire bench` standard + `--matrix` complete without validation errors.
- serve_harness battery (10 turns, greedy, thinking off): all turns complete, think 0, avg prefill 1075 / decode 73.7 tok/s, transcript at `/tmp/fuse_harness_out.json`.

## Known limitations (not regressions — framing/model scope)

- Chat-role framing makes Fuse emit meta-chatter ("The user is asking for…") and unclosed `<think>` on Q/A prompts; completion-style prompts via `--raw` are direct and correct. Fuse needs its native framing (raw or its real chat template) — the battery above measures this mismatch, not numerics.
- Prefill drift (~0.03 max-abs at depth) can flip near-tie greedy decisions late in degenerate loops. Same ULP-noise class the repo accepts between kernel variants.
- AR hipGraph stays disabled for k=2 (`is_standard_k8` guard) although the indexed path is D2H-free; re-enable needs the wrapper's legacy-stream memset made capture-safe.
- Fresh-kernel JIT is broken in this nix shell without `HIPFIRE_ROCM_DEVICE_LIB_PATH` (split clr package carries no `amdgcn/bitcode`); `fix(compiler)` passes `--rocm-device-lib-path` when discoverable.

## Follow-ups

Phase-4 follow-up candidates: k=2 AR-graph re-enable, prefill-drift reduction (batched MQ6 shared arms), Fuse chat-template framing, `serve` HTTP raw passthrough.
