# Receipt: T-hipfire-fuse-default-readiness

**Task:** Fuse-2-MoE as default-serve readiness — close HANDOFF divergence,
quality levers, thinking probe. **Date:** 2026-09-10. **Branch:** R9700.
**GPU:** gfx1201 R9700 (single slot; Qwen serve stopped for window, restored after).
**Binaries (rebuilt with fix):** `hipfire a94a660c…`, `daemon 821cd586…`.
**Env:** `HIPFIRE_HIPCC_EXTRA_FLAGS=…rocm-device-libs-22.0.0…/amdgcn/bitcode`,
`HIPFIRE_DFLASH_MODE=off`, draft unset. No push.

## 1. HANDOFF AR-decode divergence: CLOSED (not by the new fix)

Raw 1-token probe (`--raw --spec off --temp 0 --max-tokens 1`,
"P=Explain why the sky is blue in two sentences."):

| spec | max=1 | max=2 |
|---|---|---|
| off | `\n\n` | `\n\nThe` |
| ngram | `\n\n` | `\n\nThe` |
| mtp | `\n\n` | `\n\nThe` |

All three paths agree and match exporter truth (first=271 `\n\n`,
second=760 `The`). A/B with the `run_moe_decode` rotation change stashed
vs applied: byte-identical output. The closer was an earlier commit between
2026-09-03 (HANDOFF) and HEAD, not the new fix. Blocker 1 from review: closed.

## 2. MoE decode rotation parity fix (committed `4da7b018`)

`!gate_side_mq4` decode branch now rotates with first routed expert
`gate_up.awq_scale`, mirroring prefill representative-scale convention.
`hfq_dump` evidence: router `mlp.gate.weight qt=3` (Q8, no AWQ),
experts `gate_up`/`down qt=44` (MQ4G256V2) with **zero AWQ sidecar entries**
in the 1035-tensor dump → `expert_awq=None` → new branch falls through to
unscaled `rotate_x_mq`, i.e. **behavior-preserving no-op for all current
artifacts** (Fuse + A3B). Keep: protects future AWQ-carrying MoE quants from
a per-channel scale error. k2-indexed admission untouched (no AWQ → cert holds).

## 3. Presence A/B (`serve_harness battery`, thinking off, greedy, q8/contiguous)

| sampling | runaway | attractor | avg decode | notes |
|---|---|---:|---|---|
| greedy (presence 0.0) | 2 | 2 | 113.0 | reproduces prod receipt exactly |
| greedy + presence 1.0 | **0** | 2 | 114.3 | reason 2048→342 tok, instruct 2048→112 tok, both `stop`; no perf cost |
| sampled 0.7/0.8/20 + presence 1.0 | 0 | 1 | 113.3 | prose fixed, but code loses code body, factual goes repetitive, instruct **refuses**, reason confabulates (`1.5 hours` for 250 mi) |

Eyeball: presence 1.0 converts runaway→bounded-degenerate; meta-chatter
persists (model property, `--raw` equally degenerate). Sampling is NOT a lever
for Fuse — greedy+presence 1.0 is the recommended serve policy.
Raw JSON: `/tmp/fuse_presence0.json`, `/tmp/fuse_presence10.json`, `/tmp/fuse_sampled10.json`.

## 4. Thinking probe: BLOCKED on template (the default-flip gate)

- Thinking-off serve: `reasoning={contract:qwen_jinja, mode:disabled}`,
  clean stop, no think leak. (Also observed: `17*23=379` — wrong, 391.)
- Thinking-on via scratch `models.toml` edit (mode on, budget low,
  effort xhigh, max_tokens 512; pinned file restored after):
  `reasoning={contract:qwen_jinja, mode:enabled, max_think_tokens:512}` but
  warning **`reasoning.effort 'xhigh' dropped: template does not natively
  support effort`** — the 7756-char embedded template has no effort rungs.
  No `<think>` block emitted; output is pure attractor
  ("The user is asking for the result of 17*33?" ×N) and misreads 23→33.
- Per-request `reasoning_effort` with mode off is dropped (`thinking disabled`)
  — config is authoritative, as designed.

**Gate for default:** byte-diff embedded template vs HF GGUF repo
(`Akahsizrr/Fuse-2-MoE-GGUF`) `tokenizer.chat_template`; confirm
`enable_thinking`/`reasoning_effort` names + rungs + `<think>` ids. If GGUF
template supports effort and embedded does not, the embed predates it → re-export
or pin template source before any thinking claim. BF16 repo stays the clean
re-quant source; no re-quant until the diff lands.

## 5. Template diff: embedded == upstream, effort unsupported by design

- GGUF repo (`Akahsizrr/Fuse-2-MoE-GGUF` @ `4b686d2d`) carries no JSON sidecars;
  template sourced from API `gguf.chat_template`: **7756 chars, byte-identical**
  to `/tmp/fuse_embedded_tpl.j2` (`cmp` clean). No re-export needed.
- Upstream template has `enable_thinking` but **no `reasoning_effort` variable**
  → effort-rung warning is correct upstream behavior, not staleness.
  Thinking control = `enable_thinking` boolean + think-token budget only.
- Template end-logic: `enable_thinking=false` → closed empty think block
  (current thinking-off framing); otherwise open `<think>`.

## 6. Thinking + presence combined probe: FAILS CLOSED (hard error)

Scratch config (reverted after): reasoning mode on / budget low / effort xhigh
(warns+drops, but enables) / max_tokens 512 + presence 1.0. Result:
`daemon error: open think span at end of generation (validation)`, rolled back,
no answer at all. Model enters `<think>`, never emits `</think>` in budget;
daemon refuses to force-close (qwen.rs:559). Presence does not save it.
Side finding: `effort=none` forces thinking disabled even with mode on
(request reports `mode:disabled`); effort xhigh enables despite the drop warning.

**Verdict:** thinking is mechanically fully wired (contract, framing, budget,
validation) but model-incapable on this artifact — thinking-off degenerate text
becomes a thinking-on hard error. Fuse cannot take thinking traffic. Qwen stays
for all reasoning workload; Fuse default, if any, is thinking-off code/factual only.

## 7. SHIPPED: Fuse presence_penalty 0.0 → 1.0

One-line `~/.hipfire/models.toml` change (fuse generation block only; Qwen
untouched, live serve not restarted — applies on next Fuse serve). Evidence §3:
runaway 2→0, decode 113.0→114.3, greedy retained. Sampling rejected.

## 8. State on exit

- `~/.hipfire/models.toml` untouched (scratch edit reverted, verified).
- Qwen `qwen3.8:27b-mq4-pro` serve restored on `:11435`, pre-warmed.
- Committed: `4da7b018` (fix) + this receipt. Uncommitted: none (tracked).
- Not run (parked): presence 1.5 cell, kv/vmm + max_seq A/Bs, MTP/DFlash economics.
