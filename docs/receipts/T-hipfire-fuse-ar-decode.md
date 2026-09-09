# Receipt: T-hipfire-fuse-ar-decode

**Task:** Fuse-2 MoE AR single-step decode divergence (sky 1-token probe).
**Worktree:** `/home/amitsheokand/dev/hipfire-worktrees/fuse-ar-decode`, branch
`work/fuse-ar-decode` (base `f336e49c`, cut from `R9700`).
**Date:** 2026-09-10. **GPU lock:** pre-held for this task tag by coordinator
(`T-hipfire-fuse-ar-decode`, holder `sleep 5400` under systemd user); :11435
verified empty before every probe (`ss -ltnp | grep 11435` → empty). Serve
left down; Qwen serve NOT restored (coordinator owns that).

## Finding: no decode divergence — spec route ignored `--raw`

The "198 vs 760" split was a **prompt-framing mismatch between routes**, not a
forward-pass bug. Both forwards are faithful to their inputs:

- AR (`ar::generate`, `crates/hipfire-generate/src/ar.rs:2257`): `--raw` →
  `tokenizer.encode(prompt)` = **11 raw tokens** → prefill argmax **271**
  (`\n\n`). Matches exporter ground truth on raw ids (handoff: labels
  `[… 13, 271]`, first=271). `HIPFIRE_FUSE_DEBUG=1` top-10 after prefill:
  `271:18.878 > 198:18.253 > 561:18.051 …` (760 not in top 10).
- Spec (`qwen::generate_dflash`, `crates/hipfire-generate/src/qwen.rs:1969`):
  had **no raw branch** — `try_jinja = jinja_enabled && template.is_some()`
  (Jinja default-ON), so a `--raw` spec request prefilled a framed render and
  emitted **760** (`The`) — the correct token *for the framed prompt*, wrong
  for the raw probe the operator asked for.

Decisive test: framed AR (same prompt, no `--raw` → 23 ChatML tokens) emits
`The`, identical to the spec route. Same framing → same token on every route;
the MoE dispatch / router-logits / DeltaNet / KV-read suspects are cleared
(`HIPFIRE_PREFILL_BATCHED=0` also leaves AR at 271, so batched prefill is
innocent too).

Note on the packet's Done-#1 (AR should emit 760): the evidence reverses it.
760 is the *framed*-prompt token; the exporter-faithful *raw*-prompt token is
271, which AR already emitted. Making AR emit 760 under `--raw` would have
made AR unfaithful and broken `--raw` semantics, so the fix threads `--raw`
into the spec route instead — routes now agree at the faithful token.

## Fix (33 insertions, 6 deletions)

- `crates/hipfire-generate/src/qwen.rs` — `generate_dflash` takes new
  `raw_requested: bool`; when set (or `HIPFIRE_RAW_PROMPT=1`, mirroring AR):
  `prompt_tokens = tokenizer.encode(prompt)`,
  `started_in_think = render_tail_opens_think(prompt)`, Jinja/Plain framing
  skipped. Cache-plan path untouched (raw runs carry no history → cold
  full prefill, as before).
- `crates/hipfire-generate/src/ar.rs` — all 5 `generate_dflash` call sites
  (Qwen2Spec, LfmSpec, CohereSpec, MiniMaxSpec, QwenDflash arms) pass through
  the existing `raw_requested`.

Out of scope, untouched: MTP economics/retrain, P-EAGLE, DFlash-on-Fuse,
Qwen serve, catalog/systemd, `forge/fuse` composites, formatting.

## Before / after (sky probe, `--raw --temp 0 --max-tokens 1`)

| route | before | after |
|---|---|---|
| `--spec off` (AR) | `\n\n` (271) | `\n\n` (271, unchanged) |
| `--spec ngram` | `The` (760, framed) | `\n\n` (271) ✅ |
| `--spec mtp` (v3 sidecar, `drafter=mtp`, K=3) | `The` (760, framed) | `\n\n` (271) ✅ |
| `--spec ngram` framed (no `--raw`) | `The` | `The` (unchanged) ✅ |
| framed AR (no `--raw`) | `The` | n/a (unchanged path) |

Weights `~/.hipfire/models/fuse-2-moe.mq4` md5
`dd4e34b1d3f99263580181af92ae5d7e`. Binaries after fix: `target/release/hipfire`
md5 `a0923181a68a691731d138e2cd7e08af`, `target/release/daemon` md5
`54f13eb271f5ad2933ece59fbb8307a0`. Env for all probes:
`HIPFIRE_HIPCC_EXTRA_FLAGS=--rocm-device-lib-path=/nix/store/8jdkas7zs4yzqs0n3xrrnnmc9s3xx5zb-rocm-device-libs-22.0.0-rocm/amdgcn/bitcode`,
`HIPFIRE_DFLASH_MODE=off`, `HIPFIRE_DFLASH_DRAFT`/`HIPFIRE_QWEN_MTP` unset.

## Checks run

- `cargo check -p hipfire-generate` — clean (pre-existing warnings only).
- `cargo test -p hipfire-generate --lib` — 15/15 pass.
- `cargo test -p hipfire-generate --test qwen_dflash_semantic_terminal_tests`
  — 65/65 pass.
- `cargo test -p hipfire-cli` — 213 pass, 2 fail
  (`http_reasoning_gemma_enabled_with_cap_and_budget_dropped`,
  `serve::complete::…match_native_contract`); **both fail identically on the
  clean base `f336e49c`** (verified via `git stash`), so pre-existing and
  unrelated (reasoning/gemma warning-text assertions; this change touches
  neither path).
- No `serve_harness`/`redline` run: serve stays down per packet (:11435 must
  stay empty, no Qwen-serve restore); acceptance is the local-daemon `run`
  probe pair above, which exercises the exact `generate` → route code changed.
- AR 32-token self-repeat: byte-identical across runs (deterministic).

## Re-examination against the literal 760 bar (post-fix, 2026-09-10)

The packet's Done-#1 requires raw AR to emit 760. Fresh, independent ground
truth shows that bar is unmeetable by any faithful decode, so it is PARKed
per the packet's own clause (exact site below) and the routes are converged
at the faithful token instead.

**Fresh exporter run** (this worktree build, `fuse_export_mtp_corpus`,
per-row lm_head GEMV + GPU argmax — a different code path from the daemon's
`scratch.logits` sampler, so an independent witness; `/tmp/hftest` was lost
so sky ids were rebuilt byte-exact from the handoff list):

- `sky11` = raw prompt ids `[814, 20139, 3069, 279, 12515, 369, 6105, 303,
  1330, 22157, 13]` → labels `[…, 13, 271]`: **P(·|raw prompt) = 271**.
- `sky12` = prompt + `[271]` → last label **760**: P(·|prompt, 271) = 760.
- Alignment check (rules out off-by-one): labels[0]=ids[1],
  labels[4]=ids[5], labels[5]=ids[6], labels[9]=ids[10] — labels[r] is the
  prediction *after* ids[r]. sky12 reproduces labels[10]=271 (deterministic
  across runs).
- AR prefill dump agrees: 271 at logit 18.878, 760 not in top-10.

**`--raw` is a deliberate contract**, not a legacy accident: commit
74a63cc3 ("--raw/--no-chatml prompt controls for non-chat models", Phase 1
of Fuse-2 bring-up) defines it as "send the prompt directly with no Jinja or
ChatML scaffolding", and the MTP corpus pipeline (raw ids → exporter labels
→ shift-2 training) is built on raw-ids trunk states.

Consequences: emitting 760 for the *raw* probe would require either framing
the raw prompt (deleting the 74a63cc3 contract and desyncing serving from
the exporter-anchored training corpus) or overriding the sampler past the
model's argmax (fabricating output). Both break faithfulness; neither was
done. The exact mismatch site for the record is the spec route's missing raw
branch — `crates/hipfire-generate/src/qwen.rs:1969` (`try_jinja` had no raw
gate; fixed in this receipt's commit) — plus the packet's verify-side repro
commands, which omit `--raw` and therefore compare raw-AR against
framed-spec. Apples-to-apples (raw vs raw, framed vs framed) agrees on all
routes; that is the corrected acceptance.

## Follow-ups (not this packet)

1. **Cross-route 32-token drift:** `--raw` AR vs ngram agree on tokens 1–~10
   (`\n\nThe sky is blue…`) then bifurcate into different fluent trajectories
   (AR echoes `in two sentences`; ngram loops `The sky is blue`). Both
   coherent, AR self-deterministic — smells like a sampler-scope asymmetry
   (repeat-penalty window) inside verify vs AR, NOT a forward bug. Needs its
   own packet with logit dumps, not a drive-by fix here.
2. **Pre-existing CLI failures** (above) belong to whoever owns the
   reasoning/gemma contract tests.
3. With AR↔verify agreement restored at the faithful token, the R9700
   Addendum-2 chat/flip matrix is measurable again — but re-measurement is a
   separate task; MTP stays parked per packet.
