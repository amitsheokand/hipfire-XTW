# Receipt: T-hipfire-fuse-prod

**Task:** Production Fuse — native chat template + close serve decode vs 113.8 tok/s AR floor.
**Worktree:** `/home/amitsheokand/dev/hipfire-worktrees/fuse-prod`, branch
`work/fuse-prod` @ `b867f779`. **GPU:** gfx1201 R9700.
**Date:** 2026-09-10. GPU lock held (`T-hipfire-fuse-prod`).
Qwen serve (`qwen3.8:27b-mq4-pro`, pid 275960) stopped before probes,
`:11435` verified empty. **Serve left down; coordinator restores Qwen.**
Systemd default untouched (still Qwen). No MTP/DFlash work. No push.

## 1. Quality: packet premise is stale — native template already on the serve path

PACKET.md states `fuse-2-moe.mq4` has "no `tokenizer_config.chat_template`"
(1035 B HFQM JSON). The production artifact does not match that description:

- `~/.hipfire/models/fuse-2-moe.mq4` md5 `dd4e34b1d3f99263580181af92ae5d7e`
  (unchanged from the T-hipfire-fuse-ar-decode receipt) carries
  `tokenizer_config.chat_template` = **7756 chars**, the upstream GGUF
  `tokenizer.chat_template`, embedded by `d7eb7962` (in HEAD; file
  quantized 2026-09-03 after that commit). `gguf_meta` still carries the
  `tokenizer.chat_template` source key.
- Loader path (`crates/hipfire-loader/src/lib.rs:1359`): arch string
  `qwen35moe` → arch_id 6 → `qwen35_template_from_embedded` returns the
  embedded template verbatim (only `ornith-1.5-*` artifacts are adapted).
  No shadowing override exists: no `chat_template_file` in config, no
  `~/.hipfire/templates/fuse-2-moe.mq4.j2` (dir holds only ornith + qwen3.8
  files). Harness isolates `HIPFIRE_HOME`, so the same embedded path is used
  under test.
- Render probe (`render_chat_template`, framed sky prompt): native framing
  `<|im_start|>user … <|im_end|> <|im_start|>assistant <think>`, 21 tokens
  vs 11 raw — **not** the generic ChatML scaffold.
- Serve proof (15 served turns: sampled battery + greedy battery + greedy
  chain, thinking off): `think_words=0`, `atem_leak=False` on **every**
  turn, zero `<think>` tags in any transcript. The ChatML-fallback symptom
  (unclosed `<think>` spans, per `d7eb7962`) is absent.

**No template change was needed and none was made** (no re-quant of
5.5 GiB, no override file — an override would desync serve framing from the
exporter-anchored MTP corpus, which is built on the embedded template's
framed distribution).

### Residual quality (model property, out of scope)

Greedy serve still meta-chatters on prose/instruct
("The user wants a 4-sentence story…", "The user is asking for…") and hits
attractors on reason/instruct (battery greedy: runaway=2/attractor=2; chain:
attractor=2 with cross-turn context poisoning). Control: `--raw` completion
on the same prose prompt is equally degenerate (repetition loop), so this is
**model weakness, not framing** — no template/serve lever fixes it, and
training levers are out of scope (MTP parked, Path C dead). `code`/`factual`
turns are clean and on-rails. `models.toml` keeps thinking off / spec off /
greedy for `fuse-2-moe`; thinking-off framing verified (no think block
emitted, `[WARN: INVALID CONFIG] max_think_tokens/budget dropped` expected).

## 2. Speed: gap closed — serve within ~1–4% of the AR floor

Fixture: `benchmarks/prompts/fuse_prod_sky.txt` (new, committed),
md5 `e529c2204a2531d836d5ae2a4b755228`. Binaries (this worktree build):
`target/release/hipfire` md5 `b4f9e1a381fb0b7986bb533868ce8533`,
`target/release/daemon` md5 `1fccf77ba1c8339a1b95155034742180`.
Env: `HIPFIRE_HIPCC_EXTRA_FLAGS=--rocm-device-lib-path=/nix/store/8jdkas7zs4yzqs0n3xrrnnmc9s3xx5zb-rocm-device-libs-22.0.0-rocm/amdgcn/bitcode`,
`HIPFIRE_DFLASH_MODE=off`, `HIPFIRE_DFLASH_DRAFT` unset.

| path | decode tok/s |
|---|---|
| `bench` noslots q8, spec off, greedy, 3 fresh procs × 5 runs | 117.3 / 117.1 / 117.9 (floor ≈ **117.4**, above the 113.8 checkpoint on the newer build) |
| same + `HIPFIRE_AR_GRAPH=0` | 109.8 / 110.3 / 110.0 (≈ 110.0) |
| serve battery, thinking off, sampled (temp 1.0) | avg **115.9** |
| serve battery, thinking off, greedy (×2 runs) | avg **113.1 / 114.0** |
| serve chain, thinking off, greedy (prefix cache hits: cached=302) | avg **112.1** |

- **k=2 indexed graph is ON by default on the Fuse serve path**: killing
  only `HIPFIRE_AR_GRAPH` costs −6.7% (117.4 → 110.0), which proves
  `use_graph` was live — and therefore `moe_k2_indexable` certified true at
  load, `graph_ar`/`graph_moe` defaults held, and no kill-switch/legacy
  memset blocks capture. Consistent with `f04efe61` (+5%). No code change.
- **Target met**: serve greedy 113–114 vs 113.8 checkpoint floor (−0.6%),
  vs own-build bench floor 117.4 (−3%). The historic 73.7 serve number does
  not reproduce on this build. Prefill ~1350 tok/s, TTFT ~23.5 ms.

## Files touched

- `benchmarks/prompts/fuse_prod_sky.txt` (new — byte-identical bench prompt)
- `docs/receipts/T-hipfire-fuse-prod.md` (new — this file)

No engine code changed. `PACKET.md` (coordinator-provided, untracked) left
in place, uncommitted.

## Checks run

- `cargo build --release` (nix shell) — clean, warnings only.
- `render_chat_template` (lab feature) — native framing render, proof §1.
- `serve_harness.py --mode battery` ×2 (sampled + greedy, thinking/spec off,
  q8/contiguous) + `--mode chain` (greedy) — transcripts eyeballed, JSON at
  `/tmp/fuse_battery.json`, `/tmp/fuse_battery_greedy.json`,
  `/tmp/fuse_chain_greedy.json` (tmp-local; accusation-grade numbers are in
  the tables above).
- `hipfire bench` 3×(5 runs, 3 warmups) fresh-process + 3× graph-killed A/B.
- No `cargo test`: no code changed. No redline route: no kernel/dispatch
  change.

## Blockers / risks for the parent

1. Serve quality ceiling is the model, not the harness: prose/instruct/reason
   meta-chatter + attractors persist under greedy + native template + think
   off, and `--raw` is equally degenerate. Do not accept a "production Fuse
   quality" claim on chat genres without a training-side lever.
2. I did not load the `astrea` skill for KLD/PPL eval: nothing in this packet
   (no re-quant, no k-map change, training levers explicitly out of scope)
   consumes a calibration verdict. If the parent wants a promote/reject
   number on the MQ4V2 artifact, that is a separate packet.
3. Qwen serve left DOWN (`systemctl --user status hipfire-serve.service` =
   inactive). Coordinator owns the restore.
