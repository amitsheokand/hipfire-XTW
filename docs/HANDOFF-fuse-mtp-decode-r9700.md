# HANDOFF: Fuse-2-MoE MTP + AR-decode divergence (R9700)

**Written:** 2026-09-03 ~20:00 UTC. **Branch:** R9700. **Tree:** clean
(docs + `scripts/mtp_train/train_fuse_mtp.py` only; no engine changes).
**Production:** untouched, healthy on `qwen3.8:27b-mq4-pro`
(`systemctl --user status hipfire-serve.service`).

## TL;DR for the next session

1. **MTP head v3 works (tau 0.47–0.59) but is parked on economics**
   (verify-dominated: break-even needs tau≈4.4 > K=3 max).
2. **Headline finding (latest, Addendum 5): AR single-step DECODE diverges
   on Fuse-MoE.** Exporter ground truth says 2nd token = 760; all spec
   paths (MTP/ngram, batched + per-token verify) emit 760; AR decode emits
   198. Every AR quality observation this session is UNSOUND, including the
   flip-killing chat matrix. **Next task: fix AR decode, then re-measure
   everything.**
3. All evidence committed as checkpoints (see § Evidence). Artifacts backed
   up durably (see § Artifacts).

## The one-paragraph story

v1 MTP head trained on unframed text hit tau=0.00 (framing OOD: 62%→27%).
v2 retrained on Jinja-framed corpus (67%) but live tau only 0.05 — diagnosed
as a shift-1/shift-2 mismatch: the serve loop feeds EAGLE-shaped
`(E(last@P), H(@P-1))` queries while the head was trained `(E(@t), H(@t))`,
so it echoed its embed input live (Python probe: 32% serve-shaped). v3
retrained shift-2 (58% holdout) → live tau 0.47–0.59, stable. K=5/K=7 sweep:
tau flat, tok/s drops — K=3 stands. On-policy pilot (`--chain-w 0.5`):
Python chain 10%→55% but live tau REGRESSED 0.59→0.17 — reverted to v3.
Then the quality probe: AR vs MTP decoded text differs deterministically
(temp 0). Bisection (K=1, per-token verify, ngram, max-tokens=1,
exporter labels) proves the VERIFY path is faithful and AR DECODE is
divergent. Suspect: single-step MoE/dispatch path (router-logits?).

## Evidence (all committed, read newest-last)

- `docs/perf-checkpoints/2026-09-03-fuse2-mtp-tau0-framing-ood-r9700.md`
  — framing OOD diagnosis (f2468190 context).
- `docs/perf-checkpoints/2026-09-03-fuse2-mtp-v3-shift2-parked-r9700.md`
  — THE MASTER DOC: v3 result, A/B table, flip verdict, K-sweep,
  chat matrix, on-policy negative, decode divergence (Addenda 1–5).
- `docs/perf-checkpoints/2026-09-03-mtp-parallel-draft-proposal-r9700.md`
  — P-EAGLE single-pass design (gated, do not build yet).
- Code commits on R9700 (all `Co-authored-by: CommandCodeBot`):
  `f2468190` trainer+pipeline+checker · `1e3b07e4` tau0 checkpoint ·
  `c9a633b0` framer tool · `3a5e0b8c` shift-2 trainer · `3fd10763`,
  `12f58f88`, `b87afdaa`, `fb690714`, `205aa52e`, `35c45860`, `c3c5cc0f`
  checkpoints · `ac0d3b6e` + `350c04e6` `--init` warm start ·
  `ff756fb4` `--chain-w` + chain metric.

## Artifacts

**Durable** (`~/.hipfire/artifacts/mtp_fuse_v3/`, md5-verified):
`model_best.safetensors` (v3, 451 MB) · `model_v4_chain.safetensors`
(negative pilot, forensics only) · `config.json` · `train_metrics.json` ·
`fuse-2-moe-v3.mtp` (== installed sidecar, md5 `842cda4e…`).
Installed: `~/.hipfire/models/fuse-2-moe.mtp` (v3), `models.toml` keeps
`mtp = off` for fuse.

**Crash-vulnerable** (`/tmp/hftest/`, lost on reboot — rebuild recipe):
`corpus_framed/` (226 framed prompts, ~25 min via `fuse_frame_corpus` +
`fuse_export_mtp_corpus`) · `tables1/` (embed/lm_head, ~2 min via
`export_fuse_embed.py`) · `mtp_fuse_v3/`, `mtp_fuse_v4/` (train outputs) ·
`corpS*` (sky-probe files: `corpS/sky.txt`, `corpS_ids/sky{,271}.ids`,
`corpS_exp/sky{,271}.{hidden,labels}.npy`) · `ref_*.py` probes.

## Key numbers (don't re-measure blindly)

- A/B same prompt/flags (noslots, 128 tok): AR 116.4 tok/s dec /
  1433 prefill / 27 ms TTFT vs MTP-v3 (tau 0.59) 63.7 / 771 / 51 ms.
  qwen38 prod AR: 33.0 tok/s.
- Sky probe: raw ids `[814 20139 3069 279 12515 369 6105 303 1330 22157 13]`;
  exporter labels `[... 13, 271]` (first=271) and with 271 appended
  `[... 271, 760]` (second=760). AR-m1=`\n\n`, MTP-m1=`The`.
- v3 train: 1500 steps ~65 min, 0→58.3% shift-2. v4 pilot: 500 steps,
  chain 10.3→55.0%, shift-2 flat 58.33%, live tau 0.59→0.17.

## Resume commands

```bash
# Health + state
git -C ~/dev/hipfire log --oneline -3; git status --short | grep -v '^??'
systemctl --user is-active hipfire-serve.service
md5sum ~/.hipfire/models/fuse-2-moe.mtp   # expect 842cda4e… (v3)

# Serve must be DOWN for GPU work, restarted after (user-approved pattern):
systemctl --user stop hipfire-serve.service; sleep 3; … ; \
  systemctl --user reset-failed hipfire-serve.service; \
  systemctl --user start hipfire-serve.service

# Canonical MTP bench (v3 expectation: tau≈0.5, ~60 tok/s):
nix develop --command ./target/release/hipfire bench ~/.hipfire/models/fuse-2-moe.mq4 \
  --spec mtp --runs 3 --warmups 1 --max-tokens 128 --backend noslots \
  --workload stateless --json "Write a Python function…" 2>&1 | grep -E 'tau=|tok/s='

# Deterministic text probe (the decode-divergence test):
P="Explain why the sky is blue in two sentences."
nix develop --command ./target/release/hipfire run ~/.hipfire/models/fuse-2-moe.mq4 \
  --raw --spec off --temp 0 --max-tokens 1 --json "$P"   # expect '\n\n' (271)
# vs --spec mtp / ngram → expect 'The' (760) ← the divergence

# Exporter ground truth (needs serve down):
nix develop --command ./target/release/examples/fuse_export_mtp_corpus \
  --model ~/.hipfire/models/fuse-2-moe.mq4 --ids-dir <ids> --out-dir <out> --max-tokens 64
# Tokenize any text: ./target/release/examples/encode_prompt <model.mq4> <txt>
# Train: mtp-torch scripts/mtp_train/train_fuse_mtp.py --corpus … --tables … \
#   --out … [--init <safetensors>] [--chain-w 0.5] (default chain-w 0 = safe)
# Pack: ./target/release/mtp_extract --hf-dir <out> --output <x>.mtp --quant q8
```

## Next task (in priority order)

1. **Fix AR decode divergence on Fuse.** Repro: 1-token probe above
   (198 vs 760). Verify path (batched AND per-token) + exporter agree;
   only single-step decode is wrong. Suspects: single-step MoE dispatch /
   router-logits handling, DN single-step vs batch, KV-read path. Bisect
   with `HIPFIRE_PREFILL_BATCHED=0` (already shows verify innocent) then
   code-level: compare `forward_scratch*` decode vs `forward_prefill_batch`
   n=1 on identical (KV, DN, token, pos). Needs `enter_plan_mode` (multi-file,
   unfamiliar area) — do NOT start by reading files one-by-one.
2. **Re-measure AR quality + flip** once decode is fixed (Addendum 2's chat
   matrix is void). Fuse AR may be much better than measured.
3. **Only then:** on-policy via GPU-capture harness (Python chain loss
   proven non-transferring), P-EAGLE single-pass (proposal doc exists).

## Gotchas learned (read before touching anything)

- `bench` has NO `--draft-max` (run-only); K for bench comes from
  `HIPFIRE_MTP_K` env. `HIPFIRE_MTP_K=0` rejected (min 1).
- `run` temp flag is `--temp` (not `--temperature`); `temp: Option<f64>`.
  Stops are only `<|im_end|>`/`<|endoftext|>` — no fence stop.
- Daemon holds flock on `~/.hipfire/daemon.pid` — stop serve before bench.
- `mtp-torch` provides torch+numpy (system python has neither).
- Trainer: no `empty_cache()` (corrupts ROCm math); foreach=False for Adam;
  checkpoints save with `mtp.` prefix; `--init` strips it (fixed 350c04e6).
- Framed corpus ONLY (unframed = OOD); labels ≈ ids[t+1] (teacher).
- Never leave background/dev processes; never commit without reading first;
  traces must be removed before bench/commit.
