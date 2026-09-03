# Fuse-2 MTP v3: shift-2 training fixes tau 0.05 → 0.47, still uneconomic (R9700)

**Date:** 2026-09-03. **Lifecycle:** historical.
**Disposition:** mechanism proven, economics negative — MTP parked for Fuse-2 MoE.

## What was wrong

The serve draft loop feeds step-0 `(E(last@P), H(@P-1))` (EAGLE-shaped).
Heads v1/v2 were trained shift-1 `(E(@t), H(@t)) → t+1`. Live, the
embed/hidden position mismatch makes the head echo its embed input.
Proven by a Python probe on identical bytes: serve-shaped queries on the
v2 head score 32% (echo) vs 74% shift-1.

## Fix: shift-2 (EAGLE-shaped) training, same corpus, no re-export

`train_fuse_mtp.py`: inputs `(E(ids[t+1]), H(ids[t])) → labels[t+1]`
(eval + loss). Framed corpus (226 prompts) reused unchanged. 1500 steps,
~65 min: **0% → 58.3% shift-2 holdout agreement** (plateau from ~step 1000).

## Live result (gfx1201, `/tmp/hftest/bench` noslots, code prompt)

| Head | Offline agree | Live tau | Live decode tok/s | AR baseline |
|---|---|---|---|---|
| v1 (shift-1, raw) | 62% | 0.00 | ~23 | 117.8 |
| v2 (shift-1, framed) | 67% / 74% prompt | 0.05–0.07 | ~43 | 117.8 |
| **v3 (shift-2, framed)** | **58%** | **0.47 stable** | **~32.5** | **117.8** |

Shift-2 mechanism confirmed (7–9× tau lift, stable across requests —
the v2 first-request/second-request split is gone). Installed:
`~/.hipfire/models/fuse-2-moe.mtp` (md5 `842cda4e…`, Q8, 15 tensors,
round-trip PASS). `models.toml` keeps `mtp = off` for fuse.

## Why parked: verify-dominated economics at K=3 on a MoE trunk

Per window (~46 ms): batched trunk verify over K+1=4 tokens (~34 ms) +
3 head forwards (~12 ms) → 5.4× one AR token (8.5 ms) for 1.47 tokens.
Break-even needs tau ≈ 4.4 > K=3 maximum (perfect K=3 still 1.35× loss).
qwen38 won (tau 1.96) on a different trunk/overhead profile; it does not
transfer. Reviving MTP here needs a cheaper verify or larger K with a
near-perfect head — a separate project, not a follow-up tweak.

## Reproduction (all crash-vulnerable paths under /tmp/hftest/)

- Corpus: `corpus_framed/` (226, `fuse_frame_corpus` + `fuse_export_mtp_corpus`,
  ~25 min) · tables: `tables1/` (`export_fuse_embed.py`, ~2 min)
- Train: `train_fuse_mtp.py --corpus corpus_framed --tables tables1
  --out mtp_fuse_v3` (~65 min) · pack: `mtp_extract --quant q8`
- Checkpoints: `mtp_fuse_v3/model_best.safetensors` (451 MB) +
  packed `fuse-2-moe-v3.mtp`
- Commits: framer (`c9a633b0`), shift-2 trainer (`3a5e0b8c`)

## Addendum: follow-up analysis (same day)

- **A/B (same prompt/flags, noslots, max-tokens 128):** AR 116.4 tok/s
  decode / 1433 prefill / ttft 27 ms vs MTP-v3 (tau 0.59) 63.7 / 771 /
  51 ms. MTP is a 1.8× net loss at its best measured tau.
- **Production flip: NO.** Fuse decodes 3.5× faster than qwen38 prod
  (116 vs 33 tok/s) but chat quality blocks: same coding prompt →
  qwen38 emits a full correct function (dflash tau 8.15), Fuse emits
  one stub sentence then stops. Speed without following is not shippable.
- **k>0 chain:** confirmed worthless today (steps 1–2 correct only when
  step-0 was; chain feeds raw t_mtp_out never seen in training). Chained
  fine-tuning (EAGLE active style) is the fix, but payoff is gated behind
  verify economics (below) — parked, not next.
- **Cheaper verify:** break-even needs tau ≈ 4.4 > K=3 max. Larger K
  needs a near-perfect head first. Parked behind head quality.
- **Shift-2 elsewhere:** deepseek4 (qwen38) loop uses the same EAGLE shape
  (`mtp_last_hidden` + seed) and its head is consistent with it (tau 1.96
  live) — no action. Lesson recorded for future head training.
