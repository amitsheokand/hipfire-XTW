# Fuse-2 MTP tau=0 diagnosis: chat-framing OOD (gfx1201, R9700)

**Date:** 2026-09-03 (post-reboot recovery session).
**Lifecycle:** historical. **Disposition:** diagnosis evidence for the framed-corpus retrain.

## Result

Packed Fuse MTP sidecar (`fuse-2-moe.mtp`, 115 MB, Q8, from 1500-step
self-distill, 0% → 62% teacher-forced holdout agreement) benches at
**tau=0.00, ~40 tok/s vs 117.8 AR** on gfx1201. Every draft rejects,
including step-0.

## Root cause: head trained unframed, served framed

Live requests are Jinja-framed (`<|im_start|>user\n{prompt}\n<|im_end|>\n…<think></think>\n`,
26 tokens for a 14-token prompt) even under `--raw` — the qwen.rs spec
path has no raw bypass (`try_jinja = jinja_enabled && template.is_some()`).
The training corpus was tokenized raw (wiki/code, no template). Measured
head-vs-trunk agreement on identical weights (GPU harness, saved triples):

| Context | Agreement |
|---|---|
| Unframed wiki/code (p1, 117 tok) | 12/12 positions (100% spot, 62% holdout) |
| Jinja-framed (26 tok) | 7–8/26 (~27–31%) |
| Live bench `--spec mtp` | tau=0.00 (0 accepts / ~20 windows) |

Failure mode is input-echo: on framed context the head re-emits
`last_committed` (cands `[314,314,314]`), which the trunk rejects.

## Exonerated (all proven correct in isolation, GPU, same bytes)

- Head kernels end-to-end (block + partial-rope + QK-norms + Q8 GEMV): 12/12.
- Per-token AND batched MTP-KV prompt fill: 12/12 each.
- Single AND batched trunk-embed lookup (bit-exact vs exporter table).
- `prev_hidden` seeding (norms ~157–162 match exporter distribution).
- Trunk verify path: harness batched-verify reproduces live trunk exactly
  (after framed+760 → 6511, twice).
- Q8 sidecar packing/naming (offline dequant 64.2%, key check clean).
- p_min (off on gfx1201), proposal graph (off), device-token chain, positions.

## Corrections to prior notes

- The offline AR-chain d1/d2/d3 numbers (3%/6%/4%) were a **checker
  artifact**: the script fed raw `t_mtp_out` (norm ~1.2) as next-hidden
  where the head expects trunk-scale hidden (norm ~150). Not a head
  quality signal. (Small `t_mtp_out` norm is normal — verified identical
  in the 12/12 harness runs.)
- qwen38's tau~1.96 precedent (2026-08-21, chat session WITH framing)
  used a COMPRESSED head trained on a different (corpus-derived) pipeline;
  it does not transfer to this full-vocab head + raw corpus.

## Path forward (approved next work)

Retrain with **framed prompts**: wrap corpus texts in the exact live
Jinja frame before tokenize/export so train distribution == serve
distribution. Expected: live step-0 agreement back to the ~60% regime,
tau > 1. No serve-code change needed. Temp debug scaffolding used here
(HIPFIRE_MTP_TRACE sites, tmp_encode_one/tmp_head_npy harnesses) was
fully reverted; tree clean at diagnosis close.
