# Proposal: single-pass parallel MTP drafting (P-EAGLE-style) for hipfire

**Date:** 2026-09-03. **Status:** design only — build after head quality
warrants it (needs shift-2-level agreement at depth first).

## Why (numbers from this week's work)

- Serial K=3 draft costs 3 head forwards (~12 ms/window, 26% of window).
  K=5/K=7 sweep: tau flat at 0.47, tok/s drops 63.7 → 45.5. Depth is
  pure loss serially — P-EAGLE (arXiv:2602.01469, vLLM 1.69× over EAGLE-3
  at K=7) removes the linear draft cost, making K=7 viable.
- P-EAGLE also reports HIGHER acceptance length than serial at same K
  (3.94 vs 3.03 @K=7 HumanEval) — parallel training regularizes better.

## What changes

1. **Head:** add learned `mask_emb` + `mask_hidden` buffers; draft input
   for positions 2..K uses masks instead of chained outputs. Single
   forward produces K drafts (batched across K slots, causal mask over
   [prefix + K mask slots]).
2. **Train:** extend `train_fuse_mtp.py` with mask-slot rows: input
   `(E(ids[t+1]) or mask_emb, H(ids[t]) or mask_hidden)` → labels[t+k].
   Start from v3 weights (warm start needs resume support — add it).
3. **Serve:** new `spec_step_mtp_parallel` in mtp_spec.rs: one block
   forward over K slots (reuse `mtp_head_forward_block_batched` — it
   already batches slots!), one batched lm_head GEMM, K argmaxes.
   No token-chain kernel needed on this path. Verify unchanged.
4. **Packing:** two extra F32 vectors in the sidecar (negligible size).

## Sequencing (gated)

1. v3 head must first reach useful serial tau (on-policy fine-tune) —
   else parallel training has nothing to distill from.
2. Then: warm-start parallel head → bench K=7 AL vs serial → keep iff
   tok/s wins after verify costs.
3. The batched-fill/scratch infra (`Qwen35MtpHeadBatchedScratch`,
   `mtp_head_forward_block_batched`) already exists — the serve path is
   the smaller half; training is the larger half.

## References

- P-EAGLE blog (vLLM, 2026-03-13): single-pass K drafts, fused Triton
  batch-prep kernel, sequence-partition training.
- EAGLE-3: abandon feature-uncertainty (on-policy training) — do this
  first as the serial on-policy fine-tune.
