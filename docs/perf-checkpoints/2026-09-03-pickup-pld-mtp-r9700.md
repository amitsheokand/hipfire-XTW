# Pickup items 3+4: PLD operating window + MTP unblock path (Fuse-2, gfx1201)

**Date:** 2026-09-03.

## Item 3: PLD retrieval — exists, measured, bounded (no code change)

`NgramDrafter` (spec_ngram.rs) already implements PLD (suffix self-match
5/4/3 + bigram fallback) over prompt+emitted with exact verify. Grounded
bench (verbatim-repeat prompt): ngram median **142.2 vs 112.1 AR (+27%)**,
tau up to 7.0 — but sd 33 (93–181): when the weak model paraphrases
instead of copying, drafts reject and verify overhead bites.

Findings from probe instrumentation (all reverted, tree clean):

- Cold/short prompts offer NOTHING for ~11 windows (PLD needs 3–5gram
  history, bigram min_count=2): those windows run single-token verify at
  **84 vs 112 tok/s** — the verify path (recurrent snapshot +
  prefill-style batched verify kernels for 1 token + commit machinery)
  costs ~35% over a plain AR forward. This, not rejection, is the floor.
- An adaptive-backoff prototype (suppress offered-but-rejected drafts)
  was built, proven live via probes, then **reverted**: it correctly
  stays idle when nothing is offered, and offered-but-rejected windows
  are rare in practice — measured effect ~zero. The disease is
  empty-draft verify overhead, which backoff cannot suppress (empty
  proposal IS already the AR step; the cost is downstream of propose).
- Proper fix = bypass ChainSpeculator::step into a plain decode forward
  on empty drafts. That needs a new SpecTarget decode-step method across
  all arch impls (spec_advance/verify both run prefill machinery, so
  neither is the bypass). Invasive; deferred.

Operating window: enable ngram **only for long/grounded prompts**
(copy-heavy, code, RAG context); keep off otherwise. No default change.
REST-with-corpus remains a future option (needs corpus infra).

## Item 4: MTP head — blocked on training env (concrete unblock path)

- Scripts exist: scripts/mtp_train/ (0p8b 500-step wikitext template,
  27b v5/v6/v7 variants). All hardcode `device_map="cuda:0"`.
- No torch anywhere on this machine (system or nix dev shell). ROCm
  torch honors `cuda:0` naming, so the plausible path is: install
  ROCm torch → adapt a script to qwen35moe (Fuse dims/arch) →
  train head → pack .pt → .mtp sidecar → registry/models.toml → eval tau
  (qwen38 precedent tau~1.96).
- .mtp packing tooling was not located in-repo (ornith/qwen38 sidecars
  exist; pack step needs rediscovery). Flagged as part of the work.
- Estimated: env setup + port + train + pack + eval. Highest EV
  (+30–80% class) but entirely gated on the torch env decision.
