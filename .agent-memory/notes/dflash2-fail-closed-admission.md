---
title: DFlash2 selector/conv admission is fail-closed
date: 2026-08-25
tags: [dflash2,selector,admission]
---

DFlash2 knobs merge **per field** across `dflash.dflash_config` (preferred on conflict) and `config.dflash_config`. Incomplete pairs (rank without top_k, kernel without group, group that does not divide hidden) fail parse.

A declared selector must load projection + both codebooks or `DflashWeights::load` returns `HipError` — no silent greedy path at B=16. `runtime_block_size` on weights widens 8→16 only after `has_candidate_selector()`. Declared conv requires both base+proj on every attn/MLP layer.

Related: `dflash2-engine-port-2026-08-24`, selector parse fix `dc48745b`.
