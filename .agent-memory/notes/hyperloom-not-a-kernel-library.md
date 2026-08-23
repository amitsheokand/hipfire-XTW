---
title: Hyperloom is a vLLM/SGLang Instinct tuner, not a gfx1201 kernel source
date: 2026-08-23
tags: [hyperloom, geak, gfx942, ignore]
---

AMD-AGI/Hyperloom (MIT, Python, v1.0.0b2) auto-optimizes vLLM/SGLang on
MI300X/MI325X/MI355X via Claude/GEAK. Zero `.hip` in tree. Runners are
gfx942+gfx950; gfx1100 is explicitly unmapped; RDNA4 is unmerged PR #1032.
Do not wire it into hipfire (Python hot path, wrong hardware). If kernel
ideas are needed, read AMD-AGI/GEAK and AITER/CK, then re-lower to RDNA4
WMMA. Full briefing: docs/investigations/2026-08-23-r9700-fp8-external-refs.md.
