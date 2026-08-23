---
title: The-Rock8 R9700 traps — HIP_VISIBLE_DEVICES, 256 B EA, ~631 GB/s ceiling
date: 2026-08-23
tags: [gfx1201, r9700, profiling, the-rock8, rocprof]
---

The-Monk/The-Rock8 is a gfx1201 llama.cpp+Lemonade appliance on TheRock
ROCm, not a distro. Kernels: The-Monk/llama.cpp branch roc8
(`mul_mat_dense_fp8_mmq.cu`). Traps already paid: never mix
ROCR_VISIBLE_DEVICES (silent CPU fallback); STABLE_STD before rocprof or
counters read 0; no gfx12 counter XML — use PC sampling
(ROCPROFILER_PC_SAMPLING_BETA_ENABLED=ON); FETCH_SIZE must be
GL2C_EA_RDREQ_sum*256/1024; decode DRAM roofline ~631 GB/s on this card
(>631 is a measurement bug); hipBLASLt persist algo index not blob;
per-shape M-threshold (same default +32.7% 8B / −38.2% 24B). hipfire
scripts/r9700_profile.sh already encodes the FETCH_SIZE caveat. Ignore
Lemonade/GGUF serving. Related: [[fp8-g256-not-llamacpp-34b]].
