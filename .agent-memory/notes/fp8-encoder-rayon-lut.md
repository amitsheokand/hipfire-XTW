---
title: FP8 E4M3 G256 encoder uses Rayon + mag LUT
date: 2026-08-24
tags: [gfx1201, r9700, fp8, encoder]
---

`quantize_fp8e4m3_g256_2d` was a serial row loop; `f32_to_e4m3fn` scanned
0..=0x7E with `powi` per weight. Rayon pool (MoE experts only) never ran
on dense FP8, so 27B encode sat at ~99% of one core.

Fix: const MAG[127] + binary-search nearest (ties keep lower code, same
as the old `<` scan); row-parallel `par_chunks_mut`. Default threads
`cores-2` (12→10). Unit tests: LUT vs powi, convert vs brute, par vs
serial rows. Restarted Qwen3.8-27B encode with `--threads 10`.

Related: [[fp8-encoder-cpu]], [[fp8-e2e-qwen35-4b]].
