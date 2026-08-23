# R9700 FP8 branch: external references (2026-08-23)

- **Date:** 2026-08-23
- **Branch:** `r9700-fp8-adaptive` @ `7e345c4c` + uncommitted GEMM
- **Status:** investigation / briefing. Not a product default, floor, or admission.
- **Hardware:** Radeon AI PRO R9700 (`gfx1201`), 32 GB GDDR6

Five sources were fetched the same day as the FP8 GEMV landing. This note
records what they actually are, what hipfire should steal, and what to
ignore. Canonical owners for procedure remain
[`docs/VALIDATION.md`](../VALIDATION.md), [`docs/R9700.md`](../R9700.md),
and `.agents/skills/hipfire-arch-port/`.

**Fetch limits.** The RDNA4 ISA page is a PDF viewer (doc 70651, 697 pages);
instruction names below come from extracted text, not a redistributed PDF.
arXiv 2608.16157 is the FreeToken paper (same project as the GitHub repo).
Do not vendor the AMD ISA PDF (license forbids redistribution).

---

## Branch snapshot (this checkout)

| Item | State |
|---|---|
| Fork | `amitsheokand/hipfire-XTW`, branch `r9700-fp8-adaptive` (no upstream tracking) |
| Base | `master` @ `80a572c8` (2026-08-18 integration / Saddle) |
| `3b9c6b70` | Phase 0–2 groundwork + `QuantType` 40/41 slot |
| `7e345c4c` | Native E4M3 G256 **GEMV** (`v_dot4_f32_fp8_fp8`), GPU-validated |
| Working tree | **GEMM** kernel + `Gfx1201Device::fp8_gemm_e4m3_g256` + lab example, uncommitted |

GEMV result recorded in the commit: NRMSE 2.5–2.7% vs CPU FP32 (F32→E4M3
activation rounding; 5% relative tolerance), 46–58% peak BW on attention
shapes. HIP launch must pass `&mut` addresses, not pointer values. The
`__shared__` + `__syncthreads` reduce compiled to a 24-instruction stub
under `--genco`; shuffle-reduce is the live path.

Config keys already exist and are **off**: `experimental.fp8_wmma`,
`experimental.fp8_format` (default `e4m3`), `experimental.ubatch_size`,
`experimental.moe_expert_cache_mb`. No daemon / quantizer encoder / HFQ
writer for qt=40 yet.

### Layout comment bug (do not copy llama.cpp)

`crates/hipfire-quantize/src/hfq.rs` names the slot `FP8E4M3G256` and the
kernel uses **256-element** leaves + fp16 per-group scale + 16 B header.
The same comment also says “34 B/group (32 × 1B leaf + 2 B fp16 scale),
8.5 bpw”. That 32-elem / 34 B layout is **llama.cpp / The-Rock8 F8E4M3**,
not hipfire G256. Same WMMA opcode, **not drop-in**. Fix the comment when
the encoder lands.

---

## Source map

| Source | What it is | hipfire overlap | Action |
|---|---|---|---|
| [AMD-AGI/Hyperloom](https://github.com/AMD-AGI/Hyperloom) | Claude/Codex agent that auto-tunes **vLLM/SGLang** on Instinct (MI300X/MI325X/MI355X). MIT. No in-tree HIP kernels. | Domain only (AMD LLM serving). | **Ignore product.** Optional: read GEAK/AITER, not Hyperloom Python. |
| [The-Monk/The-Rock8](https://github.com/The-Monk/The-Rock8) | gfx1201 llama.cpp + Lemonade **appliance** on TheRock ROCm 7.13/7.14. Kernels live in [The-Monk/llama.cpp `roc8`](https://github.com/The-Monk/llama.cpp/tree/roc8). MIT. Validated on 2× R9700. | Same silicon ops (`v_wmma_f32_16x16x16_fp8_fp8`, `v_dot4_f32_fp8_fp8`). Different weight layout. | **Steal tiling, geom sweep, profiling traps.** Do not steal GGUF/Lemonade. |
| [RDNA4 ISA](https://docs.amd.com/v/u/en-US/rdna4-instruction-set-architecture) | Family ISA, doc **70651**, 7 Apr 2025, 697 pp. Never names gfx1200 vs gfx1201. | Confirms current GEMM fragment layout and C-map. | **Keep as ISA authority.** F8_Mode for WMMA is unspecified — prove on silicon. |
| [arXiv:2608.16157](https://arxiv.org/abs/2608.16157) | FreeToken paper (17 Aug 2026). NVIDIA-only edge MoE serving. | Serving/MoE pager, not GEMM. Matches the unused `moe_expert_cache_mb` knob. | **Park until experts do not fit VRAM.** |
| [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken) | Apache-2.0 Python/CUDA implementation of that paper. Not a tokenizer, not spec-decode. | Algorithm only (LRU experts, \(q^\star\), semantic anchors). | **Do not vendor.** Ideas → `WeightPager` later. |

---

## 1. Hyperloom — ignore the product

GitHub description: agentic auto-optimizer for LLM workloads on AMD GPUs.
It profiles a live **vLLM ≥ 0.21 / SGLang ≥ 0.5.12** job on **CDNA**
Instinct, then an LLM loop rewrites host configs and delegates kernel
rewrites to **GEAK** (optional Forge). Recursive tree: ~1.1k Python files,
**zero** `.hip`. Official GPUs: MI300X / MI325X / MI355X. Issue
[#1041](https://github.com/AMD-AGI/Hyperloom/issues/1041): runner map is
`gfx942` + `gfx950` only; tests assert `_GFX_TO_RUNNER.get("gfx1100") is None`.
RDNA4 is an unmerged draft ([PR #1032](https://github.com/AMD-AGI/Hyperloom/pull/1032)).

**Steal (from siblings, not Hyperloom itself):**

- [AMD-AGI/GEAK](https://github.com/AMD-AGI/GEAK) — HIP/Triton rewriter with
  A/B + output parity. Results are **gfx942**.
- Trace hygiene (warmup discard, IQR KEEP/REVERT) — hipfire already has a
  stricter protocol in `docs/methodology/perf-benchmarking.md`.
- Arbor methodology paper: [arXiv:2606.12563](https://arxiv.org/abs/2606.12563)
  (agent search, not a kernel).

**Ignore:** Claude/Codex loop, vLLM/SGLang/Quark/EAGLE3, FlyDSL, Instinct
quotas, Python in any hot path.

---

## 2. The-Rock8 — steal method, not layout

Not AMD TheRock, not a compiler. Docs + Podman image wrapping a gfx1201-only
llama.cpp HIP fork (`roc8`) + Lemonade. Runtime is a vendored TheRock ROCm
tree (`libamdhip64.so.7.13…`; fork also tested on 7.14). Image:
`ghcr.io/the-monk/the-rock8:rdna4-tr713`.

**Kernel file to read (fork, not the appliance repo):**
`ggml/src/ggml-cuda/mul_mat_dense_fp8_mmq.cu` — two
`__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` per K32, LDS
double-buffer, geom sweep (`GGML_HIP_DENSE_FP8_MMQ_GEOM`). Default 128×128×8;
they measured 64×64×8 faster on N≤1024. Decode: `v_dot4_f32_fp8_fp8`.
hipBLASLt F8E4M3 prefill is **opt-in** with a per-shape M-threshold; the same
threshold was **+32.7% on 8B and −38.2% on 24B**. Persist **algo index**, not
blob (blob restore segfaults). gfx1201 hipBLASLt has **no cost model**.

**Measurement traps they already paid for (align `docs/R9700.md`):**

- `HIP_VISIBLE_DEVICES` only. Mixing `ROCR_VISIBLE_DEVICES` silently
  CPU-falls-back.
- `amd-smi set --gpu N --perf-level STABLE_STD` before rocprof — RDNA4
  memory counters otherwise read 0.
- rocprofiler-sdk has **no gfx12 counter XML**. gfx11 `FETCH_SIZE` is
  wrong; EA requests are **256 B**. Working path: PC sampling
  (`ROCPROFILER_PC_SAMPLING_BETA_ENABLED=ON rocprofv3 --pc-sampling-method host_trap`).
- Measured decode DRAM roofline on this card: **~631 GB/s** (hipfire
  runbook quotes ~640 GB/s peak). Anything above ~631 GB/s is a measurement
  bug.
- MUL_MAT_ID OOB when `rows_per_block > 1` and `nrows % rows_per_block != 0`
  (wrong MoE decode). Treat as a known blocking-class failure.
- F16 `v_dot2_f32_f16` decode lost; dual-issue VOPD ~1.04× not 2×.

**Ignore:** Lemonade, GGUF Q1/Q2/Bonsai, ~123 `GGML_HIP_*` research flags,
2:4 SWMMAC as a decode lever, copying launch geometry across kernels,
their tok/s as hipfire baselines.

Toolkit: [The-Monk/rocky-hackathon `toolkit/`](https://github.com/The-Monk/rocky-hackathon/tree/master/toolkit)
(`scan-isa-gfx.sh gfx1201`, `disasm-gfx.sh`).

---

## 3. RDNA4 ISA — confirms the WIP GEMM shape

| Field | Value |
|---|---|
| Title | "RDNA4" Instruction Set Architecture: Reference Guide |
| Doc ID | 70651 |
| Date | 7 April 2025 |
| Viewer | https://docs.amd.com/v/u/en-US/rdna4-instruction-set-architecture |
| Direct PDF | https://docs.amd.com/api/khub/documents/uQpkEvk3pv~kfAb2x~j4uw/content |
| GPUOpen hub | https://gpuopen.com/amd-gpu-architecture-programming-documentation/ |
| XML ISA | https://gpuopen.com/machine-readable-isa/ |

Dense FP8 WMMA (wave32): **`V_WMMA_F32_16X16X16_FP8_FP8`** (VOP3P opcode 70).
A/B = **2 VGPRs/lane** (8 FP8), C/D = **8 F32**. No OPSEL/ABS/NEG/clamp.
Clang: `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`.
C-map (trust the worked example, not the conflicting prose sentence):
`acc[j] = C[8*(tid>>4)+j][tid&15]`. That is what
`gemm_fp8e4m3_g256_wmma.gfx1201.hip` already stores.

FP8 = E4M3, BF8 = E5M2. Two **F8_Mode** rows (bias 7 / max 448 vs bias 8 /
max 240). **The ISA does not say which mode WMMA or `V_CVT_PK_FP8_F32` use.**
Mode 0 is OCP E4M3fn and is what G256 almost certainly wants — prove with an
Inf/NaN/max oracle, not a comment.

Also present: `V_DOT4_F32_FP8_FP8` (GEMV), `V_CVT_PK_FP8_F32` (RNE pack),
`GLOBAL_LOAD_TR_B64` (8-bit 16×16 load+transpose; unused unless major-order
≠ VGPR layout). G256 leaves are K-contiguous → plain 64-bit/`v2i_t` load is
the ISA-correct row, not TR.

**gfx11→gfx12 bites that still apply:** wrong builtin family; A/B length
halves; `kABKLane` 1→2; `kRepeat` 2→1; C-map even/odd-rows → contiguous
8-row groups (`b7ac66a`-class silent corruption); WMMA ignores EXEC so
partial tiles clamp loads and predicate stores.

**Gaps (ROCm headers / silicon, not this PDF):** F8_Mode programming, exact
NaN/denorm bit mapping, HIP `hip_fp8.h` vs WMMA FP8 identity, R9700
throughput/occupancy tables, CK `WmmaTraits`.

---

## 4. FreeToken (paper + repo) — parked MoE serving

Citation: Yang et al., *FreeToken: Efficient Edge-Native MoE Serving with
Bandwidth-Adaptive Execution*, arXiv:2608.16157, 17 Aug 2026.
Code: https://github.com/FlashML-org/FreeToken (Apache-2.0, CUDA 13 / PyTorch,
NVIDIA-only). Project: https://flashml.ai

Not tokenization. Not DFlash/MTP/n-gram. Host holds the full expert pool;
leftover VRAM is a shared LRU of complete `(layer, expert)` slots. Prefill
double-buffers **full layers**. Decode splits \(m\) unique misses with
\(q^\star \approx m\, B_P / B_H\) (PCIe fill vs CPU in-place); GPU and CPU
partials merge **exactly**. Recurrent state is checkpointed at **semantic
anchors** (think / tool / turn tokens) on a radix prefix tree — this covers
harness **edits** that kill hipfire’s LCP forward-extension.

Eval is **all NVIDIA**. No AMD numbers. hipfire’s current FP8 GEMM work is
orthogonal. The overlap is the already-added `experimental.moe_expert_cache_mb`
knob and `WeightPager` v0.1 (residency map, **no real eviction yet**;
[`docs/specs/2026-07-19-weight-pager-eviction-policy.md`](../specs/2026-07-19-weight-pager-eviction-policy.md)
cites SpecMD that LRU can lose to staleness-aware eviction — empirical fork
vs FreeToken’s “global LRU beats static”).

**Steal later (algorithm):** \(q^\star\) from measured \(B_P, B_H\) on this
box; complete-expert LRU; full-layer prefill double-buffer; semantic-anchor
DeltaNet checkpoints; device-side miss classification inside hipGraph.

**Ignore:** Python/Torch hot path, CUDA Graphs / FlashInfer, NVFP4/MXFP4 as
product quants, “always LRU” without pager traces, using their 5090 tok/s as
an AMD claim. DFlash + CPU experts can tank wall-clock tok/s at unchanged τ.

---

## Recommended sequence (this branch)

Skills in play: `.agents/skills/hipfire-arch-port/` (WMMA C-map, chip tag),
`.agents/skills/hipfire-kernel-tuning/` (one lever, fresh-process measure),
`.agents/skills/agent-memory/` (this note + `.agent-memory/notes/`).

| Phase | Goal | Success metric | Est. |
|---|---|---|---|
| **3b (now)** | GPU-validate uncommitted WMMA GEMM vs CPU FP32 | `test_gemm_fp8e4m3_g256` ALL PASS; NRMSE in the same ~5% F32→E4M3 band as GEMV; no page fault | hours |
| **3c** | F8_Mode / C-map oracle (16×16×16 and 16×16×256) | `cvt_pk_fp8_f32` bytes match G256 encoder; rows 0–7 vs 8–15 C-map; max 448 not 240 | hours |
| **3d** | Fix qt=40 comment (G256 vs 32-elem 34 B) | Comment matches kernel + encoder | minutes |
| **4** | One tiling lever after 3b is green (LDS X panel **or** 128-bit K-stage **or** geom sweep) | Occupancy/ISA dump first; then NRMSE still PASS; microbench GFLOPS up. No tok/s claim | 1–2 days |
| **5** | Encoder + `fp8_wmma` dispatch into one prefill GEMM | Channel cosine on a tiny oracle; daemon still default-off | days |
| **Parked** | FreeToken \(q^\star\) / semantic anchors | Only if a MoE pool does not fit 32 GB | later |
| **Never** | Hyperloom, Lemonade, GGUF F8 layout, `HSA_OVERRIDE` as product | — | — |

hipBLASLt as a large-M prefill fallback is **opt-in, per-shape M-threshold
on hipfire G256 shapes** — do not import The-Rock8’s default 384.

Profiling on this card: `scripts/r9700_profile.sh` already encodes
`FETCH_SIZE = GL2C_EA_RDREQ_sum * 256 / 1024` and the PC-sample path.

---

## URL index (save these, not binaries)

```
https://github.com/AMD-AGI/Hyperloom
https://github.com/AMD-AGI/GEAK
https://github.com/AMD-AGI/Hyperloom/pull/1032
https://rocm.docs.amd.com/projects/hyperloom/en/latest/index.html
https://arxiv.org/abs/2606.12563

https://github.com/The-Monk/The-Rock8
https://github.com/The-Monk/llama.cpp/tree/roc8
https://github.com/The-Monk/rocky-hackathon
https://github.com/ROCm/TheRock
https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md
https://rocm.nightlies.amd.com/v2/gfx120X-all/
https://github.com/ROCm/composable_kernel/pull/3759

https://docs.amd.com/v/u/en-US/rdna4-instruction-set-architecture
https://docs.amd.com/api/khub/documents/uQpkEvk3pv~kfAb2x~j4uw/content
https://gpuopen.com/amd-gpu-architecture-programming-documentation/
https://gpuopen.com/machine-readable-isa/
https://github.com/ROCm/amd_matrix_instruction_calculator

https://arxiv.org/abs/2608.16157
https://arxiv.org/pdf/2608.16157
https://arxiv.org/html/2608.16157v1
https://github.com/FlashML-org/FreeToken
https://flashml.ai
```
