# R9700 bandwidth campaign: dense MQ4 → DFlash 2 → Whittle-MoE → adaptive residency

| Field | Value |
|---|---|
| State | **step 4 measured** (32k FA-tile tax fixed 2026-08-25; pager not indicated) |
| Date | 2026-08-24 |
| Hardware | Radeon AI PRO R9700 (`gfx1201`), 32 GB GDDR6, decode DRAM roofline ~631 GB/s |
| Dense floor | Qwen3.8-27B MQ4 (selected 2026-08-24; not an admission) |
| MoE SKU | `logic65/Qwen3.8-Whittle-MoE-27B-A17.8B` v2.1 |
| Validation | [`docs/VALIDATION.md`](../VALIDATION.md) — serve_harness for generation; redline harness only if kernels/dispatch change; [`docs/methodology/perf-benchmarking.md`](../methodology/perf-benchmarking.md) for tok/s |
| Not | Product default, registry tag, or `admissions.yml` row |

Dense-then-MoE on one 32 GB card. Compute kernels (FP8 GEMV/GEMM, fused overwrite) stay on `r9700-fp8-adaptive`. This campaign targets **HBM bytes/token**, then **HBM+DDR overlap**, not VRAM spill. Step 4 is measured: Whittle AR beats dense MQ4 on serve at `max_seq=32768` after the AR hipGraph FA tile-grid fix ([`2026-08-25-whittle-ar-fa-actual-tiles-r9700.md`](../perf-checkpoints/2026-08-25-whittle-ar-fa-actual-tiles-r9700.md)). Do not start the pager from that row.

Companion briefing: [`docs/investigations/2026-08-23-r9700-fp8-external-refs.md`](../investigations/2026-08-23-r9700-fp8-external-refs.md) § FreeToken. Pager spec: [`docs/specs/2026-07-19-weight-pager-eviction-policy.md`](../specs/2026-07-19-weight-pager-eviction-policy.md). Selection ledger: `.agent-memory/notes/qwen38-27b-r9700-selection-2026-08-24.md`.

---

## Why bandwidth, not VRAM

R9700 decode is memory-bound. Qwen3.8-27B MQ4 already **fits** (~16 GB used at seq 2048 q8). Paging a dense trunk to host would replace HBM with PCIe and lose. The sparsity lever is a model that **does not fetch** idle FFN weights.

Whittle is a post-hoc partition of the same 27B, not a new pretrain. Attention and DeltaNet stay the parent's (64 layers, hidden 5120, 3:1 hybrid). Each FFN of width 17408 is cut into **one shared expert of 5120 plus 64 routed slivers of 192**, top-16. `64×192 + 5120 = 17408`. Per token: 8192 / 17408 ≈ 47% of FFN width → **17.8B active of 27B**. Parameter count is unchanged, so MQ4 VRAM stays in the same band as dense MQ4. Idle 48/64 routed experts occupy capacity but not HBM traffic.

FreeToken ([arXiv:2608.16157](https://arxiv.org/abs/2608.16157), [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken)) is NVIDIA/CUDA. Steal algorithms only. Do not vendor Python, CUDA Graphs, FlashInfer, or their 5090 tok/s. SpecMD already warns that **LRU can lose** to staleness-aware eviction — do not copy “always LRU”.

| FreeToken idea | When it applies here |
|---|---|
| Sparse activation (the model) | Immediate HBM win if 16×192 GEMV overhead does not eat it. |
| Complete-`(layer, expert)` residency | After Whittle AR is green. 4096 slots, ~1.5 MB MQ4 each. Shared 5120 stays pinned. |
| \(q^\star \approx m B_P / B_H\) CPU/PCIe split | Only if pager traces show HBM still saturated and DDR idle. Measure \(B_P,B_H\) on this box. |
| Full-layer prefill double-buffer | Prefill; not the decode-bandwidth claim. |
| Semantic-anchor checkpoints (think / tool / turn) | Dense or MoE. Covers harness **edits** that kill LCP. Independent of dtype. |

`experimental.moe_expert_cache_mb` exists and is unused. `Qwen35Config.paged_experts` exists and defaults off. `WeightPager` v0.1 has a residency map and **no real eviction**.

---

## SKUs

### Dense floor (keep)

- Trunk: `~/.hipfire/models/qwen38-27b.mq4` (md5 `129909ad0fed21dcf72b5b9225e85604`)
- MTP sidecar: `~/.hipfire/models/qwen38-27b.mtp` (hardlinked; do not drop the inode)
- Source: `~/.hipfire/hf-cache/Qwen3.8-27B` until DFlash 2 convert is done and a re-encode is no longer needed
- Pick: MQ4 over FP8 on this card (quality pass both; MQ4 faster, ~12 GB less VRAM). Exploratory, not admission.

### MoE target

- HF: `logic65/Qwen3.8-Whittle-MoE-27B-A17.8B` **v2.1** (router-healed + anti-loop). Not the unhealed first carve.
- Shape vs A3B: **64 / 16 / 192 / 5120**, not 256 / 8 / 512 / 512. hipfire already routes `qwen3_5_moe` as **arch_id 6**; those four fields are config, not A3B hardcodes.
- Quality: research preview. Knowledge battery ~28/39; structured-output loop still ~22% on the author's harness. **Serve against the dense parent on the same prompts before any speed claim.**
- First encode risk: MQ4/FP8 **group-256 along K**. Routed `down` GEMV K = 192 is not a multiple of 256 (`test_moe_grouped_mmq_gfx12` asserts `k % 256 == 0`). Pad-to-256 or a non-G256 format on routed experts is the likely first quant fight. Shared 5120 is fine.

### Spec drafters

- **DFlash 2** (Inco / z-lab, Aug 2026): [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) is a **dense-parent** drafter. Convert via existing `dflash_convert` (arch 20). Measure on dense MQ4 first. Do **not** assume transfer onto Whittle (same mismatch class as 3.6-draft on 3.5).
- Native MTP: already works on dense 27B through thinking. Parent `.mtp` is not a Whittle MTP.
- DFlash 1 + thinking still falls back to AR on this stack.

---

## Sequence

Do not start the pager on dense FP8/MQ4. Do not start fused residual FP8 GEMMs as a substitute for this campaign.

1. **Keep dense MQ4 as the quality/speed floor.** Byte-identical prompts; record md5. Optional: honor `hipfire bench --warmups` on native-generate-v1 (local CLI fix, uncommitted as of this plan).
2. **DFlash 2 on the dense parent.** Convert `Qwen3.8-27B-DFlash2`, `serve_harness` quality, then `hipfire bench --spec dflash` vs MTP vs AR (noslots, stateless, kv q8, fresh process). τ and tok/s both required. Genre-conditional prose loss is allowed; attractors are not.
3. **Download Whittle v2.1 BF16** (exclude GGUF). Encode to arch 6. Fix the 192-K group issue. AR smoke vs parent on the same two-prompt fixture used for 27B selection (`factual` + `merge_sort`).
4. **Bytes/token + tok/s vs dense MQ4.** Same prompt md5. If 16×192 launch overhead ate the sparsity win, the kernel job is **batched expert GEMV** (DS4 E8 lesson), not FreeToken.
5. **`paged_experts` + pager P0 traces** on Whittle (`WeightId::Expert` only; shared stays resident). Caps at 25/50/75% of routed-expert bytes even though the pool fits — that is how you simulate pressure on 32 GB. LRU vs Belady vs Least-Stale. Steal \(q^\star\) only if traces say miss-bound **and** DDR is idle.
6. **A3B / DS4 inherit the pager.** Those pools are where residency fights VRAM. Whittle is the 192-wide fixture that makes traces cheap.

Success for step 4 is a real AR decode lift vs dense MQ4 at matched quality (eyeball + harness), not a paper speedup. Success for step 5 is a trace report that decides policy, including a recorded “LRU is enough, stop” if that is what the data says.

---

## Workstation disk (this R9700 box)

Root filesystem was 88% / ~54 GB free on 2026-08-24 before reclaim. Whittle BF16 is ~parent-sized (~53 GB). Keep enough headroom for the download **plus** one MQ4 encode (~14 GB) **plus** a DFlash 2 drafter.

### Keep

| Path | Why |
|---|---|
| `~/.hipfire/models/qwen38-27b.mq4` | Dense production pick |
| `~/.hipfire/models/qwen38-27b.mtp` | Native MTP sidecar (may be hardlinked) |
| `~/.hipfire/hf-cache/Qwen3.8-27B` | Re-encode / DFlash 2 convert source until those are done |
| Future: Whittle v2.1 BF16 then its MQ4 | MoE campaign |
| Future: `*-dflash*.hfq` for the dense DFlash 2 draft | Spec arm |

### Reclaim (done 2026-08-24)

Stepping-stone FP8 encodes and their HF caches; losing 27B FP8 trunk (re-encode from kept BF16 if needed). Do **not** delete `qwen38-27b.mtp` when dropping the `qwen38-27b-fp8e4m3.mtp` name — same inode.

| Path | Class |
|---|---|
| `~/.hipfire/models/qwen35-0.8b-fp8e4m3.hfq` | Small test; known-incoherent |
| `~/.hipfire/models/qwen35-4b-fp8e4m3.hfq` | Stepping-stone encode |
| `~/.hipfire/hf-cache/Qwen3.5-0.8B` | Source for the test encode |
| `~/.hipfire/hf-cache/Qwen3.5-4B` | Source for the stepping-stone |
| `~/.hipfire/models/qwen38-27b-fp8e4m3.hfq` | Losing dtype after MQ4 pick |
| `~/.hipfire/models/qwen38-27b-fp8e4m3.mtp` | Extra hardlink name only |

Also: user-level `nix-collect-garbage` (6.5 GiB on 2026-08-24). NixOS **system** generations (373 links under `/nix/var/nix/profiles/system-*`) need root:

```bash
sudo nix-collect-garbage --delete-older-than 7d
```

Do not delete `~/work/advait` or hipfire `target/`. Never `rm` `/tmp/hipfire-gpu.lock`.

### Future unused-model rule

After a SKU is encoded **and** a dtype/spec pick is recorded:

1. Drop losing dtypes of that SKU (keep the winner + its sidecars).
2. Drop stepping-stone models (0.8B / 4B class) and their `hf-cache` trees.
3. Drop BF16/`hf-cache` of a SKU only after its MQ4 (or chosen) artifact exists **and** no pending convert still needs the source (DFlash 2, MTP extract, re-quant).
4. Never `rm` a file with `nlink > 1` until every keep-name is checked (`ls -li`).
5. Prefer `nix-collect-garbage` over deleting live `/nix/store` paths by hand. Never `rm` `/tmp/hipfire-gpu.lock`.

Target: ≥80 GB free before starting a ~50 GB HF download.

---

## Out of scope

- Vendoring FreeToken, Lemonade, GGUF F8 layouts, or `HSA_OVERRIDE_GFX_VERSION` as product.
- Treating Whittle v2.1 as a registry/admission candidate before serve_harness vs parent.
- Using DFlash 2 numbers from H200 / SGLang as an R9700 claim.
- Adaptive KV (mid-sequence K/V downshift) as a substitute for expert residency — different lever, already specified elsewhere.
- PFlash.
