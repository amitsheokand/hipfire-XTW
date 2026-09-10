# Receipt: T-hipfire-fuse-qwen-bench

**Task:** Fuse vs Qwen throughput + quality snapshot on gfx1201 R9700.
**Worktree:** `/home/amitsheokand/dev/hipfire-worktrees/fuse-qwen-bench`,
branch `work/fuse-qwen-bench` @ `60ebfd4d`. **Date:** 2026-09-10.
**GPU lock:** `T-hipfire-fuse-qwen-bench`. Qwen systemd serve on `:11435`
stopped before probes; **serve left down** (coordinator restores Qwen).
No engine code changed. No push.

## Identity

| field | value |
|---|---|
| GPU | AMD Radeon RX 9070 XT (gfx1201), HIP 7.2.53211, 32624 MB VRAM |
| Branch / commit | `work/fuse-qwen-bench` @ `60ebfd4d25ce139fa2089f9d85c28df5bea4cef5` |
| `hipfire` md5 | `c0baa4dd3c28f860bbdcbe9b435ee3f2` (`target/release/hipfire`) |
| `daemon` md5 | `ac1db2ea5a2910827f56de9b3b74a8fa` (`target/release/daemon`) |
| Fuse artifact | `~/.hipfire/models/fuse-2-moe.mq4` md5 `dd4e34b1d3f99263580181af92ae5d7e` |
| Qwen artifact | `~/.hipfire/models/qwen3.8-27b.mq4-pro` md5 `279c563786499c6651de6a3a57b42b02` (tag `qwen3.8:27b-mq4-pro`) |
| KV / backend / workload | `q8`, `noslots`, `stateless` |
| Spec | `HIPFIRE_DFLASH_MODE=off`, `HIPFIRE_DFLASH_DRAFT` unset |
| ROCm JIT | `HIPFIRE_HIPCC_EXTRA_FLAGS=--rocm-device-lib-path=/nix/store/8jdkas7zs4yzqs0n3xrrnnmc9s3xx5zb-rocm-device-libs-22.0.0-rocm/amdgcn/bitcode` |
| Prompts | `fuse_prod_sky.txt` md5 `e529c2204a2531d836d5ae2a4b755228`; `humaneval_3_below_zero.txt` md5 `37c5aad9f9efe93b5c47f27256bdf149` |

## Bench (`hipfire bench`, spec off, greedy, 5 runs × 3 fresh processes)

Decode tok/s — median of per-process medians:

| model | thinking | prompt md5 (short) | decode tok/s | prefill tok/s | TTFT ms |
|---|---|---|---:|---:|---:|
| Fuse | off | `e529c220…` (sky) | **117.8** | 584.7 | 20.5 |
| Fuse | off | `37c5aad9…` (code) | **117.7** | 584.3 | 20.5 |
| Qwen | off | `e529c220…` (sky) | **33.0** | 649.0 | 265.0 |
| Qwen | off | `37c5aad9…` (code) | **33.0** | 644.9 | 266.7 |
| Qwen | on (`--reasoning-on`) | `e529c220…` (sky) | **32.9** | 639.9 | 267.2 |

Per-process decode medians (all 5-run cells):

- Fuse sky: 117.8, 117.7, 118.1
- Fuse code: 117.7, 117.6, 117.9
- Qwen sky off: 33.0, 33.0, 33.0
- Qwen code off: 33.0, 33.0, 33.0
- Qwen sky on: 33.0, 32.9, 32.9

**Headline:** Fuse AR decode is ~**3.6×** Qwen on this fixture (117.7 vs 33.0
tok/s, thinking off). Qwen `--reasoning-on` does not move decode on the sky
cell (32.9 vs 33.0). Fuse prefill is slightly lower tok/s but far lower TTFT
(~20 ms vs ~266 ms) because the MoE prompt is shorter after framing.

Raw JSON: `/tmp/fuse-qwen-bench/bench_*.json`.

## Serve (`serve_harness.py`, thinking/spec off, greedy, q8/contiguous)

`HIPFIRE_DPM_WARMUP_SECS=10`, `HIPFIRE_SERVE_ALLOW_INCOHERENT=1` (Fuse
attractors are a known model property, not a harness failure — see
`T-hipfire-fuse-prod`).

| model | mode | avg decode | avg prefill | quality notes |
|---|---|---:|---:|---|
| Fuse | battery | **113.6** | 1375.6 | code/factual on-rails; reason/instruct **runaway+attractor** (2/5); prose meta-chatter ("The user wants…") |
| Fuse | chain | **112.8** | 1247.5 | prefix-cache hits (cached=302 on t4); turns 3–5 **attractor** from context poisoning |
| Qwen | battery | **32.9** | 562.5 | all 5 genres clean; code returns proper `merge_sorted` with docstring; math/story/list coherent |
| Qwen | chain | **32.9** | 559.6 | coherent across turns; prefix reuse (cached=241) |

Serve decode tracks bench within ~3% for Fuse (113.6 serve vs 117.8 bench) and
matches Qwen bench exactly (32.9). Bench is **not** serve authority — both
reported.

Qwen with `--thinking off` still emits think-span tokens on some battery turns
(e.g. code: think 39 / ans 46 words) — native Qwen3.8 template behavior, not
Fuse-class meta-chatter. No attractors or runaways on Qwen.

Raw JSON: `/tmp/fuse-qwen-bench/serve_*.json`.

## Eyeball summary

- **Throughput:** Fuse wins decisively on AR greedy q8 (~117 vs ~33 tok/s).
  Not a product-floor claim — snapshot under fixed fixtures on gfx1201.
- **Quality:** Qwen is the clear chat winner on the genre battery. Fuse
  `code`/`factual` are usable; `reason`/`instruct` degenerate; `prose`/
  chained turns show meta-chatter and attractors (consistent with
  `T-hipfire-fuse-prod`).
- **Fair A/B caveats:** different arch (MoE k=2 vs dense 64L), different
  framing contracts, different parameter counts. Compare only at recorded md5s.

## Checks run

- `cargo build --release -p hipfire-cli -p hipfire-daemon` (nix shell) — clean.
- `hipfire bench` 15 cells (5 model×prompt/thinking combos × 3 fresh processes).
- `serve_harness.py` battery + chain × 2 models.
- No kernel/dispatch change → no redline route.
- `:11435` verified empty at end.

## Files touched

- `docs/receipts/T-hipfire-fuse-qwen-bench.md` (this file)
- `docs/perf-checkpoints/2026-09-10-fuse-vs-qwen-r9700.md`

`PACKET.md` / `run-packet.sh` (coordinator-provided) left untracked.
