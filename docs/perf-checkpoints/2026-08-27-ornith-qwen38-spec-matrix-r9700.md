# 2026-08-27 — Ornith 1.5 35B-A3B + Qwen3.8-27B spec matrix on R9700

**Lifecycle:** `historical`. Evidence under the exact fixture and method below.
Not a current default, not an automatic baseline, not an admission decision.

## Why this run exists

Matched `hipfire bench` AR / MTP / DFlash 2 matrix after Ornith qt44 MTP load
and HTTP speculation work (`f40796d8`). Serve-path tests ran first
(`scripts/serve_harness.py`); timed cells are the synthetic daemon probe
(`hipfire bench`), not production HTTP authority.

## Identity


| | |
|---|---|
| host | Radeon AI PRO R9700, gfx1201, HIP 7.2.53211, `HIP_VISIBLE_DEVICES=0` |
| source | `tmp/ornith-pr610` @ `f40796d841ea722f5b220423eeaae0c08f9cf0a9` (dirty: untracked `.commandcode/`, this checkpoint) |
| hipfire md5 | `551c229b3c06874dcfa10fa2f9b0158e` (`target/release/hipfire`, rebuilt 2026-08-27) |
| daemon md5 | `ac20666040e01f998afe4e516509f38b` (`target/release/daemon`) |
| harness | `hipfire bench --runs 5 --warmups 3 --max-tokens 128 --backend noslots --workload stateless --kv-mode q8 -j` |
| env | `nix develop`, `HIPFIRE_DPM_WARMUP_SECS=10`, GPU lock, fresh process per cell |
| sampler (bench) | `temperature=0`, `top_p=1`, `repeat_penalty=1.1`, **thinking off**, `reasoning_effort=none`, `max_think_tokens=1`, `assistant_prefix=closed_think` |
| thinking (all timed cells) | **off** — no `<think>` span; answer-only 128-token cap |
| allocated context | `memory.max_seq=262144` from `~/.hipfire/config.toml` (not filled; prompts were short) |
| KV | q8. Qwen timed cells: **vmm**, `physical_cap=262144`. Ornith timed cells: q8 contiguous-style log (10/40 KV layers). Graph verify left at product default. |
| Ornith trunk | `~/.hipfire/models/ornith-1.5-35b-a3b.mq4` md5 `8f347584806e86a5cd7751a5491a2122` (arch `qwen3_5_moe`) |
| Ornith MTP | `~/.hipfire/models/ornith-1.5-35b-a3b.mtp` md5 `0cf41c5a36430c3570c509643db102b5` |
| Qwen trunk | `~/.hipfire/models/qwen38-27b.mq4` md5 `129909ad0fed21dcf72b5b9225e85604` (arch `qwen3_5`) |
| Qwen MTP | `~/.hipfire/models/qwen38-27b.mtp` md5 `4073099a71b57c15e90b9df4723378df` |
| DFlash 2 draft | `~/.hipfire/models/qwen38-27b-dflash2.hfq` md5 `cdf5dd280cb8a33455579b7e82ef0a58` (F16, declared_B=8 runtime_B=16, windowed W=2048 from draft metadata; filename does not auto-match) |

`--spec off` / `--spec mtp` unset `HIPFIRE_DFLASH_DRAFT` and set `HIPFIRE_QWEN_MTP=0` / `=1` respectively.
`--spec dflash` pins the draft and sets `HIPFIRE_QWEN_MTP=0`.

### Prompts (byte-identical)

| id | bytes | md5 |
|---|---|---|
| default | `hipfire bench` string `Explain the theory of general relativity in simple terms.` (no trailing newline) | `d94d3115a3001f08a654d91461d6bdc4` |
| humaneval | `benchmarks/prompts/humaneval_3_below_zero.txt` | `37c5aad9f9efe93b5c47f27256bdf149` |

Warmups are `Hello` @ 16 tokens, **not** the measured prompt. First measured sample of a new prompt shape can still carry JIT/DPM (visible on humaneval AR prefill/TTFT and DFlash humaneval decode 99.1 → 153). Report **median of the 5 measured runs**.

## Thinking and context (every cell)

These tok/s numbers are **thinking-off, short filled context**. They do not measure forge/anvil (thinking on, medium/xhigh) or decode at a filled 64K/262K window.

| surface | thinking | effort | gen cap (`max_tokens`) | allocated `max_seq` | filled prompt ctx | KV |
|---|---|---|---:|---:|---|---|
| Timed Ornith × 4 | off (`max_think_tokens=1`, `closed_think`) | none | 128 | **262144** (user config) | default ~dozen tokens; humaneval short code file | q8, 10/40 layers |
| Timed Qwen × 6 | off (same) | none | 128 | **262144** (`physical_cap=262144` in log) | default short; humaneval **129** tokens (serve confirm) | q8 **vmm**, 16/64 layers; DFlash draft window W=2048 |
| Harness Ornith AR/MTP | `--thinking off --thinking-effort none` | none | **64** | **32768** (isolated harness default) | battery ctx 25–55 | q8 contiguous default |
| Harness Qwen DFlash humaneval | off / none | none | **128** | **262144** (tag-policy `qwen3.8:27b`) | **129** | q8 vmm |

Production catalog (nixos `agent-profiles.nix`) was **not** this fixture:

| lane | thinking | effort | advertised context | `max_tokens` | spec on default Ornith | spec if swapped to Qwen |
|---|---|---|---:|---:|---|---|
| forge (daily) | **on** | medium | 262144 | 32768 | off (AR) | off (AR) |
| anvil | **on** | xhigh | 262144 | 32768 | off (AR) | off (AR) |
| feather | **off** (matches matrix) | none / greedy | **65536** | 8192 | MTP (`dflash-if-capable`) | DFlash 2 |

## Serve-path tests (before timed matrix)

First attempt used a positional `battery` argument and argparse-exited 2 without loading. Correct invocation: `--mode battery`. Re-run after the timed matrix (same binaries, GPU lock, systemd `hipfire-serve` stopped).

| cell | command gist | result |
|---|---|---|
| Ornith AR | `--speculation off --thinking off --thinking-effort none --sampling greedy --max-tokens 64 --kv q8` built-in 5-genre battery | exit 0; turns=5; **attractor=0**; empty=0; `finish=length` (cap) flagged runaway; avg decode 121.2 tok/s |
| Ornith MTP | same + `--speculation mtp --mtp on` `HIPFIRE_QWEN_MTP=1` | exit 0; attractor=0; tau 2.74 / 2.52 / 2.10 / 1.75 / 1.91 by genre; avg decode 128.5 tok/s |
| Qwen DFlash humaneval | `--speculation dflash --dflash on --draft …dflash2.hfq --prompt-file humaneval_3_below_zero.txt --max-tokens 128` | exit 0; attractor=0; **τ=9.58**; decode 152.3 tok/s; preview is a real `below_zero` Python body, not a single-token loop |

Harness JSON/logs: [`2026-08-27-ornith-qwen38-spec-matrix-r9700-logs/`](2026-08-27-ornith-qwen38-spec-matrix-r9700-logs/).
Qwen harness used tag-policy `kv_backend=vmm` and `max_seq=262144`, matching Qwen timed cells. Ornith harness allocated 32768; Ornith timed cells inherited user `max_seq=262144`. Do not A/B harness decode against bench.

## Timed matrix (median of 5 measured 128-token runs)


| model | prompt | spec | env | decode | vs AR | prefill | wall | TTFT | τ | windows (measured) | VRAM free after load |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Ornith 35B-A3B | default | AR `--spec off` | draft unset, `HIPFIRE_QWEN_MTP=0` | 57.1 | — | 612.2 | 56.1 | 39.2 ms | — | — | 6644 / 32624 MB |
| Ornith 35B-A3B | default | MTP `--spec mtp` | `HIPFIRE_QWEN_MTP=1` | **112.4** | **+96.8%** | 718.7 | 109.2 | 33.0 ms | **1.92** | `mtp_windows=66` `ar_windows=0` | 6072 / 32624 MB |
| Ornith 35B-A3B | humaneval | AR | as AR row | 119.8 | — | 2169.9† | 113.5† | 59.4 ms† | — | — | 6644 / 32624 MB |
| Ornith 35B-A3B | humaneval | MTP | `HIPFIRE_QWEN_MTP=1` | **140.7** | **+17.4%** | 1276.4 | 126.6 | 100.0 ms | **2.54** | `mtp_windows=50` | 6072 / 32624 MB |
| Qwen3.8-27B | default | AR | draft unset, `HIPFIRE_QWEN_MTP=0` | 36.6 | — | 395.7 | 36.0 | 60.6 ms | — | — | 16862 / 32624 MB |
| Qwen3.8-27B | default | MTP | `HIPFIRE_QWEN_MTP=1` | **51.7** | **+41.3%** | 326.5 | 50.2 | 73.0 ms | **2.59** | `mtp_windows=49` | 16640 / 32624 MB |
| Qwen3.8-27B | default | DFlash 2 `--spec dflash` | `HIPFIRE_DFLASH_DRAFT` pinned | 49.2 | +34.4% | 261.4 | 47.6 | 91.8 ms | 2.26 | 38–39 windows | 12578 / 32624 MB |
| Qwen3.8-27B | humaneval | AR | as AR row | 36.4 | — | 580.7† | 34.2† | 222.1 ms† | — | — | 16862 / 32624 MB |
| Qwen3.8-27B | humaneval | MTP | `HIPFIRE_QWEN_MTP=1` | 55.0 | +51.1% | 425.1 | 48.6 | 303.0 ms | 2.76 | `mtp_windows=46` | 16640 / 32624 MB |
| Qwen3.8-27B | humaneval | DFlash 2 | draft pinned | **153.7** | **+322%** | 479.3 | 116.2 | 269.2 ms | **9.58** | 12 windows | 12578 / 32624 MB |

† First measured sample still cold (Hello-warmup does not cover this prompt). Decode medians are stable; do not cite the prefill/TTFT mean.

Decode samples:

- Ornith AR default: 57.8 / 57.3 / 57.1 / 57.1 / 57.1
- Ornith MTP default: 112.3 / 112.3 / 112.5 / 112.5 / 112.4; τ 1.92 × 5
- Ornith AR humaneval: 117.4 / 119.2 / 119.8 / 120.1 / 120.2
- Ornith MTP humaneval: 140.6 / 140.8 / 140.7 / 140.7 / 140.7; τ 2.54 × 5
- Qwen AR default: 36.6 × 5
- Qwen MTP default: 51.7 / 51.8 / 51.7 / 51.8 / 51.7; τ 2.59 × 5
- Qwen DFlash default: 50.5 / 49.2 / 49.2 / 49.2 / 49.2; τ 2.34 / 2.26 × 4
- Qwen AR humaneval: 36.4 / 36.4 / 36.4 / 36.4 / 36.3
- Qwen MTP humaneval: 55.0 / 55.0 / 54.9 / 55.0 / 54.9; τ 2.76 × 5
- Qwen DFlash humaneval: 99.1 / 153.5 / 154.1 / 153.7 / 153.8; τ 9.58 × 5 (first measured is the JIT cell; median 153.7)

JSON siblings: [`…-ornith-ar-default.json`](2026-08-27-ornith-qwen38-spec-matrix-r9700-ornith-ar-default.json) and the nine matching `…-{model}-{spec}-{prompt}.json` files.

DFlash 2 log (both Qwen DFlash cells): `DFlash 2 draft detected (grouped conv + candidate selector); not running as DFlash 1`; windowed W=2048 from draft metadata.

## Reading

- **Measured, not admitted.** Ornith has no DFlash draft in this matrix (catalog: spec `off`/`mtp` only).
- Qwen default AR **36.6 tok/s** matches the 2026-08-24 protocol-lite AR cell on the same trunk md5. MTP here is **51.7** vs that row’s **47.7** (different binaries). DFlash default **49.2** vs that row’s **37.2** — draft now logs windowed W=2048; do not treat as a same-fixture delta.
- Ornith AR decode is **prompt-dependent** on this MoE (57 tok/s default prose vs 120 tok/s humaneval). MTP’s relative win is large on prose (+97%) and modest on humaneval (+17%) because the AR baseline already moved.
- Qwen DFlash τ=9.58 on humaneval with stdev 0 is greedy-deterministic, not a hidden attractor: serve preview is a `below_zero` implementation and attractor=0. Same τ on the serve path (152.3 tok/s / τ=9.58).
- `hipfire bench` is **not** production HTTP authority. Production-path decode lives in the harness `done` fields above.

## Disposition

Keep as a dated R9700 spec×model×prompt ledger. Not a product floor, not a DFlash-on-Ornith claim, not pager evidence.
