# 2026-08-24 — Qwen3.8-27B MQ4 DFlash 2 vs MTP vs AR on R9700 (protocol-lite)

**Lifecycle: `historical`.** Evidence under the exact fixture and method below.
Not a current default, not an automatic baseline, not an admission decision.

## Why this run exists

Campaign step 2 of [`docs/plans/2026-08-24-r9700-whittle-bandwidth.md`](../plans/2026-08-24-r9700-whittle-bandwidth.md):
matched `hipfire bench` AR / MTP / DFlash 2 on the dense MQ4 parent after
Whittle FFN prefill was exhausted. Serve smoke already showed DFlash 2 firing
(attractor=0; τ=2.94 factual / 6.88 code). This row is the protocol-lite
tok/s + τ matrix on the **same default bench prompt** as the Whittle AR
checkpoints.

## Fixture

| | |
|---|---|
| host | Radeon AI PRO R9700, gfx1201, HIP 7.2, `HIP_VISIBLE_DEVICES=0` |
| hipfire md5 | `d8a4677d382aa36cb8def0ffbe169061` |
| daemon md5 | `51ef5fe234c0ae52c2f0c26d4b3d4127` |
| prompt | default `hipfire bench` string `Explain the theory of general relativity in simple terms.` (no trailing newline) |
| prompt md5 | `d94d3115a3001f08a654d91461d6bdc4` |
| harness | `hipfire bench --runs 3 --warmups 3 --max-tokens 128 --backend noslots --workload stateless --kv-mode q8 -j` |
| env | `nix develop`, `HIPFIRE_DPM_WARMUP_SECS=10`, GPU lock, fresh process per arm |
| trunk | `~/.hipfire/models/qwen38-27b.mq4` md5 `129909ad0fed21dcf72b5b9225e85604` |
| MTP sidecar | `~/.hipfire/models/qwen38-27b.mtp` md5 `4073099a71b57c15e90b9df4723378df` |
| DFlash 2 draft | `~/.hipfire/models/qwen38-27b-dflash2.hfq` md5 `cdf5dd280cb8a33455579b7e82ef0a58` (F16, block 8, arch 20). Filename does not auto-match; pin `HIPFIRE_DFLASH_DRAFT`. |

`--spec off` / `--spec mtp` unset `HIPFIRE_DFLASH_DRAFT`. `--spec dflash` pins
the draft above. Native Qwen MTP generate is **also** gated by
`HIPFIRE_QWEN_MTP=1`; `--spec mtp` alone loads the sidecar and still routes AR.

## Result

| arm | env | decode | prefill | wall | TTFT | τ | VRAM free after load |
|---|---|---:|---:|---:|---:|---:|---:|
| AR `--spec off` | draft unset | 36.6 | 401.0 | 36.0 | 59.8 ms | — | 16292 / 32624 MB |
| MTP `--spec mtp` no opt-in | draft unset | 36.6 | 399.3 | 36.0 | 60.1 ms | (absent) | 16292 / 32624 MB |
| **MTP `--spec mtp`** | **`HIPFIRE_QWEN_MTP=1`** | **47.7** | 330.8 | 46.5 | 72.0 ms | **2.40** | 16292 / 32624 MB |
| DFlash 2 `--spec dflash` | `HIPFIRE_DFLASH_DRAFT` pinned | 37.2 | 260.5 | 36.2 | 92.1 ms | 2.43 | 10480 / 32624 MB |

JSON:

- AR: [`2026-08-24-qwen38-27b-dflash2-vs-mtp-ar-r9700-ar.json`](2026-08-24-qwen38-27b-dflash2-vs-mtp-ar-r9700-ar.json)
- MTP silent AR: [`…-mtp-no-optin.json`](2026-08-24-qwen38-27b-dflash2-vs-mtp-ar-r9700-mtp-no-optin.json)
- MTP live: [`…-mtp.json`](2026-08-24-qwen38-27b-dflash2-vs-mtp-ar-r9700-mtp.json)
- DFlash 2: [`…-dflash.json`](2026-08-24-qwen38-27b-dflash2-vs-mtp-ar-r9700-dflash.json)

Samples (measured 128-token runs):

- AR decode 36.7 / 36.6 / 36.6
- MTP live decode 47.8 / 47.7 / 47.7, τ 2.40 / 2.40 / 2.40, `mtp_windows=53` `ar_windows=0`
- DFlash 2 decode 37.3 / 37.2 / 37.0, τ 2.43 / 2.43 / 2.43, 37 windows. Log:
  `DFlash 2 draft detected (grouped conv + candidate selector); not running as DFlash 1`

Vs AR: MTP **+30.3%** decode. DFlash 2 **+1.6%** decode at almost the same τ,
prefill **−35%**, TTFT **+54%**. Draft VRAM is the 5.8 GB free-VRAM drop.

This default prompt is an explain-prose cell, not the serve `merge_sort` smoke
(second-turn 93.5 tok/s, τ=6.88, max=64, different bytes). Do not collapse
those. Genre-conditional DFlash loss is allowed; attractors were 0 on the
earlier serve smoke.

## Reading

`--spec mtp` without `HIPFIRE_QWEN_MTP=1` is not an MTP measurement. With the
opt-in, native MTP is the spec win on this fixture (matches the older 5/10
MQ4 ~47.8 tok/s / τ=2.40 band). DFlash 2 is wired and accepting (τ≈MTP) but
does not convert τ into tok/s here — draft+verify overhead ate it on this
prompt. Not pager evidence. Do not assume transfer onto Whittle.
