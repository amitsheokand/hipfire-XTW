---
title: Reclaimed Whittle MQ4; re-encoded Qwen3.8-27B FP8E4M3
date: 2026-08-25
tags: [r9700, gfx1201, fp8, whittle, disk]
---

Whittle `qwen38-whittle-27b-a17.8.mq4` (16 GB, md5 `bb8dd08b…`) deleted.
Step 4 of the bandwidth campaign was already measured (AR 43.6 at 32k after
FA tiles); pager not indicated. Dense MQ4 trunk kept.

Re-encoded from `~/.hipfire/hf-cache/Qwen3.8-27B`:

```
hipfire-quantize --format fp8e4m3 --uniform --threads 10 \
  --input ~/.hipfire/hf-cache/Qwen3.8-27B \
  --output ~/.hipfire/models/qwen38-27b-fp8e4m3.hfq
```

- size 27354295296 (27354.3 MB written)
- md5 `62820ebc4a7f952be75d25e5bbcb9714`
- skipped 885429488 mtp/visual params (same as prior encode)
- MTP sidecar hardlink of `qwen38-27b.mtp` md5 `4073099a…`

Disk after: 42 GB free. Last FP8 protocol-lite (2026-08-24) was AR 20.6 /
MTP 38.7; not re-benched on this file yet. Do not treat this md5 as the
Aug-24 row unless it matches that artifact.

Related: [[qwen38-27b-r9700-selection-2026-08-24]], [[fp8-e2e-qwen38-27b]].
