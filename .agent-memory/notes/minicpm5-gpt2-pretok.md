---
title: MiniCPM5 GGUF pretok is ignore_merges, not a new arch
date: 2026-09-10
tags: [tokenizer, minicpm5, gpt2-bpe, ignore_merges, llama-cpp]
---

MiniCPM5-2B GGUF is llama-shaped (`arch_id=0`) with `tokenizer.ggml.model=gpt2` and
`tokenizer.ggml.pre=minicpm5`. Hipfire used to ignore `pre` and always byte-seed + BPE,
so `"hi"` split instead of emitting vocab id 7466 (llama.cpp `llama-tokenize`).

llama.cpp: `LLAMA_VOCAB_PRE_TYPE_MINICPM5` → Isolated `\p{N}{1,3}` then main GPT-2 regex
(`\p{N}+`), `byte_encode=true`, **`ignore_merges=true`** (whole encoded chunk in vocab →
skip BPE). `add_bos_token=false`, `bos_token_id=0`.

Fix: `Gpt2PreKind::MiniCpm5` in `crates/hipfire-runtime/src/tokenizer.rs`, read from
`tokenizer.ggml.pre` in `from_gguf` / `from_gguf_meta_json`. MQ4 already copies the field
via `gguf_meta`. Local encode of `"hi"` on `MiniCPM5-2B-Q8_0.mq4` is `[7466]`.

Serve after the port: `"hi"` encodes `[7466]`. Follow-up (same day): register
`<s>` (bos id 0, len 3) as a greedy special — the `len() > 3` heuristic skipped
it so jinja `{{ bos_token }}` BPE-split to two ids. MiniCPM5 jinja is Qwen-shaped:
`enable_thinking is defined` + false emits empty `<think></think>` (Python-file
attractor). Leave that jinja variable undefined when thinking is off.

Remaining: some prompts still emit think-body “user’s request…” into
`reasoning_content` (llama_ar think router / model opening `<think>`). Not a
first-class hipfire arch. Catalog window is 32k.
