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

Serve after the port: the Python-file attractor is gone. Remaining (not tokenizer):
Jinja `{{- bos_token }}` + tools XML in the GGUF chat template; `llama_ar` fake
`reasoning_content` / “the user’s request…” scaffold; `:8080` tools on llama_ar.
Not a first-class hipfire arch. Do not advertise 128k; catalog is 32k.
