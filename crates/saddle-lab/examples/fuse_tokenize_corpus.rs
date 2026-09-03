// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE in the project root.

//! Tokenize a corpus directory with the PRODUCTION GGUF tokenizer.
//!
//! Uses hipfire-runtime's Tokenizer (the same encoder the daemon runs),
//! so MTP training corpora match production tokenization exactly.
//! Reads *.txt from --input-dir, writes <name>.ids (raw u32 LE token ids)
//! to --out-dir, skipping files that encode to fewer than --min-tokens.
//!
//! Usage: fuse_tokenize_corpus --gguf <model.gguf> --input-dir <txt/>
//!          --out-dir <ids/> [--min-tokens 50]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_runtime::gguf::GgufFile;
    use hipfire_runtime::tokenizer::Tokenizer;
    use std::io::Write;
    use std::path::PathBuf;

    let argv: Vec<String> = std::env::args().collect();
    let mut gguf = String::from("/tmp/Fuse-2-MoE-BF16.gguf");
    let mut input_dir: Option<String> = None;
    let mut out_dir: Option<String> = None;
    let mut min_tokens: usize = 50;
    let mut decode_ids: Option<String> = None;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--gguf" => {
                gguf = argv[i + 1].clone();
                i += 2;
            }
            "--decode-ids" => {
                decode_ids = Some(argv[i + 1].clone());
                i += 2;
            }
            "--input-dir" => {
                input_dir = Some(argv[i + 1].clone());
                i += 2;
            }
            "--out-dir" => {
                out_dir = Some(argv[i + 1].clone());
                i += 2;
            }
            "--min-tokens" => {
                min_tokens = argv[i + 1].parse().unwrap();
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    let file = GgufFile::open(std::path::Path::new(&gguf)).expect("open gguf");
    let tok = Tokenizer::from_gguf(&file).expect("gguf tokenizer");
    eprintln!("vocab={} eos?", tok.vocab_size());
    if let Some(ids_s) = decode_ids {
        let ids: Vec<u32> = ids_s
            .split(',')
            .map(|s| s.trim().parse::<u32>().expect("id"))
            .collect();
        println!("{:?} -> {:?}", ids, tok.decode(&ids));
        return;
    }
    let (input_dir, out_dir) = match (input_dir, out_dir) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            eprintln!("Usage: fuse_tokenize_corpus --gguf <m.gguf> --input-dir <txt/> --out-dir <ids/> [--min-tokens N]");
            std::process::exit(1);
        }
    };
    std::fs::create_dir_all(&out_dir).expect("out dir");
    let mut n_ok = 0usize;
    let mut n_skip = 0usize;
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&input_dir)
        .expect("read input dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|e| e == "txt").unwrap_or(false))
        .collect();
    entries.sort();
    for path in entries {
        let text = std::fs::read_to_string(&path).unwrap_or_default();
        let ids = tok.encode(&text);
        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        if ids.len() < min_tokens {
            n_skip += 1;
            continue;
        }
        let out_path = PathBuf::from(&out_dir).join(format!("{stem}.ids"));
        let mut f = std::fs::File::create(&out_path).expect("create ids");
        for id in &ids {
            f.write_all(&id.to_le_bytes()).expect("write ids");
        }
        n_ok += 1;
    }
    eprintln!("tokenized: {n_ok} kept, {n_skip} skipped (<{min_tokens} tok)");
}
