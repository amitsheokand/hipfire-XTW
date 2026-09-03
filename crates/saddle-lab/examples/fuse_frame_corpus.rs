// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE in the project root.

//! Frame raw corpus texts through the model's own Jinja chat template so the
//! MTP training corpus matches the live serve distribution (the spec path
//! always Jinja-frames, even for `--raw` requests).
//!
//! Usage: fuse_frame_corpus --model <m.mq4> --input-dir <txt/> --out-dir <framed-txt/>
//!          [--emit-ids <ids/>] [--enable-thinking] [--verify-ids <csv>]
//!
//! `--emit-ids` additionally writes raw u32-LE `<stem>.ids` token files for
//! the framed texts (direct input to fuse_export_mtp_corpus --ids-dir).
//! `--verify-ids` checks one framed output against a known-good id list
//! (e.g. from a TEMP-MTP-TRACE prefill dump) and fails closed on mismatch.

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::prompt_frame::JinjaChatFrame;
use std::path::{Path, PathBuf};

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let mut model = String::new();
    let mut input_dir: Option<String> = None;
    let mut out_dir: Option<String> = None;
    let mut enable_thinking = false;
    let mut emit_ids: Option<String> = None;
    let mut verify_ids: Option<String> = None;
    let mut verify_name: Option<String> = None;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = argv[i + 1].clone();
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
            "--emit-ids" => {
                emit_ids = Some(argv[i + 1].clone());
                i += 2;
            }
            "--enable-thinking" => {
                enable_thinking = true;
                i += 1;
            }
            "--verify-ids" => {
                verify_ids = Some(argv[i + 1].clone());
                i += 2;
            }
            "--verify-name" => {
                verify_name = Some(argv[i + 1].clone());
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    let (input_dir, out_dir) = match (input_dir, out_dir) {
        (Some(a), Some(b)) if !model.is_empty() => (a, b),
        _ => {
            eprintln!("Usage: fuse_frame_corpus --model <m.mq4> --input-dir <txt/> --out-dir <framed/> [--enable-thinking] [--verify-ids <csv> --verify-name <stem>]");
            std::process::exit(1);
        }
    };
    let mut hfq = HfqFile::open(Path::new(&model)).expect("open model");
    let template = hfq.chat_template().expect("model carries no chat_template");
    let tok = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer");
    std::fs::create_dir_all(&out_dir).expect("out dir");
    if let Some(ids_dir) = emit_ids.as_ref() {
        std::fs::create_dir_all(ids_dir).expect("ids dir");
    }
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&input_dir)
        .expect("read input dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|e| e == "txt").unwrap_or(false))
        .collect();
    entries.sort();
    let mut n = 0usize;
    for path in entries {
        let text = std::fs::read_to_string(&path).expect("read txt");
        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        let frame = JinjaChatFrame {
            tokenizer: &tok,
            template: &template,
            system: None,
            user: text.trim(),
            enable_thinking,
            bos_token: None,
            reasoning_strength: None,
            reasoning_effort: None,
        };
        let rendered = frame.render().expect("jinja render");
        let ids = tok.encode(&rendered);
        if let (Some(ids_csv), Some(vname)) = (verify_ids.as_ref(), verify_name.as_ref()) {
            if &stem == vname {
                let expect: Vec<u32> = ids_csv
                    .split(',')
                    .map(|s| s.trim().parse::<u32>().expect("id"))
                    .collect();
                if ids != expect {
                    eprintln!("VERIFY FAIL {stem}: got {ids:?}");
                    std::process::exit(2);
                }
                eprintln!("verify {stem}: {n} framed ids match live trace", n = ids.len());
            }
        }
        if let Some(ids_dir) = emit_ids.as_ref() {
            let bytes: Vec<u8> = ids.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(PathBuf::from(ids_dir).join(format!("{stem}.ids")), &bytes)
                .expect("write ids");
        }
        std::fs::write(PathBuf::from(&out_dir).join(format!("{stem}.txt")), &rendered)
            .expect("write framed");
        n += 1;
    }
    eprintln!("framed {n} prompts -> {out_dir}");
}
