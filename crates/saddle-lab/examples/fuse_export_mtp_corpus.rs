// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE in the project root.

//! Export MTP training corpus from the production trunk.
//!
//! For each .ids file (raw u32 LE token ids): prefill to fill KV, capture
//! per-token post-norm hiddens, run the lm_head GEMV per row + GPU argmax
//! for self-distill labels, gather embedding rows. Writes one .npz-like
//! set per prompt: <name>.ids.npy (u32[T]), .hidden.npy (f32[T,H]),
//! .prevemb.npy (f32[T,H]), .labels.npy (u32[T]).
//!
//! Usage: fuse_export_mtp_corpus --model <m.mq4> --ids-dir <ids/>
//!          --out-dir <corpus/> [--max-tokens 768] [--limit N]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::qwen35::{
        self, DeltaNetState, Qwen35Scratch,
    };
    use hipfire_runtime::llama::EmbeddingFormat;
    use hipfire_dispatch::context::DispatchCtx;
    use hipfire_dispatch::pipeline::{execute_steps, GemvInput, Step};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::KvCache;
    use std::io::Write;
    use std::path::{Path, PathBuf};

    fn write_npy(path: &Path, data: &[u8], descr: &str, shape: &[usize]) {
        // Minimal numpy v1.0 writer (C-order, no fortran).
        let shape_s = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let shape_s = if shape.len() == 1 {
            format!("({shape_s},)")
        } else {
            format!("({shape_s})")
        };
        let header = format!(
            "{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_s}, }} Fonte\n"
        );
        // Pad header (incl. 10-byte prefix) to 64-byte alignment.
        let mut header = header.replace(" Fonte\n", "\n");
        let pad = 64 - ((10 + header.len()) % 64);
        header.push_str(&" ".repeat(pad - 1));
        header.push('\n');
        let mut f = std::fs::File::create(path).expect("create npy");
        f.write_all(b"\x93NUMPY").unwrap();
        f.write_all(&[1u8, 0u8]).unwrap();
        f.write_all(&(header.len() as u16).to_le_bytes()).unwrap();
        f.write_all(header.as_bytes()).unwrap();
        f.write_all(data).unwrap();
    }

    fn to_bytes_f32(v: &[f32]) -> Vec<u8> {
        let mut b = Vec::with_capacity(v.len() * 4);
        for x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        b
    }

    fn to_bytes_u32(v: &[u32]) -> Vec<u8> {
        let mut b = Vec::with_capacity(v.len() * 4);
        for x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        b
    }

    let argv: Vec<String> = std::env::args().collect();
    let mut model_path = String::new();
    let mut ids_dir: Option<String> = None;
    let mut out_dir: Option<String> = None;
    let mut max_tokens: usize = 768;
    let mut limit: usize = usize::MAX;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model_path = argv[i + 1].clone();
                i += 2;
            }
            "--ids-dir" => {
                ids_dir = Some(argv[i + 1].clone());
                i += 2;
            }
            "--out-dir" => {
                out_dir = Some(argv[i + 1].clone());
                i += 2;
            }
            "--max-tokens" => {
                max_tokens = argv[i + 1].parse().unwrap();
                i += 2;
            }
            "--limit" => {
                limit = argv[i + 1].parse().unwrap();
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    let (ids_dir, out_dir) = match (ids_dir, out_dir) {
        (Some(a), Some(b)) if !model_path.is_empty() => (a, b),
        _ => {
            eprintln!("Usage: fuse_export_mtp_corpus --model <m.mq4> --ids-dir <ids/> --out-dir <corpus/> [--max-tokens N] [--limit N]");
            std::process::exit(1);
        }
    };
    std::fs::create_dir_all(&out_dir).expect("out dir");

    let mut hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("read config");
    let dim = config.dim;
    let vocab = config.vocab_size;
    eprintln!("model: dim={} layers={} vocab={}", dim, config.n_layers, vocab);
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let weights = {
        let mut src = qwen35::HfqSource::new(&mut hfq, &config);
        let layout = qwen35::Layout::single(config.n_layers);
        qwen35::load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
    }
    .expect("load weights");
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).expect("scratch");

    let mut entries: Vec<PathBuf> = std::fs::read_dir(&ids_dir)
        .expect("read ids dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|e| e == "ids").unwrap_or(false))
        .collect();
    entries.sort();
    let hidden_dim = dim;
    // KV + recurrent state allocated ONCE at max size and reused across
    // prompts (prefill from 0 overwrites all live positions; no allocator
    // churn, no cross-prompt contamination).
    let kv_seq = (max_tokens + 32).max(512);
    let mut kv_cache = KvCache::new_gpu_q8(
        &mut gpu,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        kv_seq,
    )
    .expect("kv");
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).expect("dn");
    let mut done = 0usize;
    for path in entries {
        if done >= limit {
            break;
        }
        let raw = std::fs::read(&path).expect("read ids");
        let mut ids: Vec<u32> = raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        ids.truncate(max_tokens);
        if ids.len() < 8 {
            continue;
        }
        let t = ids.len();
        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        // Reset recurrent state per prompt (KV slots are positionally
        // overwritten by prefill from 0; DN state accumulates and must zero).
        dn_state.reset(&mut gpu).expect("dn reset");
        // Per-token hiddens via prefill capture (KV/state reused, see above).
        let hidden_buf = gpu.alloc_tensor(&[t * hidden_dim], rdna_compute::DType::F32).expect("hidden");
        qwen35::forward_prefill_batch(
            &mut gpu,
            &weights,
            &config,
            &ids,
            0,
            &mut kv_cache,
            &mut dn_state,
            &scratch,
            None,
            Some(&hidden_buf),
            None,
            None,
        )
        .expect("prefill");
        // lm_head per row + GPU argmax labels.
        let logits_row = gpu.alloc_tensor(&[vocab], rdna_compute::DType::F32).expect("logits");
        let mut labels = Vec::with_capacity(t);
        {
            let ctx = DispatchCtx::new(&gpu);
            let wr = weights.output.dispatch_ref();
            for r in 0..t {
                let hrow = hidden_buf.sub_offset(r * hidden_dim, hidden_dim);
                let step = Step::Gemv {
                    w: &wr,
                    input: GemvInput::Raw(&hrow),
                    out: &logits_row,
                };
                execute_steps(&mut gpu, &ctx, &[step]).expect("lm_head");
                labels.push(gpu.argmax_f32(&logits_row, vocab).expect("argmax"));
            }
        }
        // Embedding rows per token.
        let prevemb_buf = gpu.alloc_tensor(&[t * hidden_dim], rdna_compute::DType::F32).expect("prevemb");
        for (r, &tok) in ids.iter().enumerate() {
            let erow = prevemb_buf.sub_offset(r * hidden_dim, hidden_dim);
            match weights.embd_format {
                EmbeddingFormat::HFQ4G256 => {
                    gpu.embedding_lookup_hfq4g256(&weights.token_embd, &erow, tok, dim)
                }
                EmbeddingFormat::HFQ4G128 => {
                    gpu.embedding_lookup_hfq4g128(&weights.token_embd, &erow, tok, dim)
                }
                EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.token_embd, &erow, tok, dim),
                EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.token_embd, &erow, tok, dim),
                _ => panic!("unsupported embedding format"),
            }
            .expect("embed");
        }
        // Download + write.
        let hidden = gpu.download_f32(&hidden_buf).expect("dl hidden");
        let prevemb = gpu.download_f32(&prevemb_buf).expect("dl prevemb");
        let od = PathBuf::from(&out_dir);
        write_npy(&od.join(format!("{stem}.ids.npy")), &to_bytes_u32(&ids), "<u4", &[t]);
        write_npy(
            &od.join(format!("{stem}.hidden.npy")),
            &to_bytes_f32(&hidden),
            "<f4",
            &[t, hidden_dim],
        );
        write_npy(
            &od.join(format!("{stem}.prevemb.npy")),
            &to_bytes_f32(&prevemb),
            "<f4",
            &[t, hidden_dim],
        );
        write_npy(
            &od.join(format!("{stem}.labels.npy")),
            &to_bytes_u32(&labels),
            "<u4",
            &[t],
        );
        gpu.free_tensor(hidden_buf).ok();
        gpu.free_tensor(logits_row).ok();
        gpu.free_tensor(prevemb_buf).ok();
        done += 1;
        if done % 10 == 0 {
            eprintln!("exported {done} prompts");
        }
    }
    eprintln!("done: {done} prompts -> {out_dir}");
}
