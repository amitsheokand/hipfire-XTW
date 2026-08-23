// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 amitsheokand
// hipfire — see LICENSE and NOTICE in the project root.

//! Correctness test for the gfx1201 block-scaled FP8 E4M3 WMMA GEMM (prefill).
//!
//! Compares `gemm_fp8e4m3_g256_wmma_gfx1201` (native
//! `v_wmma_f32_16x16x16_fp8_fp8`) against a CPU FP32 reference across
//! Qwen prefill shapes. Reports NRMSE and effective bandwidth.
//!
//! Skips silently on non-gfx1201 archs.
//!
//! Run:
//!   cargo run --release -p rdna-compute --example test_gemm_fp8e4m3_g256 --features lab
//!   HIPFIRE_FP8_GEMM_QUICK=1 cargo run --release -p rdna-compute --example test_gemm_fp8e4m3_g256 --features lab

use rdna_compute::{DType, Gpu};
use std::time::Instant;

const GROUP: usize = 256;

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    if arch != "gfx1201" {
        eprintln!("=== SKIP === FP8 E4M3 GEMM is gfx1201-only (arch={arch})");
        return;
    }
    eprintln!("=== gemm_fp8e4m3_g256_wmma_gfx1201 vs CPU FP32 reference ===");
    eprintln!("  arch={arch}");

    // Small tiles first so a C-map / launch fault fails in seconds, then
    // Qwen prefill shapes. CPU ref is O(N*M*K) and the large cells take minutes.
    // HIPFIRE_FP8_GEMM_QUICK=1 keeps only the three small tiles.
    // Shapes: [N batch, M out, K in], K padded to 256.
    let quick = std::env::var("HIPFIRE_FP8_GEMM_QUICK").ok().as_deref() == Some("1");
    let shapes: Vec<(usize, usize, usize, &str)> = vec![
        (16, 16, 256, "oracle    N=16   M=16   K=256"),
        (32, 64, 256, "tile      N=32   M=64   K=256"),
        (128, 128, 512, "small     N=128  M=128  K=512"),
        (2048, 4096, 2048, "gate_up   N=2048 M=4096 K=2048"),
        (4096, 2048, 11008, "w_down    N=4096 M=2048 K=11008"),
        (512, 2048, 2048, "qkv       N=512  M=2048 K=2048"),
        (128, 11008, 2048, "ffn1      N=128  M=11008 K=2048"),
        (2048, 2048, 2048, "attn      N=2048 M=2048 K=2048"),
    ];
    let shapes: Vec<(usize, usize, usize, &str)> = if quick {
        shapes.into_iter().take(3).collect()
    } else {
        shapes
    };

    let trials = 20;
    let warmup = 3;

    let mut all_pass = true;
    for (n, m, k, label) in &shapes {
        let (n, m, k) = (*n, *m, *k);
        let n_blocks = k / GROUP;
        let scale_padded = ((n_blocks + 15) >> 4) << 4;
        let row_bytes = 16 + scale_padded * 2 + n_blocks * GROUP;
        let total_w_bytes = m * row_bytes;

        // Synthesize deterministic block-scaled FP8 weights + F32 X.
        let (w_host, x_host, y_cpu) = synth_reference(m, k, n, 0xAA00 | (m as u64) ^ (k as u64));

        let w = gpu.upload_raw(&w_host, &[total_w_bytes]).unwrap();
        let x = gpu.alloc_tensor(&[n, k], DType::F32).unwrap();
        let y = gpu.alloc_tensor(&[n, m], DType::F32).unwrap();
        gpu.hip
            .memcpy_htod(&x.buf, unsafe {
                std::slice::from_raw_parts(x_host.as_ptr() as *const u8, n * k * 4)
            })
            .unwrap();

        // Warmup (JIT-compiles the kernel on first call).
        for _ in 0..warmup {
            gpu.try_gfx1201()
                .unwrap()
                .fp8_gemm_e4m3_g256(&w, &x, &y, m, k, n)
                .unwrap();
        }
        gpu.hip.device_synchronize().unwrap();

        // Time.
        let t = Instant::now();
        for _ in 0..trials {
            gpu.try_gfx1201()
                .unwrap()
                .fp8_gemm_e4m3_g256(&w, &x, &y, m, k, n)
                .unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let us = t.elapsed().as_secs_f64() * 1e6 / trials as f64;
        let mflops = 2.0 * m as f64 * n as f64 * k as f64 / (us * 1e-6) / 1e6;

        // Correctness vs CPU FP32 reference.
        let y_gpu = gpu.download_f32(&y).unwrap();
        let mut max_abs = 0.0f64;
        let mut max_abs_ref = 0.0f64;
        let mut sum_sq_err = 0.0f64;
        let mut sum_sq_ref = 0.0f64;
        for i in 0..n * m {
            let r = y_cpu[i] as f64;
            let g = y_gpu[i] as f64;
            let abs = (r - g).abs();
            if abs > max_abs {
                max_abs = abs;
            }
            if r.abs() > max_abs_ref {
                max_abs_ref = r.abs();
            }
            sum_sq_err += (r - g) * (r - g);
            sum_sq_ref += r * r;
        }
        let nrmse = (sum_sq_err / sum_sq_ref.max(1e-30)).sqrt();
        // F32->E4M3 activation quantization: ~5% relative tolerance.
        let tol_abs = 0.05 * max_abs_ref.max(1e-3);
        let bad: usize = (0..n * m)
            .filter(|&i| ((y_cpu[i] as f64) - (y_gpu[i] as f64)).abs() > tol_abs)
            .count();
        if bad > 0 {
            eprintln!(
                "  {label:32}  FAIL {bad}/{}  max_abs={max_abs:.3e} tol={tol_abs:.3e} max|y|={max_abs_ref:.3e} NRMSE={nrmse:.3e}  {us:7.2}µs {mflops:8.0} MFLOPS",
                n * m
            );
            for probe in &[0usize, 1, m, m + 1, 8, 16, 17, 31] {
                if *probe < n * m {
                    eprintln!(
                        "    y_cpu[{}]={:.4} y_gpu[{}]={:.4} y_gpu[{}*M]={:.4}",
                        probe, y_cpu[*probe], probe, y_gpu[*probe], probe, y_gpu[*probe * m % (n * m)]
                    );
                }
            }
            all_pass = false;
        } else {
            eprintln!(
                "  {label:32}  OK  NRMSE={nrmse:.3e}  {us:7.2}µs {mflops:8.0} MFLOPS",
            );
        }
    }

    if !all_pass {
        eprintln!("\n=== FAIL ===");
        std::process::exit(1);
    }
    eprintln!("\n=== ALL PASS ===");
}

fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((b >> 3) & 0xF) as i32;
    let man = (b & 0x7) as f32;
    let v = if exp == 0 {
        man / 8.0 * 2.0f32.powi(-6)
    } else {
        (1.0 + man / 8.0) * 2.0f32.powi(exp - 7)
    };
    sign * v
}

fn f32_to_e4m3(v: f32) -> u8 {
    let neg = v.is_sign_negative();
    let a = v.abs();
    let mut exp = 0i32;
    let mut mant = 0u8;
    if a >= 2.0f32.powi(-6) {
        let mut e = 0i32;
        while (2.0f32).powi(e - 7) * 1.125 <= a {
            e += 1;
        }
        e -= 1;
        exp = e.clamp(0, 15);
        let scale = 2.0f32.powi(exp - 7);
        let m = ((a / scale - 1.0) * 8.0).round() as i32;
        mant = m.clamp(0, 7) as u8;
    }
    let raw = ((exp as u8) << 3) | mant;
    if neg {
        raw | 0x80
    } else {
        raw
    }
}

// Minimal f32<->f16 for the fp16 group scales.
trait F16Bits {
    fn to_f16_bits(self) -> u16;
    fn from_f16_bits(bits: u16) -> f32;
}
impl F16Bits for f32 {
    fn to_f16_bits(self) -> u16 {
        let f = self as f64;
        let sign = if f < 0.0 { 0x8000u16 } else { 0 };
        let a = f.abs();
        if a == 0.0 {
            return sign;
        }
        let bits = (self as f32).to_bits();
        let exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let man = bits & 0x7FFFFF;
        if exp > 15 {
            return sign | 0x7C00;
        }
        if exp < -14 {
            return sign;
        }
        let half_exp = exp + 15;
        let half = ((half_exp as u32) << 10) | (man >> 13);
        (half as u16) | sign
    }
    fn from_f16_bits(bits: u16) -> f32 {
        let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exp = ((bits >> 10) & 0x1F) as i32;
        let man = (bits & 0x3FF) as f32;
        if exp == 0 {
            sign * man / 1024.0 * 2.0f32.powi(-14)
        } else {
            sign * (1.0 + man / 1024.0) * 2.0f32.powi(exp - 15)
        }
    }
}

fn synth_reference(m: usize, k: usize, n: usize, seed: u64) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let n_blocks = k / GROUP;
    let scale_padded = ((n_blocks + 15) >> 4) << 4;
    let row_bytes = 16 + scale_padded * 2 + n_blocks * GROUP;
    let mut w = vec![0u8; m * row_bytes];
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };

    for row in 0..m {
        let row_off = row * row_bytes;
        w[row_off..row_off + 2].copy_from_slice(&1.0f32.to_f16_bits().to_le_bytes());
        w[row_off + 2..row_off + 4].copy_from_slice(&0.0f32.to_f16_bits().to_le_bytes());
        for g in 0..n_blocks {
            let gscale = 0.02f32 + ((next() & 0xFF) as f32) * 1e-3;
            let gscale_f16 = gscale.to_f16_bits();
            w[row_off + 16 + g * 2..row_off + 18 + g * 2]
                .copy_from_slice(&gscale_f16.to_le_bytes());
            for j in 0..GROUP {
                let idx = row_off + 16 + scale_padded * 2 + g * GROUP + j;
                let v = (next() & 0xFF) as f32 / 256.0 - 0.5;
                w[idx] = f32_to_e4m3(v);
            }
        }
    }

    let mut x = vec![0.0f32; n * k];
    for i in 0..n * k {
        x[i] = ((i as i64).wrapping_mul(0x91c2_a73d).wrapping_add(0x2222) & 0xFFFFFF) as f32 * 1e-7 - 0.5;
    }

    // Reference: Y[n][m] = sum_k X[n][k] * A[m][k].
    let mut y = vec![0.0f32; n * m];
    for nn in 0..n {
        for mm in 0..m {
            let row_off = mm * row_bytes;
            let mut acc = 0.0f32;
            for g in 0..n_blocks {
                let gscale_bits = u16::from_le_bytes([w[row_off + 16 + g * 2], w[row_off + 16 + g * 2 + 1]]);
                let gscale = f32::from_f16_bits(gscale_bits);
                let mut gacc = 0.0f32;
                for j in 0..GROUP {
                    let leaf = w[row_off + 16 + scale_padded * 2 + g * GROUP + j];
                    gacc += e4m3_to_f32(leaf) * x[nn * k + g * GROUP + j];
                }
                acc += gacc * gscale;
            }
            y[nn * m + mm] = acc;
        }
    }
    (w, x, y)
}
