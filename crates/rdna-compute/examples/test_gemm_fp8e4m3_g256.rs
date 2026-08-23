// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 amitsheokand
// hipfire — see LICENSE and NOTICE in the project root.

//! Correctness test for the gfx1201 block-scaled FP8 E4M3 WMMA GEMM (prefill).
//!
//! Compares `gemm_fp8e4m3_g256_wmma_gfx1201` (native
//! `v_wmma_f32_16x16x16_fp8_fp8`, X packed by `pack_f32_to_fp8_gfx12`)
//! against a CPU FP32 reference. Times are GEMM-only after the pack
//! cache warms (`ensure_fp8_x` skips reconvert for the same X pointer).
//!
//! First step is an encoder→GPU cosine oracle: F32 W is packed with the
//! same G256 E4M3fn recipe as `hipfire-quantize` (`fp8e4m3_g256.rs`;
//! copied here — rdna-compute cannot depend on that crate) and compared
//! to `W_f32 @ X` (cosine) and `dequant(enc) @ X` (NRMSE). GEMV uses F32
//! X; GEMM still packs X to E4M3.
//!
//! Skips silently on non-gfx1201 archs.
//!
//! Run:
//!   cargo run --release -p rdna-compute --example test_gemm_fp8e4m3_g256 --features lab
//!   HIPFIRE_FP8_GEMM_QUICK=1 cargo run --release -p rdna-compute --example test_gemm_fp8e4m3_g256 --features lab
//!   HIPFIRE_FP8_GEMM_QUICK=prefill cargo run --release -p rdna-compute --example test_gemm_fp8e4m3_g256 --features lab

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

    if !encoder_gpu_oracle(&mut gpu) {
        eprintln!("\n=== FAIL === encoder→GPU cosine oracle");
        std::process::exit(1);
    }

    // Small tiles first so a C-map / launch fault fails in seconds, then
    // Qwen prefill shapes. CPU ref is O(N*M*K) and w_down takes many minutes.
    // HIPFIRE_FP8_GEMM_QUICK=1 → three small tiles; =prefill → skip w_down.
    // Shapes: [N batch, M out, K in], K padded to 256.
    let quick = std::env::var("HIPFIRE_FP8_GEMM_QUICK").ok();
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
    let shapes: Vec<(usize, usize, usize, &str)> = match quick.as_deref() {
        Some("1") => shapes.into_iter().take(3).collect(),
        Some("prefill") => shapes
            .into_iter()
            .filter(|s| !s.3.contains("w_down"))
            .collect(),
        _ => shapes,
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
            gpu.fp8_gemm_e4m3_g256(&w, &x, &y, m, k, n).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();

        // Time.
        let t = Instant::now();
        for _ in 0..trials {
            gpu.fp8_gemm_e4m3_g256(&w, &x, &y, m, k, n).unwrap();
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

    if !fused_gate_up_matches_dual(&mut gpu) {
        all_pass = false;
    }
    if !fused_qkvza_matches_quad(&mut gpu) {
        all_pass = false;
    }
    if !fused_qkv_matches_triple(&mut gpu) {
        all_pass = false;
    }

    if !all_pass {
        eprintln!("\n=== FAIL ===");
        std::process::exit(1);
    }
    eprintln!("\n=== ALL PASS ===");
}

/// Fused gate+up must match two overwrite GEMMs on the same packed X.
fn fused_gate_up_matches_dual(gpu: &mut Gpu) -> bool {
    let n = 32;
    let gate_m = 64;
    let up_m = 64;
    let k = 256;
    let n_blocks = k / GROUP;
    let scale_padded = ((n_blocks + 15) >> 4) << 4;
    let row_bytes = 16 + scale_padded * 2 + n_blocks * GROUP;

    let (w_gate_host, x_host, _) = synth_reference(gate_m, k, n, 0xC0FFEE);
    let (w_up_host, _, _) = synth_reference(up_m, k, n, 0x0BADF00D);

    let w_gate = gpu
        .upload_raw(&w_gate_host, &[gate_m * row_bytes])
        .unwrap();
    let w_up = gpu.upload_raw(&w_up_host, &[up_m * row_bytes]).unwrap();
    let x = gpu.alloc_tensor(&[n, k], DType::F32).unwrap();
    gpu.hip
        .memcpy_htod(&x.buf, unsafe {
            std::slice::from_raw_parts(x_host.as_ptr() as *const u8, n * k * 4)
        })
        .unwrap();

    let y_gate_ref = gpu.alloc_tensor(&[n, gate_m], DType::F32).unwrap();
    let y_up_ref = gpu.alloc_tensor(&[n, up_m], DType::F32).unwrap();
    let y_gate = gpu.alloc_tensor(&[n, gate_m], DType::F32).unwrap();
    let y_up = gpu.alloc_tensor(&[n, up_m], DType::F32).unwrap();

    gpu.fp8_gemm_e4m3_g256(&w_gate, &x, &y_gate_ref, gate_m, k, n)
        .unwrap();
    gpu.fp8_gemm_e4m3_g256(&w_up, &x, &y_up_ref, up_m, k, n)
        .unwrap();
    gpu.fp8_gemm_gate_up_e4m3_g256(
        &w_gate, &w_up, &x, &y_gate, &y_up, gate_m, up_m, k, n,
    )
    .unwrap();
    gpu.hip.device_synchronize().unwrap();

    let g_ref = gpu.download_f32(&y_gate_ref).unwrap();
    let u_ref = gpu.download_f32(&y_up_ref).unwrap();
    let g = gpu.download_f32(&y_gate).unwrap();
    let u = gpu.download_f32(&y_up).unwrap();

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }
    let dg = max_abs(&g_ref, &g);
    let du = max_abs(&u_ref, &u);
    // Same kernel math as overwrite GEMM; only routing differs. Bit-exact
    // on this tile in practice; keep a tiny float slop for store order.
    let ok = dg <= 1e-5 && du <= 1e-5;
    if ok {
        eprintln!("  fused_gate_up vs dual GEMM     OK  max|Δ| gate={dg:.3e} up={du:.3e}");
    } else {
        eprintln!("  fused_gate_up vs dual GEMM     FAIL  max|Δ| gate={dg:.3e} up={du:.3e}");
    }
    ok
}

/// Fused QKVZA must match four overwrite GEMMs on the same packed X.
/// Uneven M so a 16-row tile can straddle bank boundaries.
fn fused_qkvza_matches_quad(gpu: &mut Gpu) -> bool {
    let n = 32;
    let qkv_m = 48;
    let z_m = 32;
    let beta_m = 16;
    let alpha_m = 32;
    let k = 256;
    let n_blocks = k / GROUP;
    let scale_padded = ((n_blocks + 15) >> 4) << 4;
    let row_bytes = 16 + scale_padded * 2 + n_blocks * GROUP;

    let (w_qkv_host, x_host, _) = synth_reference(qkv_m, k, n, 0xA11CE);
    let (w_z_host, _, _) = synth_reference(z_m, k, n, 0xBEEF);
    let (w_beta_host, _, _) = synth_reference(beta_m, k, n, 0xCAFE);
    let (w_alpha_host, _, _) = synth_reference(alpha_m, k, n, 0xF00D);

    let w_qkv = gpu
        .upload_raw(&w_qkv_host, &[qkv_m * row_bytes])
        .unwrap();
    let w_z = gpu.upload_raw(&w_z_host, &[z_m * row_bytes]).unwrap();
    let w_beta = gpu
        .upload_raw(&w_beta_host, &[beta_m * row_bytes])
        .unwrap();
    let w_alpha = gpu
        .upload_raw(&w_alpha_host, &[alpha_m * row_bytes])
        .unwrap();
    let x = gpu.alloc_tensor(&[n, k], DType::F32).unwrap();
    gpu.hip
        .memcpy_htod(&x.buf, unsafe {
            std::slice::from_raw_parts(x_host.as_ptr() as *const u8, n * k * 4)
        })
        .unwrap();

    let y_qkv_ref = gpu.alloc_tensor(&[n, qkv_m], DType::F32).unwrap();
    let y_z_ref = gpu.alloc_tensor(&[n, z_m], DType::F32).unwrap();
    let y_beta_ref = gpu.alloc_tensor(&[n, beta_m], DType::F32).unwrap();
    let y_alpha_ref = gpu.alloc_tensor(&[n, alpha_m], DType::F32).unwrap();
    let y_qkv = gpu.alloc_tensor(&[n, qkv_m], DType::F32).unwrap();
    let y_z = gpu.alloc_tensor(&[n, z_m], DType::F32).unwrap();
    let y_beta = gpu.alloc_tensor(&[n, beta_m], DType::F32).unwrap();
    let y_alpha = gpu.alloc_tensor(&[n, alpha_m], DType::F32).unwrap();

    gpu.fp8_gemm_e4m3_g256(&w_qkv, &x, &y_qkv_ref, qkv_m, k, n)
        .unwrap();
    gpu.fp8_gemm_e4m3_g256(&w_z, &x, &y_z_ref, z_m, k, n)
        .unwrap();
    gpu.fp8_gemm_e4m3_g256(&w_beta, &x, &y_beta_ref, beta_m, k, n)
        .unwrap();
    gpu.fp8_gemm_e4m3_g256(&w_alpha, &x, &y_alpha_ref, alpha_m, k, n)
        .unwrap();
    gpu.fp8_gemm_qkvza_e4m3_g256(
        &w_qkv, &w_z, &w_beta, &w_alpha, &x, &y_qkv, &y_z, &y_beta, &y_alpha,
        qkv_m, z_m, beta_m, alpha_m, k, n,
    )
    .unwrap();
    gpu.hip.device_synchronize().unwrap();

    let q_ref = gpu.download_f32(&y_qkv_ref).unwrap();
    let z_ref = gpu.download_f32(&y_z_ref).unwrap();
    let b_ref = gpu.download_f32(&y_beta_ref).unwrap();
    let a_ref = gpu.download_f32(&y_alpha_ref).unwrap();
    let q = gpu.download_f32(&y_qkv).unwrap();
    let z = gpu.download_f32(&y_z).unwrap();
    let b = gpu.download_f32(&y_beta).unwrap();
    let a = gpu.download_f32(&y_alpha).unwrap();

    fn max_abs(p: &[f32], q: &[f32]) -> f32 {
        p.iter()
            .zip(q)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }
    let dq = max_abs(&q_ref, &q);
    let dz = max_abs(&z_ref, &z);
    let db = max_abs(&b_ref, &b);
    let da = max_abs(&a_ref, &a);
    let ok = dq <= 1e-5 && dz <= 1e-5 && db <= 1e-5 && da <= 1e-5;
    if ok {
        eprintln!(
            "  fused_qkvza vs quad GEMM     OK  max|Δ| qkv={dq:.3e} z={dz:.3e} beta={db:.3e} alpha={da:.3e}"
        );
    } else {
        eprintln!(
            "  fused_qkvza vs quad GEMM     FAIL  max|Δ| qkv={dq:.3e} z={dz:.3e} beta={db:.3e} alpha={da:.3e}"
        );
    }
    ok
}

/// Fused FA QKV must match three overwrite GEMMs on the same packed X.
/// Uneven M so a 16-row tile can straddle bank boundaries.
fn fused_qkv_matches_triple(gpu: &mut Gpu) -> bool {
    let n = 32;
    let q_m = 48;
    let k_m = 32;
    let v_m = 16;
    let k = 256;
    let n_blocks = k / GROUP;
    let scale_padded = ((n_blocks + 15) >> 4) << 4;
    let row_bytes = 16 + scale_padded * 2 + n_blocks * GROUP;

    let (w_q_host, x_host, _) = synth_reference(q_m, k, n, 0x111111);
    let (w_k_host, _, _) = synth_reference(k_m, k, n, 0x222222);
    let (w_v_host, _, _) = synth_reference(v_m, k, n, 0x333333);

    let w_q = gpu.upload_raw(&w_q_host, &[q_m * row_bytes]).unwrap();
    let w_k = gpu.upload_raw(&w_k_host, &[k_m * row_bytes]).unwrap();
    let w_v = gpu.upload_raw(&w_v_host, &[v_m * row_bytes]).unwrap();
    let x = gpu.alloc_tensor(&[n, k], DType::F32).unwrap();
    gpu.hip
        .memcpy_htod(&x.buf, unsafe {
            std::slice::from_raw_parts(x_host.as_ptr() as *const u8, n * k * 4)
        })
        .unwrap();

    let y_q_ref = gpu.alloc_tensor(&[n, q_m], DType::F32).unwrap();
    let y_k_ref = gpu.alloc_tensor(&[n, k_m], DType::F32).unwrap();
    let y_v_ref = gpu.alloc_tensor(&[n, v_m], DType::F32).unwrap();
    let y_q = gpu.alloc_tensor(&[n, q_m], DType::F32).unwrap();
    let y_k = gpu.alloc_tensor(&[n, k_m], DType::F32).unwrap();
    let y_v = gpu.alloc_tensor(&[n, v_m], DType::F32).unwrap();

    gpu.fp8_gemm_e4m3_g256(&w_q, &x, &y_q_ref, q_m, k, n)
        .unwrap();
    gpu.fp8_gemm_e4m3_g256(&w_k, &x, &y_k_ref, k_m, k, n)
        .unwrap();
    gpu.fp8_gemm_e4m3_g256(&w_v, &x, &y_v_ref, v_m, k, n)
        .unwrap();
    gpu.fp8_gemm_qkv_e4m3_g256(
        &w_q, &w_k, &w_v, &x, &y_q, &y_k, &y_v, q_m, k_m, v_m, k, n,
    )
    .unwrap();
    gpu.hip.device_synchronize().unwrap();

    let q_ref = gpu.download_f32(&y_q_ref).unwrap();
    let k_ref = gpu.download_f32(&y_k_ref).unwrap();
    let v_ref = gpu.download_f32(&y_v_ref).unwrap();
    let q = gpu.download_f32(&y_q).unwrap();
    let k = gpu.download_f32(&y_k).unwrap();
    let v = gpu.download_f32(&y_v).unwrap();

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }
    let dq = max_abs(&q_ref, &q);
    let dk = max_abs(&k_ref, &k);
    let dv = max_abs(&v_ref, &v);
    let ok = dq <= 1e-5 && dk <= 1e-5 && dv <= 1e-5;
    if ok {
        eprintln!("  fused_qkv vs triple GEMM     OK  max|Δ| q={dq:.3e} k={dk:.3e} v={dv:.3e}");
    } else {
        eprintln!("  fused_qkv vs triple GEMM     FAIL  max|Δ| q={dq:.3e} k={dk:.3e} v={dv:.3e}");
    }
    ok
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

/// Encoder→GPU oracle. Keep the pack/dequant helpers in sync with
/// `hipfire-quantize/src/fp8e4m3_g256.rs` (rdna-compute cannot depend on it).
fn encoder_gpu_oracle(gpu: &mut Gpu) -> bool {
    const N: usize = 16;
    const M: usize = 16;
    const K: usize = 256;
    eprintln!("=== encoder→GPU cosine oracle  N={N} M={M} K={K} ===");

    let mut w_f32 = vec![0.0f32; M * K];
    let mut x_host = vec![0.0f32; N * K];
    let mut s = 0xC0FFEE_u64;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as u32) as f32 / (u32::MAX as f32)
    };
    for v in &mut w_f32 {
        *v = next() * 0.1 - 0.05;
    }
    for v in &mut x_host {
        *v = next() * 0.2 - 0.1;
    }

    let w_enc = encode_fp8e4m3_g256_2d(&w_f32, M, K);
    let w_dq = dequant_fp8e4m3_g256_2d(&w_enc, M, K);

    let y_true = matmul_f32(&w_f32, &x_host, M, K, N);
    let y_dq = matmul_f32(&w_dq, &x_host, M, K, N);
    let enc_nrmse = nrmse(&y_true, &y_dq);

    let w = gpu.upload_raw(&w_enc, &[w_enc.len()]).unwrap();
    let x_gemv = gpu.alloc_tensor(&[K], DType::F32).unwrap();
    let y_gemv = gpu.alloc_tensor(&[M], DType::F32).unwrap();
    gpu.hip
        .memcpy_htod(&x_gemv.buf, unsafe {
            std::slice::from_raw_parts(x_host.as_ptr() as *const u8, K * 4)
        })
        .unwrap();
    gpu.fp8_gemv_e4m3_g256(&w, &x_gemv, &y_gemv, M, K)
        .unwrap();
    let y_gemv_gpu = gpu.download_f32(&y_gemv).unwrap();
    let y_true0 = &y_true[..M];
    let y_dq0 = &y_dq[..M];
    let gemv_vs_dq = nrmse(y_dq0, &y_gemv_gpu);
    let gemv_cos = cosine(y_true0, &y_gemv_gpu);

    let x_gemm = gpu.alloc_tensor(&[N, K], DType::F32).unwrap();
    let y_gemm = gpu.alloc_tensor(&[N, M], DType::F32).unwrap();
    gpu.hip
        .memcpy_htod(&x_gemm.buf, unsafe {
            std::slice::from_raw_parts(x_host.as_ptr() as *const u8, N * K * 4)
        })
        .unwrap();
    gpu.fp8_gemm_e4m3_g256(&w, &x_gemm, &y_gemm, M, K, N)
        .unwrap();
    let y_gemm_gpu = gpu.download_f32(&y_gemm).unwrap();
    let gemm_vs_dq = nrmse(&y_dq, &y_gemm_gpu);
    let gemm_cos = cosine(&y_true, &y_gemm_gpu);

    eprintln!("  encoder  NRMSE(dequant@X vs W_f32@X)={enc_nrmse:.3e}");
    eprintln!(
        "  GEMV     NRMSE vs dequant={gemv_vs_dq:.3e}  cosine vs W_f32={gemv_cos:.6}"
    );
    eprintln!(
        "  GEMM     NRMSE vs dequant={gemm_vs_dq:.3e}  cosine vs W_f32={gemm_cos:.6}"
    );

    let ok = enc_nrmse < 0.05
        && gemv_vs_dq < 0.05
        && gemm_vs_dq < 0.05
        && gemv_cos > 0.99
        && gemm_cos > 0.99;
    if ok {
        eprintln!("  encoder oracle OK");
    } else {
        eprintln!("  encoder oracle FAIL");
    }
    ok
}

fn nrmse(a: &[f32], b: &[f32]) -> f64 {
    let mut se = 0.0;
    let mut sr = 0.0;
    for (x, y) in a.iter().zip(b) {
        let d = *x as f64 - *y as f64;
        se += d * d;
        sr += (*x as f64) * (*x as f64);
    }
    (se / sr.max(1e-30)).sqrt()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (x, y) in a.iter().zip(b) {
        let xf = *x as f64;
        let yf = *y as f64;
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-30)
}

fn matmul_f32(w: &[f32], x: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n * m];
    for nn in 0..n {
        for mm in 0..m {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += w[mm * k + kk] * x[nn * k + kk];
            }
            y[nn * m + mm] = acc;
        }
    }
    y
}

const E4M3_MAX: f32 = 448.0;

fn enc_e4m3fn_mag(u: u8) -> f32 {
    let exp = ((u >> 3) & 0xF) as i32;
    let mant = (u & 0x7) as f32;
    if exp == 0 {
        return (2.0f32).powi(-6) * mant / 8.0;
    }
    if exp == 0xF && (u & 0x7) == 7 {
        return E4M3_MAX;
    }
    (2.0f32).powi(exp - 7) * (1.0 + mant / 8.0)
}

fn enc_e4m3fn_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0f32 } else { 1.0 };
    sign * enc_e4m3fn_mag(byte & 0x7F)
}

fn enc_f32_to_e4m3fn(v: f32) -> u8 {
    if !v.is_finite() {
        return 0;
    }
    let neg = v.is_sign_negative();
    let a = v.abs();
    if a == 0.0 {
        return if neg { 0x80 } else { 0 };
    }
    if a >= E4M3_MAX {
        return if neg { 0xFE } else { 0x7E };
    }
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for code in 0u8..=0x7E {
        let err = (enc_e4m3fn_mag(code) - a).abs();
        if err < best_err {
            best_err = err;
            best = code;
        }
    }
    if neg {
        best | 0x80
    } else {
        best
    }
}

/// Truncating F32→F16, same as `hipfire-quantize::float16::f32_to_f16`.
fn enc_f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;
    if exp == 0xFF {
        let f16_frac = if frac == 0 { 0 } else { (frac >> 13) | 1 };
        return ((sign << 15) | (0x1F << 10) | f16_frac) as u16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return ((sign << 15) | (0x1F << 10)) as u16;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return (sign << 15) as u16;
        }
        let f = frac | 0x800000;
        let shift = (1 - new_exp + 13) as u32;
        return ((sign << 15) | (f >> shift)) as u16;
    }
    ((sign << 15) | ((new_exp as u32) << 10) | (frac >> 13)) as u16
}

fn enc_f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((bits >> 10) & 0x1F) as i32;
    let man = (bits & 0x3FF) as f32;
    if exp == 0 {
        return sign * man / 1024.0 * 2.0f32.powi(-14);
    }
    if exp == 31 {
        return if man == 0.0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        };
    }
    sign * (1.0 + man / 1024.0) * 2.0f32.powi(exp - 15)
}

fn encode_fp8e4m3_g256_row(row: &[f32]) -> Vec<u8> {
    let k = row.len();
    let n_blocks = k / GROUP;
    let scale_padded = ((n_blocks + 15) >> 4) << 4;
    let nbytes = 16 + scale_padded * 2 + n_blocks * GROUP;
    let mut out = vec![0u8; nbytes];
    out[0..2].copy_from_slice(&enc_f32_to_f16(1.0).to_le_bytes());
    out[2..4].copy_from_slice(&enc_f32_to_f16(0.0).to_le_bytes());
    out[4..6].copy_from_slice(&(n_blocks as u16).to_le_bytes());
    for g in 0..n_blocks {
        let block = &row[g * GROUP..(g + 1) * GROUP];
        let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let gscale = if amax > 0.0 { amax / E4M3_MAX } else { 1.0 };
        let inv = if amax > 0.0 { 1.0 / gscale } else { 0.0 };
        out[16 + g * 2..18 + g * 2].copy_from_slice(&enc_f32_to_f16(gscale).to_le_bytes());
        let leaf_off = 16 + scale_padded * 2 + g * GROUP;
        for j in 0..GROUP {
            out[leaf_off + j] = enc_f32_to_e4m3fn(block[j] * inv);
        }
    }
    out
}

fn encode_fp8e4m3_g256_2d(f32_data: &[f32], m: usize, k: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for r in 0..m {
        out.extend_from_slice(&encode_fp8e4m3_g256_row(&f32_data[r * k..(r + 1) * k]));
    }
    out
}

fn dequant_fp8e4m3_g256_2d(packed: &[u8], m: usize, k: usize) -> Vec<f32> {
    let n_blocks = k / GROUP;
    let scale_padded = ((n_blocks + 15) >> 4) << 4;
    let row_bytes = 16 + scale_padded * 2 + n_blocks * GROUP;
    let mut out = vec![0.0f32; m * k];
    for r in 0..m {
        let packed_row = &packed[r * row_bytes..(r + 1) * row_bytes];
        let row_scale = enc_f16_to_f32(u16::from_le_bytes([packed_row[0], packed_row[1]]));
        for g in 0..n_blocks {
            let gscale = enc_f16_to_f32(u16::from_le_bytes([
                packed_row[16 + g * 2],
                packed_row[16 + g * 2 + 1],
            ]));
            let leaf_off = 16 + scale_padded * 2 + g * GROUP;
            for j in 0..GROUP {
                out[r * k + g * GROUP + j] =
                    row_scale * gscale * enc_e4m3fn_to_f32(packed_row[leaf_off + j]);
            }
        }
    }
    out
}
