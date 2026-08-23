// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 amitsheokand
// hipfire — see LICENSE and NOTICE in the project root.

//! Correctness + perf test for the gfx1201 block-scaled FP8 E4M3 decode GEMV.
//!
//! Compares `gemv_fp8e4m3_g256_gfx1201` (native `v_dot4_f32_fp8_fp8`) against
//! a CPU FP32 reference across production Qwen decode shapes. Reports NRMSE,
//! timing, and % of the R9700's ~640 GB/s peak DRAM bandwidth.
//!
//! Skips silently on non-gfx1201 archs.
//!
//! Run:
//!   cargo run -p rdna-compute --example test_gemv_fp8e4m3_g256 --features lab

use rdna_compute::{DType, Gpu};
use std::time::Instant;

const PEAK_GBPS: f64 = 640.0;
const GROUP: usize = 256;

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    if arch != "gfx1201" {
        eprintln!("=== SKIP === FP8 E4M3 GEMV is gfx1201-only (arch={arch})");
        return;
    }
    eprintln!("=== gemv_fp8e4m3_g256_gfx1201 vs CPU FP32 reference ===");
    eprintln!("  arch={arch}  peak_bw_gbps={PEAK_GBPS}");

    // Qwen 9B/27B decode-path GEMV shapes (K padded to 256).
    let shapes: Vec<(usize, usize, &str)> = vec![
        (2048, 2048, "qkv-q     M=2048 K=2048"),
        (512, 2048, "qkv-kv    M=512  K=2048"),
        (11008, 2048, "gate_up   M=11008 K=2048"),
        (2048, 11008, "w_down    M=2048  K=11008"),
        (1024, 2048, "small     M=1024 K=2048"),
    ];

    let trials = 200;
    let warmup = 20;

    let mut all_pass = true;
    for (m, k, label) in &shapes {
        let (m, k) = (*m, *k);
        let n_blocks = k / GROUP;
        let scale_padded = ((n_blocks + 15) >> 4) << 4;
        let row_bytes = 16 + scale_padded * 2 + n_blocks * GROUP;
        let total_w_bytes = m * row_bytes;

        // Synthesize deterministic block-scaled FP8 weights.
        let (w_host, x_host, y_cpu) = synth_reference(m, k, 0xAA00 | (m as u64) ^ (k as u64));

        let w = gpu
            .upload_raw(&w_host, &[total_w_bytes])
            .unwrap();
        let x = gpu.alloc_tensor(&[k], DType::F32).unwrap();
        let y = gpu.alloc_tensor(&[m], DType::F32).unwrap();
        gpu.hip
            .memcpy_htod(&x.buf, unsafe {
                std::slice::from_raw_parts(x_host.as_ptr() as *const u8, k * 4)
            })
            .unwrap();

        // Warmup (JIT-compile the kernel on first call via ensure_kernel).
        for _ in 0..warmup {
            gpu.try_gfx1201()
                .unwrap()
                .fp8_gemv_e4m3_g256(&w, &x, &y, m, k)
                .unwrap();
        }
        gpu.hip.device_synchronize().unwrap();

        // Time.
        let t = Instant::now();
        for _ in 0..trials {
            gpu.try_gfx1201()
                .unwrap()
                .fp8_gemv_e4m3_g256(&w, &x, &y, m, k)
                .unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let us = t.elapsed().as_secs_f64() * 1e6 / trials as f64;
        let bw = (total_w_bytes as f64) / (us * 1e-6) / 1e9;

        // Correctness vs CPU FP32 reference.
        let y_gpu = gpu.download_f32(&y).unwrap();
        let mut max_abs = 0.0f64;
        let mut max_abs_ref = 0.0f64;
        let mut sum_sq_err = 0.0f64;
        let mut sum_sq_ref = 0.0f64;
        for i in 0..m {
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
        // Tolerance: 5% relative — the kernel converts F32 activations to
        // E4M3 inline (v_dot4 consumes fp8 x fp8), so a ~1-3% activation
        // quantization error vs the exact-F32 CPU reference is expected.
        let tol_abs = 0.05 * max_abs_ref.max(1e-3);
        let bad: usize = (0..m)
            .filter(|&i| ((y_cpu[i] as f64) - (y_gpu[i] as f64)).abs() > tol_abs)
            .count();
        if bad > 0 {
            eprintln!(
                "  {label:30}  FAIL {bad}/{m}  max_abs={max_abs:.3e} tol={tol_abs:.3e} max|y|={max_abs_ref:.3e} NRMSE={nrmse:.3e}  {us:6.2}µs ({bw:5.1} GB/s = {:4.1}%)",
                bw / PEAK_GBPS * 100.0
            );
            all_pass = false;
        } else {
            eprintln!(
                "  {label:30}  OK  NRMSE={nrmse:.3e}  {us:6.2}µs ({bw:5.1} GB/s = {:4.1}%)",
                bw / PEAK_GBPS * 100.0
            );
        }
    }

    if !all_pass {
        eprintln!("\n=== FAIL ===");
        std::process::exit(1);
    }
    eprintln!("\n=== ALL PASS ===");
}

/// E4M3 -> f32 (OCP: 1 sign, 4 exp, 3 mantissa, bias 7). NaN/Inf not used.
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
    // Clamp + round-to-nearest to E4M3 grid. Only used on synth values in
    // [-2, 2), so the finite range covers everything we emit.
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

fn synth_reference(m: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
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

    // CPU reference accumulators per row.
    let mut x = vec![0.0f32; k];
    for i in 0..k {
        x[i] = ((i as i64).wrapping_mul(0x91c2_a73d).wrapping_add(0x1111) & 0xFFFFFF) as f32 * 1e-7 - 0.5;
    }

    for row in 0..m {
        let row_off = row * row_bytes;
        // fp16 row_scale = 1.0, row_bias = 0.0 (reserved fields).
        w[row_off..row_off + 2].copy_from_slice(&1.0f32.to_f16_bits().to_le_bytes());
        w[row_off + 2..row_off + 4].copy_from_slice(&0.0f32.to_f16_bits().to_le_bytes());
        for g in 0..n_blocks {
            let gscale = 0.02f32 + ((next() & 0xFF) as f32) * 1e-3;
            let gscale_f16 = gscale.to_f16_bits();
            w[row_off + 16 + g * 2..row_off + 18 + g * 2]
                .copy_from_slice(&gscale_f16.to_le_bytes());
            for j in 0..GROUP {
                let idx = row_off + 16 + scale_padded * 2 + g * GROUP + j;
                let v = (next() & 0xFF) as f32 / 256.0 - 0.5; // in [-0.5, 0.5)
                w[idx] = f32_to_e4m3(v);
            }
        }
    }

    // Reference with proper per-group scaling.
    let mut y = vec![0.0f32; m];
    let mut dbg_sum = 0.0f32;
    for row in 0..m {
        let row_off = row * row_bytes;
        let mut acc = 0.0f32;
        for g in 0..n_blocks {
            let gscale_bits = u16::from_le_bytes([
                w[row_off + 16 + g * 2],
                w[row_off + 16 + g * 2 + 1],
            ]);
            let gscale = f32::from_f16_bits(gscale_bits);
            let mut gacc = 0.0f32;
            for j in 0..GROUP {
                let leaf = w[row_off + 16 + scale_padded * 2 + g * GROUP + j];
                gacc += e4m3_to_f32(leaf) * x[g * GROUP + j];
            }
            acc += gacc * gscale;
        }
        y[row] = acc;
        dbg_sum += acc.abs();
    }
    eprintln!("  [synth] m={m} k={k} row0_y={} sum|y|={}", y[0], dbg_sum);
    if m > 0 && n_blocks > 0 {
        let row_off = 0usize;
        let g = 0usize;
        let gscale_bits = u16::from_le_bytes([w[16 + g * 2], w[16 + g * 2 + 1]]);
        let leaf0 = w[16 + scale_padded * 2 + g * GROUP];
        let leaf1 = w[16 + scale_padded * 2 + g * GROUP + 1];
        let x0 = x[g * GROUP];
        let x1 = x[g * GROUP + 1];
        eprintln!(
            "  [synth] gscale_bits={gscale_bits:#06x} gscale={} leaf0={leaf0:#04x}({}) leaf1={leaf1:#04x}({}) x0={x0} x1={x1}",
            f32::from_f16_bits(gscale_bits),
            e4m3_to_f32(leaf0),
            e4m3_to_f32(leaf1),
        );
    }
    (w, x, y)
}

// f16 conversion helpers (hipfire-quantize has them; keep the test standalone).
#[allow(non_camel_case_types)]
type f16 = u16;
trait F16Bits {
    fn to_bits(self) -> u16;
    fn to_f16_bits(self) -> u16;
    fn from_f16_bits(bits: u16) -> f32;
}
impl F16Bits for f32 {
    fn to_bits(self) -> u16 {
        self.to_f16_bits()
    }
    fn to_f16_bits(self) -> u16 {
        let f = self as f64;
        // minimal f32->f16 round-to-nearest-even for values in normal range
        let sign = if f < 0.0 { 0x8000u16 } else { 0 };
        let a = f.abs();
        if a == 0.0 {
            return sign;
        }
        let bits = (self as f32).to_bits();
        let exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let man = bits & 0x7FFFFF;
        if exp > 15 {
            return sign | 0x7C00; // inf
        }
        if exp < -14 {
            return sign; // flush to zero (synth values won't hit this)
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
impl F16Bits for u16 {
    fn to_bits(self) -> u16 {
        self
    }
    fn to_f16_bits(self) -> u16 {
        self
    }
    fn from_f16_bits(bits: u16) -> f32 {
        f32::from_f16_bits(bits)
    }
}
