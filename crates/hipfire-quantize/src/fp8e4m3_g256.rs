//! Block-scaled OCP E4M3fn (bias 7) weights, group-256.
//!
//! Layout matches `QuantType::FP8E4M3G256` and the gfx1201 GEMV/GEMM kernels:
//! `[16 B hdr][n_blocks × 2 B fp16 scale, 16-aligned][n_blocks × 256 B E4M3]`.
//! Header: fp16 row_scale @0 (1.0), fp16 row_bias @2 (0), u16 n_blocks @4.
//! Per-group scale is `amax / 448` so leaves use the full finite E4M3 range.
//! 0x7F (NaN) is never emitted; max finite code is 0x7E (±448).

use crate::float16::{f16_to_f32, f32_to_f16};
use rayon::prelude::*;

const GROUP: usize = 256;
/// OCP E4M3fn max finite magnitude (exp=15, mant=6 → 0x7E).
const E4M3_MAX: f32 = 448.0;

const fn pow2_i(n: i32) -> f32 {
    f32::from_bits(((127 + n) as u32) << 23)
}

const fn mag_at(code: u8) -> f32 {
    let exp = (code >> 3) & 0xF;
    let mant = code & 0x7;
    if exp == 0 {
        return pow2_i(-6) * (mant as f32) / 8.0;
    }
    if exp == 0xF && mant == 7 {
        return E4M3_MAX;
    }
    pow2_i(exp as i32 - 7) * (1.0 + (mant as f32) / 8.0)
}

/// Magnitudes for codes 0..=0x7E. Strictly increasing; 0x7F (NaN) is omitted.
const MAG: [f32; 127] = {
    let mut m = [0.0f32; 127];
    let mut i = 0;
    while i < 127 {
        m[i] = mag_at(i as u8);
        i += 1;
    }
    m
};

pub fn row_bytes(k: usize) -> usize {
    assert!(
        k.is_multiple_of(GROUP),
        "FP8E4M3G256 requires K%256=0, got K={k}"
    );
    let n_blocks = k / GROUP;
    let scale_padded = ((n_blocks + 15) >> 4) << 4;
    16 + scale_padded * 2 + n_blocks * GROUP
}

pub fn e4m3fn_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let mag = e4m3fn_mag(byte & 0x7F);
    sign * mag
}

fn e4m3fn_mag(u: u8) -> f32 {
    let u = u & 0x7F;
    if u >= 127 {
        return E4M3_MAX;
    }
    MAG[u as usize]
}

/// Round-to-nearest finite E4M3fn. Saturates at ±448. Never emits 0x7F.
/// Ties keep the lower code (same as the old linear scan's `<` update).
pub fn f32_to_e4m3fn(v: f32) -> u8 {
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
    let mag = nearest_mag_code(a);
    if neg {
        mag | 0x80
    } else {
        mag
    }
}

fn nearest_mag_code(a: f32) -> u8 {
    let mut left = 0i32;
    let mut right = 127i32;
    while left < right {
        let mid = (left + right) / 2;
        if MAG[mid as usize] < a {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    let hi = left.min(126) as u8;
    if hi == 0 {
        return 0;
    }
    let lo = hi - 1;
    let err_hi = (MAG[hi as usize] - a).abs();
    let err_lo = (MAG[lo as usize] - a).abs();
    if err_hi < err_lo {
        hi
    } else {
        lo
    }
}

fn quantize_fp8e4m3_g256_row_into(row: &[f32], out: &mut [u8]) {
    let k = row.len();
    let n_blocks = k / GROUP;
    let nbytes = row_bytes(k);
    debug_assert_eq!(out.len(), nbytes);
    out[..16].fill(0);

    out[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
    out[2..4].copy_from_slice(&f32_to_f16(0.0).to_le_bytes());
    out[4..6].copy_from_slice(&(n_blocks as u16).to_le_bytes());

    let scale_padded = ((n_blocks + 15) >> 4) << 4;
    for g in 0..n_blocks {
        let block = &row[g * GROUP..(g + 1) * GROUP];
        let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let gscale = if amax > 0.0 { amax / E4M3_MAX } else { 1.0 };
        let inv = if amax > 0.0 { 1.0 / gscale } else { 0.0 };
        out[16 + g * 2..18 + g * 2].copy_from_slice(&f32_to_f16(gscale).to_le_bytes());
        let leaf_off = 16 + scale_padded * 2 + g * GROUP;
        for j in 0..GROUP {
            out[leaf_off + j] = f32_to_e4m3fn(block[j] * inv);
        }
    }
}

/// Quantize one row of K FP32 weights. K must be a multiple of 256.
pub fn quantize_fp8e4m3_g256_row(row: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; row_bytes(row.len())];
    quantize_fp8e4m3_g256_row_into(row, &mut out);
    out
}

pub fn quantize_fp8e4m3_g256_2d(f32_data: &[f32], m: usize, k: usize) -> Vec<u8> {
    assert_eq!(f32_data.len(), m * k, "FP8E4M3G256 2d length mismatch");
    let rb = row_bytes(k);
    let mut out = vec![0u8; m * rb];
    out.par_chunks_mut(rb)
        .zip(f32_data.par_chunks(k))
        .for_each(|(dst, row)| quantize_fp8e4m3_g256_row_into(row, dst));
    out
}

pub fn dequant_fp8e4m3_g256_row(packed: &[u8], k: usize) -> Vec<f32> {
    let n_blocks = k / GROUP;
    let scale_padded = ((n_blocks + 15) >> 4) << 4;
    assert_eq!(packed.len(), row_bytes(k));
    let row_scale = f16_to_f32(u16::from_le_bytes([packed[0], packed[1]]));
    let mut out = vec![0.0f32; k];
    for g in 0..n_blocks {
        let gscale = f16_to_f32(u16::from_le_bytes([
            packed[16 + g * 2],
            packed[16 + g * 2 + 1],
        ]));
        let leaf_off = 16 + scale_padded * 2 + g * GROUP;
        for j in 0..GROUP {
            out[g * GROUP + j] = row_scale * gscale * e4m3fn_to_f32(packed[leaf_off + j]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_max_is_448() {
        assert_eq!(e4m3fn_to_f32(0x7E), 448.0);
        assert_eq!(f32_to_e4m3fn(448.0), 0x7E);
        assert_eq!(f32_to_e4m3fn(10_000.0), 0x7E);
        assert_ne!(f32_to_e4m3fn(1.0), 0x7F);
    }

    #[test]
    fn row_layout_g256() {
        let k = 256;
        let row = vec![0.1f32; k];
        let packed = quantize_fp8e4m3_g256_row(&row);
        assert_eq!(packed.len(), 16 + 16 * 2 + 256);
        let n_blocks = u16::from_le_bytes([packed[4], packed[5]]);
        assert_eq!(n_blocks, 1);
        let recovered = dequant_fp8e4m3_g256_row(&packed, k);
        let nrmse = nrmse(&row, &recovered);
        assert!(nrmse < 0.02, "nrmse={nrmse}");
    }

    #[test]
    fn gaussian_rows_stay_in_e4m3_band() {
        let k = 512;
        let mut row = vec![0.0f32; k];
        let mut s = 0xC0FFEE_u64;
        for v in &mut row {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = (s as f64 / u64::MAX as f64) as f32 * 0.1 - 0.05;
        }
        let packed = quantize_fp8e4m3_g256_row(&row);
        let recovered = dequant_fp8e4m3_g256_row(&packed, k);
        let nrmse = nrmse(&row, &recovered);
        // Same band as the gfx1201 GEMM/GEMV lab (F32→E4M3 ≈ 2.5–3%).
        assert!(nrmse < 0.05, "nrmse={nrmse}");
        assert_eq!(packed.len(), row_bytes(k));
    }

    fn f32_to_e4m3fn_brute(v: f32) -> u8 {
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
            let err = (e4m3fn_mag(code) - a).abs();
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

    #[test]
    fn mag_lut_matches_powi_formula() {
        for code in 0u8..=0x7E {
            let exp = ((code >> 3) & 0xF) as i32;
            let mant = (code & 0x7) as f32;
            let want = if exp == 0 {
                (2.0f32).powi(-6) * mant / 8.0
            } else if exp == 0xF && (code & 0x7) == 7 {
                E4M3_MAX
            } else {
                (2.0f32).powi(exp - 7) * (1.0 + mant / 8.0)
            };
            assert_eq!(e4m3fn_mag(code), want, "code={code}");
        }
        assert_eq!(e4m3fn_mag(0x7F), E4M3_MAX);
    }

    #[test]
    fn fast_convert_matches_brute() {
        for code in 0u8..=0x7E {
            let x = e4m3fn_mag(code);
            assert_eq!(f32_to_e4m3fn(x), code, "exact code {code}");
            if x > 0.0 {
                assert_eq!(f32_to_e4m3fn(-x), code | 0x80, "neg code {code}");
            }
        }
        for c in 0u8..0x7E {
            let mid = 0.5 * (e4m3fn_mag(c) + e4m3fn_mag(c + 1));
            assert_eq!(
                f32_to_e4m3fn(mid),
                f32_to_e4m3fn_brute(mid),
                "midpoint {c}/{}",
                c + 1
            );
        }
        let mut s = 0xC0FFEE_u64;
        for _ in 0..80_000 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let x = f32::from_bits((s as u32) & 0x7f7f_ffff);
            if !x.is_finite() {
                continue;
            }
            assert_eq!(f32_to_e4m3fn(x), f32_to_e4m3fn_brute(x), "x={x}");
            assert_eq!(f32_to_e4m3fn(-x), f32_to_e4m3fn_brute(-x), "-x={x}");
        }
    }

    #[test]
    fn parallel_2d_matches_serial_rows() {
        let m = 32;
        let k = 256;
        let mut data = vec![0.0f32; m * k];
        let mut s = 1u64;
        for v in &mut data {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = (s as f64 / u64::MAX as f64) as f32 * 0.2 - 0.1;
        }
        let par = quantize_fp8e4m3_g256_2d(&data, m, k);
        let mut ser = Vec::new();
        for r in 0..m {
            ser.extend_from_slice(&quantize_fp8e4m3_g256_row(&data[r * k..(r + 1) * k]));
        }
        assert_eq!(par, ser);
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
}
