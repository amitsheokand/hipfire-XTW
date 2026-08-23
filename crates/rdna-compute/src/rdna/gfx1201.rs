// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact-gfx1201 operations.
//!
//! `Gfx1201Device` is the architecture proof. Its field and constructor are
//! private, it does not dereference to [`Gpu`], and it exposes only operations
//! implemented by gfx1201-owned sources.

use std::ffi::c_void;

use hip_bridge::{HipError, HipResult, KernargBlob};

use crate::{Gpu, GpuTensor};

const MQ2_LLOYD_GATE_UP_EP_SRC: &str =
    include_str!("../../../../kernels/src/gemv_mq2g256_lloyd_moe_gate_up_indexed_ep.gfx1201.hip");
const MQ2_LLOYD_GATE_UP_EP_KERNEL: &str = "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_gfx1201_ep";
const MQ2_LLOYD_GATE_UP_COMPACT_EP_SRC: &str = include_str!(
    "../../../../kernels/src/gemv_mq2g256_lloyd_moe_gate_up_indexed_compact_ep.gfx1201.hip"
);
const MQ2_LLOYD_GATE_UP_COMPACT_EP_KERNEL: &str =
    "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_compact_gfx1201_ep";
const MQ2_LLOYD_DOWN_EXPANDED_EP_SRC: &str =
    include_str!("../../../../kernels/src/gemv_mq2g256_lloyd_moe_down_expanded_k4_ep.gfx1201.hip");
const MQ2_LLOYD_DOWN_EXPANDED_EP_KERNEL: &str =
    "gemv_mq2g256_lloyd_moe_down_expanded_k4_gfx1201_ep";
const MQ2_LLOYD_DOWN_EXPANDED_COMPACT_EP_SRC: &str = include_str!(
    "../../../../kernels/src/gemv_mq2g256_lloyd_moe_down_expanded_k4_compact_ep.gfx1201.hip"
);
const MQ2_LLOYD_DOWN_EXPANDED_COMPACT_EP_KERNEL: &str =
    "gemv_mq2g256_lloyd_moe_down_expanded_k4_compact_gfx1201_ep";
const MQ2_LLOYD_DOWN_EXPANDED_LDS_EP_SRC: &str = include_str!(
    "../../../../kernels/src/gemv_mq2g256_lloyd_moe_down_expanded_k4_lds_ep.gfx1201.hip"
);
const MQ2_LLOYD_DOWN_EXPANDED_LDS_EP_KERNEL: &str =
    "gemv_mq2g256_lloyd_moe_down_expanded_k4_lds_gfx1201_ep";

const FP8_E4M3_GEMV_SRC: &str =
    include_str!("../../../../kernels/src/gemv_fp8e4m3_g256.gfx1201.hip");
const FP8_E4M3_GEMV_KERNEL: &str = "gemv_fp8e4m3_g256_gfx1201";

const FP8_E4M3_GEMM_SRC: &str =
    include_str!("../../../../kernels/src/gemm_fp8e4m3_g256_wmma.gfx1201.hip");
const FP8_E4M3_GEMM_KERNEL: &str = "gemm_fp8e4m3_g256_wmma_gfx1201";

const FP8_E4M3_GATE_UP_SRC: &str =
    include_str!("../../../../kernels/src/gemm_gate_up_fp8e4m3_g256_wmma.gfx1201.hip");
const FP8_E4M3_GATE_UP_KERNEL: &str = "gemm_gate_up_fp8e4m3_g256_wmma_gfx1201";

const FP8_E4M3_QKVZA_SRC: &str =
    include_str!("../../../../kernels/src/gemm_qkvza_fp8e4m3_g256_wmma.gfx1201.hip");
const FP8_E4M3_QKVZA_KERNEL: &str = "gemm_qkvza_fp8e4m3_g256_wmma_gfx1201";

/// A mutable GPU borrow proven to target exact gfx1201.
///
/// The constructor is intentionally available only through
/// [`Gpu::try_gfx1201`]. There is no `Deref` or escape hatch to the generic
/// context.
pub struct Gfx1201Device<'gpu> {
    gpu: &'gpu mut Gpu,
}

impl Gpu {
    /// Borrow this context as an exact-gfx1201 device.
    pub fn try_gfx1201(&mut self) -> Option<Gfx1201Device<'_>> {
        self.arch_caps
            .is_gfx1201()
            .then_some(Gfx1201Device { gpu: self })
    }

    /// Native qt=40 decode GEMV. gfx1201 only; other archs error.
    pub fn fp8_gemv_e4m3_g256(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.try_gfx1201()
            .ok_or_else(|| HipError::new(0, "FP8E4M3G256 GEMV requires gfx1201"))?
            .fp8_gemv_e4m3_g256(a, x, y, m, k)
    }

    /// Native qt=40 WMMA GEMM (packed-X). gfx1201 only; other archs error.
    pub fn fp8_gemm_e4m3_g256(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.try_gfx1201()
            .ok_or_else(|| HipError::new(0, "FP8E4M3G256 GEMM requires gfx1201"))?
            .fp8_gemm_e4m3_g256(a, x, y, m, k, n)
    }

    /// Native qt=40 fused gate+up WMMA GEMM. gfx1201 only; other archs error.
    #[allow(clippy::too_many_arguments)]
    pub fn fp8_gemm_gate_up_e4m3_g256(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.try_gfx1201()
            .ok_or_else(|| HipError::new(0, "FP8E4M3G256 fused gate+up requires gfx1201"))?
            .fp8_gemm_gate_up_e4m3_g256(a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, n)
    }

    /// Native qt=40 fused QKVZA WMMA GEMM. gfx1201 only; other archs error.
    #[allow(clippy::too_many_arguments)]
    pub fn fp8_gemm_qkvza_e4m3_g256(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.try_gfx1201()
            .ok_or_else(|| HipError::new(0, "FP8E4M3G256 fused QKVZA requires gfx1201"))?
            .fp8_gemm_qkvza_e4m3_g256(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m,
                beta_m, alpha_m, k, n,
            )
    }
}

impl Gfx1201Device<'_> {
    /// Micro-screen sister of [`Self::mq2_lloyd_moe_gate_up_ep`] which uses
    /// one workgroup per output row and loops only the rank-owned routed slots.
    #[allow(clippy::too_many_arguments)]
    pub fn mq2_lloyd_moe_gate_up_compact_ep(
        &mut self,
        expert_ptrs: &GpuTensor,
        nonowned_dummy: &GpuTensor,
        topk_indices: &GpuTensor,
        x_rot: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
    ) -> HipResult<()> {
        self.gpu.bind_thread()?;
        assert!(
            k.is_multiple_of(256),
            "gfx1201 MQ2 gate/up requires K%256=0"
        );
        self.gpu.ensure_kernel(
            MQ2_LLOYD_GATE_UP_COMPACT_EP_KERNEL,
            MQ2_LLOYD_GATE_UP_COMPACT_EP_SRC,
            MQ2_LLOYD_GATE_UP_COMPACT_EP_KERNEL,
        )?;
        let expert_ptrs_ptr = expert_ptrs.buf.as_ptr();
        let dummy_ptr = nonowned_dummy.buf.as_ptr();
        let topk_indices_ptr = topk_indices.buf.as_ptr();
        let x_ptr = x_rot.buf.as_ptr();
        let gate_ptr = y_gate.buf.as_ptr();
        let up_ptr = y_up.buf.as_ptr();
        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let k_top_i32 = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &expert_ptrs_ptr as *const _ as *mut c_void,
            &dummy_ptr as *const _ as *mut c_void,
            &topk_indices_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &gate_ptr as *const _ as *mut c_void,
            &up_ptr as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
            &k_top_i32 as *const _ as *mut c_void,
        ];
        self.gpu.launch_maybe_blob(
            MQ2_LLOYD_GATE_UP_COMPACT_EP_KERNEL,
            [m as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(expert_ptrs_ptr);
                blob.push_ptr(dummy_ptr);
                blob.push_ptr(topk_indices_ptr);
                blob.push_ptr(x_ptr);
                blob.push_ptr(gate_ptr);
                blob.push_ptr(up_ptr);
                blob.push_i32(m_i32);
                blob.push_i32(k_i32);
                blob.push_i32(k_top_i32);
                blob
            },
        )
    }

    /// DS4 TP/EP decode gate/up which keeps the fixed six-slot graph but skips
    /// MQ2 work for slots whose pointer resolves to the rank-local zero dummy.
    #[allow(clippy::too_many_arguments)]
    pub fn mq2_lloyd_moe_gate_up_ep(
        &mut self,
        expert_ptrs: &GpuTensor,
        nonowned_dummy: &GpuTensor,
        topk_indices: &GpuTensor,
        x_rot: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
    ) -> HipResult<()> {
        self.gpu.bind_thread()?;
        assert!(
            k.is_multiple_of(256),
            "gfx1201 MQ2 gate/up requires K%256=0"
        );
        self.gpu.ensure_kernel(
            MQ2_LLOYD_GATE_UP_EP_KERNEL,
            MQ2_LLOYD_GATE_UP_EP_SRC,
            MQ2_LLOYD_GATE_UP_EP_KERNEL,
        )?;
        let expert_ptrs_ptr = expert_ptrs.buf.as_ptr();
        let dummy_ptr = nonowned_dummy.buf.as_ptr();
        let topk_indices_ptr = topk_indices.buf.as_ptr();
        let x_ptr = x_rot.buf.as_ptr();
        let gate_ptr = y_gate.buf.as_ptr();
        let up_ptr = y_up.buf.as_ptr();
        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &expert_ptrs_ptr as *const _ as *mut c_void,
            &dummy_ptr as *const _ as *mut c_void,
            &topk_indices_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &gate_ptr as *const _ as *mut c_void,
            &up_ptr as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
        ];
        let bytes = k_top * (m * (k / 256) * 72 + k * 4 + m * 4);
        let timer =
            crate::profile::begin_timer(&self.gpu.hip, "gemv", MQ2_LLOYD_GATE_UP_EP_KERNEL, bytes);
        let result = self.gpu.launch_maybe_blob(
            MQ2_LLOYD_GATE_UP_EP_KERNEL,
            [m as u32, k_top as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(expert_ptrs_ptr);
                blob.push_ptr(dummy_ptr);
                blob.push_ptr(topk_indices_ptr);
                blob.push_ptr(x_ptr);
                blob.push_ptr(gate_ptr);
                blob.push_ptr(up_ptr);
                blob.push_i32(m_i32);
                blob.push_i32(k_i32);
                blob
            },
        );
        if let Some(timer) = timer {
            timer.finish(&self.gpu.hip);
        }
        result
    }

    /// Deterministic DS4 TP/EP down projection. Ownership is derived from the
    /// gate/up pointer table because non-owned down slots deliberately reuse a
    /// valid compact weight pointer.
    #[allow(clippy::too_many_arguments)]
    pub fn mq2_lloyd_moe_down_expanded_ep(
        &mut self,
        expert_ptrs: &GpuTensor,
        ownership_ptrs: &GpuTensor,
        nonowned_dummy: &GpuTensor,
        topk_indices: &GpuTensor,
        rot_batch: &GpuTensor,
        expert_outputs: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.gpu.bind_thread()?;
        assert!(k.is_multiple_of(256), "gfx1201 MQ2 down requires K%256=0");
        self.gpu.ensure_kernel(
            MQ2_LLOYD_DOWN_EXPANDED_EP_KERNEL,
            MQ2_LLOYD_DOWN_EXPANDED_EP_SRC,
            MQ2_LLOYD_DOWN_EXPANDED_EP_KERNEL,
        )?;
        let expert_ptrs_ptr = expert_ptrs.buf.as_ptr();
        let ownership_ptrs_ptr = ownership_ptrs.buf.as_ptr();
        let dummy_ptr = nonowned_dummy.buf.as_ptr();
        let topk_indices_ptr = topk_indices.buf.as_ptr();
        let rot_ptr = rot_batch.buf.as_ptr();
        let output_ptr = expert_outputs.buf.as_ptr();
        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let k_top_i32 = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &expert_ptrs_ptr as *const _ as *mut c_void,
            &ownership_ptrs_ptr as *const _ as *mut c_void,
            &dummy_ptr as *const _ as *mut c_void,
            &topk_indices_ptr as *const _ as *mut c_void,
            &rot_ptr as *const _ as *mut c_void,
            &output_ptr as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
            &k_top_i32 as *const _ as *mut c_void,
        ];
        let bytes = batch_size * k_top * (m * (k / 256) * 72 + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.gpu.hip,
            "gemv",
            MQ2_LLOYD_DOWN_EXPANDED_EP_KERNEL,
            bytes,
        );
        let result = self.gpu.launch_maybe_blob(
            MQ2_LLOYD_DOWN_EXPANDED_EP_KERNEL,
            [m as u32, k_top as u32, batch_size as u32],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(expert_ptrs_ptr);
                blob.push_ptr(ownership_ptrs_ptr);
                blob.push_ptr(dummy_ptr);
                blob.push_ptr(topk_indices_ptr);
                blob.push_ptr(rot_ptr);
                blob.push_ptr(output_ptr);
                blob.push_i32(m_i32);
                blob.push_i32(k_i32);
                blob.push_i32(k_top_i32);
                blob
            },
        );
        if let Some(timer) = timer {
            timer.finish(&self.gpu.hip);
        }
        result
    }

    /// Micro-screen sister of [`Self::mq2_lloyd_moe_down_expanded_ep`] which
    /// uses one workgroup per output row and loops rank-owned routed slots.
    #[allow(clippy::too_many_arguments)]
    pub fn mq2_lloyd_moe_down_expanded_compact_ep(
        &mut self,
        expert_ptrs: &GpuTensor,
        ownership_ptrs: &GpuTensor,
        nonowned_dummy: &GpuTensor,
        topk_indices: &GpuTensor,
        rot_batch: &GpuTensor,
        expert_outputs: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.gpu.bind_thread()?;
        assert!(k.is_multiple_of(256), "gfx1201 MQ2 down requires K%256=0");
        self.gpu.ensure_kernel(
            MQ2_LLOYD_DOWN_EXPANDED_COMPACT_EP_KERNEL,
            MQ2_LLOYD_DOWN_EXPANDED_COMPACT_EP_SRC,
            MQ2_LLOYD_DOWN_EXPANDED_COMPACT_EP_KERNEL,
        )?;
        let expert_ptrs_ptr = expert_ptrs.buf.as_ptr();
        let ownership_ptrs_ptr = ownership_ptrs.buf.as_ptr();
        let dummy_ptr = nonowned_dummy.buf.as_ptr();
        let topk_indices_ptr = topk_indices.buf.as_ptr();
        let rot_ptr = rot_batch.buf.as_ptr();
        let output_ptr = expert_outputs.buf.as_ptr();
        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let k_top_i32 = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &expert_ptrs_ptr as *const _ as *mut c_void,
            &ownership_ptrs_ptr as *const _ as *mut c_void,
            &dummy_ptr as *const _ as *mut c_void,
            &topk_indices_ptr as *const _ as *mut c_void,
            &rot_ptr as *const _ as *mut c_void,
            &output_ptr as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
            &k_top_i32 as *const _ as *mut c_void,
        ];
        self.gpu.launch_maybe_blob(
            MQ2_LLOYD_DOWN_EXPANDED_COMPACT_EP_KERNEL,
            [m as u32, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(expert_ptrs_ptr);
                blob.push_ptr(ownership_ptrs_ptr);
                blob.push_ptr(dummy_ptr);
                blob.push_ptr(topk_indices_ptr);
                blob.push_ptr(rot_ptr);
                blob.push_ptr(output_ptr);
                blob.push_i32(m_i32);
                blob.push_i32(k_i32);
                blob.push_i32(k_top_i32);
                blob
            },
        )
    }

    /// Micro-screen sister of [`Self::mq2_lloyd_moe_down_expanded_ep`] which
    /// cooperatively stages each four-entry MQ2 codebook in LDS.
    #[allow(clippy::too_many_arguments)]
    pub fn mq2_lloyd_moe_down_expanded_lds_ep(
        &mut self,
        expert_ptrs: &GpuTensor,
        ownership_ptrs: &GpuTensor,
        nonowned_dummy: &GpuTensor,
        topk_indices: &GpuTensor,
        rot_batch: &GpuTensor,
        expert_outputs: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.gpu.bind_thread()?;
        assert!(k.is_multiple_of(256), "gfx1201 MQ2 down requires K%256=0");
        self.gpu.ensure_kernel(
            MQ2_LLOYD_DOWN_EXPANDED_LDS_EP_KERNEL,
            MQ2_LLOYD_DOWN_EXPANDED_LDS_EP_SRC,
            MQ2_LLOYD_DOWN_EXPANDED_LDS_EP_KERNEL,
        )?;
        let expert_ptrs_ptr = expert_ptrs.buf.as_ptr();
        let ownership_ptrs_ptr = ownership_ptrs.buf.as_ptr();
        let dummy_ptr = nonowned_dummy.buf.as_ptr();
        let topk_indices_ptr = topk_indices.buf.as_ptr();
        let rot_ptr = rot_batch.buf.as_ptr();
        let output_ptr = expert_outputs.buf.as_ptr();
        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let k_top_i32 = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &expert_ptrs_ptr as *const _ as *mut c_void,
            &ownership_ptrs_ptr as *const _ as *mut c_void,
            &dummy_ptr as *const _ as *mut c_void,
            &topk_indices_ptr as *const _ as *mut c_void,
            &rot_ptr as *const _ as *mut c_void,
            &output_ptr as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
            &k_top_i32 as *const _ as *mut c_void,
        ];
        let bytes = batch_size * k_top * (m * (k / 256) * 72 + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.gpu.hip,
            "gemv",
            MQ2_LLOYD_DOWN_EXPANDED_LDS_EP_KERNEL,
            bytes,
        );
        let result = self.gpu.launch_maybe_blob(
            MQ2_LLOYD_DOWN_EXPANDED_LDS_EP_KERNEL,
            [m as u32, k_top as u32, batch_size as u32],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(expert_ptrs_ptr);
                blob.push_ptr(ownership_ptrs_ptr);
                blob.push_ptr(dummy_ptr);
                blob.push_ptr(topk_indices_ptr);
                blob.push_ptr(rot_ptr);
                blob.push_ptr(output_ptr);
                blob.push_i32(m_i32);
                blob.push_i32(k_i32);
                blob.push_i32(k_top_i32);
                blob
            },
        );
        if let Some(timer) = timer {
            timer.finish(&self.gpu.hip);
        }
        result
    }

    /// Block-scaled FP8 E4M3 (group-256) decode GEMV.
    ///
    /// Computes `y[row] = A[row,:] . x` for `M` rows of `K`-wide FP8 weights
    /// on the native gfx1201 `v_dot4_f32_fp8_fp8` path (one wave32 per row).
    /// Layout matches `QuantType::FP8E4M3G256`:
    /// `[16 B hdr (fp16 row_scale, fp16 row_bias)] [n_blocks×2 B fp16 scale]
    ///  [n_blocks×256 B E4M3 leaves]`, `n_blocks = K / 256`.
    ///
    /// Kernel: `kernels/src/gemv_fp8e4m3_g256.gfx1201.hip`.
    #[allow(clippy::too_many_arguments)]
    pub fn fp8_gemv_e4m3_g256(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        assert!(k % 256 == 0, "gfx1201 FP8 E4M3 GEMV requires K%256=0");
        assert!(
            x.numel() >= k && y.numel() >= m,
            "fp8_gemv_e4m3_g256 needs x>=K y>=M (x={:?} y={:?} m={m} k={k})",
            x.shape,
            y.shape
        );

        self.gpu.bind_thread()?;
        self.gpu.ensure_kernel(FP8_E4M3_GEMV_KERNEL, FP8_E4M3_GEMV_SRC, FP8_E4M3_GEMV_KERNEL)?;

        let func = &self.gpu.functions[FP8_E4M3_GEMV_KERNEL];
        let mut a_ptr = a.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        let bytes = m * (16 + (k / 256).next_multiple_of(16) * 2 + k);
        let timer = crate::profile::begin_timer(
            &self.gpu.hip,
            "gemv",
            FP8_E4M3_GEMV_KERNEL,
            bytes,
        );

        let result = unsafe {
            self.gpu.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [32, 1, 1],
                0,
                self.gpu.stream_ref(),
                &mut params,
            )
        };
        if let Some(timer) = timer {
            timer.finish(&self.gpu.hip);
        }
        result
    }

    /// Block-scaled FP8 E4M3 (group-256) WMMA GEMM (prefill).
    ///
    /// Computes `Y[N x M] = X[N x K] @ A[M x K]^T` where `A` is
    /// `QuantType::FP8E4M3G256` and `X` is F32, packed to E4M3 via
    /// [`Gpu::ensure_fp8_x`] (`pack_f32_to_fp8_gfx12`) before the MMA.
    /// Same X is cached so back-to-back GEMMs skip the pack.
    ///
    /// Kernel: `kernels/src/gemm_fp8e4m3_g256_wmma.gfx1201.hip`.
    pub fn fp8_gemm_e4m3_g256(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        assert!(k % 256 == 0, "gfx1201 FP8 E4M3 GEMM requires K%256=0");
        assert!(
            x.numel() >= n * k && y.numel() >= n * m,
            "fp8_gemm_e4m3_g256 needs X>=N*K Y>=N*M (x={:?} y={:?} n={n} m={m} k={k})",
            x.shape,
            y.shape
        );

        self.gpu.bind_thread()?;
        let x_fp8_ptr = self.gpu.ensure_fp8_x(x, n * k)?;
        self.gpu.ensure_kernel(FP8_E4M3_GEMM_KERNEL, FP8_E4M3_GEMM_SRC, FP8_E4M3_GEMM_KERNEL)?;

        let func = &self.gpu.functions[FP8_E4M3_GEMM_KERNEL];
        let mut a_ptr = a.buf.as_ptr();
        let mut x_ptr = x_fp8_ptr;
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut n_val = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let bytes = m * (16 + (k / 256).next_multiple_of(16) * 2 + k) + n * k;
        let timer = crate::profile::begin_timer(
            &self.gpu.hip,
            "gemm",
            FP8_E4M3_GEMM_KERNEL,
            bytes,
        );

        let result = unsafe {
            self.gpu.hip.launch_kernel(
                func,
                [((m + 15) / 16) as u32, ((n + 15) / 16) as u32, 1],
                [32, 1, 1],
                0,
                self.gpu.stream_ref(),
                &mut params,
            )
        };
        if let Some(timer) = timer {
            timer.finish(&self.gpu.hip);
        }
        result
    }

    /// Fused gate+up sister of [`Self::fp8_gemm_e4m3_g256`]. One launch,
    /// packed X is still `ensure_fp8_x` (cache hits the second matrix).
    #[allow(clippy::too_many_arguments)]
    pub fn fp8_gemm_gate_up_e4m3_g256(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        assert!(k % 256 == 0, "gfx1201 FP8 E4M3 fused gate+up requires K%256=0");
        assert!(
            x.numel() >= n * k
                && y_gate.numel() >= n * gate_m
                && y_up.numel() >= n * up_m,
            "fp8_gemm_gate_up_e4m3_g256 needs X>=N*K Yg>=N*gate_m Yu>=N*up_m \
             (x={:?} yg={:?} yu={:?} n={n} gate_m={gate_m} up_m={up_m} k={k})",
            x.shape,
            y_gate.shape,
            y_up.shape
        );

        self.gpu.bind_thread()?;
        let x_fp8_ptr = self.gpu.ensure_fp8_x(x, n * k)?;
        self.gpu.ensure_kernel(
            FP8_E4M3_GATE_UP_KERNEL,
            FP8_E4M3_GATE_UP_SRC,
            FP8_E4M3_GATE_UP_KERNEL,
        )?;

        let func = &self.gpu.functions[FP8_E4M3_GATE_UP_KERNEL];
        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut x_ptr = x_fp8_ptr;
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut gate_m_i = gate_m as i32;
        let mut up_m_i = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut gate_m_i as *mut _ as *mut c_void,
            &mut up_m_i as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = gate_m + up_m;
        let bytes = (gate_m + up_m) * (16 + (k / 256).next_multiple_of(16) * 2 + k) + n * k;
        let timer = crate::profile::begin_timer(
            &self.gpu.hip,
            "gemm",
            FP8_E4M3_GATE_UP_KERNEL,
            bytes,
        );

        let result = unsafe {
            self.gpu.hip.launch_kernel(
                func,
                [((total_m + 15) / 16) as u32, ((n + 15) / 16) as u32, 1],
                [32, 1, 1],
                0,
                self.gpu.stream_ref(),
                &mut params,
            )
        };
        if let Some(timer) = timer {
            timer.finish(&self.gpu.hip);
        }
        result
    }

    /// Fused QKVZA sister of [`Self::fp8_gemm_gate_up_e4m3_g256`]. One launch,
    /// packed X is still `ensure_fp8_x` (cache hits the other three matrices).
    #[allow(clippy::too_many_arguments)]
    pub fn fp8_gemm_qkvza_e4m3_g256(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        assert!(k % 256 == 0, "gfx1201 FP8 E4M3 fused QKVZA requires K%256=0");
        assert!(
            x.numel() >= n * k
                && y_qkv.numel() >= n * qkv_m
                && y_z.numel() >= n * z_m
                && y_beta.numel() >= n * beta_m
                && y_alpha.numel() >= n * alpha_m,
            "fp8_gemm_qkvza_e4m3_g256 needs X>=N*K and Y*>=N*m \
             (x={:?} yqkv={:?} yz={:?} yb={:?} ya={:?} n={n} \
              qkv_m={qkv_m} z_m={z_m} beta_m={beta_m} alpha_m={alpha_m} k={k})",
            x.shape,
            y_qkv.shape,
            y_z.shape,
            y_beta.shape,
            y_alpha.shape
        );

        self.gpu.bind_thread()?;
        let x_fp8_ptr = self.gpu.ensure_fp8_x(x, n * k)?;
        self.gpu.ensure_kernel(
            FP8_E4M3_QKVZA_KERNEL,
            FP8_E4M3_QKVZA_SRC,
            FP8_E4M3_QKVZA_KERNEL,
        )?;

        let func = &self.gpu.functions[FP8_E4M3_QKVZA_KERNEL];
        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut x_ptr = x_fp8_ptr;
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut qkv_m_i = qkv_m as i32;
        let mut z_m_i = z_m as i32;
        let mut beta_m_i = beta_m as i32;
        let mut alpha_m_i = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut qkv_m_i as *mut _ as *mut c_void,
            &mut z_m_i as *mut _ as *mut c_void,
            &mut beta_m_i as *mut _ as *mut c_void,
            &mut alpha_m_i as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let bytes = total_m * (16 + (k / 256).next_multiple_of(16) * 2 + k) + n * k;
        let timer = crate::profile::begin_timer(
            &self.gpu.hip,
            "gemm",
            FP8_E4M3_QKVZA_KERNEL,
            bytes,
        );

        let result = unsafe {
            self.gpu.hip.launch_kernel(
                func,
                [((total_m + 15) / 16) as u32, ((n + 15) / 16) as u32, 1],
                [32, 1, 1],
                0,
                self.gpu.stream_ref(),
                &mut params,
            )
        };
        if let Some(timer) = timer {
            timer.finish(&self.gpu.hip);
        }
        result
    }
}
