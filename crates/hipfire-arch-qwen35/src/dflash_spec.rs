// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Qwen3.5 DFlash / DDTree speculative-decode state and `Speculator` impl.
//!
//! Contents: [`DflashState`] (the loaded draft weights/scratch + target
//! snapshot/tape + optional [`DdtreeState`]), [`load_dflash_state`] (its
//! load-time constructor), the [`DflashSpeculator`] impl (which owns
//! `DflashState` + the divergent-render checkpoint ring) behind the arch-generic
//! [`Speculator`] trait, and [`build_dflash_speculator`] (its env-resolving
//! constructor). All types here are qwen35 + runtime types — no loader types —
//! so the loader only calls in; it never owns the DFlash mechanics.
use crate::dflash_verify_pm4::{DflashVerifyPm4, DFLASH_VERIFY_PM4_BLOCK};
use crate::qwen35::{self, DeltaNetState, Qwen35Config, Qwen35Weights, StateQuant};
use crate::speculative::{
    apply_eviction_retain_to_draft, apply_host_nucleus, apply_host_topk, sample_categorical,
    scatter_hidden_block_to_interleaved, seed_target_hidden_from_prompt_abortable,
    seed_target_hidden_suffix_abortable, softmax_temp_into, spec_step_ddtree_batched,
    spec_step_dflash, xorshift_next_unit, DdtreeScratch, DeltaNetSnapshot, GdnTape,
    HiddenStateRingBuffer, ModelSlot, SpecStepResult, VerifyScratch,
};
use hipfire_runtime::dflash::{DflashConfig, DflashScratch, DflashWeights};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::spec::{
    EvictRetain, PrefillOutcome, SpecGrammar, SpecRequestConfig, SpecStep, SpecTarget, Speculator,
};
use rdna_compute::Gpu;
use std::collections::VecDeque;
use std::path::Path;

/// τ-window trip-wire for chain DFlash block size.
///
/// Opt-in (`HIPFIRE_DFLASH_ADAPTIVE_B=1`). Default off: on Qwen3.8-27B DFlash 2
/// (trained B=8) the trip-wire 8→4 cut prose τ 2.43→1.89 and decode 44.9→38.5
/// (R9700, 2026-08-24). DFlash 1 trained at 16 can still opt in (floor 8).
struct AdaptiveBlock {
    enabled: bool,
    trained: usize,
    floor: usize,
    current: usize,
    window: VecDeque<usize>,
    tau_down: f64,
}

impl AdaptiveBlock {
    fn new(trained: usize, enabled: bool, tau_down: f64) -> Self {
        let trained = trained.max(2);
        Self {
            enabled,
            trained,
            floor: (trained / 2).max(2),
            current: trained,
            window: VecDeque::with_capacity(8),
            tau_down,
        }
    }

    fn from_env(trained: usize) -> Self {
        let trained = trained.max(2);
        let force = hipfire_config::developer_var("HIPFIRE_DFLASH_BLOCK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|b| b.clamp(2, trained));
        let enabled = force.is_none()
            && match hipfire_config::developer_var("HIPFIRE_DFLASH_ADAPTIVE_B") {
                Ok(s) => !matches!(s.as_str(), "0" | "false" | "off" | "no"),
                Err(_) => false,
            };
        let tau_down = hipfire_config::developer_var("HIPFIRE_DFLASH_ADAPTIVE_TAU_DOWN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2.5);
        if let Some(b) = force {
            eprintln!("[dflash] HIPFIRE_DFLASH_BLOCK={b} (adaptive-B off)");
            let mut s = Self::new(trained, false, tau_down);
            s.current = b;
            return s;
        }
        Self::new(trained, enabled, tau_down)
    }

    fn reset(&mut self) {
        self.window.clear();
        if self.enabled {
            self.current = self.trained;
        }
    }

    fn pick(&self) -> usize {
        self.current
    }

    fn observe(&mut self, accepted: usize) {
        if !self.enabled {
            return;
        }
        if self.window.len() == 8 {
            self.window.pop_front();
        }
        self.window.push_back(accepted);
        if self.window.len() < 4 {
            return;
        }
        let mean = self.window.iter().copied().sum::<usize>() as f64 / self.window.len() as f64;
        let next = if mean < self.tau_down {
            self.floor
        } else {
            self.trained
        };
        if next != self.current {
            eprintln!(
                "[dflash] adaptive-B {} → {next} (mean accept {mean:.2})",
                self.current
            );
            self.current = next;
        }
    }
}

/// Extract layers the retained B=16 DFlash2 verify route admits.
const DFLASH_VERIFY_PM4_EXTRACT_LAYERS: [usize; 5] = [5, 19, 33, 47, 61];

// ─── DDTree side state ────────────────────────────────────────────────

/// Side state for DDTree-mode speculative decoding.
pub struct DdtreeState {
    pub post_seed_snap: DeltaNetSnapshot,
    pub scratch: DdtreeScratch,
    pub budget: usize,
    pub topk: usize,
}

// ─── DFlash state ─────────────────────────────────────────────────────

/// Optional DFlash speculative-decoding state.
pub struct DflashState {
    pub draft_config: DflashConfig,
    pub draft_weights: DflashWeights,
    pub draft_scratch: DflashScratch,
    pub hidden_rb: HiddenStateRingBuffer,
    pub verify_scratch: VerifyScratch,
    pub target_snap: DeltaNetSnapshot,
    pub gdn_tape: GdnTape,
    pub target_hidden_host: Vec<f32>,
    pub ctx_capacity: usize,
    pub block_size: usize,
    pub ddtree: Option<DdtreeState>,
    /// Retained-PM4 route for the fixed B=16 chain target-verify forward.
    /// Always present; `Disabled` when admission fails (no controller held).
    pub verify_pm4: DflashVerifyPm4,
}

impl DflashState {
    /// Destructured without `..` on purpose: a field added later that owns GPU
    /// memory becomes a compile error here instead of a per-load leak.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let DflashState {
            draft_config: _,
            draft_weights,
            draft_scratch,
            hidden_rb,
            verify_scratch,
            target_snap,
            gdn_tape,
            target_hidden_host: _,
            ctx_capacity: _,
            block_size: _,
            ddtree,
            mut verify_pm4,
        } = self;
        // Must prove no retained IB is in flight before any captured owner is
        // freed. Unknown quiescence → intentional leak; daemon restart is the
        // containment path.
        if let Err(reason) = verify_pm4.shutdown() {
            eprintln!(
                "dflash verify PM4: refusing free after unknown quiescence: {reason}"
            );
            std::mem::forget((
                verify_pm4,
                draft_weights,
                draft_scratch,
                hidden_rb,
                verify_scratch,
                target_snap,
                gdn_tape,
                ddtree,
            ));
            let _ = gpu;
            return;
        }
        drop(verify_pm4);
        draft_weights.free_gpu(gpu);
        draft_scratch.free_gpu(gpu);
        hidden_rb.free_gpu(gpu);
        verify_scratch.free_gpu(gpu);
        target_snap.free_gpu(gpu);
        gdn_tape.free_gpu(gpu);
        if let Some(dd) = ddtree {
            dd.post_seed_snap.free_gpu(gpu);
            dd.scratch.free_gpu(gpu);
        }
    }
}

// ─── DFlash state load ────────────────────────────────────────────────

/// Default ceiling for the DFlash draft's context-indexed structures
/// (`target_hidden` [L × extract×hidden], the per-layer K/V caches, the
/// hidden ring, `mq_x_rot`, and the host hidden log). Serve loads default
/// `max_seq` to 32768+, which sized ALL of these to 32K rows — on a 27B
/// target with a 5-layer-extract MQ4 draft that is ~11 GB of draft-side
/// VRAM, vs ~1.4 GB at the ≤4K contexts DFlash benches actually run. The
/// draft only affects acceptance rate (verify is target-gated), so a
/// request that outgrows the cap simply falls back to AR in the daemon —
/// emitted tokens are never at risk. `HIPFIRE_DFLASH_CTX_CAP=0` opts out
/// (legacy uncapped behaviour); any other value overrides the ceiling.
///
/// A cap that cannot fit in remaining VRAM is not a silent AR fallback:
/// [`load_dflash_state`] halves the Legacy cap down to this default,
/// drains the GPU pool so hipFree actually returns memory, and only then
/// errors (the loader may still AR). Windowed mode does not shrink `W`.
pub const DEFAULT_DFLASH_CTX_CAP: usize = 8192;

/// Next Legacy DFlash ctx-cap to try after a VRAM OOM. Floor is
/// [`DEFAULT_DFLASH_CTX_CAP`]; `None` means the current cap is already
/// at/under that floor (caller should fail, not retry forever).
pub fn next_legacy_dflash_cap_after_oom(current: usize) -> Option<usize> {
    let next = (current / 2).max(DEFAULT_DFLASH_CTX_CAP);
    (next < current).then_some(next)
}

fn is_dflash_vram_oom(err: &str) -> bool {
    err.contains("HipError(2)")
        || err.contains("hipError=2")
        || err.to_ascii_lowercase().contains("out of memory")
}

struct DflashRuntimeAlloc {
    draft_scratch: DflashScratch,
    hidden_rb: HiddenStateRingBuffer,
    verify_scratch: VerifyScratch,
    target_snap: DeltaNetSnapshot,
    gdn_tape: GdnTape,
    target_hidden_host: Vec<f32>,
    ddtree: Option<DdtreeState>,
}

/// Allocate draft scratch + target-side DFlash GPU state. Does **not** own
/// draft weights — the caller retries this on VRAM OOM without reloading
/// the 3+ GB weight tensors.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn alloc_dflash_runtime(
    gpu: &mut Gpu,
    draft_config: &DflashConfig,
    has_mq: bool,
    block_size: usize,
    max_n: usize,
    staging_max_batch: usize,
    ctx_capacity: usize,
    window: Option<usize>,
    requested_ctx: usize,
    target_config: &Qwen35Config,
    target_dn: &DeltaNetState,
    ddtree_budget: usize,
    ddtree_topk_param: Option<usize>,
) -> Result<DflashRuntimeAlloc, String> {
    macro_rules! or_free {
        ($e:expr, $ctx:expr $(, $owned:expr)* $(,)?) => {
            match $e {
                Ok(v) => v,
                Err(e) => {
                    $($owned.free_gpu(gpu);)*
                    let ctx: &str = $ctx;
                    return Err(if ctx.is_empty() {
                        format!("{e}")
                    } else {
                        format!("{ctx}: {e}")
                    });
                }
            }
        };
    }
    let draft_scratch = or_free!(
        match window {
            Some(w) => DflashScratch::new_windowed(
                gpu,
                draft_config,
                block_size,
                w,
                // w_full UNBOUNDED: the last (full-attention) layer's ring spans
                // the whole supported context, matching the artifact's
                // `layer_types: [sliding x(n-1), full_attention]` semantics — the
                // NInfer reference keeps one layer genuinely unbounded. The prior
                // `requested_ctx.min(4 * w)` made the "full" layer a 4W-window
                // (8192 rows at W=2048), so past 8K NO layer had full reach.
                // Ring VRAM scales with requested_ctx (~270 MB at 32K rows,
                // kvd=1024, f32 — see DflashScratch::new_windowed docs).
                requested_ctx,
                requested_ctx,
                has_mq,
            ),
            None => {
                // `with_mq` allocates the FWHT rotation scratch (mq_x_rot)
                // that `gemm_dispatch` requires for MQ4/MQ3/MQ6 draft weights.
                DflashScratch::new_with_mq(
                    gpu,
                    draft_config,
                    block_size,
                    ctx_capacity,
                    has_mq,
                )
            }
        },
        "",
    );
    let hidden_rb = or_free!(
        HiddenStateRingBuffer::new(
            gpu,
            target_config.n_layers,
            draft_config.num_extract(),
            target_config.dim,
            ctx_capacity,
            staging_max_batch,
        ),
        "HiddenStateRingBuffer::new",
        draft_scratch,
    );
    let hidden_k = target_config.dim.next_power_of_two();
    let verify_scratch = or_free!(
        VerifyScratch::with_prefill(
            gpu,
            max_n,
            target_config.dim,
            target_config.vocab_size,
            hidden_k,
            target_config,
        ),
        "VerifyScratch::with_prefill",
        hidden_rb,
        draft_scratch,
    );
    let target_snap = or_free!(
        DeltaNetSnapshot::new_for(gpu, target_dn),
        "DeltaNetSnapshot::new_for",
        verify_scratch,
        hidden_rb,
        draft_scratch,
    );
    let gdn_tape = or_free!(
        GdnTape::new_for_config(gpu, target_config, max_n),
        "GdnTape::new_for_config",
        target_snap,
        verify_scratch,
        hidden_rb,
        draft_scratch,
    );
    let target_hidden_host = vec![0.0f32; ctx_capacity * target_config.dim];
    let ddtree = if ddtree_budget > 0 {
        let topk: usize = gpu.flags.ddtree_topk.or(ddtree_topk_param).unwrap_or(4);
        let post_seed_snap = or_free!(
            DeltaNetSnapshot::new_for(gpu, target_dn),
            "",
            gdn_tape,
            target_snap,
            verify_scratch,
            hidden_rb,
            draft_scratch,
        );
        let scratch = or_free!(
            DdtreeScratch::new(gpu, ddtree_budget),
            "DdtreeScratch::new",
            post_seed_snap,
            gdn_tape,
            target_snap,
            verify_scratch,
            hidden_rb,
            draft_scratch,
        );
        Some(DdtreeState {
            post_seed_snap,
            scratch,
            budget: ddtree_budget,
            topk,
        })
    } else {
        None
    };
    Ok(DflashRuntimeAlloc {
        draft_scratch,
        hidden_rb,
        verify_scratch,
        target_snap,
        gdn_tape,
        target_hidden_host,
        ddtree,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn load_dflash_state(
    draft_path: &str,
    ctx_capacity: usize,
    target_config: &Qwen35Config,
    target_dn: &DeltaNetState,
    gpu: &mut Gpu,
    // DDTree draft tuning forwarded by the loader from the unified spec config
    // (CLI `--ddtree-budget` / `--ddtree-topk`). Env wins, else these, else default.
    ddtree_budget_param: Option<usize>,
    ddtree_topk_param: Option<usize>,
    // CASK eviction active for this load. Windowed draft mode refuses the
    // combination (the eviction rebuild re-projects rows the window has
    // already dropped) and falls back to Legacy — gather-compact over the
    // rings is a follow-up.
    eviction_active: bool,
    // Retained-PM4 admission facts owned by the loader/target, not the draft.
    target_weights: &Qwen35Weights,
    kv_is_q8: bool,
    single_gpu: bool,
    // True when adaptive KV is engaged for this load (tier-switching cache).
    // Must be false for retained-PM4 admission.
    adaptive_kv: bool,
) -> Result<DflashState, String> {
    let requested_ctx = ctx_capacity;
    // Open the draft container up-front: its declared SWA window is the
    // DEFAULT window (below), so the artifact must be parsed before the
    // windowed-vs-Legacy decision.
    let draft_hfq = HfqFile::open(Path::new(draft_path)).map_err(|e| format!("{e}"))?;
    let draft_config = DflashConfig::from_hfq(&draft_hfq)
        .ok_or_else(|| "draft: failed to parse DflashConfig from HFQ metadata".to_string())?;

    // Windowed draft context:
    //   - Legacy DFlash (n−1 sliding + last full): layers 0..n−2 attend over
    //     the last `window` rows, the last layer over the entire supported
    //     context (`w_full = requested_ctx`). Draft VRAM pins at W.
    //   - DFlash2 (all layers sliding): every extract layer shares the same
    //     W ring; there is no full-attention last layer. `new_windowed`
    //     already skips the long-reach ring when `all_layers_sliding`.
    // Requests past the window degrade τ instead of hitting Legacy AR fallback.
    //
    // The window DEFAULTS to what the draft artifact declares it was trained
    // with (`config.sliding_window`, honoured only when `use_sliding_window`
    // is true and `layer_types` match a split we implement — see
    // `DflashConfig::declared_window` / `all_layers_sliding`). That is the
    // only width correct by construction.
    //   HIPFIRE_DFLASH_WINDOW=<rows>  explicit override (warns on mismatch)
    //   HIPFIRE_DFLASH_WINDOW=0       explicit Legacy (cap + AR fallback)
    //   unset                         draft-declared window, else Legacy
    let window = match hipfire_config::developer_var("HIPFIRE_DFLASH_WINDOW")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        Some(0) => None,
        Some(w) => {
            if let Some(declared) = draft_config.declared_window {
                if declared != w {
                    eprintln!(
                        "  DFlash window override {w} != draft-declared sliding_window \
                         {declared} — the draft was trained at {declared}; acceptance may \
                         degrade (output stays verify-exact)"
                    );
                }
            }
            Some(w)
        }
        None => draft_config.declared_window,
    };
    let window = match (window, eviction_active) {
        (Some(w), true) => {
            eprintln!(
                "  DFlash windowed mode ({w}) disabled: CASK eviction rebuild is not \
                 ring-aware — falling back to Legacy capped mode"
            );
            None
        }
        (w, _) => w,
    };
    let mut ctx_capacity = match window {
        Some(w) => w,
        None => match hipfire_config::developer_var("HIPFIRE_DFLASH_CTX_CAP")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
        {
            Some(0) => ctx_capacity, // explicit opt-out: legacy uncapped
            Some(cap) => ctx_capacity.min(cap),
            None => ctx_capacity.min(DEFAULT_DFLASH_CTX_CAP),
        },
    };
    if let Some(w) = window {
        let from_meta = if draft_config.declared_window == Some(w) {
            " [from draft metadata]"
        } else {
            ""
        };
        if draft_config.all_layers_sliding {
            eprintln!(
                "  DFlash2 draft windowed: all {} layers sliding at W={w} \
                 (no full-attention layer; draft VRAM pinned at W; \
                 HIPFIRE_DFLASH_WINDOW=0 for Legacy){from_meta}",
                draft_config.n_layers
            );
        } else {
            eprintln!(
                "  DFlash draft windowed: SWA W={w} rows on layers 0..n-2, full-attention \
                 last layer over all {} rows (draft VRAM pinned at W; HIPFIRE_DFLASH_WINDOW=0 for Legacy){from_meta}",
                requested_ctx
            );
        }
    } else if ctx_capacity < requested_ctx {
        eprintln!(
            "  DFlash draft ctx capped: {} -> {} rows (draft-side VRAM scales with this; \
             HIPFIRE_DFLASH_CTX_CAP=0 for uncapped, or set a larger cap)",
            requested_ctx, ctx_capacity
        );
    }
    // Every step below owns GPU memory the later ones need. A bare `?` drops
    // those without freeing (no `Drop` on the GPU-owning types), so a failed
    // DFlash load stays resident and the AR fallback it announces then OOMs.
    // `or_free` returns the buffers to the GpuPool; the OOM-retry / loader
    // paths call `drain_pool` so hipFree actually returns VRAM.
    macro_rules! or_free {
        ($e:expr, $ctx:expr $(, $owned:expr)* $(,)?) => {
            match $e {
                Ok(v) => v,
                Err(e) => {
                    $($owned.free_gpu(gpu);)*
                    let ctx: &str = $ctx;
                    return Err(if ctx.is_empty() {
                        format!("{e}")
                    } else {
                        format!("{ctx}: {e}")
                    });
                }
            }
        };
    }
    let draft_weights = or_free!(DflashWeights::load(gpu, &draft_hfq, &draft_config), "");
    let block_size = draft_config.runtime_block_size();
    if block_size != draft_config.block_size {
        eprintln!(
            "  DFlash2 runtime block: {} -> {} (selector/conv path is length-generic)",
            draft_config.block_size, block_size
        );
    }
    // DDTree verify batches up to `budget + 1` slots (seed + budget nodes), which
    // can exceed the chain block_size+1. Size verify_scratch / GdnTape / hidden
    // staging for the larger of the two so ddtree-mode serve doesn't overflow
    // ("verify_scratch max_n < b" panic). budget=0 ⇒ chain-only, unchanged.
    // Resolved through FeatureFlags (env override) so the ddtree budget has a
    // single parser shared with the dense path — env wins, else the CLI param,
    // else 0 (chain-only). An explicit `HIPFIRE_DDTREE_BUDGET=0` reads as None
    // (unset) here and falls through to the param, matching the dense semantics.
    let mut ddtree_budget: usize = gpu.flags.ddtree_budget.or(ddtree_budget_param).unwrap_or(0);
    // DFlash2 is chain-only: never construct DDTree (independent-marginal
    // tree would bypass the candidate selector).
    if draft_weights.has_candidate_selector() {
        if ddtree_budget > 0 {
            eprintln!(
                "  DFlash2 candidate selector: DDTree disabled (chain-only; \
                 independent-marginal tree would bypass the selector)"
            );
        }
        ddtree_budget = 0;
    }
    let max_n = (block_size + 1).max(ddtree_budget + 1);
    // `with_mq` allocates the FWHT rotation scratch (mq_x_rot) that
    // `gemm_dispatch` requires for MQ4/MQ3/MQ6 draft weights. The carrier
    // refactor regressed this to the `with_mq=false` `::new` constructor →
    // panic "MQ4 dispatch requires mq_x_rot scratch" on any MQ-quantized draft.
    let draft_scratch = or_free!(
        match window {
            Some(w) => DflashScratch::new_windowed(
                gpu,
                &draft_config,
                block_size,
                w,
                // Split drafts: last (full-attention) layer's ring spans the
                // whole supported context. All-sliding DFlash2: `new_windowed`
                // ignores w_full and pins every layer at W.
                requested_ctx,
                requested_ctx,
                draft_weights.has_mq,
            ),
            None => DflashScratch::new_with_mq(
                gpu,
                &draft_config,
                block_size,
                ctx_capacity,
                draft_weights.has_mq,
            ),
        },
        "",
        draft_weights,
    );
    let _ = draft_hfq;
    // The hidden-ring STAGING buffers must hold one prefill chunk. Verify
    // cycles seed only `max_n` (= block_size+1) rows, but the prompt seed
    // (`seed_target_hidden_from_prompt_abortable`) prefills the prompt in
    // chunks of up to `PREFILL_MAX_BATCH` and captures each into staging via
    // `write_rows_to_staging` (whose `n <= max_batch` guard is a debug_assert,
    // silent in release). Sizing staging to only `max_n` overflowed the d2d
    // copy on any prompt longer than block_size+1 tokens. Size it to the
    // larger of the two so both paths fit.
    let staging_max_batch = max_n.max(qwen35::PREFILL_MAX_BATCH);
    // Hidden extraction must use checkpoint target_layer_ids exactly — validate
    // against target layer range and allocate by the explicit list (not the
    // evenly-spaced fallback). Checkpoint [5,19,33,47,61] is thus captured verbatim.
    for &lid in &draft_config.target_layer_ids {
        if lid >= target_config.n_layers {
            // Free owned GPU state before returning; or_free! would have done
            // this for the ring allocation failure path — do it manually here.
            draft_scratch.free_gpu(gpu);
            draft_weights.free_gpu(gpu);
            return Err(format!(
                "draft target_layer_ids contains {} >= num_target_layers {}",
                lid, target_config.n_layers
            ));
        }
    }
    let hidden_rb = loop {
        match HiddenStateRingBuffer::new_for_layers(
            gpu,
            &draft_config.target_layer_ids,
            target_config.dim,
            ctx_capacity,
            staging_max_batch,
        ) {
            Ok(rb) => break rb,
            Err(e) => {
                gpu.drain_pool();
                let es = format!("HiddenStateRingBuffer::new_for_layers: {e}");
                if window.is_none() && is_dflash_vram_oom(&es) {
                    if let Some(next) = next_legacy_dflash_cap_after_oom(ctx_capacity) {
                        eprintln!(
                            "  DFlash draft OOM at {ctx_capacity} rows ({e}); \
                             retrying Legacy cap {next} rows"
                        );
                        ctx_capacity = next;
                        continue;
                    }
                }
                draft_scratch.free_gpu(gpu);
                draft_weights.free_gpu(gpu);
                gpu.drain_pool();
                return Err(es);
            }
        }
    };
    let hidden_k = target_config.dim.next_power_of_two();
    let verify_scratch = or_free!(
        VerifyScratch::with_prefill(
            gpu,
            max_n,
            target_config.dim,
            target_config.vocab_size,
            hidden_k,
            target_config,
        ),
        "VerifyScratch::with_prefill",
        hidden_rb,
        draft_scratch,
        draft_weights,
    );
    let target_snap = or_free!(
        DeltaNetSnapshot::new_for(gpu, target_dn),
        "DeltaNetSnapshot::new_for",
        verify_scratch,
        hidden_rb,
        draft_scratch,
        draft_weights,
    );
    let gdn_tape = or_free!(
        GdnTape::new_for_config(gpu, target_config, max_n),
        "GdnTape::new_for_config",
        target_snap,
        verify_scratch,
        hidden_rb,
        draft_scratch,
        draft_weights,
    );
    let target_hidden_host = vec![0.0f32; ctx_capacity * target_config.dim];
    // DDTree (budget read once above, used for scratch sizing).
    let ddtree = if ddtree_budget > 0 {
        let topk: usize = gpu.flags.ddtree_topk.or(ddtree_topk_param).unwrap_or(4);
        let post_seed_snap = or_free!(
            DeltaNetSnapshot::new_for(gpu, target_dn),
            "",
            gdn_tape,
            target_snap,
            verify_scratch,
            hidden_rb,
            draft_scratch,
            draft_weights,
        );
        let scratch = or_free!(
            DdtreeScratch::new(gpu, ddtree_budget),
            "DdtreeScratch::new",
            post_seed_snap,
            gdn_tape,
            target_snap,
            verify_scratch,
            hidden_rb,
            draft_scratch,
            draft_weights,
        );
        Some(DdtreeState {
            post_seed_snap,
            scratch,
            budget: ddtree_budget,
            topk,
        })
    } else {
        None
    };
    // Retained-PM4 admission (default-off). Pure gates + env opt-in; Disabled
    // holds no controller so HIP behavior stays byte-identical when not armed.
    let env_opt_in = hipfire_config::developer_var("HIPFIRE_DFLASH_VERIFY_PM4")
        .ok()
        .as_deref()
        == Some("1");
    let moe_router_present = verify_scratch
        .prefill_batch
        .as_ref()
        .map(|p| p.moe_router_logits_batch.is_some())
        .unwrap_or(false);
    let pbs_eligible_b16 = qwen35::prefill_batch_pbs_eligible(
        target_weights,
        target_config,
        target_dn,
        DFLASH_VERIFY_PM4_BLOCK,
        &gpu.arch,
        moe_router_present,
    );
    let pbs_max_batch = verify_scratch
        .prefill_batch
        .as_ref()
        .map(|p| p.max_batch)
        .unwrap_or(0);
    // The draft's selector/dynamic-conv shape is deliberately NOT a gate: the
    // draft forward is outside the tape, so DFlash2 and legacy DFlash yield an
    // identical target verify body.
    let verify_pm4 = match admit_dflash_verify_pm4(
        env_opt_in,
        &gpu.arch,
        single_gpu,
        target_config.num_experts,
        kv_is_q8,
        matches!(target_dn.quant, StateQuant::Q8),
        // finish_qwen35_load suppresses generic DFlash under adaptive KV, but
        // admit fail-closed so a future load path cannot arm a static-Q8 tape
        // against a tier-switching cache.
        !adaptive_kv,
        &hidden_rb.extract_layers,
        block_size,
        &draft_config.target_layer_ids,
        ddtree.is_some(),
        verify_scratch.prefill_batch.is_some(),
        pbs_eligible_b16,
        verify_scratch.max_n,
        pbs_max_batch,
        hidden_rb.max_batch,
        gdn_tape.max_n,
    ) {
        Ok(()) => {
            eprintln!(
                "  DFlash verify PM4: armed (B={}, exact {})",
                DFLASH_VERIFY_PM4_BLOCK, gpu.arch
            );
            DflashVerifyPm4::armed()
        }
        Err(reason) => {
            eprintln!("  DFlash verify PM4: disabled ({reason})");
            DflashVerifyPm4::disabled(reason)
        }
    };
    Ok(DflashState {
        draft_config,
        draft_weights,
        draft_scratch,
        hidden_rb,
        verify_scratch,
        target_snap,
        gdn_tape,
        target_hidden_host,
        // Windowed mode reports the TARGET's physical capacity: the draft
        // degrades τ past its window instead of refusing, so the spec
        // loop's overflow guard and the daemon's capacity fallback track
        // the true cliff, not the window.
        ctx_capacity: if window.is_some() {
            requested_ctx
        } else {
            ctx_capacity
        },
        block_size,
        ddtree,
        verify_pm4,
    })
}

// ─── DflashSpeculator ───────────────────────────────────────────────────

/// Lower a qwen35 `SpecStepResult` onto the arch-generic `SpecStep`.
///
/// The daemon-called `spec_step_*` build `committed = [seed, drafts.., bonus]`,
/// so `committed[1..]` is exactly the daemon's `committed_tail` (the tokens
/// emitted this window) and its length is `accepted + 1` — which is why the
/// unified loop advances `position` by `emit.len()`.
fn lower_qwen35(r: SpecStepResult) -> SpecStep {
    SpecStep::new(
        r.committed[1..].iter().copied(),
        r.bonus_token,
        r.drafted.len(),
        r.accepted,
    )
}

/// DFlash / DDTree speculator: wraps the qwen35 `spec_step_*` chain/tree
/// kernels behind the arch-generic [`Speculator`] trait. Chain-vs-tree is an
/// internal detail resolved at build (`ddtree` presence comes from the loaded
/// `DflashState`).
///
/// Owns the `DflashState` moved out of `LoadedModel.dflash`, plus the divergent-
/// render DeltaNet checkpoint ring folded in from `LoadedModel.dflash_checkpoints`.
pub struct DflashSpeculator {
    df: DflashState,
    rng_state: u64,
    /// Per-request sampling, set via `set_sampling` before each step loop and
    /// applied in the chain-mode `spec_step_dflash` branch of `step`. Default
    /// greedy (temp 0 / top_p 1 / top_k 0 / cactus 0) → argmax-accept, the
    /// historical DFlash posture, so an unconfigured speculator (or the
    /// greedy-only DDTree branches) decode greedily. Mirrors spec-graph's old
    /// inline `generate_dflash` call, which threaded the request temp/top_p/top_k
    /// into the same four `spec_step_dflash` args.
    sample_temp: f32,
    sample_top_p: f32,
    sample_top_k: usize,
    sample_cactus: f32,
    /// Divergent-render checkpoint ring. Populated by `prefill`'s seed when
    /// `resume_enabled`; freed on `reset`/`free`.
    checkpoints: Vec<(usize, DeltaNetSnapshot)>,
    resume_enabled: bool,
    ck_interval: usize,
    ck_cap: usize,
    /// Chain-mode block-size trip-wire. Unused on the DDTree arm.
    adaptive: AdaptiveBlock,
}

impl DflashSpeculator {
    /// `resume_enabled`/`ck_interval`/`ck_cap` mirror the daemon's
    /// `ckpt_resume_enabled()`/`ckpt_interval()`/`ckpt_max()` — passed in by
    /// `build_dflash_speculator` so `new` itself is env-free (and unit-testable).
    pub fn new(
        df: DflashState,
        resume_enabled: bool,
        ck_interval: usize,
        ck_cap: usize,
        adaptive: AdaptiveBlock,
    ) -> Self {
        Self {
            df,
            // Same fixed seed the daemon's DFlash loop used. `set_sampling`
            // re-seeds it to this value per request (matching spec-graph's local
            // `let mut rng_state = 0x13579BDF` per `generate_dflash` call) so a
            // sampled request is deterministic given its seed; greedy decode does
            // not consume it.
            rng_state: 0x13579BDF,
            // Greedy by default until a request calls `set_sampling`.
            sample_temp: 0.0,
            sample_top_p: 1.0,
            sample_top_k: 0,
            sample_cactus: 0.0,
            checkpoints: Vec::new(),
            resume_enabled,
            ck_interval,
            ck_cap,
            adaptive,
        }
    }

    /// Borrow the retained-PM4 verify route (report / phase inspection).
    pub fn verify_pm4(&self) -> &DflashVerifyPm4 {
        &self.df.verify_pm4
    }

    /// Mutable borrow for the chain verify path (`Some(&mut self.df.verify_pm4)`).
    pub fn verify_pm4_mut(&mut self) -> &mut DflashVerifyPm4 {
        &mut self.df.verify_pm4
    }
}

impl Speculator for DflashSpeculator {
    fn name(&self) -> &'static str {
        "dflash"
    }

    fn prefill(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        prompt_tokens: &[u32],
        prefill_tokens: &[u32],
        prefill_start: usize,
        cache_hit: bool,
        resume_from: Option<usize>,
        abort: &dyn Fn() -> bool,
    ) -> Result<PrefillOutcome, String> {
        let slot = target
            .as_any_mut()
            .downcast_mut::<ModelSlot>()
            .ok_or("DflashSpeculator: target is not a Qwen3.5 ModelSlot")?;

        // Mirror the daemon's pre-seed drafter setup (generate_dflash 4064-4072):
        // always clear the host hidden buffer; on a full prefill drop the draft's
        // upload/projection tracking. On a cache HIT it is PRESERVED so the draft
        // reuses the cached [0..start_pos] projections and only projects the suffix.
        self.df.target_hidden_host.clear();
        if !cache_hit {
            self.df.draft_scratch.reset_upload_tracking();
        }

        // Seed the target's hidden state into the drafter ring (chunked prefill
        // with hidden extraction). Cache hit → seed only the suffix from
        // `prefill_start`, reusing the prior turn's KV + recurrent state; miss →
        // seed the full prompt (the seed fn resets target state itself).
        let (ck_interval, ck_cap) = (self.ck_interval, self.ck_cap);
        let ckpt_sink = if self.resume_enabled {
            Some(&mut self.checkpoints)
        } else {
            None
        };
        let aborted = if cache_hit {
            seed_target_hidden_suffix_abortable(
                gpu,
                slot,
                &mut self.df.hidden_rb,
                prefill_tokens,
                prefill_start,
                abort,
                ckpt_sink,
                ck_interval,
                ck_cap,
            )
        } else {
            seed_target_hidden_from_prompt_abortable(
                gpu,
                slot,
                &mut self.df.hidden_rb,
                &mut self.df.target_hidden_host,
                prefill_tokens,
                abort,
                ckpt_sink,
                ck_interval,
                ck_cap,
            )
        }
        .map_err(|e| e.to_string())?;
        if aborted {
            // Caller resets conversation state + emits aborted/done; the slot
            // guard restores the target bundle on the way out.
            return Ok(PrefillOutcome::Aborted);
        }

        // Prime/extend the draft's GPU target_hidden buffer. On a hit, scatter
        // only the suffix rows at `prefill_start` (the prefix is preserved);
        // on a miss, scatter all prompt rows from 0.
        let (scatter_off, scatter_len) = if cache_hit {
            (prefill_start, prefill_tokens.len())
        } else {
            (0, prompt_tokens.len())
        };
        if let Err(e) = scatter_hidden_block_to_interleaved(
            gpu,
            &self.df.hidden_rb,
            &self.df.draft_scratch.target_hidden,
            scatter_off,
            scatter_len,
            scatter_len,
            self.df.draft_scratch.ctx_modulus(),
        ) {
            eprintln!("[dflash] scatter failed: {e} — falling back to per-cycle upload");
        }
        // Windowed cold prefill longer than W: the 4+1 split's last (full)
        // layer still needs K/V for every prompt row. `draft_seed_backfill`
        // is a no-op for all-sliding DFlash2 (every layer shares the W ring)
        // and for prompt_len <= W. Keep the call — it is safe on both splits.
        if !cache_hit {
            hipfire_runtime::dflash::draft_seed_backfill(
                gpu,
                &self.df.draft_weights,
                &self.df.draft_config,
                &mut self.df.draft_scratch,
                &self.df.target_hidden_host,
                prompt_tokens.len(),
            )
            .map_err(|e| e.to_string())?;
        }
        self.df.draft_scratch.thlog.seed_prompt(prompt_tokens.len());
        if let Some(ckpt) = resume_from {
            // Divergent rows [ckpt..len) were just overwritten; drop the draft's
            // projection cursor so the first spec step re-projects from `ckpt`.
            self.df.draft_scratch.thlog.set_resume_checkpoint(ckpt);
        }

        // First emit = target draw at the final prompt position (seed already
        // ran the per-token forward; scratch.logits holds the post-prompt logits).
        // temp≈0 stays the historical host argmax fold (byte-identical greedy).
        // temp>0 uses the same host nucleus sampler as chain DFlash verify so the
        // post-prefill seed is not a special greedy exception on distribution-
        // preserving requests.
        let first_logits = gpu
            .download_f32(&slot.scratch.logits)
            .map_err(|e| e.to_string())?;
        let first_token = if self.sample_temp <= 1e-6 {
            first_logits
                .iter()
                .enumerate()
                .fold((0u32, f32::NEG_INFINITY), |(best, bv), (i, &v)| {
                    if v > bv {
                        (i as u32, v)
                    } else {
                        (best, bv)
                    }
                })
                .0
        } else {
            let mut probs = Vec::with_capacity(first_logits.len());
            softmax_temp_into(&first_logits, self.sample_temp, &mut probs);
            // DDTree SWOR honors temperature only (matches step's tree arm).
            // Chain mode applies the same host top_k + nucleus cuts as
            // `spec_step_dflash` so the seed is AR-at-(top_k,top_p).
            if self.df.ddtree.is_none() {
                if self.sample_top_k > 0 && self.sample_top_k < probs.len() {
                    apply_host_topk(&mut probs, self.sample_top_k);
                }
                if self.sample_top_p < 0.999 {
                    apply_host_nucleus(&mut probs, self.sample_top_p);
                }
            }
            let u = xorshift_next_unit(&mut self.rng_state);
            sample_categorical(&probs, u)
        };
        Ok(PrefillOutcome::Ready { first_token })
    }

    /// Forced tokens (think-budget force-close) must land in the drafter's
    /// per-position `target_hidden` cache, not just the target's KV. Seeding via
    /// the same suffix path the prompt-cache HIT uses advances the target WITH
    /// hidden extraction, so the rows exist and `thlog` stays contiguous.
    ///
    /// Skipping this is what previously left an uninitialized (NaN) hole at the
    /// forced positions: the next draft forward read it, produced all-NaN logits,
    /// and `argmax` collapsed to token 0 — τ went to 0 for the rest of the
    /// session and stayed dead across prompt-cache HITs.
    fn on_forced_advance(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        tokens: &[u32],
        start_pos: usize,
        abort: &dyn Fn() -> bool,
    ) -> Result<bool, String> {
        if tokens.is_empty() {
            return Ok(true);
        }
        let slot = target
            .as_any_mut()
            .downcast_mut::<ModelSlot>()
            .ok_or("DflashSpeculator: target is not a Qwen3.5 ModelSlot")?;
        let aborted = seed_target_hidden_suffix_abortable(
            gpu,
            slot,
            &mut self.df.hidden_rb,
            tokens,
            start_pos,
            abort,
            None,
            self.ck_interval,
            self.ck_cap,
        )
        .map_err(|e| e.to_string())?;
        if aborted {
            // Caller tears the request down; leaving the rows unwritten is fine
            // because the drafter state is reset on the way out.
            return Ok(true);
        }
        scatter_hidden_block_to_interleaved(
            gpu,
            &self.df.hidden_rb,
            &self.df.draft_scratch.target_hidden,
            start_pos,
            tokens.len(),
            tokens.len(),
            self.df.draft_scratch.ctx_modulus(),
        )
        .map_err(|e| e.to_string())?;
        let co = slot.kv_cache_mut().map(|kv| kv.compact_offset).unwrap_or(0) as i32;
        self.df
            .draft_scratch
            .thlog
            .append_committed(start_pos, tokens.len(), co);
        Ok(true)
    }

    /// Temp>0 verify is distribution-correct on the ddtree-batched arm (SWOR)
    /// and on DFlash2 selector-chain rejection sampling. Legacy chain without
    /// a selector stays greedy-only at this gate.
    fn supports_temp_verify(&self) -> bool {
        self.df.ddtree.is_some() || self.df.draft_weights.has_candidate_selector()
    }

    /// Faithful top_p/top_k nucleus only on the DFlash2 candidate-selector chain.
    /// DDTree SWOR returns false so route selection still blocks user-explicit
    /// non-temperature controls for the tree path.
    fn supports_chain_nucleus_verify(&self) -> bool {
        self.df.draft_weights.has_candidate_selector()
    }

    fn step(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        emitted: &[u32],
        _grammar: Option<&mut dyn SpecGrammar>,
        temp: f32,
        max_emit: usize,
    ) -> Result<SpecStep, String> {
        let slot = target
            .as_any_mut()
            .downcast_mut::<ModelSlot>()
            .ok_or("DflashSpeculator: target is not a Qwen3.5 ModelSlot")?;

        if max_emit == 0 {
            return Err("DflashSpeculator: max_emit=0 (no remaining output budget)".into());
        }
        // Chain DFlash: emit ≤ b (accepted drafts + bonus). Cap block size so the
        // verify window cannot commit past remaining client budget. b >= 2.
        // emit = accept + 1 ≤ b when seed is excluded. Prefer b = max_emit
        // (uniform for max_emit >= 1); max_accept clamps accept before commit
        // so max_emit == 1 is a true one-token path (accept 0 + bonus).
        let block_override = {
            let cfg_b = self.df.block_size.max(2);
            let want = max_emit.max(2).min(self.adaptive.pick());
            let b = cfg_b.min(want).max(2);
            if b < cfg_b || b != self.df.draft_config.block_size {
                Some(b)
            } else {
                None
            }
        };
        // accepted drafts + bonus = emit; max accepted drafts = max_emit - 1.
        let max_accept = Some(max_emit.saturating_sub(1));

        // Two-way dispatch: DDTree-batched (SWOR) when a tree is configured
        // (never for DFlash2 selector — load refused construction), else
        // chain-mode DFlash. Selector chain uses sparse-q rejection at temp>0.
        // The grammar arg is ignored — qwen35 enforces tool-call grammar post-hoc
        // in the daemon.
        let chain = self.df.ddtree.is_none();
        let result = if let Some(dd) = self.df.ddtree.as_mut() {
            // Tree node budget is structural; max_accept is the commit bound.
            // Keep at least 1 node so the tree builder stays well-formed; the
            // accept clamp drops to 0 drafts when max_emit == 1.
            let tree_budget = dd.budget.min(max_emit.saturating_sub(1).max(1));
            spec_step_ddtree_batched(
                gpu,
                slot,
                &self.df.draft_weights,
                &self.df.draft_config,
                &mut self.df.draft_scratch,
                &mut self.df.hidden_rb,
                &mut self.df.target_hidden_host,
                &mut self.df.target_snap,
                &mut dd.post_seed_snap,
                &mut self.df.gdn_tape,
                &dd.scratch,
                &self.df.verify_scratch,
                position,
                seed,
                None, // ctx_slice = full history
                tree_budget,
                dd.topk,
                // Request temperature → distribution-preserving SWOR verify at
                // temp>0 (greedy/argmax at temp 0). The ddtree-batched arm is the
                // only DFlash mode with sampled verify; the chain below stays
                // greedy, so `supports_temp_verify` gates serve routing to ddtree.
                temp,
                &mut self.rng_state,
                max_accept,
            )
        } else {
            // Selector chain must error on invalid rewrites rather than silently
            // substituting 0/None — keep the CACTUS guard here (the other
            // rewrites are hard-wired to off/None in this path, but the direct
            // `spec_step_dflash` caller covers them).
            if self.df.draft_weights.has_candidate_selector() && self.sample_cactus != 0.0 {
                return Err(
                    "selector mode: CACTUS is not supported with DFlash2 candidate selector".into(),
                );
            }
            spec_step_dflash(
                gpu,
                slot,
                &self.df.draft_weights,
                &self.df.draft_config,
                &mut self.df.draft_scratch,
                &mut self.df.hidden_rb,
                &mut self.df.target_hidden_host,
                &mut self.df.target_snap,
                &self.df.verify_scratch,
                position,
                seed,
                None, // ctx_slice = full history
                Some(&mut self.df.gdn_tape),
                // Sampling threaded from the request via `set_sampling` (#477
                // merge re-wire). These four positions reproduce spec-graph's old
                // inline `generate_dflash` call verbatim: temp 0 ⇒ greedy/argmax;
                // temp>0 ⇒ lossless rejection sampling with the IDENTICAL
                // (top_k,top_p) nucleus truncation on draft + target. The DDTree
                // branches above stay greedy (tree-verify is greedy by
                // construction) and ignore these.
                self.sample_temp,
                self.sample_top_p, // top_p (1.0 = no truncation)
                self.sample_top_k, // top_k (0 = top_p-only)
                &mut self.rng_state,
                block_override, // remaining-output budget
                None,           // ngram_cache
                emitted,
                self.sample_cactus, // selector already checked — safe to pass through
                None,               // pld_spine
                1.0_f32,            // repeat_penalty (off)
                0,                  // repeat_window
                max_accept,
                Some(&mut self.df.verify_pm4),
            )
        };

        match result {
            Ok(r) => {
                if chain {
                    self.adaptive.observe(r.accepted);
                }
                Ok(lower_qwen35(r).cap_emit(max_emit))
            }
            Err(e) => Err(e.to_string()),
        }
    }

    fn on_evict(&mut self, gpu: &mut Gpu, retain: &EvictRetain) -> Result<(), String> {
        // Compact the drafter's cached target-hidden rows to match the target KV
        // after the FlashCASK eviction the daemon already applied to the target.
        let ne = self.df.draft_config.num_extract();
        let h = self.df.draft_config.hidden;
        apply_eviction_retain_to_draft(
            gpu,
            &mut self.df.draft_scratch,
            &retain.retain_mask,
            ne,
            h,
            retain.pre_phys,
        )
        .map_err(|e| e.to_string())
    }

    fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        // Drafter-local reset: invalidate cached suffix projections and free the
        // divergent-render checkpoint ring (the target KV/recurrent reset is the
        // daemon's job — it owns the bundle).
        self.df.draft_scratch.reset_upload_tracking();
        for (_, snap) in self.checkpoints.drain(..) {
            snap.free_gpu(gpu);
        }
        Ok(())
    }

    fn reset_state_evidence(&self) -> Option<hipfire_runtime::spec::SpecResetEvidence> {
        let th = &self.df.draft_scratch.thlog;
        Some(hipfire_runtime::spec::SpecResetEvidence {
            drafter_reset: th.uploaded_rows() == 0
                && th.proj_cached_rows() == 0
                && th.full_cached_rows() == 0,
            checkpoint_empty: self.checkpoints.is_empty(),
        })
    }

    fn block_size(&self) -> usize {
        self.df.block_size
    }

    fn ctx_capacity(&self) -> usize {
        self.df.ctx_capacity
    }

    fn checkpoint_positions(&self) -> Vec<usize> {
        self.checkpoints.iter().map(|(p, _)| *p).collect()
    }

    fn rewind_to(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
    ) -> Result<usize, String> {
        // Restore the target's DeltaNet recurrent state to the checkpoint at
        // `position` and drop the now-stale tail of the ring (mirrors the old
        // divergent-render resume at generate_dflash 4021-4036). Caller rewinds
        // seq_pos / conversation_tokens to match.
        let slot = target
            .as_any_mut()
            .downcast_mut::<ModelSlot>()
            .ok_or("DflashSpeculator: target is not a Qwen3.5 ModelSlot")?;
        if let Some(idx) = self.checkpoints.iter().rposition(|(p, _)| *p == position) {
            self.checkpoints[idx]
                .1
                .restore_to(&mut slot.dn_state, gpu)
                .map_err(|e| format!("DeltaNetSnapshot::restore_to: {e}"))?;
            for (_, snap) in self.checkpoints.drain(idx + 1..) {
                snap.free_gpu(gpu);
            }
        }
        Ok(position)
    }

    fn configure_request(&mut self, cfg: SpecRequestConfig) {
        // Store the request's sampling config for the chain-mode branch of
        // `step`. Re-seed the RNG to the same fixed value spec-graph used per
        // `generate_dflash` call (a fresh `let mut rng_state = 0x13579BDF`), so a
        // sampled request is deterministic given its seed and two identical
        // requests in one session produce identical output — preserving
        // spec-graph's behavior rather than letting the seed drift across turns.
        // New SpecRequestConfig fields (min_p / rng_seed / ngram) are ignored —
        // this path never supported them.
        self.sample_temp = cfg.temp;
        self.sample_top_p = cfg.top_p;
        self.sample_top_k = cfg.top_k;
        self.sample_cactus = cfg.cactus_delta;
        self.rng_state = 0x13579BDF;
        self.adaptive.reset();
    }

    fn requires_greedy(&self) -> bool {
        // DFlash supports faithful temp>0 decode via lossless rejection sampling
        // (set_sampling + the sampled `spec_step_dflash` path), so it does NOT
        // require greedy verification. The daemon dispatch consults this (via
        // `spec_can_sample`) to decide whether a temp>0 request may take the spec
        // path or must fall to AR — returning `false` here is what lets sampled
        // DFlash engage while greedy-only drafters (MTP/n-gram) stay on AR.
        false
    }

    fn quiesce(&mut self, _gpu: &mut Gpu) -> Result<(), String> {
        self.df.verify_pm4.shutdown()
    }

    fn verify_pm4_report(&self) -> Option<serde_json::Value> {
        Some(self.df.verify_pm4.report_json())
    }

    fn free(self: Box<Self>, gpu: &mut Gpu) {
        let DflashSpeculator {
            df, checkpoints, ..
        } = *self;
        df.free_gpu(gpu);
        for (_, snap) in checkpoints {
            snap.free_gpu(gpu);
        }
    }
}

/// Construct the DFlash speculator from a freshly-loaded `DflashState`, resolving
/// the env config the daemon's old `generate_dflash` read inline: checkpoint
/// resume (`HIPFIRE_DFLASH_CKPT_RESUME` + no-eviction) and interval/cap
/// (`HIPFIRE_CACHE_CKPT_INTERVAL`/`_MAX`, matching the daemon's
/// `ckpt_interval()`/`ckpt_max()` defaults). Called once at load.
pub fn build_dflash_speculator(df: DflashState, eviction_is_none: bool) -> Box<dyn Speculator> {
    let resume_enabled = hipfire_config::developer_var("HIPFIRE_DFLASH_CKPT_RESUME")
        .ok()
        .as_deref()
        != Some("0")
        && eviction_is_none;
    let ck_interval = hipfire_config::developer_var("HIPFIRE_CACHE_CKPT_INTERVAL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048usize)
        .max(256);
    let ck_cap = hipfire_config::developer_var("HIPFIRE_CACHE_CKPT_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8usize)
        .max(1);
    let adaptive = AdaptiveBlock::from_env(df.block_size);
    Box::new(DflashSpeculator::new(
        df,
        resume_enabled,
        ck_interval,
        ck_cap,
        adaptive,
    ))
}

#[cfg(test)]
mod adaptive_block_tests {
    use super::AdaptiveBlock;

    #[test]
    fn low_tau_on_dflash2_shrinks_to_half() {
        let mut b = AdaptiveBlock::new(8, true, 2.5);
        for _ in 0..4 {
            b.observe(2);
        }
        assert_eq!(b.pick(), 4);
    }

    #[test]
    fn high_tau_stays_at_trained() {
        let mut b = AdaptiveBlock::new(8, true, 2.5);
        for _ in 0..4 {
            b.observe(6);
        }
        assert_eq!(b.pick(), 8);
    }

    #[test]
    fn dflash1_trained_16_shrinks_to_8() {
        let mut b = AdaptiveBlock::new(16, true, 2.5);
        for _ in 0..4 {
            b.observe(2);
        }
        assert_eq!(b.pick(), 8);
    }

    #[test]
    fn reset_returns_to_trained() {
        let mut b = AdaptiveBlock::new(8, true, 2.5);
        for _ in 0..4 {
            b.observe(2);
        }
        b.reset();
        assert_eq!(b.pick(), 8);
    }

    #[test]
    fn disabled_ignores_accepts() {
        let mut b = AdaptiveBlock::new(8, false, 2.5);
        for _ in 0..8 {
            b.observe(1);
        }
        assert_eq!(b.pick(), 8);
    }
}

#[cfg(test)]
mod dflash_oom_backoff_tests {
    use super::{
        is_dflash_vram_oom, next_legacy_dflash_cap_after_oom, DEFAULT_DFLASH_CTX_CAP,
    };

    #[test]
    fn halves_until_default_floor() {
        assert_eq!(next_legacy_dflash_cap_after_oom(65536), Some(32768));
        assert_eq!(next_legacy_dflash_cap_after_oom(32768), Some(16384));
        assert_eq!(next_legacy_dflash_cap_after_oom(16384), Some(8192));
        assert_eq!(next_legacy_dflash_cap_after_oom(DEFAULT_DFLASH_CTX_CAP), None);
        assert_eq!(next_legacy_dflash_cap_after_oom(9000), Some(8192));
    }

    #[test]
    fn detects_hip_oom_display() {
        assert!(is_dflash_vram_oom(
            "HiddenStateRingBuffer::new: HipError(2): hipMalloc: out of memory"
        ));
        assert!(!is_dflash_vram_oom(
            "draft: failed to parse DflashConfig from HFQ metadata"
        ));
    }
}

// ─── Retained-PM4 admission (pure) ────────────────────────────────────

/// Pure admission predicate for the fixed B=16 DFlash retained-PM4 verify route.
///
/// The gates constrain what the tape actually captures — the *target* forward's
/// arch, shape, quantization, and scratch capacities. They deliberately say
/// nothing about the draft's selector or dynamic-conv fields: the draft forward,
/// candidate selection, and draft lm-head are all outside the tape boundary, so
/// DFlash2 and legacy DFlash produce an identical target verify body.
///
/// Each failed condition returns a **distinct** reason string so harness
/// evidence can name the gate. GPU-free: unit-tested without a device.
pub fn admit_dflash_verify_pm4(
    env_opt_in: bool,
    arch: &str,
    single_gpu: bool,
    num_experts: usize,
    kv_is_q8: bool,
    dn_state_is_q8: bool,
    adaptive_kv_absent: bool,
    runtime_extract_layers: &[usize],
    runtime_block_size: usize,
    target_layer_ids: &[usize],
    ddtree_present: bool,
    prefill_batch_present: bool,
    pbs_eligible_b16: bool,
    verify_max_n: usize,
    pbs_max_batch: usize,
    hidden_rb_max_batch: usize,
    gdn_tape_max_n: usize,
) -> Result<(), String> {
    if !env_opt_in {
        return Err("HIPFIRE_DFLASH_VERIFY_PM4 is not set to 1".into());
    }
    if arch != "gfx1201" {
        return Err(format!("arch is {arch}, not exact gfx1201"));
    }
    if !single_gpu {
        return Err("multi-GPU load is not admitted".into());
    }
    if num_experts != 0 {
        return Err(format!(
            "MoE target (num_experts={num_experts}) is not admitted"
        ));
    }
    if !kv_is_q8 {
        return Err("KV mode is not Q8".into());
    }
    if !dn_state_is_q8 {
        return Err("DeltaNet state quant is not Q8".into());
    }
    if !adaptive_kv_absent {
        return Err("adaptive KV is engaged".into());
    }
    // Consistency, not identity: whatever layers the draft declares must be the
    // layers the hidden ring actually extracts, or the captured staging writes
    // do not correspond to what the draft will read back.
    if runtime_extract_layers != target_layer_ids {
        return Err(format!(
            "hidden-ring extract layers {runtime_extract_layers:?} != draft target_layer_ids {target_layer_ids:?}"
        ));
    }
    if target_layer_ids.is_empty() {
        return Err("draft declares no target_layer_ids".into());
    }
    if runtime_block_size != DFLASH_VERIFY_PM4_BLOCK {
        return Err(format!(
            "runtime block size is {runtime_block_size}, not {DFLASH_VERIFY_PM4_BLOCK}"
        ));
    }
    if ddtree_present {
        return Err("DDTree is present".into());
    }
    if !prefill_batch_present {
        return Err("verify_scratch.prefill_batch is absent".into());
    }
    if !pbs_eligible_b16 {
        return Err("prefill_batch_pbs_eligible failed at B=16".into());
    }
    if verify_max_n < DFLASH_VERIFY_PM4_BLOCK {
        return Err(format!(
            "verify_scratch.max_n={verify_max_n} < {DFLASH_VERIFY_PM4_BLOCK}"
        ));
    }
    if pbs_max_batch < DFLASH_VERIFY_PM4_BLOCK {
        return Err(format!(
            "prefill_batch.max_batch={pbs_max_batch} < {DFLASH_VERIFY_PM4_BLOCK}"
        ));
    }
    if hidden_rb_max_batch < DFLASH_VERIFY_PM4_BLOCK {
        return Err(format!(
            "hidden_rb.max_batch={hidden_rb_max_batch} < {DFLASH_VERIFY_PM4_BLOCK}"
        ));
    }
    if gdn_tape_max_n < DFLASH_VERIFY_PM4_BLOCK {
        return Err(format!(
            "gdn_tape.max_n={gdn_tape_max_n} < {DFLASH_VERIFY_PM4_BLOCK}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod admit_dflash_verify_pm4_tests {
    use super::*;

    /// Baseline admitted args; override one field per negative case.
    struct Args {
        env_opt_in: bool,
        arch: &'static str,
        single_gpu: bool,
        num_experts: usize,
        kv_is_q8: bool,
        dn_state_is_q8: bool,
        adaptive_kv_absent: bool,
        runtime_extract_layers: &'static [usize],
        runtime_block_size: usize,
        target_layer_ids: &'static [usize],
        ddtree_present: bool,
        prefill_batch_present: bool,
        pbs_eligible_b16: bool,
        verify_max_n: usize,
        pbs_max_batch: usize,
        hidden_rb_max_batch: usize,
        gdn_tape_max_n: usize,
    }

    impl Default for Args {
        fn default() -> Self {
            Self {
                env_opt_in: true,
                arch: "gfx1201",
                single_gpu: true,
                num_experts: 0,
                kv_is_q8: true,
                dn_state_is_q8: true,
                adaptive_kv_absent: true,
        runtime_extract_layers: &DFLASH_VERIFY_PM4_EXTRACT_LAYERS,
                runtime_block_size: DFLASH_VERIFY_PM4_BLOCK,
                target_layer_ids: &DFLASH_VERIFY_PM4_EXTRACT_LAYERS,
                ddtree_present: false,
                prefill_batch_present: true,
                pbs_eligible_b16: true,
                verify_max_n: DFLASH_VERIFY_PM4_BLOCK,
                pbs_max_batch: DFLASH_VERIFY_PM4_BLOCK,
                hidden_rb_max_batch: DFLASH_VERIFY_PM4_BLOCK,
                gdn_tape_max_n: DFLASH_VERIFY_PM4_BLOCK,
            }
        }
    }

    fn admit(a: Args) -> Result<(), String> {
        admit_dflash_verify_pm4(
            a.env_opt_in,
            a.arch,
            a.single_gpu,
            a.num_experts,
            a.kv_is_q8,
            a.dn_state_is_q8,
            a.adaptive_kv_absent,
            a.runtime_extract_layers,
            a.runtime_block_size,
            a.target_layer_ids,
            a.ddtree_present,
            a.prefill_batch_present,
            a.pbs_eligible_b16,
            a.verify_max_n,
            a.pbs_max_batch,
            a.hidden_rb_max_batch,
            a.gdn_tape_max_n,
        )
    }

    #[test]
    fn admits_full_conjunction() {
        assert!(admit(Args::default()).is_ok());
    }

    #[test]
    fn rejects_env_off() {
        let err = admit(Args {
            env_opt_in: false,
            ..Args::default()
        })
        .unwrap_err();
        assert_eq!(err, "HIPFIRE_DFLASH_VERIFY_PM4 is not set to 1");
    }

    #[test]
    fn rejects_wrong_arch() {
        let err = admit(Args {
            arch: "gfx1100",
            ..Args::default()
        })
        .unwrap_err();
        assert_eq!(err, "arch is gfx1100, not exact gfx1201");
    }

    #[test]
    fn rejects_multi_gpu() {
        let err = admit(Args {
            single_gpu: false,
            ..Args::default()
        })
        .unwrap_err();
        assert_eq!(err, "multi-GPU load is not admitted");
    }

    #[test]
    fn rejects_moe() {
        let err = admit(Args {
            num_experts: 256,
            ..Args::default()
        })
        .unwrap_err();
        assert_eq!(err, "MoE target (num_experts=256) is not admitted");
    }

    #[test]
    fn rejects_non_q8_kv() {
        let err = admit(Args {
            kv_is_q8: false,
            ..Args::default()
        })
        .unwrap_err();
        assert_eq!(err, "KV mode is not Q8");
    }

    #[test]
    fn rejects_non_q8_dn_state() {
        let err = admit(Args {
            dn_state_is_q8: false,
            ..Args::default()
        })
        .unwrap_err();
        assert_eq!(err, "DeltaNet state quant is not Q8");
    }

    #[test]
    fn rejects_adaptive_kv() {
        let err = admit(Args {
            adaptive_kv_absent: false,
            ..Args::default()
        })
        .unwrap_err();
        assert_eq!(err, "adaptive KV is engaged");
    }

    #[test]
    fn admits_a_legacy_dflash_draft() {
        // The draft forward is outside the tape, so a draft without the DFlash2
        // selector or dynamic conv still yields an identical target verify body.
        assert!(admit(Args {
            runtime_extract_layers: &[3, 7, 11, 15, 19],
            target_layer_ids: &[3, 7, 11, 15, 19],
            ..Args::default()
        })
        .is_ok());
    }

    #[test]
    fn rejects_empty_extract_layers() {
        let err = admit(Args {
            runtime_extract_layers: &[],
            target_layer_ids: &[],
            ..Args::default()
        })
        .unwrap_err();
        assert_eq!(err, "draft declares no target_layer_ids");
    }

    #[test]
    fn rejects_wrong_block_size() {
        let err = admit(Args {
            runtime_block_size: 8,
            ..Args::default()
        })
        .unwrap_err();
        assert_eq!(err, "runtime block size is 8, not 16");
    }

    #[test]
    fn rejects_extract_layer_disagreement() {
        let err = admit(Args {
            target_layer_ids: &[5, 19, 33, 47, 60],
            ..Args::default()
        })
        .unwrap_err();
        assert_eq!(
            err,
            "hidden-ring extract layers [5, 19, 33, 47, 61] != draft target_layer_ids [5, 19, 33, 47, 60]"
        );
    }

    #[test]
    fn rejects_ddtree() {
        let err = admit(Args {
            ddtree_present: true,
            ..Args::default()
        })
        .unwrap_err();
        assert_eq!(err, "DDTree is present");
    }

    #[test]
    fn rejects_missing_prefill_batch() {
        let err = admit(Args {
            prefill_batch_present: false,
            ..Args::default()
        })
        .unwrap_err();
        assert_eq!(err, "verify_scratch.prefill_batch is absent");
    }

    #[test]
    fn rejects_pbs_ineligible() {
        let err = admit(Args {
            pbs_eligible_b16: false,
            ..Args::default()
        })
        .unwrap_err();
        assert_eq!(err, "prefill_batch_pbs_eligible failed at B=16");
    }

    #[test]
    fn rejects_capacity_shortfalls() {
        let cases = [
            (
                Args {
                    verify_max_n: 15,
                    ..Args::default()
                },
                "verify_scratch.max_n=15 < 16",
            ),
            (
                Args {
                    pbs_max_batch: 15,
                    ..Args::default()
                },
                "prefill_batch.max_batch=15 < 16",
            ),
            (
                Args {
                    hidden_rb_max_batch: 15,
                    ..Args::default()
                },
                "hidden_rb.max_batch=15 < 16",
            ),
            (
                Args {
                    gdn_tape_max_n: 15,
                    ..Args::default()
                },
                "gdn_tape.max_n=15 < 16",
            ),
        ];
        for (args, expected) in cases {
            assert_eq!(admit(args).unwrap_err(), expected);
        }
    }
}
