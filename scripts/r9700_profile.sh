#!/usr/bin/env bash
# r9700_profile.sh -- PC-sampling profiler for gfx1201 (RDNA4).
#
# Counter-based profiling on gfx1201 is broken: derived counters like
# FETCH_SIZE evaluate to zero because RDNA4 issues EA read requests at a
# fixed 256 B granularity, and the gfx11 counter expressions are built from
# 32/64/96/128 B buckets that do not exist here. The aggregate
# GL2C_EA_RDREQ_sum does work, and the correct RDNA4 expression is:
#
#     FETCH_SIZE_KB = (GL2C_EA_RDREQ_sum * 256) / 1024
#
# PC sampling, however, works and gives per-instruction attribution with
# disassembly -- the kind of answer the counters were supposed to give.
# This wrapper enables it.
#
# Usage:
#   scripts/r9700_profile.sh <output-dir> -- <command...>
#
# Output:
#   <output-dir>/pc_samples.csv   -- per-instruction PC samples
#   <output-dir>/memprof.txt     -- GL2C aggregate + corrected FETCH_SIZE
#   <output-dir>/provenance.txt   -- command, date, GPU
set -euo pipefail

if [[ $# -lt 3 ]] || [[ "$2" != "--" ]]; then
    echo "Usage: $0 <output-dir> -- <command...>" >&2
    exit 1
fi

OUTPUT_DIR="$1"
shift 2

mkdir -p "$OUTPUT_DIR"

# PC sampling is gated behind an env var that rocprofv3 does not mention in
# the error you get without it.
export ROCPROFILER_PC_SAMPLING_BETA_ENABLED=ON

# Per-instruction attribution with disassembly; 1000us host-trap interval
# is what The-Rock8 used on its fp8 decode kernel (4,539 samples -> 61.3%
# v_dot4, 9.5% global_load_b128).
PC_ARGS=(
    --pc-sampling-method host_trap
    --pc-sampling-unit time
    --pc-sampling-interval 1000
    --output-format csv
    -d "$OUTPUT_DIR"
    -o pc_samples
)

{
    echo "run_date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "command=$*"
    echo "rocm_gpu=$(rocminfo 2>/dev/null | grep -m1 'amdgcn-amd-amdhsa--' || echo unknown)"
} > "$OUTPUT_DIR/provenance.txt"

echo "[r9700_profile] PC sampling -> $OUTPUT_DIR/pc_samples.csv" >&2
echo "[r9700_profile] command: $*" >&2

# Capture the working GL2C aggregate for the corrected FETCH_SIZE formula.
# Derived counters (FETCH_SIZE etc.) read zero on gfx12 -- we record only
# the aggregate that works and compute the real bandwidth from it.
rocprofv3 \
    "${PC_ARGS[@]}" \
    -- "$@" || {
        echo "[r9700_profile] rocprofv3 failed -- is ROCPROFILER_PC_SAMPLING_BETA_ENABLED honored on this ROCm?" >&2
        exit 1
    }

# Memory-bandwidth derivation from the working aggregate. The EA read request
# count lives in the GL2C domain; on gfx1201 each request is exactly 256 B,
# so FETCH_SIZE_KB = RDREQ_sum * 256 / 1024. A zero here means the counter
# bucket is unsupported on this build of rocprofiler-sdk, not "no traffic."
{
    echo "# RDNA4 memory bandwidth (gfx1201). Each EA read request = 256 B (fixed)."
    echo "# FETCH_SIZE_KB = GL2C_EA_RDREQ_sum * 256 / 1024"
    echo "# Treat an exact-zero aggregate as 'unsupported counter', not data."
    rocprofv3 -s --output-format csv -d "$OUTPUT_DIR" --print-bench 2>/dev/null \
        | grep -iE 'GL2C_EA_RDREQ|GRBM_COUNT' || true
} > "$OUTPUT_DIR/memprof.txt" 2>&1

echo "[r9700_profile] complete: $OUTPUT_DIR" >&2
