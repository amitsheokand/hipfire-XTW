#!/usr/bin/env bash
# R9700 (gfx1201) A/B sweep across a fixed prompt fixture set.
#
# Runs `hipfire bench` on the stock and current builds against the same
# committed prompts so token/s deltas are attributable to code, not prompt
# shape (one newline moves tau by up to 17%). Records prompt md5 + binary
# md5 alongside the JSON so the result is reproducible.
#
# Usage:
#   scripts/r9700_ab_sweep.sh <model-tag> [prompt-file ...]
#
# Defaults to the four canonical R9700 fixtures if none are given. Pin
# --backend/--workload explicitly: the default `both` measures two things
# at once and is not a comparison.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_TAG="${1:?usage: $0 <model-tag> [prompt-file ...]}"
shift || true

PROMPTS=("$@")
if [ "${#PROMPTS[@]}" -eq 0 ]; then
    PROMPTS=(
        "$ROOT/benchmarks/prompts/merge_sort_thinking_off.txt"
        "$ROOT/benchmarks/prompts/humaneval_3_below_zero.txt"
        "$ROOT/benchmarks/prompts/lru_cache_pep8_strict.txt"
        "$ROOT/benchmarks/prompts/prose_river_short.txt"
    )
fi

RUNS="${RUNS:-5}"
WARMUPS="${WARMUPS:-3}"
MAX_TOKENS="${MAX_TOKENS:-128}"
BACKEND="${BACKEND:-noslots}"
WORKLOAD="${WORKLOAD:-stateless}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/target/validation/r9700-ab-sweep/$RUN_ID}"
mkdir -p "$OUT_ROOT"

HIPFIRE_BIN="${HIPFIRE_BIN:-$ROOT/target/release/hipfire}"
DAEMON_BIN="${DAEMON_BIN:-$ROOT/target/release/daemon}"

record_provenance() {
    local dir="$1"
    : > "$dir/provenance.txt"
    {
        echo "run_id=$RUN_ID"
        echo "model_tag=$MODEL_TAG"
        echo "runs=$RUNS warmups=$WARMUPS max_tokens=$MAX_TOKENS"
        echo "backend=$BACKEND workload=$WORKLOAD"
        echo "hipfire_md5=$(md5sum "$HIPFIRE_BIN" 2>/dev/null | awk '{print $1}')"
        echo "daemon_md5=$(md5sum "$DAEMON_BIN" 2>/dev/null | awk '{print $1}')"
        echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } >> "$dir/provenance.txt"
}

bench_prompt() {
    local prompt="$1" prompt_md5 out_dir
    prompt_md5="$(md5sum "$prompt" | awk '{print $1}')"
    out_dir="$OUT_ROOT/$(basename "${prompt%.txt}")_$prompt_md5"
    mkdir -p "$out_dir"
    cp "$prompt" "$out_dir/prompt.txt"
    record_provenance "$out_dir"

    echo ">>> bench: $(basename "$prompt") (md5=$prompt_md5)"
    "$HIPFIRE_BIN" bench "$MODEL_TAG" \
        --runs "$RUNS" --warmups "$WARMUPS" --max-tokens "$MAX_TOKENS" \
        --backend "$BACKEND" --workload "$WORKLOAD" --json \
        > "$out_dir/bench.json" 2> "$out_dir/bench.stderr" || {
            echo "bench failed for $prompt — see $out_dir/bench.stderr" >&2
            return 1
        }
    echo "    -> $out_dir/bench.json"
}

echo "R9700 A/B sweep — model=$MODEL_TAG runs=$RUNS backend=$BACKEND"
echo "out: $OUT_ROOT"
for p in "${PROMPTS[@]}"; do
    bench_prompt "$p"
    sleep 5
done

echo "R9700 A/B sweep complete: $OUT_ROOT"
