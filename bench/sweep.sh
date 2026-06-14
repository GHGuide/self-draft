#!/usr/bin/env bash
# Debug sweep: characterize Metal dual-model slowdown + measure CPU (Graviton-relevant) speedup.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMP="$ROOT/llama.cpp/build/bin/llama-completion"
SPEC="$ROOT/llama.cpp/build/bin/llama-speculative-simple"
T="$ROOT/models/gemma-4-12b-it-Q4_K_M.gguf"
D="$ROOT/models/gemma-4-12b-it-UD-Q2_K_XL.gguf"
P="<start_of_turn>user
Explain why the sky is blue.<end_of_turn>
<start_of_turn>model
"
log(){ echo "=== $* ==="; }
spec_tok(){ grep -E "decoded .* t/s" "$1" | grep -Eo "speed:[[:space:]]*[0-9.]+" | grep -Eo "[0-9.]+" | tail -1; }
spec_acc(){ grep -E "accept " "$1" | grep -Eo "[0-9.]+%" | tail -1; }
comp_tok(){ grep "eval time" "$1" | grep -v "prompt eval" | grep -Eo "[0-9.]+ tokens per second" | grep -Eo "[0-9.]+" | head -1; }

run_spec(){ # tag, ngl, ngld, nmax, extra...
  local tag=$1 ngl=$2 ngld=$3 nmax=$4; shift 4
  "$SPEC" -m "$T" -md "$D" --spec-type draft-simple -ngl "$ngl" --spec-draft-ngl "$ngld" \
    --spec-draft-n-max "$nmax" --spec-draft-p-min 0 --temp 0 -n 64 -p "$P" "$@" </dev/null \
    >"$ROOT/bench/$tag.out" 2>"$ROOT/bench/$tag.err"
  echo "[$tag] exit=$? tok/s=$(spec_tok "$ROOT/bench/$tag.err") accept=$(spec_acc "$ROOT/bench/$tag.err")"
}
run_van(){ # tag, ngl
  local tag=$1 ngl=$2
  "$COMP" -m "$T" -ngl "$ngl" -no-cnv --temp 0 -n 128 --no-display-prompt -p "$P" </dev/null \
    >"$ROOT/bench/$tag.out" 2>"$ROOT/bench/$tag.err"
  echo "[$tag] exit=$? tok/s=$(comp_tok "$ROOT/bench/$tag.err")"
}

log "R1 spec GPU n-max4 (both GPU)";            run_spec spec_gpu_n4  99 99 4
log "R2 spec GPU n-max4 no-backend-sampling";   run_spec spec_gpu_n4_nbs 99 99 4 --no-spec-draft-backend-sampling
log "R3 spec draft-on-CPU target-GPU n-max8";   run_spec spec_dcpu    99 0  8
log "R4 vanilla CPU (ngl0)";                    run_van  van_cpu      0
log "R5 spec BOTH CPU n-max8 (Graviton-like)";  run_spec spec_cpu     0  0  8
echo "ALL DONE"
