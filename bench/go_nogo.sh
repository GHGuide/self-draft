#!/usr/bin/env bash
# GO/NO-GO benchmark: vanilla vs self-speculative (--model-draft), byte-identical check.
# Usage: ./go_nogo.sh <target.gguf> <draft.gguf> [n_predict] [draft_max]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLI="$ROOT/llama.cpp/build/bin/llama-cli"
OUT="$ROOT/bench/out"
mkdir -p "$OUT"

TARGET="${1:?target gguf}"
DRAFT="${2:?draft gguf}"
NPRED="${3:-256}"
DMAX="${4:-16}"
NGL=99

# Reasoning-ish prompt (Cassandra's target workload). Gemma chat template.
read -r -d '' PROMPT <<'EOF' || true
<start_of_turn>user
A train leaves city A at 60 mph heading east. Two hours later a second train leaves city A on a parallel track at 90 mph, same direction. Show your reasoning step by step, then state after how many hours from the SECOND train's departure it overtakes the first.<end_of_turn>
<start_of_turn>model
EOF

common=( -ngl "$NGL" --temp 0 --top-k 1 --seed 1 -n "$NPRED" -no-cnv -p "$PROMPT" )

echo "=== VANILLA (target alone) ==="
"$CLI" -m "$TARGET" "${common[@]}" \
  >"$OUT/vanilla.gen.txt" 2>"$OUT/vanilla.perf.txt" || { echo "vanilla failed"; tail -20 "$OUT/vanilla.perf.txt"; exit 1; }

echo "=== SPECULATIVE (target + draft, --model-draft) ==="
"$CLI" -m "$TARGET" -md "$DRAFT" -ngld "$NGL" --draft-max "$DMAX" --draft-min 1 "${common[@]}" \
  >"$OUT/spec.gen.txt" 2>"$OUT/spec.perf.txt" || { echo "spec failed"; tail -20 "$OUT/spec.perf.txt"; exit 1; }

echo
echo "================ RESULTS ================"
hv=$(shasum -a 256 "$OUT/vanilla.gen.txt" | awk '{print $1}')
hs=$(shasum -a 256 "$OUT/spec.gen.txt"    | awk '{print $1}')
echo "vanilla gen sha256: $hv"
echo "spec    gen sha256: $hs"
if [ "$hv" = "$hs" ]; then echo "OUTPUT: BYTE-IDENTICAL ✅"; else echo "OUTPUT: DIVERGED ❌"; fi

# generation throughput ("eval time ... tokens per second")
tv=$(grep -E "eval time" "$OUT/vanilla.perf.txt" | grep -Eo "[0-9.]+ tokens per second" | head -1 | awk '{print $1}')
ts=$(grep -E "eval time" "$OUT/spec.perf.txt"    | grep -Eo "[0-9.]+ tokens per second" | head -1 | awk '{print $1}')
echo "vanilla gen tok/s: ${tv:-?}"
echo "spec    gen tok/s: ${ts:-?}"
if [ -n "${tv:-}" ] && [ -n "${ts:-}" ]; then
  echo "SPEEDUP: $(python3 -c "print(f'{$ts/$tv:.2f}x')")"
fi
# acceptance (llama.cpp prints draft accept stats)
echo "--- draft acceptance ---"
grep -E -i "accept|n_drafted|draft" "$OUT/spec.perf.txt" | head -10 || echo "(no accept lines — check perf file)"
echo "========================================="
echo "perf files: $OUT/{vanilla,spec}.perf.txt"
