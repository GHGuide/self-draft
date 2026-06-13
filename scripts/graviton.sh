#!/usr/bin/env bash
# One-shot: provision an AWS Graviton (Arm64) instance, build llama.cpp with Arm
# KleidiAI kernels, fetch Gemma 4 + its MTP head, and benchmark self-draft with a
# cost-per-token table. Tested target: Ubuntu 22.04/24.04 on c7g/c8g (Neoverse V1/V2).
#
# Usage:  bash scripts/graviton.sh [PRICE_PER_HR]
#   PRICE_PER_HR defaults to 0.145 (c7g.xlarge on-demand, us-east-1; CHECK current price).
set -euo pipefail
PRICE="${1:-0.145}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

echo "== arch check =="; uname -m   # expect aarch64
test "$(uname -m)" = "aarch64" || echo "WARNING: not aarch64 - this script targets Arm64 Graviton"

echo "== deps =="
if command -v apt-get >/dev/null; then
  sudo apt-get update -y
  sudo apt-get install -y build-essential cmake git python3 python3-pip libcurl4-openssl-dev
  pip3 install -q --user "huggingface_hub[cli]" || pipx install huggingface_hub || true
fi
HF="$(command -v hf || echo "$HOME/.local/bin/hf")"

echo "== clone + build llama.cpp WITH KleidiAI (Arm Neoverse i8mm/SVE2 matmul) =="
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
# KleidiAI = Arm's optimized micro-kernels; GGML_CPU_KLEIDIAI repacks Q4_0 for i8mm/dotprod.
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_CPU_KLEIDIAI=ON -DGGML_NATIVE=ON
cmake --build llama.cpp/build -j"$(nproc)" --target llama-server
# (optional) native --self-draft flag:
( cd llama.cpp && git apply ../patches/llama.cpp-self-draft.patch 2>/dev/null && \
  cmake --build build -j"$(nproc)" --target llama-server && echo "patched: --self-draft available" ) || \
  echo "note: running without the --self-draft patch (sd.py wires draft-mtp explicitly)"

echo "== fetch model + MTP head =="
mkdir -p models
# Q4_0 is the KleidiAI-accelerated quant on Arm; also grab Q4_K_M for comparison.
"$HF" download unsloth/gemma-4-12b-it-GGUF \
  --include "gemma-4-12b-it-Q4_0.gguf" "gemma-4-12b-it-Q4_K_M.gguf" "mtp-gemma-4-12b-it.gguf" \
  --local-dir models

NPROC="$(nproc)"
echo "== autotune n-max (Q4_0, KleidiAI, CPU) on $NPROC threads =="
python3 selfdraft/sd.py autotune models/gemma-4-12b-it-Q4_0.gguf \
  --ngl 0 --threads "$NPROC" --grid 1,2,3,4,6 --workload code --n-predict 200 \
  --json bench/graviton_autotune_q4_0.json

echo "== bench self-draft with cost (Q4_0, KleidiAI) =="
python3 selfdraft/sd.py bench models/gemma-4-12b-it-Q4_0.gguf \
  --ngl 0 --threads "$NPROC" --n-max 3 --workload code --n-predict 200 \
  --price "$PRICE" --json bench/graviton_bench_q4_0.json

echo "== bench self-draft with cost (Q4_K_M, for comparison) =="
python3 selfdraft/sd.py bench models/gemma-4-12b-it-Q4_K_M.gguf \
  --ngl 0 --threads "$NPROC" --n-max 3 --workload code --n-predict 200 \
  --price "$PRICE" --json bench/graviton_bench_q4km.json

echo
echo "== DONE =="
echo "Instance: $(uname -m), $NPROC vCPU, price \$$PRICE/hr"
echo "Results JSON in bench/graviton_*.json -> paste the numbers into README results table."
