# Devpost submission - copy-paste fields

**Track:** Cloud AI

**Project name:** self-draft

**Elevator pitch (one line):**
One flag turns your model's own MTP heads into a draft - up to 2.0x faster, ~50% cheaper LLM inference on Arm64 cloud, zero extra download. Verified in CI on a free Arm64 runner.

**Built with:** llama.cpp, Arm KleidiAI, Gemma 4, Multi-Token Prediction (MTP), Python, GitHub Actions (Arm64), AWS Graviton, Apache-2.0

**Repository:** https://github.com/GHGuide/self-draft

**Video:** (your YouTube link, optional)

---

## About the project (paste into the description field)

### Project Overview
Speculative decoding speeds up LLM inference - a cheap "draft" proposes tokens, the full
model verifies them in one batched pass - but llama.cpp's `--model-draft` needs a
separate, vocabulary-matched draft model that most models don't have. self-draft removes
that barrier: modern models (Google Gemma 4) ship Multi-Token-Prediction (MTP) heads in
the same release, and those heads *are* a draft. self-draft auto-detects the MTP sibling
and wires it into llama.cpp's `draft-mtp` path with one flag - zero extra download. It
adds a draft-length autotuner (the single knob most people get wrong) and reports the
metrics inference teams care about (tok/s, TTFT, latency, $/token). It is a drop-in
latency optimization for LLM inference servers on Arm64 cloud.

Why it should win: it is real Arm64-cloud runtime engineering, not a config tutorial; it
is verified on a free GitHub-hosted Arm64 runner (fully reproducible in CI, no credit
card); and it is honest - we characterize exactly where it helps (latency for
agentic/reasoning workloads) and that it is distributionally lossless, not bit-identical.

### Functionality / Output
- One command: `python3 selfdraft/sd.py run model.gguf --ngl 0` (or native llama.cpp
  `--self-draft` via the included patch) serves an OpenAI-compatible endpoint with MTP
  self-speculation, auto-resolving the local `mtp-*.gguf` head.
- `sd.py autotune` picks the best draft length per instance; `sd.py bench` reports tok/s,
  TTFT, latency and $/1M tokens; `sd.py agent` shows the end-to-end win on a tool-using
  ReAct loop.
- Measured (free GitHub `ubuntu-24.04-arm`, 4 vCPU aarch64, KleidiAI, gemma-4-12b-it-Q4_0,
  n-max=3): 2.0x faster decode (5.99 -> 12.01 tok/s), 1.87x lower latency
  (36.2s -> 19.3s), 76% draft acceptance, 99.1% output similarity, ~50% lower $/token.

### Setup / Build / Validate on Arm64
Zero-cost, reproducible: open the repo's Actions tab -> "arm64-cloud-benchmark" -> Run
workflow. It runs on a free GitHub Arm64 runner: builds llama.cpp with
`-DGGML_CPU_KLEIDIAI=ON`, fetches Gemma 4 + its MTP head, autotunes and benchmarks, and
uploads the results JSON.

Locally / on a Graviton instance:
```
git clone https://github.com/GHGuide/self-draft && cd self-draft
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_CPU_KLEIDIAI=ON
cmake --build llama.cpp/build -j --target llama-server
hf download unsloth/gemma-4-12b-it-GGUF --include "gemma-4-12b-it-Q4_0.gguf" "mtp-gemma-4-12b-it.gguf" --local-dir models
python3 selfdraft/sd.py autotune models/gemma-4-12b-it-Q4_0.gguf --ngl 0
```
Full instructions and honest limitations are in the README.
