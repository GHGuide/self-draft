# Devpost submission - copy-paste fields

**Track:** Cloud AI

**Project name:** Arm Agent-Memory

**Elevator pitch (one line):**
Persistent multi-agent KV memory for llama.cpp on Arm64: returning agents restore their cache
instead of re-prefilling — ~23-180x lower TTFT, bit-identical output, 4x more agents per GB,
plus a bundled 2x self-draft decode. First llama.cpp/Arm64 port of arXiv:2603.04428.

**Built with:** llama.cpp, Gemma 4, MTP, KleidiAI/i8mm, Python, GitHub Actions (Arm64), AWS Graviton, Apache-2.0

**Repository:** https://github.com/GHGuide/self-draft

**Video:** (your YouTube link)

---

## About the project (paste into the description field)

### Project Overview
Multi-agent LLM systems are the hot 2026 workload, but device RAM holds only a few KV caches at
once. When an agent is evicted and resumes, the standard serving path re-prefills its whole
context - on a 4-vCPU Arm CPU that is 13 s to 91 s of wasted compute per resumed turn. Arm
Agent-Memory persists each agent's KV to disk and restores it on resume instead. It ports the
idea of arXiv:2603.04428 (Apple-MLX only) to llama.cpp on Arm64 - the first such port - and adds
the automatic slot manager llama.cpp lacks (agent_id -> disk slot, restore-on-resume, LRU
eviction). It should win because it targets exactly what the Cloud AI rubric names - agents,
inference-server TTFT, Arm64 - with a measured, reproducible, bit-exact result, not a tutorial.

### Functionality / Output
- `agent_memory.py`: automatic persistent-KV manager over llama-server (restore-on-resume,
  save-on-turn, LRU eviction to disk when agents exceed RAM-resident slots).
- Bundled `--self-draft`: the model's own MTP heads as a zero-download speculative draft, ~2x decode.
- Measured (Gemma-4-12B Q4, Arm CPU, free GitHub Arm64 CI): returning-agent TTFT 91,043 ms cold
  -> 498 ms restored (182x); multi-agent RAM-constrained 13,124 ms -> 563 ms (23x); restored
  output byte-identical to never-evicted (sha-verified); 4x more agents per GB via q8_0/q4 KV;
  ~2.0x bundled decode.

### Setup / Build / Validate on Arm64
Zero-cost, reproducible: open the repo's Actions tab -> "arm64-cloud-benchmark" -> Run workflow.
On a free GitHub `ubuntu-24.04-arm` runner it builds llama.cpp with KleidiAI, runs the multi-agent
TTFT bench, the bit-exact check, and the self-draft decode bench, and uploads results.

Locally / on Graviton:
```
git clone https://github.com/GHGuide/self-draft && cd self-draft
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_CPU_KLEIDIAI=ON
cmake --build llama.cpp/build -j --target llama-server
hf download unsloth/gemma-4-12b-it-GGUF --include "gemma-4-12b-it-Q4_K_M.gguf" "mtp-gemma-4-12b-it.gguf" --local-dir models
python3 bench/agent_memory_bench.py --agents 4 --ram-slots 2 --rounds 2   # TTFT win
python3 bench/kv_equivalence.py                                           # bit-exact proof
```
Full instructions + honest limitations (RAM-constrained regime, Gemma --swa-full) in the README.
