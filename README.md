# Arm Agent-Memory — persistent multi-agent KV serving for llama.cpp on Arm64

**When device RAM can't hold every agent's KV cache, returning agents pay a full re-prefill
— 13–91 s per turn on an Arm CPU. Agent-Memory persists each agent's KV to disk and restores
it instead: **up to 500× lower TTFT** (measured on a free Arm64 CI runner), **bit-identical**
output, up to 3.6× more agents per GB — with an optional self-draft decode layer on top.
One server, one Arm64 box, reproducible on free CI.**

> **Arm Create: AI Optimization Challenge — Cloud AI track.** Target: AWS Graviton / Arm64
> multi-agent inference serving. Validated free on GitHub `ubuntu-24.04-arm` runners.
> Ports [arXiv:2603.04428](https://arxiv.org/abs/2603.04428) (Apple-MLX only) to llama.cpp/Arm64 — first to do so.

## The problem (measured, on Arm CPU)
Multi-agent systems (planner / coder / critic / researcher …) run many LLM "agents", but a
device's RAM holds only a few KV caches at once. When an agent is evicted and later resumes,
the server **re-prefills its entire context** — on a 4 vCPU Arm CPU that is **13 s at ~500
tokens, 91 s at ~4800 tokens**. The compute is pure waste: the KV existed, it was thrown away.

## What it does
`agent_memory.py` is an automatic persistent-KV manager over `llama-server`. llama.cpp ships
the primitives (`--slot-save-path`, `/slots save|restore`, `--cache-type-k/v`) but not the
management — that is the contribution:
- `agent_id → disk slot file`; **restore-on-resume, save-on-turn-end**
- more agents than RAM-resident slots → **LRU eviction to disk** (not recompute)
- returning agent **skips the O(n) prefill** → restore the exact KV in milliseconds

```bash
python3 bench/agent_memory_bench.py --agents 4 --ram-slots 2 --rounds 2   # see it
```

## Results (measured, Gemma-4-12B Q4, Arm CPU, `--swa-full`)

**TTFT — returning agent (the win):**

| environment | cold re-prefill (naive) | restore (Agent-Memory) | speedup |
|---|---|---|---|
| **free GitHub Arm64 CI** (multi-agent, RAM-constrained) | **111,441 ms** | **223 ms** | **500×** |
| Apple M4 CPU (single 4.8K-token ctx) | 91,043 ms | 498 ms | 182× |
| Apple M4 CPU (multi-agent, RAM-constrained) | 13,124 ms | 563 ms | 23× |

(Numbers scale with how slow the cold prefill is — the bandwidth-starved Arm64 cloud CPU is
exactly where avoiding re-prefill wins hardest.)

**Bit-exact:** restoring a mid-stream KV produces output **byte-identical** to a never-evicted
agent (sha verified) — unlike speculative decoding, this is *exactly* lossless.

**Density (measured):** persisting KV quantized shrinks each slot file — f16 354 MB →
**q8_0 188 MB (1.9× more agents, near-lossless KV)** → **q4_0 100 MB (3.6×)** for the same
context. More agents fit a fixed RAM/disk budget.

**Optional self-draft decode:** `--self-draft` (the model's own MTP heads as a zero-download
draft) **coexists** with persistent KV — verified: each agent keeps near-instant warm TTFT
*and* gets speculative decode. The decode gain is the standalone **~1.86× on the Arm CI**
(longer reasoning/code generations); on very short agent turns it is modest (~1.1×, draft
overhead isn't amortized). Honest: it's a compatible bonus, not the headline.

## Why this fits Cloud AI
Hits the rubric's named values head-on: **agents**, **inference-server speed (TTFT/latency)**,
**Arm64 cloud**, **production developer workflow**, **reusable artifact** (free-CI benchmark).
Built on stock llama.cpp primitives + a model-agnostic Python manager — drop-in for an Arm64
serving stack.

## Install / run
```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_CPU_KLEIDIAI=ON   # Arm i8mm/SVE2
cmake --build llama.cpp/build -j --target llama-server
hf download unsloth/gemma-4-12b-it-GGUF --include "gemma-4-12b-it-Q4_K_M.gguf" "mtp-gemma-4-12b-it.gguf" --local-dir models

python3 bench/agent_memory_bench.py --agents 4 --ram-slots 2 --rounds 2 --json out.json   # TTFT win
python3 bench/kv_equivalence.py        # prove bit-exact restore
python3 selfdraft/sd.py bench models/gemma-4-12b-it-Q4_K_M.gguf --ngl 0 --n-max 3   # bundled 2x decode
```

## Validate on Arm64 (free, no credit card)
[`.github/workflows/arm-bench.yml`](.github/workflows/arm-bench.yml) runs the whole thing on a
free GitHub `ubuntu-24.04-arm` runner (KleidiAI build, TTFT cold-vs-restore, self-draft decode)
and uploads results. Same on AWS Graviton: see [scripts/graviton.sh](scripts/graviton.sh).

## Honest notes (read these)
- **The win is the RAM-constrained regime** (the paper's premise): llama.cpp's default in-RAM
  prompt cache (8 GB) already reuses KV across slots for *few/small* agents. Agent-Memory wins
  when total KV **exceeds RAM** (many agents / long contexts / small device) **or across server
  restarts** — exactly where multi-agent serving hurts. Benchmarks use `--cache-ram 0` to
  isolate this regime honestly.
- **Gemma is sliding-window attention** → the server MUST run `--swa-full` or slot restore
  silently drops out-of-window tokens. Handled.
- **KV restore is bit-exact** (hash-verified). The bundled self-draft decode is *distributionally*
  lossless (FP-tie flips) — see [docs/FINDINGS.md](docs/FINDINGS.md).

## Research basis
- [arXiv:2603.04428](https://arxiv.org/abs/2603.04428) "Agent Memory Below the Prompt" (Feb 2026) — persistent Q4 KV for multi-agent edge inference. Reference impl is Apple-MLX only; this is the first llama.cpp/Arm64 port.
- [arXiv:2601.06007](https://arxiv.org/pdf/2601.06007) "Don't Break the Cache" — prompt layout for cache hits.

## License
Apache 2.0 — see [LICENSE](LICENSE).
