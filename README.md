# self-draft - drop-in latency optimization for LLM inference servers on Arm64 cloud

**One flag. Your model's own Multi-Token-Prediction heads become the draft. ~1.5x
lower-latency decoding for agentic/reasoning workloads on AWS Graviton - zero extra
model download, measurable cost-per-token reduction, stacked on Arm KleidiAI kernels.**

> **Arm Create: AI Optimization Challenge - Cloud AI track.**
> Target: AWS Graviton (Arm64) inference serving. Dev/validation: Apple Silicon (Arm64).

Speculative decoding speeds up LLM inference: a cheap "draft" proposes several tokens,
the full model verifies them in one batched pass. Stock `llama.cpp`'s `--model-draft`
needs a *separate, downloaded, vocabulary-matched* draft model - most models don't have
one. **self-draft removes the download:** modern models (Google **Gemma 4**) now ship
**MTP heads** in the same release (`mtp-*.gguf`). Those heads *are* a draft. self-draft
auto-detects and wires them into llama.cpp's `--spec-type draft-mtp` path:

```bash
python3 selfdraft/sd.py run models/gemma-4-12b-it-Q4_K_M.gguf --ngl 0   # CPU = Graviton path
# auto-resolves mtp-gemma-4-12b-it.gguf, serves OpenAI-compatible /completion with MTP self-speculation
```

## Why this is a Cloud AI optimization

- **Inference-server flag**, not a one-off script: it's a `llama-server` option; we report
  the metrics serving teams care about - **tokens/sec, TTFT, end-to-end latency, $/1M tokens**.
- **Arm64 cloud**: validated on **AWS Graviton**, built with **KleidiAI** (Arm Neoverse
  i8mm/SVE2 matmul micro-kernels) - our optimization layers on top of Arm's own kernels.
- **Agents**: speculative decoding wins exactly where agents live - long, structured,
  predictable outputs (tool calls, reasoning/CoT). We measure **76-87% draft acceptance**
  on code/reasoning vs ~54% on short prose, and ship a working ReAct agent demo.
- **Production DX**: one flag, autotuned to the instance, drop-in for an existing Arm64
  inference server.

## Results (measured)

`gemma-4-12b-it` Q4_K_M, MTP draft `mtp-gemma-4-12b-it.gguf` (0.47 GB), greedy,
reasoning/code workload, CPU-only (`--ngl 0`, the Graviton-equivalent path).
**Dev numbers below are Apple M4 Pro (Arm64); reproduce on Graviton with
[`scripts/graviton.sh`](scripts/graviton.sh).**

**Draft-length autotune** (`sd.py autotune`):

| n-max | tok/s | speedup | accept |
|------:|------:|--------:|-------:|
| vanilla | 18.6 | 1.00x | - |
| 2 | 20.2 | 1.09x | 86% |
| **3** | **28.6** | **1.54x** | 81% |
| 4 | 28.3 | 1.52x | 77% |
| 6 | 20.5 | 1.10x | 65% |

**Server metrics + cost** (`sd.py bench --price`, n-max=3; $/hr is illustrative
c7g.xlarge - replace with your Graviton numbers):

| | tok/s | TTFT | latency | $/1M tok |
|---|------:|-----:|--------:|---------:|
| vanilla | 21.3 | 3267 ms | 10.8 s | $1.89 |
| **self-draft** | **27.8** | 3049 ms | **8.8 s** | **$1.45 (-23%)** |

**Agent loop** (`sd.py agent`, ReAct + calculator tool, correct answer both):
~1.1-1.2x end-to-end (short per-step generations are TTFT-bound; the decode win
dominates on longer outputs).

Key takeaways: **~1.5x decode / ~23% lower $/token, zero extra download**, n-max tuning
is decisive (too long *loses* - the autotuner picks the sweet spot per instance).

## Install / build

Requires `cmake`, a C++ compiler, Python 3, the HF CLI (`hf`).

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release    # +Arm: -DGGML_CPU_KLEIDIAI=ON
cmake --build llama.cpp/build -j --target llama-server
hf download unsloth/gemma-4-12b-it-GGUF --include "gemma-4-12b-it-Q4_K_M.gguf" "mtp-gemma-4-12b-it.gguf" --local-dir models
```

## Usage

```bash
python3 selfdraft/sd.py autotune models/gemma-4-12b-it-Q4_K_M.gguf --ngl 0          # pick n-max
python3 selfdraft/sd.py bench     models/...Q4_K_M.gguf --ngl 0 --n-max 3 --price 0.145   # tok/s, TTFT, latency, $/1M tok
python3 selfdraft/sd.py agent     models/...Q4_K_M.gguf --ngl 0 --n-max 3            # agent e2e latency, vanilla vs self-draft
python3 selfdraft/sd.py run       models/...Q4_K_M.gguf --ngl 0 --n-max 3            # serve /completion on :8099
```

`--ngl 0` = CPU (the native Graviton path; **required on Apple Silicon**, see below).
Optional native one-flag: [`patches/`](patches/) adds `--self-draft` to llama.cpp itself.

## Validate on AWS Graviton (Arm64)

```bash
bash scripts/graviton.sh 0.145    # provision deps, build llama.cpp +KleidiAI, fetch model+MTP,
                                  # autotune + bench with cost table. Pass your instance $/hr.
```
`--ngl 0` is the native CPU path on Graviton; KleidiAI accelerates the Q4_0 quant via
Arm i8mm/dotprod. The autotuner adapts the draft length to the instance core count.

## Honest notes (read these)

- **Distributionally lossless, not bit-identical.** The full model verifies every drafted
  token, so output is always a valid greedy decode of the target. But batched verification
  sums floating-point in a different order than sequential decoding, so a near-tied argmax
  can flip ~1 token over a long generation (we observe identical text up to a tie, then a
  valid alternative continuation). `sd.py bench` reports this transparently.
- **Latency, not max throughput.** Speculative decoding trades compute for latency - it
  wins single-stream / low-concurrency / latency-sensitive (agentic) serving, not
  max-batch throughput. Use it where per-request latency matters.
- **Apple Silicon: use `--ngl 0` (CPU).** On Metal GPU, running target+draft on one
  UMA device is a `llama.cpp` dual-context slowdown (<1x, even for the tiny MTP head).
  Graviton (CPU) is unaffected - and is the contest target.

## License
Apache 2.0 - see [LICENSE](LICENSE).
