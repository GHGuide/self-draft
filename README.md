# self-draft - drop-in latency optimization for LLM inference servers on Arm64 cloud

**One flag. Your model's own Multi-Token-Prediction heads become the draft. Up to ~2x
lower-latency decoding for agentic/reasoning workloads on Arm64 cloud - zero extra
model download, ~50% lower cost-per-token, stacked on Arm KleidiAI kernels.**

> **Verified on a free GitHub-hosted Arm64 runner (4 vCPU aarch64): 2.0x speedup,
> 1.87x lower latency, 99.1% output similarity, n-max=3** - reproducible in CI, see below.

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

`gemma-4-12b-it`, MTP draft `mtp-gemma-4-12b-it.gguf` (0.47 GB), greedy, reasoning/code
workload, CPU-only (`--ngl 0`, the native Arm64-cloud path).

### Arm64 cloud (headline) - free GitHub `ubuntu-24.04-arm` runner, 4 vCPU aarch64, Q4_0 + KleidiAI

Fully reproducible in CI ([`.github/workflows/arm-bench.yml`](.github/workflows/arm-bench.yml)):

**Draft-length autotune** (`sd.py autotune`):

| n-max | tok/s | speedup | accept |
|------:|------:|--------:|-------:|
| vanilla | 5.99 | 1.00x | - |
| 1 | 7.74 | 1.29x | 90% |
| 2 | 9.21 | 1.54x | 82% |
| **3** | **12.11** | **2.02x** | 78% |
| 4 | 7.53 | 1.26x | 70% |

**Bench** (n-max=3): vanilla **5.99 -> 12.01 tok/s = 2.0x**; end-to-end latency
**36.2 s -> 19.3 s (1.87x lower)**; draft acceptance 76%; output similarity **99.1%**
(1 floating-point-tie flip, see notes). Since cost is inversely proportional to
throughput, 2.0x tok/s = **~50% lower $/1M tokens** on any priced Arm instance.

### Apple M4 Pro (Arm64 laptop, dev) - Q4_K_M

For reference, a fast M-series core (less memory-bound) shows a smaller but real win:
n-max=3 -> **1.54x** decode (vanilla 18.6 -> 28.6 tok/s, 81% accept), latency 1.23x lower,
~23% lower $/token. **Self-draft helps *more* on the weaker, more memory-bound Arm cloud
cores** - exactly where it matters for cost.

**Agent loop** (`sd.py agent`, ReAct + calculator tool, correct answer both): end-to-end
win on agentic workloads (short per-step generations are TTFT-bound; the decode win
dominates on longer outputs).

Key takeaways: **up to 2.0x decode / ~50% lower $/token on Arm64 cloud, zero extra
download.** n-max tuning is decisive (too long *loses*) - the autotuner picks the sweet
spot per instance.

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

**Free Arm64-cloud run (no instance, no credit card):**
[`.github/workflows/arm-bench.yml`](.github/workflows/arm-bench.yml) runs the whole
benchmark on a GitHub-hosted `ubuntu-24.04-arm` runner (free for public repos) - build
with KleidiAI, fetch model + MTP, autotune, bench - and uploads the results JSON as an
artifact. Trigger it from the repo's **Actions** tab ("Run workflow"). This is also the
reproducible CI/DX artifact for the submission.

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

## Engineering findings
The rigorous path to these numbers (why a same-size Q2 draft fails, why draft length is
decisive, the floating-point equivalence analysis, the Metal/UMA issue) is written up in
[docs/FINDINGS.md](docs/FINDINGS.md).

## License
Apache 2.0 - see [LICENSE](LICENSE).
