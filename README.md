# self-draft — zero-download self-speculative decoding for llama.cpp

**One command. Any model that ships MTP heads. ~1.5× faster decoding on Arm64 CPU. No draft model to download.**

Speculative decoding makes LLM inference faster: a cheap "draft" proposes several
tokens, the full model verifies them in one batched pass. The catch in stock
`llama.cpp` is `--model-draft`: you must find and download a *separate*,
vocabulary-matched draft model. Most models don't have one.

**self-draft removes the download.** Modern open models increasingly ship
**Multi-Token Prediction (MTP)** heads in the same release (e.g. Google's
**Gemma 4**, whose GGUFs include an `mtp-*.gguf` sibling). Those heads *are* a
draft — built from the target's own training, distributed alongside it.
self-draft auto-detects the MTP sibling and wires it into llama.cpp's
`--spec-type draft-mtp` path for you:

```bash
python3 selfdraft/sd.py run models/gemma-4-12b-it-Q4_K_M.gguf --ngl 0
# auto-resolves mtp-gemma-4-12b-it.gguf, launches a server with MTP self-speculation
```

> Built for the **Arm Create: AI Optimization Challenge — Cloud AI track**.
> Target hardware: AWS Graviton (Arm64). Dev/validation: Apple Silicon (Arm64).

---

## Results (measured)

`gemma-4-12b-it`, target `Q4_K_M`, MTP draft `mtp-gemma-4-12b-it.gguf` (0.47 GB),
greedy decode, reasoning/code workload. **Apple M4 Pro, CPU-only (`--ngl 0`)** —
the configuration that mirrors Graviton (no GPU):

| n-max | tok/s | speedup | draft accept |
|------:|------:|--------:|-------------:|
| vanilla | 22.4 | 1.00× | — |
| 2 | 19–23 | ~1.05× | 87% |
| **3** | **34.4** | **1.54×** | 76% |
| 6 | 15.9 | 0.88× | 61% |

- **~1.5× faster** at the tuned draft length (`--spec-draft-n-max 3`), **zero extra download** (MTP head ships with the model).
- Acceptance is **workload-dependent**: reasoning/code 76–87% vs short prose ~54%. Speculative shines on the structured, predictable outputs (long CoT, code) that matter for agents.
- **n-max tuning is decisive** — too long (6) over-drafts and *loses*. `sd.py autotune` finds the sweet spot automatically.

### Honest note on output equivalence

Speculative decoding is **distributionally lossless**: the full model verifies
every drafted token, so self-draft only ever emits tokens the target itself
would. It is **not guaranteed bit-identical** to *sequential* decoding, because
batched verification sums floating-point in a different order than one-token-at-
a-time decoding; at a near-tied argmax this can flip a single token (we observe
~1 flip after 232 identical characters, then a valid alternative continuation).
Every self-draft output is a **valid greedy decode of the target model**.
`sd.py bench` reports this transparently (exact-match flag + similarity + common
prefix) rather than claiming a hash match it can't always honor.

### Known issue: Apple Metal (GPU) dual-context slowdown

On Apple Silicon **GPU** (`--ngl 99`), running target + draft contexts on the
same Metal/UMA device is **catastrophically slow** (<1× — observed even with the
featherweight MTP head), a `llama.cpp` dual-context contention issue independent
of draft choice. **Use `--ngl 0` (CPU) on Apple Silicon.** On Graviton (CPU,
no GPU) the issue does not arise — which is the contest target anyway.

---

## Install / build

Requires: `cmake`, a C++ compiler, Python 3, the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) (`hf`).

```bash
# 1. clone + build llama.cpp into this repo
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release   # add -DGGML_METAL=ON on macOS
cmake --build llama.cpp/build -j

# 2. fetch a target model + its MTP sibling (Gemma 4 example)
hf download unsloth/gemma-4-12b-it-GGUF --include "gemma-4-12b-it-Q4_K_M.gguf" --local-dir models
hf download unsloth/gemma-4-12b-it-GGUF --include "mtp-gemma-4-12b-it.gguf"      --local-dir models
```

## Usage

```bash
# benchmark vanilla vs self-draft (speedup + acceptance + equivalence)
python3 selfdraft/sd.py bench models/gemma-4-12b-it-Q4_K_M.gguf --ngl 0 --n-max 3

# autotune the draft length for your model + workload
python3 selfdraft/sd.py autotune models/gemma-4-12b-it-Q4_K_M.gguf --ngl 0 --grid 1,2,3,4,6,8

# launch a self-draft server (OpenAI-compatible /completion on :8099)
python3 selfdraft/sd.py run models/gemma-4-12b-it-Q4_K_M.gguf --ngl 0 --n-max 3
```

Flags: `--ngl` GPU layers (**use 0 on Apple Silicon**), `--threads`, `--ctx`,
`--workload code|prose|<literal prompt>`, `--n-predict`, `--n-max`, `--json out.json`.

`sd.py` works with **stock** llama.cpp. Optionally, [`patches/`](patches/) adds a
native one-flag `--self-draft` to llama.cpp itself
(`llama-server -m model.gguf --self-draft`), which auto-resolves a local
`mtp-*.gguf` sibling — see [patches/README.md](patches/README.md).

## Validate on AWS Graviton (Arm64)

```bash
# on a Graviton instance (e.g. c7g/t4g, Ubuntu): build is identical, no -DGGML_METAL
sudo apt-get update && sudo apt-get install -y build-essential cmake python3 git pipx && pipx install huggingface_hub
git clone --depth 1 https://github.com/ggml-org/llama.cpp && cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release && cmake --build llama.cpp/build -j
hf download unsloth/gemma-4-12b-it-GGUF --include "gemma-4-12b-it-Q4_K_M.gguf" "mtp-gemma-4-12b-it.gguf" --local-dir models
python3 selfdraft/sd.py autotune models/gemma-4-12b-it-Q4_K_M.gguf --ngl 0   # CPU = native Graviton path
```

`--ngl 0` is the native path on Graviton (no GPU). Expect the same ~1.5× on
reasoning/code workloads; the autotuner picks the best `--spec-draft-n-max` for
the instance's core count.

## License

Apache 2.0 — see [LICENSE](LICENSE).
