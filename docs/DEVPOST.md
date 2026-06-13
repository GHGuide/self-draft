# Devpost submission — self-draft

**Track: Cloud AI** (Arm64 cloud inference performance + production developer workflow).

## Tagline
A one-flag, drop-in latency optimization for LLM inference servers on AWS Graviton:
your model's own MTP heads become the draft - ~1.5x faster decoding and ~23% lower
$/token on agentic/reasoning workloads, zero extra download, stacked on Arm KleidiAI.

## Inspiration
Speculative decoding is the highest-leverage inference optimization that *doesn't*
touch model quality: a cheap draft guesses tokens, the full model verifies them in
one batched pass. But `llama.cpp`'s `--model-draft` needs a separate, vocabulary-
matched draft model you have to find and download — and most models don't have one.
Meanwhile, a new generation of open models (Google **Gemma 4**, and others) now ship
**Multi-Token Prediction (MTP)** heads right in the release. Those heads *are* a
draft — trained with the model, distributed with it. Nobody was wiring them up
automatically. So we did.

## What it does
`self-draft` is a thin tool over `llama.cpp` that:
1. **Auto-detects** a model's MTP draft sibling (`mtp-*.gguf`) — no separate download.
2. **Wires it** into llama.cpp's `--spec-type draft-mtp` self-speculation path.
3. **Autotunes** the draft length (`--spec-draft-n-max`) — the single most important
   knob, which most users get wrong (too long actually *slows you down*).
4. **Benchmarks honestly** — tok/s, draft acceptance, and a transparent
   output-equivalence report.

One command:
```
python3 selfdraft/sd.py run models/gemma-4-12b-it-Q4_K_M.gguf --ngl 0
```

## Results
On `gemma-4-12b-it` (Q4_K_M), CPU-only (the Graviton-equivalent path), reasoning/code
workload: **1.54x faster decoding** at the autotuned draft length, **TTFT and end-to-end
latency down**, **~23% lower $/1M tokens**, **76-87% draft acceptance**, **0.47 GB MTP
head** vs downloading a whole second model. A working ReAct agent demo (tool-calling,
correct answer) shows the end-to-end serving benefit. Numbers reproduce on Graviton via
`scripts/graviton.sh` (builds with Arm KleidiAI kernels).

## How we built it (and what we learned)
We ran a disciplined go/no-go before committing:
- A **naive Q2 re-quant of the same model as a draft** (the folk trick) is a dead end:
  it has the *same layer count / same FLOPs* as the target, so it's not actually
  cheaper — we measured **<1×** on every configuration. The win has to come from a
  draft that is genuinely cheap, which is exactly what MTP heads are.
- We found and characterized two real `llama.cpp` behaviors:
  (a) On **Apple Metal/UMA GPU**, running target+draft contexts on one device is
  catastrophically slow (a dual-context contention issue) — so we default to CPU on
  Apple Silicon; Graviton (CPU) is unaffected.
  (b) Speculative decoding is **distributionally lossless** but **not bit-identical**
  to sequential decoding (batched-vs-sequential floating-point ordering flips rare
  argmax ties). We report this honestly instead of overclaiming a hash match.

## Built with
`llama.cpp` (`draft-mtp` speculative path), Python (stdlib), Gemma 4 GGUF + MTP heads,
Apache 2.0.

## Arm64 / Graviton validation
Full build + run instructions for AWS Graviton in the README. `--ngl 0` is the native
CPU path; the autotuner adapts the draft length to the instance.

## What's next
- Upstream a `--self-draft` convenience flag to `llama.cpp` (auto-enable MTP when the
  model has heads).
- Investigate the Metal dual-context slowdown for a possible upstream fix.
- Extend autotuning to adapt n-max online from a rolling acceptance estimate.
