# Engineering findings: making self-speculative decoding pay off on Arm64

A short, reproducible account of what we measured building self-draft. Useful to anyone
trying speculative decoding in llama.cpp on Arm. All numbers: `gemma-4-12b-it`, greedy,
reasoning/code workload; Apple M4 Pro (Arm64) dev box unless noted; reproduce on Graviton
with `scripts/graviton.sh`.

## 1. A same-size quantized draft does NOT work (the folk trick fails)
The common suggestion is "re-quantize your model to Q2 and use it as the draft." We tested
it (`--spec-type draft-simple`, target Q4_K_M, draft Q2_K):

| config | speedup |
|---|---|
| GPU Metal | 0.03x |
| CPU (Graviton-like) | 0.28x |

A Q2 copy of a 12B model has the **same layer count and same FLOPs** as the target - only
~1.5x less memory bandwidth. Drafting N tokens costs ~N full forward passes, so the
theoretical ceiling is <1x even at 100% acceptance. **The draft must be genuinely cheap,
not just lower-precision.** That is what MTP heads (and layer-pruned drafts) provide.

## 2. MTP heads are the cheap draft - and they ship with the model
Gemma 4 releases include an `mtp-*.gguf` Multi-Token-Prediction head (0.47 GB). Wired via
`--spec-type draft-mtp`, on CPU:

| n-max | speedup | accept |
|---|---|---|
| 2 | 1.09x | 86% |
| **3** | **1.54x** | 81% |
| 6 | 1.10x | 65% |

Two lessons: **acceptance is workload-dependent** (76-87% on code/reasoning vs ~54% on
short prose), and **draft length is decisive** - too long over-drafts and *loses* (n=6 <
n=3). Hence the autotuner.

## 3. "Byte-identical" is the wrong claim on CPU
We hash-checked output vs sequential decode. vanilla, draft-mtp, and draft-simple all
produced **different** hashes - but draft-mtp and draft-simple agreed with each other and
diverged from sequential vanilla at a single point (a near-tied argmax: "memo dictionary"
vs "dictionary"), then continued validly. Root cause: **batched verification sums
floating-point in a different order than one-token-at-a-time decoding**; at a tie this
flips one token. Speculative decoding is **distributionally lossless** (the target verifies
every token) but **not bit-identical** to sequential decode. We report similarity + common
prefix honestly rather than claiming a hash match.

## 4. Speculative decoding is broken on Apple Metal/UMA (use CPU)
On Apple Silicon GPU, target+draft on one Metal/UMA device gave **<1x for every draft**,
including the featherweight MTP head - a llama.cpp dual-context contention issue, not a
draft-cost problem. We default to `--ngl 0` (CPU) on Apple Silicon. Graviton (CPU-only) is
unaffected, and is the Cloud AI target anyway. (Worth an upstream issue.)

## 5. It is a latency optimization, not a throughput one
Speculative decoding trades extra compute for lower latency. It wins single-stream /
low-concurrency / latency-sensitive (agentic) serving; it does not raise max-batch
throughput. We frame and benchmark it accordingly (TTFT, per-request latency, $/token).

## Takeaway
On Arm64 cloud, the winning recipe is: a draft that is *actually cheap* (MTP head, shipped
free with the model) + per-instance draft-length autotuning + honest latency/cost metrics,
stacked on Arm's KleidiAI kernels. One flag, no extra download.
