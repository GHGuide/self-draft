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

## 6. We tried to make it work WITHOUT MTP heads - and measured why it doesn't
MTP only works on the handful of models that ship MTP heads. We attempted a universal,
training-free alternative: derive a draft from the target's OWN layers (early-exit / layer
subset) + its tied embedding head, verified losslessly by `draft-simple`. We built a GGUF
surgery tool (`selfdraft/make_self_draft.py`) that slices the first K of N=48 Gemma-4-12B
layers (copying quantized bytes verbatim, reslicing the per-layer `head_count_kv` and
`sliding_window_pattern` arrays) and swept K (CPU, code workload):

| K (of 48) | draft accept | speedup |
|---|---|---|
| 44 | 27.3% | 0.23x |
| 40 | 15.6% | 0.14x |
| 36 | 6.7%  | 0.10x |
| 32 | 3.2%  | 0.12x |
| 24 | 0.0%  | 0.14x |

**Conclusion: training-free early-exit self-draft does not work on stock Gemma 4.** Even
keeping 44/48 layers (skipping just 4), the draft matches the full model only 27% of the
time - the model is not trained for early exit, so projecting an intermediate hidden state
through the tied head gives near-random tokens. And a 44-layer draft is barely cheaper than
the 48-layer target, so you pay near-full draft cost for almost no accepted tokens => 4-10x
SLOWER. Output stays correct (draft-simple verifies; 92-100% similarity) - it is lossless
but useless. Salience-based layer selection cannot close a 0-27% acceptance gap; the fix
requires *training* a draft (which is exactly what MTP/Eagle/LayerSkip do, and what breaks
"zero-download"). This is why self-draft uses the model's shipped, trained MTP heads.

## 7. Memory-cache knobs (flash-attn + q8_0 KV) help only at LONG context - measured
We exposed Arm memory-bound knobs (`--fa on --ctk/--ctv q8_0`, draft-KV-quant, `--mlock`)
and A/B'd on the free Arm64 runner at our agentic/code context (n_predict=256, ctx 4096):

| config | self-draft tok/s | speedup |
|---|---|---|
| MTP baseline | 12.15 | 2.01x |
| MTP + flash-attn + q8_0 KV + mlock | 10.89 | 1.90x (-10%) |

KV-cache quantization is a long-context optimization: at a few hundred tokens the KV cache
is small, so the per-access q8_0 dequant overhead outweighs the bandwidth saving, and the
CPU flash-attn path adds cost on small attention. The knobs are kept as OPTIONAL flags
(they pay off when the verify step re-reads a large KV cache, i.e. long-context serving),
but they are NOT the default - the simple MTP path at n-max=3 is faster for typical
agentic/coding generations. Lesson: match the optimization to the context length.

## 8. Adaptive draft-length (online n-max) loses over HTTP - needs an in-loop controller
We patched llama.cpp to enable per-request `speculative.n_max` (it ships `#if 0`'d out;
3-file change: server-task parse, server-context `get_n_draft_max` clamp, MTP loop honoring
the per-call ceiling). Verified live: n_max=1 -> draft_n=61 accept 95%; n_max=8 -> draft_n=184
accept 52%. We then built an EMA controller (`sd.py adaptive`) that varies n_max per chunk
from rolling acceptance. On a mixed code+prose+JSON workload (where a static n is wrong for
half the run):

| | tok/s |
|---|---|
| best static (n-max=3) | 24.87 |
| adaptive (EMA controller) | 15.83 (0.64x) |

Adaptive LOST by 36%. Two reasons: (1) the controller lags at acceptance-region boundaries
(it raised n right as a region went cold); (2) more fundamentally, an HTTP-chunked
controller pays per-request overhead (15 requests vs static's 1) that swamps any gain. The
literature's +5-15% assumes the controller runs INSIDE the decode loop with zero request
overhead. Doing it right requires an in-server C++ controller (adjust slot n_max per step) -
a large change with modest, workload-specific upside; not worth it given the pattern below.
The per-request-n_max patch itself is a real, working capability (kept in `patches/`).

## Takeaway: five measured negative/null results, one robust win
We systematically tried to beat the simple MTP self-draft (static n-max=3, ~2.0x on Arm64):
same-size Q2 draft (slower), early-exit layer-subset draft (4-10x slower), MTP+n-gram
cascade (wash on Arm), flash-attn + q8_0 KV-cache (slower at short ctx), and adaptive n-max
(slower over HTTP). None beat it. The lesson: on this stack the cheap, shipped MTP head with
a tuned static draft length is at the throughput ceiling for short agentic/coding workloads;
the extra machinery costs more than it saves. The winning recipe stays simple - and we have
the data to prove the simple thing is right.
On Arm64 cloud, the winning recipe is: a draft that is *actually cheap* (MTP head, shipped
free with the model) + per-instance draft-length autotuning + honest latency/cost metrics,
stacked on Arm's KleidiAI kernels. One flag, no extra download. We also showed, with data,
that the obvious training-free shortcuts (same-size Q2 draft; early-exit layer-subset draft)
do NOT work - useful negative results for anyone optimizing inference on Arm.
