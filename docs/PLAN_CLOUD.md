# Cloud AI track — winning plan

Track: **Cloud AI** ("scalable infrastructure, Arm64 cloud, inference performance,
frameworks, agents, production-ready developer workflows"). Prize target: Best-in-
Category Cloud AI ($1,000), with a shot at Overall via WOW + rigor.

## Positioning (one sentence)
**self-draft: a drop-in, one-flag latency optimization for LLM inference servers on
Arm64 cloud (AWS Graviton) — your model's own MTP heads become the draft, so you get
~1.5x lower-latency decoding on agentic/reasoning workloads with zero extra download,
stacked on Arm's KleidiAI kernels, with a measured cost-per-token reduction.**

## Why this fits Cloud AI (not just "model speed")
- Inference-server optimization: implemented as a `llama-server` flag; we report TTFT,
  per-request latency, tokens/sec, and $/1M-tokens on Graviton.
- Arm64 cloud: benchmarked on Graviton (c7g/c8g), built with KleidiAI (Arm Neoverse
  SVE2 / i8mm matmul) — our layer sits on top of Arm's optimized kernels.
- Agents: speculative decoding wins exactly where agents live - long, structured,
  predictable outputs (tool calls, reasoning/CoT). 76-87% draft acceptance measured.
- Production DX: one flag, autotuned to the instance, OpenAI-compatible server.

## Honest framing (do not overclaim)
- Latency optimization for low-concurrency / latency-sensitive / agentic serving.
  Speculative decoding trades compute for latency: it wins single-stream and small
  batch, NOT max-throughput high-concurrency. State the use case explicitly.
- Distributionally lossless, not bit-identical vs sequential decode (FP-tie analysis).
- MTP heads are shipped by the model (Gemma 4); our contribution is the auto-wiring,
  the autotuner, the Arm64-cloud benchmarking + cost analysis, and the llama.cpp flag.

## Metrics to report (Graviton)
- tokens/sec (decode), TTFT, end-to-end request latency (vanilla vs self-draft).
- draft acceptance % by workload (code/reasoning vs prose).
- $/1M tokens = (instance $/hr) / (tokens/sec * 3600 / 1e6), vanilla vs self-draft.
- KleidiAI on vs off (Arm-specific gain), and self-draft stacked on KleidiAI.
- Instance sweep if budget allows: c7g.large/xlarge (and c8g if available).

## Work items
1. `sd.py`: add TTFT + latency + $/token reporting; `serve`/agent-demo subcommand. [me]
2. Graviton one-shot script: provision deps, build llama.cpp +KleidiAI, fetch model+MTP,
   autotune, full bench + cost table. [me writes / user runs]
3. Reframe README + DEVPOST for Cloud AI server/latency/agents/cost. [me]
4. Agent-loop demo (tool-calling / multi-step reasoning) showing latency cut. [me]
5. Publish repo, record <=3min demo on Graviton, Devpost form. [user]

## Stretch
- File llama.cpp ISSUE for Metal/UMA dual-context speculative slowdown (impact/learning).
- Arm Performix run for authoritative Arm benchmarks.
