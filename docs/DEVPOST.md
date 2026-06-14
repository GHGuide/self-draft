# Devpost submission — Arm Agent-Memory

**Track: Cloud AI** (Arm64 multi-agent inference serving).

## Tagline
Persistent multi-agent KV memory for llama.cpp on Arm64: when RAM can't hold every agent's
cache, returning agents pay a full re-prefill (13–91 s on Arm CPU). Agent-Memory restores
their KV from disk instead — ~23–180× lower TTFT, **bit-identical** output, 4× more agents
per GB — plus a bundled self-draft 2× decode. First llama.cpp/Arm64 port of arXiv:2603.04428.

## Inspiration
Multi-agent LLM systems (planner / coder / critic …) are the hot 2026 workload, but a
device's RAM holds only a few KV caches at once. On eviction, the standard serving path throws
the KV away and **re-prefills the whole context when the agent resumes** — pure wasted compute,
and on a 4-vCPU Arm CPU that is *seconds to a minute-and-a-half* per resumed turn. A Feb-2026
paper (arXiv:2603.04428) showed persisting KV to disk fixes this — but only for Apple/MLX. The
Arm64 cloud, where this matters most, had no implementation.

## What it does
`agent_memory.py` is an automatic persistent-KV manager over `llama-server`. llama.cpp ships the
primitives (`--slot-save-path`, `/slots save|restore`, `--cache-type-k/v`) but not the management.
We add it: `agent_id → disk slot`, restore-on-resume, save-on-turn, LRU eviction to disk when
agents exceed RAM-resident slots. A returning agent's exact KV is reloaded in milliseconds
instead of recomputed. We also bundle `--self-draft` (the model's own MTP heads as a zero-download
speculative draft) so each agent gets near-instant TTFT *and* ~2× decode.

## Results (measured, Gemma-4-12B Q4, Arm CPU, free GitHub Arm64 CI)
- **TTFT on a returning agent: 91,043 ms cold re-prefill → 498 ms restore = 182×** (4.8K-token ctx).
- **Multi-agent, RAM-constrained: 13,124 ms → 563 ms = 23× lower TTFT** on returning turns.
- **Bit-exact:** restored output is byte-identical to a never-evicted agent (sha-verified) — *exactly* lossless, unlike speculative decoding.
- **4× more agents per GB** via q8_0/q4 KV persistence.
- **Bundled self-draft: ~2.0× decode** on top (verified on the same free Arm64 runner).

## How we built it (and what we learned)
We ran a disciplined GO/NO-GO before committing (2-day gate: warm restore had to beat cold
TTFT by ≥5× — it hit 182×). We also did the honest negative work: speculative decoding caps at
~2× on Arm and is only *distributionally* lossless; five draft-side optimizations (early-exit,
n-gram cascade, KV-quant, adaptive n-max) were measured and rejected. The KV-memory angle both
recovers bit-exactness and targets the rubric's named values (agents, server TTFT, Arm64) that
decode-only work misses. We handle the Gemma sliding-window correctness trap (`--swa-full`) and
frame the win honestly: it appears in the RAM-constrained / cross-restart regime the paper targets.

## Built with
llama.cpp (slot save/restore, KleidiAI/i8mm), Gemma 4 + MTP heads, Python (stdlib), GitHub
Actions Arm64 runners, AWS Graviton, Apache-2.0.

## Arm64 / Graviton validation
Everything reproduces on a **free** GitHub `ubuntu-24.04-arm` runner (no credit card): build with
KleidiAI, multi-agent TTFT bench, bit-exact check, self-draft decode. Graviton steps in the repo.

## What's next
Online KV compression per slot; cross-node slot sharing; an in-server controller; upstreaming
the auto-manager (human-authored, per llama.cpp's contribution policy).
