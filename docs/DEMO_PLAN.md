# Demo video plan — Arm Agent-Memory (<3 min; aim ~2:25)

Rule: <=3 min, must show it running on Arm64, public YouTube/Vimeo/Youku, English, no
copyrighted music/trademarks. Judges may score on video+text alone -> every number on screen.

## Timed shot list (~2:25)
- 0:00-0:12  HOOK. Title card. "Multi-agent LLM on Arm CPU: every resumed agent re-prefills
  its whole context - up to 91s of pure waste."
- 0:12-0:35  THE WASTE. Naive terminal: an agent resumes, re-prefill timer ticks 13-91s.
  "The KV existed. It was thrown away."
- 0:35-1:20  HERO (split screen, same Arm box). Agent-Memory restores the agent's KV ->
  first token in ~0.2s, decoding immediately. Big number: 500x lower TTFT.
- 1:20-1:45  BIT-EXACT. Two sha256 side by side, identical. "Provably the same output."
- 1:45-2:05  DENSITY + BUNDLE. q4 KV = 3.6x more agents/GB; flip on --self-draft, decode races.
- 2:05-2:25  PROOF + CLOSE. Green free Arm64 GitHub CI run (reproducible, $0) + repo URL +
  "Cloud AI . Arm64 . zero extra downloads."

## Maximize the rubric (judges = Arm staff engineers)
- Tech 40 (also the tiebreaker): show the slot manager, --swa-full handling, CI building
  llama.cpp with KleidiAI i8mm/SVE2 on Neoverse, "first llama.cpp/Arm64 port of arXiv:2603.04428".
- WOW 25: the 500x cold-vs-warm visual + bit-exact - lead with it.
- Impact 20: multi-agent = 2026 workload; model-agnostic; first Arm64 port; forkable free-CI bench.
- DX 15: one command, copy-paste README, free-CI reproducibility.

## Capture notes
- Record on an Arm64 box (M4 / Graviton) or screen-capture the public Arm64 CI run (counts as
  "on the device for which it was built"). Pre-warm model loads off-camera so the demo shows
  TTFT/decode, not load time. asciinema or screen recorder; large high-contrast terminals.
