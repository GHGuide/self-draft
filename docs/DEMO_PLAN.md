# Demo video plan (≤3 min)

Goal: judge sees self-draft running on the target device (Arm64) and getting faster
with zero extra download, honestly. No copyrighted material.

## Shot list

**0:00–0:20 — Hook.** Title card: "self-draft — your model already ships a draft."
One line: stock llama.cpp needs a separate downloaded draft model; most models have
none; MTP heads change that.

**0:20–0:50 — The download story.** Terminal: `ls -lh models/` showing the 0.47 GB
`mtp-gemma-4-12b-it.gguf` sitting next to the model. "No second model to download —
this head shipped with Gemma 4." Run `sd.py run … --ngl 0`; show it auto-resolving
the MTP sibling in the log.

**0:50–1:50 — Side-by-side speed.** Split screen, same Arm64 box, same prompt
(a coding/reasoning task), greedy:
- left: vanilla `llama-server`
- right: `sd.py` self-draft
Token counters racing; right finishes clearly first. Cut to `sd.py bench` output:
`SPEEDUP: 1.54x`, `accept 76%`.

**1:50–2:20 — Autotune.** Run `sd.py autotune … --grid 1,2,3,4,6,8`. Show the curve:
n=3 wins, n=6 *loses*. "Draft length is the knob everyone gets wrong — we tune it for you."

**2:20–2:45 — Honesty + Arm.** Show the equivalence line (distributionally lossless,
1 FP-tie flip; valid greedy decode). Flash the Graviton build commands from the README.

**2:45–3:00 — Close.** "One command. Any model with MTP heads. ~1.5× on Arm64. Zero
download." Repo URL + Apache 2.0.

## Capture notes
- Record on the Graviton instance if available (best — "on target device"); else
  Apple Silicon CPU path (`--ngl 0`) and state the Graviton numbers from the README.
- Use `asciinema` or screen capture; keep terminals large and high-contrast.
- Pre-warm model loads off-camera so the demo shows *decode* speed, not load time.
- Upload public to YouTube/Vimeo; link in Devpost.
