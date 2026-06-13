# Plan: Novel non-MTP self-speculative decoding ("self-draft for ANY model")

Executor: this is a staged plan with HARD go/no-go gates. Do NOT write C++ until Stage 1
proves the method works. Fail fast. Be empirical: measure before building. Every claim in
the repo must stay honest (distributionally lossless, real numbers only).

## 0. Goal & thesis

Make self-speculative decoding work on **any GGUF model** - no shipped MTP head, no
separately downloaded draft. Derive a **cheap draft from the target model's OWN weights**
(a subset of its transformer layers + its own embedding/output head), then use llama.cpp's
existing, provably-lossless `draft-simple` verification. One flag, zero download, any model.

This is the "Cassandra" idea done for real: salience-aware layer pruning to build a
self-draft. If it works on stock (un-finetuned) models it is genuinely novel and broadly
useful - the current MTP project only works on the ~handful of models that ship MTP heads.

**The make-or-break unknown (be honest):** stock models are NOT trained for early exit, so
a truncated-layer draft may have LOW token-agreement with the full model -> low acceptance
-> no speedup. Stage 1 exists solely to measure this in hours. If it fails, salience
selection (Stage 2) is the rescue; if that fails too, we abandon and keep the clean MTP 2x.

## 1. Background facts (verified in this llama.cpp, b1-ebc1077)

- Gemma 4 arch string = `gemma4`; layer count key = `gemma4.block_count`. Graph layer loop:
  `src/models/gemma4.cpp:202` `for (int il = 0; il < n_layer; ++il)`, output norm/head after.
- GGUF tensors: `token_embd.weight`, per-layer `blk.{i}.*`, final `output_norm.weight`,
  `output.weight` (Gemma ties output to token_embd - verify; if tied, no separate `output`).
- `draft-simple` (`--spec-type draft-simple -md <draft.gguf>`) is the lossless draft-model
  path. Draft + target only need identical vocab (a layer-subset of the same model = same
  vocab). We already proved draft-simple is distributionally lossless (FP-tie flips only).
- Gemma 3/4 use **interleaved local/global (sliding-window) attention** - the attention
  pattern may be layer-indexed metadata. Truncating the FIRST K layers preserves the
  prefix pattern; REORDERING layers (salience) must remap any per-layer metadata. Handle.
- gguf-py lives in `llama.cpp/gguf-py` (read/write GGUF from Python).

## 2. Method: Early-Exit Self-Draft (EESD), then Salience-Selected Self-Draft (SSD)

- **EESD (Stage 1):** draft = layers [0..K) of target + target's output_norm + output head.
- **SSD (Stage 2):** draft = the K *most important* layers (chosen by a calibration pass),
  remapped to 0..K-1. Higher acceptance per FLOP than uniform truncation.
- Verification is always the full target via `draft-simple` -> output stays a valid greedy
  decode of the target (lossless in exact arithmetic).

Rough speedup model (use to set expectations & gates): draft cost ~ K/N of a full pass.
With draft length n and average accepted a, speedup ~= (a+1) / (n*(K/N) + 1).
- K=N/2, n=4, a=3 -> ~1.33x ; K=N/3, n=4, a=3 -> ~1.7x ; K=N/4, n=5, a=4 -> ~2.2x.
=> need BOTH smallish K AND high acceptance. That is the whole game.

---

## STAGE 1 - Feasibility (PYTHON ONLY, no C++). ~half a day. GATE.

### 1.1 Build the draft generator
Create `selfdraft/make_self_draft.py` using `gguf-py`:
- Args: `--in target.gguf --out draft.gguf --keep K` (and later `--layers i,j,k...`).
- Read target GGUF. Copy: all non-`blk.*` tensors (token_embd, output_norm, output if
  present, rope/other globals). Copy `blk.0..blk.{K-1}` verbatim.
- Set metadata `gemma4.block_count = K` (use the model's arch prefix read from
  `general.architecture`). Copy ALL other metadata unchanged.
- For Gemma sliding-window/attention-pattern keys that are per-layer arrays: truncate them
  to length K (first K). If a key is a scalar pattern (e.g., "every Nth layer global"),
  leave as-is but VERIFY the truncated model loads without assertion.
- Write draft.gguf. Print size (expect ~K/N of target + embeddings/head).

### 1.2 Validate the draft loads & runs
`llama-server -m draft.gguf -ngl 0 -c 2048` then a /completion. It will produce LOW-quality
text (that's fine - it is only a draft). MUST load without crash and emit tokens.
If the loader errors on missing/extra tensors or the attention pattern: fix metadata in 1.1.

### 1.3 Measure acceptance vs K (THE GATE)
For K in {N, 3N/4, N/2, N/3} (N = target layer count; Gemma-4-12B N=48 -> K in {36,24,16}):
- Generate draft_K.gguf.
- `llama-server -m target.gguf -md draft_K.gguf --spec-type draft-simple --spec-draft-n-max 4
   --spec-draft-p-min 0 -ngl 0` ; run sd.py-style bench (reuse `selfdraft/sd.py bench` by
   pointing `--methods draft-simple` - add a `--draft-model` passthrough to sd.py if needed).
- Record: tok/s, speedup vs vanilla, draft acceptance %, output similarity.

**GO/NO-GO:**
- GO if any K gives **net speedup >= 1.5x** with similarity ~vanilla (lossless). Proceed to Stage 2/3.
- MARGINAL (1.2-1.5x): proceed to Stage 2 (salience) to push higher.
- NO-GO (<1.2x at all K, i.e. acceptance collapses): uniform early-exit fails on stock
  Gemma 4 (expected possibility). Try ONE salience pass (Stage 2 quick test); if still <1.2x,
  STOP, write up the negative finding, keep the MTP 2x as the submission. Do NOT sink days.

Deliverable of Stage 1: a table (K, layers, tok/s, speedup, accept%) + the generator script.

---

## STAGE 2 - Salience-selected self-draft (the novelty). ~1 day. Only if Stage 1 GO/MARGINAL.

### 2.1 Layer-importance calibration
Add `selfdraft/calibrate_layers.py`:
- Run the TARGET on a small calibration set (20-50 reasoning/code prompts, ~50 tok each).
- Estimate each layer's importance. Two cheap options (implement the simpler first):
  (a) **Angular/residual contribution**: cosine distance between hidden state before vs
      after each layer (small change = prunable). Needs hidden states - get via
      `llama-server` with `--embeddings`? Simpler: a tiny C++ debug dumper OR use the
      `llama-perplexity`/`llama-eval-callback` (`examples/eval-callback`) which already
      prints per-layer tensors. Reuse `eval-callback` to dump per-layer hidden norms.
  (b) **Drop-one acceptance**: for each layer, build a draft that skips just that layer,
      measure acceptance; rank layers by how little skipping them hurts. Slower but directly
      optimizes the target metric. Use if (a) is fiddly.
- Output: ranked layer list -> the K most important layer indices to KEEP.

### 2.2 Generate salience draft
Extend `make_self_draft.py --layers i,j,k,...`: keep those layers, **remap blk indices to
0..K-1**, and remap/rebuild any per-layer metadata (attention pattern!) to match the new
order. This is the trickiest correctness point - verify the model loads and acceptance
actually improves vs uniform-first-K at the same K.

### 2.3 Re-measure vs Stage 1
Compare SSD vs EESD at equal K. Expect SSD acceptance > EESD. Pick the (K, layer-set) with
best speedup. Re-run the GO/NO-GO bar (>=1.5x lossless).

---

## STAGE 3 - DX integration: one flag, auto-derive. ~half a day. After a GO.

- Extend `selfdraft/sd.py`: add `derive` subcommand (wraps make_self_draft + calibrate) and
  make `--self-draft` auto-generate-and-cache the draft from the target if none exists:
  `sd.py run model.gguf --self-draft-derive --keep 16` -> builds `model.selfdraft-k16.gguf`
  once, then serves with `draft-simple`. Zero download: the draft is derived locally from
  the model the user already has.
- Add `autotune` over K (reuse the autotuner; sweep generated drafts).
- Update README/DEVPOST: "self-draft works on ANY model - derives a salience-pruned draft
  from the model's own layers, one command, zero download." Add the EESD/SSD results table.
  Keep MTP as the zero-config path for models that ship heads; EESD/SSD as the universal path.
- Add the free Arm CI job variant that derives + benchmarks the self-draft on a non-MTP
  model (proves "any model" on Arm64).

---

## STAGE 4 - OPTIONAL C++: shared-KV early-exit (max efficiency). Only if time + a strong GO.

The GGUF-surgery draft (Stages 1-3) loads a second (smaller) model and runs it independently
- it does NOT reuse the first-K layers' KV between draft and target. True LayerSkip-style
self-speculation reuses them (the draft's first-K compute IS the target's prefix), roughly
doubling the efficiency. This needs real C++:
- Add a context mode `n_exit_layer` (cparams) in `include/llama.h` + `src/llama-context.*`.
- In `src/models/gemma4.cpp` (and any arch you target): when `n_exit_layer>0`, after layer
  `n_exit_layer-1` apply `output_norm` + `output` and return early (skip remaining layers).
- New `common/speculative.cpp` impl `draft-self-exit`: ONE model, ONE context; draft pass
  runs to the exit layer, target pass continues the SAME forward from the exit layer reusing
  the shared activations/KV. This is the hard part (graph + KV bookkeeping). Reference the
  existing `draft-mtp` impl and the `target_layer_ids` extraction infra (speculative.cpp:423).
- High risk/complexity; only attempt if Stages 1-3 show a clear, defensible win and you have
  days to spare. Otherwise the GGUF-surgery version is a complete, shippable contribution.

---

## Risks & honest expectations (put these in the writeup)
- Stock Gemma 4 is not early-exit-trained; uniform truncation may collapse acceptance. SSD
  (salience) is the mitigation but may still underperform MTP's 2x. A clean negative result
  ("training-free universal self-draft is hard; here is exactly why, measured") is itself a
  valid, judge-respected Impact/learning artifact - write it up either way.
- Gemma sliding-window attention metadata is the main correctness trap on layer reordering.
- Lossless guarantee comes from draft-simple verification; do NOT claim bit-identical
  (FP-tie flips), claim distributionally lossless + measured similarity.
- Keep llama.cpp build green; all C++ (Stage 4 only) in small commits.

## Verification commands (run at every stage)
```
# build draft, sanity-load
python3 selfdraft/make_self_draft.py --in models/gemma-4-12b-it-Q4_0.gguf --out /tmp/d.gguf --keep 24
./llama.cpp/build/bin/llama-server -m /tmp/d.gguf -ngl 0 -c 1024   # must load + emit
# acceptance/speedup vs vanilla (lossless check via similarity)
python3 selfdraft/sd.py bench models/gemma-4-12b-it-Q4_0.gguf --methods draft-simple \
  --draft-model /tmp/d.gguf --ngl 0 --n-max 4 --workload code --n-predict 200
# Arm64 truth: run the same on the free GitHub ubuntu-24.04-arm CI runner.
```

## Deliverables
- `selfdraft/make_self_draft.py`, `selfdraft/calibrate_layers.py`
- `sd.py` `derive` subcommand + `--draft-model` passthrough + K autotune
- Results table (EESD vs SSD vs MTP, Arm64), README/DEVPOST update, honest findings
- (optional) Stage-4 C++ `draft-self-exit` patch
- If NO-GO: a written negative-result finding; revert to MTP 2x submission.

## First action for the executor
1. Read `selfdraft/sd.py`, `common/speculative.cpp` (draft-simple + draft-mtp impls),
   `src/models/gemma4.cpp`, `src/llama-model-loader.cpp` (tensor load + kv overrides).
2. Confirm Gemma 4's output head tensor names + whether output is tied to token_embd.
3. Implement `make_self_draft.py` (Stage 1.1), run 1.2/1.3, report the GATE table. STOP there
   for a human decision before Stage 2+.
