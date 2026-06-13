# Optional: native `--self-draft` flag for llama.cpp

`selfdraft/sd.py` works with **stock** llama.cpp (it passes `-md <mtp> --spec-type
draft-mtp` explicitly). This patch is optional: it adds a native one-flag
`--self-draft` to llama.cpp itself, so the CLI/server gain zero-download MTP
self-speculation directly.

## What `llama.cpp-self-draft.patch` does
`common/arg.cpp`, +61 lines:
1. Adds a `--self-draft` flag that enables the `draft-mtp` speculative type.
2. Adds `common_find_local_mtp_sibling()` — when a **local** `-m model.gguf` is
   used with draft-mtp and no explicit `-md`, it scans the model's directory for
   an `mtp-*.gguf` / `*-MTP.gguf` sibling and wires it as the draft. Stock
   llama.cpp only auto-resolves the MTP head for `-hf` downloads, not local files.

Result: `llama-server -m models/gemma-4-12b-it-Q4_K_M.gguf --self-draft` just works.

## Apply + build
```bash
cd llama.cpp
git apply ../patches/llama.cpp-self-draft.patch
cmake --build build -j --target llama-server llama-cli
./build/bin/llama-server -m ../models/gemma-4-12b-it-Q4_K_M.gguf --self-draft --ngl 0
# log: "self-draft: using local MTP head 'mtp-gemma-4-12b-it.gguf'"
```

## Upstreaming note (read before submitting)
llama.cpp's `AGENTS.md`/`CONTRIBUTING.md` state the project does **not** accept
fully/predominantly AI-generated PRs, and that automated agents must not open PRs.
This patch is provided as an **assistive draft**: if you want it upstream, you must
fully understand it, be able to explain/defend it to maintainers without AI, write
your own PR description and commit message, and submit it yourself. The patch is
deliberately small, reuses existing `draft-mtp` infrastructure, and adds no new
subsystem. Until then it lives in your own fork (private forks are exempt from the
policy).
