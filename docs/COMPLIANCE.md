# Compliance audit — Arm Create / Cloud AI (Arm Agent-Memory)

Checked against the official rules (reviewed 2026-06-14). Submission window Jun 10 – Aug 14, 2026.

## Hard submission requirements
- [x] **Create/migrate/optimize AI on Arm** — optimizes multi-agent LLM inference on Arm64 (TTFT).
- [x] **Track selected: Cloud AI** (optimization-output; includes source code → satisfies the Track-1/2 source rule).
- [x] **Public repo, all source + instructions, functional** — https://github.com/GHGuide/self-draft; every README-referenced file is tracked; CI proves the exact build/run flow is green.
- [x] **OSS license detectable in About** — Apache-2.0, auto-detected by GitHub (verified via API: `license.spdx_id = Apache-2.0`).
- [x] **Write-up: Overview / Functionality / Setup** — `docs/SUBMISSION_FORM.md` (paste-ready) + `docs/DEVPOST.md`.
- [x] **Runs consistently on Arm64** — reproduced green on the free GitHub `ubuntu-24.04-arm` runner (build + all benches).
- [x] **New / significantly updated in submission period** — repo created 2026-06-12..14.
- [x] **Original work + OSS used lawfully + enhanced** — builds on llama.cpp (MIT) + Gemma 4 (Apache-2.0); adds an automatic manager + benchmarks (enhances the underlying OSS). Implements a *published method* (arXiv:2603.04428) independently (no MLX code copied); paper cited.
- [x] **English.**
- [ ] **Register Devpost ("Join Hackathon")** — USER action.
- [x] **Arm Developer Program** — USER already a Member (confirmed).
- [ ] **Submit the Devpost form** (select Cloud AI, paste write-up, repo URL) — USER action.
- [ ] **Demo video (optional)** — USER; if made: ≤3 min, show it running on Arm64, public YouTube/Vimeo/Youku, no copyrighted music/trademarks.

## Models not committed (intentional, compliant)
Multi-GB GGUFs are not in the repo (gitignored); the README gives exact `hf download` commands +
the free CI downloads them automatically. "Instructions required for the project to be functional"
are present — committing 7 GB of weights is impractical and against GitHub norms. Judges run free.

## Judging-criteria fit (Cloud AI)
- **Tech 40** — automatic persistent-KV manager + slot lifecycle + bit-exact verification + bundled self-draft; ports a Feb-2026 paper to Arm64; rigorous measured findings (incl. negatives).
- **DX 15** — one manager, free-CI-reproducible, copy-paste docs, drop-in over llama-server.
- **Impact 20** — multi-agent serving is the 2026 workload; model-agnostic; first llama.cpp/Arm64 port; reusable free-CI benchmark.
- **WOW 25** — 500× lower TTFT on a free Arm runner, bit-exact, visceral cold-vs-warm demo.

## RISK FLAGS — read these (honest)
1. **Verification of your role (rule §8B).** Prizes require verifying the winner's IDENTITY,
   QUALIFICATIONS and ROLE in creating the submission. This was built with heavy AI assistance.
   Rules permit technical assistance *if you own the work and can stand behind it*. You MUST be
   able to explain, without help: how persistent-KV restore avoids re-prefill; why `--swa-full`
   is required for Gemma; what the 500× / bit-exact numbers mean and the RAM-constrained caveat;
   how the slot manager works. Treat the repo as yours and learn it cold before submitting.
2. **Claims must match reality (rule: "function as depicted").** All README/Devpost numbers are
   measured and CI-reproduced; keep it that way. The video must show REAL behavior, not staged.
3. **The 500× is the RAM-constrained / cross-restart regime** (the paper's premise; `--cache-ram 0`).
   README states this honestly. Do not imply it's universal — present it as the edge/multi-agent
   regime where it genuinely applies.
4. **Repo name** is `self-draft` (legacy) while the project is "Arm Agent-Memory" — cosmetic, not a
   rule issue. Optional: rename the GitHub repo to `arm-agent-memory` and update the URL in the
   README/SUBMISSION_FORM before submitting (your call; updates 3 doc references).

## Remaining to submit (USER, all free)
1. Devpost → Join Hackathon. 2. Record ≤3-min Arm64 demo (optional but high-WOW). 3. Devpost →
Enter Submission → Cloud AI → paste `docs/SUBMISSION_FORM.md` → repo URL → submit before Aug 14, 4pm PT.
