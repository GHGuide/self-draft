# Rules compliance + competitiveness checklist

Source: Arm Create: AI Optimization Challenge official rules (reviewed 2026-06-13).
Submission period: Jun 10 - Aug 14, 2026 (4:00pm PT). Judging Aug 17 - Sep 4.

## Hard requirements
- [x] Open-source license MIT/Apache-2.0 present from first commit (Apache-2.0).
- [ ] Repo PUBLIC, license visible in GitHub About section (push needed; GitHub auto-detects LICENSE).
- [x] Repo contains all source + instructions to build/run/validate on Arm64 (README: Graviton steps, sd.py, patches/).
- [x] Write-up: Project Overview / Functionality / Setup (docs/DEVPOST.md).
- [x] New / significantly updated within submission period (repo created 2026-06-12/13).
- [x] Original work; OSS (llama.cpp MIT, Gemma 4 Apache) used and *enhanced* (sd.py + arg.cpp patch).
- [ ] Demo video <=3 min, shows it running on target device, public (YouTube/Vimeo/Youku), no copyrighted music (plan: docs/DEMO_PLAN.md).
- [x] English.
- [ ] Register on Devpost ("Join Hackathon") + join Arm Developer Program (developer.arm.com).
- [ ] Select track: Cloud AI category; optimization-output (source code) submission.

## Judging criteria (100 pts) - self-assessment
- Technological Implementation (40): native llama.cpp `--self-draft` patch, n-max autotuner, rigorous go/no-go, Metal-bug characterization. Foreground the engineering so it's not read as a thin wrapper over existing MTP.
- UX/DX (15): one flag, zero download, clean README + Graviton steps. Strongest axis.
- Potential Impact (20): reusable on any MTP-equipped model; portable CPU path. (Upstream PR blocked by llama.cpp no-AI-PR policy.)
- WOW (25): "your model already ships a draft" + side-by-side token race. Weakest: ~1.5x modest, byte-identical not bit-exact - pitch on free/zero-download/one-flag, not raw multiplier.

## TOP RISK: no Arm-cloud (Graviton) numbers
Cloud AI track judges expect Arm64 *cloud* results. We have Apple M4 (Arm64) only.
Rules cite Arm Performix for exact Arm benchmarks. ACTION: run `sd.py autotune --ngl 0`
on a Graviton instance (c7g/t4g); optionally via Arm Performix. Highest-leverage move
for Tech-Impl + Impact.

## Remaining actions (owner: user)
1. Run on AWS Graviton, capture numbers -> add to README results table.
2. Publish repo public on GitHub (Apache LICENSE -> auto-shown in About).
3. Record <=3 min demo (docs/DEMO_PLAN.md), upload public.
4. Register Devpost + Arm Developer Program; fill submission form (paste docs/DEVPOST.md).
5. (optional) File llama.cpp ISSUE for the Metal/UMA dual-context speculative slowdown.
