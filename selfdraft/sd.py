#!/usr/bin/env python3
"""
self-draft — zero-download self-speculative decoding for llama.cpp.

Wraps llama.cpp's `--spec-type draft-mtp` path: if a model ships Multi-Token
Prediction (MTP) heads (e.g. Gemma 4's `mtp-*.gguf` sibling), self-draft wires
them up automatically — no separate draft model to download, one command.

Subcommands:
  run       launch a llama-server with self-draft enabled (interactive/serving)
  bench     vanilla vs self-draft: tok/s, speedup, draft acceptance, equivalence
  autotune  sweep --spec-draft-n-max, pick the value with best tok/s

Equivalence note: speculative decoding is *distributionally* lossless (the target
verifies every drafted token), but batched verification is not bit-identical to
sequential decoding due to floating-point non-associativity — expect rare
argmax-tie flips on long generations. `bench` quantifies this honestly.
"""
import argparse, json, os, signal, subprocess, sys, time, hashlib, difflib, glob, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

def find_server():
    for p in [os.path.join(ROOT, "llama.cpp/build/bin/llama-server"),
              os.path.join(ROOT, "llama.cpp/build/llama-server")]:
        if os.path.exists(p):
            return p
    p = subprocess.run(["bash","-lc","command -v llama-server"], capture_output=True, text=True).stdout.strip()
    if p:
        return p
    sys.exit("ERROR: llama-server not found. Build llama.cpp first (see README).")

def pick_draft(args):
    """Draft model path: explicit --draft-model wins (e.g. a derived self-draft gguf),
    else auto-resolve an mtp-*.gguf sibling."""
    if getattr(args, "draft_model", None):
        return args.draft_model
    return resolve_mtp(args.model, args.mtp)

def perf_kwargs(args):
    """Arm/CPU perf flags, applied to BOTH vanilla and self-draft servers for a fair ratio."""
    return dict(fa=args.fa, ctk=args.ctk, ctv=args.ctv,
                ctkd=getattr(args, "ctkd", None), ctvd=getattr(args, "ctvd", None),
                mlock=args.mlock, poll=args.poll)

def resolve_mtp(model_path, explicit=None):
    """Find the MTP sibling for a target gguf. unsloth convention: mtp-<stem>.gguf"""
    if explicit:
        return explicit
    d = os.path.dirname(os.path.abspath(model_path))
    cands = sorted(glob.glob(os.path.join(d, "mtp-*.gguf")) + glob.glob(os.path.join(d, "*-MTP.gguf")))
    if not cands:
        sys.exit(f"ERROR: no MTP sibling (mtp-*.gguf) found next to {model_path}.\n"
                 f"       Download it, e.g.: hf download <repo> --include 'mtp-*.gguf' --local-dir {d}")
    if len(cands) > 1:
        print(f"[self-draft] multiple MTP candidates, using {os.path.basename(cands[0])}", file=sys.stderr)
    return cands[0]

class Server:
    def __init__(self, binary, model, port=8099, ngl=99, threads=None, ctx=4096,
                 mtp=None, n_max=None, p_min=0.0, backend_sampling=True, spec_type=None,
                 fa=None, ctk=None, ctv=None, ctkd=None, ctvd=None, mlock=False, poll=None):
        self.binary, self.model, self.port = binary, model, port
        self.ngl, self.threads, self.ctx = ngl, threads, ctx
        self.mtp, self.n_max, self.p_min = mtp, n_max, p_min
        self.backend_sampling = backend_sampling
        # speculation method(s): None = vanilla; "draft-mtp" (default when mtp set);
        # comma list e.g. "draft-mtp,ngram-mod" runs a cascade.
        self.spec_type = spec_type
        # Arm/CPU perf knobs (memory-bound wins). fa: 'on'|'off'|'auto'. ctk/ctv: KV cache
        # quant type (q8_0...) for TARGET (changes the lossless baseline -> hash both runs
        # at the SAME setting). ctkd/ctvd: DRAFT KV quant (correctness-safe, only affects
        # acceptance). -ctv requires -fa on. mlock: pin weights. poll: spin-wait.
        self.fa, self.ctk, self.ctv = fa, ctk, ctv
        self.ctkd, self.ctvd, self.mlock, self.poll = ctkd, ctvd, mlock, poll
        self.proc = None

    def cmd(self):
        c = [self.binary, "-m", self.model, "-ngl", str(self.ngl), "-c", str(self.ctx),
             "--port", str(self.port), "--host", "127.0.0.1"]
        if self.threads:
            c += ["-t", str(self.threads)]
        if self.fa:    c += ["-fa", self.fa]
        if self.ctk:   c += ["-ctk", self.ctk]
        if self.ctv:   c += ["-ctv", self.ctv]   # requires -fa on
        if self.mlock: c += ["--mlock", "--no-mmap"]
        if self.poll is not None: c += ["--poll", str(self.poll)]
        spec = self.spec_type or ("draft-mtp" if self.mtp else None)
        if spec:
            c += ["--spec-type", spec]
            # draft-model methods (draft-*: mtp / eagle3 / simple) need the draft model +
            # draft flags; ngram-* methods draft from context and need neither.
            if self.mtp and "draft-" in spec:
                c += ["-md", self.mtp, "--spec-draft-ngl", str(self.ngl),
                      "--spec-draft-p-min", str(self.p_min)]
                if self.n_max is not None:
                    c += ["--spec-draft-n-max", str(self.n_max)]
                if not self.backend_sampling:
                    c += ["--no-spec-draft-backend-sampling"]
                if self.ctkd: c += ["-ctkd", self.ctkd]   # draft KV quant: correctness-safe
                if self.ctvd: c += ["-ctvd", self.ctvd]
        return c

    def __enter__(self):
        os.makedirs(os.path.join(ROOT, "bench"), exist_ok=True)
        self.log = open(os.path.join(ROOT, "bench", f"server_{self.port}.log"), "w")
        self.proc = subprocess.Popen(self.cmd(), stdout=self.log, stderr=subprocess.STDOUT,
                                     preexec_fn=os.setsid)
        # wait for /health
        url = f"http://127.0.0.1:{self.port}/health"
        for _ in range(180):
            try:
                if json.load(urllib.request.urlopen(url, timeout=2)).get("status") == "ok":
                    return self
            except Exception:
                pass
            if self.proc.poll() is not None:
                sys.exit(f"ERROR: server exited early. See bench/server_{self.port}.log")
            time.sleep(1)
        self.__exit__(None, None, None)
        sys.exit("ERROR: server did not become ready in 180s")

    def __exit__(self, *a):
        if self.proc:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except Exception:
                pass
            self.proc.wait(timeout=15)
        if getattr(self, "log", None):
            self.log.close()

    def complete(self, prompt, n_predict, temperature=0.0, top_k=1, n_max=None, cache_prompt=False):
        body = {"prompt": prompt, "n_predict": n_predict, "temperature": temperature,
                "top_k": top_k, "cache_prompt": cache_prompt}
        if n_max is not None:
            body["speculative.n_max"] = n_max   # per-request draft length (needs the server patch)
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/completion",
                                     json.dumps(body).encode(), {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=900))

# --- prompts (a reasoning/code prompt is the speculative-friendly default) ---
PROMPTS = {
    "code": "<start_of_turn>user\nWrite a Python function `fib(n)` that returns the nth Fibonacci "
            "number using memoization. Then explain step by step how it works and give the time "
            "complexity.<end_of_turn>\n<start_of_turn>model\n",
    "prose": "<start_of_turn>user\nExplain why the sky is blue.<end_of_turn>\n<start_of_turn>model\n",
    # mixed: alternates high-acceptance (code/structured) and low-acceptance (free prose)
    # regions -> a static n-max is wrong for half the run; this is where adaptive wins.
    "mixed": "<start_of_turn>user\nFirst write a Python function fib(n) using memoization. "
             "Then write a short, original, free-flowing motivational paragraph about persistence "
             "(no code). Then give a JSON object {\"name\":..., \"complexity\":...} for the function."
             "<end_of_turn>\n<start_of_turn>model\n",
}

def gen_metrics(r):
    t = r["timings"]
    m = {"tok_s": t["predicted_per_second"], "n": t["predicted_n"],
         "ttft_ms": t.get("prompt_ms"),                          # time to first token
         "latency_ms": (t.get("prompt_ms", 0) + t.get("predicted_ms", 0))}  # end-to-end
    if t.get("draft_n"):
        m["accept_pct"] = 100.0 * t["draft_n_accepted"] / t["draft_n"]
        m["draft_n"] = t["draft_n"]
    return m

def cost_per_mtok(tok_s, price_per_hr):
    # $ per 1M output tokens = ($/hr) / (tokens/hr) * 1e6
    if not price_per_hr or not tok_s:
        return None
    return price_per_hr / (tok_s * 3600.0) * 1e6

def equivalence(a, b):
    """Honest equivalence report between two generations (text)."""
    ha, hb = (hashlib.sha256(x.encode()).hexdigest() for x in (a, b))
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]:
        i += 1
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    return {"identical": ha == hb, "sha_a": ha[:12], "sha_b": hb[:12],
            "common_prefix_chars": i, "len_a": len(a), "len_b": len(b),
            "similarity": round(ratio, 4)}

def do_bench(args):
    binary = find_server()
    mtp = pick_draft(args)
    prompt = PROMPTS.get(args.workload, args.workload)
    print(f"[self-draft] target={os.path.basename(args.model)} mtp={os.path.basename(mtp)} "
          f"ngl={args.ngl} n-max={args.n_max} workload={args.workload} n_predict={args.n_predict}")

    with Server(binary, args.model, port=args.port, ngl=args.ngl, threads=args.threads,
                ctx=args.ctx, **perf_kwargs(args)) as s:
        v = s.complete(prompt, args.n_predict)
    vm = gen_metrics(v)

    with Server(binary, args.model, port=args.port, ngl=args.ngl, threads=args.threads,
                ctx=args.ctx, mtp=mtp, n_max=args.n_max, p_min=args.p_min, spec_type=args.methods, **perf_kwargs(args)) as s:
        d = s.complete(prompt, args.n_predict)
    dm = gen_metrics(d)

    eq = equivalence(v["content"], d["content"])
    speedup = dm["tok_s"] / vm["tok_s"]
    print("\n================ self-draft bench (Arm64 cloud) ================")
    print(f"{'':12} {'tok/s':>8} {'TTFT ms':>9} {'latency ms':>11}")
    print(f"vanilla    : {vm['tok_s']:8.2f} {vm['ttft_ms']:9.1f} {vm['latency_ms']:11.1f}")
    print(f"self-draft : {dm['tok_s']:8.2f} {dm['ttft_ms']:9.1f} {dm['latency_ms']:11.1f}   "
          f"(accept {dm.get('accept_pct',0):.1f}%, n-max={args.n_max})")
    print(f"DECODE SPEEDUP : {speedup:.2f}x   |   LATENCY: {vm['latency_ms']/dm['latency_ms']:.2f}x lower")
    if args.price:
        cv, cd = cost_per_mtok(vm["tok_s"], args.price), cost_per_mtok(dm["tok_s"], args.price)
        print(f"COST @ ${args.price}/hr : vanilla ${cv:.3f}/1M tok  ->  self-draft ${cd:.3f}/1M tok  "
              f"({(1-cd/cv)*100:.0f}% cheaper)")
    print("--- output equivalence (distributional, not bit-exact) ---")
    print(f"exact byte-identical : {eq['identical']}  | similarity {eq['similarity']*100:.2f}% "
          f"(common prefix {eq['common_prefix_chars']}/{eq['len_a']} chars)")
    print("===============================================================")
    if args.json:
        json.dump({"vanilla": vm, "self_draft": dm, "speedup": speedup, "equivalence": eq,
                   "price_per_hr": args.price,
                   "cost_per_mtok": {"vanilla": cost_per_mtok(vm["tok_s"], args.price),
                                     "self_draft": cost_per_mtok(dm["tok_s"], args.price)}},
                  open(args.json, "w"), indent=2)
        print(f"[self-draft] wrote {args.json}")

def do_autotune(args):
    binary = find_server()
    mtp = pick_draft(args)
    prompt = PROMPTS.get(args.workload, args.workload)
    grid = [int(x) for x in args.grid.split(",")]
    print(f"[self-draft] autotune n-max over {grid} (workload={args.workload})")
    # vanilla baseline
    with Server(binary, args.model, port=args.port, ngl=args.ngl, threads=args.threads, ctx=args.ctx, **perf_kwargs(args)) as s:
        vm = gen_metrics(s.complete(prompt, args.n_predict))
    print(f"vanilla: {vm['tok_s']:.2f} tok/s")
    rows, best = [], None
    for nm in grid:
        with Server(binary, args.model, port=args.port, ngl=args.ngl, threads=args.threads,
                    ctx=args.ctx, mtp=mtp, n_max=nm, p_min=args.p_min, spec_type=args.methods, **perf_kwargs(args)) as s:
            m = gen_metrics(s.complete(prompt, args.n_predict))
        sp = m["tok_s"] / vm["tok_s"]
        rows.append((nm, m["tok_s"], sp, m.get("accept_pct", 0)))
        print(f"  n-max={nm:2d}  {m['tok_s']:6.2f} tok/s  {sp:.2f}x  accept={m.get('accept_pct',0):.1f}%")
        if best is None or sp > best[2]:
            best = (nm, m["tok_s"], sp, m.get("accept_pct", 0))
    print(f"\nBEST n-max={best[0]} -> {best[2]:.2f}x ({best[1]:.2f} tok/s, accept {best[3]:.1f}%)")
    if args.json:
        json.dump({"vanilla_tok_s": vm["tok_s"],
                   "grid": [{"n_max": r[0], "tok_s": r[1], "speedup": r[2], "accept_pct": r[3]} for r in rows],
                   "best_n_max": best[0]}, open(args.json, "w"), indent=2)

def do_run(args):
    binary = find_server()
    mtp = pick_draft(args)
    cmd = Server(binary, args.model, port=args.port, ngl=args.ngl, threads=args.threads,
                 ctx=args.ctx, mtp=mtp, n_max=args.n_max, p_min=args.p_min, spec_type=args.methods, **perf_kwargs(args)).cmd()
    print("[self-draft] launching:\n  " + " ".join(cmd))
    os.execv(binary, cmd)

def run_adaptive(srv, prompt, total_n, n_cap, chunk=24, beta=0.5):
    """Online adaptive draft-length controller (GammaTune-style EMA). Issues chunked
    completions reusing KV (cache_prompt), reads per-chunk acceptance, and steers the
    per-request n_max: high acceptance -> draft longer, low acceptance -> draft shorter.
    Lossless: only changes HOW MANY tokens are proposed; the target verifies every one."""
    text, n_max, ema = "", 3, None
    tok, ms, traj = 0, 0.0, []
    while tok < total_n:
        r = srv.complete(prompt + text, min(chunk, total_n - tok), n_max=n_max, cache_prompt=True)
        t = r["timings"]; text += r["content"]
        tok += t["predicted_n"]; ms += t["predicted_ms"]
        a = (t["draft_n_accepted"] / t["draft_n"]) if t.get("draft_n") else 0.0
        ema = a if ema is None else (1 - beta) * ema + beta * a
        traj.append((n_max, round(a, 2)))
        # proportional controller: optimal draft length grows with acceptance
        n_max = max(1, min(n_cap, round(1 + 0.7 * ema / (1 - min(ema, 0.92)))))
    return {"tok_s": tok / (ms / 1000.0) if ms else 0, "tokens": tok, "n_max_traj": traj}

def do_adaptive(args):
    binary = find_server()
    mtp = pick_draft(args)
    prompt = PROMPTS.get(args.workload, args.workload)
    n_cap = args.n_cap
    print(f"[self-draft] adaptive n-max vs static-best (n_cap={n_cap}, workload={args.workload})")
    # one server launched at n_cap; per-request n_max selects the effective draft length
    with Server(binary, args.model, port=args.port, ngl=args.ngl, threads=args.threads,
                ctx=args.ctx, mtp=mtp, n_max=n_cap, p_min=args.p_min, spec_type=args.methods,
                **perf_kwargs(args)) as s:
        # static sweep on the SAME server (per-request n_max)
        static = {}
        for k in [int(x) for x in args.grid.split(",")]:
            r = s.complete(prompt, args.n_predict, n_max=k)
            static[k] = r["timings"]["predicted_per_second"]
            print(f"  static n-max={k}: {static[k]:.2f} tok/s")
        best_k = max(static, key=static.get)
        # adaptive
        ad = run_adaptive(s, prompt, args.n_predict, n_cap, chunk=args.chunk)
    sp = ad["tok_s"] / static[best_k]
    print(f"\nbest static : n-max={best_k}  {static[best_k]:.2f} tok/s")
    print(f"ADAPTIVE    : {ad['tok_s']:.2f} tok/s  ({sp:.2f}x vs best static)")
    print(f"  n-max trajectory (n,accept): {ad['n_max_traj']}")
    if args.json:
        json.dump({"static": static, "best_static_k": best_k, "best_static_tok_s": static[best_k],
                   "adaptive_tok_s": ad["tok_s"], "adaptive_vs_static": sp,
                   "n_max_traj": ad["n_max_traj"]}, open(args.json, "w"), indent=2)

def do_agent(args):
    from agent_demo import run_agent
    binary = find_server()
    mtp = pick_draft(args)
    print(f"[self-draft] agent end-to-end latency: vanilla vs self-draft (n-max={args.n_max})")
    with Server(binary, args.model, port=args.port, ngl=args.ngl, threads=args.threads, ctx=args.ctx, **perf_kwargs(args)):
        v = run_agent(args.port, verbose=args.verbose)
    with Server(binary, args.model, port=args.port, ngl=args.ngl, threads=args.threads, ctx=args.ctx,
                mtp=mtp, n_max=args.n_max, p_min=args.p_min, spec_type=args.methods, **perf_kwargs(args)):
        d = run_agent(args.port, verbose=args.verbose)
    print("\n============ agent loop (ReAct + calculator tool) ============")
    print(f"vanilla    : {v['wall_ms']/1000:6.2f}s  steps={v['steps']} tokens={v['tokens']} answer={v['answer']} correct={v['correct']}")
    print(f"self-draft : {d['wall_ms']/1000:6.2f}s  steps={d['steps']} tokens={d['tokens']} answer={d['answer']} correct={d['correct']}")
    if d["wall_ms"]:
        print(f"END-TO-END AGENT SPEEDUP : {v['wall_ms']/d['wall_ms']:.2f}x faster")
    print("==============================================================")
    if args.json:
        json.dump({"vanilla": v, "self_draft": d, "speedup": v["wall_ms"]/d["wall_ms"]},
                  open(args.json, "w"), indent=2)

def main():
    ap = argparse.ArgumentParser(prog="self-draft", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    def common(p):
        p.add_argument("model", help="path to target .gguf")
        p.add_argument("--mtp", help="explicit MTP draft .gguf (default: auto-resolve mtp-*.gguf sibling)")
        p.add_argument("--draft-model", help="explicit draft .gguf (e.g. a derived self-draft); overrides MTP auto-resolve")
        p.add_argument("--ngl", type=int, default=99, help="GPU layers (0 = CPU only; CPU avoids Metal dual-context bug)")
        p.add_argument("--threads", type=int, default=None)
        p.add_argument("--ctx", type=int, default=4096)
        p.add_argument("--port", type=int, default=8099)
        p.add_argument("--p-min", type=float, default=0.0)
        p.add_argument("--methods", default="draft-mtp",
                       help="speculation method(s): draft-mtp | ngram-mod | 'draft-mtp,ngram-mod' (cascade, usually fastest)")
        # Arm/CPU perf knobs (applied to BOTH vanilla and self-draft for a fair ratio)
        p.add_argument("--fa", choices=["on", "off", "auto"], help="flash attention (set 'on' for CPU; required for -ctv)")
        p.add_argument("--ctk", help="target KV-cache K quant (e.g. q8_0). Changes lossless baseline; hash both runs at same setting")
        p.add_argument("--ctv", help="target KV-cache V quant (e.g. q8_0). Requires --fa on")
        p.add_argument("--ctkd", help="DRAFT KV-cache K quant (correctness-safe; only affects acceptance)")
        p.add_argument("--ctvd", help="DRAFT KV-cache V quant (correctness-safe)")
        p.add_argument("--mlock", action="store_true", help="pin weights (--mlock --no-mmap): lower tail latency")
        p.add_argument("--poll", type=int, help="busy-poll wait (e.g. 50): cut per-token wakeup latency")
        p.add_argument("--workload", default="code", help="'code', 'prose', or a literal prompt string")
        p.add_argument("--n-predict", type=int, default=200)
        p.add_argument("--price", type=float, default=0.0, help="instance price $/hr (e.g. c7g.xlarge ~0.145) -> reports $/1M tokens")
        p.add_argument("--json", help="write metrics JSON here")
    b = sub.add_parser("bench"); common(b); b.add_argument("--n-max", type=int, default=3); b.set_defaults(fn=do_bench)
    a = sub.add_parser("autotune"); common(a); a.add_argument("--grid", default="1,2,3,4,6,8"); a.set_defaults(fn=do_autotune)
    r = sub.add_parser("run"); common(r); r.add_argument("--n-max", type=int, default=3); r.set_defaults(fn=do_run)
    g = sub.add_parser("agent"); common(g); g.add_argument("--n-max", type=int, default=3)
    g.add_argument("-v", "--verbose", action="store_true"); g.set_defaults(fn=do_agent)
    ad = sub.add_parser("adaptive"); common(ad)
    ad.add_argument("--n-cap", type=int, default=8, help="max draft length the controller may use")
    ad.add_argument("--chunk", type=int, default=24, help="tokens per control step")
    ad.add_argument("--grid", default="1,2,3,4,6", help="static n-max values to compare against")
    ad.set_defaults(fn=do_adaptive)
    args = ap.parse_args()
    args.fn(args)

if __name__ == "__main__":
    main()
