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
                 mtp=None, n_max=None, p_min=0.0, backend_sampling=True):
        self.binary, self.model, self.port = binary, model, port
        self.ngl, self.threads, self.ctx = ngl, threads, ctx
        self.mtp, self.n_max, self.p_min = mtp, n_max, p_min
        self.backend_sampling = backend_sampling
        self.proc = None

    def cmd(self):
        c = [self.binary, "-m", self.model, "-ngl", str(self.ngl), "-c", str(self.ctx),
             "--port", str(self.port), "--host", "127.0.0.1"]
        if self.threads:
            c += ["-t", str(self.threads)]
        if self.mtp:
            c += ["-md", self.mtp, "--spec-type", "draft-mtp",
                  "--spec-draft-ngl", str(self.ngl), "--spec-draft-p-min", str(self.p_min)]
            if self.n_max is not None:
                c += ["--spec-draft-n-max", str(self.n_max)]
            if not self.backend_sampling:
                c += ["--no-spec-draft-backend-sampling"]
        return c

    def __enter__(self):
        self.log = open(os.path.join(ROOT, "bench", f"server_{self.port}.log"), "w")
        os.makedirs(os.path.join(ROOT, "bench"), exist_ok=True)
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

    def complete(self, prompt, n_predict, temperature=0.0, top_k=1):
        body = json.dumps({"prompt": prompt, "n_predict": n_predict,
                           "temperature": temperature, "top_k": top_k,
                           "cache_prompt": False}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/completion", body,
                                     {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=900))

# --- prompts (a reasoning/code prompt is the speculative-friendly default) ---
PROMPTS = {
    "code": "<start_of_turn>user\nWrite a Python function `fib(n)` that returns the nth Fibonacci "
            "number using memoization. Then explain step by step how it works and give the time "
            "complexity.<end_of_turn>\n<start_of_turn>model\n",
    "prose": "<start_of_turn>user\nExplain why the sky is blue.<end_of_turn>\n<start_of_turn>model\n",
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
    mtp = resolve_mtp(args.model, args.mtp)
    prompt = PROMPTS.get(args.workload, args.workload)
    print(f"[self-draft] target={os.path.basename(args.model)} mtp={os.path.basename(mtp)} "
          f"ngl={args.ngl} n-max={args.n_max} workload={args.workload} n_predict={args.n_predict}")

    with Server(binary, args.model, port=args.port, ngl=args.ngl, threads=args.threads,
                ctx=args.ctx) as s:
        v = s.complete(prompt, args.n_predict)
    vm = gen_metrics(v)

    with Server(binary, args.model, port=args.port, ngl=args.ngl, threads=args.threads,
                ctx=args.ctx, mtp=mtp, n_max=args.n_max, p_min=args.p_min) as s:
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
    mtp = resolve_mtp(args.model, args.mtp)
    prompt = PROMPTS.get(args.workload, args.workload)
    grid = [int(x) for x in args.grid.split(",")]
    print(f"[self-draft] autotune n-max over {grid} (workload={args.workload})")
    # vanilla baseline
    with Server(binary, args.model, port=args.port, ngl=args.ngl, threads=args.threads, ctx=args.ctx) as s:
        vm = gen_metrics(s.complete(prompt, args.n_predict))
    print(f"vanilla: {vm['tok_s']:.2f} tok/s")
    rows, best = [], None
    for nm in grid:
        with Server(binary, args.model, port=args.port, ngl=args.ngl, threads=args.threads,
                    ctx=args.ctx, mtp=mtp, n_max=nm, p_min=args.p_min) as s:
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
    mtp = resolve_mtp(args.model, args.mtp)
    cmd = Server(binary, args.model, port=args.port, ngl=args.ngl, threads=args.threads,
                 ctx=args.ctx, mtp=mtp, n_max=args.n_max, p_min=args.p_min).cmd()
    print("[self-draft] launching:\n  " + " ".join(cmd))
    os.execv(binary, cmd)

def do_agent(args):
    from agent_demo import run_agent
    binary = find_server()
    mtp = resolve_mtp(args.model, args.mtp)
    print(f"[self-draft] agent end-to-end latency: vanilla vs self-draft (n-max={args.n_max})")
    with Server(binary, args.model, port=args.port, ngl=args.ngl, threads=args.threads, ctx=args.ctx):
        v = run_agent(args.port, verbose=args.verbose)
    with Server(binary, args.model, port=args.port, ngl=args.ngl, threads=args.threads, ctx=args.ctx,
                mtp=mtp, n_max=args.n_max, p_min=args.p_min):
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
        p.add_argument("--ngl", type=int, default=99, help="GPU layers (0 = CPU only; CPU avoids Metal dual-context bug)")
        p.add_argument("--threads", type=int, default=None)
        p.add_argument("--ctx", type=int, default=4096)
        p.add_argument("--port", type=int, default=8099)
        p.add_argument("--p-min", type=float, default=0.0)
        p.add_argument("--workload", default="code", help="'code', 'prose', or a literal prompt string")
        p.add_argument("--n-predict", type=int, default=200)
        p.add_argument("--price", type=float, default=0.0, help="instance price $/hr (e.g. c7g.xlarge ~0.145) -> reports $/1M tokens")
        p.add_argument("--json", help="write metrics JSON here")
    b = sub.add_parser("bench"); common(b); b.add_argument("--n-max", type=int, default=3); b.set_defaults(fn=do_bench)
    a = sub.add_parser("autotune"); common(a); a.add_argument("--grid", default="1,2,3,4,6,8"); a.set_defaults(fn=do_autotune)
    r = sub.add_parser("run"); common(r); r.add_argument("--n-max", type=int, default=3); r.set_defaults(fn=do_run)
    g = sub.add_parser("agent"); common(g); g.add_argument("--n-max", type=int, default=3)
    g.add_argument("-v", "--verbose", action="store_true"); g.set_defaults(fn=do_agent)
    args = ap.parse_args()
    args.fn(args)

if __name__ == "__main__":
    main()
