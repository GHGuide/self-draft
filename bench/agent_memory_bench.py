#!/usr/bin/env python3
"""
Multi-agent serving bench: persistent KV (agent_memory) vs naive re-prefill, on Arm CPU.

Scenario: N agents, only K RAM-resident slots (K < N), R rounds round-robin. After round 1
every agent has been evicted, so every round-2+ turn is a "return". Naive serving re-prefills
the agent's long context (slow); our manager restores its saved KV (fast). Measures per-turn
TTFT for both and the aggregate win. Run with the server's --swa-full (Gemma SWA).
"""
import json, os, sys, time, subprocess, signal, urllib.request, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "selfdraft"))
from agent_memory import AgentMemory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
        json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))

def agent_prompt(aid, pad):
    roles = ["a senior backend engineer", "a security auditor", "a data scientist",
             "a devops specialist", "a frontend architect", "a QA lead"]
    role = roles[aid % len(roles)]
    ctx = (f"You are agent #{aid}, {role}, working in a large monorepo. " * pad)
    return (f"<start_of_turn>user\n{ctx}\nGiven your role, state your single top priority "
            f"this sprint in one short sentence.<end_of_turn>\n<start_of_turn>model\n")

def launch_server(model, port, n_slots, ctx, slot_dir, cache_ram=0, self_draft=False):
    # cache_ram=0 disables llama.cpp's in-RAM prompt cache, isolating the paper's regime:
    # RAM too small to hold every agent's KV -> the ONLY way to avoid re-prefill is our
    # disk save/restore. (With the default 8GB RAM cache, the server already reuses KV
    # across slots for small/few agents, so disk persistence only wins once RAM is exceeded
    # or across server restarts.)
    bin = os.path.join(ROOT, "llama.cpp/build/bin/llama-server")
    log = open(os.path.join(ROOT, "bench", f"agentmem_srv_{port}.log"), "w")
    cmd = [bin, "-m", model, "-ngl", "0", "-t", "8", "-c", str(ctx),
        "--swa-full", "--slot-save-path", slot_dir, "-np", str(n_slots),
        "--cache-ram", str(cache_ram), "--port", str(port), "--host", "127.0.0.1"]
    if self_draft:   # bundle the MTP self-draft decode layer
        mtp = os.path.join(os.path.dirname(model), "mtp-gemma-4-12b-it.gguf")
        cmd += ["-md", mtp, "--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--spec-draft-p-min", "0"]
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    for _ in range(120):
        try:
            if json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)).get("status") == "ok":
                return p
        except Exception: pass
        if p.poll() is not None: sys.exit("server died; see log")
        time.sleep(1)
    os.killpg(os.getpgid(p.pid), signal.SIGTERM); sys.exit("server timeout")

def kill(p):
    try: os.killpg(os.getpgid(p.pid), signal.SIGTERM); p.wait(timeout=15)
    except Exception: pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models/gemma-4-12b-it-Q4_K_M.gguf"))
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--ram-slots", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--pad", type=int, default=60, help="context padding (~tokens = pad*12)")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--n-predict", type=int, default=24)
    ap.add_argument("--self-draft", action="store_true", help="bundle self-draft decode (needs -md launch)")
    ap.add_argument("--json")
    a = ap.parse_args()
    slot_dir = os.path.join(ROOT, "slots"); os.makedirs(slot_dir, exist_ok=True)
    os.system(f"rm -f {slot_dir}/agent_*.bin")
    prompts = {i: agent_prompt(i, a.pad) for i in range(a.agents)}
    order = [i for _ in range(a.rounds) for i in range(a.agents)]  # round-robin

    # ---- NAIVE: server auto-manages K slots; returns after eviction re-prefill ----
    p = launch_server(a.model, a.port, a.ram_slots, a.ctx, slot_dir)
    naive = []
    for aid in order:
        r = post(a.port, "/completion", {"prompt": prompts[aid], "n_predict": a.n_predict,
                 "temperature": 0, "cache_prompt": True})
        naive.append({"agent": aid, "ttft_ms": r["timings"]["prompt_ms"], "reproc": r["timings"]["prompt_n"]})
    kill(p)

    # ---- PERSISTENT: our manager saves+restores KV across evictions ----
    os.system(f"rm -f {slot_dir}/agent_*.bin")
    p = launch_server(a.model, a.port, a.ram_slots, a.ctx, slot_dir, self_draft=a.self_draft)
    mem = AgentMemory(a.port, a.ram_slots, slot_dir)
    for aid in order:
        mem.turn(aid, prompts[aid], n_predict=a.n_predict, self_draft_n_max=(3 if a.self_draft else None))
    kill(p)
    persistent = mem.stats

    # ---- compare (returns = turns after the first `agents` turns) ----
    nret = a.agents
    naive_ret = naive[nret:]; pers_ret = persistent[nret:]
    n_ttft = sum(x["ttft_ms"] for x in naive_ret) / max(1, len(naive_ret))
    p_ttft = sum(x["ttft_ms"] for x in pers_ret) / max(1, len(pers_ret))
    print(f"\n===== multi-agent serving: {a.agents} agents, {a.ram_slots} RAM slots, {a.rounds} rounds =====")
    print(f"context ~{a.pad*12} tokens/agent")
    print(f"{'turn':>4} {'agent':>5} | {'NAIVE ttft':>11} {'reproc':>7} | {'PERSIST ttft':>13} {'warm':>5}")
    for i,(nv,ps) in enumerate(zip(naive, persistent)):
        print(f"{i:>4} {nv['agent']:>5} | {nv['ttft_ms']:>9.0f}ms {nv['reproc']:>7} | {ps['ttft_ms']:>11.0f}ms {str(ps['warm']):>5}")
    print(f"\nRETURNING-turn avg TTFT:  naive {n_ttft:.0f} ms  ->  persistent {p_ttft:.0f} ms"
          f"  ({n_ttft/max(p_ttft,0.01):.1f}x lower)")
    gts = [x["gen_tok_s"] for x in persistent if x.get("gen_tok_s")]
    if gts:
        print(f"persistent decode: {sum(gts)/len(gts):.1f} tok/s avg"
              + ("  (self-draft bundled)" if a.self_draft else "  (no draft)"))
    if a.json:
        json.dump({"naive": naive, "persistent": persistent,
                   "naive_return_ttft_ms": n_ttft, "persistent_return_ttft_ms": p_ttft,
                   "ttft_speedup": n_ttft/max(p_ttft,0.01),
                   "config": {"agents": a.agents, "ram_slots": a.ram_slots, "rounds": a.rounds, "pad": a.pad}},
                  open(a.json, "w"), indent=2)
        print(f"wrote {a.json}")
    print("AGENTMEM_BENCH_DONE")

if __name__ == "__main__":
    main()
