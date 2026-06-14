#!/usr/bin/env python3
"""
Measure persisted-KV slot-file size at different KV-cache quant levels -> "agents per GB".
Each agent's saved slot is the KV cache; quantizing it (--cache-type-k/v) shrinks the file
and lets more agents' contexts persist in a fixed budget. Restore stays bit-exact relative
to the SAME quant config. V-cache quant requires flash-attn on.
"""
import json, os, sys, time, subprocess, signal, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8099
SLOTS = os.path.join(ROOT, "slots"); os.makedirs(SLOTS, exist_ok=True)
MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "models/gemma-4-12b-it-Q4_K_M.gguf")

def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))

def measure(ktype, vtype):
    os.system(f"rm -f {SLOTS}/dens.bin")
    binp = os.path.join(ROOT, "llama.cpp/build/bin/llama-server")
    log = open(os.path.join(ROOT, "bench", "kv_density_srv.log"), "w")
    p = subprocess.Popen([binp, "-m", MODEL, "-ngl", "0", "-t", "8", "-c", "8192", "--swa-full",
        "-fa", "on", "-ctk", ktype, "-ctv", vtype, "--slot-save-path", SLOTS, "-np", "1",
        "--cache-ram", "0", "--port", str(PORT), "--host", "127.0.0.1"],
        stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    ok = False
    for _ in range(120):
        try:
            if json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)).get("status")=="ok":
                ok = True; break
        except Exception: pass
        if p.poll() is not None: break
        time.sleep(1)
    sz = -1; ntok = -1
    if ok:
        prompt = "<start_of_turn>user\n" + ("You are an autonomous agent in a large repo. " * 110) + \
                 "State your top priority.<end_of_turn>\n<start_of_turn>model\n"
        r = post("/completion", {"prompt": prompt, "n_predict": 1, "temperature": 0, "cache_prompt": True})
        ntok = r["timings"]["prompt_n"]
        post("/slots/0?action=save", {"filename": "dens.bin"})
        fp = os.path.join(SLOTS, "dens.bin")
        sz = os.path.getsize(fp) if os.path.exists(fp) else -1
    try: os.killpg(os.getpgid(p.pid), signal.SIGTERM); p.wait(timeout=15)
    except Exception: pass
    return sz, ntok

print(f"{'KV (k/v)':>14} {'slot MB':>9} {'tokens':>7} {'agents/16GB':>12}")
base = None
for k, v in [("f16","f16"), ("q8_0","q8_0"), ("q4_0","q4_0")]:
    sz, nt = measure(k, v)
    mb = sz/1e6
    if base is None: base = mb
    per16 = int(16000/mb) if mb > 0 else -1
    rel = base/mb if mb > 0 else 0
    print(f"{k+'/'+v:>14} {mb:>9.0f} {nt:>7} {per16:>12}   ({rel:.1f}x more agents vs f16)")
print("KV_DENSITY_DONE")
