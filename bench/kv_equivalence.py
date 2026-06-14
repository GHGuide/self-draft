#!/usr/bin/env python3
"""
Prove KV slot restore is BIT-EXACT: evicting an agent mid-generation and restoring its
saved KV yields output byte-identical to never evicting. Unlike speculative decoding
(distributionally lossless, FP-tie flips), persistent KV is exactly the same tensors ->
exactly the same logits -> identical tokens. Recovers the 'byte-identical' guarantee.
"""
import json, os, sys, time, subprocess, signal, urllib.request, hashlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8099
SLOTS = os.path.join(ROOT, "slots"); os.makedirs(SLOTS, exist_ok=True)
MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "models/gemma-4-12b-it-Q4_K_M.gguf")

def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))

def gen(prompt, n, cache=True):
    return post("/completion", {"prompt": prompt, "n_predict": n, "temperature": 0,
                                "top_k": 1, "cache_prompt": cache, "id_slot": 0})["content"]

bin = os.path.join(ROOT, "llama.cpp/build/bin/llama-server")
log = open(os.path.join(ROOT, "bench", "kv_eq_srv.log"), "w")
os.system(f"rm -f {SLOTS}/eq.bin")
p = subprocess.Popen([bin, "-m", MODEL, "-ngl", "0", "-t", "8", "-c", "4096", "--swa-full",
    "--slot-save-path", SLOTS, "-np", "1", "--cache-ram", "0", "--port", str(PORT),
    "--host", "127.0.0.1"], stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
for _ in range(120):
    try:
        if json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)).get("status")=="ok": break
    except Exception: pass
    time.sleep(1)

P = "<start_of_turn>user\nList 12 distinct uses for a paperclip, numbered, one per line.<end_of_turn>\n<start_of_turn>model\n"
try:
    # Reference: generate 72 tokens straight through (no eviction)
    post("/slots/0?action=erase", {})
    ref = gen(P, 72, cache=False)

    # Restore path: generate 36, save+evict+restore, continue 36 from restored KV
    post("/slots/0?action=erase", {})
    a = gen(P, 36, cache=False)
    post("/slots/0?action=save", {"filename": "eq.bin"})
    post("/slots/0?action=erase", {})
    post("/slots/0?action=restore", {"filename": "eq.bin"})
    b = gen(P + a, 36, cache=True)
    restored = a + b

    h1 = hashlib.sha256(ref.encode()).hexdigest()
    h2 = hashlib.sha256(restored.encode()).hexdigest()
    print(f"reference (no evict) sha: {h1[:16]}")
    print(f"restore mid-stream  sha: {h2[:16]}")
    print("BYTE-IDENTICAL: " + ("YES - KV restore is bit-exact (lossless recovered)" if h1==h2
          else f"NO (ref len {len(ref)} vs restored {len(restored)})"))
    print("KV_EQ_DONE")
finally:
    try: os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except Exception: pass
