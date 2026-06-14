#!/usr/bin/env bash
# GO/NO-GO gate for the Arm Agent-Memory pivot: does restoring a saved KV slot beat a
# cold re-prefill on TTFT? (arXiv 2603.04428). GATE: warm restore >= 5x faster prefill.
set -uo pipefail
R="/Users/leonardo/Downloads/5 hackathons quick wins/Arm Create"
SRV="$R/llama.cpp/build/bin/llama-server"
MODEL="$R/models/gemma-4-12b-it-Q4_K_M.gguf"
PORT=8099
mkdir -p "$R/slots"
pkill -f "llama-server.*$PORT" 2>/dev/null; sleep 1

# Gemma is sliding-window attention -> --swa-full required or slot restore drops
# out-of-window tokens (llama.cpp discussion #20572). -np 1 single slot.
"$SRV" -m "$MODEL" -ngl 0 -t 8 -c 8192 --swa-full --slot-save-path "$R/slots" \
  -np 1 --port $PORT --host 127.0.0.1 > "$R/bench/kv_gate_srv.log" 2>&1 &
n=0; until grep -qi "server is listening" "$R/bench/kv_gate_srv.log" 2>/dev/null || grep -qiE "error|assert" "$R/bench/kv_gate_srv.log" 2>/dev/null || [ $n -ge 60 ]; do sleep 2; n=$((n+1)); done

python3 - "$PORT" "$R" <<'PY'
import json, sys, urllib.request, time
port, R = sys.argv[1], sys.argv[2]
def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
        json.dumps(body).encode(), {"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))

# ~3-4k token agent context (long system+tools+history prefix is the expensive prefill)
para = ("You are an autonomous coding agent operating inside a large repository. "
        "Your tools are read_file, write_file, run_tests, grep, and git. Follow the plan, "
        "reason step by step, cite file paths, and keep changes minimal and reviewable. ")
prompt = "<start_of_turn>user\n" + (para * 90) + \
         "\nNow summarize your operating rules in one sentence.<end_of_turn>\n<start_of_turn>model\n"

# COLD: fresh slot, full prefill
c = post("/completion", {"prompt": prompt, "n_predict": 1, "temperature": 0, "cache_prompt": True})
cold_ms = c["timings"]["prompt_ms"]; pn = c["timings"]["prompt_n"]

# SAVE -> ERASE (simulate eviction) -> RESTORE (no recompute)
post("/slots/0?action=save", {"filename": "agent0.bin"})
post("/slots/0?action=erase", {})
t0 = time.monotonic()
post("/slots/0?action=restore", {"filename": "agent0.bin"})
restore_ms = (time.monotonic() - t0) * 1000.0

# WARM: identical prompt -> restored KV should match prefix -> ~no prefill
w = post("/completion", {"prompt": prompt, "n_predict": 1, "temperature": 0, "cache_prompt": True})
warm_ms = w["timings"]["prompt_ms"]; wpn = w["timings"]["prompt_n"]

import os
slot_mb = os.path.getsize(os.path.join(R, "slots", "agent0.bin"))/1e6 if os.path.exists(os.path.join(R,"slots","agent0.bin")) else -1
ratio = cold_ms / warm_ms if warm_ms > 0.01 else float('inf')
print(f"prompt tokens (cold prefill_n) : {pn}")
print(f"COLD prefill TTFT  : {cold_ms:8.1f} ms  ({pn} tokens)")
print(f"slot restore (disk): {restore_ms:8.1f} ms  (file {slot_mb:.0f} MB)")
print(f"WARM prefill TTFT  : {warm_ms:8.1f} ms  (reprocessed_n={wpn})")
print(f"effective warm TTFT (restore+warm prefill): {restore_ms+warm_ms:8.1f} ms")
print(f"TTFT SPEEDUP (cold / (restore+warm)) : {cold_ms/(restore_ms+warm_ms):.1f}x")
gate = cold_ms/(restore_ms+warm_ms)
print("GATE >=5x : " + ("PASS -> GO" if gate >= 5 else ("MARGINAL(3-5x)" if gate>=3 else "FAIL -> reconsider")))
PY
pkill -f "llama-server.*$PORT" 2>/dev/null
echo "KV_GATE_DONE"
