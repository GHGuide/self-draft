#!/usr/bin/env bash
# MTP self-speculative bench via llama-server (MTP is wired in server, not speculative-simple).
# Compares vanilla vs --spec-type draft-mtp. Greedy => byte-identical check. Reports tok/s + accept.
# Usage: ./mtp_bench.sh <target.gguf> <mtp.gguf> [n_predict] [n_max] [ngl]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRV="$ROOT/llama.cpp/build/bin/llama-server"
T="${1:?target}"; M="${2:?mtp draft}"; NPRED="${3:-128}"; NMAX="${4:-4}"; NGL="${5:-99}"
PORT=8099
PROMPT='<start_of_turn>user
Explain why the sky is blue.<end_of_turn>
<start_of_turn>model
'

kill_srv(){ pkill -f "llama-server.*--port $PORT" 2>/dev/null; sleep 2; }
wait_ready(){
  local log=$1 n=0
  until grep -qiE "server is listening|all slots are idle|HTTP server listening" "$log" 2>/dev/null; do
    sleep 1; n=$((n+1)); [ $n -ge 180 ] && { echo "server timeout"; tail -5 "$log"; return 1; }
    grep -qiE "error|failed|abort|assert" "$log" 2>/dev/null && { echo "server error"; tail -8 "$log"; return 1; }
  done
}
hit(){ # -> writes JSON to $1
  python3 - "$PORT" "$PROMPT" "$NPRED" "$2" <<'PY'
import sys,json,urllib.request
port,prompt,npred,outf=sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4]
body=json.dumps({"prompt":prompt,"n_predict":npred,"temperature":0,"top_k":1,"cache_prompt":False}).encode()
req=urllib.request.Request(f"http://127.0.0.1:{port}/completion",body,{"Content-Type":"application/json"})
r=json.load(urllib.request.urlopen(req,timeout=600))
open(outf,"w").write(json.dumps(r))
t=r.get("timings",{})
print("tok/s=%.3f predicted_n=%s | draft_n=%s draft_accepted=%s | accept=%s%%"%(
  t.get("predicted_per_second",0), t.get("predicted_n"),
  t.get("draft_n"), t.get("draft_n_accepted"),
  ("%.1f"%(100*t.get("draft_n_accepted",0)/t["draft_n"]) if t.get("draft_n") else "n/a")))
PY
}
content(){ python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['content'])" "$1"; }

kill_srv
echo "=== VANILLA server (no draft) ==="
"$SRV" -m "$T" -ngl "$NGL" --port $PORT --host 127.0.0.1 -c 2048 >"$ROOT/bench/srv_v.log" 2>&1 &
wait_ready "$ROOT/bench/srv_v.log" || exit 1
echo -n "[vanilla] "; hit x "$ROOT/bench/mtp_v.json"
kill_srv

echo "=== MTP server (--spec-type draft-mtp) ==="
"$SRV" -m "$T" -md "$M" --spec-type draft-mtp -ngl "$NGL" --spec-draft-ngl "$NGL" \
  --spec-draft-n-max "$NMAX" --spec-draft-p-min 0 --port $PORT --host 127.0.0.1 -c 2048 >"$ROOT/bench/srv_m.log" 2>&1 &
wait_ready "$ROOT/bench/srv_m.log" || { echo "MTP server failed:"; grep -iE "error|assert|abort|mtp|spec" "$ROOT/bench/srv_m.log" | tail -10; exit 1; }
echo -n "[mtp]     "; hit x "$ROOT/bench/mtp_m.json"
kill_srv

echo "=== EQUIVALENCE ==="
hv=$(content "$ROOT/bench/mtp_v.json" | shasum -a256 | awk '{print $1}')
hm=$(content "$ROOT/bench/mtp_m.json" | shasum -a256 | awk '{print $1}')
echo "vanilla sha=$hv"; echo "mtp     sha=$hm"
[ "$hv" = "$hm" ] && echo "BYTE-IDENTICAL ✅" || echo "DIVERGED ❌"
echo "MTP_BENCH_DONE"
