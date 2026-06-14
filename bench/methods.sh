#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; SRV="$ROOT/llama.cpp/build/bin/llama-server"
T="$ROOT/models/gemma-4-12b-it-Q4_K_M.gguf"; MTP="$ROOT/models/mtp-gemma-4-12b-it.gguf"
PORT=8099; NGL=0; TH=8
ready(){ local n=0; until grep -qi "server is listening" "$1" 2>/dev/null || grep -qiE "error|assert|abort" "$1" 2>/dev/null || [ $n -ge 50 ]; do sleep 2; n=$((n+1)); done; }
tok(){ python3 -c "import json,sys;print('%.2f'%json.load(open(sys.argv[1]))['timings']['predicted_per_second'])" "$1" 2>/dev/null || echo ERR; }
run(){ # tag, reqfile, extra server args...
  local tag=$1 req=$2; shift 2
  pkill -f "llama-server.*$PORT" 2>/dev/null; sleep 2
  "$SRV" -m "$T" -ngl $NGL -t $TH --port $PORT --host 127.0.0.1 -c 4096 "$@" >"$ROOT/bench/m.log" 2>&1 &
  ready "$ROOT/bench/m.log"
  /usr/bin/curl -s --max-time 600 http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' --data @"$req" >"$ROOT/bench/m_out.json" 2>/dev/null
  echo "$tag: $(tok "$ROOT/bench/m_out.json") tok/s"
  pkill -f "llama-server.*$PORT" 2>/dev/null; sleep 1
}
for W in "code:$ROOT/bench/req2.json" "json:$ROOT/bench/req3.json"; do
  name="${W%%:*}"; req="${W#*:}"
  echo "===== workload: $name ====="
  run "  vanilla      " "$req"
  run "  mtp(n3)      " "$req" -md "$MTP" --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-ngl $NGL --spec-draft-p-min 0
  run "  ngram-mod    " "$req" --spec-type ngram-mod
  run "  mtp+ngram    " "$req" -md "$MTP" --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-draft-ngl $NGL --spec-draft-p-min 0
done
echo "METHODS_DONE"
