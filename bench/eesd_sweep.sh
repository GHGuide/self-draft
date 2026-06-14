#!/usr/bin/env bash
set -uo pipefail
R="/Users/leonardo/Downloads/5 hackathons quick wins/Arm Create"
T="$R/models/gemma-4-12b-it-Q4_K_M.gguf"
echo "K,draft_GB,vanilla_toks,selfdraft_toks,speedup,accept_pct,similarity_pct"
for K in 44 40 36 32 24; do
  D="$R/models/sd-k$K.gguf"
  [ -f "$D" ] || PYTHONPATH="$R/llama.cpp/gguf-py" python3 "$R/selfdraft/make_self_draft.py" --in "$T" --out "$D" --keep $K >/dev/null 2>&1
  gb=$(python3 -c "import os;print('%.2f'%(os.path.getsize('$D')/1e9))")
  J="$R/bench/eesd_k$K.json"
  python3 "$R/selfdraft/sd.py" bench "$T" --methods draft-simple --draft-model "$D" \
    --ngl 0 --threads 8 --n-max 4 --workload code --n-predict 160 --json "$J" >/dev/null 2>&1
  python3 - "$J" "$K" "$gb" <<'PY'
import json,sys
j=json.load(open(sys.argv[1])); k=sys.argv[2]; gb=sys.argv[3]
v=j["vanilla"]["tok_s"]; d=j["self_draft"]["tok_s"]; sp=j["speedup"]
ac=j["self_draft"].get("accept_pct",0); sim=j["equivalence"]["similarity"]*100
print(f"{k},{gb},{v:.2f},{d:.2f},{sp:.2f},{ac:.1f},{sim:.1f}")
PY
  rm -f "$D"
done
echo "EESD_SWEEP_DONE"
