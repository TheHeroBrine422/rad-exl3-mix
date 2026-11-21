#!/bin/bash
# run_grid.sh PAR  -- restartable driver: every (matrix, setting) job of the w211 grid, P parallel single-threaded jobs, heaviest first.
# env: W211_ROWS (rows per slice, 0 = all), W211_SLICE, OUT dir, SETTINGS list. Needs experts.json (pick_experts.py). Skips finished (matrix, setting, K).
cd /data/q27/w211; P=${1:-30}; ROWS_GU=${W211_ROWS_GU:-576}; ROWS_DN=${W211_ROWS_DN:-640}; SL=${W211_SLICE:-0}
SETS=${SETTINGS:-"base d0.001 d0.1 pooled mix1 mix0.1 d0.005 d0.025 d0.05"}
export E3_V2=3 E3_TB=1 OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p out logs
python3 - > jobs.$SL.txt <<PY
import json
d = json.load(open("experts.json")); sets = "$SETS".split()
jobs = []
for s in sets:
    for L, v in d.items():
        for cls in ("hot", "medium", "cold"):
            for x in v[cls]:
                for kind in ("gate_up", "down"):
                    if kind == "down" and s in ("pooled", "mix1", "mix0.1"): continue
                    jobs.append((s, L, x["e"], kind))
for s, L, e, kind in jobs:       # settings in priority order (stable), per setting gate_up before down
    print(L, e, kind, s)
PY
cat jobs.$SL.txt | while read L e kind s; do
  r=$ROWS_GU; [ $kind = down ] && r=$ROWS_DN
  echo "$L $e $kind $s $r"
done | xargs -P $P -L 1 sh -c 'W211_ROWS=$4 W211_SLICE='$SL' /data/q27/venv/bin/python -I -u ds_check.py $0 $1 $2 $3 out/res-s'$SL'-$2-L$0-e$1-$3.jsonl > logs/s'$SL'-$2-L$0-e$1-$3.log 2>&1; echo "$(date -u +%H:%M) $0 $1 $2 $3 rc=$?" >> logs/s'$SL'.done'
echo "GRID slice $SL ALL DONE $(date -u +%H:%M)" >> logs/s$SL.done
