#!/bin/bash
# run_grid.sh TOOL PAR [LAYERS]  -- restartable xargs driver (box2 CPU): one single-thread job per (matrix) of w211's 18 experts x {gate_up, down}.
# TOOL = heph (heph_check.py, specs in HEPH_SPECS) | ours (ours_metrics.py, Ks in W235_KS). Results out/<tool>-<tag>/..., one jsonl per matrix.
# env: TAG (default main), W235_ROWS_GU / W235_ROWS_DN (env.sh: 288 / 320; 0 = all rows). Finished (matrix, spec) lines are skipped.
cd /data/q27/w235; . code/heph235/env.sh
TOOL=$1; P=${2:-8}; LAYERS=${3:-"5 20 35"}; TAG=${TAG:-main}
SCRIPT=code/heph235/heph_check.py; [ $TOOL = ours ] && SCRIPT=code/heph235/ours_metrics.py
mkdir -p out/$TOOL-$TAG logs/$TOOL-$TAG
python3 - "$LAYERS" > jobs.$TOOL-$TAG.txt <<PY
import json, sys
d = json.load(open("code/w211/experts.json")); Ls = sys.argv[1].split()
for L in Ls:
    for cls in ("hot", "medium", "cold"):
        for x in d[L][cls]:
            for kind in ("gate_up", "down"): print(L, x["e"], kind)
PY
cat jobs.$TOOL-$TAG.txt | xargs -P $P -L 1 sh -c "nice -n 10 $PY -I -u $SCRIPT \$0 \$1 \$2 out/$TOOL-$TAG/\$2-L\$0-e\$1.jsonl > logs/$TOOL-$TAG/\$2-L\$0-e\$1.log 2>&1; echo \"\$(date +%H:%M) \$0 \$1 \$2 rc=\$?\" >> logs/$TOOL-$TAG.done"
echo "GRID $TOOL-$TAG DONE $(date +%H:%M)" >> logs/$TOOL-$TAG.done
