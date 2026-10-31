#!/bin/bash
# vtmlslice.sh -- golden-ml: every multilingual document of the fit pool (67 docs, 77k tokens, capped at 8192),
# the guard slice for the vocab-tiered embedding. The vocab ranking used search + qad only, so this is out of sample.
: ${RXM_WORK:=/root}; export RXM_WORK
cd $RXM_WORK; [ -d golden-ml ] && exit 0
D=$(python3 -I - <<'PY'
import json, os
W = os.environ["RXM_WORK"]
pools = json.load(open(f"{W}/challenge/pools.json"))["pool"]
docs = [json.loads(l) for l in open(f"{W}/challenge/corpus/calib.jsonl")]
print(" ".join(f"{i}:8192" if int(d["tokens"]) > 8192 else str(i) for i, d in enumerate(docs)
               if pools[i] == "fit" and d["category"] == "multilingual"))
PY
)
nice python3 -I golden_pick.py challenge/golden/calib challenge/corpus/calib.jsonl golden-ml corpus/ml.jsonl $D
