#!/bin/bash
# k4grid.sh -- V3 menu ablation (quality only) and speed frontier with knap4.py. Out: /workspace/logs/knap4/g-*.jsonl
cd "$(dirname "$0")"; M=../../data/v3/kfit2-model-final.json A=../e1b/data/aq.jsonl F=../e1b/data/aqfactors.json P=../e1b/data/e0-ident.map
export KN_SPEED=${KN_SPEED:-../bench/data/speedcal-stock.jsonl}      # measured cores; KN_A4=a4time.jsonl adds decode
O=${OUT:-/workspace/logs/knap4}; mkdir -p $O; T=1.75,2.0,2.5,3.0,3.5,4.0,4.5,5.0
run() { local n=$1; shift; [ -s $O/g-$n.jsonl ] || env "$@" python3 knap4.py $M $A $F $P $O/g-$n.jsonl $T 0 > $O/g-$n.log 2>&1; }
run v2 KN_MODES=f16 &
run frac KN_MODES=f16 KN_FRAC=1 &
run emb KN_MODES=f16 KN_EMBT=1 &
run damp KN_MODES=f16 KN_DAMP=1 &
run v3 KN_MODES=f16 KN_FRAC=1 KN_EMBT=1 KN_DAMP=1 &
run v3qh KN_MODES=f16 KN_FRAC=1 KN_EMBT=1 KN_DAMP=1 KN_HEAD_STEP=0.25 &
wait
[ -s $O/g-v3speed.jsonl ] || KN_FRAC=1 KN_EMBT=1 KN_DAMP=1 python3 knap4.py $M $A $F $P $O/g-v3speed.jsonl 2.0,2.5,3.0,4.0 0,1e-6,3e-6,1e-5,3e-5 > $O/g-v3speed.log 2>&1
echo K4GRID DONE >> $O/g-v3speed.log
