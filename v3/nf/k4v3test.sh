#!/bin/bash
# k4v3test.sh -- KN_V3 option table (critique C3.2): regression (switch off == HEAD's knap4, byte for byte) + V3 solves.
# Out: /workspace/logs/knap4v3/. Background job (~20-40 min on the VM).
cd "$(dirname "$0")"; O=/workspace/logs/knap4v3; mkdir -p $O; rm -f $O/*.jsonl
M=../../data/v3/kfit2-model-final.json A=../e1b/data/aq.jsonl F=../e1b/data/aqfactors.json P=../e1b/data/e0-ident.map
export KN_SPEED=../bench/data/speedcal-stock.jsonl
git show HEAD:v3/nf/knap4.py > .knap4_head.py
COMMON="KN_FRAC=1 KN_EMBT=2 KN_HEADSLAB=1"
env $COMMON python3 .knap4_head.py $M $A $F $P $O/old.jsonl 2.5,4.0 0,3e-6 > $O/old.log 2>&1 &
env $COMMON python3 knap4.py $M $A $F $P $O/new.jsonl 2.5,4.0 0,3e-6 > $O/new.log 2>&1 &
env $COMMON KN_V3=1 KN_LEVELS=../../data/v3/levels KN_DAMP=1 python3 knap4.py $M $A $F $P $O/v3.jsonl \
  2.0,2.5,3.0,3.5,4.0,4.5,5.0,6.0 0,3e-6 > $O/v3.log 2>&1 &
env $COMMON KN_V3=1 KN_GROUPS=0 KN_LEVELS=../../data/v3/levels KN_DAMP=1 python3 knap4.py $M $A $F $P $O/v3pt.jsonl \
  2.5,3.0,4.0 3e-6 > $O/v3pt.log 2>&1 &
wait; rm -f .knap4_head.py
cmp -s $O/old.jsonl $O/new.jsonl && echo "REGRESSION OK: identical" || echo "REGRESSION FAIL: outputs differ"
echo DONE
