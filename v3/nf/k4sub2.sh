#!/bin/bash
# k4sub2.sh -- how much of knap4's low-bpw gain rests on sub-2 fractional levels (critique C3.7): quality-only solves at
# 1.75-3.0 bpw with V2's menu, the old V3 menu (K1.25-1.75 interpolated) and the V3 table (ladder levels >= 2 only).
cd "$(dirname "$0")"; O=/workspace/logs/knap4sub2; mkdir -p $O; rm -f $O/*.jsonl
M=../../data/v3/kfit2-model-final.json A=../e1b/data/aq.jsonl F=../e1b/data/aqfactors.json P=../e1b/data/e0-ident.map
export KN_SPEED=../bench/data/speedcal-stock.jsonl KN_MODES=f16; T=1.75,2.0,2.25,2.5,3.0
python3 knap4.py $M $A $F $P $O/v2.jsonl $T 0 > $O/v2.log 2>&1 &
KN_FRAC=1 KN_EMBT=2 python3 knap4.py $M $A $F $P $O/oldv3.jsonl $T 0 > $O/oldv3.log 2>&1 &
KN_V3=1 KN_FRAC=1 KN_EMBT=2 KN_LEVELS=../../data/v3/levels python3 knap4.py $M $A $F $P $O/v3.jsonl $T 0 > $O/v3.log 2>&1 &
wait; echo DONE
