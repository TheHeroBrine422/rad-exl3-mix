#!/bin/bash
# k4levels.sh -- which extra trunk levels pay (pre-requant checklist #3): V3 table (quality only), base ladder vs
# + K5.25/5.75, + 1/8 steps K3-5, + both. Interpolated extras (no containers: KN_WRITE unset). Out /workspace/logs/k4levels/
cd "$(dirname "$0")"; O=/workspace/logs/k4levels; mkdir -p $O; rm -f $O/*.jsonl
M=../../data/v3/kfit2-model-final.json A=../e1b/data/aq.jsonl F=../e1b/data/aqfactors.json P=../e1b/data/e0-ident.map
export KN_SPEED=../bench/data/speedcal-stock.jsonl KN_MODES=f16 KN_V3=1 KN_FRAC=1 KN_EMBT=2 KN_HEADSLAB=1 KN_LEVELS=../../data/v3/levels
T=2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0
E8=$(python3 -c "print(','.join(f'{3 + i / 8:g}' for i in range(1, 16) if (i % 2)))")
python3 knap4.py $M $A $F $P $O/base.jsonl $T 0 > $O/base.log 2>&1 &
KN_FRACS_EXTRA=5.25,5.75 python3 knap4.py $M $A $F $P $O/q5.jsonl $T 0 > $O/q5.log 2>&1 &
KN_FRACS_EXTRA=$E8 python3 knap4.py $M $A $F $P $O/e35.jsonl $T 0 > $O/e35.log 2>&1 &
KN_FRACS_EXTRA=$E8,5.25,5.75 python3 knap4.py $M $A $F $P $O/both.jsonl $T 0 > $O/both.log 2>&1 &
wait; echo DONE
