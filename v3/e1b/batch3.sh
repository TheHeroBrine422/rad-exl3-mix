#!/bin/bash
# batch3.sh -- E1b A4 fixes on ours-K6, calib-A: group-128 activation scales and clipping, on gate_up and down
: ${RXM_WORK:=/root} ${RXM_OURS:=/data/ours}; export RXM_WORK RXM_OURS   # paths (README: Paths)
cd ${RXM_WORK}/e1b; MAP=runs/e0-ident.map; M=${RXM_OURS}/ours-K6.rad
r() { ./e1brun.sh "$1" $M golden-calA corpus/calA.jsonl "${@:2}"; }
mk() { python3 -I mkmask.py $MAP "$@"; }
for f in gate_up down; do
  r a4-$f-g128 EXL3_AQ_MASK=$(mk 4 $f all) EXL3_AQ_GROUP=128
  for c in 0.85 0.7 0.55; do r a4-$f-clip$c EXL3_AQ_MASK=$(mk 4 $f all) EXL3_AQ_CLIP=$c; done
  r a4-$f-g128-clip0.85 EXL3_AQ_MASK=$(mk 4 $f all) EXL3_AQ_GROUP=128 EXL3_AQ_CLIP=0.85
done
echo BATCH3DONE
