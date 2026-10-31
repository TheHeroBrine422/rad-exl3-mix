#!/bin/bash
# batch4.sh -- E1b group-256 activation scales on ours-K6, calib-A
: ${RXM_WORK:=/root} ${RXM_OURS:=/data/ours}; export RXM_WORK RXM_OURS   # paths (README: Paths)
cd ${RXM_WORK}/e1b; MAP=runs/e0-ident.map; M=${RXM_OURS}/ours-K6.rad
r() { ./e1brun.sh "$1" $M golden-calA corpus/calA.jsonl "${@:2}"; }
mk() { python3 -I mkmask.py $MAP "$@"; }
r a8-all-g256 EXL3_AQ_MASK=$(mk 8 all all) EXL3_AQ_GROUP=256
for f in gate_up down; do
  r a4-$f-g256 EXL3_AQ_MASK=$(mk 4 $f all) EXL3_AQ_GROUP=256
  r a4-$f-g256-clip0.85 EXL3_AQ_MASK=$(mk 4 $f all) EXL3_AQ_GROUP=256 EXL3_AQ_CLIP=0.85
done
r a4-all-g128-clip0.85 EXL3_AQ_MASK=$(mk 4 all all) EXL3_AQ_GROUP=128 EXL3_AQ_CLIP=0.85
echo BATCH4DONE
