#!/bin/bash
# batch6.sh -- E1b group-512 A4 scales on ours-K6, calib-A (after batch5)
: ${RXM_WORK:=/root} ${RXM_OURS:=/data/ours}; export RXM_WORK RXM_OURS   # paths (README: Paths)
cd ${RXM_WORK}/e1b; MAP=runs/e0-ident.map; M=${RXM_OURS}/ours-K6.rad
until grep -q BATCH5DONE batch5.out 2>/dev/null; do sleep 60; done
r() { ./e1brun.sh "$1" $M golden-calA corpus/calA.jsonl "${@:2}"; }
mk() { python3 -I mkmask.py $MAP "$@"; }
for f in gate_up down; do
  r a4-$f-g512 EXL3_AQ_MASK=$(mk 4 $f all) EXL3_AQ_GROUP=512
  r a4-$f-g512-clip0.85 EXL3_AQ_MASK=$(mk 4 $f all) EXL3_AQ_GROUP=512 EXL3_AQ_CLIP=0.85
done
echo BATCH6DONE
