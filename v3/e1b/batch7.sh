#!/bin/bash
# batch7.sh -- E1b A8 group size sweep on ours-K6, calib-A (after batch6): which group keeps A8 ~free
: ${RXM_WORK:=/root} ${RXM_OURS:=/data/ours}; export RXM_WORK RXM_OURS   # paths (README: Paths)
cd ${RXM_WORK}/e1b; MAP=runs/e0-ident.map; M=${RXM_OURS}/ours-K6.rad
until grep -q BATCH6DONE batch6.out 2>/dev/null; do sleep 60; done
r() { ./e1brun.sh "$1" $M golden-calA corpus/calA.jsonl "${@:2}"; }
mk() { python3 -I mkmask.py $MAP "$@"; }
for g in 512 1024; do r a8-all-g$g EXL3_AQ_MASK=$(mk 8 all all) EXL3_AQ_GROUP=$g; done
r a8-all-clip0.9 EXL3_AQ_MASK=$(mk 8 all all) EXL3_AQ_CLIP=0.9
echo BATCH7DONE
