#!/bin/bash
# batch1.sh -- E1b first pass on ours-K6, calib-A (fit pool): floors, per family (A4), per depth band (A4)
: ${RXM_WORK:=/root} ${RXM_OURS:=/data/ours}; export RXM_WORK RXM_OURS   # paths (README: Paths)
cd ${RXM_WORK}/e1b; MAP=runs/e0-ident.map; M=${RXM_OURS}/ours-K6.rad
r() { ./e1brun.sh "$1" $M golden-calA corpus/calA.jsonl "${@:2}"; }
mk() { python3 -I mkmask.py $MAP "$@"; }
r a8-all   EXL3_AQ_MASK=$(mk 8 all all)
r a4-all   EXL3_AQ_MASK=$(mk 4 all all)
for f in attn_in attn_out gdn_in ssm_out gate_up down head; do r a4-$f EXL3_AQ_MASK=$(mk 4 $f all); done
for b in 0-15 16-31 32-47 48-63; do r a4-L$b EXL3_AQ_MASK=$(mk 4 all $b); done
r a4-all-g128 EXL3_AQ_MASK=$(mk 4 all all) EXL3_AQ_GROUP=128
echo BATCH1DONE
