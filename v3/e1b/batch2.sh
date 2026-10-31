#!/bin/bash
# batch2.sh -- E1b A8 follow-up on ours-K6, calib-A: which tensors carry the A8 cost, and finer scales
: ${RXM_WORK:=/root} ${RXM_OURS:=/data/ours}; export RXM_WORK RXM_OURS   # paths (README: Paths)
cd ${RXM_WORK}/e1b; MAP=runs/e0-ident.map; M=${RXM_OURS}/ours-K6.rad
r() { ./e1brun.sh "$1" $M golden-calA corpus/calA.jsonl "${@:2}"; }
mk() { python3 -I mkmask.py $MAP "$@"; }
r a8-all-g128 EXL3_AQ_MASK=$(mk 8 all all) EXL3_AQ_GROUP=128
r a8-all-g32  EXL3_AQ_MASK=$(mk 8 all all) EXL3_AQ_GROUP=32
for f in attn_in attn_out gdn_in ssm_out gate_up down head; do r a8-$f EXL3_AQ_MASK=$(mk 8 $f all); done
r a8-nodownhead EXL3_AQ_MASK=$(mk 8 attn_in,attn_out,gdn_in,ssm_out,gate_up all)
echo BATCH2DONE
