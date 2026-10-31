#!/bin/bash
# bf16donor.sh -- the unquantized donor for the knapsack's bf16 option: $RXM_MODELS/q38-27b-bf16.rad (plain
# checkpoint conversion, qwen35 entry names), its entry sizes appended to radls.jsonl, and one test mix (layer 0's
# ffn_down in bf16 on SC 4.00) queued to check the exl3 architecture serves a bf16 weight on the card.
# Runs beside the evals (a plain conversion fits in the ~15 GB they leave).
: ${RXM_WORK:=/root} ${RXM_MODELS:=$RXM_WORK/models/final} ${RXM_RADIANCE:=$RXM_WORK/install-card}; export RXM_WORK RXM_MODELS RXM_RADIANCE   # paths (README: Paths)
cd ${RXM_WORK}
O=${RXM_MODELS}/q38-27b-bf16.rad
if [ ! -f $O ]; then
  ( exec 9>${RXM_WORK}/conv.lock; flock 9   # no ram.lock: two overlapping eval workers would starve an exclusive wait; ~15 GB is free
    export PATH=${RXM_RADIANCE}/bin:$PATH RADIANCE_HOME=${RXM_RADIANCE}/share/radiance RADIANCE_HOST_DEVICES=1
    cd ${RXM_CONVDIR:-/workspace} && nice rad-convert models/qwen38-27b-bf16 -o $O.part > ${RXM_WORK}/bf16donor.log 2>&1 && mv $O.part $O ) \
    || { echo "$(date +%H:%M) bf16 donor conversion FAIL" >> ourchain.log; exit 1; }
fi
python3 -I hproxy/radls.py $O >> radls.jsonl
echo "blk.0.ffn_down.weight $O" > evq/mixes/bf16test-sc4.00-e4-L0down.mix
echo "bf16test-sc4.00-e4-L0down ${RXM_MODELS}/q38-27b-sc4.00-e4.rad ${RXM_WORK}/evq/mixes/bf16test-sc4.00-e4-L0down.mix" >> evq/queue-cal.txt
echo "$(date +%H:%M) bf16 donor ready ($(du -h $O | cut -f1)), test mix queued" >> ourchain.log
