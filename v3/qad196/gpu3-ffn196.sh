#!/bin/bash
# gpu3-ffn196.sh (w182, #196): the GPU part of the un-shrink (ffn_down, the slow CPU type) on box2 card 3, then the
# identity check of the un-shrink path (export-only with c forced to 1, run196.sh's own card lock). Card 3 lock held
# for the GPU part only; restartable (finished per-type files are skipped).
cd /data/q27/w182 && source env.sh
flock -n /data/q27/locks/card3.lock -c "
  export ROCR_VISIBLE_DEVICES=3
  TYPES=ffn_down DEV=cuda:0 ./code/v3/qad196/ushr196.sh /data/q27/w170/vl/inc2.5m 25
  TYPES=ffn_down DEV=cuda:0 ./code/v3/qad196/ushr196.sh /data/q27/w182/vl/inc4.0m 40
" || { echo "card 3 busy"; exit 75; }
./code/v3/qad196/run196.sh ONE 3 --unshrink /data/q27/w182/data/one25.safetensors --export-only
echo "done $(date -u +%H:%M)"
