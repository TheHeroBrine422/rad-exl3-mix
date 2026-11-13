#!/bin/bash
# host144.sh -- (w144, #174-02 amendment 2, ds8) the host-candidate arms: K8 mul1 donor, int8-row reference, then the
# three A8 arms on the 2.5 incumbent. Restartable (every step skips finished output).
set -u
Q=/data/q27 W=/data/q27/w144 D=/data/q27/emb/trellis; cd $W
echo "t-m1 8 mul1" > $W/list-k8.txt
bash $W/donors144all.sh $W/list-k8.txt
if [ ! -f $D/emb-i8row.rad ]; then
  B0=$(ls $Q/packs/q27v3-144dd6e1-k0.125-i8-s*.rad | head -n 1); rm -f $D/emb-i8row.rad.part
  flock $Q/locks/conv.lock flock -s $Q/locks/ram.lock cp $B0 $D/emb-i8row.rad.part &&
  nice -n 10 $Q/venv/bin/python -I $W/embrtn144.py $D/emb-i8row.rad.part && mv $D/emb-i8row.rad.part $D/emb-i8row.rad
fi
I=v3s171-2.5-explore-00419
bash $W/a8anchor144.sh $I:bf16 $I:t8m $I:i8row
