#!/bin/bash
# qimport.sh MACHINE -- (box2) results.tsv rows for the reads done outside the queue: w221's scores (L32, L48, A8-25 under its dispatchers) and w220's 25C control. Links their .json.rows into QROOT/scores
# (never the .json, so no queue job reads as done) so the paired-vs-control columns resolve; skips names already in results.tsv. Sizes / controls per arm below. Safe to rerun.
set -u
MACHINE=$1; HERE0=$(dirname "$(readlink -f "$0")"); source $HERE0/qenv.sh $MACHINE
declare -A SZ=([C30]=3.00 [L32-30]=3.00 [A8-30]=3.00) CT=([L32]=25C [L48]=25C [T128]=25C [CB]=25C [A8-25]=C25-3 [L32-30]=C30 [A8-30]=C30)
for spec in /data/q27/w221:w221 /data/q27/w220:w220; do
  D=${spec%%:*}
  for f in $D/scores/*.json.rows; do
    [ -e "$f" ] || continue
    N=$(basename $f .json.rows); ARM=${N%-ckpt-*}; CK=ckpt-${N##*-ckpt-}
    case $D/$ARM in /data/q27/w220/25C) ;; /data/q27/w220/*) continue;; esac       # w220: only the 25C control
    ln -sfn $f $QROOT/scores/$N.json.rows
    grep -q "	$N$" $QROOT/results.tsv 2>/dev/null && continue
    CTL=-; [ -n "${CT[$ARM]:-}" ] && CTL=${CT[$ARM]}-$CK
    $PY $HERE/qrow.py add $QROOT --machine "$MACHINE(import)" --card - --name $N --arm $ARM --ckpt $CK --size ${SZ[$ARM]:-2.50} --golden $J217 --ctldir $CTLDIR \
      --ctl $CTL --run $D/runs/$ARM
  done
done
