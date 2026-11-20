#!/bin/bash
# run.sh BINDIR OUT.jsonl LOCK [cb208 args ..] -- every candidate, one at a time, each under the card lock.
# A candidate that already has a "done" line in OUT.done is skipped (rerun-safe).
set -u
B=$1; O=$2; L=$3; shift 3
ALL=${CANDS:-"Raw Base Mad MadF Mad4 Mad4F Mul1I8 Int4Grid Qtip1Mad Qtip3Inst Cptc6Lds Cptc6Glb Cptc6Eq3 Cptc6FLds Cptc6FGlb Cptc5Lds Cptc5Glb Cptc5Reg St6Lds St6Glb St8Lds St8Glb St10Lds St10Glb St12Lds St12Glb CptcMin Lc1Lds Lc1Glb"}
touch $O.done
for c in $ALL; do
  grep -qx "$c" $O.done && continue
  rm -f $O.$c
  x=""; [ $c = Raw ] && x="--ctx 1"
  if flock $L $B/cb208_$c --out $O.$c $x "$@" > $O.$c.log 2>&1; then cat $O.$c >> $O; echo $c >> $O.done; echo "ok $c"
  else echo "FAIL $c: $(tail -1 $O.$c.log)"; fi
done
echo ALLDONE
