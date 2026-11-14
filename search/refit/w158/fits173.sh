#!/bin/bash
# fits173.sh STAGE OBS.. -- (w158, RULES-173-03) the colo fits, one single-threaded process each, nice 19, in parallel; outputs in
# the current dir (menu173m.json, models173.json there). Restartable (finished outputs skipped).
#   w1      full fits + 5-fold CV of M1m-out / M1m-in / M1m-out-noV2 / M1m-in-noV2 on the given obs (wave 1)
#   sub1 M  sub-1 rule for the adopted model M: fits + curves of F1 = M, F2 = M + sat, F3 = M + lin2; CV on the S1 / S2 rows
set -u; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
C=/data/work/w158/code; ST=$1; shift
job() { local o=$1; shift; [ -s $o ] && return 0; nice -n 19 python3 $C/fin158.py "$@" > $o.log 2>&1 && echo "$o done" || echo "$o FAILED"; }
case $ST in
w1)
  for m in M1m-out M1m-in M1m-out-noV2 M1m-in-noV2; do
    job fit-$m.json fit fit-$m.json menu173m.json models173.json $m --obs "$@" &
    job cv-$m.json cv cv-$m.json menu173m.json models173.json $m --obs "$@" &
  done; wait ;;
sub1)
  m=$1; shift
  python3 - $m <<'EOF'
import json, sys
d = json.load(open("models173.json")); m = sys.argv[1]; b = d[m]
d[f"{m}-F2"] = {"ext": dict(b["ext"]), "opts": {**b["opts"], "sat": True}}
d[f"{m}-F3"] = {"ext": {**b["ext"], "lin2": True}, "opts": dict(b["opts"])}
json.dump(d, open("models173.json", "w"), indent=1)
EOF
  for f in "" -F2 -F3; do
    ( job fit-$m$f-w12.json fit fit-$m$f-w12.json menu173m.json models173.json $m$f --obs "$@"
      job curve-$m$f-w12.json curve curve-$m$f-w12.json menu173m.json models173.json $m$f fit-$m$f-w12.json --obs "$@" ) &
    job cvsub1-$m$f.json cv cvsub1-$m$f.json menu173m.json models173.json $m$f --obs "$@" --on S1,S2 &
  done; wait ;;
esac
