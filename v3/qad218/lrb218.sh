#!/bin/bash
# lrb218.sh (iceman /data/w218, after lr218.sh) -- the low-rank direction D_4 applied to the RELEASED codes (re-encode of released + step, no
# --from-orig): few tiles change, so there is no fresh-encode noise; tune eval vs cur.
set -u
cd /data/w218; G=code/v3/qad218/gpu218.sh; PY=/data/venv-exl3/bin/python; C=code/v3/qad218; R=runs/r0
ETAS=${ETAS:-0.2,0.5,1,2}
say() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $*" >> logs/driver.log; }
run() { local log=$1; shift; while true; do $G $log "$@"; rc=$?; case $rc in 75|76) sleep 30;; *) return $rc;; esac; done; }
while pgrep -f "^/bin/bash code/v3/qad218/[l]r218.sh" > /dev/null; do sleep 30; done
say "lrb: encode released + D4 $ETAS"; run logs/lr.log $PY $C/encode218.py $R/enclr4b --grad $R/lr4 --step quant --etas $ETAS || exit 1
V="cur=rel"; for e in ${ETAS//,/ }; do V="$V,c$e=$R/enclr4b/eta-$e"; done
say "lrb: tune"; run logs/lr.log $PY $C/probe218.py eval $R/tunelrb --set tune --variants "$V" || exit 1
say "lrb done"
