#!/bin/bash
# ls2218.sh (iceman /data/w218, #218, after ls218.sh) -- (a) the preconditioned direction M_d^-1 G (precond218.py) on round 0's merged gradient,
# the same normalized-step line search; (b) fresh encodes of the true weight (--from-orig): eta 0 (the encode-to-encode noise floor) and
# the true weight minus the preconditioned step. Tune eval of each.
set -u
cd /data/w218; G=code/v3/qad218/gpu218.sh; PY=/data/venv-exl3/bin/python; C=code/v3/qad218
ETAS=${ETAS:-0.005,0.01,0.02,0.05,0.1,0.2}; OETAS=${OETAS:-0,0.05,0.2}
say() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $*" >> logs/driver.log; }
run() { local log=$1; shift; while true; do $G $log "$@"; rc=$?; case $rc in 75|76) sleep 30;; *) return $rc;; esac; done; }
while pgrep -f "^/bin/bash code/v3/qad218/[l]s218.sh" > /dev/null; do sleep 30; done
[ -f runs/r0/gradp/grad.json ] || { say "ls2: precond"; run logs/ls2.log $PY $C/precond218.py runs/r0/enc/grad runs/r0/gradp || exit 1; }
say "ls2: encode precond $ETAS"; run logs/ls2.log $PY $C/encode218.py runs/r0/encp --grad runs/r0/gradp --step quant --etas $ETAS || exit 1
say "ls2: encode from orig $OETAS"; run logs/ls2.log $PY $C/encode218.py runs/r0/enco --grad runs/r0/gradp --step quant --from-orig --etas $OETAS || exit 1
V="cur=rel"; for e in ${ETAS//,/ }; do V="$V,p$e=runs/r0/encp/eta-$e"; done; for e in ${OETAS//,/ }; do V="$V,o$e=runs/r0/enco/eta-$e"; done
say "ls2: tune"; run logs/ls2.log $PY $C/probe218.py eval runs/r0/tunep --set tune --variants "$V" || exit 1
say "ls2 done"
