#!/bin/bash
# lr218.sh (iceman /data/w218, #218 follow-up 11-22) -- low-rank-projected code-QAD step at round 0: lowrank218.py (subspace from chunk g2,
# signal = mean of g0, g1 projected) -> fresh encodes of the true weight minus eta x delta x D / rms(D) (--from-orig --step quant) for r = 4, 16
# and the preconditioned M_d^-1 D_4 -> tune eval of cur (released), o0 (the eta-0 fresh encode) and every variant. Restartable.
set -u
cd /data/w218; G=code/v3/qad218/gpu218.sh; PY=/data/venv-exl3/bin/python; C=code/v3/qad218; R=runs/r0
ETAS=${ETAS:-0.05,0.1,0.2,0.5,1}; PETAS=${PETAS:-0.05,0.2,0.5,1}; O0=$R/enco/eta-0
say() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $*" >> logs/driver.log; }
run() { local log=$1; shift; while true; do $G $log "$@"; rc=$?; case $rc in 75|76) sleep 30;; *) return $rc;; esac; done; }
[ -f $R/lr16/grad.json ] || { say "lr: lowrank218"; run logs/lr.log $PY $C/lowrank218.py $R || exit 1; }
[ -f $R/lr4p/grad.json ] || { say "lr: precond lr4"; run logs/lr.log $PY $C/precond218.py $R/lr4 $R/lr4p || exit 1; }
for g in lr4 lr16; do say "lr: encode $g $ETAS"; run logs/lr.log $PY $C/encode218.py $R/enc$g --grad $R/$g --step quant --from-orig --cmp $O0 --etas $ETAS || exit 1; done
say "lr: encode lr4p $PETAS"; run logs/lr.log $PY $C/encode218.py $R/enclr4p --grad $R/lr4p --step quant --from-orig --cmp $O0 --etas $PETAS || exit 1
V="cur=rel,o0=$O0"
for e in ${ETAS//,/ }; do V="$V,a$e=$R/enclr4/eta-$e,b$e=$R/enclr16/eta-$e"; done
for e in ${PETAS//,/ }; do V="$V,p$e=$R/enclr4p/eta-$e"; done
say "lr: tune"; run logs/lr.log $PY $C/probe218.py eval $R/tunelr --set tune --variants "$V" || exit 1
say "lr done"
