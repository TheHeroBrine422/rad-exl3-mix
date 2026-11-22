#!/bin/bash
# ls218.sh (iceman /data/w218, #218 redesign 11-21) -- the fine line search at round 0 with the per-tensor normalized step (--step quant:
# the step's rms = eta x the tensor's quantization step), on round 0's merged 300k-row gradient at the released codes; tune eval of
# cur + every eta. Also: the encoder on the true weight of 23.self_attn.k_proj (reencode_fixed of W_hat_orig at the released scales).
set -u
cd /data/w218; G=code/v3/qad218/gpu218.sh; PY=/data/venv-exl3/bin/python; C=code/v3/qad218
ETAS=${ETAS:-0.005,0.01,0.02,0.05,0.1,0.2}
say() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $*" >> logs/driver.log; }
run() { local log=$1; shift; while true; do $G $log "$@"; rc=$?; case $rc in 75|76) sleep 30;; *) return $rc;; esac; done; }
[ -f runs/diag/orig/out.json ] || { run logs/diag.log bin/exl3v3_tool reencode --list runs/diag/orig/list.txt --hess /data/challenge/hessians/calib
  grep -h '^{' logs/diag.log | tail -1 > runs/diag/orig/out.json; }
say "ls218: encode --step quant $ETAS"
run logs/lsenc.log $PY $C/encode218.py runs/r0/encq --grad runs/r0/enc/grad --step quant --etas $ETAS || exit 1
V="cur=rel"; for e in ${ETAS//,/ }; do V="$V,q$e=runs/r0/encq/eta-$e"; done
say "ls218: tune"
run logs/lstune.log $PY $C/probe218.py eval runs/r0/tuneq --set tune --variants "$V" || exit 1
say "ls218 done"
