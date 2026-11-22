#!/bin/bash
# lrj218.sh (iceman /data/w218, after lr218.sh + lrb218.sh) -- judge read of the low-rank line searches' tune pick (the lowest tune KL among
# every non-control variant of runs/r0/tunelr and tunelrb; name lr-<variant>), then agg218.py -> runs/agg218.txt. Pick rule fixed before reading.
set -u
cd /data/w218; G=code/v3/qad218/gpu218.sh; PY=/data/venv-exl3/bin/python; C=code/v3/qad218
say() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $*" >> logs/driver.log; }
run() { local log=$1; shift; while true; do $G $log "$@"; rc=$?; case $rc in 75|76) sleep 30;; *) return $rc;; esac; done; }
while pgrep -f "^/bin/bash code/v3/qad218/[l]rb\?218.sh" > /dev/null; do sleep 30; done
grep -q "lrb done" logs/driver.log || { say "lrj: lrb not done, no judge read"; exit 1; }
PICK=$($PY - <<'PEOF'
import json, glob
v = [json.load(open(f)) for f in glob.glob("runs/r0/tunelr*/*.json")]
v = [x for x in v if x["name"] not in ("cur", "o0")]
b = min(v, key=lambda x: x["kl"]); print(f"lr-{b['name']}={b['codes']}")
PEOF
)
say "lrj: judge $PICK"; run logs/judge.log $PY $C/probe218.py eval runs/judge --set judge --variants "$PICK" || exit 1
$PY $C/agg218.py runs --json runs/agg218.json > runs/agg218.txt 2>&1
say "lr all done (judge $PICK; runs/agg218.txt)"
