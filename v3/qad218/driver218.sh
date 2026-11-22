#!/bin/bash
# driver218.sh [ROUNDS] (iceman /data/w218, card 0) -- #218's whole probe, restartable (every step skipped by its output):
# per round r: grad over chunks CPR r .. CPR r + CPR - 1 (100k gen rows each, at the current codes; merged rows-weighted) -> encode at ETAS -> tune eval of cur + every eta -> pick -> judge eval of the pick;
# then the scale-only comparison on the same documents. Pick rule (fixed before any read): the eta of the lowest tune KL;
# if no eta beats cur on tune, the round keeps cur (its codes do not change). Log logs/driver.log.
set -u
cd /data/w218
R=${1:-3}; ETAS=${ETAS:-0.01,0.03,0.1,0.3}; CPR=${CPR:-3}
G=code/v3/qad218/gpu218.sh; PY=/data/venv-exl3/bin/python; C=code/v3/qad218
say() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $*" >> logs/driver.log; }
run() { local log=$1; shift; while true; do $G $log "$@"; rc=$?; case $rc in 75|76) sleep 60;; *) return $rc;; esac; done; }
CUR=""
for r in $(seq 0 $((R - 1))); do
  D=runs/r$r
  [ $r -gt 0 ] && [ -f runs/r$((r - 1))/pick.txt ] && CUR=$(cat runs/r$((r - 1))/pick.txt)     # a restart picks up the codes so far
  GD=""
  for j in $(seq 0 $((CPR - 1))); do
    c=$((CPR * r + j)); GD="$GD${GD:+,}$D/g$j"
    [ -f $D/g$j/grad.json ] || { say "r$r grad chunk $c"; run logs/r${r}grad.log $PY $C/probe218.py grad $D/g$j --chunk $c --codes "$CUR" || exit 1; }
  done
  E=$ETAS; [ $r = 0 ] && E=0,$ETAS
  say "r$r encode $E"; run logs/r${r}enc.log $PY $C/encode218.py $D/enc --grad $GD --base "$CUR" --etas $E || exit 1
  V="cur=${CUR:-rel}"; for e in ${ETAS//,/ }; do V="$V,e$e=$D/enc/eta-$e"; done
  say "r$r tune"; run logs/r${r}tune.log $PY $C/probe218.py eval $D/tune --set tune --variants "$V" || exit 1
  PICK=$($PY - "$D/tune" <<'EOF'
import json, sys, glob, os
d = sys.argv[1]; v = {json.load(open(f))["name"]: json.load(open(f)) for f in glob.glob(os.path.join(d, "*.json"))}
best = min(v, key=lambda k: v[k]["kl"])
print("" if best == "cur" else v[best]["codes"])
EOF
)
  if [ -n "$PICK" ]; then CUR=$PICK; fi
  echo "${CUR}" > $D/pick.txt
  say "r$r pick: ${PICK:-cur (no eta beat cur)} -> codes ${CUR:-rel}"
  J="r$r=${CUR:-rel}"; [ $r = 0 ] && J="rel=rel,$J"
  say "r$r judge"; run logs/judge.log $PY $C/probe218.py eval runs/judge --set judge --variants "$J" || exit 1
done
[ -f runs/scale/scale.json ] || { say "scale-only"; run logs/scale.log $PY $C/probe218.py scale runs/scale --chunks 0-$((CPR * R - 1)) --compile-scales || exit 1; }
say "driver done"
