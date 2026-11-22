#!/bin/bash
# post218.sh (iceman /data/w218, after driver218.sh) -- the readout and the one served read: agg218.py -> runs/agg218.json; the best
# code-QAD round = the lowest judge KL among r0..r2; if it is below rel's, swap its records into a copy of the released container
# (swap218.py) and score rel + best on j214 (score218.sh, card 0 lock), then cmp214.py. Restartable (outputs skip steps).
set -u
cd /data/w218; PY=/data/venv-exl3/bin/python; C=code/v3/qad218
say() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $*" >> logs/driver.log; }
REL=rel/host-emb/qwen3.8-27b-exl3-v3-2.50bpw.rad
$PY $C/agg218.py runs --json runs/agg218.json > runs/agg218.txt 2>&1; say "post: agg218 written"
BEST=$($PY - <<'EOF'
import json, glob, os
v = {json.load(open(f))["name"]: json.load(open(f)) for f in glob.glob("runs/judge/*.json")}
r = {k: x for k, x in v.items() if k.startswith("r")}
b = min(r, key=lambda k: r[k]["kl"]) if r else None
print(r[b]["codes"] if b and r[b]["kl"] < v["rel"]["kl"] and r[b]["codes"] not in ("", "rel") else "")
EOF
)
[ -n "$BEST" ] || { say "post: no round beats rel on judge: no served read"; exit 0; }
grep -q "RADBUILD218 DONE" logs/radbuild.log || { say "post: radiance build missing"; exit 1; }
[ -f best.rad ] || $PY $C/swap218.py $REL lv/2.50 $BEST best.rad >> logs/post.log 2>&1 || { say "post: swap failed"; exit 1; }
for x in "rel $REL" "best best.rad"; do
  set -- $x
  while true; do
    exec {fd}>/root/card0.lock; flock -n $fd && break; exec {fd}>&-; sleep 60
  done
  $C/score218.sh $1 $2 >> logs/post.log 2>&1; exec {fd}>&-
done
$PY code/v3/qad214/cmp214.py slices/golden-j214 scores/rel.json.rows scores/best.json.rows --json scores/cmp218.json >> logs/post.log 2>&1
say "post: served read done ($BEST): $(tail -1 logs/post.log | cut -c1-200)"
