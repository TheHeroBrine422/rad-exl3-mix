#!/bin/bash
# stageC3.sh -- (ds8) #177-03 stage C, third schedule (lead note 11-14 2:10 AM CST: quant is the bottleneck now). As stageC2.sh
# (two quant lanes HIP 0-2 / 3-4, each claiming the next level and building it with its own rxrun.run; the swj loops swjC.sh on
# HIP 5 / 6) plus:
#   - order: levels already started first (job outputs present), then menu-critical: K2.0-5.0, then 5.25-7.5, then 1.0-1.875,
#     then sub-1 (each high K first inside the last two);
#   - an opportunistic third lane on HIP 5-6: it takes a level only when no swj pass runs or waits and no quantized level is
#     waiting for swj; it holds card5.lock + card6.lock for that whole level (its rxrun pool uses private lock files), so a
#     swj pass that lands meanwhile waits for the level's end (<= ~40 min) instead of stalling a lane's tail; then the cards go
#     back to swj.
# status-c.txt and the end (QUANT-DONE, wait for the swj loops, "STAGE C DONE" for after.sh) as stageC2.sh. Restartable;
# stop: touch /data/q27/mad/STOP. Process names `bash /data/q27/mad/tools/stageC3.sh [lane CARDS | lane3]`. (w151)
set -u; Q=/data/q27; T=$Q/mad/tools; R=$Q/ladder-mad; K=$Q/mad/conf; PY=$Q/venv/bin/python; LG=$Q/mad/logs/stageC.log
LANES=${LANES:-"0,1,2 3,4"}
ORDER="2 2.125 2.25 2.375 2.625 2.75 2.875 3 3.125 3.25 3.375 3.5 3.625 3.75 3.875 4.125 4.25 4.375 4.5 4.625 4.75 4.875 5
7.5 7 6.75 6.5 6.25 5.75 5.5 5.25
1.875 1.75 1.625 1.5 1.375 1.25 1.125 1
0.875 0.75 0.625 0.5 0.375 0.25 0.125"
say() { echo "$(date -u +%T) $*" >> $LG; }
nm() { echo q27v3-635bac02-k$1-mad; }
status() {
  local nq=0 ns=0 nn=0 k
  for k in $ORDER; do [ -f $R/$(nm $k)/DONE ] && nq=$((nq + 1)); done
  for k in $ORDER; do awk "BEGIN {exit !($k <= 4.5)}" && { nn=$((nn + 1)); grep -q '"scales": "' $R/$(nm $k)/DONE 2>/dev/null && ns=$((ns + 1)); }; done
  $PY - $nq $ns $nn $Q/mad/logs/lanes.jsonl > $R/status-c.txt.part <<'EOF' && mv $R/status-c.txt.part $R/status-c.txt
import json, os, sys, time
nq, ns, nn, p = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
w = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
cst = lambda t: time.strftime("%m-%d %H:%M", time.gmtime(t - 6 * 3600)) + " CST"
cs = sum(x["s"] * len(x["cards"]) for x in w[-8:]) / len(w[-8:]) if w else 3600.0   # card-s a level, last 8 levels
qh = (46 - nq) * cs / 3600 / 6                                   # 5 lane cards + HIP 5-6 about half the time
sh = (nn - ns) / 2 * 0.85 / 2
eta = time.time() + max(qh, sh) * 3600 + (0.85 * 3600 if nn - ns else 0)
print(f"{cst(time.time())} | stage C quant {nq}/46 levels (lanes HIP 0-2, 3-4, + 5-6 when swj is idle) | swj {ns}/{nn} levels "
      f"(HIP 5-6) | ETA ~{cst(eta)} (quant {qh:.1f} h, swj {sh:.1f} h + last pass)")
EOF
}
next_level() {   # claim: first a started level, then ORDER; prints the level name
  local k t
  for k in $ORDER; do t=$(nm $k); [ -f $R/$t/DONE ] || [ -d $R/swjclaim/$t ] && continue
    [ -n "$(ls $R/$t/out 2>/dev/null)" ] && mkdir $R/qclaim/$t 2>/dev/null && { echo $t; return; }; done
  for k in $ORDER; do t=$(nm $k); [ -f $R/$t/DONE ] || [ -d $R/swjclaim/$t ] && continue
    mkdir $R/qclaim/$t 2>/dev/null && { echo $t; return; }; done
}
build() {   # build LEVEL CARDS [PRIVATE_LOCKS]
  local n=$1 c=$2 k a t
  k=${n#q27v3-635bac02-k}; k=${k%-mad}
  $PY $T/mkconf.py $K/q-$k.json --root $R --cards $c --levels $k-mad > /dev/null
  [ -n "${3:-}" ] && $PY -c "import json,sys; p=sys.argv[1]; c=json.load(open(p))
for d in c['devices']: d['lock'] = '/data/q27/mad/l3/card%d.lock' % d['device']
json.dump(c, open(p, 'w'), indent=1)" $K/q-$k.json
  t=$(date +%s); say "lane $c: $n start"
  for a in 1 2 3 4; do
    env PYTHONPATH=$Q/stage/code/pipeline $PY -m rxrun.run $K/q-$k.json >> $Q/mad/logs/q-$k.log 2>&1 && break
    say "lane $c: $n attempt $a failed"; [ -f $Q/mad/STOP ] && return; sleep 30
  done
  if [ -f $R/$n/DONE ]; then echo "{\"level\": \"$n\", \"cards\": [$c], \"s\": $(( $(date +%s) - t ))}" >> $Q/mad/logs/lanes.jsonl
    say "lane $c: $n DONE in $(( ($(date +%s) - t) / 60 )) min"
  else say "lane $c: $n NOT DONE after 4 attempts (claim kept)"; fi
  bash $T/trim.sh $R >> $LG 2>&1; status
}
lane() { local n; while [ ! -f $Q/mad/STOP ]; do n=$(next_level); [ -z "$n" ] && break; build $n $1; done; }
swj_waiting() {   # a swj pass runs / waits, or a quantized level waits for its pass
  pgrep -f "rxrun.run /data/q27/mad/conf/swj-" > /dev/null && return 0
  local D K
  for D in $R/q27v3-*-mad; do [ -f $D/DONE ] && [ ! -d $R/swjclaim/$(basename $D) ] || continue; K=${D##*-k}; K=${K%-mad}
    $PY -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if float(sys.argv[2]) <= 4.5 and not d.get('scales') else 1)" $D/DONE $K && return 0
  done; return 1
}
lane3() {
  local n; mkdir -p /data/q27/mad/l3
  while [ ! -f $Q/mad/STOP ]; do
    if swj_waiting; then sleep 60; continue; fi
    n=$(next_level); [ -z "$n" ] && break
    exec 5>$Q/locks/card5.lock 6>$Q/locks/card6.lock; flock 5; flock 6
    build $n 5,6 private
    exec 5>&- 6>&-
  done
}
case "${1:-}" in lane) lane $2; exit 0;; lane3) lane3; exit 0;; esac
mkdir -p $R/qclaim $K; rm -f $R/QUANT-DONE
for d in $R/qclaim/*; do [ -d "$d" ] && [ ! -f $R/$(basename $d)/DONE ] && [ ! -d $R/swjclaim/$(basename $d) ] && rmdir $d; done   # no lane runs now
for c in 5 6; do pgrep -f "^bash $T/swjC.sh $c\$" > /dev/null || { (setsid nohup bash $T/swjC.sh $c > /dev/null 2>&1 < /dev/null &); say "swj loop on card $c"; }; done
say "stageC3: lanes $LANES + lane3 HIP 5,6 (when swj idle); order menu-critical first"; status
for c in $LANES; do bash $T/stageC3.sh lane $c & done; bash $T/stageC3.sh lane3 & wait
[ -f $Q/mad/STOP ] && { say "STOP"; exit 2; }
touch $R/QUANT-DONE; say "QUANT-DONE"; status
while pgrep -f "^bash $T/swjC.sh" > /dev/null; do sleep 120; bash $T/trim.sh $R >> $LG 2>&1; status; done
bash $T/trim.sh $R >> $LG 2>&1; status; say "STAGE C DONE"
