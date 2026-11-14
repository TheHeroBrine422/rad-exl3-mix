#!/bin/bash
# stageC2.sh -- (ds8) #177-03 stage C, rescheduled (lead note 11-13 9:40 PM CST: stageC.sh stalled because its quant run
# shared card 5 with a swj pass). Quant and swj never share a card now: two quant LANES (default "0,1,2" and "3,4") each
# take the next unclaimed level in the stage-C order (G1 K2.625-4.5, G2 K0.125-2.375, G3 K4.625-7.5) and build it with its
# own rxrun.run (steps jobs / assemble / post / recipe, that level only, that lane's cards), so two levels run at once and
# a level's tail idles at most one lane; the swj loops (swjC.sh) keep HIP 5 and 6 to themselves. Claims: ROOT/qclaim/<level>
# (cleared for unfinished levels when stageC2 starts: no lane runs then). Writes ROOT/status-c.txt (levels done, swj done,
# ETA) after every level, ROOT/QUANT-DONE at the end, then waits for the swj loops and logs "STAGE C DONE" (after.sh starts
# on it). Restartable; stop: touch /data/q27/mad/STOP. Process names `bash /data/q27/mad/tools/stageC2.sh`, `... lane N`. (w151)
set -u; Q=/data/q27; T=$Q/mad/tools; R=$Q/ladder-mad; K=$Q/mad/conf; PY=$Q/venv/bin/python; LG=$Q/mad/logs/stageC.log
LANES=${LANES:-"0,1,2 3,4"}; SC=${SWJCARDS:-5 6}
ORDER="2.625 2.75 2.875 3 3.125 3.25 3.375 3.5 3.625 3.75 3.875 4.125 4.25 4.375 4.5
0.125 0.25 0.375 0.5 0.625 0.75 0.875 1 1.125 1.25 1.375 1.5 1.625 1.75 1.875 2 2.125 2.25 2.375
4.625 4.75 4.875 5 5.25 5.5 5.75 6.25 6.5 6.75 7 7.5"
say() { echo "$(date -u +%T) $*" >> $LG; }
nm() { echo q27v3-635bac02-k$1-mad; }
status() {   # one line: quant levels done (of the 46), swj levels done, ETA from this run's measured level walls
  local nq=0 ns=0 nn=0 k
  for k in $ORDER; do [ -f $R/$(nm $k)/DONE ] && nq=$((nq + 1)); done
  for k in $ORDER; do awk "BEGIN {exit !($k <= 4.5)}" && { nn=$((nn + 1)); grep -q '"scales": "' $R/$(nm $k)/DONE 2>/dev/null && ns=$((ns + 1)); }; done
  $PY - $nq $ns $nn $Q/mad/logs/lanes.jsonl > $R/status-c.txt.part <<'EOF' && mv $R/status-c.txt.part $R/status-c.txt
import json, os, sys, time
nq, ns, nn, p = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
w = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
cst = lambda t: time.strftime("%m-%d %H:%M", time.gmtime(t - 6 * 3600)) + " CST"
cs = sum(x["s"] * len(x["cards"]) for x in w[-8:]) / len(w[-8:]) if w else 3600.0   # card-s a level (prior: stage B, 3 levels in 25 min on 7 cards)
qh = (46 - nq) * cs / 3600 / 5                                   # on the 5 quant cards
sh = (nn - ns) / 2 * 0.85 / 2                                    # pairs x 0.85 h on 2 swj cards
eta = time.time() + max(qh, sh) * 3600 + (0.85 * 3600 if nn - ns else 0)
print(f"{cst(time.time())} | stage C quant {nq}/46 levels (2 lanes, HIP 0-4) | swj {ns}/{nn} levels (HIP 5-6) | "
      f"ETA ~{cst(eta)} (quant {qh:.1f} h, swj {sh:.1f} h + last pass)")
EOF
}
lane() {   # lane CARDS
  local c=$1 k n a t
  while [ ! -f $Q/mad/STOP ]; do
    n=""; for k in $ORDER; do t=$(nm $k); [ -f $R/$t/DONE ] || [ -d $R/swjclaim/$t ] && continue; mkdir $R/qclaim/$t 2>/dev/null && { n=$t; break; }; done
    [ -z "$n" ] && break
    k=${n#q27v3-635bac02-k}; k=${k%-mad}
    $PY $T/mkconf.py $K/q-$k.json --root $R --cards $c --levels $k-mad > /dev/null
    t=$(date +%s); say "lane $c: $n start"
    for a in 1 2 3 4; do
      env PYTHONPATH=$Q/stage/code/pipeline $PY -m rxrun.run $K/q-$k.json >> $Q/mad/logs/q-$k.log 2>&1 && break
      say "lane $c: $n attempt $a failed"; [ -f $Q/mad/STOP ] && return; sleep 30
    done
    if [ -f $R/$n/DONE ]; then echo "{\"level\": \"$n\", \"cards\": \"$c\", \"s\": $(( $(date +%s) - t ))}" | sed 's/"cards": "\([^"]*\)"/"cards": [\1]/' >> $Q/mad/logs/lanes.jsonl
      say "lane $c: $n DONE in $(( ($(date +%s) - t) / 60 )) min"
    else say "lane $c: $n NOT DONE after 4 attempts (claim kept)"; fi
    bash $T/trim.sh $R >> $LG 2>&1; status
  done
}
if [ "${1:-}" = lane ]; then lane $2; exit 0; fi
mkdir -p $R/qclaim $K; rm -f $R/QUANT-DONE
for d in $R/qclaim/*; do [ -d "$d" ] && [ ! -f $R/$(basename $d)/DONE ] && rmdir $d; done     # stale claims (no lane runs now)
for c in $SC; do pgrep -f "^bash $T/swjC.sh $c\$" > /dev/null || { (setsid nohup bash $T/swjC.sh $c > /dev/null 2>&1 < /dev/null &); say "swj loop on card $c"; }; done
say "stageC2: lanes $LANES, swj on $SC"; status
for c in $LANES; do bash $T/stageC2.sh lane $c & done; wait
[ -f $Q/mad/STOP ] && { say "STOP"; exit 2; }
touch $R/QUANT-DONE; say "QUANT-DONE"; status
while pgrep -f "^bash $T/swjC.sh" > /dev/null; do sleep 120; bash $T/trim.sh $R >> $LG 2>&1; status; done
bash $T/trim.sh $R >> $LG 2>&1; status; say "STAGE C DONE"
