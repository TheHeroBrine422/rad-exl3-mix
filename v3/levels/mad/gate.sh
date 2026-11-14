#!/bin/bash
# gate.sh [CARDS] -- (ds8) #177-03 gate: the merged tree's K4.125 mad jobs for layers 2-3 (one GDN, one attention layer;
# 8 group jobs, H_out, tile CD, C9) vs w139's q27v3-144dd6e1-k4.125-hash jobs (box2 /data/q27/w139/s3/..., pulled over
# the 10 GbE link): rxrun.cmp --tensors (tensor bytes; the stamps differ by design) must print CMP SAME. Takes the first
# free card of CARDS (default "7 6 5 4"). Restartable. Out /data/q27/mad/gate (gate.log, GATE.ok | GATE.fail)
set -u; Q=/data/q27; G=$Q/mad/gate; T=$Q/stage/code/v3/levels/mad; mkdir -p $G; exec >> $G/gate.log 2>&1
say() { echo "$(date -u +%T) $*"; }
source $Q/tools/env-ds8.sh
REF=/data/q27/w139/s3/q27v3-144dd6e1-k4.125-hash/out
if [ ! -d $G/ref ]; then
  mkdir -p $G/ref.part
  for j in L02-g0 L02-g1 L02-g2 L02-g3 L03-g0 L03-g1 L03-g2 L03-g3; do
    rsync -a -e "ssh -i /root/.ssh/dbox -o StrictHostKeyChecking=no" root@10.29.6.6:$REF/$j $G/ref.part/ || { say "pull $j FAIL"; exit 1; }
  done; mv $G/ref.part $G/ref; say "ref pulled: $(ls $G/ref | wc -l) jobs"
fi
for c in ${1:-7 6 5 4}; do flock -n $Q/locks/card$c.lock true && { CARD=$c; break; }; done
[ -n "${CARD:-}" ] || { say "no free card"; exit 1; }
$PY $T/mkconf.py $G/conf.json --root $G/ladder --cards $CARD --steps jobs --levels 4.125-mad --layers 2-3 --dup 0
say "jobs on card $CARD"
env PYTHONPATH=$Q/stage/code/pipeline $PY -m rxrun.run $G/conf.json >> $G/run.log 2>&1 || { say "rxrun.run FAIL"; touch $G/GATE.fail; exit 1; }
L=$(ls -d $G/ladder/q27v3-*-k4.125-mad); say "level $L"
env PYTHONPATH=$Q/stage/code/pipeline $PY -m rxrun.cmp $G/ref $L/out --tensors > $G/cmp.txt 2>&1
tail -n 3 $G/cmp.txt
tail -n 1 $G/cmp.txt | grep -q "CMP SAME" && { touch $G/GATE.ok; say "GATE PASS"; } || { touch $G/GATE.fail; say "GATE FAIL"; }
