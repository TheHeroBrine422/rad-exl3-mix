#!/bin/bash
# chain233c.sh (box2; setsid nohup) -- w233 GPU block 1 (replaces chain233a/b): (1) prof1: prof233 on card 2 (one-card segment profile, cap 8192);
# (2) the finals-rank plan fin12 (plan231: V 12 = one finals rank on one card pair; calib 8k 0.4 / tset1 branch sessions 0.3 / xg-scr1 0.3 / 2 deep
# traces of ~250k started at step 0) on cards 2,3 in the finals flags, three windows: W0 steps 1-8 (fresh start), W1 resumed at step 120 (traces at
# ~120k), W2 resumed at step 200 (~200k); each window: one step under torch.profiler (GPU busy per card), py-spy over the next two (GIL share).
# The chain holds both card locks for all of it (W233_LOCKED=1 for run233.sh: no gaps between windows) and logs its release to cards.log.
set -u
W=/data/q27/w233; C=$W/code/v3/qad233; PY=/data/q27/venv/bin/python
log() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M:%S %p CST') $*" | tee -a $W/run233.log; }
log "chain233c waiting for cards 2,3 (one holder for every window)"
for c in 2 3; do eval "exec {l}>/data/q27/locks/card$c.lock"; flock $l; done
export W233_LOCKED=1; log "chain233c holds cards 2,3"
[ -e $W/runs/prof1/prof233.json ] || $C/run233.sh 2 prof1 prof233.py --budgets 2e6 --cap 8192 --compile-scales --compile-ew --a8-int8 --dynamo-limit 64
echo '[0, 1]' > $W/tr2.json
[ -e $W/plan-fin12.json ] || (cd $W && PYTHONPATH=/data/q27/w214/pylib-fla $PY $W/code/v3/qad231/plan231.py $C/specs/fin12.json $W/plan-fin12.json --steps-tsv $W/steps-fin12.tsv > $W/plan-fin12.out 2>&1) || { log "plan fin12 failed"; exit 1; }
FL="--budgets 3e6 --lr 3e-4 --compile-scales --compile-ew --a8-int8 --dynamo-limit 64 --cap 262144 --V 12 --plan231 $W/plan-fin12.json --split-layers 32 --pipeline \
 --attn-gqa --state-card-gb 4 --state-budget-gb 8 --card-mem-gb 28.5 --pipe-xfer host --snap-at none"
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:256
win() {   # win NAME FROM_STEP: resume a copy of W0's state at FROM_STEP (0 = fresh), run 8 steps, profile step FROM+4, py-spy steps FROM+5..6
  local N=$1 S=$2 OUT=$W/runs/$1
  [ -e $OUT/prof233-step$((S + 4)).json ] && { log "$N done before"; return; }
  rm -rf $OUT; mkdir -p $OUT
  [ $S -gt 0 ] && $PY $C/mkstate233.py $W/runs/fin-w0/state.pt $S $OUT/state.pt
  W233_PROF_STEPS=$((S + 4)) $C/run233.sh 2,3 $N $C/pipe233.py $FL --max-steps $((S + 8)) &
  local R=$!
  until [ "$(grep -c '"what": "main"' $OUT/train.jsonl 2>/dev/null)" -ge 5 ]; do sleep 10; kill -0 $R 2>/dev/null || { wait $R; return; }; done
  local P=$(pgrep -f "^/data/q27/venv/bin/python $C/pipe233.py $OUT " | head -1)
  py-spy record -p $P -r 40 -d 50 --idle --threads -f raw -o $OUT/pyspy-all.txt --nonblocking > $OUT/pyspy-all.log 2>&1
  py-spy record -p $P -r 40 -d 50 --gil --threads -f raw -o $OUT/pyspy-gil.txt --nonblocking > $OUT/pyspy-gil.log 2>&1
  wait $R
}
win fin-w0 0
win fin-w1 120
win fin-w2 200
log "chain233c finished"; echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') w233 releases cards 2,3" >> $W/cards.log
