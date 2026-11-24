#!/bin/bash
# stud228.sh (box2; setsid nohup, log /data/q27/onpol/w228/stud.log) -- w228 student step (lead 11-24, user approved): the same on-policy data
# written by the STUDENT (the released 2.50 container with I25-step-10M's 10M scales) instead of IQ6R-V4, then its bf16 goldens and two queue arms.
# Restartable: every step is skipped when its output exists (the generators skip finished cuts). Steps:
#   0. student container /data/q27/onpol/student10m.rad (swap222.py --verify, as qread.sh; CPU, runs at once)
#   1. wait for chain228 done + arm228.sh's release (q.log "arm228: v2: single-card jobs queued"): cards 1-3 are ours via only-card<N> = NONE
#      (no locks; a restart re-writes NONE and waits until no queue job holds the card's lock)
#   2. 3 servers, TP1 on cards 1, 2, 3 (ports 8031-8033): the queue's radiance build (qenv.sh RB, as qscore.sh), A8 as released, embedding-placement
#      ram, i8 KV, prefix cache, max-model-len 135168
#   3. EXACTLY the teacher batches' cuts (the cut ids gen228.py finished in out/, out-calib-chat/, out-calib-text/), same sampling, raw completions,
#      thinking included: three generators at once (6 + 2 + 2 slots a server), no token target (every cut)
#   4. servers stopped, only-card1 removed (card 1 back to the queue)
#   5. bf16 goldens TP2 on cards 2,3 (docs <= 8k): ext/gold/stud-b1 (traces) and ext/gold/stud-calib-b1 (calib chat + text), onpol-*'s layout
#      (gen.u8 all ones, corpus.meta.jsonl, genmask/<name>.u8), check228.py
#   6. cont files as arm228.sh (seeded permutation, rows >= 16 at cap 8192, until 1,552,745 rows): cont228/cont-stud-b1.json, cont-stud-calib-b1.json
#   7. jobs.txt lines (under queue.lock): T C228-stud / C228-studcal + their JC / plain / JL reads
#   8. only-card2 / only-card3 removed; "stud228 done"
# On any exit: the servers are stopped and only-card1..3 removed, so no card is stranded.
# PRIVATE (stud-b1 holds the user's sessions): box2 only.
set -u
Q=/data/q27; QROOT=$Q/queue; W=$Q/onpol/w228; E=$QROOT/ext; C=$W/code/v3/qad228; L=$W/stud.log
source $QROOT/code/v3/qadq/qenv.sh box2
log() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $*" >> $L; }
STUD=$Q/onpol/student10m.rad; CK=$QROOT/runs-ice/I25-step-10M/ckpt-1e+07
PIDS=""
cleanup() {
  for p in $PIDS; do kill $p 2>/dev/null; done
  rm -f $QROOT/only-card1 $QROOT/only-card2 $QROOT/only-card3
  log "stud228 exit (rc $1): servers stopped, only-card1..3 removed"
}
log "stud228 start (pid $$)"
cd $W
# ---- 0 (in the background: CPU work while the teacher goldens record)
if [ ! -e $STUD.ok ]; then
  ( while [ "$(df -BG --output=avail $Q | tail -1 | tr -dc 0-9)" -lt 120 ]; do sleep 120; done
    $PY $QROOT/code/v3/qadq/swap222.py $IN/rel/host-emb/qwen3.8-27b-exl3-v3-2.50bpw.rad $IN/lv/2.50 $CK $STUD.part --verify > $W/logs/swap-stud.log 2>&1 \
      && mv $STUD.part $STUD && touch $STUD.ok; log "student container rc $? ($(tail -n 1 $W/logs/swap-stud.log))" ) & SW=$!
fi
# ---- 1
until grep -q "chain228 done" $W/chain.log && grep -q "arm228: v2: single-card jobs queued" $QROOT/q.log; do sleep 60; done
trap 'cleanup $?' EXIT
for c in 1 2 3; do echo NONE > $QROOT/only-card$c; done
[ -n "${SW:-}" ] && wait $SW
[ -e $STUD.ok ] || { log "no student container: stop"; exit 1; }
for c in 1 2 3; do ( exec 9>$(CARDLOCK $c); flock 9 ); done              # a queue job that was already on the card finishes first
log "cards 1-3 ours (only-card NONE)"
# ---- 2 + 3
mk_cuts() {   # mk_cuts RENDER GENDIR OUT: the teacher's finished cuts of RENDER, for the student
  [ -e $3/cuts.jsonl ] && return
  mkdir -p $3; ln -sfn $(readlink -f $1/sess) $3/sess
  $PY -c "
import json, os, sys
done = {int(f[:-5]) for f in os.listdir('$2/gen') if f.endswith('.json')}
rows = [l for l in open('$1/cuts.jsonl') if json.loads(l)['cut'] in done]
open('$3/cuts.jsonl.part', 'w').writelines(rows); os.replace('$3/cuts.jsonl.part', '$3/cuts.jsonl'); print(len(rows), 'cuts')" >> $L 2>&1
}
mk_cuts $W/render-b $W/out $W/render-stud-trace
mk_cuts $W/render-calib-chat $W/out-calib-chat $W/render-stud-calib-chat
mk_cuts $W/render-calib-text $W/out-calib-text $W/render-stud-calib-text
if [ ! -e $W/stud-gen.done ]; then
  for c in 1 2 3; do
    cat > $W/stud$c.yaml <<YEOF
server:
  host: 127.0.0.1
  port: 803$c
memory:
  host-headroom-mib: 8192
models:
  stud:
    model: $STUD
    tp: 1
    kv-cache-dtype: i8
    max-model-len: 135168
    max-num-seqs: 10
    embedding-placement: ram
    checkpoint-slots: 16
    prefix-cache-mib: 6144
    mm-max-patches: 0
YEOF
    LD_LIBRARY_PATH="$RLIBS" RADIANCE_HOME=$RH ROCR_VISIBLE_DEVICES=$c $RB --config $W/stud$c.yaml > $W/logs/stud$c.log 2>&1 < /dev/null &
    PIDS="$PIDS $!"
  done
  for c in 1 2 3; do
    for i in $(seq 120); do curl -sf localhost:803$c/health > /dev/null && break; sleep 10; done
    curl -sf localhost:803$c/health > /dev/null || { log "student server card $c did not come up: $(tail -n 3 $W/logs/stud$c.log | tr '\n' ' ')"; exit 1; }
  done
  log "student servers up (cards 1-3, ports 8031-8033)"
  G="$PY -I $C/gen228.py"; P=8031,8032,8033; T=1000000000
  TZ=America/Chicago $G $W/render-stud-trace $W/out-stud-trace --ports $P --slots 6 --target $T > $W/logs/gen-stud-trace.out 2>&1 & g1=$!
  TZ=America/Chicago $G $W/render-stud-calib-chat $W/out-stud-calib-chat --ports $P --slots 2 --target $T > $W/logs/gen-stud-chat.out 2>&1 & g2=$!
  TZ=America/Chicago $G $W/render-stud-calib-text $W/out-stud-calib-text --ports $P --slots 2 --target $T > $W/logs/gen-stud-text.out 2>&1 & g3=$!
  wait $g1; r1=$?; wait $g2; r2=$?; wait $g3; r3=$?
  log "student generators rc $r1 / $r2 / $r3: trace $(tail -n 1 $W/out-stud-trace/gen.log) | chat $(tail -n 1 $W/out-stud-calib-chat/gen.log) | text $(tail -n 1 $W/out-stud-calib-text/gen.log)"
  [ $r1 = 0 ] && [ $r2 = 0 ] && [ $r3 = 0 ] || exit 1
  for n in trace calib-chat calib-text; do
    want=$(wc -l < $W/render-stud-$n/cuts.jsonl); have=$(ls $W/out-stud-$n/gen | grep -c '\.json$')
    [ "$want" = "$have" ] || { log "student $n: $have of $want cuts (failed requests): stop"; exit 1; }
  done
  touch $W/stud-gen.done
fi
for p in $PIDS; do kill $p 2>/dev/null; done; for p in $PIDS; do while kill -0 $p 2>/dev/null; do sleep 5; done; done; PIDS=""
rm -f $QROOT/only-card1; log "student servers stopped; card 1 back to the queue"
# ---- 5
record() {   # record NAME PARTS...
  local N=$1; shift; local G=$E/gold/$N CORP=$E/onpol/$N.jsonl
  [ -e $CORP ] || { $PY -I $C/docs228.py $E/onpol/$N "$@" >> $L 2>&1; log "docs228 $N rc $?"; }
  [ -e $CORP ] || return 1
  if [ ! -e $G/.done ]; then
    [ "$(df -BG --output=avail $E | tail -1 | tr -dc 0-9)" -lt 120 ] && { log "disk low: not recording $N"; return 1; }
    rm -rf $G.part
    cat > $W/rec-$N.yaml <<YEOF
memory:
  host-headroom-mib: 8192
models:
  $N:
    model: $MODEL
    tp: 2
    kv-cache-dtype: bf16
    max-model-len: 9216
    max-num-seqs: 2
    kld-record: $G.part
    kld-corpus: $CORP
    kld-top: 512
YEOF
    local s=$(date +%s)
    LD_LIBRARY_PATH="$RLIBS" RADIANCE_HOME=$RH ROCR_VISIBLE_DEVICES=2,3 $RB --config $W/rec-$N.yaml > $W/rec-$N.log 2>&1; local rc=$?
    log "record $N (TP2, cards 2,3) rc $rc in $(( ($(date +%s) - s) / 60 )) min"
    [ $rc = 0 ] && [ -e $G.part/kld.json ] || return 1
    $PY -c "import json,numpy as np; m=json.load(open('$G.part/kld.json')); np.ones(m['rows'],np.uint8).tofile('$G.part/gen.u8')"
    cp $E/onpol/$N.jsonl.meta.jsonl $G.part/corpus.meta.jsonl; touch $G.part/.done; mv $G.part $G
  fi
  [ -e $E/genmask/$N.u8 ] || { cp $G/gen.u8 $E/genmask/$N.u8.part && mv $E/genmask/$N.u8.part $E/genmask/$N.u8; }
  $PY -I $C/check228.py $G $CORP $MODEL >> $L 2>&1; log "check228 $N rc $?"
}
record stud-b1 --part $W/render-stud-trace $W/out-stud-trace pi-studpol || { log "stud-b1 golden failed: stop"; exit 1; }
record stud-calib-b1 --part $W/render-stud-calib-chat $W/out-stud-calib-chat calib-studpol --part $W/render-stud-calib-text $W/out-stud-calib-text calib-studpol \
  || { log "stud-calib-b1 golden failed: stop"; exit 1; }
# ---- 6
CD=$QROOT/cont228
if [ ! -e $CD/cont-stud-calib-b1.json ]; then
  $PY - <<PYEOF >> $L 2>&1 || { log "cont files failed: stop"; exit 1; }
import json, os, sys
import numpy as np
sys.path.insert(0, "$QROOT/code/v3/qad181"); import data181 as D
T = 1552745
for nm in ("stud-b1", "stud-calib-b1"):
    g = D.Gold("$E/gold/" + nm, genfile="$E/genmask/" + nm + ".u8")
    n = len(json.load(open("$E/gold/" + nm + "/kld.json"))["docs"]); it = []
    for i in np.random.default_rng(0).permutation(n):
        r = g.nrows(int(i), 8192)
        if r >= 16: it.append([nm, int(i), int(r)])
    out, s = [], 0
    for x in it:
        if s >= T: break
        out.append(x); s += x[2]
    f = "$CD/cont-" + nm + ".json"
    json.dump({"goldens": [nm], "seed": 0, "items": out}, open(f + ".part", "w")); os.replace(f + ".part", f)
    print("stud228:", nm, "docs", len(it), "used", len(out), "rows", s, "available", sum(x[2] for x in it))
PYEOF
  log "cont files written"
fi
# ---- 7
if ! grep -q "^T C228-stud " $QROOT/jobs.txt; then
  A="--lr 3e-4 --lr-steps 3e-4:1.5e6,1e-4:1.11e7 --plan-ext {ext}/ext-plan.json --ext-root {ext} --compile-scales --compile-ew --a8-int8"
  F="--fork-from $QROOT/runs-ice/I25-step-10M/state.pt"
  ( exec 9>$QROOT/queue.lock; flock 9; cp $QROOT/jobs.txt $QROOT/jobs.txt.pre-stud228; cat >> $QROOT/jobs.txt <<JEOF
# w228 stud228.sh (lead 11-24, user): student-generated on-policy arms (the 2.50 + I25-step-10M student wrote the turns at the teacher batches' cuts),
# mirroring C228-optr / C228-opcal: ~1.55M rows each from I25-step-10M's 10M state; judged on jc228 (JC-*), j217 and golden-jlong (JL-*).
T C228-stud 2.50 1.15e7 ckpt-1e+07 $F --cont $CD/cont-stud-b1.json $A
R JC-C228-stud 2.50 C228-stud/ckpt-1e+07
R C228-stud 2.50 C228-stud/ckpt-1e+07
R JL-C228-stud 2.50 C228-stud/ckpt-1e+07
T C228-studcal 2.50 1.15e7 ckpt-1e+07 $F --cont $CD/cont-stud-calib-b1.json $A
R JC-C228-studcal 2.50 C228-studcal/ckpt-1e+07
R C228-studcal 2.50 C228-studcal/ckpt-1e+07
R JL-C228-studcal 2.50 C228-studcal/ckpt-1e+07
JEOF
  )
  qsay "stud228 (w228): C228-stud / C228-studcal + reads queued; cards 1-3 back to the queue"
  log "jobs.txt: C228-stud / C228-studcal lines appended"
fi
# ---- 8 (the trap removes only-card1..3)
log "stud228 done"
