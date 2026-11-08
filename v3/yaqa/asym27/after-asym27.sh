#!/bin/bash
# writers asym rho0 on the K2.5 s1 / base arms (v3/yaqa/ASYM27.md; w37), after q2.sh trunk arms (interleaves via locks).
# w78 (11-07): every arm's exit status is logged, and "asym27: done" only when both arms finished (pair.json) and the
# verdict printed; anything else ends in "asym27: FAIL ...". Idempotent: rerun resumes each arm where it stopped.
# Status: /data/asym27/queue.log (watch.conf: "ASYM27 QUEUE DONE"); headline lines also in /data/q1106/q.log.
# Env passed to asym27.sh (go.sh sets MINMEM_GB=4, TRITON_CACHE_DIR=/data/asym27/triton).
A=/data/asym27 QL=/data/q1106/q.log SL=/data/asym27/queue.log
say() { echo "$(date -u +%T) asym27: $*" | tee -a $SL >> $QL; }
until grep -q "trunk arms done" $QL; do sleep 300; done
say "start (pid $$)"
cd $A
arm() { local name=$1; shift
  if [ -s $A/$name/pair.json ] && grep -q "ASYM27 $name DONE" $A/$name/asym27.log 2>/dev/null; then say "$name already done"; return 0; fi
  env ARM=$name "$@" bash $A/code/asym27.sh >> $A/run.out 2>&1; local rc=$?
  if [ $rc = 0 ] && [ -s $A/$name/pair.json ]; then say "$name ok"; return 0; fi
  [ $rc = 3 ] && say "$name stopped (STOP file, rc=3)" || say "$name FAIL rc=$rc (see $A/$name/asym27.log)"
  return 1; }
arm s1   BASE=/data/swap27/q-s1/K2.5 CTL=s1;   r1=$?
[ -f $A/STOP ] && { say "queue STOP file: not starting base"; exit 3; }
arm base BASE=/data/v3ladder2/K2.5   CTL=base; r2=$?
if [ $r1 = 0 ] && [ $r2 = 0 ]; then
  /data/venv-exl3/bin/python -I $A/code/verdict.py $A base s1 --paired /data/swap27/tools/paired.py \
    --golden /data/swap27/golden-search27 2>&1 | tee -a $A/result.txt > $A/verdict.out; rv=${PIPESTATUS[0]}
  v=$(grep -o "ASYM27 VERDICT.*" $A/verdict.out)
  if [ $rv = 0 ] && [ -n "$v" ]; then say "$v"; say "done (s1 ok, base ok, verdict rc 0)"; echo "ASYM27 QUEUE DONE" >> $SL; exit 0; fi
  say "FAIL verdict rc=$rv"; exit 1
fi
say "FAIL (s1 rc-state $r1, base rc-state $r2; no verdict): rerun /data/asym27/go.sh to resume"; exit 1
