#!/bin/bash
# knapwait.sh -- #46-03 steps 3-4 driver on the VM (w72): waits for the gpubox queue's cost tables (q.log "(g64 phase)
# A4COST VERDICT" -> version g64, "A4COST DONE" -> version final), then on colo (cpu1, /data/work/w72): mkcost.py ->
# a4cost-<ver>.jsonl, knap5a jobs (mkjobs.py, arm a4m-<ver>, run6.py 8 at a time, nice 19), step4.py vs noa4. Results
# copied to v3/e3/a4/cost/data/; one 'STEP4 VERDICT (<ver>)' line a version in data/knapwait.log. Idempotent: a version
# whose step4-<ver>.txt exists is skipped. Light (one ssh a poll, every 10 min).
# Run: setsid nohup bash v3/e3/a4/cost/knapwait.sh > /dev/null 2>&1 < /dev/null &   (from the repo root)
cd "$(dirname "$0")"; C=$PWD; D=$C/data; mkdir -p $D; LOG=$D/knapwait.log
GB=../../../../local/bin/ssh-gpubox.sh; CO=../../../../local/bin/ssh-cpu1.sh; W=/data/work/w72
log() { echo "$(date -u +%T) $*" >> $LOG; }
exec 3>/tmp/w72-knapwait.lock; flock -n 3 || exit 0
log "knapwait start (pid $$)"
while :; do
  ql=$($GB 'cat /data/a4cost/q.log 2>/dev/null' < /dev/null) || { sleep 600; continue; }
  for ver in g64 final; do
    [ -f $D/step4-$ver.txt ] && continue
    case $ver in g64) echo "$ql" | grep -q "(g64 phase) A4COST VERDICT" || continue ;;
                 final) echo "$ql" | grep -q "A4COST DONE" || continue ;; esac
    if [ ! -f $D/table-$ver.jsonl ]; then
      $GB 'cat /data/a4cost/table.jsonl' < /dev/null > $D/table-$ver.jsonl.part && $GB 'cat /data/a4cost/results.jsonl' < /dev/null > $D/results-$ver.jsonl \
        && $GB "cat /data/a4cost/analysis$([ $ver = g64 ] && echo -g64).txt" < /dev/null > $D/analysis-$ver.txt && mv $D/table-$ver.jsonl.part $D/table-$ver.jsonl || continue
      log "$ver: tables fetched ($(wc -l < $D/table-$ver.jsonl) cells)"
    fi
    $CO "cat > $W/table-$ver.jsonl" < $D/table-$ver.jsonl; $CO "cat > $W/results-$ver.jsonl" < $D/results-$ver.jsonl
    $CO "cd $W/tree/v3/nf && python3 mkcost.py $W/table-$ver.jsonl $W/results-$ver.jsonl $W/a4cost-$ver.jsonl && cd $W && python3 mkjobs.py jobs-$ver.txt $W/a4cost-$ver.jsonl a4m-$ver" \
      < /dev/null >> $LOG 2>&1 || { log "$ver: mkcost / mkjobs FAIL"; continue; }
    $CO "cd $W && pgrep -f '^python3 run6.py jobs-$ver.txt' > /dev/null || grep -q 'W72 QUEUE DONE' logs/queue-$ver.log 2>/dev/null || (setsid nohup python3 run6.py jobs-$ver.txt 8 > logs/queue-$ver.log 2>&1 < /dev/null &)" < /dev/null
    log "$ver: knap5a queue running on colo"
    until $CO "grep -q 'W72 QUEUE DONE' $W/logs/queue-$ver.log" < /dev/null; do sleep 300; done
    $CO "grep -c FAIL $W/logs/queue-$ver.log; cd $W && python3 step4.py out a4m-$ver noa4 && echo && python3 step4.py out a4e1c noa4" < /dev/null > $D/step4-$ver.txt.part 2>&1
    $CO "cat $W/a4cost-$ver.jsonl" < /dev/null > $D/a4cost-$ver.jsonl
    $CO "cd $W/out && tar czf - noa4-* a4e1c-* a4m-$ver-*" < /dev/null > $D/knap-$ver.tgz
    mv $D/step4-$ver.txt.part $D/step4-$ver.txt
    log "STEP4 VERDICT ($ver): $(grep '^STEP4 VERDICT' $D/step4-$ver.txt | head -1 | cut -d: -f2-) [$(grep '^STEP4 [0-9]' $D/step4-$ver.txt | head -3 | tr '\n' ';')]"
  done
  [ -f $D/step4-final.txt ] && { log "KNAPWAIT DONE"; exit 0; }
  sleep 600
done
