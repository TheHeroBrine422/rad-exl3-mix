#!/bin/bash
# run.sh -- w09-racing (ranked.md G3, G6): CPU replay of the V2 stream's per-row scores. NO GPU anywhere.
# Run from the VM. The data is only on gpubox (/root/evq/scores, written by search/auto/evalq2.sh); this copies the
# stream candidates' score files of calA / fit2 / fit3 (+ goldens' kld.json / gen.u8, results.jsonl, frontier
# reports, calib.jsonl for categories) to cpu1 and runs race.py there (CPU, ~1-3 min a slice). Idempotent: the copy is
# skipped once its marker exists, a slice once its label is in result.txt. Needs gpubox up (read-only use there:
# one tar of existing files, niced).
# Result: cpu1 /data/work/w09/w09-racing/result.txt (one JSON line a slice) + <slice>.txt (tables).
set -e
H=$(cd $(dirname $0) && pwd); B=$H/../../../local/bin
W=/data/work/w09/w09-racing; V=$W/v2
$B/ssh-cpu1.sh "mkdir -p $V"
for f in race.py aggsim.py; do cat $H/$f | $B/ssh-cpu1.sh "cat > $W/$f"; done
if ! $B/ssh-cpu1.sh "test -f $V/.copied"; then
  # gpubox side: list the stream candidates (a|f2|f3-knap-...) that have a score file, tar them with the metadata
  $B/ssh-gpubox.sh 'cd /root && nice -n 19 python3 -c "
import json, os, re
KR = re.compile(r\"^(a|f2|f3)-knap-\")
for l in open(\"evq/results.jsonl\"):
    t = json.loads(l).get(\"tag\", \"\")
    if KR.match(t) and os.path.exists(f\"evq/scores/{t}.f32\"): print(f\"evq/scores/{t}.f32\")
" > /tmp/w09-list.txt && ls golden-calA/kld.json golden-calA/gen.u8 golden-fit2/kld.json golden-fit2/gen.u8 \
      golden-fit3/kld.json golden-fit3/gen.u8 evq/results.jsonl auto/frontier-*.txt challenge/corpus/calib.jsonl >> /tmp/w09-list.txt &&
    wc -l < /tmp/w09-list.txt >&2 && nice -n 19 tar c -T /tmp/w09-list.txt' | $B/ssh-cpu1.sh "tar x -C $V && touch $V/.copied"
fi
for s in calA fit2 fit3; do
  $B/ssh-cpu1.sh "grep -q '\"label\": \"v2-$s\"' $W/result.txt 2>/dev/null" && continue
  $B/ssh-cpu1.sh "cd $W && OMP_NUM_THREADS=4 nice python3 race.py $V/golden-$s $V/evq/scores --results $V/evq/results.jsonl \
    --pred '$V/auto/frontier-*.txt' --corpus $V/challenge/corpus/calib.jsonl --orders 20 --label v2-$s --out $W/result.txt \
    > $W/$s.txt 2>&1; cat $W/$s.txt"
done
$B/ssh-cpu1.sh "cd $W && python3 aggsim.py $V/evq/results.jsonl > aggsim.txt; tail -3 result.txt | cut -c1-300"
