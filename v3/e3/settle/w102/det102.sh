#!/bin/bash
# det102.sh [JOB...] -- w102 #136-06 (../w102.md amendment 1): GPU repeat-run determinism. Reruns w92's finished fs0 jobs
# (default L03-g0 L02-g0 L02-g2) with w92's own staged code / exllamav3 and a copy of its extension build, card 0 under
# card0.lock + ram.lock (shared), into /data/w102/det/<job>, then compares every output with /data/w92/out/fs0/<job>:
# t-*.safetensors byte for byte (cmp), tilecd.jsonl / log.jsonl with time fields removed. Result: /data/w102/det/det.txt.
set -u
W=/data/w102; D=$W/det; A=/data/w92; CODE=$A/code; EXL3=$A/exl3; PY=/data/venv-exl3/bin/python
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
       TORCH_EXTENSIONS_DIR=$D/build OMP_NUM_THREADS=4 MAX_JOBS=4 PATH=/opt/rocm/core-10.1/bin:$PATH ROCM_PATH=/opt/rocm/core-10.1
RX=(env PYTHONPATH=$CODE/pipeline:$EXL3:/data/exllamav3 RXQ_RXM=$CODE RXQ_EXL3_SRC=/data/exl3frac/src RXQ_REFIT_CPU_FALLBACK=1 $PY)
mkdir -p $D; [ -d $D/build ] || cp -r $A/build $D/build
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $D/det.txt; }
strip() { $PY -c "
import json, sys
def s(o):
    if isinstance(o, dict): return {k: s(v) for k, v in o.items() if not any(t in k.lower() for t in ('sec', 'time', '_s', 'date', 'host', 'pid', 'wall'))}
    if isinstance(o, list): return [s(v) for v in o]
    return o
for l in open(sys.argv[1]):
    l = l.strip()
    if l: print(json.dumps(s(json.loads(l)), sort_keys=True))
" "$1"; }
for j in ${*:-L03-g0 L02-g0 L02-g2}; do
  o=$D/$j; ref=$A/out/fs0/$j
  if [ ! -f $o/.done ]; then
    rm -rf $o; t=$(date +%s)
    ( exec 5>/root/card0.lock; flock 5; exec 7>/root/ram.lock; flock -s 7
      log "$j: card 0 held after $(( $(date +%s) - t )) s"
      /data/swap27/tools/memguard.sh 3000 nice -n 5 "${RX[@]}" -m rxq job $A/jobs/fs0/$j.json -o $o --device 0 > $o.log 2>&1 )
    rc=$?; log "$j: rc $rc, $(( $(date +%s) - t )) s wall"
    [ $rc = 0 ] && touch $o/.done || continue
  fi
  n=0; bad=0
  for f in $ref/t-*.safetensors; do
    b=$(basename $f); n=$((n + 1))
    if cmp -s $f $o/$b; then log "$j: $b IDENTICAL"; else bad=$((bad + 1)); log "$j: $b DIFFERS ($(cmp $f $o/$b 2>&1 | head -1))"; fi
  done
  for x in tilecd.jsonl log.jsonl; do
    [ -f $ref/$x ] || continue
    if diff <(strip $ref/$x) <(strip $o/$x) > $o/$x.diff; then log "$j: $x same (time fields removed)"
    else log "$j: $x differs (time fields removed): $(wc -l < $o/$x.diff) diff lines, $o/$x.diff"; fi
  done
  log "$j: $((n - bad)) / $n tensor files byte-identical"
done
log "det102 done"
