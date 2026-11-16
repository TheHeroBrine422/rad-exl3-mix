#!/bin/bash
# score197x.sh NAME MANIFEST GOLDEN -- #197 (w185): score197.sh plus (1) EXTRA flags (e.g. --embedding-placement ram) and
# (2) a VRAM sampler: every 2 s the used / total VRAM of every amdgpu card (sysfs mem_info_vram_*) into NAME.vram.tsv,
# so the peak rise on the card the read used can be read off (the card index <-> HIP index map is not assumed).
# The sampler is a child of this script and is killed by its own PID. Same lock / engine / flags as score197.sh.
set -eu
# the vload build keeps every level shard it reads open (64 per level): raise the 1024 soft limit (hard 524288)
ulimit -n 65536
N=$1; M=$2; G=$3; W=/data/q27/w185
BIN=${BIN:-$W/rad-vload/install/bin/radiance}; HOMEX=${HOMEX:-/data/q27/rad10-e157b/home}; EXTRA=${EXTRA:-}
mkdir -p $W/scores $W/logs
[ -s $W/scores/$N.json.rows ] && { echo "$N skipped (rows present)"; exit 0; }
cd $W/scores
sampler() {
  while :; do
    t=$(date -u +%s)
    for d in /sys/class/drm/card[0-9]*/device; do
      [ -f $d/mem_info_vram_used ] || continue
      echo "$t $(basename $(dirname $d)) $(cat $d/mem_info_vram_used) $(cat $d/mem_info_vram_total)"
    done
    sleep 2
  done
}
while :; do for c in ${CARDS:-0 2}; do
  exec 9>/data/q27/locks/card$c.lock
  if flock -n 9; then
    echo "$(date -u +%FT%TZ) $N start card $c extra [$EXTRA]" >> $W/logs/score.status
    sampler > $N.vram.tsv 2>/dev/null & SP=$!
    set +e
    flock -s /data/q27/locks/ram.lock env ROCR_VISIBLE_DEVICES=$c RADIANCE_HOME=$HOMEX $BIN --model $M --tp 1 $EXTRA \
      --kv-cache-dtype bf16 --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 \
      --kld-ref $G --kld-out $N.json.part > $W/logs/$N.log 2>&1
    rc=$?
    set -e
    kill $SP 2>/dev/null || true; wait $SP 2>/dev/null || true
    if [ $rc = 0 ] && [ -f $N.json.part.rows ]; then mv $N.json.part.rows $N.json.rows; fi
    [ -f $N.json.part ] && mv $N.json.part $N.json
    echo "$(date -u +%FT%TZ) $N rc $rc card $c" >> $W/logs/score.status
    exit $rc
  fi
  exec 9>&-
done; sleep 20; done
