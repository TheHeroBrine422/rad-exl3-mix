#!/bin/bash
# s3score-ds8.sh "CARDS" -- RULES-177 stage 3 scores on ds8 (box2's cards were reserved by other work 11-13 ~12:50 PM CST):
# the four A8 mixes (s3score.sh's mix-a8.rad, confounder checks passed on box2) and the two plugin homes rsynced from
# box2 into /data/q27/w139s3/; every (K, golden, arm) scored here, so each paired comparison runs on one machine.
# Same radiance command as s3score.sh (rad10-perf install, home-base3 for i8 / home-hq for hash, bf16 KV,
# max-model-len 16384, max-num-seqs 1). Units claimed by mkdir (run one per card); card lock (first free of CARDS,
# polled) + ram.lock shared. GOLDS = golden dirs (default search27c-c and ver3-8k; amendment 1 adds slices/golden-h177-x).
# Out /data/q27/w139s3/<level>/score-<golden>.json(.rows). (w139, #177)
set -u; Q=/data/q27; S=$Q/w139s3; L=$Q/locks; RP=$Q/rad10-perf/install/bin/radiance; LOG=$S/score-ds8.log
say() { echo "$(date -u +%T) $*" >> $LOG; }
for K in 2.625 4.125; do for gp in ${GOLDS:-$Q/goldens/golden-search27c-c $Q/goldens/golden-ver3-8k}; do g=$(basename $gp); for a in i8 hash; do
  D=$S/q27v3-144dd6e1-k$K-$a; hm=base3; [ $a = hash ] && hm=hq; o=$D/score-$g
  [ -f $o.json ] && continue; mkdir $o.claim 2>/dev/null || continue
  while :; do for c in $1; do exec 5>$L/card$c.lock; flock -n 5 && break 2; exec 5>&-; done; sleep 20; done
  ( flock -s 6; say "score $(basename $D) $g card $c"
    ROCR_VISIBLE_DEVICES=$c RADIANCE_HOME=$S/home-$hm $RP --model $D/mix-a8.rad --tp 1 --kv-cache-dtype bf16 --max-model-len 16384 \
      --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $gp --kld-out $o.json.part > $o.log 2>&1 \
      && mv $o.json.part.rows $o.json.rows && mv $o.json.part $o.json; say "score $(basename $D) $g rc $?" ) 6>$L/ram.lock
  exec 5>&-; [ -f $o.json ] || rmdir $o.claim
done; done; done
say "worker ($1) done"
