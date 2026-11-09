#!/bin/bash
# q94.sh -- #125's CPU proxy jobs on colo (RULES.md): preps, global scales, then every (variant, arm, K, 8-block chunk)
# quant job, P at a time, single-threaded, nice 19. Restartable: hq94.py skips finished outputs (prep / scale files,
# per-block .npy). Run from /data/work/w13/w94 with env.sh. P (default 16) = parallel jobs.
cd /data/work/w13/w94 && . ./env.sh
P=${P:-16} O=out H="nice -n 19 python3 -W ignore code/v3/yaqa/w94/hq94.py"
mkdir -p $O logs
log() { echo "$(date -u +%H:%M:%S) $*" >> logs/q94.log; }
log start
for v in raw mr mr-r4 mr-r8; do [ -s $O/prep-$v.pt ] || $H prep $v $O > logs/prep-$v.log 2>&1 & done; wait
for v in raw mr mr-r4 mr-r8; do [ -s $O/prep-$v.pt ] || { log "prep $v FAIL"; exit 1; }; done
log preps done
for v in raw mr mr-r4 mr-r8; do for k in 3 4; do $H scale $v $k $O > logs/scale-$v-$k.log 2>&1 & done; done; wait
log scales done
{
  for k in 3 4; do
    for va in raw/one mr/one mr/two:0.01 mr/two:0.1 mr/two:1 mr-r4/one mr-r8/one; do
      for c in 0 1 2 3 4 5 6 7; do echo "${va%%/*} ${va#*/} $k $c"; done
    done
  done
} | xargs -P $P -L 1 sh -c "$H quant \$0 \$1 \$2 \$3 $O >> logs/quant.log 2>&1"
log "quants done rc $?"
echo Q94DONE >> logs/q94.log
