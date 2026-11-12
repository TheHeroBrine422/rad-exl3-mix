#!/bin/bash
# run122.sh STEP ... -- w122 on ds8 (RULES.md): restartable steps, outputs written .part then renamed, finished units
# skipped. Locks: $LK/card<c>.lock per card (HIP index), $LK/ram.lock shared (a head quant peaks ~22 GB host RAM; a
# quant also waits for MemAvailable >= 30 GB). Stop: touch $W/STOP (checked between units).
#   run122.sh pull                   w119's code / fork / perms / inputs / dumps / donors / scores from box2 (env BOX2=user@host,
#                                    BOX2KEY=ssh key; 10 GbE link)
#   run122.sh worker CARD QUEUE      head quants from QUEUE lines "NAME K SIGMA_OUT [CODEBOOK]" (R16 order; claims by mkdir)
#   run122.sh ident                  RULES identity gate: q/repro/K4 vs w119's R16s003 K4 (sha256 of both files)
#   run122.sh score CARD SET         score119 on dump SET for every R16 donor not yet scored (+ repro K4)
set -u
Q=/data/q27 W=/data/q27/heads LK=${LK:-/data/q27/locks}
source $Q/tools/env-ds8.sh
# w119's tree under its box2 path: ourq's code stamp hashes absolute paths, so /data/q27/w119 is a symlink to
# $W/w119 and every path below uses it (same stamp as w119's donors). Own JIT dir.
W1=/data/q27/w119; [ -e $W1 ] || ln -s $W/w119 $W1; [ "$(readlink $W1)" = $W/w119 ] || { echo "$W1 is not our symlink"; exit 1; }
export PYTHONPATH=$W1/exl3:$Q/extbase TORCH_EXTENSIONS_DIR=$W/build
C=$W1/code; CK=$Q/models/qwen38-27b-bf16; SRC=$W1/exl3/exllamav3/exllamav3_ext
export RXM_WORK=$Q RXM_CKPT=$CK RXM_TEMPLATE=$W1/in/template-config.json HS_CKPT=$CK
mkdir -p $W/logs
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $W/run122.log; }
stop() { [ -f $W/STOP ] && { log "STOP"; exit 0; }; }
waitmem() { while [ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -lt ${1:-30} ]; do sleep 20; done; }
step=$1; shift
case $step in
pull)
  : ${BOX2:?} ${BOX2KEY:?}; R="ssh -i $BOX2KEY -o StrictHostKeyChecking=no -c aes128-gcm@openssh.com"; B=/data/q27/w119
  mkdir -p $W/w119 $W/q/R16s003 $W/q/R16s003m1 $W/q/S $W/scores/search $W/scores/fit
  for d in code exl3 perms in dump slab rep; do rsync -a --partial -e "$R" --exclude '*.npz' $BOX2:$B/$d $W/w119/ || log "FAIL pull $d"; done
  for k in 3 4 5 6 6.5; do rsync -a --partial -e "$R" $BOX2:$B/q/R16s003/K$k $W/q/R16s003/ || log "FAIL pull R16s003 K$k"; done
  rsync -a --partial -e "$R" $BOX2:$B/q/R16s003m1/K7 $W/q/R16s003m1/ || log "FAIL pull R16s003m1 K7"
  for k in 3 4 5 7; do rsync -a --partial -e "$R" --exclude q-head.safetensors $BOX2:$B/q/S/K$k $W/q/S/ || log "FAIL pull S K$k"; done
  for s in search fit; do
    rsync -a --partial -e "$R" --include 'ref.npz' --include 'R16s003*.npz' --include 'S-K*.npz' --exclude '*-a8.npz' --exclude '*' \
      $BOX2:$B/scores/$s/ $W/scores/w119-$s/ || log "FAIL pull scores $s"
  done
  log "pull done: $(du -sh $W | cut -f1)"
  ;;
worker)
  card=$1 qf=$2
  while read -r name k sig cb; do
    [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
    d=$W/q/$name/K$k
    [ -f $d/wq/lm_head.rowerr.npz ] && continue
    mkdir -p $W/q/$name; mkdir $d.claim 2>/dev/null || continue
    stop; waitmem 30
    log "quant $name K$k (R16, sigma_out $sig, ${cb:-i8}) card $card start"
    ( exec 5>$LK/card$card.lock; flock 5; exec 7>$LK/ram.lock; flock -s 7; cd $W1
      ROCR_VISIBLE_DEVICES=$card nice -n 5 $PY $C/quantize/ourq.py -K $k -o $d --only-head --device 0 --save-wq \
        --head-perm $W1/perms/R16.npy --qtf2 --frac-ext $SRC --scaled-hessian --sigma-reg 0.025 --codebook ${cb:-i8} \
        --hout $W1/perms/hd-R16 --hout-blocks group --sigma-reg-out $sig < /dev/null ) > $d.log 2>&1
    if [ -f $d/wq/lm_head.rowerr.npz ]; then log "quant $name K$k OK: $(tail -1 $d/proxy.jsonl 2>/dev/null | cut -c1-120)"
    else log "quant $name K$k FAIL ($d.log)"; rmdir $d.claim; fi
  done < $qf
  log "worker card $card: queue $qf done"
  ;;
ident)
  a=$W/q/repro/K4 b=$W/q/R16s003/K4; ok=1
  for f in q-head.safetensors wq/lm_head.safetensors wq/lm_head.rowerr.npz; do
    x=$(sha256sum < $a/$f | cut -c1-16); y=$(sha256sum < $b/$f | cut -c1-16)
    [ "$x" = "$y" ] && log "ident $f SAME $x" || { log "ident $f DIFFERS ds8 $x box2 $y"; ok=0; }
  done
  log "ident proxy ds8 $(cut -c1-90 $a/proxy.jsonl) | box2 $(cut -c1-90 $b/proxy.jsonl)"
  [ $ok = 1 ] && { touch $W/ident.ok; log "IDENT PASS"; } || { log "IDENT FAIL: STOP"; exit 1; }
  ;;
score)
  card=$1 set=$2; sp=()
  for d in $W/q/R16s003*/K* $W/q/repro/K4; do
    [ -f $d/wq/lm_head.rowerr.npz ] || continue
    fam=$(basename $(dirname $d)); k=$(basename $d); n=$fam-$k; [ $fam = repro ] && n=repro-K4
    [ -f $W/scores/$set/$n.npz ] || sp+=("$n=exl3:wq=$d/wq/lm_head.safetensors,perm=$W1/perms/R16.npy")
  done
  [ ${#sp[@]} -eq 0 ] && { log "score $set: nothing new"; exit 0; }
  ( exec 5>$LK/card$card.lock; flock 5; exec 7>$LK/ram.lock; flock -s 7
    ROCR_VISIBLE_DEVICES=$card $PY -I $C/v3/yaqa/w119/score119.py $W1/dump/$set $CK $W/scores/$set "${sp[@]}" ) >> $W/logs/score-$set.log 2>&1
  log "score $set rc $? (${#sp[@]} options)"
  ;;
esac
