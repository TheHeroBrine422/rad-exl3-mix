#!/bin/bash
# run119.sh STEP ... -- w119 on box2 (RULES.md): restartable steps, every output written .part then renamed; finished
# units are skipped. Locks (box2): $LK/card<c>.lock per card, $LK/ram.lock shared (head quants ~22 GB host RAM each; a
# quant also waits for MemAvailable >= 30 GB). Stop: touch $W/STOP (checked between units).
#   run119.sh dumps CARD              headx94 dumps of docs119.json (search, fit, verify) + w94's qad set, G1 gates
#   run119.sh worker CARD QUEUE       head quants from QUEUE lines "NAME K ORDER SIGMA_OUT [CODEBOOK]" (claims by mkdir)
#   run119.sh score CARD SET SPEC...  score119 on dump SET (outputs $W/scores/SET)
#   run119.sh int8 CARD               E9 int8 heads: deadcode's GPTQ head (HTTP range read) + RTN / RTN-fwht
set -u
Q=/data/q27 W=/data/q27/w119 LK=${LK:-/data/q27/locks}
source $Q/tools/env-box2.sh
# own copy of the fork (b0052f7c) and own JIT dir: a JIT build must never write into the shared stage tree or build dir
export PYTHONPATH=$W/exl3:$Q/extbase TORCH_EXTENSIONS_DIR=$W/build
C=$W/code; CK=$Q/models/qwen38-27b-bf16; G=$Q/challenge/golden/calib; SRC=$W/exl3/exllamav3/exllamav3_ext
export RXM_WORK=$Q RXM_CKPT=$CK RXM_TEMPLATE=$W/in/template-config.json HS_CKPT=$CK
mkdir -p $W/logs
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $W/run119.log; }
stop() { [ -f $W/STOP ] && { log "STOP"; exit 0; }; }
waitmem() { while [ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -lt ${1:-30} ]; do sleep 20; done; }
step=$1; shift
case $step in
dumps)
  card=$1
  ( exec 5>$LK/card$card.lock; flock 5; exec 7>$LK/ram.lock; flock -s 7
    for s in search fit verify; do
      [ -f $W/dump/$s/done.json ] && continue; stop
      log "dump $s start (card $card)"
      ROCR_VISIBLE_DEVICES=$card $PY -I $C/v3/yaqa/w94/headx94.py $CK $G $C/v3/yaqa/w119/docs119.json $W/dump --sets $s >> $W/logs/headx-$s.log 2>&1
      log "dump $s rc $? $(cat $W/dump/$s/done.json 2>/dev/null)"
    done
    if [ ! -f $W/dump94/qad/done.json ]; then
      ROCR_VISIBLE_DEVICES=$card $PY -I $C/v3/yaqa/w94/headx94.py $CK $G $C/v3/yaqa/w94/docs94.json $W/dump94 --sets qad >> $W/logs/headx-qad.log 2>&1
      log "dump qad (w94 docs) rc $? $(cat $W/dump94/qad/done.json 2>/dev/null)"
    fi
    for d in $W/dump/search $W/dump/fit $W/dump/verify $W/dump94/qad; do
      s=$(basename $d); o=$W/gate-$s.json; [ -f $o ] && continue
      ROCR_VISIBLE_DEVICES=$card $PY -I $C/v3/yaqa/w94/gate94.py $d $G $CK $o 1000000 >> $W/logs/gate.log 2>&1
      log "G1 $s: $(python3 -c "import json; r=json.load(open('$o')); print(r['pass'], round(r['kl_top_gen'], 6), round(r['top1_gen'], 4))")"
    done )
  ;;
worker)
  card=$1 qf=$2
  while read -r name k order sig cb; do
    [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
    d=$W/q/$name/K$k
    [ -f $d/wq/lm_head.rowerr.npz ] && continue
    mkdir -p $W/q/$name; mkdir $d.claim 2>/dev/null || continue
    stop; waitmem 30
    log "quant $name K$k ($order, sigma_out $sig) card $card start"
    ( exec 5>$LK/card$card.lock; flock 5; exec 7>$LK/ram.lock; flock -s 7; cd $W
      ROCR_VISIBLE_DEVICES=$card nice -n 5 $PY $C/quantize/ourq.py -K $k -o $d --only-head --device 0 --save-wq \
        --head-perm $W/perms/$order.npy --qtf2 --frac-ext $SRC --scaled-hessian --sigma-reg 0.025 --codebook ${cb:-i8} \
        --hout $W/perms/hd-$order --hout-blocks group --sigma-reg-out $sig < /dev/null ) > $d.log 2>&1
    if [ -f $d/wq/lm_head.rowerr.npz ]; then log "quant $name K$k OK: $(cat $d/proxy.jsonl 2>/dev/null | tail -1 | cut -c1-200)"
    else log "quant $name K$k FAIL ($d.log)"; rmdir $d.claim; fi
  done < $qf
  log "worker card $card: queue $qf done"
  ;;
score)
  card=$1 set=$2; shift 2
  case $set in qad) D=$W/dump94/qad ;; *) D=$W/dump/$set ;; esac
  ( exec 5>$LK/card$card.lock; flock 5; exec 7>$LK/ram.lock; flock -s 7
    ROCR_VISIBLE_DEVICES=$card $PY -I $C/v3/yaqa/w119/score119.py $D $CK $W/scores/$set "$@" ) >> $W/logs/score-$set.log 2>&1
  log "score $set rc $? ($# options)"
  ;;
scoreall)
  # every finished donor / option not yet scored on SET (a8 arms on search and fit only)
  card=$1 set=$2; sp=()
  for d in $W/q/*/K*/wq/lm_head.rowerr.npz; do
    [ -f "$d" ] || continue; qd=${d%/wq/lm_head.rowerr.npz}; name=$(basename $(dirname $qd)); k=$(basename $qd); k=${k#K}
    case $name in S|T) ord=$name; bl=1 ;; Sm1) ord=S; bl=1 ;; T03) ord=T; bl=1 ;; R16*) ord=R16; bl=0 ;; N) ord=N; bl=0 ;; *) continue ;; esac
    [ -f $W/scores/$set/$name-K$k.npz ] || sp+=("$name-K$k=exl3:wq=$qd/wq/lm_head.safetensors,perm=$W/perms/$ord.npy,blocks=$bl")
    if [ ${name:0:1} != T ] && [ "${name:0:4}" != R16s ] && { [ $set = search ] || [ $set = fit ]; }; then
      [ -f $W/scores/$set/$name-K$k-a8.npz ] || sp+=("$name-K$k-a8=exl3:wq=$qd/wq/lm_head.safetensors,perm=$W/perms/$ord.npy,act=a8,qhead=$qd/q-head.safetensors")
    fi
  done
  [ -n "${T_BF16:-}" ] && [ ! -f $W/scores/$set/T-bf16.npz ] && sp+=("T-bf16=bf16:perm=$W/perms/T.npy,blocks=1")
  for v in gptq rtn rtn-fwht; do
    [ -f $W/i8/$v.npz ] || continue; f=$([ $v = rtn-fwht ] && echo ",fwht=1")
    [ -f $W/scores/$set/i8-$v.npz ] || sp+=("i8-$v=int8:npz=$W/i8/$v.npz$f")
    [ -f $W/scores/$set/i8-$v-a8.npz ] || sp+=("i8-$v-a8=int8:npz=$W/i8/$v.npz$f,act=a8")
  done
  [ -n "${BF16_FULL:-}" ] && [ ! -f $W/scores/$set/bf16-full.npz ] && sp+=("bf16-full=bf16:perm=id")
  [ ${#sp[@]} -eq 0 ] && { log "scoreall $set: nothing new"; exit 0; }
  bash $0 score $card $set "${sp[@]}"
  ;;
int8)
  card=$1; mkdir -p $W/i8
  U=https://huggingface.co/StillDeadcode/qwen3.8-27b-iq8r-dflash/resolve/main/qwen3.8-27b-iq8r-dflash.rad
  [ -f $W/i8/gptq.npz ] || { $PY -I $C/v3/yaqa/w119/fetch119.py $U output.weight $W/i8/gptq.part.npz >> $W/logs/i8.log 2>&1 && mv $W/i8/gptq.part.npz $W/i8/gptq.npz; log "i8 gptq fetched: $(tail -1 $W/logs/i8.log)"; }
  ( exec 5>$LK/card$card.lock; flock 5; exec 7>$LK/ram.lock; flock -s 7
    for v in rtn rtn-fwht; do
      [ -f $W/i8/$v.npz ] && continue
      ROCR_VISIBLE_DEVICES=$card $PY -I $C/v3/yaqa/w119/int8rtn119.py $CK $W/i8/$v.part.npz $([ $v = rtn-fwht ] && echo fwht) >> $W/logs/i8.log 2>&1 \
        && mv $W/i8/$v.part.npz $W/i8/$v.npz; log "i8 $v: $(tail -1 $W/logs/i8.log)"
    done )
  ;;
esac
