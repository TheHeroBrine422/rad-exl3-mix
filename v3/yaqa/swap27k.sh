#!/bin/bash
# swap27k.sh ARM K V2 [ourq args...] -- one arm of the 27B swap test at level K (H27.md amendment 3): quantize the 64
# trunk layers from a shared queue of 4-layer chunks (card 0 always; card 1 too while $STOPF does not exist: it may be
# taken away at any time -- stop2.sh kills its chunk and releases the claim, card 0 redoes it), pack (exl3i, mul1i8,
# rate K), mix into the V2 container, census, score on golden-search27 (card 0). Reuses swap27.sh's functions; idempotent.
ARM=$1 K=$2 V2=$3; shift 3
source <(sed -n '/^: \${RXM_WORK/,/^quant() /p' /data/swap27/tools/swap27.sh | sed '$d')
packmixk() { arm=$1 src=$2 P=$S/pack-$1.rad M=$S/mix-$1.rad
  [ -f $M ] && [ -f $S/census-$arm.ok ] && return 0
  [ $(ls $src/q-L*.safetensors 2>/dev/null | wc -l) = 64 ] || { log "$arm: $src has not 64 layers"; return 1; }
  TRUNK | sed 's/^blk\.[0-9]*\./blk.*./' | sort -u | sed "s|\$| exl3i src=$src codebook=mul1i8 rate=$K|" > $S/$arm.recipe
  if [ ! -f $P ]; then
    ( exec 9>$RXM_WORK/conv.lock; flock 9; exec 7>$RXM_WORK/ram.lock; flock -s 7; cd /workspace; s=$(date +%s)
      RADIANCE_HOME=$H RADIANCE_HOST_DEVICES=1 nice $T/memguard.sh 2500 $I/bin/rad-convert models/qwen38-27b-bf16 \
        --recipe $S/$arm.recipe --set quantization_config.quant_method=exl3 -o $P.part > $S/conv-$arm.log 2>&1 && mv $P.part $P
      echo "pack $arm rc=$? $(( $(date +%s) - s ))s" ) | tee -a $S/result.txt
    [ -f $P ] || { log "pack $arm FAIL ($S/conv-$arm.log)"; return 1; }
  fi
  TRUNK | sed "s|\$| $P|" > $S/$arm.mix
  $T/radmix $M.part $V2 $S/$arm.mix > $S/mix-$arm.log 2>&1 && mv $M.part $M || { log "radmix $arm FAIL"; return 1; }
  $PY -I $T/census.py $M --mix $S/$arm.mix --base $V2 > $S/census-$arm.txt 2>&1 || { log "census $arm FAIL"; return 1; }
  grep -qE "exl3i  *$K(\.0+)?  *mul1i8 *f16  *336 " $S/census-$arm.txt || { log "$arm: not 336 exl3i K$K trunk weights"; return 1; }
  touch $S/census-$arm.ok; rm -f $P; log "$arm: mix ready, $($PY -c "import os; print(f'{os.path.getsize(\"$M\") * 8 / 26895998464:.4f} bpw')")"; }
STOPF=${STOPF:-/data/night1105/STOP2}; O=$S/q-$ARM/K$K; CL=$S/claims-$ARM; mkdir -p $CL
done_chunk() { for l in $(seq $1 $(( $1 + 3 ))); do [ -f $O/q-L$(printf %02d $l).safetensors ] || return 1; done; }
worker() { c=$1; shift
  while :; do left=0
    for a in $(seq 0 4 60); do
      done_chunk $a && continue; left=1
      [ $c = 1 ] && [ -f $STOPF ] && return 0
      mkdir $CL/$a 2>/dev/null || continue
      echo $a > $S/card$c-chunk
      ( exec 5>$RXM_WORK/card$c.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7; cd $RXM_WORK
        ROCR_VISIBLE_DEVICES=$c EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=/data/exl3frac/build \
          exec $PY -I ourq/ourq.py -K $K -o "$S/q-$ARM/K{K}" --device 0 --layers $a-$(( a + 3 )) --qtf2 --frac-ext $SRC --scaled-hessian \
          --sigma-reg 0.025 --sigma-reg-in 0.005 --codebook i8 "$@" >> $S/quant-$ARM-c$c.log 2>&1 ) & echo $! > $S/card$c-quant.pid; wait $!
      rmdir $CL/$a 2>/dev/null; rm -f $S/card$c-chunk $S/card$c-quant.pid
    done
    [ $left = 0 ] && return 0; [ $c = 1 ] && return 0; sleep 30   # card 0: wait for chunks card 1 still holds
  done; }
if [ $(ls $O/q-L*.safetensors 2>/dev/null | wc -l) != 64 ]; then
  log "quant $ARM (K$K, queue; card 1 $([ -f $STOPF ] && echo off || echo on)): start"; s=$(date +%s)
  rmdir $CL/* 2>/dev/null   # stale claims from a killed run
  worker 0 "$@" & W0=$!; [ -f $STOPF ] || { worker 1 "$@" & }; wait $W0
  n=$(ls $O/q-L*.safetensors 2>/dev/null | wc -l); [ "$n" = 64 ] || { log "quant $ARM: FAIL ($n layers)"; exit 1; }
  log "quant $ARM: done in $(( ($(date +%s) - s) / 60 )) min"
fi
packmixk $ARM $O && C=0 score $ARM
