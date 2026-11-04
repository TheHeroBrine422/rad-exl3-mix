#!/bin/bash
# swap27.sh STAGE -- the 27B H_out swap test at K2.5 (rule: v3/yaqa/H27.md, amendment 2026-11-04), on iceman.
#   prep   no card: plugin home (radiance 1.16.0 + our plugins), V2 3.00 base, the base arm's pack + mix + census
#   run    waits for the Σdy² collection to end, builds the block-normalized diagonal, then per arm (bn, dy, s1):
#          quantize the K2.5 trunk on card 0, pack (exl3i, mul1i8, rate 2.5), mix into V2 3.00, census, score; then the
#          base arm is scored and the paired comparisons are written. Idempotent: rerun after any failure.
# Everything under /data/swap27; log result.txt. Stop: kill the driver; a rerun resumes per arm and per layer.
: ${RXM_WORK:=/root}; export RXM_WORK
S=/data/swap27 T=/data/swap27/tools I=/data/install-116 B=/data/install-116.build L=/data/v3ladder2
PY=/data/venv-exl3/bin/python SRC=/data/exl3frac/src HD=/data/yaqa/hdiag NSEQ=${NSEQ:-2048}
G=$S/golden-search27 V2=/data/release/qwen3.8-27b-exl3-v2-3.00bpw.rad H=$S/home C=${CARD:-0}
mkdir -p $S; cd $S; log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $S/result.txt; }
export PATH=/opt/rocm/core-10.1/bin:$PATH ROCM_PATH=/opt/rocm/core-10.1

TRUNK() { for l in $(seq 0 63); do
    if [ $((l % 4)) = 3 ]; then for t in attn_qg attn_k attn_v attn_output; do echo blk.$l.$t.weight; done
    else for t in ssm_qkv ssm_z ssm_out; do echo blk.$l.$t.weight; done; fi
    echo blk.$l.ffn_gate_up.weight; echo blk.$l.ffn_down.weight; done; }

# pack + mix + census for one arm: SRCDIR (ourq output with 64 q-LNN shards) -> $S/mix-ARM.rad
packmix() { arm=$1 src=$2 P=$S/pack-$1.rad M=$S/mix-$1.rad
  [ -f $M ] && [ -f $S/census-$arm.ok ] && return 0
  [ $(ls $src/q-L*.safetensors 2>/dev/null | wc -l) = 64 ] || { log "$arm: $src has not 64 layers"; return 1; }
  TRUNK | sed 's/^blk\.[0-9]*\./blk.*./' | sort -u | sed "s|\$| exl3i src=$src codebook=mul1i8 rate=2.5|" > $S/$arm.recipe
  if [ ! -f $P ]; then
    ( exec 9>$RXM_WORK/conv.lock; flock 9; exec 7>$RXM_WORK/ram.lock; flock -s 7; cd /workspace; s=$(date +%s)
      RADIANCE_HOME=$H RADIANCE_HOST_DEVICES=1 nice $T/memguard.sh 2500 $I/bin/rad-convert models/qwen38-27b-bf16 \
        --recipe $S/$arm.recipe --set quantization_config.quant_method=exl3 -o $P.part > $S/conv-$arm.log 2>&1 && mv $P.part $P
      echo "pack $arm rc=$? $(( $(date +%s) - s ))s" ) | tee -a $S/result.txt
    [ -f $P ] || { log "pack $arm FAIL ($S/conv-$arm.log)"; return 1; }
  fi
  TRUNK | sed "s|\$| $P|" > $S/$arm.mix
  $T/radmix $M.part $V2 $S/$arm.mix > $S/mix-$arm.log 2>&1 && mv $M.part $M || { log "radmix $arm FAIL"; return 1; }
  $PY -I $T/census.py $M --mix $S/$arm.mix --base $V2 > $S/census-$arm.txt 2>&1 || { log "census $arm FAIL ($S/census-$arm.txt)"; return 1; }
  grep -q "exl3i  *2.5  *mul1i8 *f16  *336 " $S/census-$arm.txt || { log "$arm: not 336 exl3i K2.5 trunk weights"; return 1; }
  touch $S/census-$arm.ok; rm -f $P   # the mix holds reflink clones of the pack's weights: the pack can go
  log "$arm: mix ready, $($PY -c "import os; print(f'{os.path.getsize(\"$M\") * 8 / 26895998464:.4f} bpw')")"
}

score() { arm=$1 M=$S/mix-$1.rad
  [ -f $S/$arm.json ] && return 0
  ( exec 5>$RXM_WORK/card$C.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7; s=$(date +%s)
    ROCR_VISIBLE_DEVICES=$C RADIANCE_HOME=$H $I/bin/radiance --model $M --tp 1 --kv-cache-dtype bf16 \
      --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $G --kld-out $S/$arm.json.part > $S/score-$arm.log 2>&1 \
      && mv $S/$arm.json.part $S/$arm.json; [ -f $S/$arm.json.part.rows ] && mv $S/$arm.json.part.rows $S/$arm.json.rows
    echo "score $arm rc=$? $(( $(date +%s) - s ))s" ) | tee -a $S/result.txt
  [ -f $S/$arm.json ] && log "$arm: $(grep -o '"kld[^,]*,' $S/$arm.json | head -2 | tr '\n' ' ')" || { log "score $arm FAIL ($S/score-$arm.log)"; return 1; }
}

quant() { arm=$1; shift; O=$S/q-$arm
  [ $(ls $O/K2.5/q-L*.safetensors 2>/dev/null | wc -l) = 64 ] && return 0
  log "quant $arm: start"; s=$(date +%s)
  ( exec 5>$RXM_WORK/card$C.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7; cd $RXM_WORK
    ROCR_VISIBLE_DEVICES=$C EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=/data/exl3frac/build \
      $PY -I ourq/ourq.py -K 2.5 -o "$O/K{K}" --device 0 --qtf2 --frac-ext $SRC --scaled-hessian --sigma-reg 0.025 \
      --sigma-reg-in 0.005 --codebook i8 "$@" >> $S/quant-$arm.log 2>&1 )
  n=$(ls $O/K2.5/q-L*.safetensors 2>/dev/null | wc -l)
  [ "$n" = 64 ] && log "quant $arm: done in $(( ($(date +%s) - s) / 60 )) min" || { log "quant $arm: FAIL ($n layers)"; return 1; }
}

case $1 in
prep)
  # plugin home: radiance 1.16.0 + our plugins as built by eval/port116/build.sh (all their tests passed)
  if [ ! -f $H/.ok ]; then rm -rf $H; cp -a $I/share/radiance $H
    cp $B/exl3dense/radiance_home/kernels/exl3dense.so $B/exl3r4d/radiance_home/kernels/exl3_r4d.so $B/exl3i/radiance_home/kernels/exl3i.so $H/kernels/
    cp $B/exl3quant/radiance_home/quantizers/exl3quant.so $H/quantizers/
    cp $B/qwen35_exl3/radiance_home/architectures/qwen35_exl3.so $H/architectures/
    touch $H/.ok; log "home ready (radiance 1.16.0 + plugins)"; fi
  [ -f $V2 ] || { mkdir -p /data/release; nice $T/memguard.sh 2500 /data/venv-exl3/bin/hf download Hero422/qwen3.8-27b-exl3-v2-rad \
      $(basename $V2) --local-dir /data/release > $S/dl.log 2>&1 && log "V2 3.00 downloaded" || { log "V2 download FAIL"; exit 1; }; }
  packmix base $L/K2.5 || exit 1
  log "PREP DONE" ;;
run)
  while pgrep -f "yaqa/[c]ollect.py" > /dev/null; do sleep 120; done
  [ -d $HD/hdiag-snap$NSEQ-dy ] || { log "no $NSEQ snapshot"; exit 1; }
  [ -f $S/hout-bn/mkbn.json ] || { nice $PY -I $T/mkbn.py $HD $NSEQ $S/hout-bn >> $S/result.txt 2>&1 || { log "mkbn FAIL"; exit 1; }; }
  score base || exit 1
  quant bn --hout $S/hout-bn --sigma-reg-out 0.01 && packmix bn $S/q-bn/K2.5 && score bn
  quant dy --hout $HD/hdiag-snap$NSEQ-dy --sigma-reg-out 0.01 && packmix dy $S/q-dy/K2.5 && score dy
  quant s1 --seed-offset 1 && packmix s1 $S/q-s1/K2.5 && score s1
  for a in bn dy s1; do [ -f $S/$a.json.rows ] && log "pair base -> $a: $($PY -I $T/cmp115.py $G $S/base.json.rows $S/$a.json.rows 2>&1 | tail -1)"; done
  [ -f $S/dy.json.rows ] && [ -f $S/bn.json.rows ] && log "pair dy -> bn: $($PY -I $T/cmp115.py $G $S/dy.json.rows $S/bn.json.rows 2>&1 | tail -1)"
  log "RUN DONE" ;;
*) echo "swap27.sh prep|run"; exit 2 ;;
esac
