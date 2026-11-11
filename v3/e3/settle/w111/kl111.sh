#!/bin/bash
# kl111.sh [STEP...] -- w111 #146-02 (../w111.md): w101's #130-03 pair with the 16 self_attn.v_proj added. v_proj units
# (N / P16, seeds 0 / 1, K2.5, w101's staged (c) code, stamp 87a7e4ab) -> dense bf16 patch -> rad-convert --reuse + radmix
# of the 16 attn_v into w101's mix-{N,P}{0,1}.rad -> gates -> 16 scores (new + w101's mixes) on the fresh slice -> verdict.
# gpubox; quants and scores on card 0 under /root/card0.lock + /root/ram.lock (shared), packs / radmix under conv.lock +
# ram.lock exclusive. Restartable: every step skips finished outputs. Stop: touch $C/STOP.
# Steps: prep -> per seed S: qN S, qP S, splice S, pack S, gate S -> score (needs $C/slice.env: GX=.. GC=..) -> verdict.
# KL111_LOCKDIR=<dir>: take the step locks there (dummy files) because hold111.sh already holds /root's card0, conv and
# ram locks for the whole run (one wait instead of one per step behind a chunked level build).
# Usage (gpubox, code staged in the repo layout): setsid nohup bash /data/w111/code/v3/e3/settle/w111/kl111.sh > /data/w111/run.out 2>&1 < /dev/null &
set -u
C=/data/w111; S=$C/code/v3/e3/settle/w111; S101=$C/code/v3/e3/settle/w101; W101=/data/w101/c; CODE=$W101/code; EXL3=/data/w101/exl3; PD=$C/p16
HOUT=/data/swap27/hout-bn MODEL=/data/workspace/models/qwen38-27b-bf16 HESS=/root/challenge/hessians/calib PY=/data/venv-exl3/bin/python
T=/data/swap27/tools I8=/data/install-116 H8=/data/swap27/home RXM_WORK=${KL111_LOCKDIR:-/root} STAMP=87a7e4ab32f9d5b9
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
       TORCH_EXTENSIONS_DIR=$W101/build PYTORCH_HIP_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MAX_JOBS=4 \
       PATH=/opt/rocm/core-10.1/bin:$PATH ROCM_PATH=/opt/rocm/core-10.1
RX=(env PYTHONPATH=$CODE/pipeline:$EXL3:/data/exllamav3 RXQ_RXM=$CODE RXQ_EXL3_SRC=/data/exl3frac/src RXQ_REFIT_CPU_FALLBACK=1 $PY)
P=model.language_model.layers
KEYS=$(for l in $(seq 3 4 63); do echo $P.$l.self_attn.v_proj; done | paste -sd,)
NAMES=$(for l in $(seq 3 4 63); do echo blk.$l.attn_v.weight; done)
declare -A CMIX=([0]=/data/swap27/mix-base.rad [1]=/data/swap27/mix-s1.rad)
mkdir -p $C/out $C/ck $C/mix $C/scores $C/logs; cd $C
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $C/result.txt; }
die() { log "KL111 STOP: $*"; exit 1; }
stopped() { [ -f $C/STOP ]; }
memok() { local m=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo); [ $m -ge $1 ] || die "MemAvailable $m GB < $1 GB ($2)"; }
cardrun() { ( exec 5>$RXM_WORK/card0.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7; "$@" ); }
exrun() { ( exec 9>$RXM_WORK/conv.lock; flock 9; exec 7>$RXM_WORK/ram.lock; flock 7; "$@" ); }

prep() {
  [ -f $PD/.done ] && return 0; memok 12 prep
  ( exec 7>$RXM_WORK/ram.lock; flock -s 7; nice -n 10 $PY $S101/mkperm101.py $MODEL $HOUT $PD ${KEYS//,/ } ) > $C/logs/prep.log 2>&1 \
    || die "prep FAILED ($C/logs/prep.log)"
  log "prep: $(tail -1 $C/logs/prep.log)"
}

unit() { # unit ARM SEED
  local arm=$1 s=$2 d=$C/out/$1-K2.5-s$2; [ -f $d/.done ] && return 0; stopped && die "STOP before $arm$s"
  local ck=$MODEL ho=$HOUT; [ $arm = P ] && { ck=$PD/ckpt; ho=$PD/hout; }
  local t=$(date +%s); mkdir -p $d
  cardrun bash -c "echo \"\$(date -u +%H:%M:%S) $arm$s: card 0 held after \$(( \$(date +%s) - $t )) s\" >> $C/result.txt
    $T/memguard.sh 3000 nice -n 5 ${RX[*]} -m rxq quant --keys $KEYS -K 2.5 --seed-offset $s --ckpt $ck --hess $HESS \
      -o $d --device 0 --no-prof --save-wq --hout $ho >> $d.log 2>&1"
  local rc=$? n=$(ls $d/wq/*.safetensors 2>/dev/null | wc -l)
  grep -q "$STAMP" $d.log $d/*.json 2>/dev/null || die "$arm$s: settings stamp $STAMP not found in the unit's log / outputs (G3)"
  [ $rc = 0 ] && [ $n = 16 ] && touch $d/.done
  log "$arm$s: rc $rc, $n wq, $(( $(date +%s) - t )) s wall (incl. lock wait)"; [ -f $d/.done ] || die "unit $arm$s"
}

splice() { local s=$1
  for a in N P; do local o=$C/ck/$a$s; [ -f $o/patch.json ] && continue; memok 8 "splice $a$s"
    exrun nice -n 10 $PY $S/mksplice111.py $C/out/$a-K2.5-s$s $MODEL $PD/perm.json $o 16 > $C/logs/splice-$a$s.log 2>&1 \
      || die "mksplice $a$s ($C/logs/splice-$a$s.log)"
    log "splice $a$s: $(tail -1 $C/logs/splice-$a$s.log)"; done
}

pack() { local s=$1
  grep -vF "$NAMES" $W101/mix/pack-$s.recipe > $C/mix/pack-$s.recipe
  [ $(wc -l < $C/mix/pack-$s.recipe) = 208 ] || die "pack recipe $s: $(wc -l < $C/mix/pack-$s.recipe) rules (want 224 - 16)"
  for a in N P; do local PK=$C/mix/pack-$a$s.rad MX=$C/mix/mix-$a$s-v.rad
    [ -f $MX ] && [ -f $C/mix/census-$a$s.ok ] && continue; stopped && die "STOP before pack $a$s"
    if [ ! -f $PK ]; then memok 20 "pack $a$s"
      exrun bash -c "s0=\$(date +%s); rm -f $PK.part
        RADIANCE_HOME=$H8 RADIANCE_HOST_DEVICES=1 nice $T/memguard.sh 2500 $I8/bin/rad-convert $C/ck/$a$s --recipe $C/mix/pack-$s.recipe \
          --set quantization_config.quant_method=exl3 --reuse ${CMIX[$s]} --changed $C/ck/$a$s/keys.txt -o $PK.part > $C/logs/conv-$a$s.log 2>&1 \
          && mv $PK.part $PK; echo \"\$(date -u +%H:%M:%S) pack $a$s rc=\$? \$(( \$(date +%s) - s0 ))s\" >> $C/result.txt"
      [ -f $PK ] || { rm -f $PK.part; die "pack $a$s FAIL ($C/logs/conv-$a$s.log)"; }
    fi
    echo "$NAMES" | sed "s|\$| $PK|" > $C/mix/$a$s.mix
    exrun bash -c "rm -f $MX.part; $T/radmix $MX.part $W101/mix/mix-$a$s.rad $C/mix/$a$s.mix > $C/logs/radmix-$a$s.log 2>&1 && mv $MX.part $MX" || die "radmix $a$s"
    $PY -I $T/census.py $MX --mix $C/mix/$a$s.mix --base $W101/mix/mix-$a$s.rad > $C/mix/census-$a$s.txt 2>&1 || die "census $a$s ($C/mix/census-$a$s.txt)"
    touch $C/mix/census-$a$s.ok; rm -f $PK
    log "mix $a$s-v ready ($(du -h --apparent-size $MX | cut -f1))"
  done
}

gate() { local s=$1; [ -f $C/hyg-$s.ok ] && return 0
  local r1=$(nice -n 10 "${RX[@]}" -m rxq.pairs mixes $C/mix/mix-P$s-v.rad $C/mix/mix-N$s-v.rad --arm ssm_qkv,ssm_z,attn_qg,attn_v 2>&1 | tail -n 1)
  local r2=$(nice -n 10 "${RX[@]}" -m rxq.pairs mixes $C/mix/mix-N$s-v.rad $W101/mix/mix-N$s.rad --arm attn_v 2>&1 | tail -n 1)
  local r3=$(nice -n 10 "${RX[@]}" -m rxq.pairs mixes $C/mix/mix-P$s-v.rad $W101/mix/mix-P$s.rad --arm attn_v 2>&1 | tail -n 1)
  local r4=$(nice -n 10 $PY -I $S/bytes111.py $T/census.py $C/mix/mix-N$s-v.rad $C/mix/mix-P$s-v.rad $C/ck/N$s $C/ck/P$s 2>&1 | tail -n 1)
  log "G1 P$s-v vs N$s-v: ${r1:0:240}"; log "G1 N$s-v vs w101 N$s: ${r2:0:240}"; log "G1 P$s-v vs w101 P$s: ${r3:0:240}"; log "G2 $s: $r4"
  [[ "$r1" == "PAIR OK"* && "$r2" == "PAIR OK"* && "$r3" == "PAIR OK"* && "$r4" == "BYTES111 PASS"* ]] && touch $C/hyg-$s.ok || die "gate $s FAIL"
}

# one score (inside the card hold of score()); a failure leaves no rows file
score1() { local tag=$1 part=$2 mix=$3 G o=$C/scores/$1-$2
  [ $part = x ] && G=$GX || G=$GC
  [ -f $o.json.rows ] && return 0; stopped && return 1; memok 6 "score $tag"
  local s0=$(date +%s); RADIANCE_HOME=$H8 $I8/bin/radiance --model $mix --tp 1 --kv-cache-dtype bf16 \
      --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $G --kld-out $o.json.part > $C/logs/score-$tag-$part.log 2>&1; local rc=$?
  [ $rc = 0 ] && [ -f $o.json.part.rows ] && mv $o.json.part.rows $o.json.rows && mv $o.json.part $o.json
  log "score $tag-$part rc=$rc $(( $(date +%s) - s0 ))s"; [ -f $o.json.rows ]
}

scoreall() {
  for s in 0 1; do for part in x c; do
    score1 N$s-v $part $C/mix/mix-N$s-v.rad && score1 P$s-v $part $C/mix/mix-P$s-v.rad \
      && score1 N$s $part $W101/mix/mix-N$s.rad && score1 P$s $part $W101/mix/mix-P$s.rad || return 1
  done; done
}

score() {   # all 16 scores in one card hold (card0.lock, ram.lock shared): ~1-1.5 min each
  [ -f $C/slice.env ] || die "no slice.env (GX= GC=): the slice is fixed by ../w111.md, write it first"
  . $C/slice.env; [ -f $GX/kld.json ] && [ -f $GC/kld.json ] || die "slice not built ($GX, $GC)"
  for s in 0 1; do [ -f $C/hyg-$s.ok ] || die "no gate pass for seed $s: nothing is scored"; done
  export -f score1 scoreall log die memok stopped; export C GX GC H8 I8 W101
  local t=$(date +%s)
  cardrun bash -c "log \"scores: card 0 held after \$(( \$(date +%s) - $t )) s\"; scoreall" || die "scores FAIL or stopped (see $C/logs/score-*)"
  log "all 16 scores done"
}

STEPS=${*:-prep 0 1}
log "kl111 start: $STEPS; settings $("${RX[@]}" -m rxq stamp --config $CODE/pipeline/config/v3.toml 2>&1 | tail -1)"
for st in $STEPS; do
  case $st in
    prep) prep ;;
    0|1) unit N $st; unit P $st; splice $st; pack $st; gate $st; log "seed $st gated" ;;
    score) score ;;
    verdict) . $C/slice.env; $PY $S/verdict111.py $C $W101 $GX $GC > $C/verdict111.txt 2>&1; log "$(tail -n 1 $C/verdict111.txt)" ;;
    *) die "unknown step $st" ;;
  esac
done
log "KL111 DONE: $STEPS"
