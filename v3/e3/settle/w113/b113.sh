#!/bin/bash
# b113.sh [STEP...] -- w113 #148-02 (../w113.md): 27B KL of P16-TP (T) vs natural on golden-search27e. The scope (families
# that passed #148-01's gate) is read from $C/scope (one line, e.g. "ssm_qkv,ssm_z,attn_qg"). T units (K2.5, seeds 0 / 1,
# w101's staged (c) code, stamp 87a7e4ab) -> dense bf16 patch (mksplice113) -> rad-convert --reuse + radmix of the scope's
# weights into w111's mix-N_S-v -> mix-T_S -> gates -> 12 scores (N_S-v, T_S, P_S-v x both parts) -> verdict.
# gpubox; quants and scores on card 0 under /root/card0.lock then /root/ram.lock (shared), one unit / one score per hold;
# packs / radmix under conv.lock + ram.lock exclusive. Restartable: every step skips finished outputs. Stop: touch $C/STOP.
# Steps: prep -> per seed S: S (unit, splice, pack, gate) -> score (needs $C/slice.env: GX=.. GC=..) -> verdict.
# Usage (gpubox): setsid nohup bash /data/w113/code/v3/e3/settle/w113/b113.sh > /data/w113/b/run.out 2>&1 < /dev/null &
set -u
C=/data/w113/b; S=/data/w113/code/v3/e3/settle/w113; W101=/data/w101/c; CODE=$W101/code; EXL3=/data/w101/exl3; PD=$C/permT
M111=/data/w111/mix HOUT=/data/swap27/hout-bn MODEL=/data/workspace/models/qwen38-27b-bf16 HESS=/root/challenge/hessians/calib
PY=/data/venv-exl3/bin/python T=/data/swap27/tools I8=/data/install-116 H8=/data/swap27/home STAMP=87a7e4ab32f9d5b9
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
       TORCH_EXTENSIONS_DIR=$W101/build PYTORCH_HIP_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MAX_JOBS=4 \
       PATH=/opt/rocm/core-10.1/bin:$PATH ROCM_PATH=/opt/rocm/core-10.1
RX=(env PYTHONPATH=$CODE/pipeline:$EXL3:/data/exllamav3 RXQ_RXM=$CODE RXQ_EXL3_SRC=/data/exl3frac/src RXQ_REFIT_CPU_FALLBACK=1 $PY)
declare -A CMIX=([0]=/data/swap27/mix-base.rad [1]=/data/swap27/mix-s1.rad)
mkdir -p $C/out $C/ck $C/mix $C/scores $C/logs; cd $C
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $C/result.txt; }
die() { log "B113 STOP: $*"; exit 1; }
stopped() { [ -f $C/STOP ]; }
memok() { local m=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo); [ $m -ge $1 ] || die "MemAvailable $m GB < $1 GB ($2)"; }
cardrun() { ( exec 5>/root/card0.lock; flock 5; exec 7>/root/ram.lock; flock -s 7; "$@" ); }
exrun() { ( exec 9>/root/conv.lock; flock 9; exec 7>/root/ram.lock; flock 7; "$@" ); }

[ -f $C/scope ] || die "no $C/scope (written from #148-01's gate line)"
SCOPE=$(cat $C/scope); P=model.language_model.layers
KEYS=""; NAMES=""
for l in $(seq 0 63); do
  if [ $((l % 4)) = 3 ]; then
    [[ ,$SCOPE, == *,attn_qg,* ]] && { KEYS+="$P.$l.self_attn.q_proj,"; NAMES+="blk.$l.attn_qg.weight"$'\n'; }
    [[ ,$SCOPE, == *,attn_v,* ]] && { KEYS+="$P.$l.self_attn.v_proj,"; NAMES+="blk.$l.attn_v.weight"$'\n'; }
  else
    [[ ,$SCOPE, == *,ssm_qkv,* ]] && { KEYS+="$P.$l.linear_attn.in_proj_qkv,"; NAMES+="blk.$l.ssm_qkv.weight"$'\n'; }
    [[ ,$SCOPE, == *,ssm_z,* ]] && { KEYS+="$P.$l.linear_attn.in_proj_z,"; NAMES+="blk.$l.ssm_z.weight"$'\n'; }
  fi
done
KEYS=${KEYS%,}; NAMES=${NAMES%$'\n'}; NK=$(echo "$NAMES" | wc -l)
[ -n "$KEYS" ] || die "empty scope"

prep() {
  [ -f $PD/.done ] && return 0; memok 24 prep
  ( exec 7>/root/ram.lock; flock -s 7; nice -n 10 $PY $S/mkperm113.py T $MODEL $HOUT $PD ${KEYS//,/ } ) > $C/logs/prep.log 2>&1 \
    || die "prep FAILED ($C/logs/prep.log)"
  log "prep: $(tail -1 $C/logs/prep.log)"
}

unit() { # unit SEED
  local s=$1 d=$C/out/T-K2.5-s$1; [ -f $d/.done ] && return 0; stopped && die "STOP before T$s"
  local t=$(date +%s); mkdir -p $d
  cardrun bash -c "echo \"\$(date -u +%H:%M:%S) T$s: card 0 held after \$(( \$(date +%s) - $t )) s\" >> $C/result.txt
    $T/memguard.sh 3000 nice -n 5 ${RX[*]} -m rxq quant --keys $KEYS -K 2.5 --seed-offset $s --ckpt $PD/ckpt --hess $HESS \
      -o $d --device 0 --no-prof --save-wq --hout $PD/hout >> $d.log 2>&1"
  local rc=$? n=$(ls $d/wq/*.safetensors 2>/dev/null | wc -l)
  grep -q "$STAMP" $d.log $d/*.json 2>/dev/null || die "T$s: settings stamp $STAMP not found in the unit's log / outputs (G3)"
  [ $rc = 0 ] && [ $n = $NK ] && touch $d/.done
  log "T$s: rc $rc, $n wq, $(( $(date +%s) - t )) s wall (incl. lock wait)"; [ -f $d/.done ] || die "unit T$s"
}

splice() { local s=$1 o=$C/ck/T$1; [ -f $o/patch.json ] && return 0; memok 16 "splice T$s"
  exrun nice -n 10 $PY $S/mksplice113.py $C/out/T-K2.5-s$s $MODEL $PD/perm.json $o $NK > $C/logs/splice-T$s.log 2>&1 \
    || die "mksplice T$s ($C/logs/splice-T$s.log)"
  log "splice T$s: $(tail -1 $C/logs/splice-T$s.log)"
}

pack() { local s=$1 PK=$C/mix/pack-T$1.rad MX=$C/mix/mix-T$1.rad
  [ -f $MX ] && [ -f $C/mix/census-T$s.ok ] && return 0; stopped && die "STOP before pack T$s"
  [ $(wc -l < $M111/pack-$s.recipe) = 208 ] || die "w111 pack recipe $s is not the 208-rule one"
  if [ ! -f $PK ]; then memok 20 "pack T$s"
    exrun bash -c "s0=\$(date +%s); rm -f $PK.part
      RADIANCE_HOME=$H8 RADIANCE_HOST_DEVICES=1 nice $T/memguard.sh 2500 $I8/bin/rad-convert $C/ck/T$s --recipe $M111/pack-$s.recipe \
        --set quantization_config.quant_method=exl3 --reuse ${CMIX[$s]} --changed $C/ck/T$s/keys.txt -o $PK.part > $C/logs/conv-T$s.log 2>&1 \
        && mv $PK.part $PK; echo \"\$(date -u +%H:%M:%S) pack T$s rc=\$? \$(( \$(date +%s) - s0 ))s\" >> $C/result.txt"
    [ -f $PK ] || { rm -f $PK.part; die "pack T$s FAIL ($C/logs/conv-T$s.log)"; }
  fi
  echo "$NAMES" | sed "s|\$| $PK|" > $C/mix/T$s.mix
  exrun bash -c "rm -f $MX.part; $T/radmix $MX.part $M111/mix-N$s-v.rad $C/mix/T$s.mix > $C/logs/radmix-T$s.log 2>&1 && mv $MX.part $MX" || die "radmix T$s"
  $PY -I $T/census.py $MX --mix $C/mix/T$s.mix --base $M111/mix-N$s-v.rad > $C/mix/census-T$s.txt 2>&1 || die "census T$s ($C/mix/census-T$s.txt)"
  touch $C/mix/census-T$s.ok; rm -f $PK
  log "mix T$s ready ($(du -h --apparent-size $MX | cut -f1))"
}

gate() { local s=$1; [ -f $C/hyg-$s.ok ] && return 0
  local r1=$(nice -n 10 "${RX[@]}" -m rxq.pairs mixes $C/mix/mix-T$s.rad $M111/mix-N$s-v.rad --arm $SCOPE 2>&1 | tail -n 1)
  local r2=$(nice -n 10 $PY -I $S/bytes113.py $T/census.py $M111/mix-N$s-v.rad $C/mix/mix-T$s.rad $C/ck/T$s 2>&1 | tail -n 1)
  log "G1 T$s vs N$s-v: ${r1:0:240}"; log "G2 $s: $r2"
  [[ "$r1" == "PAIR OK"* && "$r2" == "BYTES113 PASS"* ]] && touch $C/hyg-$s.ok || die "gate $s FAIL"
}

# one score per card hold; a failure leaves no rows file
score1() { local tag=$1 part=$2 mix=$3 G o=$C/scores/$1-$2
  [ $part = x ] && G=$GX || G=$GC
  [ -f $o.json.rows ] && return 0; stopped && die "STOP before score $tag-$part"; memok 6 "score $tag"
  local t=$(date +%s)
  cardrun bash -c "s0=\$(date +%s); RADIANCE_HOME=$H8 $I8/bin/radiance --model $mix --tp 1 --kv-cache-dtype bf16 \
      --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $G --kld-out $o.json.part > $C/logs/score-$tag-$part.log 2>&1; rc=\$?
    [ \$rc = 0 ] && [ -f $o.json.part.rows ] && mv $o.json.part.rows $o.json.rows && mv $o.json.part $o.json
    echo \"\$(date -u +%H:%M:%S) score $tag-$part rc=\$rc \$(( \$(date +%s) - s0 ))s (lock wait \$(( s0 - $t ))s)\" >> $C/result.txt"
  [ -f $o.json.rows ] || die "score $tag-$part FAIL ($C/logs/score-$tag-$part.log)"
}

score() {
  [ -f $C/slice.env ] || die "no slice.env (GX= GC=)"; . $C/slice.env
  [ -f $GX/kld.json ] && [ -f $GC/kld.json ] || die "slice not built ($GX, $GC)"
  for s in 0 1; do [ -f $C/hyg-$s.ok ] || die "no gate pass for seed $s: nothing is scored"; done
  for s in 0 1; do for part in x c; do
    score1 N$s-v $part $M111/mix-N$s-v.rad; score1 T$s $part $C/mix/mix-T$s.rad; score1 P$s-v $part $M111/mix-P$s-v.rad
  done; done
  log "all 12 scores done"
}

STEPS=${*:-prep 0 1}
log "b113 start: $STEPS; scope $SCOPE ($NK tensors); settings $("${RX[@]}" -m rxq stamp --config $CODE/pipeline/config/v3.toml 2>&1 | tail -1)"
for st in $STEPS; do
  case $st in
    prep) prep ;;
    0|1) unit $st; splice $st; pack $st; gate $st; log "seed $st gated" ;;
    score) score ;;
    verdict) . $C/slice.env; $PY $S/verdict113.py $C $GX $GC > $C/verdict113.txt 2>&1; log "$(tail -n 1 $C/verdict113.txt)" ;;
    *) die "unknown step $st" ;;
  esac
done
log "B113 DONE: $STEPS"
