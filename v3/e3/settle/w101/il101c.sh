#!/bin/bash
# il101c.sh [STEP...] -- w101 #130-03 (../w101.md amendment 1): 27B dense-splice KL pair, natural vs P16 output-row
# interleave of all 112 ssm_qkv / ssm_z / attn_qg tensors at K2.5 (i8), seed pairs 0 / 1, scored on golden-search27c.
# gpubox; quants and scores on card 0 under /root/card0.lock + /root/ram.lock (shared), packs / radmix under conv.lock +
# ram.lock exclusive. Restartable: every step skips finished outputs (.done / .ok / rows files). Stop: touch $C/STOP.
# Steps: stage -> prep -> per seed S: qN S, qP S, splice S, pack S, gate S, score S -> f2 -> verdict.
# Usage (gpubox): setsid nohup bash /data/w101/c/code/v3/e3/settle/w101/il101c.sh > /data/w101/c/run.out 2>&1 < /dev/null &
set -u
C=/data/w101/c; CODE=$C/code; S=$CODE/v3/e3/settle/w101; EXL3=/data/w101/exl3; PD=$C/p16; HOUT=/data/swap27/hout-bn
MODEL=/data/workspace/models/qwen38-27b-bf16 HESS=/root/challenge/hessians/calib PY=/data/venv-exl3/bin/python
T=/data/swap27/tools I8=/data/install-116 H8=/data/swap27/home GD=/data/w92 RXM_WORK=/root STAMP=87a7e4ab32f9d5b9
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
       TORCH_EXTENSIONS_DIR=$C/build PYTORCH_HIP_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MAX_JOBS=4 \
       PATH=/opt/rocm/core-10.1/bin:$PATH ROCM_PATH=/opt/rocm/core-10.1
RX=(env PYTHONPATH=$CODE/pipeline:$EXL3:/data/exllamav3 RXQ_RXM=$CODE RXQ_EXL3_SRC=/data/exl3frac/src RXQ_REFIT_CPU_FALLBACK=1 $PY)
P=model.language_model.layers
KEYS=$(for l in $(seq 0 63); do if [ $((l % 4)) = 3 ]; then echo $P.$l.self_attn.q_proj
       else echo $P.$l.linear_attn.in_proj_qkv; echo $P.$l.linear_attn.in_proj_z; fi; done | paste -sd,)
ARM=ssm_qkv,ssm_z,attn_qg
declare -A CMIX=([0]=/data/swap27/mix-base.rad [1]=/data/swap27/mix-s1.rad) CREC=([0]=/data/swap27/base.recipe [1]=/data/swap27/s1.recipe)
mkdir -p $C/out $C/ck $C/mix $C/scores $C/logs; cd $C
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $C/result.txt; }
die() { log "IL101C STOP: $*"; exit 1; }
stopped() { [ -f $C/STOP ]; }
memok() { local m=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo); [ $m -ge $1 ] || die "MemAvailable $m GB < $1 GB ($2)"; }
cardrun() { ( exec 5>$RXM_WORK/card0.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7; "$@" ); }
exrun() { ( exec 9>$RXM_WORK/conv.lock; flock 9; exec 7>$RXM_WORK/ram.lock; flock 7; "$@" ); }
TRUNK() { "${RX[@]}" -c "from rxq import mix; print(''.join(n + chr(10) for n in mix.trunk_names()), end='')"; }
radname() { echo $1 | sed -E 's/^model\.language_model\.layers\.([0-9]+)\.[a-z_]+\.([a-z_]+)$/blk.\1.\2.weight/; s/in_proj_qkv/ssm_qkv/; s/in_proj_z/ssm_z/; s/q_proj/attn_qg/'; }

prep() {
  [ -f $PD/.done ] && return 0; memok 20 prep
  ( exec 7>$RXM_WORK/ram.lock; flock -s 7; nice -n 10 $PY $S/mkperm101.py $MODEL $HOUT $PD ${KEYS//,/ } ) > $C/logs/prep.log 2>&1 \
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
  grep -q "$STAMP" $d.log $d/*.json 2>/dev/null || die "$arm$s: settings stamp $STAMP not found in the unit's log / outputs (G-c3)"
  [ $rc = 0 ] && [ $n = 112 ] && touch $d/.done
  log "$arm$s: rc $rc, $n wq, $(( $(date +%s) - t )) s wall (incl. lock wait)"; [ -f $d/.done ] || die "unit $arm$s"
}

splice() { local s=$1
  for a in N P; do local o=$C/ck/$a$s; [ -f $o/patch.json ] && continue; memok 16 "splice $a$s"
    exrun nice -n 10 $PY $S/mksplice101.py $C/out/$a-K2.5-s$s $MODEL $PD/perm.json $o > $C/logs/splice-$a$s.log 2>&1 \
      || die "mksplice $a$s ($C/logs/splice-$a$s.log)"
    log "splice $a$s: $(tail -1 $C/logs/splice-$a$s.log)"; done
}

pack() { local s=$1 NAMES=$(for k in ${KEYS//,/ }; do radname $k; done)
  local SRC=$(awk 'NR == 1 {for (i = 3; i <= NF; i++) printf "%s ", $i}' ${CREC[$s]})
  TRUNK | grep -vxF "$NAMES" | sed "s|\$| exl3i $SRC|" > $C/mix/pack-$s.recipe
  [ $(wc -l < $C/mix/pack-$s.recipe) = 224 ] || die "pack recipe $s: $(wc -l < $C/mix/pack-$s.recipe) rules (want 336 - 112)"
  for a in N P; do local PK=$C/mix/pack-$a$s.rad MX=$C/mix/mix-$a$s.rad
    [ -f $MX ] && [ -f $C/mix/census-$a$s.ok ] && continue; stopped && die "STOP before pack $a$s"
    if [ ! -f $PK ]; then memok 20 "pack $a$s"
      exrun bash -c "s0=\$(date +%s); rm -f $PK.part
        RADIANCE_HOME=$H8 RADIANCE_HOST_DEVICES=1 nice $T/memguard.sh 2500 $I8/bin/rad-convert $C/ck/$a$s --recipe $C/mix/pack-$s.recipe \
          --set quantization_config.quant_method=exl3 --reuse ${CMIX[$s]} --changed $C/ck/$a$s/keys.txt -o $PK.part > $C/logs/conv-$a$s.log 2>&1 \
          && mv $PK.part $PK; echo \"\$(date -u +%H:%M:%S) pack $a$s rc=\$? \$(( \$(date +%s) - s0 ))s\" >> $C/result.txt"
      [ -f $PK ] || { rm -f $PK.part; die "pack $a$s FAIL ($C/logs/conv-$a$s.log)"; }
    fi
    echo "$NAMES" | sed "s|\$| $PK|" > $C/mix/$a$s.mix
    exrun bash -c "rm -f $MX.part; $T/radmix $MX.part ${CMIX[$s]} $C/mix/$a$s.mix > $C/logs/radmix-$a$s.log 2>&1 && mv $MX.part $MX" || die "radmix $a$s"
    $PY -I $T/census.py $MX --mix $C/mix/$a$s.mix --base ${CMIX[$s]} > $C/mix/census-$a$s.txt 2>&1 || die "census $a$s ($C/mix/census-$a$s.txt)"
    touch $C/mix/census-$a$s.ok; rm -f $PK
    log "mix $a$s ready ($(du -h --apparent-size $MX | cut -f1))"
  done
}

gate() { local s=$1; [ -f $C/hyg-$s.ok ] && return 0
  local r1=$(nice -n 10 "${RX[@]}" -m rxq.pairs mixes $C/mix/mix-P$s.rad $C/mix/mix-N$s.rad --arm $ARM 2>&1 | tail -n 1)
  local r2=$(nice -n 10 "${RX[@]}" -m rxq.pairs mixes $C/mix/mix-N$s.rad ${CMIX[$s]} --arm $ARM 2>&1 | tail -n 1)
  local r3=$(nice -n 10 $PY -I $S/bytes101.py $T/census.py $C/mix/mix-N$s.rad $C/mix/mix-P$s.rad $C/ck/N$s $C/ck/P$s 2>&1 | tail -n 1)
  log "G-c1 P$s vs N$s: ${r1:0:200}"; log "G-c1 N$s vs control: ${r2:0:200}"; log "G-c2 $s: $r3"
  [[ "$r1" == "PAIR OK"* && "$r2" == "PAIR OK"* && "$r3" == "BYTES101 PASS"* ]] && touch $C/hyg-$s.ok || die "gate $s FAIL"
}

score1() { local tag=$1 part=$2 G=$GD/golden-search27c-$2 o=$C/scores/$1-$2
  [ -f $o.json.rows ] && return 0; stopped && die "STOP before score $tag-$part"; memok 6 "score $tag"
  cardrun bash -c "s0=\$(date +%s); RADIANCE_HOME=$H8 $I8/bin/radiance --model $C/mix/mix-$tag.rad --tp 1 --kv-cache-dtype bf16 \
      --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $G --kld-out $o.json.part > $C/logs/score-$tag-$part.log 2>&1; rc=\$?
    [ \$rc = 0 ] && [ -f $o.json.part.rows ] && mv $o.json.part.rows $o.json.rows && mv $o.json.part $o.json
    echo \"\$(date -u +%H:%M:%S) score $tag-$part rc=\$rc \$(( \$(date +%s) - s0 ))s\" >> $C/result.txt"
  [ -f $o.json.rows ] || die "score $tag-$part FAIL ($C/logs/score-$tag-$part.log)"
}

STEPS=${*:-prep 0 1 f2 verdict}
log "il101c start: $STEPS; settings $("${RX[@]}" -m rxq stamp --config $CODE/pipeline/config/v3.toml 2>&1 | tail -1)"
for st in $STEPS; do
  case $st in
    prep) prep ;;
    0|1) unit N $st; unit P $st; splice $st; pack $st; gate $st
         for part in x c; do score1 N$st $part; score1 P$st $part; done; log "seed $st scored" ;;
    f2) ( exec 7>$RXM_WORK/ram.lock; flock -s 7; THREADS=4 W101_PIPE=$CODE/pipeline nice -n 10 $PY $S/f2il101.py $C/out $MODEL $HESS $HOUT $PD ) \
          >> $C/logs/f2.log 2>&1 && log "f2: done" || log "f2 FAILED ($C/logs/f2.log)" ;;
    verdict) $PY $S/verdict130c.py $C $GD > $C/verdict130c.txt 2>&1; log "$(tail -n 1 $C/verdict130c.txt)" ;;
    *) die "unknown step $st" ;;
  esac
done
log "IL101C DONE: $STEPS"
