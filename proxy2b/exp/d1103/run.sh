#!/bin/bash
# run.sh -- d1103 follow-ups (README.md), 0.8B, 4070, one GPU job at a time; idempotent: rerun after any failure.
# STEPS (default: all, in this order). A failed step is logged and the next one runs. Stop: touch /data/yb/night.stop.
P=/workspace/rad-exl3-mix/proxy2b X=$P/exp/d1103 D7=$P/exp/w07-denoise X2=$P/exp/w02-fixedscale E6=$P/exp/w06-heal
M=/data/models/qwen35-08b S=/data/g08-search1 F=/data/g08-fit1 Y=/data/yb O=/data/yb/d1103 H=/data/h08/calib
META=/data/challenge/calib-meta C=/data/yb/c18 HO="--hout /data/yb/c18/hdiag-snap1024-dy --sigma-reg-out 0.01"
ONLY=/data/yb/w07-seqsib/only.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
export POOLS=/workspace/rad-exl3-mix/eval/pools/pools-v3.json
STEPS=${STEPS:-stack rho gmdy scale allasym k45 heal cub dose}
mkdir -p $O; cd /tmp
log() { echo "$(date -u +%T) $*" | tee -a $O/result.txt; }
sfx() { [ $1 = 0 ] && echo "" || echo "_s$1"; }
# f32 TAG: the score file wherever an earlier experiment left it
f32() { for d in $O $Y $Y/w10-hout-attrib $Y/w07-seqsib $Y/w02-fixedscale; do [ -f $d/$1.f32 ] && { echo $d/$1.f32; return; }; done; echo $O/$1.f32; }
# nq DIR: its *.safetensors count; wqd: only a finished wq dir (150 files; a killed quant leaves fewer: ourq renames a
# layer's wq files after its shard)
nq() { ls $1/*.safetensors 2>/dev/null | wc -l; }
wqd() { for d in $O $Y $Y/w10-hout-attrib $Y/w02-fixedscale; do [ $(nq $d/q-$1/wq) -ge 150 ] && { echo $d/q-$1/wq; return; }; done; }
# ev TAG WQ [GOLDEN] [SUFFIX] [MODEL]: MODEL (default M) = a healed arm's folded checkpoint (fold.py OUT/model: norms)
ev() { tag=$1 wq=$2 g=${3:-$S} x=${4:-} em=${5:-$M}; [ -f $O/$tag$x.f32 ] && return
  log "$tag$x: $(python3 $P/evalq.py $em $g --wq $wq --out $O/$tag$x.f32 2>/dev/null | tail -1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"; }
# hm DIR: the checkpoint a healed arm is scored with (its folded norms when it trained any), as w06-heal/run.sh
hm() { [ -f $1/model/.done ] && echo $1/model || echo $M; }
pair() { local g=${3:-$S} a=$1 b=$2;   # local: callers keep their own $a / $b (bug found 11-06 by the watchdog)
  [ "$g" = $F ] && { a=$O/$1.fit.f32; b=$O/$2.fit.f32; } || { a=$(f32 $1); b=$(f32 $2); }
  log "pair $1 -> $2$([ "$g" = $F ] && echo " (fit)"): $(python3 $P/pcmp.py $g $a $b | grep -o '"ratio": [0-9.]*, "diff": [0-9.e-]*, "se": [0-9.e-]*, "z": [0-9.e-]*')"; }
# q TAG K CB args..: full quant (150 tensors) + search eval
q() { t=$1 k=$2 cb=$3; shift 3; [ -f $(f32 $t) ] && return
  [ $(ls $O/q-$t/wq/*.safetensors 2>/dev/null | wc -l) -ge 150 ] || $P/q2b.sh 08 $k $cb $O/q-$t "$@" > $O/q-$t.log 2>&1
  [ $(nq $O/q-$t/wq) -ge 150 ] || { log "QUANT FAIL $t"; return 1; }   # never score a partial quant
  ev $t $O/q-$t/wq; }
# qo TAG K BASEWQ args..: requantize the 48 writers (only.txt), the rest from BASEWQ -> O/q-TAG/wqall
qo() { t=$1 k=$2 bw=$3; shift 3
  [ $(ls $O/q-$t/wq/*.safetensors 2>/dev/null | wc -l) -ge 48 ] || $P/q2b.sh 08 $k i8 $O/q-$t --only $ONLY "$@" > $O/q-$t.log 2>&1
  [ $(ls $O/q-$t/wq/*.safetensors 2>/dev/null | wc -l) = 48 ] || { log "QUANT FAIL $t"; return 1; }
  d=$O/q-$t/wqall; mkdir -p $d
  for f in $bw/*.safetensors; do ln -sfn $(readlink -f $f) $d/; done
  for f in $O/q-$t/wq/*.safetensors; do ln -sfn $f $d/; done
  [ $(ls $d/*.safetensors | wc -l) = 150 ] || { log "ASSEMBLE FAIL $t"; return 1; }; }
# mom OUT ARGS..: moments in two layer passes; mk MOM OUT lam rho
mom() { [ -f $1.done ] && return
  for r in 0-11 12-23; do python3 $D7/moments.py $M $META $1 --layers $r --tokens 1000000 "${@:2}" >> $1.log 2>&1 || { log "MOMENTS FAIL $1 $r"; return 1; }; done
  touch $1.done; log "moments $(basename $1): $(grep '^MOMENTS DONE' $1.log | tail -1)"; }
mk() { [ -f $2.done ] && return; python3 $D7/mkdenoise.py $H $1 $2 --lam $3 --rho $4 --ckpt $M > $2.log 2>&1 || { log "MKDENOISE FAIL $2"; return 1; }; touch $2.done; }
want() { [[ " $STEPS " == *" $1 "* ]] && [ ! -f /data/yb/night.stop ]; }
step() { log "== step $1"; }

if want stack; then step stack   # 87-02: asym writers on top of edy
  for k in 3 2; do for s in 0 1 2; do b=k${k}edy$(sfx $s); t=k${k}edyasym$(sfx $s)
    mom $O/mom-$b --mode sib --wq $(wqd $b) --seed 0 && mk $O/mom-$b $O/dn-asym-$b 1 0.5 || continue
    [ -f $(f32 $t) ] || { qo $t $k $(wqd $b) --hin $O/dn-asym-$b/hin --wpre $O/dn-asym-$b/wpre $HO --seed-offset $s && ev $t $O/q-$t/wqall; }
    pair $b $t; pair k${k}asym$(sfx $s) $t
  done; done
  python3 $X/summ.py $O k3edy k3edyasym k2edy k2edyasym k3asym k3edyasym k2asym k2edyasym | tee -a $O/result.txt
fi

if want rho; then step rho   # 87-03: rho screen on the fit slice (plain base, seed 0; moments from #87)
  for k in 3 2; do for r in 0 0.25 0.75; do t=k${k}asym_r$r; dn=$O/dn-r$r-k${k}base
    mk /data/yb/w07-seqsib/mom-k${k}base $dn 1 $r || continue
    [ -f $O/$t.fit.f32 ] || { qo $t $k $Y/q-k${k}base/wq --hin $dn/hin $([ -d $dn/wpre ] && echo --wpre $dn/wpre) && ev $t $O/q-$t/wqall $F .fit; }
  done
  cp -n /data/yb/w07-seqsib/k${k}asym.fit.f32 /data/yb/w07-seqsib/k${k}base.fit.f32 $O/ 2>/dev/null
  for r in 0 0.25 0.75; do pair k${k}asym k${k}asym_r$r $F; done; done
fi

if want gmdy; then step gmdy   # 83-02: gen-masked loss, Σdy² diagonal
  CG=$Y/c23
  if [ ! -f $CG/hdiag-snap1024-dy/model.language_model.layers.23.mlp.down_proj.safetensors ]; then
    W=/data/c08work; mkdir -p $W/challenge/golden; ln -sfn $META $W/challenge/golden/calib
    RXM_WORK=$W RXM_CKPT=$M YAQA_POOLS=$POOLS YAQA_FP32=1 YAQA_DIAG=1 YAQA_DIAG_EXTRA=1 YAQA_GENMASK=1 YAQA_SEED=0 \
    YAQA_SNAP=512,1024 YAQA_SAVE_EVERY=64 python3 /workspace/rad-exl3-mix/v3/yaqa/collect.py $CG 1024 2048 > $O/c23.log 2>&1
    grep -q "^DONE" $O/c23.log && log "c23 collected: $(grep -h '^DONE' $O/c23.log)" || log "c23 COLLECT FAIL"
  fi
  if [ -d $CG/hdiag-snap1024-dy ]; then
    for s in 0 1 2; do for k in 3 2; do t=k${k}dygm$(sfx $s)
      q $t $k i8 --hout $CG/hdiag-snap1024-dy --sigma-reg-out 0.01 --seed-offset $s; pair k${k}edy$(sfx $s) $t
    done; done
    python3 $X/summ.py $O k3edy k3dygm k2edy k2dygm | tee -a $O/result.txt
  fi
fi

if want scale; then step scale   # 85-02
  STAGE=b1 B1_FULL=1 bash $X2/run.sh b1 > $O/w02-b1full.log 2>&1; grep -A40 "^K3" /data/yb/w02-fixedscale/result.txt | tail -40 >> $O/result.txt
  O2=/data/yb/w02-fixedscale
  for k in 3 2; do A=$(grep "k${k}base alpha\* = " $O2/result.txt | tail -1 | sed 's/.*alpha\* = //')
    [ -n "$A" ] || { log "no alpha* for K$k"; continue; }
    for b in k${k}base k${k}base_s1 k${k}edy k${k}edy_s1; do
      sw=$O2/q-${b}_swj; [ -f $sw/.done ] || sw=$O/q-${b}_swj
      if [ ! -f $sw/.done ]; then rm -rf $sw $O/q-${b}_swu
        python3 $X2/swiglufit.py $M $(dirname $(wqd $b)) $F $sw $O/q-${b}_swu > $O/q-${b}_swj.log 2>&1 || { log "swiglufit FAIL $b"; continue; }; fi
      ev ${b}_swj $sw/wq
      t=${b}_swj_astar; [ -f $O/q-$t/.done ] || { rm -rf $O/q-$t; python3 $X2/refitq.py $sw $O/q-$t --alpha $A > $O/q-$t.log 2>&1 && touch $O/q-$t/.done; }
      [ -f $O/q-$t/.done ] || { log "refit FAIL $t"; continue; }
      ev $t $O/q-$t/wq; pair $b ${b}_swj; pair ${b}_swj $t
      [ -f $O2/${b}_astar.f32 ] && pair ${b}_astar $t
      pair $b $t
    done; done
fi

if want allasym; then step allasym   # 88-02: all 150 tensors, own-seed quantized upstream
  for k in 3 2; do for s in 0 1 2; do b=k${k}base$(sfx $s); t=k${k}allasym$(sfx $s)
    mom $O/mom-q-$b --mode quant --wq $Y/q-$b/wq --seed 0 && mk $O/mom-q-$b $O/dn-allasym-$b 1 0.5 || continue
    q $t $k i8 --hin $O/dn-allasym-$b/hin --wpre $O/dn-allasym-$b/wpre --seed-offset $s
    pair $b $t; pair k${k}asym$(sfx $s) $t
  done; done
  python3 $X/summ.py $O k3base k3allasym k2base k2allasym k3asym k3allasym k2asym k2allasym | tee -a $O/result.txt
fi

if want k45; then step k45   # 87-04
  for k in 4 5; do for s in 0 1; do b=k${k}base$(sfx $s); t=k${k}asym$(sfx $s); bw=$(wqd $b)
    [ -n "$bw" ] || { q $b $k i8 --seed-offset $s; bw=$O/q-$b/wq; }
    mom $O/mom-$b --mode sib --wq $bw --seed 0 && mk $O/mom-$b $O/dn-asym-$b 1 0.5 || continue
    [ -f $(f32 $t) ] || { qo $t $k $bw --hin $O/dn-asym-$b/hin --wpre $O/dn-asym-$b/wpre --seed-offset $s && ev $t $O/q-$t/wqall; }
    pair $b $t
  done; done
  python3 $X/summ.py $O k4base k4asym k5base k5asym | tee -a $O/result.txt
fi

if want heal; then step heal   # 89-02: heal the edy + asym stack (seed 0)
  for k in 3 2; do t=k${k}edyasym; d=$O/h-$t; [ -f $O/${t}_heal.f32 ] && continue
    [ -d $O/q-$t/wqall ] || { log "no stack $t"; continue; }
    mkdir -p $d; [ -f $d/params.pt ] || python3 $E6/heal.py train $d --wq $O/q-$t/wqall >> $d.log 2>&1 || { log "heal FAIL $t"; continue; }
    log "${t}_heal: $(grep '^HEAL DONE' $d.log | tail -1)"
    [ -f $d/wq/.done ] || python3 $E6/fold.py $d/params.pt $O/q-$t/wqall $d $M >> $d.log 2>&1 || { log "fold FAIL $t"; continue; }
    ev ${t}_heal $d/wq $S "" $(hm $d); pair $t ${t}_heal; pair k${k}base ${t}_heal
  done
fi

if want cub; then step cub   # 78-04: i8c + edy vs edy, seeds 3-5
  for s in 0 1 2; do for k in 3 2; do pair k${k}edy$(sfx $s) k${k}edycub$(sfx $s); done; done
  for s in 3 4 5; do for k in 3 2; do
    q k${k}edy_s$s $k i8 $HO --seed-offset $s; q k${k}edycub_s$s $k i8c $HO --seed-offset $s; pair k${k}edy_s$s k${k}edycub_s$s
  done; done
  python3 $X/summ.py $O k3edy k3edycub k2edy k2edycub | tee -a $O/result.txt
fi

if want dose; then step dose   # 33-23: 512 vs 1,024 windows
  for s in 0 1 2; do for k in 3 2; do q k${k}dy512$(sfx $s) $k i8 --hout $C/hdiag-snap512-dy --sigma-reg-out 0.01 --seed-offset $s; pair k${k}edy$(sfx $s) k${k}dy512$(sfx $s); done; done
  python3 $X/summ.py $O k3edy k3dy512 k2edy k2dy512 | tee -a $O/result.txt
fi
log "D1103 DONE"
