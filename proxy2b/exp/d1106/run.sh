#!/bin/bash
# run.sh -- d1106 (README.md), 0.8B, 4070, shares /data/yb/gpu.lock. Idempotent. Stop: touch /data/yb/night.stop.
P=/workspace/rad-exl3-mix/proxy2b X3=$P/exp/d1103 X2=$P/exp/w02-fixedscale
source <(sed -n '/^P=/,/^step() /p' $X3/run.sh)
O=/data/yb/d1106 D3=/data/yb/d1103 D4=/data/yb/d1104 D5=/data/yb/d1105; mkdir -p $O
f32() { for d in $O $D5 $D4 $D3 $Y $Y/w10-hout-attrib $Y/w07-seqsib $Y/w02-fixedscale; do [ -f $d/$1.f32 ] && { echo $d/$1.f32; return; }; done; echo $O/$1.f32; }
wqd() { for d in $O $D5 $D4 $D3 $Y; do [ -d $d/q-$1/wqall ] && { echo $d/q-$1/wqall; return; }; [ -d $d/q-$1/wq ] && { echo $d/q-$1/wq; return; }; done; }
want() { [[ " ${D1106_STEPS:-k15 a4} " == *" $1 "* ]] && [ ! -f /data/yb/night.stop ]; }
lock() { exec 9>/data/yb/gpu.lock; flock 9; while pgrep -f "^bash /workspace/rad-exl3-mix/v3/yaqa/hconv/[h]conv.sh" > /dev/null; do sleep 60; done; }
unlock() { flock -u 9; }
A4='mlp\.(gate|up|down)_proj:4:64'; A4C='mlp\.(gate|up|down)_proj:4:64:0.85'; A8='layers\.:8:128'
# aqev TAG WQ AQ SUDIRS [MODEL]
aqev() { local t=$1 w=$2 aq=$3 su=$4 em=${5:-$M}; [ -f $O/$t.f32 ] && return
  ( export EVALQ_AQ="$aq" EVALQ_AQ_SU="$su"; ev $t $w $S "" $em ); }
if want k15; then step k15   # the K1.5 stack, as d1105's k45
  for s in 0 1; do k=1.5; b=k${k}edy$(sfx $s); sw=${b}_swj; t=k${k}edyswjasym_r0$(sfx $s)
    lock
    [ -n "$(wqd $b)" ] || q $b $k i8 $HO --seed-offset $s
    if [ ! -f $O/q-$sw/.done ]; then src=$O/q-${b}src; mkdir -p $src; ln -sfn $(wqd $b) $src/wq; rm -rf $O/q-$sw $O/q-${b}_swu
      python3 $X2/swiglufit.py $M $src $F $O/q-$sw $O/q-${b}_swu > $O/q-$sw.log 2>&1 || { log "swiglufit FAIL $b"; unlock; continue; }; fi
    bw=$O/q-$sw/wq; mom $O/mom-$sw --mode sib --wq $bw --seed 0 && mk $O/mom-$sw $O/dn-r0-$sw 1 0 || { unlock; continue; }
    [ -f $(f32 $t) ] || { qo $t $k $bw --hin $O/dn-r0-$sw/hin $([ -d $O/dn-r0-$sw/wpre ] && echo --wpre $O/dn-r0-$sw/wpre) $HO --seed-offset $s && ev $t $O/q-$t/wqall; }
    unlock
  done
fi
if want a4; then step a4
  for k in 2 1.5 3; do for s in 0 1; do B=k${k}edyswjasym_r0; st=$B$(sfx $s); w=$(wqd $st); [ -n "$w" ] || { log "no $st"; continue; }
    sd=$(dirname $w); be=$(dirname $(wqd k${k}edy$(sfx $s))); SUD="$sd:$be"
    lock
    aqev ${B}_a4$(sfx $s) $w "$A4" $SUD; pair $st ${B}_a4$(sfx $s)
    aqev ${B}_a8$(sfx $s) $w "$A8" $SUD; pair $st ${B}_a8$(sfx $s)
    [ $s = 0 ] && [ $k = 2 ] && { aqev ${B}_a4c $w "$A4C" $SUD; pair ${B}_a4 ${B}_a4c; }
    h=$D5/h-${B}_sp$(sfx $s)
    if [ -d $h/wq ]; then aqev ${B}_sp_a4$(sfx $s) $h/wq "$A4" $SUD $(hm $h); pair ${B}_sp$(sfx $s) ${B}_sp_a4$(sfx $s); fi
    unlock
  done
  python3 $X3/summ.py $O $B ${B}_a4 $B ${B}_a8 ${B}_sp ${B}_sp_a4 | tee -a $O/result.txt; done
fi
log "D1106 DONE"
