#!/bin/bash
# run.sh -- d1104 (README.md), 0.8B, 4070. STEPS="swj order heal0". Stop: touch /data/yb/night.stop.
P=/workspace/rad-exl3-mix/proxy2b X3=$P/exp/d1103 X=$P/exp/d1104 X2=$P/exp/w02-fixedscale E6=$P/exp/w06-heal
source <(sed -n '/^P=/,/^step() /p' $X3/run.sh)
O=/data/yb/d1104; D3=/data/yb/d1103; mkdir -p $O
f32() { for d in $O $D3 $Y $Y/w10-hout-attrib $Y/w07-seqsib $Y/w02-fixedscale; do [ -f $d/$1.f32 ] && { echo $d/$1.f32; return; }; done; echo $O/$1.f32; }
wqd() { for d in $O $D3 $Y $Y/w10-hout-attrib $Y/w02-fixedscale; do [ -d $d/q-$1/wqall ] && { echo $d/q-$1/wqall; return; }; [ -d $d/q-$1/wq ] && { echo $d/q-$1/wq; return; }; done; }
STEPS=${STEPS:-swj order heal0}
want() { [[ " $STEPS " == *" $1 "* ]] && [ ! -f /data/yb/night.stop ]; }
while pgrep -f "quantize/[o]urq.py|proxy2b/[e]valq.py|yaqa/[c]ollect.py" > /dev/null; do sleep 60; done
if want swj; then step swj
  for k in 3 2; do for s in 0 1 2; do b=k${k}edy$(sfx $s); t=${b}_swj; sw=$O/q-$t
    [ -f $sw/.done ] || { src=$O/q-${b}src; mkdir -p $src; ln -sfn $(wqd $b) $src/wq; rm -rf $sw $O/q-${b}_swu
      python3 $X2/swiglufit.py $M $src $F $sw $O/q-${b}_swu > $sw.log 2>&1 || { log "swiglufit FAIL $b"; continue; }; }
    ev $t $sw/wq; pair $b $t
  done; done
  python3 $X3/summ.py $O k3edy k3edy_swj k2edy k2edy_swj | tee -a $O/result.txt
fi
if want order; then step order
  for k in 3 2; do for s in 0 1 2; do b=k${k}edy$(sfx $s)_swj; t=k${k}edyswjasym_r0$(sfx $s); bw=$O/q-$b/wq
    [ -d $bw ] || { log "no $b"; continue; }
    mom $O/mom-$b --mode sib --wq $bw --seed 0 && mk $O/mom-$b $O/dn-r0-$b 1 0 || continue
    [ -f $(f32 $t) ] || { qo $t $k $bw --hin $O/dn-r0-$b/hin $([ -d $O/dn-r0-$b/wpre ] && echo --wpre $O/dn-r0-$b/wpre) $HO --seed-offset $s && ev $t $O/q-$t/wqall; }
    pair k${k}edyasym_r0$(sfx $s) $t; pair k${k}edy$(sfx $s) $t
  done; done
  python3 $X3/summ.py $O k3edyasym_r0 k3edyswjasym_r0 k2edyasym_r0 k2edyswjasym_r0 k3edy k3edyswjasym_r0 k2edy k2edyswjasym_r0 | tee -a $O/result.txt
fi
if want heal0; then step heal0
  for k in 3 2; do
    # the better stack by the order step's mean (swj version only if its K pair beat plain ρ 0 on every seed)
    st=k${k}edyasym_r0; grep -q "^k${k}edyswjasym_r0 vs k${k}edyasym_r0: .*every seed < 1: True" $O/result.txt && st=k${k}edyswjasym_r0
    log "heal0 K$k on $st"
    for s in 0 1; do t=$st$(sfx $s); d=$O/h-$t; [ -f $O/${t}_heal.f32 ] && continue
      w=$(wqd $t); [ -n "$w" ] || { log "no stack $t"; continue; }
      mkdir -p $d; [ -f $d/params.pt ] || python3 $E6/heal.py train $d --wq $w >> $d.log 2>&1 || { log "heal FAIL $t"; continue; }
      log "${t}_heal: $(grep '^HEAL DONE' $d.log | tail -1)"
      [ -f $d/wq/.done ] || python3 $E6/fold.py $d/params.pt $w $d $M >> $d.log 2>&1 || { log "fold FAIL $t"; continue; }
      ev ${t}_heal $d/wq $S "" $(hm $d); pair $t ${t}_heal; pair k${k}edy$(sfx $s) ${t}_heal; pair k${k}base$(sfx $s) ${t}_heal
    done
    python3 $X3/summ.py $O $st ${st}_heal k${k}base ${st}_heal | tee -a $O/result.txt
  done
fi
log "D1104 DONE"
