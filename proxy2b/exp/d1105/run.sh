#!/bin/bash
# run.sh -- d1105 (README.md), 0.8B, 4070. STEPS="sp mix k45". Each GPU unit holds /data/yb/gpu.lock (hconv shares the
# card). Idempotent. Stop: touch /data/yb/night.stop.
P=/workspace/rad-exl3-mix/proxy2b X3=$P/exp/d1103 X2=$P/exp/w02-fixedscale E6=$P/exp/w06-heal
source <(sed -n '/^P=/,/^step() /p' $X3/run.sh)
O=/data/yb/d1105 D3=/data/yb/d1103 D4=/data/yb/d1104; mkdir -p $O
f32() { for d in $O $D4 $D3 $Y $Y/w10-hout-attrib $Y/w07-seqsib $Y/w02-fixedscale; do [ -f $d/$1.f32 ] && { echo $d/$1.f32; return; }; done; echo $O/$1.f32; }
wqd() { for d in $O $D4 $D3 $Y $Y/w10-hout-attrib $Y/w02-fixedscale; do [ -d $d/q-$1/wqall ] && { echo $d/q-$1/wqall; return; }; [ -d $d/q-$1/wq ] && { echo $d/q-$1/wq; return; }; done; }
STEPS=${D1105_STEPS:-sp mix k45 k45lr}
want() { [[ " $STEPS " == *" $1 "* ]] && [ ! -f /data/yb/night.stop ]; }
lock() { exec 9>/data/yb/gpu.lock; flock 9; while pgrep -f "^bash /workspace/rad-exl3-mix/v3/yaqa/hconv/[h]conv.sh" > /dev/null; do sleep 60; done; }
unlock() { flock -u 9; }
MIX="/data/xcorp/v1/meta:0.8,/data/pisess/v1/meta@pisess-main.jsonl:0.2"
# heal1 TAG STACK ARGS..: one healing arm on STACK's weights -> TAG (h-TAG dir), scored and paired
heal1() { local t=$1 st=$2; shift 2; d=$O/h-$t; [ -f $O/$t.f32 ] && return; w=$(wqd $st); [ -n "$w" ] || { log "no stack $st"; return 1; }
  lock; mkdir -p $d
  [ -f $d/params.pt ] || python3 $E6/heal.py train $d --wq $w "$@" >> $d.log 2>&1 || { log "heal FAIL $t"; unlock; return 1; }
  log "$t: $(grep '^HEAL DONE' $d.log | tail -1)"
  [ -f $d/wq/.done ] || python3 $E6/fold.py $d/params.pt $w $d $M >> $d.log 2>&1 || { log "fold FAIL $t"; unlock; return 1; }
  ev $t $d/wq $S "" $(hm $d); unlock; }
# tags put the seed suffix last (summ.py's convention); d1104's heal0 scores aliased so: k3edyswjasym_r0_heal_s1
for k in 3 2; do for s in 1; do ln -sfn $D4/k${k}edyswjasym_r0_s${s}_heal.f32 $O/k${k}edyswjasym_r0_heal_s$s.f32; done; done
if want sp; then step sp
  for k in 3 2; do for s in 0 1; do B=k${k}edyswjasym_r0; st=$B$(sfx $s)
    heal1 ${B}_sp$(sfx $s) $st --epochs 1 --ntrain 2048 --per-doc 64 && { pair ${B}_heal$(sfx $s) ${B}_sp$(sfx $s); pair $st ${B}_sp$(sfx $s); }
  done; python3 $X3/summ.py $O k${k}edyswjasym_r0_heal k${k}edyswjasym_r0_sp k${k}edyswjasym_r0 k${k}edyswjasym_r0_sp | tee -a $O/result.txt; done
fi
if want mix; then step mix
  for k in 3 2; do for s in 0 1; do B=k${k}edyswjasym_r0; st=$B$(sfx $s)
    heal1 ${B}_mix$(sfx $s) $st --data mix --mix $MIX --epochs 1 --ntrain 2048 --per-doc 8 && { pair ${B}_sp$(sfx $s) ${B}_mix$(sfx $s); pair ${B}_heal$(sfx $s) ${B}_mix$(sfx $s); }
  done; python3 $X3/summ.py $O k${k}edyswjasym_r0_sp k${k}edyswjasym_r0_mix | tee -a $O/result.txt; done
fi
if want k45; then step k45
  for k in 4 5; do for s in 0 1; do b=k${k}edy$(sfx $s); sw=${b}_swj; t=k${k}edyswjasym_r0$(sfx $s)
    lock
    [ -n "$(wqd $b)" ] || q $b $k i8 $HO --seed-offset $s
    if [ ! -f $O/q-$sw/.done ]; then src=$O/q-${b}src; mkdir -p $src; ln -sfn $(wqd $b) $src/wq; rm -rf $O/q-$sw $O/q-${b}_swu
      python3 $X2/swiglufit.py $M $src $F $O/q-$sw $O/q-${b}_swu > $O/q-$sw.log 2>&1 || { log "swiglufit FAIL $b"; unlock; continue; }; fi
    ev $sw $O/q-$sw/wq; bw=$O/q-$sw/wq
    mom $O/mom-$sw --mode sib --wq $bw --seed 0 && mk $O/mom-$sw $O/dn-r0-$sw 1 0 || { unlock; continue; }
    [ -f $(f32 $t) ] || { qo $t $k $bw --hin $O/dn-r0-$sw/hin $([ -d $O/dn-r0-$sw/wpre ] && echo --wpre $O/dn-r0-$sw/wpre) $HO --seed-offset $s && ev $t $O/q-$t/wqall; }
    unlock; pair k${k}base$(sfx $s) $t
    h=k${k}edyswjasym_r0_heal$(sfx $s); heal1 $h $t && { pair $t $h; pair k${k}base$(sfx $s) $h; }
  done; python3 $X3/summ.py $O k${k}edyswjasym_r0 k${k}edyswjasym_r0_heal k${k}base k${k}edyswjasym_r0_heal | tee -a $O/result.txt; done
fi
if want k45lr; then step k45lr   # 89-06b: lr 1e-3 never beat step 0 at K4 / K5; screen lower lr on seed 0
  for a in "4 3e-4" "4 1e-4" "5 1e-4" "5 3e-5"; do set -- $a; k=$1 lr=$2; t=k${k}edyswjasym_r0
    h=k${k}edyswjasym_r0_lr$lr; heal1 $h $t --lr $lr && pair $t $h; done
fi
log "D1105 DONE"
