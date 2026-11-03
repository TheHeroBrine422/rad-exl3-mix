#!/bin/bash
# run2.sh -- d1103 tail (README "run2"), queued behind run.sh: waits for "D1103 DONE", then
#   s35:   #87-02 stack (edy + asym) at seeds 3-5, K3 / K2 (edy seeds 3-5 come from run.sh's cub step)
#   full:  edy + asym + C10 swj + C9 alpha* (the whole fixed-code stack), seeds 0-2, K3 / K2
#   heal1: healing on the edy + asym stack at seed 1
P=/workspace/rad-exl3-mix/proxy2b X=$P/exp/d1103 O=/data/yb/d1103
while ! grep -q "D1103 DONE" $O/result.txt 2>/dev/null; do sleep 120; done
source <(sed -n '/^P=/,/^step() /p' $X/run.sh)       # the helpers (paths, q / qo / mom / mk / ev / pair) of run.sh
STEPS=${STEPS2:-s35 full heal1}
want() { [[ " $STEPS " == *" $1 "* ]] && [ ! -f /data/yb/night.stop ]; }
if want s35; then step s35
  for s in 3 4 5; do for k in 3 2; do b=k${k}edy_s$s t=k${k}edyasym_s$s
    [ -n "$(wqd $b)" ] || q $b $k i8 $HO --seed-offset $s
    mom $O/mom-$b --mode sib --wq $(wqd $b) --seed 0 && mk $O/mom-$b $O/dn-asym-$b 1 0.5 || continue
    [ -f $(f32 $t) ] || { qo $t $k $(wqd $b) --hin $O/dn-asym-$b/hin --wpre $O/dn-asym-$b/wpre $HO --seed-offset $s && ev $t $O/q-$t/wqall; }
    pair $b $t
  done; done
  python3 $X/summ.py $O k3edy k3edyasym k2edy k2edyasym | tee -a $O/result.txt
fi
if want full; then step full
  O2=/data/yb/w02-fixedscale X2=$P/exp/w02-fixedscale
  for k in 3 2; do A=$(grep "k${k}base alpha\* = " $O2/result.txt | tail -1 | sed 's/.*alpha\* = //'); [ -n "$A" ] || continue
    for s in 0 1 2; do b=k${k}edyasym$(sfx $s); src=$O/q-$b; [ -d $src/wqall ] || continue
      d=$O/q-${b}src; mkdir -p $d; ln -sfn $src/wqall $d/wq
      sw=$O/q-${b}_swj; [ -f $sw/.done ] || { rm -rf $sw $O/q-${b}_swu; python3 $X2/swiglufit.py $M $d $F $sw $O/q-${b}_swu > $sw.log 2>&1 || { log "swiglufit FAIL $b"; continue; }; }
      t=${b}_full; [ -f $O/q-$t/.done ] || { rm -rf $O/q-$t; python3 $X2/refitq.py $sw $O/q-$t --alpha $A > $O/q-$t.log 2>&1 && touch $O/q-$t/.done; }
      [ -f $O/q-$t/.done ] || { log "refit FAIL $t"; continue; }
      ev $t $O/q-$t/wq; pair $b $t; pair k${k}edy$(sfx $s) $t; pair k${k}base$(sfx $s) $t
    done; done
  python3 $X/summ.py $O k3edyasym k3edyasym_full k2edyasym k2edyasym_full k3base k3edyasym_full k2base k2edyasym_full | tee -a $O/result.txt
fi
if want heal1; then step heal1
  for k in 3 2; do t=k${k}edyasym_s1; d=$O/h-$t; [ -f $O/${t}_heal.f32 ] && continue; [ -d $O/q-$t/wqall ] || continue
    mkdir -p $d; [ -f $d/params.pt ] || python3 $E6/heal.py train $d --wq $O/q-$t/wqall >> $d.log 2>&1 || { log "heal FAIL $t"; continue; }
    [ -f $d/wq/.done ] || python3 $E6/fold.py $d/params.pt $O/q-$t/wqall $d $M >> $d.log 2>&1 || { log "fold FAIL $t"; continue; }
    ev ${t}_heal $d/wq; pair $t ${t}_heal
  done
fi
log "D1103 RUN2 DONE"
