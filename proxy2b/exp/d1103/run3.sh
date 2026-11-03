#!/bin/bash
# run3.sh -- #87-05 (README "run3"), queued behind run2.sh: the rho screen picked rho 0 (fit: K3 0.947, K2 0.911 vs
# rho 0.5), so: rho 0 at seeds 0-2 on search, plain (vs base and vs rho 0.5) and stacked on edy (vs edy + rho 0.5);
# plus a rho -0.25 screen on fit (does the trend continue past the clean-output target?).
P=/workspace/rad-exl3-mix/proxy2b X=$P/exp/d1103 O=/data/yb/d1103
while ! grep -q "D1103 RUN2 DONE" $O/result.txt 2>/dev/null; do sleep 120; done
source <(sed -n '/^P=/,/^step() /p' $X/run.sh)
step r0
for k in 3 2; do for s in 0 1 2; do
  b=k${k}base$(sfx $s); t=k${k}asym_r0$(sfx $s); mo=/data/yb/w07-seqsib/mom-$b; dn=$O/dn-r0-$b
  mk $mo $dn 1 0 || continue
  [ -f $(f32 $t) ] || { qo $t $k $Y/q-$b/wq --hin $dn/hin $([ -d $dn/wpre ] && echo --wpre $dn/wpre) --seed-offset $s && ev $t $O/q-$t/wqall; }
  pair $b $t; pair k${k}asym$(sfx $s) $t
  b=k${k}edy$(sfx $s); t=k${k}edyasym_r0$(sfx $s); dn=$O/dn-r0-$b
  mk $O/mom-$b $dn 1 0 || continue
  [ -f $(f32 $t) ] || { qo $t $k $(wqd $b) --hin $dn/hin $([ -d $dn/wpre ] && echo --wpre $dn/wpre) $HO --seed-offset $s && ev $t $O/q-$t/wqall; }
  pair k${k}edyasym$(sfx $s) $t; pair $b $t
done; done
python3 $X/summ.py $O k3asym k3asym_r0 k2asym k2asym_r0 k3edyasym k3edyasym_r0 k2edyasym k2edyasym_r0 k3edy k3edyasym_r0 k2edy k2edyasym_r0 | tee -a $O/result.txt
step rneg
for k in 3 2; do t=k${k}asym_rm0.25; dn=$O/dn-rm0.25-k${k}base
  mk /data/yb/w07-seqsib/mom-k${k}base $dn 1 -0.25 || continue
  [ -f $O/$t.fit.f32 ] || { qo $t $k $Y/q-k${k}base/wq --hin $dn/hin --wpre $dn/wpre && ev $t $O/q-$t/wqall $F .fit; }
  pair k${k}asym_r0 $t $F
done
log "D1103 RUN3 DONE"
