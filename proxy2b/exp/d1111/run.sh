#!/bin/bash
# run.sh -- d1111 (README.md). 0.8B, 4070, gpu.lock per unit; idempotent.
P=/workspace/rad-exl3-mix/proxy2b X3=$P/exp/d1103
source <(sed -n '/^P=/,/^step() /p' $X3/run.sh)
X=$P/exp/d1111 X8=$P/exp/d1108
O=/data/yb/d1111 D8=/data/yb/d1108 D3=/data/yb/d1103; mkdir -p $O
f32() { for d in $O $D8 $D3 $Y; do [ -f $d/$1.f32 ] && { echo $d/$1.f32; return; }; done; echo $O/$1.f32; }
wqd() { for d in $D8 $D3 $Y; do [ -d $d/q-$1/wqall ] && { echo $d/q-$1/wqall; return; }; [ -d $d/q-$1/wq ] && { echo $d/q-$1/wq; return; }; done; }
lock() { exec 9>/data/yb/gpu.lock; flock 9; }; unlock() { flock -u 9; }
C23=$Y/c23/hdiag-snap1024-dy GX=/data/g08-xcorp2 GA=$D8/g-agent
# xcorp meta for H_in: 1,150 docs x <=2,048 tokens (xgold.py's stratified pick, seed 1111, excluding the xcorp goldens)
[ -f $O/mX/kld.json ] || CUDA_VISIBLE_DEVICES= python3 -I $P/exp/d1110/xgold.py /data/xcorp/v1/meta $O/mX 1150 1111 2048 \
   /data/yb/d1110/xmeta /data/yb/d1110/xmeta-g08-xcorp2 /data/yb/d1110/xmeta-g08-xcorp3 /data/yb/d1110/xmeta-g08-xcorp4 >> $O/prep.log 2>&1
for c in X B; do m=$([ $c = X ] && echo $O/mX || echo $D8/mB)
  [ -f $O/hin$c/.done ] || { lock; python3 $P/hcollect.py $M $m $O/hin$c > $O/hin$c.log 2>&1 && touch $O/hin$c/.done; unlock; log "hin $c: $(tail -n 1 $O/hin$c.log)"; }; done
for b in X:0.25 X:0.5 B:0.25; do c=${b%:*} w=${b#*:}; d=$O/hi-$c$w
  [ -f $d/.done ] || python3 -I $X8/blend.py hin /data/h08/calib $O/hin$c $w $d >> $O/blend.log 2>&1; done
for s in 0 1; do for k in 3 2; do base=k${k}dygm$(sfx $s); bw=$(wqd $base)
  lock; ev xc-$base $bw $GX; ev ag-$base $bw $GA; unlock
  for b in X0.25 X0.5 B0.25; do t=k${k}hi$b$(sfx $s)
    lock; q $t $k i8 --hin $O/hi-$b --hout $C23 --sigma-reg-out 0.01 --seed-offset $s; ev xc-$t $O/q-$t/wq $GX; ev ag-$t $O/q-$t/wq $GA; unlock
    pair $base $t; pair xc-$base xc-$t $GX; pair ag-$base ag-$t $GA; done; done; done
for b in X0.25 X0.5 B0.25; do for k in 3 2; do python3 $X3/summ.py $O k${k}dygm k${k}hi$b xc-k${k}dygm xc-k${k}hi$b ag-k${k}dygm ag-k${k}hi$b | tee -a $O/result.txt; done; done
log "D1111 DONE"
