#!/bin/bash
# run_v.sh -- d1111 amendment 3 (README): xcorp H_in follow-ups on the 0.8B. Waits for d1109/run2.sh; idempotent;
# gpu.lock per unit. Order = value: verify check, weight sweep, 3rd seed, bigger sample, K4/K5 transfer, category
# subsets, extra seeds.
P=/workspace/rad-exl3-mix/proxy2b X3=$P/exp/d1103
source <(sed -n '/^P=/,/^step() /p' $X3/run.sh)
X=$P/exp/d1111 X8=$P/exp/d1108
O=/data/yb/d1111 D8=/data/yb/d1108 D3=/data/yb/d1103
f32() { for d in $O $D8 $D3 $Y; do [ -f $d/$1.f32 ] && { echo $d/$1.f32; return; }; done; echo $O/$1.f32; }
wqd() { for d in $O $D8 $D3 $Y; do [ $(nq $d/q-$1/wqall) -ge 150 ] && { echo $d/q-$1/wqall; return; }; [ $(nq $d/q-$1/wq) -ge 150 ] && { echo $d/q-$1/wq; return; }; done; }   # finished dirs only (d1103 nq)
lock() { exec 9>/data/yb/gpu.lock; flock 9; }; unlock() { flock -u 9; }
log() { echo "$(date -u +%T) $*" | tee -a $O/result-v.txt; }
while pgrep -f "^bash /workspace/rad-exl3-mix/proxy2b/exp/d1109/[r]un2.sh" > /dev/null; do sleep 120; done
C23=$Y/c23/hdiag-snap1024-dy GX=/data/g08-xcorp2 GA=$D8/g-agent GV=/data/g08-verify1 HO="--hout $C23 --sigma-reg-out 0.01"
evall() { local t=$1 w=$2; ev xc-$t $w $GX; ev ag-$t $w $GA; ev vf-$t $w $GV; }
# base arm (dygm = calib H_in + c23 H_out, as d1103) at K / seed; K4/K5 bases are new (same args)
# (stdout of q / evall dropped: only the tag reaches arm's $(base | tail); their log lines still go to result-v.txt)
base() { local k=$1 s=$2 b=k${1}dygm$(sfx $2); [ -n "$(wqd $b)" ] || { lock; q $b $k i8 $HO --seed-offset $s > /dev/null; unlock; }
  [ -n "$(wqd $b)" ] || return 1
  lock; evall $b $(wqd $b) > /dev/null; unlock; echo $b; }
arm() { local k=$1 s=$2 h=$3 t b; t=k${k}hi$h$(sfx $s); b=$(base $k $s | tail -n 1)
  [ -n "$b" ] && [ -f $O/hi-$h/.done ] || { log "SKIP $t: base quant or hi-$h (blend) not finished"; return 1; }
  lock; q $t $k i8 --hin $O/hi-$h $HO --seed-offset $s && evall $t $O/q-$t/wq; unlock
  pair $b $t; pair xc-$b xc-$t $GX; pair ag-$b ag-$t $GA; pair vf-$b vf-$t $GV; }
blendx() { [ -f $O/hi-$2/.done ] || python3 -I $X8/blend.py hin /data/h08/calib $O/hin$1 $3 $O/hi-$2 >> $O/blend.log 2>&1 && touch $O/hi-$2/.done; }
collect() { local c=$1; [ -f $O/hin$1/.done ] || { lock; python3 $P/hcollect.py $M $2 $O/hin$1 > $O/hin$1.log 2>&1 && touch $O/hin$1/.done; unlock; log "hin $1: $(tail -n 1 $O/hin$1.log)"; }; }
summ() { python3 $X3/summ.py $O k3dygm k3hi$1 vf-k3dygm vf-k3hi$1 xc-k3dygm xc-k3hi$1 ag-k3dygm ag-k3hi$1 \
  k2dygm k2hi$1 vf-k2dygm vf-k2hi$1 xc-k2dygm xc-k2hi$1 ag-k2dygm ag-k2hi$1 2>&1 | tee -a $O/result-v.txt; }
# 1. verify golden on the passing / existing arms
for s in 0 1; do for k in 3 2; do for h in X0.5 X0.25; do arm $k $s $h; done; done; done
# 2. weight sweep
for ww in 0.75 1.0; do blendx X X$ww $ww; for ss in 0 1; do for kk in 3 2; do arm $kk $ss X$ww; done; done; done
# 3. third seed for w0.5
for k in 3 2; do arm $k 2 X0.5; done
# 4. bigger xcorp sample (~3M tokens, superset of mX)
[ -f $O/mX3/kld.json ] || CUDA_VISIBLE_DEVICES= python3 -I $P/exp/d1110/xgold.py /data/xcorp/v1/meta $O/mX3 3000 1112 2048 \
   /data/yb/d1110/xmeta /data/yb/d1110/xmeta-g08-xcorp2 /data/yb/d1110/xmeta-g08-xcorp3 /data/yb/d1110/xmeta-g08-xcorp4 >> $O/prep.log 2>&1
collect X3 $O/mX3; blendx X3 X3_0.5 0.5
for s in 0 1; do for k in 3 2; do arm $k $s X3_0.5; done; done
# 5. transfer to K4 / K5
for k in 4 5; do for s in 0 1; do arm $k $s X0.5; done; done
# 6. category subsets of mX at w0.5
for c in code:code,math,reasoning,science prose:web,chat,qa,encyclopedic,books,legal mt:multilingual,tools,agentic,long; do
  n=${c%%:*}; [ -f $O/m-$n/kld.json ] || CUDA_VISIBLE_DEVICES= python3 -I $X/subset.py $O/mX $O/m-$n ${c#*:} >> $O/prep.log 2>&1
  collect $n $O/m-$n; blendx $n $n-0.5 0.5; for s in 0 1; do for k in 3 2; do arm $k $s $n-0.5; done; done; done
# 7. extra seeds
for h in X0.75 X1.0 X3_0.5 code-0.5 prose-0.5 mt-0.5; do for k in 3 2; do arm $k 2 $h; done; done
for k in 4 5; do arm $k 2 X0.5; done
for h in X0.25 X0.5 X0.75 X1.0 X3_0.5 code-0.5 prose-0.5 mt-0.5; do summ $h; done
python3 $X3/summ.py $O k4dygm k4hiX0.5 vf-k4dygm vf-k4hiX0.5 xc-k4dygm xc-k4hiX0.5 k5dygm k5hiX0.5 vf-k5dygm vf-k5hiX0.5 xc-k5dygm xc-k5hiX0.5 | tee -a $O/result-v.txt
log "D1111V DONE"
