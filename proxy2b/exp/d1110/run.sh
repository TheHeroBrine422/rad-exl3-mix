#!/bin/bash
# run.sh -- d1110 (README.md): fresh verify golden for the 0.8B, re-score the adopted stack. 4070, gpu.lock per unit.
P=/workspace/rad-exl3-mix/proxy2b X3=$P/exp/d1103
source <(sed -n '/^P=/,/^step() /p' $X3/run.sh)
O=/data/yb/d1110 D3=/data/yb/d1103 D4=/data/yb/d1104 D5=/data/yb/d1105 D6=/data/yb/d1106; mkdir -p $O
f32() { for d in $O; do [ -f $d/$1.f32 ] && { echo $d/$1.f32; return; }; done; echo $O/$1.f32; }
wqd() { for d in $D6 $D5 $D4 $D3 $Y; do [ -d $d/q-$1/wqall ] && { echo $d/q-$1/wqall; return; }; [ -d $d/q-$1/wq ] && { echo $d/q-$1/wq; return; }; done; }
lock() { exec 9>/data/yb/gpu.lock; flock 9; }; unlock() { flock -u 9; }
V=/data/g08-verify1
if [ ! -f $V/rows.f32 ]; then
  idx=$(python3 -I - <<'PY'
import json, random
m = json.load(open("/data/challenge/calib-meta/kld.json")); p = json.load(open("/workspace/rad-exl3-mix/eval/pools/pools-v3.json"))["pool"]
v = [i for i, d in enumerate(m["docs"]) if p[i] == "verify"]; random.Random(1110).shuffle(v)
print(" ".join(f"{i}:2048" for i in sorted(v[:180])))
PY
)
  lock; python3 $P/mkgold.py $M /data/challenge/calib-meta $V $idx > $O/gold.log 2>&1; unlock
  log "g08-verify1: $(tail -1 $O/gold.log)"
fi
XG=/data/g08-xcorp1
if [ ! -f $XG/rows.f32 ]; then
  [ -f $O/xmeta/kld.json ] || CUDA_VISIBLE_DEVICES= python3 -I $P/exp/d1110/xgold.py /data/xcorp/v1/meta $O/xmeta 180 1110 >> $O/gold.log 2>&1
  n=$(python3 -c "import json; print(len(json.load(open('$O/xmeta/kld.json'))['docs']))")
  lock; python3 $P/mkgold.py $M $O/xmeta $XG $(seq 0 $((n - 1))) >> $O/gold.log 2>&1; unlock
  log "g08-xcorp1: $(tail -1 $O/gold.log)"
fi
for G in $V $XG; do gx=$([ $G = $XG ] && echo xc- || echo "")
for s in 0 1; do for k in 3 2; do x=$(sfx $s)
  prev=""
  for t in k${k}base k${k}edy k${k}edy_swj k${k}edyswjasym_r0 HEAL; do
    if [ $t = HEAL ]; then h=$D4/h-k${k}edyswjasym_r0$x; [ -d $h/wq ] || continue; t=k${k}edyswjasym_r0_heal; w=$h/wq; em=$(hm $h)
    else t=$t$x; w=$(wqd $t); em=$M; [ -n "$w" ] || { log "no $t"; continue; }; fi
    lock; ev $gx$t $w $G "" $em; unlock
    [ -n "$prev" ] && pair $gx$prev $gx$t $G; [ "$t" != "k${k}base$x" ] && pair $gx""k${k}base$x $gx$t $G; prev=$t
  done
done; done; done
log "D1110 DONE"
