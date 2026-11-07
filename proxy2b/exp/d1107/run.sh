#!/bin/bash
# run.sh -- d1107 (README.md), 0.8B, 4070; shares /data/yb/gpu.lock with hconv / d1105 / d1106. Idempotent.
P=/workspace/rad-exl3-mix/proxy2b X3=$P/exp/d1103 X=$P/exp/d1107
source <(sed -n '/^P=/,/^step() /p' $X3/run.sh)
X=$P/exp/d1107   # the sourced d1103 block sets X to d1103 (11-04 runs found no a1q.py)
O=/data/yb/d1107; mkdir -p $O
f32() { for d in $O /data/yb /data/yb/d1103 /data/yb/w10-hout-attrib; do [ -f $d/$1.f32 ] && { echo $d/$1.f32; return; }; done; echo $O/$1.f32; }
wqd() { for d in /data/yb /data/yb/d1103 /data/yb/w10-hout-attrib; do [ -d $d/q-$1/wq ] && { echo $d/q-$1/wq; return; }; done; }
lock() { exec 9>/data/yb/gpu.lock; flock 9; }; unlock() { flock -u 9; }
DOWN=$O/down.txt; for l in $(seq 0 23); do echo "model.language_model.layers.$l.mlp.down_proj"; done > $DOWN
for k in 2 3 4 5; do for s in 0 1; do b=k${k}base$(sfx $s); bw=$(wqd $b); [ -n "$bw" ] || { log "no $b"; continue; }
  [ -f $(f32 $b) ] || { lock; ev $b $bw; unlock; }
  for arm in u ug; do t=k${k}a1$arm$(sfx $s); d=$O/$t
    [ -f $O/$t.f32 ] && { pair $b $t; continue; }
    [ -f $d/res.json ] || CUDA_VISIBLE_DEVICES= python3 -I $X/a1q.py res /data/models/qwen35-08b $bw $d $arm >> $d.log 2>&1 || { log "res FAIL $t"; continue; }
    lock
    [ $(ls $d/q/wq/*.safetensors 2>/dev/null | wc -l) -ge 24 ] || Q2B_CKPT=$d/ckpt $P/q2b.sh 08 $k i8 $d/q --only $DOWN --seed-offset $s >> $d.log 2>&1
    unlock
    [ -f $d/wqall/.done ] || CUDA_VISIBLE_DEVICES= python3 -I $X/a1q.py expand $d $d/q/wq $bw >> $d.log 2>&1 || { log "expand FAIL $t"; continue; }
    lock; ev $t $d/wqall; unlock; pair $b $t
  done; done
  python3 $X3/summ.py $O k${k}base k${k}a1u k${k}base k${k}a1ug k${k}a1u k${k}a1ug | tee -a $O/result.txt
done
log "D1107 DONE"
