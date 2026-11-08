#!/bin/bash
# run.sh -- w79 #89-06c (README.md): settle healing at K4 / K5 on the 0.8B edy -> swj -> asym rho 0 stacks (d1105).
# 4070, one GPU unit at a time under /data/yb/gpu.lock, VRAM_CAP_GB 7.2. Idempotent (score files, params.pt, wq/.done);
# rerun after any kill. Stop: touch /data/yb/w79/STOP. NOT LAUNCHED (4070 off limits since 11-08 1:20 AM CST).
P=/workspace/rad-exl3-mix/proxy2b W=$P/exp/w79-heal45 E6=$P/exp/w06-heal
M=/data/models/qwen35-08b S=/data/g08-search1 V1=/data/g08-verify1 V2=/data/g08-verify2 D5=/data/yb/d1105 O=/data/yb/w79
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
export POOLS=/workspace/rad-exl3-mix/eval/pools/pools-v3.json
mkdir -p $O/check; cd /tmp
log() { echo "$(date -u +%T) $*" | tee -a $O/result.txt; }
stop() { [ -f $O/STOP ]; }
lock() { exec 9>/data/yb/gpu.lock; flock 9; }; unlock() { flock -u 9; }
sfx() { [ $1 = 0 ] && echo "" || echo "_s$1"; }
gs() { case $1 in $S) echo "";; $V1) echo .v1;; $V2) echo .v2;; esac; }          # score-file suffix per golden
f32() { for d in $O $D5 /data/yb/d1104; do [ -f $d/$1.f32 ] && { echo $d/$1.f32; return; }; done; echo $O/$1.f32; }
# ev TAG WQ GOLDEN [MODEL]: fold path (evalq: f16 wq cast to bf16); em TAG PARAMS WQ GOLDEN: in-memory student (evalm)
ev() { local x=$(gs $3); [ -f $O/$1$x.f32 ] || [ -f $D5/$1$x.f32 ] && return
  log "$1$x: $(python3 $P/evalq.py ${4:-$M} $3 --wq $2 --out $O/$1$x.f32 2>/dev/null | tail -1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"; }
em() { local x=$(gs $4); [ -f $O/$1$x.f32 ] && return
  log "$1$x: $(python3 $W/evalm.py $2 $3 $4 $O/$1$x.f32 2>&1 | tail -1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"; }
pair() { local g=${3:-$S} x=$(gs ${3:-$S}); local a=$(f32 $1$x) b=$(f32 $2$x); [ -f $a ] && [ -f $b ] || { log "pair $1 -> $2$x: missing"; return; }
  log "pair $1 -> $2$x: $(python3 $P/pcmp.py $g $a $b | grep -o '"ratio": [0-9.]*, "diff": [0-9.e-]*, "se": [0-9.e-]*, "z": [0-9.e-]*')"; }
hm() { [ -f $1/model/.done ] && echo $1/model || echo $M; }
st() { echo $D5/q-k$1edyswjasym_r0$(sfx $2)/wqall; }                                # stack wq (150 files)
hdir() { for d in $O $D5; do [ -f $d/h-$1/params.pt ] && { echo $d/h-$1; return; }; done; echo $O/h-$1; }
# heal1 K SEED LR: heal the stack (d1105's recipe, --lr) -> O/h-TAG, fold, score on search
heal1() { local k=$1 s=$2 lr=$3 t=k$1edyswjasym_r0_lr$3$(sfx $2); local d=$(hdir $t)
  [ -f $(f32 $t) ] && return; lock
  [ -f $d/params.pt ] || python3 $E6/heal.py train $d --wq $(st $k $s) --lr $lr >> $d.log 2>&1 || { log "heal FAIL $t"; unlock; return 1; }
  log "$t: $(grep '^HEAL DONE' $d.log | tail -1)"
  [ -f $d/wq/.done ] || python3 $E6/fold.py $d/params.pt $(st $k $s) $d $M >> $d.log 2>&1 || { log "fold FAIL $t"; unlock; return 1; }
  ev $t $d/wq $S $(hm $d); unlock; pair k${k}edyswjasym_r0$(sfx $s) $t; }

# step 0 -- confounder check (README): fold vs in-memory student; re-rounding jitter at K4 / K5
lock
em m-k5stack - $(st 5 0) $S
for lr in 3e-5 1e-4; do em m-k5lr$lr $D5/h-k5edyswjasym_r0_lr$lr/params.pt $(st 5 0) $S; done
em m-k4stack - $(st 4 0) $S
em m-k4lr3e-4 $D5/h-k4edyswjasym_r0_lr3e-4/params.pt $(st 4 0) $S
for k in 4 5; do j=$O/jit-k$k; [ -f $j/wq/.done ] || python3 $E6/fold.py - $(st $k 0) $j $M --jitter 0.002 > $j.log 2>&1; ev k${k}jit $j/wq $S; done
unlock
for k in 4 5; do pair k${k}edyswjasym_r0 k${k}jit; pair k${k}edyswjasym_r0 m-k${k}stack; done
for lr in 3e-5 1e-4; do pair m-k5stack m-k5lr$lr; done; pair m-k4stack m-k4lr3e-4
stop && exit 0
# step 1 -- seed 1 for both K4 lrs (seed 0 exists in d1105), K5 lr 1e-4 seed 1 (read only if step 0 clears the fold)
heal1 4 1 3e-4; stop && exit 0; heal1 4 1 1e-4; stop && exit 0; heal1 5 1 1e-4; stop && exit 0
# step 2 -- pick the K4 lr by mean held-out gain (heal logs); verify-pool check of the picked lr, both seeds
pick=$(for lr in 3e-4 1e-4; do echo "$lr $(cat $(hdir k4edyswjasym_r0_lr$lr).log $(hdir k4edyswjasym_r0_lr${lr}_s1).log | grep -o '^HEAL DONE.*(\S*%)' | grep -o '[-+][0-9.]*%' | tr -d '%' | awk '{s+=$1} END {print s/NR}')"; done | sort -k2 -g | head -1 | cut -d' ' -f1)
log "K4 lr picked by held-out: $pick"
for s in 0 1; do t=k4edyswjasym_r0_lr$pick$(sfx $s); d=$(hdir $t); b=k4edyswjasym_r0$(sfx $s)
  lock; for g in $V1 $V2; do ev $b $(st 4 $s) $g; ev $t $d/wq $g $(hm $d); done; unlock
  for g in $V1 $V2; do pair $b $t $g; done; done
python3 $W/readout.py $O | tee -a $O/result.txt
log "W79 HEAL DONE"
