#!/bin/bash
# w79s.sh -- #89-06c's step-0 branch (../w79-heal45/README.md, rule written by w79 before data; not changed here):
# step 0 flagged the fold path (K5 lr 3e-5: student 0.9930 vs fold 1.1441, 11-08 18:42 UTC), so "every K4 / K5 arm is
# then read on the student path (stack through the same wrappers at identity)" and "K5 lr 1e-4 seed 0 / 1 are re-read
# on the student path with the K4 rule". w79's run.sh scores only the fold path; this scores the student path
# (w79's evalm.py) of the stacks and healed arms on g08-search1 and g08-verify1 / 2 into /data/yb/w90/w79s, then
# readout79s.py applies the K4 rule to K4 (lr picked by held-out, w79's step 2) and K5 (lr 1e-4). Idempotent.
P=/workspace/rad-exl3-mix/proxy2b W79=$P/exp/w79-heal45 W=$P/exp/w90-a4heal
S=/data/g08-search1 V1=/data/g08-verify1 V2=/data/g08-verify2 D5=/data/yb/d1105 O7=/data/yb/w79 O=/data/yb/w90/w79s
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
mkdir -p $O; cd /tmp
log() { echo "$(date -u +%T) w79s $*" | tee -a /data/yb/w90/result.txt; }
sfx() { [ $1 = 0 ] && echo "" || echo "_s$1"; }
gs() { case $1 in $S) echo "";; $V1) echo .v1;; $V2) echo .v2;; esac; }
st() { echo $D5/q-k$1edyswjasym_r0$(sfx $2)/wqall; }
hdir() { for d in $O7 $D5; do [ -f $d/h-$1/params.pt ] && { echo $d/h-$1; return; }; done; }
em() { local x=$(gs $4); [ -f $O/$1$x.f32 ] && return 0
  [ "$2" = - ] || [ -f "$2" ] || { log "FAIL no params for $1 ($2)"; return 1; }
  python3 $W79/evalm.py $2 $3 $4 $O/$1$x.f32 > $O/$1$x.log 2>&1
  [ -f $O/$1$x.f32.part ] && [ ! -f $O/$1$x.f32 ] && mv $O/$1$x.f32.part $O/$1$x.f32
  [ -f $O/$1$x.f32 ] && log "$1$x: $(tail -n 1 $O/$1$x.log | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')" || { log "FAIL evalm $1$x ($O/$1$x.log)"; return 1; }; }
pick=$(grep -o "K4 lr picked by held-out: \S*" $O7/result.txt | tail -n 1 | awk '{print $NF}')
[ -n "$pick" ] || { log "FAIL no K4 lr pick in $O7/result.txt"; exit 1; }
echo $pick > $O/pick
exec 9>/data/yb/gpu.lock; flock 9
for s in 0 1; do
  for g in $S $V1 $V2; do
    em m-k4stack$(sfx $s) - $(st 4 $s) $g; em m-k5stack$(sfx $s) - $(st 5 $s) $g
    t=k4edyswjasym_r0_lr$pick$(sfx $s); em m-$t $(hdir $t)/params.pt $(st 4 $s) $g
    t=k5edyswjasym_r0_lr1e-4$(sfx $s); em m-$t $(hdir $t)/params.pt $(st 5 $s) $g
  done
  for lr in 3e-4 1e-4; do [ $lr = $pick ] && continue                     # the other K4 lr: search only, reported
    t=k4edyswjasym_r0_lr$lr$(sfx $s); em m-$t $(hdir $t)/params.pt $(st 4 $s) $S; done
done
flock -u 9
python3 $W/readout79s.py $O $P/pcmp.py $S | tee -a /data/yb/w90/result.txt
log "W79S DONE"
