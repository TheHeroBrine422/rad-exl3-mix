#!/bin/bash
# healfix.sh -- w11 (lead launches, 4070 GPU, ~25 s per eval): re-score d1103's healed arms WITH their folded norms
# (fold.py OUT/model: q_norm / k_norm / final-norm gains), which d1103/run.sh's heal step and run2.sh's heal1 drop
# (their ev() always scores on the stock checkpoint). Run after those steps; idempotent. Writes
# /data/yb/w11-review/<tag>_healn.f32 and pairs (vs the stack, vs the norm-less score) to /data/yb/w11-review/healfix.txt.
P=/workspace/rad-exl3-mix/proxy2b S=/data/g08-search1 D=/data/yb/d1103 O=/data/yb/w11-review
mkdir -p $O; cd /tmp; log() { echo "$(date -u +%T) $*" | tee -a $O/healfix.txt; }
for t in k3edyasym k2edyasym k3edyasym_s1 k2edyasym_s1; do d=$D/h-$t
  [ -f $d/wq/.done ] || { log "$t: not folded yet"; continue; }
  [ -f $d/model/.done ] || { log "$t: no norms trained, nothing to fix"; continue; }
  [ -f $O/${t}_healn.f32 ] || log "${t}_healn: $(python3 $P/evalq.py $d/model $S --wq $d/wq --out $O/${t}_healn.f32 2>/dev/null | tail -1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"
  for a in $D/$t.f32 $D/${t}_heal.f32; do [ -f $a ] && log "pair $(basename $a .f32) -> ${t}_healn: $(python3 $P/pcmp.py $S $a $O/${t}_healn.f32 | grep -o '"ratio": [0-9.]*, "diff": [0-9.e-]*, "se": [0-9.e-]*, "z": [0-9.e-]*')"; done
done
log "HEALFIX DONE"
