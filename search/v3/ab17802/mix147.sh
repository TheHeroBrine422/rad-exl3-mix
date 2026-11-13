#!/bin/bash
# mix147.sh SIZE -- (w147, #178-02, ds8) the containers of the before / after rows at SIZE (2.5: v3s171-2.5-block-00626):
#   B   the incumbent's hostd .mix + `act <w> a8g128` on every blk.* trunk weight (w142's arm B form), stock radmix
#   Bx  the same mixfile through radmix147 (the add / drop patch): must be byte-identical to B (cmp)
#   T4  B's lines + per GDN layer `add blk.N.ssm_ab.weight W/donor-ab-K4.rad`, `drop blk.N.ssm_a.weight` / `ssm_b`,
#       `act blk.N.ssm_ab.weight a8g128` (radmix147)
# conv.lock + ram.lock shared (as run178.sh). Restartable (skips finished containers). Out W/mix/<SIZE>-{B,T4}.rad.
set -u
Q=/data/q27; W=$Q/w147; mkdir -p $W/mix
BASE=$Q/mixes/qwen3.8-27b-exl3-v2-3.00bpw.rad
declare -A TAG=([2.5]=v3s171-2.5-block-00626 [4.0]=v3s171-4-block-00420)
s=$1; t=${TAG[$s]}
log() { echo "$(TZ=America/Chicago date '+%m-%d %H:%M:%S CST') $*" | tee -a $W/mix/mix147.log; }
mf=$W/mix/$s-B.mix
{ cat $Q/mixes/$t.mix; awk '$1 ~ /^blk\./ {print "act", $1, "a8g128"}' $Q/mixes/$t.mix; } > $mf
{ cat $mf
  for l in $(seq 0 63); do [ $((l % 4)) = 3 ] && continue
    echo "add blk.$l.ssm_ab.weight $W/donor-ab-K4.rad"; echo "drop blk.$l.ssm_a.weight"; echo "drop blk.$l.ssm_b.weight"
    echo "act blk.$l.ssm_ab.weight a8g128"; done; } > $W/mix/$s-T4.mix
mk() {  # out tool mixfile
  [ -f $1 ] && return 0; rm -f $1.part
  flock $Q/locks/conv.lock flock -s $Q/locks/ram.lock $2 $1.part $BASE $3 > $1.mixlog 2>&1 && mv $1.part $1 \
    && log "$(basename $1): $(tail -n 1 $1.mixlog)" || { log "radmix $(basename $1) FAILED"; return 1; }
}
mk $W/mix/$s-B.rad $Q/tools/radmix $mf
mk $W/mix/$s-Bx.rad $W/radmix147 $mf
cmp $W/mix/$s-B.rad $W/mix/$s-Bx.rad && log "$s: radmix147 = radmix on the B mixfile (byte-identical)" && rm -f $W/mix/$s-Bx.rad
mk $W/mix/$s-T4.rad $W/radmix147 $W/mix/$s-T4.mix
