#!/bin/bash
# mix144.sh STEP ... -- (w144, #174-02, box2) the identity / VRAM / time containers on one trunk,
# B = /data/q27/w138/packs/v3-k4.125-r16k4-a8.rad (K4.125 i8 ladder trunk, R16 K4 head, A8; its token_embd plain bf16):
#   mix144.sh l3        M/k4-l3.rad      B + V2 3.00's Lloyd l3 token table (radmix)
#   mix144.sh t K       M/k4-tK.rad      B + the trellis donor's token table (D/emb-t-i8-K<k>.rad, built on ds8 by donor144.sh)
#   mix144.sh ref K     M/k4-tK-ref.rad  a copy of B with token_embd = bf16 oracle rows of the same trellis (embref.py)
#   mix144.sh check K   the confounder check: the weight directories of B, tK, tK-ref (rad-info -v without offsets) must
#                       agree on every entry but token_embd.weight, and their metadata (rad-info --meta) on every key
# radmix carries the trunk's A8 act keys from w138's r16k4-a8.mix. conv.lock exclusive + ram.lock shared for radmix / copy
# (as w135's a8anchor174.sh), .part then rename, skips finished outputs.
set -eu
Q=/data/q27 W=/data/q27/w144 M=/data/q27/w144/mix D=/data/q27/emb/trellis
B=$Q/w138/packs/v3-k4.125-r16k4-a8.rad V2=$Q/mixes/qwen3.8-27b-exl3-v2-3.00bpw.rad I=$Q/rad10-perf/install
mkdir -p $M
excl() { flock $Q/locks/conv.lock flock -s $Q/locks/ram.lock "$@"; }
mix() {  # out donor
  local o=$1 d=$2; [ -f $o ] && return 0
  { echo "token_embd.weight $d"; grep '^act ' $Q/w138/recipes/r16k4-a8.mix; } > $o.mix
  rm -f $o.part; excl nice -n 10 $Q/tools/radmix $o.part $B $o.mix > $o.mixlog 2>&1 && mv $o.part $o
  echo "$o: $(tail -n 1 $o.mixlog)"
}
case $1 in
l3)  mix $M/k4-l3.rad $V2 ;;
t)   mix $M/k4-t$2.rad $D/emb-t-i8-K$2.rad ;;
ref) o=$M/k4-t$2-ref.rad; [ -f $o ] && exit 0
     rm -f $o.part; excl cp $B $o.part
     excl nice -n 10 $Q/venv/bin/python -I $W/embref.py $D/t-i8/K$2/src mul1i8 $o.part 12 && mv $o.part $o ;;
check)
     for f in $B $M/k4-t$2.rad $M/k4-t$2-ref.rad; do
       $I/bin/rad-info -v $f | sed -n '/^weight directory/,$p' | grep -E '^ {6}[a-z]|^ {10}plane' | sed -E 's/@[0-9]+//g; s/ +/ /g' > $f.info
       $I/bin/rad-info --meta $f > $f.meta
     done
     for f in $M/k4-t$2.rad $M/k4-t$2-ref.rad; do
       n=$(diff <(awk '/^ plane / {if (!skip) print; next} /^ token_embd.weight/ {skip=1; next} {skip=0; print}' $B.info) \
                <(awk '/^ plane / {if (!skip) print; next} /^ token_embd.weight/ {skip=1; next} {skip=0; print}' $f.info) | grep -c '^[<>]' || true)
       m=$(diff $B.meta $f.meta | grep -c '^[<>]' || true)
       echo "$(basename $f): $n weight-directory lines differ outside token_embd ($(grep -c -v '^ plane ' $f.info) weights), $m metadata lines differ"
       grep -A4 '^ token_embd.weight' $f.info
     done ;;
esac
