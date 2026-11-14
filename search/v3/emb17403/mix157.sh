#!/bin/bash
# mix157.sh ARM... -- (w157, #174-03, box2 CPU) the containers of RULES §1 / §3 / §5, W/mix/<ARM>.rad:
#   B       the mad K2.5 pack (ds8 packs-mad q27v3-635bac02-k2.5-mad-s6c57a3d1.rad, bf16 head + embedding) with
#           `act <w> a8g128` on the 336 trunk linears (w138's r16k4-a8.mix act lines, as #177-03's A8 mixes)
#   T       B + token_embd.weight from w144's donor emb-t-i8-K2.5.rad (full container)
#   Ts      the same from the slim K2.5 donor W/dtest/slim-t2.5.rad (slim157.sh's form): must equal T byte for byte
#   R       a copy of B whose bf16 token table = the oracle rows of t-i8/K2.5 (w144 embref.py, libexl3iref.so)
#   t<K>    B + token_embd.weight from /data/q27/emb/trellis/emb-t-i8-K<k>.rad (eighth donors: slim157.sh)
#   P3t4    w153's P3-mx (drafter pack, copied to W/in) + token_embd.weight from emb-t-i8-K4.rad
#   check A B   confounder check: weight directories (rad-info -v without offsets) and metadata of W/mix/A and B agree
#           on every entry but token_embd.weight (w144's mix144.sh check)
# radmix = W/tools/radmix (the repo's, with add / drop). conv.lock exclusive + ram.lock shared, .part then rename,
# finished outputs skipped.
set -eu
Q=/data/q27 W=/data/q27/w157 M=/data/q27/w157/mix D=/data/q27/emb/trellis I=/data/q27/rad10-perf/install
BASE=$W/in/q27v3-635bac02-k2.5-mad-s6c57a3d1.rad P3=$W/in/P3-mx.rad
mkdir -p $M
excl() { flock $Q/locks/conv.lock flock -s $Q/locks/ram.lock "$@"; }
mix() {  # out base donor|- [act]
  local o=$1 b=$2 d=$3 a=${4:-}; [ -f $o ] && return 0
  { [ "$d" = - ] || echo "token_embd.weight $d"; [ -z "$a" ] || grep '^act ' $Q/w138/recipes/r16k4-a8.mix; } > $o.mix
  rm -f $o.part; excl nice -n 10 $W/tools/radmix $o.part $b $o.mix > $o.mixlog 2>&1 && mv $o.part $o
  echo "$o: $(tail -n 1 $o.mixlog)"
}
info() { [ -f $1.info ] || { $I/bin/rad-info -v $1 | sed -n '/^weight directory/,$p' | grep -E '^ {6}[a-z]|^ {10}plane' \
           | sed -E 's/@[0-9]+//g; s/ +/ /g' > $1.info; $I/bin/rad-info --meta $1 > $1.meta; }; }
for a in "$@"; do
  case $a in
  B)    mix $M/B.rad $BASE - act ;;
  T)    mix $M/T.rad $BASE $D/emb-t-i8-K2.5.rad act ;;
  Ts)   mix $M/Ts.rad $BASE $W/dtest/slim-t2.5.rad act; cmp $M/Ts.rad $M/T.rad && echo "Ts == T (byte-identical)" ;;
  R)    o=$M/R.rad; [ -f $o ] && continue; [ -f $M/B.rad ] || { echo "R needs B"; exit 1; }
        rm -f $o.part; excl cp $M/B.rad $o.part
        excl env EXL3IREF=$Q/w144/libexl3iref.so nice -n 10 $Q/venv/bin/python -I $Q/w144/embref.py $D/t-i8/K2.5/src mul1i8 $o.part 12 \
          && mv $o.part $o && echo "$o: oracle rows written" ;;
  P3t4) mix $M/P3t4.rad $P3 $D/emb-t-i8-K4.rad ;;
  t*)   k=${a#t}; [ -f $D/emb-t-i8-K$k.rad ] || { echo "$a: no donor yet"; continue; }; mix $M/$a.rad $BASE $D/emb-t-i8-K$k.rad act ;;
  check) shift; A=$M/$1.rad; Bb=$M/$2.rad; info $A; info $Bb
        sk='/^ plane / {if (!skip) print; next} /^ token_embd.weight/ {skip=1; next} {skip=0; print}'
        n=$(diff <(awk "$sk" $A.info) <(awk "$sk" $Bb.info) | grep -c '^[<>]' || true)
        m=$(diff $A.meta $Bb.meta | grep -c '^[<>]' || true)
        echo "check $1 vs $2: $n weight-directory lines differ outside token_embd ($(grep -c -v '^ plane ' $A.info) weights), $m metadata lines differ"
        grep -A0 '^ token_embd.weight' $A.info $Bb.info; break ;;
  *) echo "unknown arm $a"; exit 2 ;;
  esac
done
