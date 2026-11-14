#!/bin/bash
# mp.sh CARD "K..." -- w151's stage-B mad levels (/data/q27/mad-packs, #177-03) through this branch: each level packed with
# rad-convert (rad10-perf/install, build b8's home: this branch's exl3quant -- the exl3i importer's codebook=mad and its
# stamp check -- and qwen35_exl3, which reads encoding version 4; rad10/home-pack's refuses mad/...), act a8g128 on the 336 trunk linears (radmix + w138's a8.mix) -> packs/<level>-a8.rad; w151's shim pack
# mixed the same way -> packs/<level>-shim-a8.rad; whole-model KLD rows of both (shim: w139's home-hq, the reference
# dequant path; mad: build b8) under card CARD's lock; rows compared. Restartable. (w150, #177-02)
C=$1; KS=$2; Q=/data/q27; W=$Q/w150; MP=$Q/mad-packs; L=$Q/locks; HP=$W/b8/home
exec >> $W/logs/mp.log 2>&1; say() { echo "$(date -u +%T) $*"; }
for K in $KS; do
  N=q27v3-635bac02-k$K-mad; P=$W/packs/$N
  if [ ! -f $P-a8.rad ]; then
    if [ ! -f $P.rad ]; then
      ( flock -x 7; flock -s 6
        while [ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -lt 16 ]; do sleep 20; done; say "pack $N"
        cd $Q && RADIANCE_HOME=$HP RADIANCE_HOST_DEVICES=1 nice -n 10 $Q/rad10-perf/install/bin/rad-convert --no-error \
          models/qwen38-27b-bf16 --recipe $MP/$N.recipe --set quantization_config.quant_method=exl3 -o $P.rad.part > $P.pack.log 2>&1 \
          && mv $P.rad.part $P.rad; say "pack $N rc $?" ) 7>$L/conv.lock 6>$L/ram.lock
    fi
    [ -f $P.rad ] || { say "PACK FAIL $N"; rm -f $P.rad.part; exit 1; }
    ( flock -s 6; $Q/tools/radmix $P-a8.rad.part $P.rad $Q/w138/recipes/a8.mix > $P.mix.log 2>&1 && mv $P-a8.rad.part $P-a8.rad
      say "mix $N: $(tail -n 1 $P.mix.log)" ) 6>$L/ram.lock
    [ -f $P-a8.rad ] && rm -f $P.rad
  fi
  if [ ! -f $P-shim-a8.rad ]; then
    ( flock -s 6; $Q/tools/radmix $P-shim-a8.rad.part $MP/$N-shim.rad $Q/w138/recipes/a8.mix > $P-shim.mix.log 2>&1 \
      && mv $P-shim-a8.rad.part $P-shim-a8.rad; say "mix shim $N: $(tail -n 1 $P-shim.mix.log)" ) 6>$L/ram.lock
  fi
  $W/tools/kld150.sh $C k$K-w151-shim-hq=$Q/w139/home-hq=$P-shim-a8.rad k$K-w151-mad-b8=$W/b8/home=$P-a8.rad
  cmp -s $W/res/kld/k$K-w151-shim-hq.json.rows $W/res/kld/k$K-w151-mad-b8.json.rows && say "K$K: ROWS IDENTICAL" || say "K$K: ROWS DIFFER"
done
say "mp done"
