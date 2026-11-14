#!/bin/bash
# profq.sh CARD TAG HOME -- the profile queue for one home: per pack (K2.5, K4.125; env PACKS) dprof units (env UNITS,
# default d0 d8 p8): decode at ctx 0 / 8k (320 tokens), p8 the 8k prefill alone (1 token) -> res/prof/TAG-PACK-UNIT.log.
# Skips finished units. (rocprofv3 sees none of radiance's kernels: it writes its own AQL packets.) (w152, #179)
C=$1 TAG=$2 H=$3; Q=/data/q27; W=$Q/w152; P8=$Q/w137/prompts/ctx-8k.txt
mkdir -p $W/res/prof $W/res/kt; exec >> $W/logs/profq.log 2>&1
say() { echo "$(date -u +%T) [$TAG] $*"; }
for p in ${PACKS:-v3-k2.5-r16k2.5-a8 v3-k4.125-r16k4-a8}; do M=$Q/w138/packs/$p.rad
  for u in ${UNITS:-d0 d8 p8}; do o=$TAG-$p-$u
    case $u in
      d0) [ -s $W/res/prof/$o.log.resp ] || { bash $W/tools/dprof.sh $C $H $M $W/res/prof/$o.log; say "$o $(grep -c 'per-op timing, rank 0' $W/res/prof/$o.log) dumps"; } ;;
      d8) [ -s $W/res/prof/$o.log.resp ] || { bash $W/tools/dprof.sh $C $H $M $W/res/prof/$o.log $P8; say "$o $(grep -c 'per-op timing, rank 0' $W/res/prof/$o.log) dumps"; } ;;
      p8) [ -s $W/res/prof/$o.log.resp ] || { NTOK=1 bash $W/tools/dprof.sh $C $H $M $W/res/prof/$o.log $P8; say "$o $(grep -c 'per-op timing, rank 0' $W/res/prof/$o.log) dumps"; } ;;


    esac
  done
done
say DONE
