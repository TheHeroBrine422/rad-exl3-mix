#!/bin/bash
# kld.sh CARD TAG... -- whole-model KLD rows of the A8 pack on golden-search27c-c, one config per TAG (skips finished):
#   base10    today's kernels: rad10-perf install + home (5c9956c, ROCm 10.1), FUSE off
#   v10-off / v10-on    rad10-perf install + b10v home (the fix, ROCm 10.1), EXL3I_A8_FUSE 0 / 1
#   v714-off / v714-on  rad/install (7.14 core) + b714v home (the fix, pip ROCm 7.14.1), EXL3I_A8_FUSE 0 / 1
#   d10 / d714          the final tree (srcd: + FUSE on by default) on 10.1 / 7.14 as above, EXL3I_A8_FUSE unset
# Under card CARD's lock and ram.lock shared. Out /data/q27/w137/kld/TAG.json(.rows). (w137, #175)
C=$1; shift; Q=/data/q27; W=$Q/w137; P=${PACK:-$Q/w138/packs/v3-k4.125-bf16-a8.rad}; G=$Q/goldens/golden-search27c-c
mkdir -p $W/kld
for t in "$@"; do
  [ -f $W/kld/$t.json ] && continue
  case $t in
    base10) I=$Q/rad10-perf/install H=$Q/rad10-perf/home F=0 ;;
    v10-*)  I=$Q/rad10-perf/install H=$W/b10v/home F=$([ ${t#v10-} = on ] && echo 1 || echo 0) ;;
    v714-*) I=$Q/rad/install H=$W/b714v/home F=$([ ${t#v714-} = on ] && echo 1 || echo 0) ;;
    d10)    I=$Q/rad10-perf/install H=$W/b10d/home F= ;;
    d714)   I=$Q/rad/install H=$W/b714d/home F= ;;
    *) echo "bad tag $t"; continue ;;
  esac
  if [ -n "$F" ]; then E=(EXL3I_A8_FUSE=$F); else E=(-u EXL3I_A8_FUSE); fi
  flock -s $Q/locks/ram.lock flock $Q/locks/card$C.lock env "${E[@]}" timeout 1800 bash $Q/tools/score.sh $I $H $P $G $W/kld/$t $C
done
