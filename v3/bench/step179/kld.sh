#!/bin/bash
# kld.sh CARD PACKNAME|/PACK/PATH.rad TAG... -- whole-model KLD rows of box2 w138/packs/PACKNAME.rad on golden-search27c-c (a kernel
# numerics check: rows compared between homes, never a selection), one config per TAG = HOMENAME-MODE (skips finished):
#   HOMENAME  b10<HOMENAME>/home under /data/q27/w152 (base = qc-exl3-v3 179220f), rad10-perf install (ROCm 10.1);
#             b714<HOMENAME>/home with the frozen 7.14 install /data/q27/rad/install when HOMENAME ends in -714
#   MODE      def: score.sh's settings (prefill chunks: the A8 path); c64: --max-num-batched-tokens 64 (64-row steps:
#             the bf16 norms and the attention prologue's decode band)
# Env FUSE=... passes QWEN35_EXL3_FUSE. Under card CARD's lock and ram.lock shared. Out res/kld/PACK-TAG.json(.rows). (w152)
C=$1 PN=$2; shift 2; Q=/data/q27; W=$Q/w152; P=$Q/w138/packs/$PN.rad
case $PN in /*) P=$PN; PN=$(basename $PN .rad) ;; esac   # a pack by path (the mad packs)
 G=$Q/goldens/golden-search27c-c
mkdir -p $W/res/kld; ulimit -n 65536
for t in "$@"; do
  o=$W/res/kld/$PN-$t; [ -f $o.json ] && continue
  hn=${t%-*} mode=${t##*-}
  case $hn in *-714) I=$Q/rad/install H=$W/b714${hn%-714}/home ;; *) I=$Q/rad10-perf/install H=$W/b10$hn/home ;; esac
  X=(); [ $mode = c64 ] && X=(--max-num-batched-tokens 64)
  E=(); [ -n "${FUSE:-}" ] && E=(QWEN35_EXL3_FUSE=$FUSE)
  s=$(date +%s)
  flock -s $Q/locks/ram.lock flock $Q/locks/card$C.lock env "${E[@]}" ROCR_VISIBLE_DEVICES=$C RADIANCE_HOME=$H timeout 3600 \
    $I/bin/radiance --model $P --tp 1 --kv-cache-dtype bf16 --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 \
    "${X[@]}" --kld-ref $G --kld-out $o.json.part > $o.log 2>&1; rc=$?
  [ $rc = 0 ] && mv $o.json.part $o.json && mv $o.json.part.rows $o.json.rows
  echo "$(date -u +%T) kld $PN-$t rc=$rc $(( $(date +%s) - s ))s"
done
