#!/bin/bash
# f16ref.sh -- #164-02 (w136): the reference that separates the quantized-head path from the donors' bf16 rounding.
# F<h> = rad-convert (rad10-perf/home-pack) of the same K4 trunk recipe from a checkpoint view whose lm_head.weight is the
# R16 head's f16 dequant un-permuted to vocab order (mkf16ck.py; also checks donor hd-u<h> = bf16 of it); the head keeps
# the checkpoint's f16. KLD on golden-ver3-8k (cards CARD1 / CARD2, default 1 / 2: off r16check's timing card), then
# kldpair A-F (the path: exl3i trellis + R16 un-permute vs the same weights dense) and B-F (the donor's bf16 rounding).
# Restartable. Log /data/q27/w136/f16ref.log.
set -u; Q=/data/q27; O=$Q/rad10-perf; W=$Q/w136; L=$Q/locks; G=$Q/goldens/golden-ver3-8k; PY=$Q/venv/bin/python
S=$Q/tools/rad10perf; P=$W/packs; R=$W/res; C1=${CARD1:-1}; C2=${CARD2:-2}
exec >> $W/f16ref.log 2>&1; ulimit -n 65536; say() { echo "$(date -u +%T) $*"; }
pf() {   # pf H CARD
  local h=$1 c=$2 n=F$1 ck=$W/ck-f16-K$1
  [ -f $R/f16ck-$h.json ] || nice -n 10 $PY -I $S/mkf16ck.py $Q/models/qwen38-27b-bf16 $W/heads/q/R16s003/K$h $Q/w119/perms/R16.npy $ck \
      $W/heads/donors/hd-u$h.rad > $R/f16ck-$h.json
  say "f16ck $h: $(cat $R/f16ck-$h.json)"
  if [ ! -f $P/$n.rad ]; then
    ( flock -x 7; flock -x 6; say "pack $n: start"; t0=$(date +%s)
      cd $Q && RADIANCE_HOME=$O/home-pack RADIANCE_HOST_DEVICES=1 nice -n 10 $O/install/bin/rad-convert --no-error $ck \
        --recipe $W/recipes/K4.recipe --set quantization_config.quant_method=exl3 -o $P/$n.rad.part > $R/pack-$n.log 2>&1 && mv $P/$n.rad.part $P/$n.rad
      say "pack $n: rc $? $(stat -c %s $P/$n.rad 2>/dev/null) bytes, $(( $(date +%s) - t0 )) s" ) 7>$L/conv.lock 6>$L/ram.lock
  fi
  [ -f $R/ents-F$h.json ] || $PY -I $S/radents.py $P/A$h.rad $P/$n.rad > $R/ents-F$h.json; say "ents A$h F$h: $(cut -c1-400 $R/ents-F$h.json)"
  [ -f $R/kld-$n.json ] || ( flock 5; flock -s 6; say "kld $n (card $c)"
    ROCR_VISIBLE_DEVICES=$c RADIANCE_HOME=$O/home timeout 3600 $O/install/bin/radiance --model $P/$n.rad --tp 1 --kv-cache-dtype bf16 \
      --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $G --kld-out $R/kld-$n.json.part > $R/kld-$n.log 2>&1 \
      && mv $R/kld-$n.json.part.rows $R/kld-$n.json.rows && mv $R/kld-$n.json.part $R/kld-$n.json
    say "kld $n rc $?" ) 5>$L/card$c.lock 6>$L/ram.lock
}
pf 4 $C1 & pf 2.5 $C2 & wait
$PY -I $S/kldpair.py $G $R A4 F4 B4 F4 A2.5 F2.5 B2.5 F2.5 > $R/kldpair-f16.json 2> $R/kldpair-f16.err; say "kldpair-f16: $(cat $R/kldpair-f16.json $R/kldpair-f16.err)"
say "F16REF DONE"
