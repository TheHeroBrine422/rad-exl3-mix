#!/bin/bash
# qdonor.sh -- #164-02 (w136): can the search splice a QUANTIZED R16 head the way it splices its bf16 donors? C4 = radmix
# of B4 (bf16 donor head) with output.weight taken from A4 (the rad-convert pack with the exl3i R16 head); then radents A4
# vs C4 (want: no payload differs) and KLD rows (card CARD1, default 1) byte-identical to A4's. Log /data/q27/w136/qdonor.log.
set -u; Q=/data/q27; O=$Q/rad10-perf; W=$Q/w136; L=$Q/locks; G=$Q/goldens/golden-ver3-8k; PY=$Q/venv/bin/python
S=$Q/tools/rad10perf; P=$W/packs; R=$W/res; c=${CARD1:-1}
exec >> $W/qdonor.log 2>&1; say() { echo "$(date -u +%T) $*"; }
echo "output.weight $P/A4.rad" > $W/recipes/C4.mix
[ -f $P/C4.rad ] || ( flock -s 6; $Q/tools/radmix $P/C4.rad.part $P/B4.rad $W/recipes/C4.mix > $R/mix-C4.log 2>&1 && mv $P/C4.rad.part $P/C4.rad
    say "mix C4: $(tail -n 1 $R/mix-C4.log) $(stat -c %s $P/C4.rad) bytes (A4 $(stat -c %s $P/A4.rad))" ) 6>$L/ram.lock
[ -f $R/ents-C4.json ] || $PY -I $S/radents.py $P/A4.rad $P/C4.rad > $R/ents-C4.json; say "ents A4 C4: $(cut -c1-400 $R/ents-C4.json)"
[ -f $R/kld-C4.json ] || ( flock 5; flock -s 6; say "kld C4 (card $c)"
  ROCR_VISIBLE_DEVICES=$c RADIANCE_HOME=$O/home timeout 3600 $O/install/bin/radiance --model $P/C4.rad --tp 1 --kv-cache-dtype bf16 \
    --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $G --kld-out $R/kld-C4.json.part > $R/kld-C4.log 2>&1 \
    && mv $R/kld-C4.json.part.rows $R/kld-C4.json.rows && mv $R/kld-C4.json.part $R/kld-C4.json
  say "kld C4 rc $?" ) 5>$L/card$c.lock 6>$L/ram.lock
$PY -I $S/kldpair.py $G $R A4 C4 > $R/kldpair-C4.json 2> $R/kldpair-C4.err; say "kldpair A4 C4: $(cat $R/kldpair-C4.json $R/kldpair-C4.err)"
say "QDONOR DONE"
