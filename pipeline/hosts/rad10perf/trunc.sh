#!/bin/bash
# trunc.sh [trunc|stoch] -- #164-02 (w136): B4$X = A4 with output.weight from hd-u4$X.rad (donortrunc.py: the same R16 K4 head, bf16 by
# truncation), KLD on golden-ver3-8k (card CARD1, default 1), then kldpair B4-B4$X (the bf16 rounding's own size), A4-B4$X,
# A4-B4 (again, for the table). Restartable. Log /data/q27/w136/trunc.log.
set -u; Q=/data/q27; O=$Q/rad10-perf; W=$Q/w136; L=$Q/locks; G=$Q/goldens/golden-ver3-8k; PY=$Q/venv/bin/python
S=$Q/tools/rad10perf; P=$W/packs; R=$W/res; c=${CARD1:-1}; M=${1:-trunc}; X=${M:0:1}   # t / s
exec >> $W/$M.log 2>&1; say() { echo "$(date -u +%T) $*"; }
[ -f $W/heads/donors/hd-u4$X.rad ] || { nice -n 10 $PY -I $S/donortrunc.py $W/heads/donors/hd-u4.rad $W/heads/q/R16s003/K4 \
    $Q/w119/perms/R16.npy $W/heads/donors/hd-u4$X.rad $M > $R/donortrunc-4$X.json; say "donortrunc: $(cat $R/donortrunc-4$X.json)"; }
echo "output.weight $W/heads/donors/hd-u4$X.rad" > $W/recipes/B4$X.mix
[ -f $P/B4$X.rad ] || ( flock -s 6; $Q/tools/radmix $P/B4$X.rad.part $P/A4.rad $W/recipes/B4$X.mix > $R/mix-B4$X.log 2>&1 && mv $P/B4$X.rad.part $P/B4$X.rad
    say "mix B4$X: $(tail -n 1 $R/mix-B4$X.log)" ) 6>$L/ram.lock
[ -f $R/ents-B4$X.json ] || $PY -I $S/radents.py $P/A4.rad $P/B4$X.rad > $R/ents-B4$X.json; say "ents A4 B4$X: $(cut -c1-300 $R/ents-B4$X.json)"
[ -f $R/kld-B4$X.json ] || ( flock 5; flock -s 6; say "kld B4$X (card $c)"
  ROCR_VISIBLE_DEVICES=$c RADIANCE_HOME=$O/home timeout 3600 $O/install/bin/radiance --model $P/B4$X.rad --tp 1 --kv-cache-dtype bf16 \
    --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $G --kld-out $R/kld-B4$X.json.part > $R/kld-B4$X.log 2>&1 \
    && mv $R/kld-B4$X.json.part.rows $R/kld-B4$X.json.rows && mv $R/kld-B4$X.json.part $R/kld-B4$X.json
  say "kld B4$X rc $?" ) 5>$L/card$c.lock 6>$L/ram.lock
$PY -I $S/kldpair.py $G $R B4 B4$X A4 B4$X A4 B4 > $R/kldpair-$M.json 2> $R/kldpair-$M.err; say "kldpair-$M: $(cat $R/kldpair-$M.json $R/kldpair-$M.err)"
say "${M^^} DONE"
