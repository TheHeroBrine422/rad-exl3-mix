#!/bin/bash
# embprep.sh BASE.rad TAG FMT... -- one fake-quantised copy a format: /root/emb/TAG-FMT.rad (log /root/emb/prep.log)
B=$1 T=$2; shift 2
mkdir -p /root/emb
OFF=$(PATH=/root/install-card/bin:$PATH rad-info -v $B 2>&1 | grep " token_embd.weight " | grep -o "@[0-9]*" | tr -d @)
[ -n "$OFF" ] || { echo "no token_embd offset in $B" >> /root/emb/prep.log; exit 1; }
for f in "$@"; do
  o=/root/emb/$T-$f.rad
  cp $B $o.part && python3 -I /root/embq.py $o.part $OFF 248320 5120 $f --write >> /root/emb/prep.log 2>&1 && mv $o.part $o \
    && echo "$T $f ready" >> /root/emb/prep.log || echo "$T $f FAIL" >> /root/emb/prep.log
done
