#!/bin/bash
# push-ds8.sh -- (w157, #174-03; NOT run: the lead decides) the menu's 36 trellis embedding donors onto ds8's
# /data/q27/emb/trellis/ at the menu's paths, as slim containers (token_embd.weight only, byte-identical to the box2
# donors': embcmp157.py): the 21 eighth donors (already slim on box2) + the 15 half-step / mul1 donors' slim copies
# (box2 /data/q27/w157/slim/). ~22 GB over the 10 GbE link. Run on ds8 (w144's link144.env transport).
. /data/q27/w144/link144.env
D=/data/q27/emb/trellis; mkdir -p $D
for k in 1.625 1.75 1.875 2.125 2.25 2.375 2.625 2.75 2.875 3.125 3.25 3.375 3.625 3.75 3.875 4.125 4.25 4.375 4.625 4.75 4.875; do
  rsync -a --partial -e "$R" $B:$D/emb-t-i8-K$k.rad $D/ < /dev/null
done
rsync -a --partial -e "$R" "$B:/data/q27/w157/slim/emb-t-*.rad" $D/ < /dev/null
ls $D/emb-t-*.rad | wc -l
