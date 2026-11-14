#!/bin/bash
# flip.sh -- (w156) the tower matters: K4 scored against its own mini reference (expect KL ~ 0), then with every picture
# flipped top-bottom (same size, so the same expansion; the reference's manifest repointed) -> KL must rise
W=/data/q27/w156; Q=/data/q27; I=$W/rad/install; H=$W/rad/home; L=$Q/locks; CARD=7; M=$W/mini
rm -rf $M/goldflip; cp -r $M/gold $M/goldflip
/data/q27/venv/bin/python - $M <<'PY'
import json, sys, os
from PIL import Image, ImageOps
m = sys.argv[1]; k = json.load(open(f"{m}/goldflip/kld.json")); os.makedirs(f"{m}/flip", exist_ok=True)
for d in k["docs"]:
    nm = []
    for p in d.get("media", []):
        q = f"{m}/flip/" + os.path.basename(p).rsplit(".", 1)[0] + ".png"
        ImageOps.flip(Image.open(p).convert("RGB")).save(q); nm.append(q)
    if nm: d["media"] = nm
json.dump(k, open(f"{m}/goldflip/kld.json", "w"))
PY
( flock 5; flock -s 4
  for x in self:gold flip:goldflip; do
    ROCR_VISIBLE_DEVICES=$CARD RADIANCE_HOME=$H timeout 1800 $I/bin/radiance --model $W/packs/K4.rad --tp 1 --kv-cache-dtype bf16 \
      --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --mm-max-patches 4096 --kld-ref $M/${x#*:} --kld-out $M/k4-${x%%:*}.json > $M/k4-${x%%:*}.log 2>&1; echo "${x%%:*} rc $?"
  done ) 5>$L/card$CARD.lock 4>$L/ram.lock
