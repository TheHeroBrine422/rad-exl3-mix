#!/bin/bash
# fakeradiance.sh -- stand-in for bin/radiance in agent32.sh's dry run: reads --model / --kld-ref / --kld-out, checks the model
# and golden exist, writes OUT (a stub report) and OUT.rows (random {kl, nll_q, nll_p, top1} rows, one per golden row).
while [ $# -gt 0 ]; do case $1 in --model) M=$2; shift;; --kld-ref) G=$2; shift;; --kld-out) O=$2; shift;; esac; shift; done
[ -f "$M" ] && [ -f "$G/kld.json" ] || { echo "fake radiance: model $M or golden $G missing"; exit 1; }
echo "fake radiance: model $M ref $G out $O (ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES RADIANCE_HOME=$RADIANCE_HOME)"
/data/venv-exl3/bin/python -I - "$G" "$O" "$M" <<'PY'
import json, sys, zlib
import numpy as np
g, o, m = sys.argv[1:4]
n = json.load(open(f"{g}/kld.json"))["rows"]
r = np.random.default_rng(zlib.crc32(m.encode()))
s = np.stack([r.exponential(0.02, n), r.exponential(1, n), r.exponential(1, n), (r.random(n) < 0.9)], 1).astype(np.float32)
s.tofile(o + ".rows"); json.dump({"fake": True, "model": m, "ref_dir": g}, open(o, "w"))
PY
