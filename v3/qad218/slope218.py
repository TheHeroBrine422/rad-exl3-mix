#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""slope218.py GT DIR [DIR ...] -- #218 (11-22): the directional derivative dKL/deta of a direction (encode218.py --step quant units:
W_hat(eta) = W_hat - eta delta D / rms(D)) predicted by three gradients: GT (a check gradient on the FD batch's own documents:
probe218.py grad --grad-set tune --grad-docs 17), the round-0 300k mean (runs/r0/enc/grad: contains the chunks D was built from, so its
dot with D is biased by their projected noise), and chunk g2 alone (it chose the subspace). Per block and total."""
import json, os, sys

import numpy as np

R = "/data/w218/runs/r0"; gt = sys.argv[1]; qe = json.load(open("/data/w218/runs/qerr.json"))
for d in sys.argv[2:]:
    meta = json.load(open(f"{d}/grad.json"))["tensors"]; tot = {}
    for k, m in meta.items():
        b = int(k.split(".")[3]); unit = qe[k]["delta"] / m["g_rms"]
        D = np.fromfile(f"{d}/{k}.g", np.float32).astype(np.float64)
        for name, g in (("check", gt), ("r0mean", f"{R}/enc/grad"), ("g2", f"{R}/g2")):
            G = np.fromfile(f"{g}/{k}.g", np.float32).astype(np.float64)
            v = -float(G @ D) * unit
            tot.setdefault(name, {}).setdefault(b, 0.0); tot[name][b] += v
    for name, bl in tot.items():
        print(f"{os.path.basename(d)} {name:7s} dKL/deta total {sum(bl.values()):+.4e}  " + "  ".join(f"L{b} {v:+.3e}" for b, v in sorted(bl.items())), flush=True)
