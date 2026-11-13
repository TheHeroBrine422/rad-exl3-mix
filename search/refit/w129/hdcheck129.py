#!/usr/bin/env python3
"""hdcheck129.py DONORS_DIR PERM.npy QROOT PLAN... -- spot check of hd129.sh's donors: 4,096 random vocab rows of each
donor's output.weight == the plan's dequantized R16 head row (bf16 bits), and the whole head differs from the base's."""
import json, os, sys
import numpy as np, torch
from safetensors import safe_open
sys.path.insert(0, "/data/q27/stage/code/v3/yaqa")
import headslab as H  # noqa: E402

D, permf, qroot = sys.argv[1:4]; perm = np.load(permf); P = json.load(open(f"{D}/plans-u.json"))["plans"]
rng = np.random.default_rng(0); ii = rng.choice(len(perm), 4096, replace=False)
for pn in sys.argv[4:]:
    d = f"{D}/hd-{pn}.rad"; off, nb, shape = H.entry(d)
    out = np.memmap(d, np.uint16, "r", offset=off, shape=tuple(shape))
    dn = P[pn]["donor"]; qd = f"{qroot}/{dn.rsplit('-K', 1)[0]}/K{P[pn]['avg_bits']:g}"
    with safe_open(f"{qd}/wq/lm_head.safetensors", "pt", device="cpu") as f: W = f.get_tensor("weight")
    want = W[torch.from_numpy(ii)].to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    got = np.asarray(out[perm[ii]])
    print(f"{pn}: {qd} rows equal {int((want == got).all(axis=1).sum())} / {len(ii)}")
