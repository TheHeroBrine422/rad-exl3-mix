#!/usr/bin/env python3
"""mkgroups96.py calib GOLDEN POOLS.json POOLS-V3.json SPLIT.json OUT.npy
   mkgroups96.py xc GOLDEN EXCLUDE.docs OUT.npy
Row groups for fisher96.py (int8 per golden row, −1 = skip).
calib: 0 = split half A gen rows, 1 = half B gen rows, 2 = gen rows of the qad documents of pools.json that pools-v3
moved to verify (so 0 + 1 + 2 = today's headsens qad-gen set, for G0), 3 / 4 = non-gen rows of half A / B.
xc: 0 = every scored row of the golden's pieces except those listed in EXCLUDE.docs (golden-search27b-x's picks)."""
import json, sys, numpy as np

def rows_of(m):
    for i, d in enumerate(m["docs"]):
        yield i, d["row0"], d["tokens"] - 1 - d["score_from"]

mode, G = sys.argv[1], sys.argv[2]
m = json.load(open(f"{G}/kld.json")); R = m["rows"]; out = np.full(R, -1, np.int8)
if mode == "calib":
    P = json.load(open(sys.argv[3]))["pool"]; P3 = json.load(open(sys.argv[4]))["pool"]; S = json.load(open(sys.argv[5]))
    A, B = set(S["A"]), set(S["B"]); gen = np.fromfile(f"{G}/gen.u8", np.uint8).astype(bool); assert len(gen) == R
    for i, r0, n in rows_of(m):
        sl = slice(r0, r0 + n)
        if P3[i] == "qad":
            g = 0 if i in A else 1 if i in B else None; assert g is not None, i
            out[sl] = np.where(gen[sl], g, 3 + g)
        elif P[i] == "qad":
            out[sl] = np.where(gen[sl], 2, -1)
    np.save(sys.argv[6], out)
else:
    ex = {int(x) for x in open(sys.argv[3]).read().split()}
    for i, r0, n in rows_of(m):
        if i not in ex: out[r0:r0 + n] = 0
    np.save(sys.argv[4], out); print(f"{len(ex)} pieces excluded")
print({g: int((out == g).sum()) for g in range(-1, int(out.max()) + 1)})
