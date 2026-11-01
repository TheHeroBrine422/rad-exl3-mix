#!/usr/bin/env python3
"""posbins.py GOLDEN REF.f32 LABEL:A.f32 ... -- mining merge B4: does KLD build up with context position, and does it
build up faster for some quants than others? Mean KLD per position bin (the scored, gen-marked rows), per document
and pooled, for every score file; then each LABEL's ratio to REF per bin. If a quant's ratio to the reference grows
with position, its error compounds with context (not explained by text type, which is the same in both: paired rows).
Bins: 0-2k, 2-8k, 8-32k, 32-64k, 64-128k, 128k+."""
import json, sys
import numpy as np
g, ref = sys.argv[1:3]; arms = [a.split(":", 1) for a in sys.argv[3:]]
m = json.load(open(f"{g}/kld.json")); gen = np.fromfile(f"{g}/gen.u8", np.uint8).astype(bool)
EDGES = [0, 2048, 8192, 32768, 65536, 131072, 1 << 40]
NAMES = ["0-2k", "2-8k", "8-32k", "32-64k", "64-128k", "128k+"]
pos = np.zeros(m["rows"], np.int64); doc = np.zeros(m["rows"], np.int64)
for j, d in enumerate(m["docs"]):
    n = d["tokens"] - 1 - d["score_from"]; s = slice(d["row0"], d["row0"] + n)
    pos[s] = d["score_from"] + np.arange(n); doc[s] = j
b = np.digitize(pos, EDGES) - 1
load = lambda f: np.fromfile(f, np.float32).reshape(-1, 4)[:, 0].astype(np.float64)
R = load(ref)
def table(x):   # [doc or pooled][bin] mean over gen rows
    out = {}
    for j in list(range(len(m["docs"]))) + [None]:
        sel = gen & ((doc == j) if j is not None else True)
        out[j] = [x[sel & (b == k)].mean() if (sel & (b == k)).any() else np.nan for k in range(len(NAMES))]
    return out
TR = table(R)
print(f"{'':18s} " + " ".join(f"{n:>9s}" for n in NAMES))
print(f"{'ref KLD (pooled)':18s} " + " ".join(f"{v:9.5f}" for v in TR[None]))
for lab, f in arms:
    T = table(load(f))
    for j in [None] + list(range(len(m["docs"]))):
        tag = "pooled" if j is None else f"doc {m['docs'][j].get('orig', j)}"
        print(f"{lab[:10]:10s} {tag:>7s} " + " ".join(f"{a / r:9.3f}" for a, r in zip(T[j], TR[j])) + "   (ratio to ref)")
