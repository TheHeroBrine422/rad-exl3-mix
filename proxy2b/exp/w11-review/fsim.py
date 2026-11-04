#!/usr/bin/env python3
"""fsim.py GROUP DN1 DN2 [DN3 ..] -- w11: how much of the asym target transform is systematic vs realization-specific:
cosine between (F - I) of mkdenoise dirs built on different sibling quants (seeds 0 / 1 / 2, K3 vs K2), plus the
diagonal-only share. A high cross-seed cosine = the correction transfers to another rounding of the siblings (e.g. the
27B's knapsack picking another K or seed for them); a low one = it fits this rounding only. CPU, numpy."""
import sys, json, itertools
import numpy as np
from safetensors.numpy import load_file
g, dns = sys.argv[1], sys.argv[2:]
def dev(d):
    F = load_file(f"{d}/wpre/{g}.safetensors")["Ft"].astype(np.float64).T
    return F - np.eye(len(F))
E = {d: dev(d) for d in dns}
cos = lambda a, b: float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
out = {"group": g, "norm": {d.split("/")[-1]: float(np.linalg.norm(E[d]) / len(E[d]) ** 0.5) for d in dns}, "cos": {}}
for a, b in itertools.combinations(dns, 2):
    out["cos"][f"{a.split('/')[-1]}~{b.split('/')[-1]}"] = round(cos(E[a], E[b]), 4)
    da, db = np.diag(E[a]), np.diag(E[b])
    out["cos"][f"diag {a.split('/')[-1]}~{b.split('/')[-1]}"] = round(cos(da, db), 4)
print(json.dumps(out))
