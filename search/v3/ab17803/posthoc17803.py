#!/usr/bin/env python3
"""posthoc17803.py DOCSUMS.json REGISTRY -- descriptive only, NOT part of RULES-17803's reading (written after the data):
how much of each arm's d comes from the few documents that dominate s173f's KL at 5.0-6.5. Per (size, arm): the KL share
of the top document per part in B, d with the 2 documents of largest |change| per part removed, and the median per-document
log ratio (documents with B sum > 1 nat)."""
import json, math, sys
import numpy as np
D = json.load(open(sys.argv[1])); docs = json.load(open(sys.argv[2]))["slices"]["s173f"]["docs"]
parts = np.array([d["part"] for d in docs]); ids = np.array([d["id"] for d in docs])
out = {}
for s in ("5.0", "6.0", "6.5"):
    B = np.array([x[0] for x in D[f"{s}|B"]["docs"]])
    top = {p: (ids[parts == p][np.argmax(B[parts == p])], float(B[parts == p].max() / B[parts == p].sum())) for p in "cx"}
    print(f"{s} B: top-doc share c {top['c'][0]} {top['c'][1]:.0%}, x {top['x'][0]} {top['x'][1]:.0%}")
    for a in ("T4", "T5", "T6"):
        A = np.array([x[0] for x in D[f"{s}|{a}"]["docs"]]); dd = A - B; keep = np.ones(len(B), bool)
        for p in "cx":
            ix = np.where(parts == p)[0]; keep[ix[np.argsort(-np.abs(dd[ix]))[:2]]] = False
        dt = float(np.mean([math.log(A[keep & (parts == p)].sum() / B[keep & (parts == p)].sum()) for p in "cx"]))
        m = B > 1; med = float(np.median(np.log(A[m] / B[m])))
        out[f"{s}|{a}"] = {"d_trim2": dt, "median_doc_logr": med, "top_share": top}
        print(f"  {a}: d without the 2 largest-change docs per part {dt:+.4f}; median per-doc log ratio {med:+.4f} ({m.sum()} docs)")
json.dump(out, open(sys.argv[3], "w"), indent=1) if len(sys.argv) > 3 else None
