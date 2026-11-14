#!/usr/bin/env python3
"""docsums157.py GOLDEN OUT.json ARM=ROWS... -- (w157, #174-03, box2) per-document KL sums over the golden's gen
positions for each scored arm (rxrun.hostd.doc_sums, inlined: the box2 stage code predates it), plus the arm's mean KL,
position count, top-1 agreement and the rows file's sha256 (the identity check compares those)."""
import hashlib, json, sys
import numpy as np

G, OUT = sys.argv[1:3]
m = json.load(open(f"{G}/kld.json")); gen = np.fromfile(f"{G}/gen.u8", np.uint8).astype(bool)
out = {}
for a in sys.argv[3:]:
    arm, rows = a.split("=", 1)
    K = np.fromfile(rows, np.float32).reshape(-1, 4); docs, allk = [], []
    for d in m["docs"]:
        s = slice(d["row0"], d["row0"] + d["tokens"] - 1 - d["score_from"]); g = gen[s]
        k = K[s, 0][g].astype(np.float64); docs.append([round(float(k.sum()), 6), int(g.sum())]); allk.append(k)
    kl = np.concatenate(allk); sc = json.load(open(rows[:-len(".rows")]))
    out[arm] = {"docs": docs, "kld": float(kl.mean()), "positions": int(len(kl)), "top1": (sc.get("all") or {}).get("top1_agreement"),
                "rows_sha256": hashlib.sha256(open(rows, "rb").read()).hexdigest()}
json.dump(out, open(OUT, "w"))
print({k: (round(v["kld"], 6), v["positions"], v["rows_sha256"][:16]) for k, v in out.items()})
