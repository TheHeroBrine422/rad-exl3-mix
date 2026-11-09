#!/usr/bin/env python3
"""simjson94.py DATA ROWSET OUT.json -- headsim.py's JSON (per plan: avg_bits, kl, top1, guard_kl, the paired difference
vs its uniform reference with a 95% interval) from headsim94's per-row output on one of report94's row sets
(search-all, search-gen, qad-gen), for w88's headgrid.py / knap5 KN_HEADGRID. The interval is a document bootstrap
(2,000 draws) of the mean paired difference (headsim.py used 512-position blocks)."""
import json, sys
import numpy as np

D, RS, OUT = sys.argv[1:4]
s = RS.split("-")[0]
z = np.load(f"{D}/sim-{s}.npz"); doc = np.fromfile(f"{D}/meta/{s}/doc.i32", np.int32)
gen = np.fromfile(f"{D}/meta/{s}/gen.u8", np.uint8).astype(bool)
msk = np.ones_like(gen) if RS.endswith("-all") else gen
names = [str(n) for n in z["names"]]; ix = {n: i for i, n in enumerate(names)}
KL, HIT, top = z["kl"][:, msk].astype(np.float64), z["hit"][:, msk], z["top"][msk]
u, inv = np.unique(doc[msk], return_inverse=True); cnt = np.bincount(inv); nd = len(u)
boot = np.random.default_rng(0).integers(0, nd, (2000, nd)); guard = top >= 20000
KS = sorted(float(n[1:]) for n in names if n[0] == "u" and n[1:].replace(".", "").isdigit())
res = {}
for n in names:
    i = ix[n]; bits = float(z["avg_bits"][i])
    r = {"avg_bits": bits, "kl": float(KL[i].mean()), "top1": float(HIT[i].mean()), "guard_kl": float(KL[i][guard].mean()),
         "guard_n": int(guard.sum()), "proxy_train_vs_uniform": None}
    B = n.lstrip("usb").split("-")[0]; ref = f"u{B}"
    if ref not in ix:
        lo = [k for k in KS if k <= float(B) + 1e-9]; ref = f"u{max(lo):g}" if lo else ref
    if ref in ix and ref != n:
        d = KL[i] - KL[ix[ref]]; sd = np.bincount(inv, d)
        bs = sd[boot].sum(1) / cnt[boot].sum(1)
        r.update(vs=ref, diff=float(d.mean()), rel=float(d.mean() / KL[ix[ref]].mean()), se=float(bs.std()),
                 ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    res[n] = r
json.dump({"n": int(msk.sum()), "rowset": RS, "guard_rows": 20000, "plans": res}, open(OUT, "w"), indent=1)
print(RS, len(res), "plans", int(msk.sum()), "rows", nd, "docs")
