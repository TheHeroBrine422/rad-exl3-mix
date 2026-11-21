#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""mkpools214.py OUTDIR --calib GOLDEN --pools POOLS --chunks DIR,... --genmask DIR [--frac 0.2] [--cap 8192] [--seed 214] -- #214's training data:
the calib documents of the pools {qad, fit, search} (never verify) with >= 16 gen rows in their first CAP tokens, a seeded subset (shuffle seed)
sized so that they hold --frac of the mixed gen rows (the xcorp / trace chunks' rows are counted with their gen masks, first CAP tokens of
each document, >= 16 rows), relabelled `train214` in OUTDIR/pools214.json (same layout as pools-v3.json; qad200's --pools / --pool train214);
OUTDIR/qad214-kld.json = ledger_add.py's input (the chosen documents, role train, use qad214); OUTDIR/summary.json."""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "qad181"))
import data181 as D  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--calib", required=True); ap.add_argument("--pools", required=True)
ap.add_argument("--chunks", required=True); ap.add_argument("--genmask", required=True)
ap.add_argument("--frac", type=float, default=0.2); ap.add_argument("--cap", type=int, default=8192); ap.add_argument("--seed", type=int, default=214)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
P = json.load(open(a.pools)); pool = P["pool"]
xrows, xs = 0, {}
for d in a.chunks.split(","):
    g = D.Gold(d, genfile=os.path.join(a.genmask, os.path.basename(d.rstrip("/")) + ".u8"))
    nr = [g.nrows(i, a.cap) for i in range(len(g.docs))]
    k = sum(r for r in nr if r >= 16); xrows += k; xs[os.path.basename(d.rstrip("/"))] = {"docs": sum(r >= 16 for r in nr), "gen_rows": int(k)}
gc = D.Gold(a.calib, "calib")
cand = [i for i, p in enumerate(pool) if p in ("qad", "fit", "search") and gc.nrows(i, a.cap) >= 16]
rng = np.random.default_rng(a.seed); rng.shuffle(cand)
want = a.frac / (1 - a.frac) * xrows; got = 0; take = []
for i in cand:
    if got >= want: break
    take.append(i); got += gc.nrows(i, a.cap)
new = [("train214" if i in set(take) else p) for i, p in enumerate(pool)]
assert all(new[i] == "train214" and pool[i] != "verify" for i in take)
P2 = dict(P); P2["pool"] = new; P2["derived214"] = {"from": os.path.basename(a.pools), "seed": a.seed, "frac": a.frac, "cap": a.cap}
json.dump(P2, open(f"{a.out}/pools214.json", "w"))
kl = {k: v for k, v in gc.m.items() if k != "docs"}
kl["docs"] = [{"source": gc.docs[i]["source"], "tokens": min(a.cap, gc.docs[i]["tokens"]), "score_from": gc.docs[i]["score_from"], "orig": i} for i in sorted(take)]
json.dump(kl, open(f"{a.out}/qad214-kld.json", "w"))
from collections import Counter
s = {"xcorp": xs, "xcorp_gen_rows": int(xrows), "calib_docs": len(take), "calib_gen_rows": int(got), "calib_share": got / (got + xrows),
     "calib_pools": dict(Counter(pool[i] for i in take)), "calib_candidates": len(cand), "cap": a.cap, "seed": a.seed}
json.dump(s, open(f"{a.out}/summary.json", "w"), indent=1); print(json.dumps(s, indent=1))
