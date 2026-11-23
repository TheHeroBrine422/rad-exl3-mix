#!/usr/bin/env python3
"""w225: where deep-context training documents are: per golden (and per calib pool of each pools file) the documents longer than 8k / 32k / 64k / 128k
tokens, their tokens beyond 32k, and how many of them are golden-jlong's (the long judge, kept at 8k). Metadata only (kld.json token counts).
python3 deepdocs225.py --calib GOLDEN --pools P1.json,P2.json --jlong IDS.json [--gold DIR,DIR..]"""
import argparse, json, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--calib"); ap.add_argument("--pools", default=""); ap.add_argument("--jlong"); ap.add_argument("--gold", default="")
a = ap.parse_args()
T = (8192, 32768, 65536, 131072)
def row(name, n, jl=None):
    n = np.asarray(n); s = f"{name:42s} docs {len(n):5d} max {int(n.max()) if len(n) else 0:7d} " + " ".join(f">{t // 1024}k {int((n > t).sum()):4d}" for t in T)
    s += f"  tokens>32k {int(np.clip(n - 32768, 0, None).sum()):8d}"
    if jl is not None: s += f"  (of >32k: {jl} golden-jlong)"
    print(s)
jl = set(json.load(open(a.jlong))) if a.jlong else set()
if a.calib:
    n = np.array([d["tokens"] for d in json.load(open(os.path.join(a.calib, "kld.json")))["docs"]])
    row("calib (all)", n, sum(1 for i in np.nonzero(n > 32768)[0] if int(i) in jl))
    for pf in [p for p in a.pools.split(",") if p]:
        pl = json.load(open(pf))["pool"]
        for p in sorted(set(pl)):
            ix = [i for i, x in enumerate(pl) if x == p]
            row(f"  {os.path.basename(pf)}:{p}", n[ix], sum(1 for i in ix if n[i] > 32768 and i in jl))
for g in [x for x in a.gold.split(",") if x]:
    row(os.path.basename(g.rstrip("/")), [d["tokens"] for d in json.load(open(os.path.join(g, "kld.json")))["docs"]])
