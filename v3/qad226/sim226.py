#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""sim226.py META PLANDIR JLONG_IDS DEEP_IDS --cap C --budget B [--place end|exthead|front] -- metadata-only simulation of the QAD stream plan
(data181.plan, V 24, Q 1024) for the extended plan + extra calib documents at cap C: per extra document the step it starts, the depth it reaches
by the budget, the step at which the budget is reached. Used to choose where the deep documents go (w226)."""
import json, sys, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "qad181"))
import data181 as D
meta, pdir, jlp, deepp = sys.argv[1:5]; a = sys.argv[5:]
CAP = int(a[a.index("--cap") + 1]); B = float(a[a.index("--budget") + 1]); place = a[a.index("--place") + 1] if "--place" in a else "end"
V, Q, MINR = 24, 1024, 16


class G:
    def __init__(self, name, d, gen):
        self.name = name; self.docs = json.load(open(f"{d}/kld.json"))["docs"]; self.gen = gen
    def ntok(self, i): return self.docs[i]["tokens"]
    def rows(self, i, a_, b_):
        d = self.docs[i]; j = np.arange(max(a_, d["score_from"]), min(b_, d["tokens"] - 1)); r = d["row0"] + j - d["score_from"]
        return int(self.gen[r].sum()) if len(r) else 0


names = ["xc8k-b2-c0", "xc8k-b2-c1", "tr32k-b2-c0"]
gs = [G(c, f"{meta}/{c}", np.fromfile(f"{pdir}/genmask/{c}.u8", np.uint8)) for c in names]
cal = G("calib", f"{meta}/calib", np.fromfile(f"{meta}/calib/gen.u8", np.uint8)); gs.append(cal); CG = 3
keep = set(json.load(open(jlp))); fe = set(json.loads(open(f"{meta}/../fe_and_md5.txt").readline()))
pl = json.load(open(f"{meta}/../pools214.json"))["pool"]
def item(gi, i):
    g = gs[gi]; n = min(CAP, g.ntok(i))
    if gi == CG and i in keep: n = min(n, 8192)
    return (gi, i, n, g.rows(i, 0, n))
items = []
for gi, g in enumerate(gs):
    for i in range(len(g.docs)):
        if gi == CG and (pl[i] != "train214" or i in fe): continue
        x = item(gi, i)
        if (gi == CG or True) and g.rows(i, 0, min(8192, g.ntok(i))) >= MINR: items.append(x)
n_old = len(items); order0 = np.random.default_rng(0).permutation(n_old)
px = json.load(open(f"{pdir}/ext-plan.json"))
for nm in px["new_goldens"]: gs.append(G(nm, f"{meta}/{nm}", np.fromfile(f"{pdir}/genmask/{nm}.u8", np.uint8)))
gi_of = {g.name: k for k, g in enumerate(gs)}
ext = [item(gi_of[nm], i) for nm, i, _ in px["items"]]
deep = [item(CG, i) for i in json.load(open(deepp))]
if place == "end": full = items + ext + deep; order = np.concatenate([order0, n_old + np.arange(len(ext) + len(deep))])
elif place == "exthead": full = items + deep + ext; order = np.concatenate([order0, n_old + np.arange(len(ext) + len(deep))])
else:   # front: the deep documents (longest first) take the first places of the order (one per stream), then w214's order, then the extension
    deep = sorted(deep, key=lambda x: -x[2]); full = deep + items + ext; nd = len(deep)
    order = np.concatenate([np.arange(nd), nd + order0, nd + n_old + np.arange(len(ext))])
steps = D.plan(full, order, V, Q)
cum, t = [], 0
for st in steps:
    for segs in st:
        for (it, sa, sb) in segs: gi, i, n, nr = full[it]; t += gs[gi].rows(i, sa, sb)
    cum.append(t)
cum = np.array(cum); last = int(np.searchsorted(cum, B)) + 1
print(f"cap {CAP} place {place}: {len(full)} items, {len(steps)} steps, {cum[-1]} rows; budget {B:.3g} reached at step {last}")
dset = {(x[0], x[1]) for x in deep}; reach = {}
for k, st in enumerate(steps[:last]):
    for segs in st:
        for (it, sa, sb) in segs:
            gi, i, n, nr = full[it]
            if (gi, i) in dset: r = reach.setdefault(i, [k, 0, n]); r[1] = max(r[1], sb)
for x in sorted(deep, key=lambda x: -x[2]):
    r = reach.get(x[1]); print(f"  calib {x[1]:5d} tokens {x[2]:7d} rows {x[3]:7d}: " + (f"starts step {r[0]}, depth {r[1]} by the budget" if r else "not reached"))
