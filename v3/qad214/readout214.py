#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""readout214.py [OUT.json] -- (box2, /data/q27/w214) every finished served read of #214 against its size's control: ratio of gen-row KL (per-document
bootstrap, cmp214's arithmetic), CI, documents better / worse, step vs the previous checkpoint of the arm, and the per-category ratio (descriptive).
Arms / sizes / checkpoint order from reads214.txt."""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmp214 import doc_sums, paired

W = "/data/q27/w214"; G = f"{W}/slices/golden-j214"
meta = json.load(open(f"{G}/kld.json"))["docs"]
cal = [json.loads(l) for l in open("/data/q27/challenge/corpus/calib.jsonl").read().rstrip("\n").split("\n")]
cat = np.array([cal[d["orig"]]["category"] for d in meta])
ctl = {}
rep = {"controls": {}, "reads": {}}
rows = [l.split() for l in open(f"{W}/reads214.txt") if l.strip() and not l.startswith("#")]
arms = collections.OrderedDict()
for a, s, c in rows: arms.setdefault((a, s), []).append(c)
print("| arm | size | ckpt | KL | ratio vs unhealed | 95 % CI | docs better / worse | median doc ratio | ratio without the 3 worst docs | vs previous ckpt |\n|---|---|---|---|---|---|---|---|---|---|")
for (a, s), cks in arms.items():
    cr = f"{W}/scores/ctl-{s}.json.rows"
    if not os.path.exists(cr): continue
    xc, n = doc_sums(G, cr); rep["controls"][s] = float(xc.sum() / n.sum()); prev = None
    for c in cks:
        r = f"{W}/scores/{a}-{c}.json.rows"
        if not os.path.exists(r): continue
        x, _ = doc_sums(G, r); d = paired(xc, x, n); d["kld"] = float(x.sum() / n.sum())
        d["step"] = paired(prev, x, n) if prev is not None else None
        w = np.argsort(-(x - xc))[:3]; k = np.ones(len(x), bool); k[w] = False
        d["median_doc_ratio"] = float(np.median(x[n > 0] / xc[n > 0])); d["ratio_without_3_worst_docs"] = float(x[k].sum() / xc[k].sum()); d["worst_docs"] = [int(meta[i]["orig"]) for i in w]
        d["by_category"] = {k: float(x[cat == k].sum() / xc[cat == k].sum()) for k in sorted(set(cat))}
        rep["reads"][f"{a}-{c}"] = d; prev = x
        st = f"{d['step']['ratio']:.4f} [{d['step']['ci95'][0]:.4f}, {d['step']['ci95'][1]:.4f}]" if d["step"] else ""
        print(f"| {a} | {s} | {c} | {d['kld']:.6f} | {d['ratio']:.4f} | [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}] | {d['docs_better']} / {d['docs_worse']} | {d['median_doc_ratio']:.3f} | {d['ratio_without_3_worst_docs']:.4f} | {st} |")
print("\n| lr contrast (same size, same budget) | ckpt | KL lr 3e-4 / KL lr 1e-3 | 95 % CI | docs better / worse |\n|---|---|---|---|---|")
rep["lr_contrast"] = {}
for lo, hi, s_ in (("25A", "25A-lr3e-4", "2.50"), ("30A", "30A-lr3e-4", "3.00"), ("40A", "40A-lr3e-4", "4.00"), ("40A", "40A-lr3e-4-1M", "4.00"), ("40Q", "40A-lr3e-4", "4.00")):
    for c in ("ckpt-2e+05", "ckpt-5e+05", "ckpt-1e+06"):
        a_, b_ = f"{W}/scores/{lo}-{c}.json.rows", f"{W}/scores/{hi}-{c}.json.rows"
        if not (os.path.exists(a_) and os.path.exists(b_)): continue
        xa, n = doc_sums(G, a_); xb, _ = doc_sums(G, b_); d = paired(xa, xb, n); rep["lr_contrast"][f"{hi}-vs-{lo}-{c}"] = d
        print(f"| {hi} vs {lo} ({s_}) | {c} | {d['ratio']:.4f} | [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}] | {d['docs_better']} / {d['docs_worse']} |")
if len(sys.argv) > 1: json.dump(rep, open(sys.argv[1], "w"), indent=1)
