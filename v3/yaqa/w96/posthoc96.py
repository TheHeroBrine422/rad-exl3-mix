#!/usr/bin/env python3
"""posthoc96.py -- report only (after the verdict): is w94's qad-vs-search gap Fisher in-sample overfit? On the qad-dump
gen rows (all in half B), plans from today's full qad Fisher (contains those documents) vs from half A (does not),
paired document bootstrap; and the slab / uniform ratio of each on qad rows vs search rows."""
import json, os, sys
import numpy as np
sys.argv = [sys.argv[0]]; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plans96 as p
Q, S = p.Sums("qad"), p.Sums("search")
full = p.splans(np.load(f"{p.HSD}/headsens.npz")["fisher_qad_gen"]); F, n, N = p.fisher_set([0]); half = p.splans(F / N)
out = {}
for k in p.PLANS3:
    u = p.UNI["u" + k[1]]
    a, b = Q.kl(full[k]), Q.kl(half[k]); m, se, ci, nd = p.boot(Q, a, b)
    out[k] = {"qad_full": float(a.mean()), "qad_halfA": float(b.mean()), "full_vs_halfA": [m, ci],
              "slab_over_uniform_qad_full": float(a.mean() / Q.kl(u).mean()),
              "slab_over_uniform_qad_halfA": float(b.mean() / Q.kl(u).mean()),
              "slab_over_uniform_search_full": float(S.kl(full[k]).mean() / S.kl(u).mean()),
              "slab_over_uniform_search_halfA": float(S.kl(half[k]).mean() / S.kl(u).mean())}
    print(k, json.dumps(out[k]), flush=True)
json.dump(out, open(f"{p.O}/posthoc.json", "w"), indent=1)
