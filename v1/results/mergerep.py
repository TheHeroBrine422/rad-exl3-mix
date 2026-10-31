#!/usr/bin/env python3
"""mergerep.py GOLDEN OUT.json REPORT.json... -- one candidate's tier reports (radiance --kld-out or
llama_kld, each with its .rows) as one report over every document they cover, in the engines'
layout: `all`, `by_source`, and `coverage` (documents and positions of GOLDEN covered). The
summaries are the engines' own: mean, linearly interpolated quantiles, top-1 agreement, and the
perplexities from the summed NLLs. When GOLDEN holds gen.u8 (genmask.py), every summary is taken over
the rows it marks, the tokens the model writes itself."""
import json
import os
import sys

import numpy as np

gd, out, reps = sys.argv[1], sys.argv[2], sys.argv[3:]
mg = json.load(open(os.path.join(gd, "kld.json")))
kl, nq, np_, t1 = (np.full(mg["rows"], np.nan, np.float32) for _ in range(4))
covered = np.zeros(len(mg["docs"]), bool)
first = None
for p in reps:
    rep = json.load(open(p))
    first = first or rep
    ref = json.load(open(os.path.join(rep["ref_dir"], "kld.json")))
    rows = np.fromfile(p + ".rows", np.float32).reshape(-1, 4)
    assert len(rows) == ref["rows"], p
    for i, d in enumerate(ref["docs"]):
        o = d.get("orig", i)  # a subset reference maps its documents to GOLDEN's
        g = mg["docs"][o]
        n = d["tokens"] - 1 - d["score_from"]
        assert (g["source"], g["tokens"], g["score_from"]) == (d["source"], d["tokens"], d["score_from"])
        if covered[o]:
            sys.exit(f"{p}: document {o} is in two reports")
        covered[o] = True
        s = slice(g["row0"], g["row0"] + n)
        kl[s], nq[s], np_[s], t1[s] = rows[d["row0"]:d["row0"] + n].T


def summary(idx):
    k = np.sort(kl[idx].astype(np.float64))
    n = len(k)
    q = lambda x: float(np.quantile(k, x)) if n else 0.0
    sq, sp = float(nq[idx].astype(np.float64).sum()), float(np_[idx].astype(np.float64).sum())
    return {"positions": n, "kld": {"mean": float(k.mean()) if n else 0.0, "median": q(0.5), "p90": q(0.9),
                                     "p95": q(0.95), "p99": q(0.99), "p99_9": q(0.999), "p99_99": q(0.9999),
                                     "max": float(k[-1]) if n else 0.0},
            "top1_agreement": float(t1[idx].mean()) if n else 0.0,
            "ppl": {"candidate": float(np.exp(sq / n)), "reference": float(np.exp(sp / n)),
                    "ratio": float(np.exp((sq - sp) / n))}}


gp = os.path.join(gd, "gen.u8")
gen = np.fromfile(gp, np.uint8).astype(bool) if os.path.exists(gp) else np.ones(mg["rows"], bool)
assert len(gen) == mg["rows"], gp


def rows_of(docs):
    r = np.concatenate([np.arange(mg["docs"][i]["row0"],
                                  mg["docs"][i]["row0"] + mg["docs"][i]["tokens"] - 1 - mg["docs"][i]["score_from"])
                        for i in docs]) if len(docs) else np.zeros(0, np.int64)
    return r[gen[r]]


docs = np.nonzero(covered)[0]
by = {}
for i in docs:
    by.setdefault(mg["docs"][i]["source"], []).append(i)
rep = {k: first[k] for k in ("model", "engine", "reference", "ref_form", "kv") if k in first}
rep.update({"ref_dir": gd, "parts": reps, "docs": int(len(docs)),
            "coverage": {"docs": int(len(docs)), "of_docs": len(mg["docs"]),
                         "positions": int(len(rows_of(docs))), "of_positions": int(gen.sum())},
            "positions": "the tokens the model writes (gen.u8)" if os.path.exists(gp) else "every scored row",
            "percentile": "linear interpolation between order statistics",
            "all": summary(rows_of(docs)),
            "by_source": {s: dict(summary(rows_of(ds)), docs=len(ds)) for s, ds in sorted(by.items())}})
json.dump(rep, open(out, "w"), indent=1)
a = rep["all"]
print(f"{out}: {rep['coverage']['positions']} of {rep['coverage']['of_positions']} positions, "
      f"mean KL {a['kld']['mean']:.6g}, top-1 {a['top1_agreement'] * 100:.3f}%")
