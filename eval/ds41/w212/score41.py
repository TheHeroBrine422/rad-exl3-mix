#!/usr/bin/env python3
"""score41.py GOLD_DIR MANIFEST.json REPORT.json [REPORT2.json] [--out OUT.json] -- ds41-verify-v1 summary of radiance --kld-out reports.

Rows are masked by GOLD_DIR/gen.u8 (model-written tokens, like the challenge's mergerep.py). Mean KLD with a per-document bootstrap CI
(documents resampled with replacement, 2000 draws, seed 0; the statistic is the mean over the resampled documents' scored positions),
per category, and the headline merged numbers (median, p99, top-1, ppl ratio). With a second report: run-to-run paired difference
(same documents, same bootstrap draws) and the max per-row difference.
"""
import json, sys, os
import numpy as np

a = [x for x in sys.argv[1:] if not x.startswith("--")]
out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
if out: a.remove(out)
gd, man, reps = a[0], json.load(open(a[1])), a[2:]
kj = json.load(open(os.path.join(gd, "kld.json")))
gen = np.fromfile(os.path.join(gd, "gen.u8"), np.uint8).astype(bool)
docs = kj["docs"]
cat = {x["name"]: x["category"] for x in man["docs"]}
doc_of = np.concatenate([np.full(d["tokens"] - 1, i) for i, d in enumerate(docs)])
cats = np.array([cat[d["source"]] for d in docs])
assert len(doc_of) == len(gen) == kj["rows"]


def load(p):
    r = np.fromfile(p + ".rows", np.float32).reshape(-1, 4)
    assert len(r) == len(gen), (len(r), len(gen))
    return r


def boot(v, mask, rng, draws=2000):
    """mean of v over mask rows, bootstrap over documents: returns (mean, lo, hi)."""
    nd = len(docs)
    s = np.bincount(doc_of[mask], weights=v[mask], minlength=nd)
    n = np.bincount(doc_of[mask], minlength=nd).astype(float)
    ok = n > 0
    s, n = s[ok], n[ok]
    m = s.sum() / n.sum()
    idx = rng.integers(0, len(s), (draws, len(s)))
    bm = s[idx].sum(1) / n[idx].sum(1)
    return float(m), float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975))


def summary(r):
    kl = r[:, 0].astype(np.float64)
    res = {}
    rng = np.random.default_rng(0)
    m = gen
    k = kl[m]
    res["all"] = dict(positions=int(m.sum()), docs=int(len(np.unique(doc_of[m]))), kld_mean=boot(kl, m, rng),
                      median=float(np.median(k)), p99=float(np.quantile(k, 0.99)), p999=float(np.quantile(k, 0.999)),
                      top1=float(r[m, 3].mean()),
                      ppl_ratio=float(np.exp((r[m, 1].astype(np.float64).sum() - r[m, 2].astype(np.float64).sum()) / m.sum())))
    res["unmasked_all_rows_kld_mean"] = float(kl.mean())
    res["by_category"] = {}
    for c in sorted(set(cats)):
        mc = m & np.isin(doc_of, np.flatnonzero(cats == c))
        if mc.sum():
            res["by_category"][c] = dict(positions=int(mc.sum()), docs=int(len(np.unique(doc_of[mc]))), kld_mean=boot(kl, mc, rng),
                                         top1=float(r[mc, 3].mean()))
    return res


R = [load(p) for p in reps]
res = {"reports": reps, "runs": [summary(r) for r in R]}
if len(R) == 2:
    rng = np.random.default_rng(1)
    d = R[0][:, 0].astype(np.float64) - R[1][:, 0].astype(np.float64)
    res["pair"] = dict(diff_mean_masked=boot(d, gen, rng), max_abs_row_diff_all=float(np.abs(d).max()),
                       max_abs_row_diff_masked=float(np.abs(d[gen]).max()), rows_identical_masked=float((d[gen] == 0).mean()))
    res["pair_by_category"] = {c: boot(d, gen & np.isin(doc_of, np.flatnonzero(cats == c)), rng) for c in sorted(set(cats))}
s = json.dumps(res, indent=1)
if out: open(out, "w").write(s)
print(s)
