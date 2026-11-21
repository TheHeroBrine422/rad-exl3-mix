#!/usr/bin/env python3
"""gate41.py GOLD_DIR MANIFEST.json BASE.json CAND.json [--out OUT.json] -- w213's port gate: a candidate's radiance --kld-out
report against the baseline's (w212b's mik1 on bakon's tree), on the golden's own gen.u8 rows (never rebuilt).

Gate (user 11-21): paired per-document difference of the masked mean KLD within +-3 % of the baseline mean with its bootstrap CI
inside +-5 %; no category's paired difference beyond 2 of its own paired SE (bootstrap SD over the category's documents);
top-1 within 0.3 pp. Also: the per-row difference distribution (share identical, p99 |d|, max), the documents with the largest
paired difference, and the prompt-side (not generated) rows as a diagnostic only.
Bootstrap: documents resampled with replacement, 2000 draws, seed 1; the statistic is the mean over the resampled documents'
scored positions (score41.py's). Rows files: <report>.rows, [kld, lp_ref, lp_cand, top1] f32.
"""
import json, sys, os
import numpy as np

a = [x for x in sys.argv[1:] if not x.startswith("--")]
out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
if out: a.remove(out)
gd, man, base_p, cand_p = a[0], json.load(open(a[1])), a[2], a[3]
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


B, C = load(base_p), load(cand_p)
kb, kc = B[:, 0].astype(np.float64), C[:, 0].astype(np.float64)
d = kc - kb
nd = len(docs)


def boot(v, mask, rng, draws=2000):
    """(mean over mask rows, CI lo, CI hi, bootstrap SD) with documents resampled."""
    s = np.bincount(doc_of[mask], weights=v[mask], minlength=nd)
    n = np.bincount(doc_of[mask], minlength=nd).astype(float)
    ok = n > 0
    s, n = s[ok], n[ok]
    m = s.sum() / n.sum()
    idx = rng.integers(0, len(s), (draws, len(s)))
    bm = s[idx].sum(1) / n[idx].sum(1)
    return float(m), float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975)), float(bm.std())


rng = np.random.default_rng(1)
m = gen
base_mean = float(kb[m].mean())
cand_mean = float(kc[m].mean())
dm, lo, hi, sd = boot(d, m, rng)
res = dict(base=base_p, cand=cand_p, positions=int(m.sum()), base_mean=base_mean, cand_mean=cand_mean,
           diff=dict(mean=dm, ci=[lo, hi], sd=sd, rel=dm / base_mean, rel_ci=[lo / base_mean, hi / base_mean]))
t1b, t1c = float(B[m, 3].mean()), float(C[m, 3].mean())
res["top1"] = dict(base=t1b, cand=t1c, diff_pp=100 * (t1c - t1b))
ad = np.abs(d[m])
res["rows"] = dict(identical_share=float((d[m] == 0).mean()), p50_abs=float(np.quantile(ad, 0.5)), p99_abs=float(np.quantile(ad, 0.99)),
                   p999_abs=float(np.quantile(ad, 0.999)), max_abs=float(ad.max()),
                   top1_flips=int((B[m, 3] != C[m, 3]).sum()),
                   lp_ref_max_abs=float(np.abs(B[m, 1] - C[m, 1]).max()))
cg = {}
for c in sorted(set(cats)):
    mc = m & np.isin(doc_of, np.flatnonzero(cats == c))
    if not mc.sum(): continue
    cm, clo, chi, csd = boot(d, mc, rng)
    cg[c] = dict(positions=int(mc.sum()), docs=int(len(np.unique(doc_of[mc]))), base=float(kb[mc].mean()), diff=cm, ci=[clo, chi],
                 se=csd, z=(cm / csd if csd > 0 else (0.0 if cm == 0 else float("inf"))))
res["by_category"] = cg
# documents by paired difference (sum over their scored rows, and mean)
s = np.bincount(doc_of[m], weights=d[m], minlength=nd)
n = np.bincount(doc_of[m], minlength=nd)
order = np.argsort(-np.abs(s))[:10]
res["top_docs"] = [dict(doc=int(i), source=docs[i]["source"], category=str(cats[i]), scored=int(n[i]), diff_sum=float(s[i]),
                        diff_mean=float(s[i] / max(n[i], 1))) for i in order]
u = ~gen
res["not_generated_rows_diag"] = dict(positions=int(u.sum()), base_mean=float(kb[u].mean()), cand_mean=float(kc[u].mean()),
                                      identical_share=float((d[u] == 0).mean()), max_abs=float(np.abs(d[u]).max()))
g = dict(mean_within_3pct=abs(res["diff"]["rel"]) <= 0.03,
         ci_within_5pct=max(abs(res["diff"]["rel_ci"][0]), abs(res["diff"]["rel_ci"][1])) <= 0.05,
         categories_within_2se=all(abs(v["z"]) <= 2 for v in cg.values()),
         top1_within_0p3pp=abs(res["top1"]["diff_pp"]) <= 0.3)
g["pass"] = all(g.values())
res["gate"] = g
txt = json.dumps(res, indent=1)
if out: open(out, "w").write(txt)
print(txt)
