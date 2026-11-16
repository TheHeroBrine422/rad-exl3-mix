#!/usr/bin/env python3
"""second_slice197.py SCORES_DIR GOLDEN_DIR OUT.json PREFIX -- #197 (w185): the second-slice check (t186n) of the head arms.
Reads PREFIX-<base>-<arm>.json.rows (column 0 = KL per golden row; gen rows of the golden), prints and writes the mean KL
per arm and each arm's paired document-bootstrap difference to (a) of the same base (paired.py's statistic, 2000
resamples, seed 0). Run with plain python3 (numpy)."""
import json, os, sys
import numpy as np
SD, GD, OUT, PRE = sys.argv[1:5]
kld = json.load(open(f"{GD}/kld.json")); n = kld["rows"]
gen = np.fromfile(f"{GD}/gen.u8", np.uint8).astype(bool)
assert gen.size == n
def col0(name):
    a = np.fromfile(f"{SD}/{name}.json.rows", np.float32)
    return a.reshape(n, a.size // n)[:, 0].astype(np.float64)
res = {}
perdoc = {}
names = sorted(f[:-len(".json.rows")] for f in os.listdir(SD) if f.startswith(PRE) and f.endswith(".json.rows"))
for nm in names:
    c = col0(nm)
    res[nm] = {"kl_mean_gen": float(c[gen].mean())}
    perdoc[nm] = [[float(c[d["row0"]:d["row0"] + d["tokens"] - 1 - d["score_from"]][gen[d["row0"]:d["row0"] + d["tokens"] - 1 - d["score_from"]]].sum()),
                   int(gen[d["row0"]:d["row0"] + d["tokens"] - 1 - d["score_from"]].sum())] for d in kld["docs"]]
for nm in names:
    base = nm[len(PRE):].split("-")[0]; ref = f"{PRE}{base}-a"
    if ref not in res or nm == ref or nm.endswith("-a"): continue
    A, B = col0(ref), col0(nm)
    sa, sb, ns = [], [], []
    for d in kld["docs"]:
        k = d["tokens"] - 1 - d["score_from"]; s = slice(d["row0"], d["row0"] + k); mk = gen[s]
        sa.append(A[s][mk].sum()); sb.append(B[s][mk].sum()); ns.append(mk.sum())
    sa, sb, ns = map(np.array, (sa, sb, ns))
    ka, kb = sa.sum() / ns.sum(), sb.sum() / ns.sum()
    rng = np.random.default_rng(0); bs = []
    for _ in range(2000):
        i = rng.integers(0, len(ns), len(ns)); bs.append((sb[i].sum() - sa[i].sum()) / ns[i].sum())
    se = float(np.std(bs))
    res[nm].update({"vs": ref, "rel_pct": 100 * (kb - ka) / ka, "ci95_pct": 100 * 1.96 * se / ka})
json.dump({"golden": GD, "rows": n, "gen_positions": int(gen.sum()), "docs": len(kld["docs"]), "arms": res, "perdoc_sum_n": perdoc},
          open(OUT, "w"), indent=1)
for nm in names:
    r = res[nm]
    extra = f"  vs {r['vs']}: {r['rel_pct']:+.2f} % (±{r['ci95_pct']:.2f})" if "vs" in r else ""
    print(f"{nm}: {r['kl_mean_gen']:.6f}{extra}")
