#!/usr/bin/env python3
"""cmp147.py GOLDEN OUT.json NAME=ROWS... -- (w147, #178-02) KLD rows of the before / after containers against w142's arms on
one golden (s171d). Per row file: mean KL over the golden's generated positions (rxrun.hostd.doc_sums' positions), per part
(c, x). Pairs (A=ROWS, B=ROWS given as A~B on the command line after the files): byte-identical or not, mean |row diff|, the
difference of the means, and d = pmean over parts of log(sum KL_A / sum KL_B) with a 95% CI from 2,000 paired document
bootstraps stratified by part (ab178.py's statistic)."""
import json, math, sys
import numpy as np

G, OUT = sys.argv[1], sys.argv[2]
m = json.load(open(f"{G}/kld.json")); gen = np.fromfile(f"{G}/gen.u8", np.uint8).astype(bool)
docs = m["docs"]; parts = np.array([d["part"] for d in docs]); P = sorted(set(parts.tolist()))
rows, pairs = {}, []
for a in sys.argv[3:]:
    if "~" in a: pairs.append(a.split("~")); continue
    n, p = a.split("=", 1); rows[n] = p
def load(p):
    K = np.fromfile(p, np.float32).reshape(-1, 4)[:, 0]
    s, ks = [], []
    for d in docs:
        sl = slice(d["row0"], d["row0"] + d["tokens"] - 1 - d["score_from"]); g = gen[sl]
        k = K[sl][g].astype(np.float64); s.append(k.sum()); ks.append(k)
    return np.array(s), np.concatenate(ks)
R = {n: load(p) for n, p in rows.items()}
npos = np.array([int(gen[slice(d["row0"], d["row0"] + d["tokens"] - 1 - d["score_from"])].sum()) for d in docs])
out = {"golden": G, "positions": int(npos.sum()), "rows": {}, "pairs": {}}
for n, (s, k) in R.items():
    out["rows"][n] = {"kld": float(k.mean()), **{f"kld_{p}": float(s[parts == p].sum() / npos[parts == p].sum()) for p in P}}
    print(f"{n:>10}: KLD {k.mean():.6f}  " + "  ".join(f"{p} {out['rows'][n]['kld_' + p]:.6f}" for p in P))
grp = [np.where(parts == p)[0] for p in P]
rng = np.random.default_rng(17802); draws = [np.concatenate([g[rng.integers(0, len(g), len(g))] for g in grp]) for _ in range(2000)]
def dstat(A, B, ix): pi = parts[ix]; return float(np.mean([math.log(A[ix][pi == p].sum() / B[ix][pi == p].sum()) for p in P]))
for a, b in pairs:
    (sa, ka), (sb, kb) = R[a], R[b]
    same = open(rows[a], "rb").read() == open(rows[b], "rb").read()
    d = dstat(sa, sb, np.arange(len(docs))); bs = sorted(dstat(sa, sb, ix) for ix in draws)
    pp = {p: math.log(sa[parts == p].sum() / sb[parts == p].sum()) for p in P}
    r = {"identical": same, "mean_abs_row_diff": float(np.abs(ka - kb).mean()), "kld_diff": float(ka.mean() - kb.mean()),
         "d": d, "ci95": [bs[50], bs[1949]], "per_part": pp}
    out["pairs"][f"{a}~{b}"] = r
    print(f"{a:>10} vs {b:<10}: {'IDENTICAL ROWS' if same else 'rows differ'}; KLD diff {r['kld_diff']:+.2e}, mean |row diff| "
          f"{r['mean_abs_row_diff']:.2e}; d {100 * d:+.3f}% [{100 * bs[50]:+.3f}, {100 * bs[1949]:+.3f}]  " +
          "  ".join(f"{p} {100 * v:+.3f}%" for p, v in pp.items()))
json.dump(out, open(OUT, "w"), indent=1)
