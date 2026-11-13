#!/usr/bin/env python3
"""ab178.py DOCSUMS.json REGISTRY OUT.json -- RULES-178 (w142, #178): each a/b arm vs B at 2.5 and 4.0 on s171d. d = pmean
over parts {c, x} of log(sum KL_arm / sum KL_B); 95% CIs from 2,000 paired document bootstraps stratified by part; trunk
price g = b_s x bpw saved (b_s: the driver's last slope, RULES-value); net N = d - g; reading as RULES-178."""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import slices as SL          # noqa: E402

D = json.load(open(sys.argv[1])); reg = SL.Registry(sys.argv[2])
PARAMS = 26_895_998_464; NW = 2 * 48 * 5120 * 48                    # a + b weights, 48 GDN layers
BITS = {"B": 16, "G": 8 + 16 / 128, "R": 8 + 16 / 128, "T4": 4 * 128 / 96, "T5": 5 * 128 / 96, "T6": 6 * 128 / 96}
SAVED = {a: (16 - b) * NW / PARAMS for a, b in BITS.items()}
BS = {"2.5": 1.495, "4.0": 1.145}
parts = np.array(reg.parts_of("s171d")); P = sorted(set(parts.tolist())); grp = [np.where(parts == p)[0] for p in P]
rng = np.random.default_rng(178); draws = [np.concatenate([g[rng.integers(0, len(g), len(g))] for g in grp]) for _ in range(2000)]


def vec(k): return np.array([d[0] for d in D[k]["docs"]])


def pmean(A, B, ix):
    pi = parts[ix]; return float(np.mean([math.log(B[ix][pi == p].sum() / A[ix][pi == p].sum()) for p in P]))


out = {"n_docs": {p: int((parts == p).sum()) for p in P}, "bpw_saved": SAVED, "b": BS, "arms": {}}
print(f"docs {out['n_docs']}  (s171d)")
print(f"{'size':>4} {'arm':>3} {'bpw saved':>9} {'d':>8} {'95% CI':>19} {'c':>8} {'x':>8} {'g':>7} {'N=d-g':>8} {'reading':>8}  KLD")
for s in ("2.5", "4.0"):
    if f"{s}|B" not in D: continue
    A = vec(f"{s}|B"); ax = np.arange(len(A))
    print(f"{s:>4} {'B':>3} {0:9.5f} {'':>8} {'':>19} {'':>8} {'':>8} {'':>7} {'':>8} {'':>8}  {D[f'{s}|B']['kld']:.5f}")
    for a in ("G", "R", "T4", "T5", "T6"):
        k = f"{s}|{a}"
        if k not in D: continue
        B_ = vec(k); d = pmean(A, B_, ax); bs = np.array([pmean(A, B_, ix) for ix in draws]); lo, hi = np.percentile(bs, [2.5, 97.5])
        g = BS[s] * SAVED[a]; n = d - g
        rd = "adopted" if n < 0 and hi < g + 0.005 else "rejected" if lo > g else "null"
        pp = {p: math.log(B_[parts == p].sum() / A[parts == p].sum()) for p in P}
        out["arms"][k] = {"d": d, "ci95": [float(lo), float(hi)], "per_part": pp, "g": g, "net": n, "reading": rd,
                          "kld": D[k]["kld"], "kld_B": D[f"{s}|B"]["kld"], "ratio": math.exp(d), "bpw_saved": SAVED[a]}
        print(f"{s:>4} {a:>3} {SAVED[a]:9.5f} {d:+8.4f} [{lo:+.4f}, {hi:+.4f}] {pp['c']:+8.4f} {pp['x']:+8.4f} {g:7.4f} {n:+8.4f} {rd:>8}  {D[k]['kld']:.5f}")
json.dump(out, open(sys.argv[3], "w"), indent=1)
