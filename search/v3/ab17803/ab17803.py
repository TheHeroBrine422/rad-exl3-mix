#!/usr/bin/env python3
"""ab17803.py DOCSUMS.json REGISTRY OUT.json -- RULES-17803 (w149, #178-03): a/b arms T4 / T5 / T6 vs B (bf16 a/b) on the
5.0 / 6.0 / 6.5 incumbents, s173f. d = pmean over parts {c, x} of log(sum KL_arm / sum KL_B); 95% CIs from 2,000 paired
document bootstraps stratified by part; trunk price g = b_s x bpw saved (b_s: the driver's last slope, RULES-value);
net N = d - g; RULES-178 reading per arm; per-size choice: T4 if N_T4 < 0, else the first of T5, T6 with N < 0, else B."""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import slices as SL          # noqa: E402

D = json.load(open(sys.argv[1])); reg = SL.Registry(sys.argv[2])
PARAMS = 26_895_998_464; PARAMS_NOEMB = PARAMS - 248_320 * 5_120    # container metric / embedding excluded
NW = 2 * 48 * 5120 * 48                                              # a + b weights, 48 GDN layers
SCALES = 0.009356717 - 0.00918                                       # w147: packed T4 saves 0.00918 incl. scale vectors
SAVED = {f"T{K}": (16 - K * 128 / 96) * NW / PARAMS - SCALES for K in (4, 5, 6)}
SAVED_NOEMB = {a: v * PARAMS / PARAMS_NOEMB for a, v in SAVED.items()}
BS = {"5.0": 0.781, "6.0": 0.350, "6.5": 0.158}
ARMS = ("T4", "T5", "T6")
parts = np.array(reg.parts_of("s173f")); P = sorted(set(parts.tolist())); grp = [np.where(parts == p)[0] for p in P]
rng = np.random.default_rng(17803); draws = [np.concatenate([g[rng.integers(0, len(g), len(g))] for g in grp]) for _ in range(2000)]


def vec(k): return np.array([d[0] for d in D[k]["docs"]])


def pmean(A, B, ix):
    pi = parts[ix]; return float(np.mean([math.log(B[ix][pi == p].sum() / A[ix][pi == p].sum()) for p in P]))


out = {"n_docs": {p: int((parts == p).sum()) for p in P}, "bpw_saved": SAVED, "bpw_saved_noemb": SAVED_NOEMB, "b": BS,
       "arms": {}, "choice": {}}
print(f"docs {out['n_docs']}  (s173f)")
print(f"{'size':>4} {'arm':>3} {'bpw saved':>9} {'d':>8} {'95% CI':>19} {'c':>8} {'x':>8} {'g':>7} {'N=d-g':>8} {'reading':>8}  KLD")
for s in BS:
    if f"{s}|B" not in D or any(f"{s}|{a}" not in D for a in ARMS): print(f"{s}: incomplete"); continue
    A = vec(f"{s}|B"); ax = np.arange(len(A))
    assert len(A) == len(parts)
    print(f"{s:>4} {'B':>3} {0:9.5f} {'':>8} {'':>19} {'':>8} {'':>8} {'':>7} {'':>8} {'':>8}  {D[f'{s}|B']['kld']:.5f}")
    for a in ARMS:
        k = f"{s}|{a}"
        B_ = vec(k); d = pmean(A, B_, ax); bs = np.array([pmean(A, B_, ix) for ix in draws]); lo, hi = np.percentile(bs, [2.5, 97.5])
        g = BS[s] * SAVED[a]; n = d - g
        rd = "adopted" if n < 0 and hi < g + 0.005 else "rejected" if lo > g else "null"
        pp = {p: math.log(B_[parts == p].sum() / A[parts == p].sum()) for p in P}
        out["arms"][k] = {"d": d, "ci95": [float(lo), float(hi)], "per_part": pp, "g": g, "net": n, "reading": rd,
                          "kld": D[k]["kld"], "kld_B": D[f"{s}|B"]["kld"], "ratio": math.exp(d), "bpw_saved": SAVED[a],
                          "bpw_saved_noemb": SAVED_NOEMB[a], "unverified": abs(d) > 0.03}
        print(f"{s:>4} {a:>3} {SAVED[a]:9.5f} {d:+8.4f} [{lo:+.4f}, {hi:+.4f}] {pp['c']:+8.4f} {pp['x']:+8.4f} {g:7.4f} {n:+8.4f} {rd:>8}  {D[k]['kld']:.5f}"
              + ("  UNVERIFIED |d|>3%" if abs(d) > 0.03 else ""))
    pick = next((a for a in ARMS if out["arms"][f"{s}|{a}"]["net"] < 0), "B")
    noise = pick != "T4" and out["arms"][f"{s}|T4"]["reading"] == "null"
    out["choice"][s] = {"arm": pick, "noise_level": noise}
    print(f"{s:>4} choice: {pick}" + ("  (noise-level: T4's reading is null)" if noise else ""))
json.dump(out, open(sys.argv[3], "w"), indent=1)
