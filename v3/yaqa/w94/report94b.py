#!/usr/bin/env python3
"""report94b.py DATA -- #125-01 (RULES.md amendment 1): exact full-head KL of the real head quants C0' (u3 / u4), B
mean-row (u13 / u14), T mean-row + two-sided (u23 / u24) from headsim94 (sim1-<set>.npz), ratios with 2,000-draw
document-bootstrap 95% intervals, guard ratios, the verdicts, and the check C0' == today's donors (sim-<set>.npz)."""
import json, sys
import numpy as np

D = sys.argv[1]; rng = np.random.default_rng(0)
def gm(s): return np.fromfile(f"{D}/meta/{s}/gen.u8", np.uint8).astype(bool), np.fromfile(f"{D}/meta/{s}/doc.i32", np.int32)
M = {s: gm(s) for s in ("search", "qad")}
SETS = {"search-gen": ("search", M["search"][0]), "search-all": ("search", np.ones_like(M["search"][0])),
        "qad-gen": ("qad", M["qad"][0]), "qad-even": ("qad", M["qad"][0] & (M["qad"][1] % 2 == 0)),
        "qad-odd": ("qad", M["qad"][0] & (M["qad"][1] % 2 == 1))}
S1 = {s: np.load(f"{D}/sim1-{s}.npz") for s in ("search", "qad")}; S0 = {s: np.load(f"{D}/sim-{s}.npz") for s in ("search", "qad")}
def kl(z, n): return z["kl"][[str(x) for x in z["names"]].index(n)].astype(np.float64)
def ratio(rs, x, y):
    s, m = SETS[rs]; d = M[s][1][m]; u, inv = np.unique(d, return_inverse=True)
    sx, sy = np.bincount(inv, x[m]), np.bincount(inv, y[m]); bi = rng.integers(0, len(u), (2000, len(u)))
    r = sx[bi].sum(1) / sy[bi].sum(1); return [float(sx.sum() / sy.sum()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]
res = {"check": {}, "kl": {}, "ratios": {}, "guard": {}}
for s in ("search", "qad"):
    for K in (3, 4):
        res["check"][f"{s}-K{K}"] = float(np.abs(kl(S1[s], f"u{K}") - kl(S0[s], f"u{K}")).max())
print("C0' vs today's donors (max |dKL| per row):", res["check"])
arms = {"C0'": "u{K}", "B": "u1{K}", "T": "u2{K}"}
for K in (3, 4):
    for rs in SETS:
        s = SETS[rs][0]; res["kl"][f"{rs}-K{K}"] = {a: float(kl(S1[s], n.format(K=K))[SETS[rs][1]].mean()) for a, n in arms.items()}
    for a, b in (("T", "B"), ("B", "C0'"), ("T", "C0'")):
        for rs in SETS:
            s = SETS[rs][0]
            res["ratios"][f"{a}/{b}-K{K}-{rs}"] = ratio(rs, kl(S1[s], arms[a].format(K=K)), kl(S1[s], arms[b].format(K=K)))
        s, m = SETS["search-gen"]; g = m & (S1["search"]["top"] >= 20000)
        res["guard"][f"{a}/{b}-K{K}"] = float(kl(S1["search"], arms[a].format(K=K))[g].sum() / kl(S1["search"], arms[b].format(K=K))[g].sum())
print("| K | pair | search-gen | qad-even | qad-odd | search-all | guard (search-gen) |"); print("|---|---|---|---|---|---|---|")
f = lambda r: f"{r[0]:.4f} [{r[1]:.4f}, {r[2]:.4f}]"
for K in (3, 4):
    for a, b in (("T", "B"), ("B", "C0'"), ("T", "C0'")):
        print(f"| {K} | {a} / {b} | " + " | ".join(f(res["ratios"][f"{a}/{b}-K{K}-{rs}"]) for rs in ("search-gen", "qad-even", "qad-odd", "search-all"))
              + f" | {res['guard'][f'{a}/{b}-K{K}']:.3f} |")
for K in (3, 4): print(f"K{K} KL search-gen:", {a: f"{v:.6f}" for a, v in res["kl"][f"search-gen-K{K}"].items()})
R = res["ratios"]
res["T_adopt"] = all(R[f"T/B-K{K}-{rs}"][0] <= 0.97 for K in (3, 4) for rs in ("search-gen", "qad-even")) and \
                 all(res["guard"][f"T/B-K{K}"] <= 1.3 for K in (3, 4))
res["mean_row"] = ("recommend" if all(R[f"B/C0'-K{K}-search-gen"][0] < 1 for K in (3, 4)) else
                   "recommend dropping" if all(R[f"B/C0'-K{K}-{rs}"][0] > 1 for K in (3, 4) for rs in ("search-gen", "qad-even")) else "no recommendation")
print("T adopted:", res["T_adopt"], "| mean-row:", res["mean_row"])
json.dump(res, open(f"{D}/report94b.json", "w"), indent=1)
