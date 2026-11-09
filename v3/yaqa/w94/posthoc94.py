#!/usr/bin/env python3
"""posthoc94.py DATA -- REPORT-ONLY (written after #125's G2 failed; not a verdict): the weighted joint estimate of the
proxy's head-only KL from hq94 score2's per-stratum sums, KL = -sum_s w_s a_s + log(1 + sum_s w_s b_s) (w_A = 1), next
to the additive estimate of the rule; its agreement with headsim94's full-head u3 / u4 for the real donors' rows, and
the arm ratios vs B under it (search-gen, qad-even, qad-odd; document bootstrap)."""
import json, sys
import numpy as np

D = sys.argv[1]; rng = np.random.default_rng(0)
B = {"wB": 10.0, "wC": (1940 - 256) / 24}
def gm(s): return np.fromfile(f"{D}/meta/{s}/gen.u8", np.uint8).astype(bool), np.fromfile(f"{D}/meta/{s}/doc.i32", np.int32)
M = {s: gm(s) for s in ("search", "qad")}
SETS = {"search-gen": ("search", M["search"][0]), "qad-even": ("qad", M["qad"][0] & (M["qad"][1] % 2 == 0)),
        "qad-odd": ("qad", M["qad"][0] & (M["qad"][1] % 2 == 1))}
sim = {s: np.load(f"{D}/sim-{s}.npz") for s in ("search", "qad")}
def simkl(s, n): z = sim[s]; i = [str(x) for x in z["names"]].index(n); return z["kl"][i].astype(np.float64)
def est(s, tag, joint=True):
    z = np.load(f"{D}/sums/{s}/{tag}.npz")
    if not joint: return z["kA"] + B["wB"] * z["kB"] + B["wC"] * z["kC"]
    a = z["aA"] + B["wB"] * z["aB"] + B["wC"] * z["aC"]; b = z["bA"] + B["wB"] * z["bB"] + B["wC"] * z["bC"]
    return -a + np.log1p(b)
def ratio(rs, x, y):
    s, m = SETS[rs]; d = M[s][1][m]; u, inv = np.unique(d, return_inverse=True)
    sx, sy = np.bincount(inv, x[m]), np.bincount(inv, y[m]); bi = rng.integers(0, len(u), (2000, len(u)))
    r = sx[bi].sum(1) / sy[bi].sum(1); return sx.sum() / sy.sum(), np.percentile(r, 2.5), np.percentile(r, 97.5)
print("estimator check (search-gen, real donor rows vs headsim full head):")
for K in (3, 4):
    m = SETS["search-gen"][1]; h = simkl("search", f"u{K}")[m].mean()
    for j in (False, True):
        e = est("search", f"real_one_K{K}", j)[m].mean()
        print(f"  K{K} {'joint' if j else 'additive'}: est / headsim = {e / h:.4f}")
    e1, e2 = est("search", f"raw_one_K{K}"), est("search", f"real_one_K{K}")
    print(f"  K{K} joint C0 / real: {ratio('search-gen', e1, e2)}")
print("arm ratios vs B (joint estimate):")
for K in (3, 4):
    for tag in ("raw_one", "mr_two0.01", "mr_two0.1", "mr_two1", "mr-r4_one", "mr-r8_one"):
        r = {rs: ratio(rs, est(SETS[rs][0], f"{tag}_K{K}"), est(SETS[rs][0], f"mr_one_K{K}")) for rs in SETS}
        s, m = SETS["search-gen"]; g = m & (sim["search"]["top"] >= 20000)
        gr = est("search", f"{tag}_K{K}")[g].sum() / est("search", f"mr_one_K{K}")[g].sum()
        print(f"  K{K} {tag:11s} " + "  ".join(f"{rs} {v[0]:.4f} [{v[1]:.4f}, {v[2]:.4f}]" for rs, v in r.items()) + f"  guard {gr:.3f}")
