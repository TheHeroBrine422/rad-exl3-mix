#!/usr/bin/env python3
"""verdict122.py E3D_DIR SLICES_DIR E3_JSON OUT.json -- w122 E3 (d) rule verdicts (RULES.md "E3 (d)").

Per pair (trunk t3 / t4, budget B): A = S-slab rows, B = U rows on golden-hd122-x and -c (gen-masked scored positions,
readout.py's load_part). Per part mean KL and ratio U / S; pooled ratio = sum KL U / sum KL S over both parts; 95% CI
from a 2,000-resample paired document bootstrap, seed 0, resampled within each part. Criterion from w119's search-gen
head-KL ratio U / S-slab (E3_JSON budgets[B].S_over_U inverted): inside [0.95, 1.05] -> no-harm (|pooled - 1| <= 0.003),
else pass (U better on both parts and pooled upper bound < 1). V2's own head and trunk + bf16 head are reported beside."""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../eval/docstat"))
from readout import load_part

E, S, E3, OUT = sys.argv[1:5]
PAIRS = [("t3", 3), ("t3", 4), ("t4", 4), ("t4", 5)]
e3 = json.load(open(E3))["budgets"]


def sums(arm, p):
    f = f"{E}/scores/{arm}-{p}.json.rows"
    if not os.path.exists(f): return None
    return np.array([d["kl"].sum() for d in load_part(f"{S}/golden-hd122-{p}", f)]), np.array([len(d["kl"]) for d in load_part(f"{S}/golden-hd122-{p}", f)])


def boot(parts, n=2000, seed=0):
    rng = np.random.default_rng(seed); num = np.zeros(n); den = np.zeros(n)
    for a, b in parts:
        idx = rng.integers(0, len(a), (n, len(a))); num += b[idx].sum(1); den += a[idx].sum(1)
    return np.percentile(num / den, [2.5, 97.5]).tolist()


res = {}
for t, B in PAIRS:
    key = f"{t}-B{B}"; sa = {p: sums(f"{t}-S{B}", p) for p in "xc"}; sb = {p: sums(f"{t}-U{B}", p) for p in "xc"}
    if any(v is None for v in list(sa.values()) + list(sb.values())): res[key] = {"status": "missing"}; continue
    head = 1 / e3[str(B)]["S_over_U"][0]; crit = "no-harm" if 0.95 <= head <= 1.05 else "pass"
    r = {"head_ratio_search_gen": head, "criterion": crit, "parts": {}}
    for p in "xc":
        a, n = sa[p]; b, _ = sb[p]
        r["parts"][p] = {"docs": len(a), "positions": int(n.sum()), "kl_S": a.sum() / n.sum(), "kl_U": b.sum() / n.sum(),
                         "ratio": b.sum() / a.sum(), "ci95": boot([(a, b)])}
        for other in ("own", "bf16"):
            o = sums(f"{t}-{other}", p)
            if o is not None: r["parts"][p][f"kl_{other}"] = o[0].sum() / o[1].sum()
    pooled = sum(sb[p][0].sum() for p in "xc") / sum(sa[p][0].sum() for p in "xc")
    ci = boot([(sa[p][0], sb[p][0]) for p in "xc"])
    r["pooled"] = {"ratio": pooled, "ci95": ci}
    if crit == "no-harm": ok = abs(pooled - 1) <= 0.003
    else: ok = all(r["parts"][p]["ratio"] < 1 for p in "xc") and ci[1] < 1
    r["verdict"] = "PASS" if ok else "FAIL (unverified until the confounder check)"
    res[key] = r
    print(f"{key}: head U/S {head:.3f} -> {crit}; x {r['parts']['x']['ratio']:.4f} {np.round(r['parts']['x']['ci95'], 4).tolist()} "
          f"c {r['parts']['c']['ratio']:.4f} {np.round(r['parts']['c']['ci95'], 4).tolist()} pooled {pooled:.4f} "
          f"{np.round(ci, 4).tolist()} -> {r['verdict']}")
    for p in "xc":
        q = r["parts"][p]
        print(f"   {p}: KL S {q['kl_S']:.5f} U {q['kl_U']:.5f} own {q.get('kl_own', float('nan')):.5f} bf16-head {q.get('kl_bf16', float('nan')):.5f}")
done = [v for v in res.values() if "verdict" in v]
res["all_pass"] = len(done) == len(PAIRS) and all(v["verdict"] == "PASS" for v in done)
print("E3 (d):", "U CONFIRMED whole-model (every pair passes)" if res["all_pass"] else f"{sum(v['verdict'] == 'PASS' for v in done)} / {len(PAIRS)} pairs pass ({len(done)} read)")
json.dump(res, open(OUT, "w"), indent=1)
