#!/usr/bin/env python3
"""aqcal.py -- calibrate the activation-quant proxy (aq.jsonl) against the E1b KLD runs: for each run, the proxy sum
over the tensors its mask quantizes (at its bits / group / clip) vs the measured KLD increment over the no-op run."""
import json, math, os, collections
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
aq = {}
for l in open(f"{D}/aq.jsonl"):
    r = json.loads(l); aq[(r["i"], r["bits"], r["group"], round(r["clip"], 2))] = r
base = None; runs = []
for l in open(f"{D}/e1b-results.jsonl"):
    r = json.loads(l)
    if "kld" not in r: continue
    if r["tag"] == "e0-ident": base = r["kld"]; continue
    env = dict(e.split("=", 1) for e in r["env"])
    if "EXL3_AQ_MASK" not in env: continue
    runs.append((r["tag"], env, r["kld"]))
rows = []
for tag, env, kld in runs:
    m = env["EXL3_AQ_MASK"]; g = int(env.get("EXL3_AQ_GROUP", 0)); c = round(float(env.get("EXL3_AQ_CLIP", 1.0)), 2)
    idx = [i for i, ch in enumerate(m) if ch in "48"]
    if not idx: continue
    b = int(m[idx[0]])
    keys = [(i, b, g, c) for i in idx]
    if not all(k in aq for k in keys): print("no proxy for", tag, g, c); continue
    p_out = sum(aq[k]["rel_out"] for k in keys); p_in = sum(aq[k]["rel_in"] for k in keys)
    rows.append((tag, kld - base, p_out, p_in))
for name, j in (("rel_out", 2), ("rel_in", 3)):
    xs = [math.log(r[j]) for r in rows]; ys = [math.log(max(r[1], 1e-6)) for r in rows]
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys)); sxx = sum((x - mx) ** 2 for x in xs); syy = sum((y - my) ** 2 for y in ys)
    print(f"{name}: log-log corr {sxy / math.sqrt(sxx * syy):.3f}, slope {sxy / sxx:.2f}, n={n}")
kglob = k = sorted(r[1] / r[2] for r in rows)[len(rows) // 2]
print(f"\nmedian dKLD / sum(rel_out) = {k:.4f}")
print(f"{'run':28} {'dKLD':>9} {'proxy*k':>9} ratio")
for tag, d, po, pi in sorted(rows, key=lambda r: r[1]):
    print(f"{tag:28} {d:9.5f} {po * k:9.5f} {d / (po * k):5.2f}")

# per-family factors: KLD ~ sum_f k_f * sum(rel_out over the run's tensors of family f); least squares in log space
import numpy as np
lab = None
import mkmask
mp = sorted(tuple(map(int, l.split())) for l in open(f"{D}/e0-ident.map") if l.strip())
lab = mkmask.label(mp)
fams = sorted(set(f for f, _ in lab))
X = []; Y = []; tags = []
for tag, env, kld in runs:
    m = env["EXL3_AQ_MASK"]; g = int(env.get("EXL3_AQ_GROUP", 0)); c = round(float(env.get("EXL3_AQ_CLIP", 1.0)), 2)
    idx = [i for i, ch in enumerate(m) if ch in "48"]
    keys = [(i, int(m[i]), g, c) for i in idx]
    if not idx or not all(k in aq for k in keys) or kld - base < 2e-5: continue
    v = np.zeros(len(fams))
    for k in keys: v[fams.index(lab[k[0]][0])] += aq[k]["rel_out"]
    X.append(v); Y.append(kld - base); tags.append(tag)
X = np.array(X); Y = np.array(Y)
from scipy.optimize import least_squares
f = lambda lk: np.log(X @ np.exp(lk)) - np.log(Y)
sol = least_squares(f, np.full(len(fams), math.log(0.011)))
kf = np.exp(sol.x); res = f(sol.x)
print("\nper-family factors:", {fa: round(float(v), 4) for fa, v in zip(fams, kf)})
print(f"log-RMSE {math.sqrt((res ** 2).mean()):.3f} (one global factor: {math.sqrt(((np.log(X.sum(1) * kglob) - np.log(Y)) ** 2).mean()):.3f}), n={len(Y)}")
for t, r in sorted(zip(tags, res), key=lambda z: -abs(z[1]))[:6]: print(f"  worst {t:28} {math.exp(r):.2f}x")

json.dump({"factors": {fa: float(v) for fa, v in zip(fams, kf)}, "log_rmse": float(math.sqrt((res ** 2).mean())),
           "n": int(len(Y)), "note": "dKLD ~ sum_f k_f * sum(rel_out at clip 1); calib-A, K6 base"},
          open(f"{D}/aqfactors.json", "w"), indent=1)
