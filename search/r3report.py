#!/usr/bin/env python3
"""r3report.py -- round-3 report on calib-A (run from /root after the queue is scored):
  1. k2 mixes (directional model + DP): predicted before measurement vs measured, and vs the reference containers'
     curve (log-KLD interpolated between whole containers measured on the same slice).
  2. perturbation designs (dz-*): spread of the design points around their centre; antithetic pairs:
     curvature = (K+ + K-)/2 / K0 - 1 (0 = locally additive), and the pair's half-difference.
  3. background swaps: dKLD per unit proxy error across backgrounds, per band."""
import json, re, glob, math, collections
import numpy as np
import os
# paths (README: Paths)
RXM_WORK = os.environ.get("RXM_WORK", "/root")
PARAMS = 26895998464
res = {}
for l in open(f"{RXM_WORK}/evq/results.jsonl"):
    r = json.loads(l)
    if "fail" not in r: res[r["tag"]] = r
bpw = lambda t: res[t]["bytes"] * 8 / PARAMS
refs = sorted([(bpw(t), res[t]["kld"], t) for t in res if t.startswith("a-ref-") or t == "base-sc4.00-e4"])
allrefs = refs; refs = []                     # Pareto envelope only: a bigger and worse container is not a reference
for r in allrefs:
    if not refs or r[1] < refs[-1][1]: refs.append(r)
xs = [r[0] for r in refs]; ys = [math.log(r[1]) for r in refs]
curve = lambda b: math.exp(np.interp(b, xs, ys))
print("reference frontier (calib-A):", ", ".join(f"{t.replace('a-ref-', '')} {b:.3f}:{k:.5f}" for b, k, t in refs))
print("dominated (not used):", ", ".join(t.replace('a-ref-', '') for b, k, t in allrefs if (b, k, t) not in refs))
pred = {json.loads(l)["name"]: json.loads(l) for l in open(f"{RXM_WORK}/knap/frontier-k2-r3pred.jsonl")}
print(f"\n== 1. k2 mixes\n{'mix':24} {'bpw':>6} {'pred':>9} {'meas':>9} {'meas/pred':>9} {'refs@bpw':>9} {'vs refs':>8} {'vs kf/fix same target':>22}")
for n in sorted(pred, key=lambda n: pred[n]["target"]):
    t = "a-" + n
    if t not in res: print(f"{n[8:]:24} not measured"); continue
    k = res[t]["kld"]; b = bpw(t); p = pred[n]["pred_kld"]
    tgt = n.rsplit("-", 1)[1]; old = [(o, res[f"a-knap-{o}-sc4.00-e4-{tgt}"]["kld"]) for o in ("kf", "fix") if f"a-knap-{o}-sc4.00-e4-{tgt}" in res]
    print(f"{n[8:]:24} {b:6.3f} {p:9.5f} {k:9.5f} {k/p:9.2f} {curve(b):9.5f} {k/curve(b)-1:+8.1%} " +
          " ".join(f"{o} {kk:.5f}" for o, kk in old))
print("\n== 2. perturbation designs (antithetic pairs 2k / 2k+1)")
for c, centre in (("d180", "1.80"), ("d250", "2.50"), ("d350", "3.50")):
    k0t = f"a-knap-k2-sc4.00-e4-{centre}"
    if k0t not in res: print(c, "centre not measured"); continue
    k0 = res[k0t]["kld"]; pts = sorted(t for t in res if t.startswith(f"a-dz-{c}-"))
    ks = [res[t]["kld"] / k0 for t in pts]
    if not ks: continue
    print(f"  {c}: centre {k0:.5f}; {len(ks)} points, KLD/centre min {min(ks):.3f} median {np.median(ks):.3f} max {max(ks):.3f}; "
          f"better than centre: {sum(k < 1 for k in ks)}")
    for i in range(0, len(pts) - 1, 2):
        a, b = res[pts[i]]["kld"], res[pts[i + 1]]["kld"]
        if pts[i + 1][-2:] != f"{int(pts[i][-2:]) + 1:02d}": continue
        print(f"     pair {pts[i][-2:]}/{pts[i+1][-2:]}: +{a/k0-1:+.1%} -{b/k0-1:+.1%}  curvature {(a+b)/2/k0-1:+.1%}")
print("\n== 3. background swaps (one step down; dKLD, ratio)")
bg = {t.replace("a-ref-", ""): res[t]["kld"] for t in res if t.startswith("a-ref-")}; bg["sc4.00-e4"] = res["base-sc4.00-e4"]["kld"]
for t in sorted(t for t in res if t.startswith("a-xp-sc")):
    m = re.match(r"a-xp-(sc[\d.]+-e\d)-(.+)$", t); g, band = m.groups()
    print(f"  {band:10} on {g:10}: {bg[g]:.5f} -> {res[t]['kld']:.5f}  dKLD {res[t]['kld']-bg[g]:+.5f}  ratio {res[t]['kld']/bg[g]:.3f}")
