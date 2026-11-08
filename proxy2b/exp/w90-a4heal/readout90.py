#!/usr/bin/env python3
"""readout90.py O -- README.md's rules on O/pairs.jsonl (run90.sh): per arm and seed r = KLD(arm) / KLD(heal0) and the
gain kept g = (stack - arm) / (stack - heal0); item 2 (t512) and item 3 (q4; q8 reported) verdicts, the rerun's noise
floor n = |log r(rr)| widening the bands to max(band, 2n)."""
import json, math, sys

O = sys.argv[1]
P = {}
for l in open(f"{O}/pairs.jsonl"):
    x = json.loads(l); P[x["pair"]] = x["r"]


def get(tag):
    """(r, z, g) of an arm tag, or None"""
    a, b = P.get(f"heal0 -> {tag}"), P.get(f"stack -> {tag}")
    if a is None or b is None: return None
    g = (b["kld_A"] - b["kld_B"]) / (b["kld_A"] - a["kld_A"])
    return a["ratio"], a["z"], g, a["kl99_ratio"], a["top1_diff"]


rr = P.get("heal0 -> k3rr")
n = abs(math.log(rr["ratio"])) if rr else 0.0
print(f"noise floor (heal0 rerun, K3 s0): {'r %.4f (z %+.2f), n = |log r| %.4f' % (rr['ratio'], rr['z'], n) if rr else 'missing'}")
band = lambda b: max(b, 2 * n)
res = {}
for arm in ("t512", "q4", "q8"):
    for k in (3, 2):
        rows = [(s, get(f"k{k}{arm}{'' if s == 0 else '_s1'}")) for s in (0, 1)]
        for s, v in rows:
            if v: print(f"{arm} K{k} s{s}: r {v[0]:.4f} (z {v[1]:+.2f}), gain kept {v[2]:.3f}, KL99 ratio {v[3]:.4f}, top-1 {v[4]:+.4f}")
            else: print(f"{arm} K{k} s{s}: missing")
        vs = [v for _, v in rows if v]
        if len(vs) == 2:
            res[(arm, k)] = {"mean_r": sum(v[0] for v in vs) / 2, "max_r": max(v[0] for v in vs), "mean_g": sum(v[2] for v in vs) / 2}
            print(f"{arm} K{k}: mean r {res[(arm, k)]['mean_r']:.4f}, mean gain kept {res[(arm, k)]['mean_g']:.3f}")
t = [res.get(("t512", k)) for k in (3, 2)]
if all(t):
    if all(x["mean_r"] <= 1 + band(0.01) and x["max_r"] <= 1 + band(0.02) for x in t): v2 = "SAME (a stored top-512 + tail teacher heals as the full vocabulary does)"
    elif any(x["mean_r"] > 1 + band(0.03) for x in t): v2 = "LOSES (the 27B pilot needs the full-vocabulary teacher)"
    else: v2 = "SMALL LOSS (between the bands; the lead weighs it)"
else: v2 = "INCOMPLETE"
q = [res.get(("q4", k)) for k in (3, 2)]
if all(q):
    if all(x["mean_g"] >= 0.95 for x in q): v3 = "SATURATED (1/4 of the data keeps >= 95% of the gain at both K)"
    elif any(x["mean_g"] < 0.90 for x in q): v3 = "DATA-LIMITED (1/4 of the data keeps < 90% of the gain at some K)"
    else: v3 = "IN BETWEEN (1/4 keeps 90-95%)"
else: v3 = "INCOMPLETE"
print(f"W90 ITEM2 (top-512 target): {v2}")
print(f"W90 ITEM3 (data starvation, 1/4): {v3}")
json.dump({"noise_n": n, "res": {f"{a}|K{k}": v for (a, k), v in res.items()}, "item2": v2, "item3": v3},
          open(f"{O}/readout90.json", "w"), indent=1)
