#!/usr/bin/env python3
"""prelim.py -- preliminary EXL3-V2 table: round-8 mixes on the held-out ver1 slice, scaled to the leaderboard by the
ratio (leaderboard / ver1) of the same turboderp containers, vs the leaderboard Pareto frontier (others' entries) and
the unsloth UD frontier."""
import json, math, os, re, sys
D = os.path.dirname(os.path.abspath(__file__))
lb = json.load(open(f"{D}/results-2026-10-30.json")); P = lb["params"]
ent = [(e["bytes"] * 8 / P, e["scores"]["kld_mean"], e["scores"]["top1"], e["author"], e["group"], e["name"])
       for e in lb["entries"] if e.get("scores", {}).get("kld_mean")]
txt = open(f"{D}/ver1.txt").read()
refs = {m[0]: (float(m[1]), float(m[2])) for m in re.findall(r"(s?c?[\w.]+-e\d) ([\d.]+):([\d.]+)", txt)}
ours = sorted((float(a), float(b), float(c)) for a, b, c in re.findall(r"^\d\.\d\d ([\d.]+) ([\d.e-]+) ([\d.]+)$", txt, re.M))
name = {"sc1.40-e3": "EXL3-SC-1.40bpw", "sc1.60-e3": "EXL3-SC-1.60bpw", "sc1.80-e3": "EXL3-SC-1.80bpw",
        "sc2.00-e3": "EXL3-SC-2.00bpw", "sc2.20-e3": "EXL3-SC-2.20bpw", "plain2.50-e3": "EXL3-2.50bpw",
        "sc3.00-e4": "EXL3-SC-3.00bpw", "plain3.00-e4": "EXL3-3.00bpw", "plain3.50-e4": "EXL3-3.50bpw",
        "sc4.00-e4": "EXL3-SC-4.00bpw", "sc5.00-e5": "EXL3-SC-5.00bpw", "sc6.00-e6": "EXL3-SC-6.00bpw"}
byname = {e[5]: e for e in ent}
cal = sorted((refs[c][0], byname[n][1] / refs[c][1]) for c, n in name.items() if c in refs and n in byname)
def interp(pts, x, log=True):
    pts = sorted(pts)
    if x <= pts[0][0]: i = 0
    elif x >= pts[-1][0]: i = len(pts) - 2
    else: i = max(j for j in range(len(pts) - 1) if pts[j][0] <= x)
    (x0, y0), (x1, y1) = pts[i], pts[i + 1]
    t = (x - x0) / (x1 - x0)
    return math.exp(math.log(y0) + t * (math.log(y1) - math.log(y0))) if log else y0 + t * (y1 - y0)
def pareto(pts):
    out = []
    for b, k in sorted(pts):
        if not out or k < out[-1][1]: out.append((b, k))
    return out
others = [e for e in ent if e[3] != "Hero422" and e[4] != "official"]
front = pareto([(e[0], e[1]) for e in others])
ud = pareto([(e[0], e[1]) for e in ent if e[4] == "ud"])
best_at = lambda b: min((e for e in others if e[0] <= b * 1.0025), key=lambda e: e[1], default=None)
print("calibration leaderboard/ver1 (same containers):", " ".join(f"{b:.2f}:{r:.2f}" for b, r in cal))
print(f"\n{'bpw':>6} {'ver1 KLD':>9} {'est. LB':>9} {'top1':>7} | {'best other at <= size':32} | {'vs LB front':>11} {'vs unsloth':>10}")
for b, k, t1 in ours:
    est = k * interp(cal, b, log=False)
    be = best_at(b)
    fr = interp(front, b); u = interp(ud, b)
    print(f"{b:6.3f} {k:9.5f} {est:9.5f} {t1:7.4f} | {(be[5] + f' {be[0]:.2f} {be[1]:.5f}') if be else '-':32} | {est / fr:10.2f}x {est / u:9.2f}x")
