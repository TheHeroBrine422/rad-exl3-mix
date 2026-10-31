#!/usr/bin/env python3
"""frontplot.py LEADERBOARD.json PUBLIC_FINALS.jsonl CALIB_RESULTS.jsonl OUT.png -- size/KLD frontiers:
leaderboard IQR (Deadcode, best of every version) and unsloth UD, turboderp's EXL3 exports (ours served, public),
and our knapsack mixes. The mixes are measured on calib half A only: their public value is ESTIMATED as
(mix / turboderp curve on calib-A at the mix's bpw) x (turboderp curve on public at that bpw), drawn dashed."""
import json, sys, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
LB, PUB, CAL, OUT = sys.argv[1:5]
PARAMS = 26895998464
lb = json.load(open(LB))
def env(pts):                                    # lower-left Pareto envelope
    out = []
    for b, k, *r in sorted(pts):
        if not out or k < out[-1][1]: out.append((b, k, *r))
    return out
iqr = [(e["bytes"] * 8 / PARAMS, e["scores"]["kld_mean"], e["name"]) for e in lb["entries"] if e["author"] == "Deadcode"]
ud = [(e["bytes"] * 8 / PARAMS, e["scores"]["kld_mean"], e["name"]) for e in lb["entries"] if e.get("group") == "ud"]
# turboderp exports on public (bytes: container sizes from the calib results' a-ref-/base entries)
cal = {}
for l in open(CAL):
    r = json.loads(l)
    if "fail" not in r: cal[r["tag"]] = r
size = {t[6:]: cal[t]["bytes"] * 8 / PARAMS for t in cal if t.startswith("a-ref-")}
size["sc4.00-e4"] = cal["base-sc4.00-e4"]["bytes"] * 8 / PARAMS
pub = {}
for l in (l for l in open(PUB) if l.startswith("{")):    # later lines (uncapped re-runs) win
    r = json.loads(l); pub[r["tag"]] = r["kld"]
td = [(size[t], pub[t], t) for t in pub if t in size]
calref = [(size[t], (cal["base-sc4.00-e4"] if t == "sc4.00-e4" else cal["a-ref-" + t])["kld"]) for t in size]
ce, pe = env(calref), env([(b, k) for b, k, _ in td])
curve = lambda e, b: math.exp(np.interp(b, [x for x, _ in e], [math.log(y) for _, y in e]))
mix = []
for t, r in cal.items():
    if t.startswith("a-knap-k2-"):
        b = r["bytes"] * 8 / PARAMS
        if b > pe[-1][0] + 0.05: continue        # no reference above the largest export: no estimate
        mix.append((b, r["kld"] / curve(ce, b) * curve(pe, b), t))
ours = env([(b, k, n) for b, k, n in td] + mix)

XL, YL = (2.0, 7.0), (5e-4, 0.35)
fig, ax = plt.subplots(figsize=(11, 6.5), dpi=130)
for x0, x1, nm in ((0, 3, "Nano"), (3, 4, "Tiny"), (4, 5.5, "Small"), (5.5, 7.5, "Medium"), (7.5, 9.5, "Large")):
    ax.axvspan(x0, x1, color="0.5", alpha=0.06 if nm in ("Tiny", "Medium") else 0.0)
    lo, hi = max(x0, XL[0]), min(x1, XL[1])
    if lo < hi: ax.text((lo + hi) / 2, YL[1] * 0.92, nm, ha="center", va="top", color="0.45", fontsize=9)
def series(pts, style, label, front=True, **kw):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    ax.scatter(xs, ys, s=18, **kw)
    if front:
        e = env(pts); ax.plot([p[0] for p in e], [p[1] for p in e], style, color=kw.get("color"), lw=1.6, label=label)
series(ud, "-", "unsloth UD (leaderboard)", color="#999999", marker="s")
series(env(iqr), "-", "Deadcode IQR frontier, best of all versions (leaderboard)", color="#d62728", marker="D")
series(td, "-", "turboderp EXL3 exports, our kernel (public, measured)", color="#1f77b4", marker="o")
ax.scatter([p[0] for p in mix], [p[1] for p in mix], s=26, color="#2ca02c", marker="*", zorder=5)
ax.plot([p[0] for p in ours], [p[1] for p in ours], "--", color="#2ca02c", lw=2,
        label="our frontier: exports + knapsack mixes (mixes ESTIMATED from calib-A)")
ax.set_yscale("log"); ax.set_xlim(*XL); ax.set_ylim(*YL)
ax.set_xlabel("real bpw (embedding and head included)"); ax.set_ylabel("mean KLD vs bf16 (log)")
ax.set_title("Qwen3.8-27B: size vs KLD frontiers (leaderboard: partial private scoring; ours: public corpus)")
ax.grid(True, which="both", alpha=0.25); ax.legend(loc="upper right", fontsize=8.5, bbox_to_anchor=(1, 0.93))
fig.tight_layout(); fig.savefig(OUT)
for b, k, n in ours: print(f"{b:6.3f} {k:.5f} {n}")
