#!/usr/bin/env python3
"""roundplot.py RESULTS.jsonl OUT.png -- EXL3-V2 progress by round: each round's mean-model mixes, as KLD relative to
the turboderp reference curve (sc/plain containers, log-interpolated in bpw) measured on the SAME slice, so rounds on
different fit slices (a, f2, f3) are comparable. Held-out verify-slice points (w- tags) are drawn hollow.
Prints a table: per round, the geometric-mean ratio over bpw bands (low < 2.5 <= mid < 5 <= high)."""
import json, math, re, sys, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = 26895998464
res = {}
for l in open(sys.argv[1]):
    r = json.loads(l)
    if "kld" in r and r.get("bytes"): res[r["tag"]] = r
bpw = lambda r: r["bytes"] * 8 / P
REF = r"(sc\d\.\d\d-e\d|plain\d\.\d\d-e\d)"


def refcurve(prefix):
    pts = sorted((bpw(r), r["kld"]) for t, r in res.items()
                 if re.fullmatch(rf"{re.escape(prefix)}ref-{REF}", t) and re.search(r"-e\d$", t))
    best = []                                          # lower envelope: keep a point only if it beats every smaller one
    for b, k in pts:
        if not best or k < best[-1][1]: best.append((b, k))
    return best


def ratio(curve, b, k):
    if len(curve) < 2 or not (curve[0][0] <= b <= curve[-1][0]): return None
    x = [c[0] for c in curve]; y = [math.log(c[1]) for c in curve]
    return k / math.exp(np.interp(b, x, y))


slices = {"a": "a-", "f2": "f2-", "f3": "f3-", "ver1": "w-"}
curves = {s: refcurve(p if s != "ver1" else "w-") for s, p in slices.items()}
rows = collections.defaultdict(list)                   # (round, slice) -> [(bpw, ratio)]
for t, r in res.items():
    m = re.fullmatch(r"(a|f2|f3|w)-knap-(r\d+|s\d+|k2)-sc4\.00-e4-([\d.]+)", t)
    if not m: continue
    s = "ver1" if m.group(1) == "w" else m.group(1)
    q = ratio(curves[s], bpw(r), r["kld"])
    if q is not None: rows[(m.group(2), s)].append((bpw(r), q))

def rkey(x): return -1 if x == "k2" else (int(x[1:].split("-")[0]) + (1000 if x[0] == "s" else 0))
# streaming fits: grouped in blocks of 10 fits so each point has enough targets
for k in list(rows):
    if k[0].startswith("s"):
        blk = f"s{(int(k[0][1:]) - 1) // 10 * 10 + 1}-{(int(k[0][1:]) - 1) // 10 * 10 + 10}"
        rows[(blk, k[1])] += rows.pop(k)
fig, ax = plt.subplots(figsize=(11, 6))
cm = plt.get_cmap("viridis")
rounds = sorted({k[0] for k in rows}, key=rkey)
for i, rd in enumerate(rounds):
    for s in slices:
        pts = sorted(rows.get((rd, s), []))
        if not pts: continue
        c = cm(i / max(1, len(rounds) - 1))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-" if s != "ver1" else "--", color=c, lw=1.2,
                marker="o", mfc=c if s != "ver1" else "none", ms=4, label=f"{rd} ({s})")
ax.axhline(1.0, color="k", lw=0.8)
ax.set_xlabel("real bpw"); ax.set_ylabel("KLD / turboderp reference curve (same slice)")
ax.set_title("EXL3-V2 mean mixes by round (solid: fit slice, dashed hollow: held-out ver1)")
ax.legend(fontsize=7, ncol=3); ax.grid(alpha=0.3); ax.set_ylim(0.3, 1.3)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=110)

print(f"{'round':6} {'slice':5} {'n':>3} {'<2.5':>7} {'2.5-5':>7} {'>=5':>7}")
for rd in rounds:
    for s in slices:
        pts = rows.get((rd, s))
        if not pts: continue
        def gm(lo, hi):
            v = [q for b, q in pts if lo <= b < hi]
            return f"{math.exp(sum(map(math.log, v)) / len(v)):7.3f}" if v else "      -"
        print(f"{rd:6} {s:5} {len(pts):3} {gm(0, 2.5)} {gm(2.5, 5)} {gm(5, 99)}")
