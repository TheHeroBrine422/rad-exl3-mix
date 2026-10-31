#!/usr/bin/env python3
"""convplot.py RESULTS.jsonl OUT.png [LB.json] -- EXL3-V2 convergence ("loss") curve.

Every mix eval on a fit slice (a / f2 / f3) is put on one scale: its KLD mapped to the leaderboard via the same EXL3-V1
containers measured on that slice (ratio leaderboard / slice, linear in bpw, clamped), divided by the unsloth UD
frontier's KLD at the mix's real bpw (log-linear between UD quants, extended at the ends) -- the leaderboard's own
score, 1.0 = unsloth, lower is better. x = eval step (order in results.jsonl).

Per bpw band (<3, 3-5, >=5), per step:
  solid  : best so far -- geometric mean over the band's targets of the best score measured at that target
  dots   : each eval (faint), held-out ver1 evals as open circles at the step the mix was first measured
           on a fit slice (not where the ver1 eval ran); not in the solid line
Targets 1.6-8.0 only (8.5-9.5 dropped 2026-10-30; below 1.6 the UD line is extrapolated).
Vertical lines: the K1/K2 fit fix, the start of streaming.

Lower panel (2026-10-31, overfit check): at each eval step, the best-so-far pick per target the way wsnap.py /
finalsel.py pick it (fit-slice KLD / slice offset) at 2.0, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7 bpw; per band, the geo-mean
score of those picks on the fit slices (dashed) and on held-out ver1 (solid, markers), wherever every target of the
band has its pick measured on ver1. (2026-10-31: the stopping signal is the held-out score itself -- the
validation loss -- not the held-out / fit ratio: overfitting can grow while the held-out score still improves.)"""
import json, math, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES, OUT = sys.argv[1], sys.argv[2]
LBF = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "lb/results-2026-10-30.json")
lb = json.load(open(LBF)); P = lb["params"]
ent = {e["name"]: e for e in lb["entries"] if e.get("scores", {}).get("kld_mean") is not None}
ud = sorted((e["bytes"] * 8 / P, e["scores"]["kld_mean"]) for e in ent.values() if e["group"] == "ud")
UD = []
for b, k in ud:
    if not UD or k < UD[-1][1]: UD.append((b, k))


def loginterp(pts, b):
    i = 0 if b <= pts[0][0] else len(pts) - 2 if b >= pts[-1][0] else max(j for j in range(len(pts) - 1) if pts[j][0] <= b)
    (x0, y0), (x1, y1) = pts[i], pts[i + 1]
    return math.exp(math.log(y0) + (b - x0) / (x1 - x0) * (math.log(y1) - math.log(y0)))


rows = [json.loads(l) for l in open(RES)]
res = {r["tag"]: r for r in rows if "kld" in r and r.get("bytes")}
bpw = lambda r: r["bytes"] * 8 / P


def lbname(c):
    m = re.fullmatch(r"(sc|plain)(\d\.\d\d)-e\d", c)
    return None if not m else f"EXL3-{'SC-' if m.group(1) == 'sc' else ''}{m.group(2)}bpw"


cal = {}
for s in ("a", "f2", "f3", "w"):
    pts = sorted((bpw(r), ent[lbname(t[len(s) + 5:])]["scores"]["kld_mean"] / r["kld"]) for t, r in res.items()
                 if t.startswith(s + "-ref-") and lbname(t[len(s) + 5:]) in ent)
    if len(pts) >= 2: cal[s] = pts
calf = lambda s, b: float(np.interp(b, [p[0] for p in cal[s]], [p[1] for p in cal[s]]))

band = lambda t: 0 if t < 2.75 else 1 if t < 4.5 else 2 if t < 6 else 3   # = the stream's weighted bands (2026-10-31)
BN = ["< 2.75 bpw", "2.75–4.5 bpw", "4.5–6 bpw", "≥ 6 bpw"]; COL = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
NB = len(BN)
pts = [[] for _ in range(NB)]; held = [[] for _ in range(NB)]; lsp = [[] for _ in range(NB)]; marks = {}
best = [{} for _ in range(NB)]; curve = [[] for _ in range(NB)]
first = {}   # mix name -> step of its first fit-slice eval (ver1 circles go there, not where the ver1 eval ran)
for step, r in enumerate(rows):
    m0 = re.fullmatch(r"(?:a|f2|f3)-(knap-.+)", r["tag"])
    if m0 and "kld" in r: first.setdefault(m0.group(1), step)
for step, r in enumerate(rows):
    t = r["tag"]
    if "s1" not in marks and re.match(r"(a|f2|f3)-knap-s\d+", t): marks["s1"] = step
    if "k1fix" not in marks and t.startswith("cal-sc4.00-e4-ours-K1-"): marks["k1fix"] = step
    m = re.fullmatch(r"(a|f2|f3|w)-knap-(?:r\d+|s\d+|k2)(?:oo)?(?:-(?:ts|ls)\d)?-sc4\.00-e4-([\d.]+)", t)
    if m and t in res and m.group(1) in cal:
        rr = res[t]; b = bpw(rr); T = float(m.group(2)); bd = band(T)
        GRID = {1.6, 1.8, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25,
                6.5, 6.75, 7.0, 7.25, 7.5, 8.0}
        if T not in GRID: continue   # current targets with a UD reference nearby (8.5-9.5 dropped; < 1.6 extrapolated)
        sc = rr["kld"] * calf(m.group(1), b) / loginterp(UD, b)
        if m.group(1) == "w": held[bd].append((first.get(t[2:], step), sc))   # at the mix's first fit-slice eval
        else:
            (lsp if "-ls" in t else pts)[bd].append((step, sc))
            if sc < best[bd].get(T, 1e9): best[bd][T] = sc
    for bd in range(NB):
        if best[bd]: curve[bd].append((step, math.exp(np.mean([math.log(v) for v in best[bd].values()]))))

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
for bd in range(NB):
    if pts[bd]: ax.scatter(*zip(*pts[bd]), s=6, alpha=0.18, color=COL[bd])
    if held[bd]: ax.scatter(*zip(*held[bd]), s=22, facecolors="none", edgecolors=COL[bd], linewidths=0.8)
    if lsp[bd]: ax.scatter(*zip(*lsp[bd]), s=30, marker="x", color=COL[bd], linewidths=1.1, alpha=0.9)
    if curve[bd]: ax.plot(*zip(*curve[bd]), color=COL[bd], lw=2.2, label=f"{BN[bd]}: best so far (geo-mean over targets)")
for step, r in enumerate(rows):
    if "ls" not in marks and re.match(r"(a|f2|f3)-knap-s\d+-ls\d", r["tag"]): marks["ls"] = step
for k, lab in (("k1fix", "K1/K2 fit fix"), ("s1", "streaming starts"), ("ls", "local search")):
    if k in marks: ax.axvline(marks[k], color="gray", ls="--", lw=1); ax.text(marks[k], 1.35, " " + lab, fontsize=8, color="gray")
# IQR-V3 (group iqr3) on the same scale: its real leaderboard KLD / UD frontier, geo-mean over its entries per band
iqr = [[] for _ in range(NB)]
for e in ent.values():
    if e["group"] != "iqr3": continue
    b = e["bytes"] * 8 / P
    if 1.6 <= b <= 8.0: iqr[band(b)].append((b, e["scores"]["kld_mean"] / loginterp(UD, b)))
for bd in range(NB):
    if iqr[bd]:
        g = math.exp(np.mean([math.log(v) for _, v in iqr[bd]]))
        ax.axhline(g, color=COL[bd], ls=":", lw=1.6)
        ax.text(ax.get_xlim()[0] if False else min(x for x, _ in curve[bd]) if curve[bd] else 0, g, f" IQR-V3 {BN[bd]} ({len(iqr[bd])} entries, {g:.2f})",
                fontsize=8, color=COL[bd], va="bottom")
        print(f"IQR-V3 {BN[bd]}: " + ", ".join(f"{b:.2f}:{v:.2f}" for b, v in sorted(iqr[bd])) + f"  -> geo-mean {g:.3f}")
ax.axhline(1.0, color="k", lw=0.8)
ax.set_yscale("log"); ax.set_ylim(0.2, 1.5)
ax.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5]); ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel("eval step (all evals, in order)"); ax.set_ylabel("est. leaderboard KLD / unsloth UD frontier at same bpw")
ax.set_title("EXL3-V2 convergence (dots: mix evals; x: local search; open circles: held-out ver1; dotted: IQR-V3; lower is better)")
ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3, which="both")
# ---- lower panel: held-out ver1 of the best-so-far picks vs their fit-slice scores
OFF = {"a": 1.0, "f2": 0.848, "f3": 0.958}            # = wsnap.py (slice offsets of the late-stream fits)
HT = {0: ["2.00", "2.50"], 1: ["3.00", "3.50", "4.00"], 2: ["4.50", "5.00"], 3: ["6.00", "7.00"]}
PR = re.compile(r"(a|f2|f3)-(knap-.+-sc4\.00-e4-([\d.]+))")
pick, fitsc = {}, {}
TL = sum(HT.values(), [])
val = {bd: ([], []) for bd in list(HT) + ["all"]}; fitc = ([], [])
def gm(v): return math.exp(np.mean([math.log(x) for x in v]))
def ver1(t):
    w = res.get(f"w-{pick[t][1]}") if t in pick else None
    return w["kld"] * calf("w", bpw(w)) / loginterp(UD, bpw(w)) if w else None
for step, r in enumerate(rows):
    m = PR.fullmatch(r["tag"])
    if m and r["tag"] in res and m.group(3) in TL and m.group(1) in cal:
        k = res[r["tag"]]["kld"] / OFF[m.group(1)]
        if k < pick.get(m.group(3), (9,))[0]:
            rr = res[r["tag"]]; b = bpw(rr)
            pick[m.group(3)] = (k, m.group(2)); fitsc[m.group(3)] = rr["kld"] * calf(m.group(1), b) / loginterp(UD, b)
    if step % 5 and step != len(rows) - 1: continue
    for bd, tl in HT.items():                      # a point only where EVERY target of the band has its pick on ver1
        v = [ver1(t) for t in tl]
        if all(v): val[bd][0].append(step); val[bd][1].append(gm(v))
    v = [ver1(t) for t in TL]
    if all(v):
        val["all"][0].append(step); val["all"][1].append(gm(v))
        fitc[0].append(step); fitc[1].append(gm([fitsc[t] for t in TL]))
for bd, (xs, ys) in val.items():
    if not xs: continue
    if bd == "all":
        ax2.plot(xs, ys, color="k", lw=2.6, marker="o", ms=3, label="VALIDATION: all 9 targets on held-out ver1 (geo-mean)")
    else:
        ax2.plot(xs, ys, color=COL[bd], lw=1.3, alpha=0.75, marker="o", ms=2, label=f"ver1, {BN[bd]} ({', '.join(HT[bd])})")
    print(f"validation {bd if bd == 'all' else BN[bd]}: {ys[0]:.3f} -> min {min(ys):.3f} @ {xs[ys.index(min(ys))]} -> now {ys[-1]:.3f} "
          f"(steps {xs[0]}-{xs[-1]}, {len(xs)} points)")
if fitc[0]:
    ax2.plot(*fitc, color="k", lw=1.2, ls="--", alpha=0.6, label="same picks on the fit slices (training)")
ax2.set_yscale("log"); ax2.set_ylim(0.25, 0.8)
ax2.set_yticks([0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8]); ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax2.set_xlabel("eval step (all evals, in order)"); ax.set_xlabel("")
ax2.set_ylabel("score of the best-so-far picks")
ax2.set_title("Validation loss: the search's current picks (2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7 bpw) scored on held-out ver1; "
              "stop when it stops falling", fontsize=10)
ax2.legend(loc="upper right", fontsize=8); ax2.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(OUT, dpi=110)

for bd in range(NB):
    c = curve[bd]
    if c:
        def at(frac): return c[min(len(c) - 1, int(frac * (len(c) - 1)))][1]
        print(f"{BN[bd]}: best-so-far now {c[-1][1]:.3f} (half-way {at(0.5):.3f}, 3/4 {at(0.75):.3f}); "
              f"{len(best[bd])} targets, {len(pts[bd])} evals")
