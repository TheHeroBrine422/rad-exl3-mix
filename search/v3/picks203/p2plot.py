# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""p2plot.py CANDS-P2.json OUT.png -- phase-2 progress chart (RULES-picks am. 11-17), in p1plot's style.

Top 4 x 4: one panel per size, one row per phase-2 candidate, vs the size's current pick (dashed 0 line):
  filled dot + bar  phase 2, trimmed (the 5 heaviest documents by median KL share left out; the decision statistic), 95 % CI
  x                 phase 2, untrimmed (all documents)
  hollow diamond    phase 1 (s195c2-c5, c:4151 left out), vs the same pick
  hollow dot (k/6)  partial phase-2 read: untrimmed, on the slices done so far
Panel title: pick, trimmed documents' share of the pick's KL, whether the trim changed the pick.
Bottom row: (a) phase 1 vs phase 2 (trimmed), every candidate with a full read, % vs its size's pick, one colour a size;
(b) per size: the trimmed documents' share of KL and the untrimmed vs trimmed spread (how much the trim moves the
ranking). Size adjustment: the frozen measured b (CANDS-P2["_b"])."""
import json, math, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2lib as L

cp, outp = sys.argv[1:3]
C = json.load(open(cp)); B = {float(k): v for k, v in C.get("_b", {}).items()}
R2 = L.Reads(2); R1 = L.Reads(1); K1 = L.keep_mask(1)
COL = {"incumbent": "#d62728", "pool": "#ff7f0e", "confirmed": "#1f77b4", "narrow-miss": "#9467bd", "round-confirmed": "#17becf",
       "round-near-miss": "#8c564b", "extra-solve": "#7f7f7f", "late-confirmed": "#bcbd22"}
def colour(why): return next((COL[w] for w in COL if w in why), "#555555")
sizes = sorted((k for k in C if not k.startswith("_")), key=float)
fig = plt.figure(figsize=(16, 3.2 * 4 + 4.2))
gs = fig.add_gridspec(5, 4, height_ratios=[1, 1, 1, 1, 1.25], hspace=0.45, wspace=0.55, top=0.955, bottom=0.04)
scatter, tstat = [], []
nfull = npart = 0
for k, s in enumerate(sizes):
    ax = fig.add_subplot(gs[k // 4, k % 4]); S = float(s); b = B.get(S, 0.0)
    cand = C[s]; tags = [c["tag"] for c in cand]
    full = [t for t in tags if R2.full(t)]; part = [t for t in tags if t in R2.d and not R2.full(t)]
    nfull += len(full); npart += len(part)
    adj = lambda R, t: b * ((R.bpw.get(t) or S) - S)
    keep = None; pick = None; changed = False; share = 0.0
    if full:
        V = {t: R2.vec(t) for t in full}
        med = np.median(np.stack([V[t] / V[t].sum() for t in full]), axis=0)
        trim = np.argsort(-med)[:5]; keep = np.ones(len(med), bool); keep[trim] = False
        tr = {t: math.log(V[t][keep].sum()) + adj(R2, t) for t in full}
        un = {t: math.log(V[t].sum()) + adj(R2, t) for t in full}
        pick = min(full, key=tr.get); changed = pick != min(full, key=un.get)
        share = 1 - V[pick][keep].sum() / V[pick].sum()
        ranks_t = sorted(full, key=tr.get); ranks_u = sorted(full, key=un.get)
        spread = np.mean([abs(ranks_t.index(t) - ranks_u.index(t)) for t in full])
        tstat.append((S, 100 * share, spread, changed))
    elif part:                                             # no full read yet: reference = best partial read so far
        pick = min(part, key=lambda t: (-len(R2.d[t]), math.log(sum(R2.d[t][x][:, 0].sum() for x in R2.d[t])) + adj(R2, t)))
    rows = []
    for c in cand:
        t = c["tag"]
        if t not in R2.d or pick is None: continue
        if R2.full(t) and R2.full(pick):
            d = adj(R2, t) - adj(R2, pick)
            et = (0, 0, 0) if t == pick else L.boot(V[t][keep], V[pick][keep], d)
            eu = 0 if t == pick else math.log(V[t].sum() / V[pick].sum()) + d
            e1 = None
            if R1.full(t) and R1.full(pick):
                e1 = math.log(R1.vec(t, K1).sum() / R1.vec(pick, K1).sum()) + adj(R1, t) - adj(R1, pick)
                scatter.append((S, L.pct(e1), L.pct(et[0])))
            rows.append((et[0], et, eu, e1, t, c["why"], 6))
        else:
            sl = [x for x in L.P2 if x in R2.d[t] and x in R2.d.get(pick, {})]
            if not sl: continue
            a = np.concatenate([R2.d[t][x][:, 0] for x in sl]); r = np.concatenate([R2.d[pick][x][:, 0] for x in sl])
            e = (0, 0, 0) if t == pick else L.boot(a, r, adj(R2, t) - adj(R2, pick))
            rows.append((e[0] + 1, e, None, None, t, c["why"], len(R2.d[t])))
    rows.sort(key=lambda r: r[0])
    for i, (_, et, eu, e1, t, why, ns) in enumerate(rows):
        y = len(rows) - 1 - i; col = colour(why); fullr = ns == 6
        ax.plot([L.pct(et[1]), L.pct(et[2])], [y, y], color=col, lw=1.2, alpha=0.7)
        ax.plot(L.pct(et[0]), y, "o", ms=5, mfc=col if fullr else "white", mec=col, mew=1.3)
        if eu is not None: ax.plot(L.pct(eu), y, "x", ms=5, color="k", mew=1.0)
        if e1 is not None: ax.plot(L.pct(e1), y, "D", ms=5, mfc="none", mec="#2ca02c", mew=1.2)
        ax.text(1.0, y, t.split("-", 2)[-1] + ("" if fullr else f" ({ns}/6)"), transform=ax.get_yaxis_transform(), fontsize=6.5, va="center", ha="left")
    ax.axvline(0, color="k", lw=0.8, ls="--"); ax.set_yticks([]); ax.set_ylim(-0.7, max(len(rows), 1) - 0.3)
    ttl = f"{S:g} bpw — {len(full)}/{len(cand)} full"
    if full: ttl += f", trim {100 * share:.1f} % KL" + (", TRIM CHANGED PICK" if changed else "")
    ax.set_title(ttl, fontsize=8.5); ax.tick_params(labelsize=7); ax.grid(axis="x", alpha=0.3)
    if k // 4 == 3: ax.set_xlabel("size-adjusted KLD vs the pick (%), 95 % CI", fontsize=7.5)
ax = fig.add_subplot(gs[4, 0:2])
if scatter:
    cm = plt.get_cmap("viridis"); sz = sorted({x[0] for x in scatter})
    for S in sz:
        p = [(a, b_) for s_, a, b_ in scatter if s_ == S]
        ax.scatter([x[0] for x in p], [x[1] for x in p], s=14, color=cm(sz.index(S) / max(1, len(sz) - 1)), label=f"{S:g}")
    lim = [min(min(x[1] for x in scatter), min(x[2] for x in scatter)) - 0.3, max(max(x[1] for x in scatter), max(x[2] for x in scatter)) + 0.3]
    ax.plot(lim, lim, "k--", lw=0.8); ax.set_xlim(lim); ax.set_ylim(lim)
    r = np.corrcoef([x[1] for x in scatter], [x[2] for x in scatter])[0, 1] if len(scatter) > 2 else float("nan")
    ax.set_title(f"(a) phase 1 vs phase 2 (trimmed), % vs each size's pick — r = {r:.2f}, n = {len(scatter)}", fontsize=9)
    ax.legend(fontsize=6.5, ncol=2, loc="upper left", title="bpw", title_fontsize=7)
ax.set_xlabel("phase 1 (s195c2–c5, c:4151 out)", fontsize=8); ax.set_ylabel("phase 2 (trimmed)", fontsize=8); ax.grid(alpha=0.3)
ax = fig.add_subplot(gs[4, 2:4])
if tstat:
    xs = np.arange(len(tstat)); ax.bar(xs - 0.2, [t[1] for t in tstat], 0.4, color="#1f77b4", label="trimmed docs' share of the pick's KL (%)")
    ax2 = ax.twinx(); ax2.bar(xs + 0.2, [t[2] for t in tstat], 0.4, color="#ff7f0e", label="mean rank shift, trimmed vs untrimmed")
    for i, t in enumerate(tstat):
        if t[3]: ax.text(i, t[1] + 0.3, "pick\nchanged", ha="center", fontsize=6, color="#d62728")
    ax.set_xticks(xs); ax.set_xticklabels([f"{t[0]:g}" for t in tstat], fontsize=7)
    ax.set_ylabel("KL share (%)", fontsize=8); ax2.set_ylabel("rank shift", fontsize=8)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    ax.set_title("(b) what the trim does: 5 heaviest documents per size", fontsize=9)
h = [plt.Line2D([], [], marker="o", ls="", color=v, label=k) for k, v in COL.items()]
h += [plt.Line2D([], [], marker="x", ls="", color="k", label="untrimmed"), plt.Line2D([], [], marker="D", ls="", mfc="none", mec="#2ca02c", label="phase 1"),
      plt.Line2D([], [], marker="o", ls="", mfc="white", mec="gray", label="partial (slices/6)")]
fig.legend(handles=h, loc="upper center", ncol=11, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.995))
tot = sum(len(C[s]) for s in sizes)
fig.suptitle(f"Phase 2 (final750 + s195c6–c10, ~2.8M rows, 5 heaviest docs/size trimmed): {nfull} complete + {npart} partial of {tot} — lower is better",
             fontsize=11, y=1.012)
fig.savefig(outp + ".part.png", dpi=110, bbox_inches="tight"); os.replace(outp + ".part.png", outp)
print(outp, nfull, npart, tot)
