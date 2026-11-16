#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""statusplot.py STATE.jsonl RESULTS.jsonl REFS.jsonl OUT.png -- one readable status figure (user 11-16): each size's
incumbent on the fixed held-out slice t183 (trk183-* reads), as KLD ÷ V2 at the same bpw (V2 log-linear between its
measured sizes), a step line per size over the eval step at which that incumbent took over. Two panels: release sizes
2.0-6.0 and the others (in-between, top, research). Legend: latest t183 KLD and vs IQR-V4 at the same bpw."""
import json, math, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOP = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
COLS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:olive"]


def loglin(pts, b):
    pts = sorted(pts)
    if len(pts) < 2 or not (pts[0][0] - 0.15 <= b <= pts[-1][0] + 0.1): return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= b <= x1: break
    else:
        (x0, y0), (x1, y1) = (pts[0], pts[1]) if b < pts[0][0] else (pts[-2], pts[-1])
    return math.exp(math.log(y0) + (math.log(y1) - math.log(y0)) * (b - x0) / (x1 - x0))


def main(state, results, refs, out):
    st = [json.loads(l) for l in open(state) if l.strip()]
    R = [json.loads(l) for l in open(refs) if l.strip()]
    V2 = [(r["bpw"], r["kld"]) for r in R if r["name"].startswith("v2")]
    IQ = [(r["bpw"], r["kld"]) for r in R if r["name"].startswith("iq")]
    trk = {}
    for l in open(results):
        if '"trk183-' not in l: continue
        r = json.loads(l)
        if r.get("kld"): trk[r["tag"][7:]] = r["kld"]
    step, inc_at = 0, {}
    for e in st:
        if e.get("ev") == "result": step += 1
        elif e.get("ev") == "incumbent": inc_at.setdefault(float(e["size"]), []).append((step, e["tag"]))
    nstep = step
    sizes = sorted(inc_at)
    plt.rcParams.update({"font.size": 15})
    fig, axs = plt.subplots(2, 1, figsize=(16, 13), sharex=True)
    groups = [[s for s in sizes if s in TOP], [s for s in sizes if s not in TOP and s != 1.25]]
    titles = ["Release sizes 2.0–6.0", "In-between (2.75–3.75), top (5.5–7.0) and research (1.5–1.75) sizes"]
    for ax, grp, title in zip(axs, groups, titles):
        for i, s in enumerate(grp):
            ref = loglin(V2, s)
            pts = [(x, trk[t]) for x, t in inc_at[s] if t in trk and ref]
            if not pts: continue
            xs, ys = [], []
            for (x, k) in pts:
                if ys and abs(k / ref - ys[-1]) < 1e-12: continue
                xs.append(x); ys.append(k / ref)
            xs.append(nstep); ys.append(ys[-1])
            iq = loglin(IQ, s); last = pts[-1][1]
            lab = f"{s:g} bpw: {last:.5f}" + (f"  ({(last / iq - 1) * 100:+.0f}% vs IQR-V4)" if iq else "")
            ax.step(xs, ys, where="post", lw=2.4, color=COLS[i % len(COLS)], label=lab)
        ax.axhline(1.0, color="k", lw=1)
        ax.set_title(title, fontsize=17); ax.set_ylabel("t183 KLD ÷ V2 (same bpw)"); ax.grid(alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=14)
    axs[-1].set_xlabel("eval step (search results, in order)")
    fig.suptitle("EXL3-V3 search: each size's incumbent on the held-out slice t183 (lower is better)", fontsize=18)
    fig.subplots_adjust(left=0.07, right=0.66, top=0.92, bottom=0.07, hspace=0.18)
    fig.savefig(out, dpi=90)


if __name__ == "__main__":
    main(*sys.argv[1:5])
