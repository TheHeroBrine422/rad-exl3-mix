#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""trackplot183.py RESULTS.jsonl REFS.jsonl OUT.png -- #183 tracking-slice plot (user 11-14): raw KL of each size's
incumbents on the fixed slice t183 (trk183-* results, in submit order), every value ÷ a reference at the SAME bpw on the
SAME documents: IQR-V4 (top) and V2 (bottom), each interpolated log-linearly in bpw between its measured sizes (REFS:
{"name", "bpw", "kld"} lines, name "iq*" / "v2-*"). No chaining, no anchoring: a point is one measurement."""
import json, math, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def interp(pts, b):
    pts = sorted(pts)
    if len(pts) < 2 or not (pts[0][0] - 0.15 <= b <= pts[-1][0] + 0.15): return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= b <= x1: break
    else:
        (x0, y0), (x1, y1) = (pts[0], pts[1]) if b < pts[0][0] else (pts[-2], pts[-1])
    t = (b - x0) / (x1 - x0); return math.exp(math.log(y0) + t * (math.log(y1) - math.log(y0)))


def main(res, refs, out):
    R = [json.loads(l) for l in open(res) if l.strip() and '"trk183-' in l]
    F = [json.loads(l) for l in open(refs) if l.strip()]
    iq = [(r["bpw"], r["kld"]) for r in F if r["name"].startswith("iq")]; v2 = [(r["bpw"], r["kld"]) for r in F if r["name"].startswith("v2")]
    series = {}
    for i, r in enumerate(R):
        kld = r.get("kld")
        if kld is None: continue
        size = float(r["tag"].split("-")[2]); bpw = r["bpw"]
        series.setdefault(size, []).append((i, kld, bpw))
    fig, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    for k, (refpts, name) in enumerate(((iq, "IQR-V4"), (v2, "V2"))):
        a = ax[k]
        for size in sorted(series):
            xs, ys = [], []
            for i, kld, bpw in series[size]:
                ref = interp(refpts, bpw)
                if ref: xs.append(i); ys.append(kld / ref)
            if xs: a.step(xs + [len(R)], ys + [ys[-1]], where="post", label=f"{size:g} ({ys[-1]:.2f})", marker="o", ms=3)
        a.axhline(1.0, color="k", lw=1); a.set_yscale("log"); a.grid(alpha=.3)
        a.set_ylabel(f"KL ÷ {name} at the same bpw (t183)"); a.legend(fontsize=8, ncol=4, title="size (latest)")
        a.set_title(f"vs {name}: {len(refpts)} reference sizes measured on t183" + ("" if len(refpts) >= 2 else " — waiting for references"))
    ax[1].set_xlabel("tracking eval (each = one incumbent scored in full on t183; never decides)")
    fig.suptitle("EXL3-V3 search #183 — fixed tracking slice t183 (48 calib + 130 xcorp docs, board rows), raw measurements")
    plt.tight_layout(); plt.savefig(out, dpi=90)
    print(f"{len(R)} tracking evals, {len(series)} sizes, refs iq {len(iq)} v2 {len(v2)}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
