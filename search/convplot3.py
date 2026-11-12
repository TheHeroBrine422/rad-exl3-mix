#!/usr/bin/env python3
"""convplot3.py RESULTS.jsonl OUT.png [--board TSV] [--events EVENTS.jsonl] [--v2 V2RESULTS.jsonl] [--ref-re RE]
[--heldout-re RE] -- the V3 search's live convergence plot (w120, 11-12; lead notes 2-9: V2's convplot idea, own
script, scale 1.0 = V2).

RESULTS: rxrun.multi's results.jsonl (one line per eval x slice: tag, bytes, bpw, positions, kld, p99, top1, slice;
failed / misconfigured evals carry "fail" or "flag": counted, never plotted). Order of lines = eval step.
Scale: y = KLD / V2's KLD at the same bpw on the same slice. Every slice has V2 reference containers scored on it
(tags matching --ref-re, default ^ref-V2-(\\d+\\.\\d+)$, e.g. ref-V2-3.00): V2 on that slice is log-linear in log(KLD)
vs bpw between the two neighbouring references, the end segment's slope outside them (lead note 8). A slice with one
reference borrows the board's V2 curve shape (search/lb/board-*.tsv, the EXL3-V2-* rows) scaled through it; a slice
with none is skipped (counted). 1.0 = V2, lower is better.
Top: per bpw band (< 2.75, 2.75-4.5, 4.5-6, >= 6) every eval (faint dots) and the best so far (line); grey vertical
lines = EVENTS ({"step": N | "t": ISO, "label": ...}: refits, model / method / slice changes, lead note 4); dotted
horizontal line = IQR-V4 at ~0.88 x V2, which holds over 2.5-5 bpw only (lead note 9).
Right: V2's own search (--v2 = data/v2/results.jsonl, through search/convplot.py's own computation) per band, each
rescaled to end at 1.0: how much V2's search moved its bands -- an upper bound (it mixes fixes into the curve; note 4).
Bottom: held-out evals (slices matching --heldout-re, default verify) of the picks, per band, and their best so far."""
import argparse, json, math, os, re, runpy, sys, tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PARAMS = 26895998464
HERE = os.path.dirname(os.path.abspath(__file__))
BN = ["< 2.75 bpw", "2.75–4.5 bpw", "4.5–6 bpw", "≥ 6 bpw"]; COL = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
band = lambda b: 0 if b < 2.75 else 1 if b < 4.5 else 2 if b < 6 else 3


def loglin(pts, b):
    """log-linear in KLD between the neighbouring points, end segments extended; (value, extrapolated?)"""
    pts = sorted(pts)
    if len(pts) == 1: return pts[0][1], b != pts[0][0]
    i = 0 if b <= pts[0][0] else len(pts) - 2 if b >= pts[-1][0] else max(j for j in range(len(pts) - 1) if pts[j][0] <= b)
    (x0, y0), (x1, y1) = pts[i], pts[i + 1]
    return math.exp(math.log(y0) + (b - x0) / (x1 - x0) * (math.log(y1) - math.log(y0))), not (pts[0][0] <= b <= pts[-1][0])


def board_v2(tsv):
    out = []
    for l in open(tsv):
        f = l.rstrip("\n").split("\t")
        if f[0].startswith("EXL3-V2-"): out.append((float(f[4]), float(f[5])))
    return sorted(out)


def v2_history(v2res):
    """search/convplot.py's per-band best-so-far curves of V2's run (its own scale), via runpy (convplot is a script)"""
    tmp = tempfile.mkdtemp(prefix="cp3-"); argv = sys.argv
    try:
        sys.argv = ["convplot.py", v2res, f"{tmp}/v2.png"]
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            g = runpy.run_path(os.path.join(HERE, "convplot.py"))
        return g["curve"], len(g["rows"])
    finally:
        sys.argv = argv
        for f in os.listdir(tmp): os.remove(f"{tmp}/{f}")
        os.rmdir(tmp)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="convplot3.py")
    ap.add_argument("results"); ap.add_argument("out")
    ap.add_argument("--board", default=os.path.join(HERE, "lb", "board-2026-11-12.tsv"))
    ap.add_argument("--events"); ap.add_argument("--v2", default=os.path.join(HERE, "..", "data", "v2", "results.jsonl"))
    ap.add_argument("--ref-re", default=r"^ref-V2-(\d+\.\d+)$"); ap.add_argument("--heldout-re", default=r"verify")
    ap.add_argument("--iqr", type=float, default=0.88)
    a = ap.parse_args(argv)
    BV2 = board_v2(a.board)
    rows = [json.loads(l) for l in open(a.results) if l.strip()]
    bad = [r for r in rows if r.get("fail") or r.get("flag")]
    good = [r for r in rows if not (r.get("fail") or r.get("flag")) and r.get("kld") and r.get("bytes")]
    refre = re.compile(a.ref_re); held_re = re.compile(a.heldout_re)
    refs = {}
    for r in good:
        if refre.match(r["tag"]): refs.setdefault(r["slice"], {})[r["tag"]] = (r["bytes"] * 8 / PARAMS, r["kld"])
    def v2at(sl, b):
        R = list(refs.get(sl, {}).values())
        if len(R) >= 2: return loglin(R, b)
        if len(R) == 1:
            (b0, k0), = R; s = k0 / loglin(BV2, b0)[0]; v, _ = loglin(BV2, b); return v * s, True
        return None, True
    pts = [[] for _ in range(4)]; ext = [[] for _ in range(4)]; held = [[] for _ in range(4)]; best = [None] * 4; curve = [[] for _ in range(4)]
    hbest = [None] * 4; hcurve = [[] for _ in range(4)]; noref = 0; steps = {}
    for step, r in enumerate(rows):
        steps.setdefault(r.get("t"), step)
        if r not in good or refre.match(r["tag"]): continue
        b = r["bytes"] * 8 / PARAMS; v2, x = v2at(r["slice"], b)
        if v2 is None: noref += 1; continue
        y = r["kld"] / v2; bd = band(b)
        if held_re.search(r["slice"]):
            held[bd].append((step, y)); hbest[bd] = min(hbest[bd] or 9, y); hcurve[bd].append((step, hbest[bd])); continue
        (ext if x else pts)[bd].append((step, y))
        best[bd] = min(best[bd] or 9, y); curve[bd].append((step, best[bd]))
    ev = []
    if a.events and os.path.exists(a.events):
        ts = [(r.get("t"), i) for i, r in enumerate(rows) if r.get("t")]
        for l in open(a.events):
            e = json.loads(l)
            if "step" in e: ev.append((e["step"], e["label"]))
            elif "t" in e: ev.append((next((i for t, i in ts if t >= e["t"]), len(rows)), e["label"]))
    fig = plt.figure(figsize=(15, 10)); gs = fig.add_gridspec(2, 2, width_ratios=[3, 1.2], height_ratios=[3, 2])
    ax = fig.add_subplot(gs[0, 0]); axv = fig.add_subplot(gs[0, 1]); ax2 = fig.add_subplot(gs[1, 0], sharex=ax); axn = fig.add_subplot(gs[1, 1]); axn.axis("off")
    for bd in range(4):
        if pts[bd]: ax.scatter(*zip(*pts[bd]), s=8, alpha=0.25, color=COL[bd])
        if ext[bd]: ax.scatter(*zip(*ext[bd]), s=10, alpha=0.35, color=COL[bd], marker="^")
        if curve[bd]: ax.step(*zip(*curve[bd]), where="post", color=COL[bd], lw=2.2, label=f"{BN[bd]}: best so far ({curve[bd][-1][1]:.3f})")
    for s, lab in ev:
        for x_ in (ax, ax2): x_.axvline(s, color="gray", ls="--", lw=1)
        ax.text(s, 1.25, " " + lab, fontsize=8, color="gray", rotation=90, va="top")
    ax.axhline(1.0, color="k", lw=0.9, label="V2 (1.0)")
    ax.axhline(a.iqr, color="k", ls=":", lw=1.6, label=f"IQR-V4 (~{a.iqr:.2f} × V2, 2.5–5 bpw)")
    for x_ in (ax, ax2):
        x_.set_yscale("log"); x_.grid(alpha=0.3, which="both"); x_.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        x_.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    top = max([1.3] + [min(2.0, y * 1.05) for c in curve for _, y in c[-1:]])
    ax.set_ylim(0.5, top); ax.set_yticks([t for t in (0.5, 0.6, 0.7, 0.8, 0.88, 1.0, 1.2, 1.5, 2.0) if t <= top])
    ax.set_ylabel("KLD ÷ V2 at the same bpw, same slice"); ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"EXL3-V3 search convergence (dots: evals; ▲: bpw outside the slice's V2 references; lower is better)\n"
                 f"{len(good)} evals, {len(bad)} failed / flagged (not plotted), {noref} without V2 refs on their slice", fontsize=10)
    for bd in range(4):
        if held[bd]: ax2.scatter(*zip(*held[bd]), s=22, facecolors="none", edgecolors=COL[bd])
        if hcurve[bd]: ax2.step(*zip(*hcurve[bd]), where="post", color=COL[bd], lw=1.6, label=f"held-out {BN[bd]} ({hcurve[bd][-1][1]:.3f})")
    ax2.axhline(1.0, color="k", lw=0.9); ax2.axhline(a.iqr, color="k", ls=":", lw=1.2); ax2.set_ylim(0.5, 1.3)
    ax2.set_yticks([0.5, 0.6, 0.7, 0.8, 0.88, 1.0, 1.2]); ax2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_xlabel("eval step (results.jsonl order)"); ax2.set_ylabel("held-out (verify) evals ÷ V2")
    ax2.set_title("Validation: the picks on held-out slices (best so far per band)", fontsize=10)
    if any(hcurve): ax2.legend(loc="upper right", fontsize=8)
    note = ["V2's own search (right): each band's best-so-far", "curve rescaled to end at 1.0. Upper bound on what", "a search adds: V2's run mixed in a K1/K2 fit fix,",
            "streaming / local search switched on partway,", "early slice / repricing changes (lead note 4)."]
    if a.v2 and os.path.exists(a.v2):
        try:
            cv, n2 = v2_history(a.v2)
            for bd in range(4):
                c = cv[bd]
                if not c: continue
                xs = [s / n2 for s, _ in c]; ys = [v / c[-1][1] for _, v in c]
                axv.plot(xs, ys, color=COL[bd], lw=1.6, label=f"{BN[bd]}: start {ys[0]:.2f}")
            axv.axhline(1.0, color="k", lw=0.9); axv.set_yscale("log"); axv.set_ylim(0.8, 2.5); axv.grid(alpha=0.3, which="both")
            axv.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter()); axv.set_yticks([0.8, 1.0, 1.25, 1.5, 2.0, 2.5])
            axv.set_xlabel(f"fraction of V2's {n2} evals"); axv.set_title("V2's search, rescaled to its end", fontsize=10); axv.legend(fontsize=7)
        except Exception as e:                    # noqa: BLE001
            note.append(f"(V2 history failed: {e!r:.80})")
    axn.text(0, 1, "\n".join(note), va="top", fontsize=8)
    fig.tight_layout(); fig.savefig(a.out, dpi=105)
    for bd in range(4):
        if curve[bd]: print(f"{BN[bd]}: best so far {curve[bd][-1][1]:.3f} ({len(pts[bd]) + len(ext[bd])} evals)")
    print(f"{len(good)} evals, {len(bad)} failed / flagged, {noref} without refs, {len(ev)} events")
    return 0


if __name__ == "__main__":
    sys.exit(main())
