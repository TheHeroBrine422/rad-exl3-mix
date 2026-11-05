#!/usr/bin/env python3
"""r97hull.py [embmenu.jsonl embmenu2.jsonl] -- #97 verdict (sub1/README.md "#97"): which new embedding levels lie on
the lower convex hull of reconstruction error rel vs bits per value (bpv), together with the existing menu (trellis
t1..t6, one scale per row = G5120, embmenu.jsonl) and Lloyd e1..e8 (run7, same 4096 rows).
A level measured twice (the t1 / t6 anchors of run7) counts with its lower error (both are valid encodings).
Also prints each level's error ratio to the level a quarter below and the hull slope (error saved per bit)."""
import json, os, sys

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
old_p, new_p = (sys.argv[1:3] if len(sys.argv) > 2 else (f"{D}/embmenu.jsonl", f"{D}/embmenu2.jsonl"))
old = [json.loads(l) for l in open(old_p)]; new = [json.loads(l) for l in open(new_p)]
NEW = {0.25, 0.5, 0.75, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0}
pts = {}                                  # name -> (bpv, rel, source)
for src, rows in (("embmenu", old), ("run7", new)):
    for r in rows:
        if r["method"] == "trellis" and r.get("group") != 5120: continue
        name = f"t{r['bits']:g}" if r["method"] == "trellis" else f"e{r['bits']}"
        if name not in pts or r["rel"] < pts[name][1]: pts[name] = (r["bpv"], r["rel"], src)
anch = {n: [r["rel"] for src, rows in (("embmenu", old), ("run7", new)) for r in rows
            if r["method"] == "trellis" and r.get("group") == 5120 and f"t{r['bits']:g}" == n] for n in ("t1", "t6")}
print("anchors (embmenu, run7): " + "; ".join(f"{n} " + " / ".join(f"{v:.6g}" for v in a) +
      (f"  run7/old {a[-1] / a[0]:.4f}" if len(a) > 1 else "") for n, a in anch.items()))


def lower_hull(P):
    P = sorted(P); H = []
    for p in P:
        while len(H) >= 2 and (H[-1][0] - H[-2][0]) * (p[1] - H[-2][1]) - (H[-1][1] - H[-2][1]) * (p[0] - H[-2][0]) <= 0:
            H.pop()
        H.append(p)
    m = min(range(len(H)), key=lambda i: H[i][1])
    return H[:m + 1]                      # past the lowest error more bits only cost: dominated


P = [(b, r, n) for n, (b, r, _) in pts.items()]
hull = lower_hull(P); on = {p[2] for p in hull}
order = sorted(pts, key=lambda n: pts[n][0])
print(f"\n{'level':6s} {'bpv':>8s} {'rel':>12s} {'x prev t':>9s} {'hull':>5s}  source")
prev = None
for n in order:
    b, r, src = pts[n]
    ratio = ""
    if n.startswith("t"):
        if prev: ratio = f"{pts[prev][1] / r:9.3f}"
        prev = n
    print(f"{n:6s} {b:8.4f} {r:12.6g} {ratio:>9s} {'yes' if n in on else '-':>5s}  {src}{'  NEW' if n.startswith('t') and float(n[1:]) in NEW else ''}")
print("\nhull slopes (Δrel per bit, between hull neighbours): " +
      ", ".join(f"{a[2]}→{b[2]} {(a[1] - b[1]) / (b[0] - a[0]):.3g}" for a, b in zip(hull, hull[1:])))
newl = [n for n in order if n.startswith("t") and float(n[1:]) in NEW]
print("\nRule (a level joins the menu if it is on the hull): " +
      ", ".join(f"{n} {'JOINS' if n in on else 'dominated'}" for n in newl))
print("(the lowest-bit and lowest-error points are hull vertices by construction: for t0.25 and t8 the slope is the test)")
