#!/usr/bin/env python3
"""headgrid.py SPEC... -- the LM head's budget grid (1/16 bit) from headsim results, for knap5's KN_HEADGRID (w88).

SPEC = SIM.json or SIM.json:PLAN.json (headsim.py output; PLAN.json = the headslab.py plan file it ran, optional: it
gives each plan's exact donor set). Every measured plan of every file counts (uniform u<K>, slab s<B>-f<floor> of
every floor, headslab2's s<B>-<S> / b<B>): the floor is chosen per budget by the measurement, as headsim does.
  - plan KL: headsim's mean head-only KL; ci=True (default, as KN_HEADSLAB): the conservative end of the 95% interval,
    KL(ref) + the upper bound of the paired difference vs its uniform ref (uniform plans: their measured KL).
  - donors: the plan's K_blocks (PLAN.json) or, without it, the file's donors (its u<K> plans) at or above the plan's
    floor (all of them when the name has no floor). donors=set(...) keeps only plans built from those donors, i.e.
    plans whose containers can be built today.
  - guard=x: drops a non-uniform plan whose rare-token KL is above x times the uniform head's at the same bits
    (uniform guard log-linear in K between the measured u<K>).
  - lower convex hull of (bits, KL) over the kept plans; a grid budget B (multiples of `step` from the lowest to the
    highest hull bits) gets the hull's KL at B: linear between the two neighbouring vertices, i.e. a block mix of two
    measured plans (KL ~ additive over row blocks), an upper bound on the best plan at B if the curve is convex.
    When the 17-donor curves arrive at 1/16 steps, the vertices are the measured plans themselves.
Returns {"grid": [(B, kl, desc)], "uniform": {K: kl}, "vertices": [(bits, kl, name)], "kept", "dropped"}.
CLI: prints the grid with the change per step and vs the uniform head of the same bits."""
import json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = ["lcurve.json", "fcurve-A.json", "fcurve-B.json", "headslab2-sim.json:headslab2-plan.json"]   # v3/yaqa/data
DATA = os.environ.get("KN_HG_DIR") or next((d for d in (f"{HERE}/../../yaqa/data", f"{HERE}/../yaqa/data", f"{HERE}/../../../v3/yaqa/data")
                                            if os.path.isdir(d)), f"{HERE}/../../yaqa/data")

def load(spec, ci=True):
    sim, _, planf = spec.partition(":")
    S = json.load(open(sim))["plans"]
    P = json.load(open(os.path.join(os.path.dirname(sim), planf) if planf and not os.path.isabs(planf) else planf))["plans"] if planf else None
    fd = {float(n[1:]) for n in S if re.fullmatch(r"u[\d.]+", n)}          # the file's donors
    out = []
    for n, r in S.items():
        if "nofloor" in n: continue                                         # no tail floor: 3-12x worse (#34)
        uni = re.fullmatch(r"u([\d.]+)", n)
        if P and n in P: don = set(map(float, P[n]["K_blocks"]))
        elif uni: don = {float(uni.group(1))}
        else:
            m = re.search(r"-f([\d.]+)$", n); fl = float(m.group(1)) if m else -1
            don = {k for k in fd if k >= fl}
        kl = r["kl"]
        if ci and not uni and "ci95" in r: kl = (r["kl"] - r["diff"]) + r["ci95"][1]
        out.append(dict(name=n, src=os.path.basename(sim), bits=r["avg_bits"], kl=kl, kl_mean=r["kl"],
                        guard=r.get("guard_kl"), uniform=bool(uni), donors=don))
    return out

def hull(pts):
    """lower convex hull of [(x, y, tag)], x ascending; drops points that do not lower y"""
    h = []
    for p in sorted(pts):
        if h and p[1] >= h[-1][1]: continue
        while len(h) >= 2 and (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) >= (p[1] - h[-2][1]) * (h[-1][0] - h[-2][0]): h.pop()
        h.append(p)
    return h

def grid(specs=None, step=1 / 16, ci=True, guard=None, donors=None):
    specs = specs or [os.path.join(DATA, s) for s in DEFAULT]
    plans = [p for s in specs for p in load(s, ci)]
    uni = {}
    for p in plans:
        if p["uniform"]: uni.setdefault(p["bits"], []).append(p)
    U = {k: min(v, key=lambda p: p["kl"]) for k, v in uni.items()}
    def ug(b):                                                 # uniform guard KL at b bits (log-linear in K)
        ks = sorted(k for k in U if U[k]["guard"]); b = min(max(b, ks[0]), ks[-1])
        lo = max(k for k in ks if k <= b); hi = min(k for k in ks if k >= b)
        if lo == hi: return U[lo]["guard"]
        f = (b - lo) / (hi - lo); return math.exp((1 - f) * math.log(U[lo]["guard"]) + f * math.log(U[hi]["guard"]))
    kept, dropped = [], []
    for p in plans:
        why = None
        if donors is not None and not p["donors"] <= donors: why = "donors"
        elif guard and not p["uniform"] and p["guard"] and p["guard"] > guard * ug(p["bits"]): why = "guard"
        (dropped if why else kept).append((p, why))
    kept = [p for p, _ in kept]
    H = hull([(p["bits"], p["kl"], f"{p['src']}:{p['name']}") for p in kept])
    g = []
    j0, j1 = math.ceil(H[0][0] / step - 1e-9), math.floor(H[-1][0] / step + 1e-9)
    for j in range(j0, j1 + 1):
        B = j * step
        v = next((v for v in H if abs(v[0] - B) < 1e-9), None)
        if v: g.append((B, v[1], v[2])); continue
        a, b = next((a, b) for a, b in zip(H, H[1:]) if a[0] <= B <= b[0])
        w = (B - a[0]) / (b[0] - a[0])
        g.append((B, a[1] + w * (b[1] - a[1]), f"mix {1 - w:.2f} {a[2]} + {w:.2f} {b[2]}"))
    return {"grid": g, "uniform": {k: v["kl"] for k, v in sorted(U.items())}, "vertices": H, "kept": len(kept),
            "dropped": [(p["src"], p["name"], why) for p, why in dropped]}

if __name__ == "__main__":
    specs = sys.argv[1:] or None
    G = grid(specs, step=float(os.environ.get("KN_HG_STEP", 1 / 16)), ci=os.environ.get("KN_HG_CI", "1") == "1",
             guard=float(os.environ["KN_HG_GUARD"]) if os.environ.get("KN_HG_GUARD") else None,
             donors={float(x) for x in os.environ["KN_HG_DONORS"].split(",")} if os.environ.get("KN_HG_DONORS") else None)
    U = G["uniform"]; ks = sorted(U)
    def uk(b):
        b = min(max(b, ks[0]), ks[-1]); lo = max(k for k in ks if k <= b); hi = min(k for k in ks if k >= b)
        return U[lo] if lo == hi else math.exp(math.log(U[lo]) + (b - lo) / (hi - lo) * (math.log(U[hi]) - math.log(U[lo])))
    print(f"{G['kept']} plans kept, {len(G['dropped'])} dropped; {len(G['vertices'])} hull vertices; uniform {U}")
    prev = None
    for B, kl, d in G["grid"]:
        print(f"{B:7.4f}  KL {kl:.3e}  step {'' if prev is None else f'{kl / prev - 1:+6.1%}'}  vs uniform(log-interp) {kl / uk(B) - 1:+6.1%}  {d}")
        prev = kl
