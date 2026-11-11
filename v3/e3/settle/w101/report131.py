#!/usr/bin/env python3
"""report131.py SECDIR [--json F] -- w101 #131 verdict (../w101.md): per-section K at equal bytes, held out over the two
block halves. E_s^X(K) = R_s * sum_{b in X, s} e_b(K) / sum_{b in X, s} ref_b; split chosen on X, scored on Y; gain
g_Y = K_eq - B with E_1^Y(K_eq) = E^Y(split) (log-linear in K between grid points); cell gain = mean over (A->B, B->A).
Pass: mean over the 16 cells >= 0.10 bit."""
import glob, itertools, json, math, os, sys

import numpy as np
import torch

D = sys.argv[1]
GRID = [1 + 0.25 * i for i in range(19)]
NS = {"ssm_qkv": {"q": 2048, "k": 2048, "v": 6144}, "attn_qg": {"q": 6144, "gate": 6144}}
TENS = [(3, "attn_qg"), (19, "attn_qg"), (39, "attn_qg"), (63, "attn_qg"), (2, "ssm_qkv"), (10, "ssm_qkv"), (30, "ssm_qkv"), (50, "ssm_qkv")]


def keq(E1, e):
    """K with E1(K) = e, log-linear between grid points; None outside the grid"""
    l = np.log(np.array([E1[k] for k in GRID])); x = math.log(e)
    for i in range(len(GRID) - 1):
        if l[i] >= x >= l[i + 1]: return GRID[i] + 0.25 * (l[i] - x) / (l[i] - l[i + 1])
    return None


def splits(tensor, B):
    secs = list(NS[tensor]); n = NS[tensor]; tot = sum(n.values()) * B
    for ks in itertools.product(GRID, repeat=len(secs) - 1):
        rest = (tot - sum(n[s] * k for s, k in zip(secs, ks))) / n[secs[-1]]
        if any(abs(rest - g) < 1e-9 for g in GRID): yield dict(zip(secs, list(ks) + [round(rest * 4) / 4]))


out = {"cells": [], "missing": []}
for L, T in TENS:
    pf = f"{D}/prep/{L}-{T}.pt"
    if not os.path.exists(pf): out["missing"].append(f"{L}-{T} prep"); continue
    o = torch.load(pf, weights_only=False); ref = o["ref_all"].numpy(); smp = o["sample"]
    from sec101 import sections  # noqa: E402  (same dir)
    sec = sections(T, o["N"]); R = {s: float(ref[bl].sum()) for s, bl in sec.items()}
    e = {}
    for K in GRID:
        f = f"{D}/rows/{L}-{T}-K{K:g}.json"
        if not os.path.exists(f): out["missing"].append(f"{L}-{T} K{K:g}"); continue
        r = json.load(open(f)); e[K] = dict(zip(r["blocks"], r["e2"]))
    if len(e) < len(GRID): continue
    ES = {X: {s: {K: R[s] * sum(e[K][b] for b in smp[s][X]) / sum(ref[b] for b in smp[s][X]) for K in GRID} for s in sec} for X in "AB"}
    E1 = {X: {K: sum(ES[X][s][K] for s in sec) for K in GRID} for X in "AB"}
    nsum = sum(NS[T].values()); rho = {s: R[s] / NS[T][s] for s in sec}
    am = sum(NS[T][s] / nsum * rho[s] for s in sec); gm = math.exp(sum(NS[T][s] / nsum * math.log(rho[s]) for s in sec))
    for B in (2.5, 4.0):
        cand = list(splits(T, B)); cell = {"tensor": f"L{L} {T}", "B": B, "share": {s: R[s] / sum(R.values()) for s in sec},
                                           "ceiling_bit": 0.5 * math.log2(am / gm)}
        gs = []
        for X, Y in (("A", "B"), ("B", "A")):
            best = min(cand, key=lambda sp: sum(ES[X][s][sp[s]] for s in sec))
            eY = sum(ES[Y][s][best[s]] for s in sec); eX = sum(ES[X][s][best[s]] for s in sec)
            kY, kX = keq(E1[Y], eY), keq(E1[X], eX)
            g = None if kY is None else kY - B
            cell[f"{X}->{Y}"] = {"split": best, "gain_bit": g, "insample_bit": None if kX is None else kX - B,
                                 "proxy_ratio": eY / E1[Y][B], "edge": any(best[s] in (GRID[0], GRID[-1]) for s in sec)}
            gs.append(g)
        cell["gain_bit"] = None if None in gs else sum(gs) / 2
        out["cells"].append(cell)
cells = [c for c in out["cells"] if c["gain_bit"] is not None]
for c in out["cells"]:
    ab, ba = c["A->B"], c["B->A"]
    print(f"{c['tensor']:14s} B{c['B']:<4} share {', '.join(f'{s} {v:.2f}' for s, v in c['share'].items()):28s} ceil {c['ceiling_bit']:.3f}  "
          f"A->B {ab['split']} g {ab['gain_bit']:+.3f} (in {ab['insample_bit']:+.3f}, x{ab['proxy_ratio']:.3f}{' EDGE' if ab['edge'] else ''})  "
          f"B->A {ba['split']} g {ba['gain_bit']:+.3f}  cell {c['gain_bit']:+.3f}")
if out["missing"]: print("missing:", out["missing"][:10], "..." if len(out["missing"]) > 10 else "")
if len(cells) == 16:
    m = sum(c["gain_bit"] for c in cells) / 16
    for f in ("attn_qg", "ssm_qkv"):
        for B in (2.5, 4.0):
            xs = [c["gain_bit"] for c in cells if f in c["tensor"] and c["B"] == B]
            print(f"  {f} B{B}: mean gain {sum(xs) / len(xs):+.3f} bit")
    print(f"\nVERDICT #131: mean held-out gain {m:+.3f} bit over 16 cells vs >= 0.10 -> {'PASS' if m >= 0.10 else 'FAIL'}")
    out.update(mean_gain_bit=m, verdict="PASS" if m >= 0.10 else "FAIL")
else: print(f"\nVERDICT #131: incomplete ({len(cells)} / 16 cells)"); out["verdict"] = "incomplete"
if "--json" in sys.argv: json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
