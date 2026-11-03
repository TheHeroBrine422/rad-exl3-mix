#!/usr/bin/env python3
"""a9cmp.py RAWDIR MODEL BDIR [--xw DIR] [--dy DIR] [--out OUT.jsonl] -- A9's CPU gate: the analytic gate / up diagonals
(swiglu_diag.py raw-snap<n>) against the measured Sketch B diagonal BDIR (collect.py YAQA_DIAG hdiag-snap<n>, or a
yaqaB-derived dv<n>/hout), per layer and projection.

Variants: a9x = ‖W_d[:, r]‖² Σ a² ‖x‖², a9d = ‖W_d[:, r]‖² Σ a²; with --dy (collect.py YAQA_DIAG_EXTRA's Σ_t dy_t²
dir, which has down_proj's too) also a9g = (Σ_i W_d[i, r]² dy_down[i]) Σ a² ‖x‖² (the isotropy assumption replaced by
the measured diagonal of the residual gradient: isolates what assuming E[δδᵀ] ∝ I costs). Measured references beside
them: --xw (Σ_t ‖x‖² dy², per token) and --dy. Reported: cosine of the raw vectors (the ranked.md rule), and the
Pearson correlation of their logs (shape over the small entries). BDIR's halves.json gives the reference's own
split-half cosine (its noise floor)."""
import argparse, json, os
import numpy as np, torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("raw"); ap.add_argument("model"); ap.add_argument("bdir")
ap.add_argument("--xw"); ap.add_argument("--dy"); ap.add_argument("--out")
a = ap.parse_args()
def vec(d, key):
    p = f"{d}/{key}.safetensors"
    if not os.path.exists(p): return None
    with safe_open(p, "pt") as f:
        return (f.get_tensor("hout_diag") if "hout_diag" in f.keys() else f.get_tensor("hout").diagonal()).double()
def ld(p):
    with safe_open(p, "pt") as f: return {k: f.get_tensor(k).double() for k in f.keys()}
cos = lambda u, v: float((u * v).sum() / (u.norm() * v.norm()))
lcor = lambda u, v: float(np.corrcoef(u.clamp_min(1e-30).log().numpy(), v.clamp_min(1e-30).log().numpy())[0, 1])
hv = json.load(open(f"{a.bdir}/halves.json"))["cos"] if os.path.exists(f"{a.bdir}/halves.json") else {}
idx = json.load(open(f"{a.model}/model.safetensors.index.json"))["weight_map"]
rows = []
nl = len([f for f in os.listdir(a.raw) if f.startswith("L")])
for l in range(nl):
    r = ld(f"{a.raw}/L{l}.safetensors"); p = f"model.language_model.layers.{l}.mlp"
    dyd = vec(a.dy, f"{p}.down_proj") if a.dy else None
    if dyd is not None:
        k = f"{p}.down_proj.weight"
        with safe_open(f"{a.model}/{idx[k]}", "pt") as f: Wd = f.get_tensor(k).double()      # [hidden, inter]
        gw = (Wd.pow(2) * dyd[:, None]).sum(0)
    for proj, s in (("gate_proj", "g"), ("up_proj", "u")):
        key = f"{p}.{proj}"; B = vec(a.bdir, key)
        if B is None: continue
        v = {"a9x": r["wd2"] * r[s + "x"], "a9d": r["wd2"] * r[s + "d"]}
        if dyd is not None: v["a9g"] = gw * r[s + "x"]
        for nm, d in (("xw", a.xw), ("dy", a.dy)):
            m = vec(d, key) if d else None
            if m is not None: v[nm] = m
        row = {"layer": l, "proj": proj, "halves_B": hv.get(key)}
        for nm, x in v.items(): row[f"cos_{nm}"] = cos(x, B); row[f"lcor_{nm}"] = lcor(x, B)
        if "xw" in v: row["cos_a9x_vs_xw"] = cos(v["a9x"], v["xw"])
        if "dy" in v: row["cos_a9d_vs_dy"] = cos(v["a9d"], v["dy"])
        rows.append(row)
if a.out:
    with open(a.out, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
cols = [c for c in rows[0] if c.startswith(("cos_", "lcor_"))] + (["halves_B"] if rows[0]["halves_B"] is not None else [])
print(f"raw {a.raw}  vs  B {a.bdir}")
print("per layer (gate | up):\n  L  " + "  ".join(f"{c:>12s}" for c in cols if c.startswith("cos")))
for l in range(nl):
    g = [r for r in rows if r["layer"] == l]
    print(f"  {l:2d} " + "  ".join(f"{g[0][c]:.3f}|{g[1][c]:.3f}" for c in cols if c.startswith("cos")))
print("summary over layers, median / min:")
for proj in ("gate_proj", "up_proj"):
    g = [r for r in rows if r["proj"] == proj]
    print(f"  {proj:9s} " + "  ".join(f"{c} {np.median([r[c] for r in g]):.3f}/{min(r[c] for r in g):.3f}" for c in cols))
