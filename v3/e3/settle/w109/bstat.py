# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""bstat.py TODAY PROP [--sub] -- #144 (b) judge (rule v3/e3/settle/w109.md 2b): per source and layer the held-out gain
G = heldout_base - heldout_joint of TODAY's and PROP's swjfit OUT dirs, the seed spread = mean over layers of
|G(fs0) - G(fs1)| / mean G (today's), the loss = mean over layers of (G_today - G_prop) / G_today per source; PASS if
every source's loss <= the spread. --sub: PROP = TODAY's own heldout_joint_sub (the fit on 1/K of the fit half).
Also the scales' movement: share of channels whose c differs, mean / max |dc|."""
import json, os, sys

import numpy as np
import torch
from safetensors import safe_open

T, P = sys.argv[1:3]; SUB = "--sub" in sys.argv
names = sorted(n for n in os.listdir(T) if os.path.isdir(f"{T}/{n}"))
rd = lambda d, n: [json.loads(l) for l in open(f"{d}/{n}/swiglu.jsonl")]
G = {n: np.array([r["heldout_base"] - r["heldout_joint"] for r in rd(T, n)]) for n in names}
spread = float(np.mean(np.abs(G[names[0]] - G[names[1]])) / np.mean([G[n].mean() for n in names]))
print(f"seed spread (|G fs0 - G fs1| / mean G, mean over {len(G[names[0]])} layers): {spread:.4f}")
ok = True
for n in names:
    rp = rd(T, n) if SUB else rd(P, n)
    Gp = np.array([r["heldout_base"] - r[("heldout_joint_sub" if SUB else "heldout_joint")] for r in rp])
    loss = float(np.mean((G[n] - Gp) / G[n])); ok &= loss <= spread
    line = f"{n}: G today {G[n].mean():.4e}, prop {Gp.mean():.4e}, loss {loss:+.4f} ({'<=' if loss <= spread else '>'} spread)"
    if not SUB:
        cg = cc = tot = 0; md = []
        for f in sorted(os.listdir(f"{T}/{n}/scales")):
            with safe_open(f"{T}/{n}/scales/{f}", "pt") as a, safe_open(f"{P}/{n}/scales/{f}", "pt") as b:
                x, y = a.get_tensor("c"), b.get_tensor("c")
            d = (x - y).abs(); cc += int((d > 0).sum()); tot += d.numel(); md.append(float(d.max()))
            if "gate_proj" in f: cg += int((d > 0).sum())
        line += f"; c moved on {cc / tot:.3%} of channels (gate {2 * cg / tot:.3%}), max |dc| {max(md):.3g}"
    print(line)
print("PASS" if ok else "FAIL")
