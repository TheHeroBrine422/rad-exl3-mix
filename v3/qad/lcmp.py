#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""lcmp.py RUN_A RUN_B NAME [--b 1.48] [--lora-run RUN] -- arm L's judgement (#165 amendment 2): two runs' torch-path
scores (qad.py --score: RUN/tscore-NAME.npz, same documents), ratio B / A of the pooled KL with its 95% document
bootstrap interval (2,000 resamples, seed 0), documents better / worse; with --lora-run the adapter bytes (bf16 and
int8 + one f16 scale per adapter row), their extra bpw over the board's 26,895,998,464 parameters, and the
size-adjusted ratio ratio x exp(b x extra bpw) (log KL + b bpw, b = the frontier's local slope)."""
import argparse, json, math, os

import numpy as np

P = 26895998464


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("name")
    ap.add_argument("--b", dest="slope", type=float, default=1.48); ap.add_argument("--b2", type=float, default=1.6)
    ap.add_argument("--lora-run", default=None)
    a = ap.parse_args()
    A = np.load(os.path.join(a.a, f"tscore-{a.name}.npz")); B = np.load(os.path.join(a.b, f"tscore-{a.name}.npz"))
    assert (A["rows"] == B["rows"]).all()
    xa, xb, n = A["sums"], B["sums"], A["rows"]
    rng = np.random.default_rng(0); bs = []
    for _ in range(2000):
        i = rng.integers(0, len(n), len(n)); bs.append(xb[i].sum() / xa[i].sum())
    lo, hi = np.percentile(bs, [2.5, 97.5]); ratio = xb.sum() / xa.sum()
    rep = {"a": a.a, "b": a.b, "slice": a.name, "kl_a": float(xa.sum() / n.sum()), "kl_b": float(xb.sum() / n.sum()),
           "ratio": float(ratio), "ci95": [float(lo), float(hi)], "docs_better": int(((xb < xa) & (n > 0)).sum()),
           "docs_worse": int(((xb > xa) & (n > 0)).sum()), "docs": int((n > 0).sum()), "positions": int(n.sum())}
    if a.lora_run:
        from safetensors import safe_open
        with safe_open(os.path.join(a.lora_run, "lora.safetensors"), "pt") as f:
            sh = [f.get_slice(k).get_shape() for k in f.keys()]
        vals = sum(x[0] * x[1] for x in sh); rows = sum(x[0] for x in sh)
        for tag, nbytes in (("bf16", 2 * vals), ("int8", vals + 2 * rows)):
            bpw = nbytes * 8 / P
            rep[f"size_{tag}"] = {"bytes": int(nbytes), "extra_bpw": bpw,
                                  "adj_ratio": float(ratio * math.exp(a.slope * bpw)),
                                  "adj_ci95": [float(lo * math.exp(a.slope * bpw)), float(hi * math.exp(a.slope * bpw))],
                                  f"adj_ratio_b{a.b2:g}": float(ratio * math.exp(a.b2 * bpw))}
        rep["adapter_values"] = int(vals)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
