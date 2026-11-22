#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""gradnoise217.py RUNDIR [N=40] -- the readout of qad200.py --gradnoise: RUNDIR/gradnoise.npz holds, per trainable scale vector, the mean and std (ddof 1) of
its gradient over N batches (the first N plan steps, the unhealed start state). Prints, over all scales (and over the input-side `r` / output-side `c`
multipliers separately): the median of |mean|, the median of the across-batch std, their ratio, the median |mean| a pure-noise gradient would show
(0.6745 std / sqrt(N): the batch mean's own noise), and the share of scales whose |mean| is under 1.96 std / sqrt(N) (not distinguishable from 0). JSON to RUNDIR/gradnoise-summary.json."""
import json, sys
import numpy as np

run = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
z = np.load(f"{run}/gradnoise.npz")
names = sorted({k.rsplit("|", 1)[0] for k in z.files})
def stats(sel):
    m = np.concatenate([z[n + "|mean"] for n in sel]).astype(np.float64); s = np.concatenate([z[n + "|std"] for n in sel]).astype(np.float64)
    am = np.abs(m); floor = 0.6745 * s / np.sqrt(N)
    return {"scales": int(len(m)), "median_abs_mean": float(np.median(am)), "median_std": float(np.median(s)), "ratio": float(np.median(am) / np.median(s)),
            "median_abs_mean_if_pure_noise": float(np.median(floor)), "share_abs_mean_within_1.96_se_of_0": float((am < 1.96 * s / np.sqrt(N)).mean()),
            "p10_std": float(np.percentile(s, 10)), "p90_std": float(np.percentile(s, 90))}
out = {"batches": N, "all": stats(names), "r (input side)": stats([n for n in names if n.endswith(":r")]), "c (output side)": stats([n for n in names if n.endswith(":c")])}
json.dump(out, open(f"{run}/gradnoise-summary.json", "w"), indent=1); print(json.dumps(out, indent=1))
