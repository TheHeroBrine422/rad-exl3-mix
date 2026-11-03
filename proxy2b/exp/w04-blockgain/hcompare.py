#!/usr/bin/env python3
"""hcompare.py REF.npz EST.npz... -- how a diagonal estimate (a smaller snapshot, another estimator, another data set)
would do as the output factor, scored in REF's h metric (REF: the 8K Sketch B diagonal, measured quants in it for
the loss weights L): per estimate, the L-weighted predicted two-sided gain designed on EST and evaluated in REF
(blk.block_costs(d_est, d_ref)), next to REF's own and to EST's self-prediction (designed and scored on EST: the
optimism a noisy or different h shows when it predicts its own gain)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blk
R = np.load(sys.argv[1]); keys = sorted({k.split("/")[0] for k in R.files if "/" in k})
def L(k):
    t = [x for x in R.files if x.startswith(k + "/e/") and "base" in x]
    return np.mean([(R[f"{k}/h"] * R[x]).sum() for x in t]) / float(R[f"{k}/trH"]) if t else 1.0
Lw = {k: L(k) for k in keys}
def d(Z, k): h = Z[f"{k}/h"]; return np.maximum(h, 1e-30 * h.max()) * R[f"{k}/sv2"]
def agg(f): return sum(Lw[k] * f(k) for k in keys) / sum(Lw.values())
ref = agg(lambda k: (lambda c: c[0].sum() / c[1].sum())(blk.block_costs(d(R, k))))
print(f"{os.path.basename(sys.argv[1])}: own predicted gain {1 - ref:.1%}")
for f in sys.argv[2:]:
    E = np.load(f)
    cross = agg(lambda k: (lambda c: c[0].sum() / c[1].sum())(blk.block_costs(d(E, k), d(R, k))))
    self_ = agg(lambda k: (lambda c: c[0].sum() / c[1].sum())(blk.block_costs(d(E, k))))
    cos = np.median([float((R[f"{k}/h"] * E[f"{k}/h"]).sum() / np.linalg.norm(R[f"{k}/h"]) / np.linalg.norm(E[f"{k}/h"])) for k in keys])
    print(f"{os.path.basename(f):24s} gain in REF's metric {1 - cross:6.1%} ({(1 - cross) / (1 - ref):5.1%} of REF's), "
          f"self-predicted {1 - self_:6.1%}, median cos(h, h_ref) {cos:.4f}")
