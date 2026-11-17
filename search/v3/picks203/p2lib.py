# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""p2lib.py -- shared readers for the final picks (RULES-picks amendments 2026-11-16 / 17).

Per-document KL sums come from the coordinator's ROOT/docs/<evaltag>-<slice>.json ({"m": [[kl_sum, positions], ...]},
documents in the golden's order). Phase 1 (s195c2-c5): eval tags p1-<tag> (4 slices in one eval) or p1s<N>-<tag>.
Phase 2 (final750, s195c6-c10): p2f-<tag>, p26-<tag> ... p210-<tag>."""
import json, math, os

import numpy as np

ROOT = "/workspace/rad-local/multi/v3b"
P1 = ["s195c2", "s195c3", "s195c4", "s195c5"]
P2 = ["final750", "s195c6", "s195c7", "s195c8", "s195c9", "s195c10"]
P2SHORT = {"final750": "f", **{f"s195c{i}": str(i) for i in (6, 7, 8, 9, 10)}}
PARAMS = 26895998464
REG = "/workspace/rad-local/workers/lead-srch/run/registry.json"
F750 = "/workspace/rad-local/workers/lead-srch/picks203/final750-docs.json"


def _tags(phase, tag, s):
    if phase == 1: return [f"p1-{tag}", f"p1s{s[-1]}-{tag}"]
    return [f"p2{P2SHORT[s]}-{tag}"]


class Reads:
    """per (tag, slice): np.array [[kl_sum, positions], ...] and bpw (container bytes)"""

    def __init__(self, phase, root=ROOT):
        if phase == 2 and os.environ.get("P2TEST"): phase = 1
        self.phase, self.root = phase, root
        self.SL = P1 if phase == 1 else P2
        self.d, self.bpw = {}, {}
        want = ('"p1' if phase == 1 else '"p2')
        for l in open(f"{root}/results.jsonl"):
            if want not in l: continue
            r = json.loads(l)
            if r.get("fail") or r.get("stopped") or r.get("slice") not in self.SL: continue
            t = r["tag"]; base = t.split("-", 1)[1]
            if t not in _tags(phase, base, r["slice"]): continue
            p = f"{root}/docs/{t}-{r['slice']}.json"
            if not os.path.exists(p): continue
            m = json.load(open(p))["m"]
            if any(not x or len(x) < 2 or x[0] is None for x in m): continue
            self.d.setdefault(base, {})[r["slice"]] = np.array([[x[0], x[1]] for x in m], float)
            if r.get("bytes"): self.bpw[base] = r["bytes"] * 8 / PARAMS

    def full(self, tag): return tag in self.d and all(s in self.d[tag] for s in self.SL)

    def pos(self, tag): return np.concatenate([self.d[tag][s][:, 1] for s in self.SL])

    def vec(self, tag, keep=None):
        """the candidate's per-document KL sums over the phase's slices (concatenated, slice order); keep: bool mask"""
        v = np.concatenate([self.d[tag][s][:, 0] for s in self.SL])
        return v if keep is None else v[keep]


def keep_mask(phase, drop_ids=("c:4151",)):
    """phase 1: every document but drop_ids (registry ids); phase 2: all (trim masks come from trim_docs)"""
    if phase == 2: return None
    R = json.load(open(REG)); R = R.get("slices", R)
    return np.concatenate([np.array([d["id"] not in drop_ids for d in R[s]["docs"]]) for s in P1])


def doc_ids(phase):
    if phase == 1 or os.environ.get("P2TEST"):
        R = json.load(open(REG)); R = R.get("slices", R)
        return [d["id"] for s in P1 for d in R[s]["docs"]]
    R = json.load(open(REG)); R = R.get("slices", R)
    f = json.load(open(F750))
    return [f"f:{d['part']}:{d.get('orig')}" for d in f] + [d["id"] for s in P2[1:] for d in R[s]["docs"]]


def verify_mask():
    """phase 2: final750's calib-verify documents (parts c1 = ver4, c3)"""
    if os.environ.get("P2TEST"): return np.array([i % 5 == 0 for i in range(len(doc_ids(1)))])
    f = json.load(open(F750)); R = json.load(open(REG)); R = R.get("slices", R)
    n_rest = sum(len(R[s]["docs"]) for s in P2[1:])
    return np.array([d["part"] in ("c1", "c3") for d in f] + [False] * n_rest)


def boot(a, r, adj=0.0, n=2000, seed=203):
    """paired per-document bootstrap of log(sum a / sum r) + adj -> (est, lo95, hi95) as log ratios"""
    rng = np.random.default_rng(seed)
    est = math.log(a.sum() / r.sum()) + adj
    idx = rng.integers(0, len(a), (n, len(a)))
    bs = np.log(a[idx].sum(1) / r[idx].sum(1)) + adj
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, float(lo), float(hi)


def pct(x): return 100 * (math.exp(x) - 1)


def measured_b(reads, cands, keep=None):
    """frozen measured b per size: slope of ln KLD vs bpw between neighbouring sizes' search incumbents on these reads
    (central difference; one-sided at the ends); sizes whose incumbent lacks a full read borrow the nearest size's b"""
    k = {}
    for s, v in cands.items():
        inc = next((c["tag"] for c in v if "incumbent" in c["why"]), None)
        if inc and reads.full(inc): k[float(s)] = (reads.bpw[inc], math.log(reads.vec(inc, keep).sum()))
    S = sorted(k); b = {}
    for i, s in enumerate(S):
        lo, hi = S[max(0, i - 1)], S[min(len(S) - 1, i + 1)]
        if lo == hi: continue
        b[s] = -(k[hi][1] - k[lo][1]) / (k[hi][0] - k[lo][0])
    for s in map(float, cands):
        if s not in b and b: b[s] = b[min(b, key=lambda x: abs(x - s))]
    return b
