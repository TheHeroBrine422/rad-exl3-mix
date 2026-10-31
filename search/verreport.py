#!/usr/bin/env python3
"""verreport.py PREFIX GOLDEN -- mixes vs the reference frontier on one slice, with 95% intervals from a document
bootstrap (2000 resamples: every reference and mix recomputed on the same resampled documents, the Pareto envelope
re-interpolated in log KLD each time). PREFIX: tag prefix of the slice's results (w- for golden-ver1, a- for calib-A).
Refs: PREFIX + "ref-*"; mixes: PREFIX + "knap-*". Run from /root."""
import json, sys, math
import numpy as np
import os
# paths (README: Paths)
RXM_WORK = os.environ.get("RXM_WORK", "/root")
P, G = sys.argv[1], sys.argv[2]
PARAMS = 26895998464
res = {}
for l in open(f"{RXM_WORK}/evq/results.jsonl"):
    r = json.loads(l)
    if "fail" in r: continue
    if r["tag"].startswith(P): res[r["tag"]] = r
    elif P == "a-" and r["tag"] == "base-sc4.00-e4": res["a-ref-sc4.00-e4"] = r      # calib-A's base

m = json.load(open(f"{RXM_WORK}/{G}/kld.json")); gen = np.fromfile(f"{RXM_WORK}/{G}/gen.u8", np.uint8).astype(bool)
spans = [(d["row0"], d["row0"] + d["tokens"] - 1 - d["score_from"]) for d in m["docs"]]
nd = np.array([gen[a:b].sum() for a, b in spans], float)
def docs(t):
    t = "base-sc4.00-e4" if t == "a-ref-sc4.00-e4" else t
    s = np.fromfile(f"{RXM_WORK}/evq/scores/{t}.f32", np.float32).reshape(-1, 4)[:, 0].astype(np.float64)
    return np.array([s[a:b][gen[a:b]].sum() for a, b in spans])
bpw = lambda t: res[t]["bytes"] * 8 / PARAMS
refs = sorted((t for t in res if t.startswith(P + "ref-")), key=bpw)
mixes = sorted((t for t in res if t.startswith(P + "knap-")), key=bpw)
R = np.array([docs(t) for t in refs]); M = np.array([docs(t) for t in mixes]); rb = np.array([bpw(t) for t in refs])
def curve(k, b):                                     # Pareto envelope of refs with KLDs k, log-interpolated at b
    keep = []; 
    for i in range(len(k)):
        if not keep or k[i] < k[keep[-1]]: keep.append(i)
    return np.exp(np.interp(b, rb[keep], np.log(k[keep])))
rng = np.random.default_rng(0); NB = 2000
ix = rng.integers(0, len(spans), (NB, len(spans)))
kr = R.sum(1) / nd.sum(); km = M.sum(1) / nd.sum()
Rb = np.stack([R[:, i].sum(1) / nd[i].sum() for i in ix]); Mb = np.stack([M[:, i].sum(1) / nd[i].sum() for i in ix])
print(f"{G}: {len(spans)} docs, {int(nd.sum())} positions; refs: " + ", ".join(f"{t[len(P)+4:]} {bpw(t):.3f}:{k:.5f}" for t, k in zip(refs, kr)))
print(f"{'mix':30} {'bpw':>6} {'KLD':>9} {'refs@bpw':>9} {'vs refs':>8} {'95% CI':>18}")
for j, t in enumerate(mixes):
    b = bpw(t); c = curve(kr, b)
    v = np.array([Mb[s, j] / curve(Rb[s], b) - 1 for s in range(NB)])
    print(f"{t[len(P):]:30} {b:6.3f} {km[j]:9.5f} {c:9.5f} {km[j]/c-1:+8.1%} [{np.percentile(v, 2.5):+7.1%},{np.percentile(v, 97.5):+7.1%}]")
