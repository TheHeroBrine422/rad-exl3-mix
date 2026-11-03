#!/usr/bin/env python3
"""aggsim.py RESULTS.jsonl [--se 0.016,0.020] [--stages ...] -- racing estimate from the V2 stream's slice-level KLDs
alone (no per-row data: those are on iceman, race.py replays them). Pairs as race.py --results: every stream candidate
on calA / fit2 / fit3 against the best mix measured before it at the same target on the same slice. The true full-slice
log ratio L of each pair is known; the stage estimate after a token fraction f is simulated as L + N(0, se^2 (1-f)/f)
(the deviation of a random prefix from the full slice, documents iid), and its predictive P(win) is
Phi(-l_f / (se sqrt((1-f)/f))). se = the paired document-bootstrap se of log(B/A) on a slice (eval/capstudy.py on V2:
1.6% at 8k positions, 1.7-1.9% at 2k). Heavy-tailed documents (the long traces) are what this misses: race.py checks."""
import argparse, json, math, re
import numpy as np
from math import erf

ap = argparse.ArgumentParser()
ap.add_argument("results"); ap.add_argument("--se", default="0.016,0.02")
ap.add_argument("--stages", default="0.1,0.2,0.35,0.5,0.7"); ap.add_argument("--alphas", default="0.001,0.003,0.01,0.02,0.05")
ap.add_argument("--t0", default="80,15"); ap.add_argument("--reps", type=int, default=200)
ap.add_argument("--scale", default="1,2", help="slice size x today's (se / sqrt(scale), token time x scale)")
a = ap.parse_args()
KR = re.compile(r"^(a|f2|f3)-knap-(?:r\d+|s\d+)(?:oo)?(?:-(?:ts|ls|tl|rd)\d+)?-sc4\.00-e4-([\d.]+)$")
best, L, wall = {}, [], []
for l in open(a.results):
    r = json.loads(l); k = KR.match(r.get("tag", ""))
    if not k or "kld" not in r: continue
    key = (k.group(1), k.group(2))
    if key in best: L.append(math.log(r["kld"] / best[key]))
    w = re.search(r"wall=(\d+)s", r.get("run", "")); wall.append(int(w.group(1)) if w else np.nan)
    if key not in best or r["kld"] < best[key]: best[key] = r["kld"]
L = np.array(L); acc = L < 0
print("time = mean racing eval time / today's full eval at the same T0 (< 1: faster)")
print(f"{len(L)} stream candidates with an incumbent; {acc.sum()} beat it ({100 * acc.mean():.1f}%); "
      f"median |log ratio| {np.median(np.abs(L)):.3f}; share within 2%: {(np.abs(L) < 0.02).mean():.2f}; "
      f"median wall {np.nanmedian(wall):.0f} s")
print("log ratio quantiles (5/25/50/75/95%):", np.round(np.quantile(L, [.05, .25, .5, .75, .95]), 3).tolist())
st = np.array([float(x) for x in a.stages.split(",")]); rng = np.random.default_rng(0)
Phi = np.vectorize(lambda z: 0.5 * (1 + erf(z / math.sqrt(2))))
W = np.nanmedian(wall)
for se0, sc in ((float(x), float(y)) for x in a.se.split(",") for y in a.scale.split(",")):
    se = se0 / math.sqrt(sc)
    # nested prefixes: simulate the prefix means as a random walk in the document order (independent increments)
    sd = se * np.sqrt((1 - st) / st)
    ests = []
    for _ in range(a.reps):
        z = rng.standard_normal((len(L), len(st)))
        # Brownian-bridge construction: sum over a prefix of fraction f deviates from f * total by se_tot*sqrt(f(1-f))
        cum = np.cumsum(np.c_[z[:, :1] * np.sqrt(st[0]), z[:, 1:] * np.sqrt(np.diff(st))], 1)     # W(f)
        bridge = cum - st * (cum[:, -1:] + rng.standard_normal((len(L), 1)) * np.sqrt(1 - st[-1]))  # W(f) - f W(1)
        ests.append(L[:, None] + se * bridge / st)                                                 # prefix mean dev.
    E = np.stack(ests)                                                                             # (reps, n, K)
    PW = Phi(-E / sd)
    print(f"\nse {se0} (slice x{sc:g}: se {se:.4f}):  alpha  rej%  wrong/acc%  " + "  ".join(f"time@T0={t} (vs today)" for t in a.t0.split(",")))
    for al in (float(x) for x in a.alphas.split(",")):
        rej = PW < al; first = np.where(rej.any(2), rej.argmax(2), -1)
        used = np.where(first >= 0, st[np.maximum(first, 0)], 1.0)
        wrong = (first >= 0) & acc[None]
        out = []
        for t0 in (float(x) for x in a.t0.split(",")):
            var = W - 80.0 if not math.isnan(W) else 100.0          # token-proportional part of a V2 eval
            out.append((t0 + used * var * sc).mean() / (t0 + var))     # mean eval time / today's full eval
        print(f"        {al:6.3f} {100 * (first >= 0).mean():5.1f} {100 * wrong.sum() / (acc.sum() * a.reps):9.2f}   " +
              "  ".join(f"{o:20.3f}" for o in out))
