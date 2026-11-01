#!/usr/bin/env python3
"""seedreport.py SEED.jsonl -- best-of-N rotation seeds. Per (tensor, K): each seed's log error relative to the seed mean,
on row sample A and on row sample B. A real (tensor-level) seed effect shows the same ranking on both samples: we
report the correlation of the seed effects A vs B over all (tensor, K), and the honest best-of-4 gain = pick the best
seed on A, measure it on B (vs the mean seed on B). Noise alone gives correlation ~0 and gain ~0."""
import collections, json, math, statistics as st, sys
R = collections.defaultdict(dict)
for l in open(sys.argv[1]):
    d = json.loads(l); R[(d["layer"], d["w"], d["bits"])][(d["seed"], d["rowseed"])] = math.log(d["rel"])
xa, xb, gains, spread = [], [], collections.defaultdict(list), []
for key, v in R.items():
    seeds = sorted({s for s, _ in v})
    if not all((s, r) in v for s in seeds for r in (0, 1)) or len(seeds) < 2: continue
    ma = st.mean(v[(s, 0)] for s in seeds); mb = st.mean(v[(s, 1)] for s in seeds)
    for s in seeds: xa.append(v[(s, 0)] - ma); xb.append(v[(s, 1)] - mb)
    best = min(seeds, key=lambda s: v[(s, 0)])
    gains[key[2]].append(math.exp(v[(best, 1)] - mb) - 1)
    spread.append(max(v[(s, 0)] for s in seeds) - min(v[(s, 0)] for s in seeds))
if not xa: sys.exit("no complete (tensor, K) yet")
n = len(xa); mxa, mxb = st.mean(xa), st.mean(xb)
cov = sum((a - mxa) * (b - mxb) for a, b in zip(xa, xb)) / n
corr = cov / (st.pstdev(xa) * st.pstdev(xb))
print(f"{len(spread)} (tensor, K) complete; seed spread on one sample (max-min error): median {math.exp(st.median(spread)) - 1:.2%}")
print(f"seed effect, sample A vs sample B: correlation {corr:+.2f}  (0 = noise, 1 = a real per-tensor seed effect)")
for K in sorted(gains):
    g = gains[K]; print(f"  K{K}: best-of-{len(seeds)} on A, measured on B: {st.mean(g):+.2%} error (median {st.median(g):+.2%}, n={len(g)})")
