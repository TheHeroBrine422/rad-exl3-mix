#!/usr/bin/env python3
"""r100report.py R100.jsonl... -- #100 numbers and verdict (README "#100" + its amendment, rules fixed before data).

ratio = rel(iqrh) / rel(mul1_i8) per (tensor, K): LDLQ stage, same rows / seed / scale grid (r100.sh, trellis4.py).
Per K: the parameter-weighted mean ratio (weight N x K of the whole tensor), the plain mean and its SE over tensors,
the worst tensor. Equal-quality bpw cost of the weighted ratio: log2(ratio) / s(K), s(K) = mul1_i8's own local
slope of log2 rel per bit at K (parameter-weighted mean over the tensors of the central difference between the K
grid's neighbours), i.e. the bits a tensor would need to buy back the error at that K.
Rule: ADOPT if the weighted ratio <= 1.005 at every K in 1..6 and no tensor > 1.02 there. REJECT if > 1.05 at any K
in 2..5. Otherwise: no automatic call -- the user decides from the trade-off table (this one) and the measured
decode / prefill speed (GEMV at M 1 / 4 / 16, prefill decode pass, on a card).
Follow-up codebooks iqrh:A:B (another global A, B) found in the files get a side table (no verdict).
Lines are deduplicated per (layer, w, method, bits), last wins (a resumed chunk reruns deterministically)."""
import json, math, sys
from collections import defaultdict

R = {}
for p in sys.argv[1:]:
    for l in open(p):
        try: o = json.loads(l)
        except ValueError: continue
        R[(o["layer"], o["w"], o["method"], float(o["bits"]))] = o
T = sorted({(k[0], k[1]) for k in R}, key=lambda t: (t[1], t[0]))
KS = sorted({k[3] for k in R})
A, B = "ldlq_mul1_i8", "ldlq_iqrh"
size = {t: next(o["N"] * o["K"] for k, o in R.items() if k[:2] == t) for t in T}


def rel(t, m, K):
    o = R.get((t[0], t[1], m, K)); return o["rel"] if o else None


ratio = {(t, K): rel(t, B, K) / rel(t, A, K) for t in T for K in KS if rel(t, A, K) and rel(t, B, K)}

# mul1_i8's local slope (log2 rel per bit) at each K: central difference on the K grid, weighted over tensors
slope = {}
for i, K in enumerate(KS):
    lo, hi = KS[max(i - 1, 0)], KS[min(i + 1, len(KS) - 1)]
    s = [(size[t], (math.log2(rel(t, A, lo)) - math.log2(rel(t, A, hi))) / (hi - lo)) for t in T
         if rel(t, A, lo) and rel(t, A, hi)]
    if s: slope[K] = sum(w * v for w, v in s) / sum(w for w, _ in s)

print(f"#100 iqrh / mul1_i8 proxy error (LDLQ), {len(T)} tensors, {len(ratio)} pairs of {len(T) * len(KS)}\n")
print("| K | n | weighted ratio | mean ± SE | min | max (tensor) | mul1_i8 slope (log2/bit) | equal-quality bpw cost |")
print("|---|---|---|---|---|---|---|---|")
W = {}
for K in KS:
    r = [(size[t], ratio[(t, K)], t) for t in T if (t, K) in ratio]
    if not r: continue
    w = sum(a * b for a, b, _ in r) / sum(a for a, _, _ in r); W[K] = (w, len(r), max(r, key=lambda x: x[1]))
    m = sum(b for _, b, _ in r) / len(r)
    se = (sum((b - m) ** 2 for _, b, _ in r) / max(len(r) - 1, 1) / len(r)) ** 0.5
    mx = max(r, key=lambda x: x[1]); mn = min(b for _, b, _ in r)
    dbpw = math.log2(w) / slope[K] if K in slope and slope[K] > 0 else float("nan")
    print(f"| {K:g} | {len(r)} | {w:.4f} | {m:.4f} ± {se:.4f} | {mn:.4f} | {mx[1]:.4f} ({mx[2][0]}:{mx[2][1]}) | "
          f"{slope.get(K, float('nan')):.2f} | {dbpw:+.4f} |")

print("\nper tensor (ratio):\n")
print("| tensor | " + " | ".join(f"K{K:g}" for K in KS) + " |")
print("|---" * (len(KS) + 1) + "|")
for t in T:
    print(f"| {t[0]}:{t[1]} | " + " | ".join(f"{ratio[(t, K)]:.3f}" if (t, K) in ratio else "-" for K in KS) + " |")

sp = {(t, K): (R[(t[0], t[1], A, K)]["scale_f"], R[(t[0], t[1], B, K)]["scale_f"]) for (t, K) in ratio}
print("\nscale factor picks (mul1_i8 -> iqrh) where they differ: " + (", ".join(
    f"{t[0]}:{t[1]} K{K:g} {a:g}->{b:g}" for (t, K), (a, b) in sorted(sp.items(), key=lambda kv: (kv[0][1], kv[0][0])) if a != b) or "none"))
edge = sorted({f for v in sp.values() for f in v} & {0.5, 1.5})
if edge: print(f"NOTE: some picks sit on the grid's edge {edge}")

for X in sorted({k[2] for k in R if k[2].startswith("ldlq_iqrh:")}):   # follow-ups (other global A, B): no verdict
    print(f"\n{X[5:]} (side check, no rule): parameter-weighted ratio vs mul1_i8 / vs iqrh, tensors\n")
    print("| K | n | vs mul1_i8 | vs iqrh | max vs mul1_i8 (tensor) |\n|---|---|---|---|---|")
    for K in KS:
        r = [(size[t], rel(t, X, K) / rel(t, A, K), rel(t, X, K) / rel(t, B, K), t) for t in T
             if rel(t, X, K) and rel(t, A, K) and rel(t, B, K)]
        if not r: continue
        sw = sum(x[0] for x in r); mx = max(r, key=lambda x: x[1])
        print(f"| {K:g} | {len(r)} | {sum(x[0] * x[1] for x in r) / sw:.4f} | {sum(x[0] * x[2] for x in r) / sw:.4f} | "
              f"{mx[1]:.4f} ({mx[3][0]}:{mx[3][1]}) |")

core = [K for K in KS if 1 <= K <= 6]; mid = [K for K in KS if 2 <= K <= 5]
complete = len(ratio) == len(T) * len(KS) and len(T) == 20
adopt = all(K in W and W[K][0] <= 1.005 for K in core) and \
        all(ratio[(t, K)] <= 1.02 for t in T for K in core if (t, K) in ratio)
reject = any(K in W and W[K][0] > 1.05 for K in mid)
why = [f"K{K:g} {W[K][0]:.4f}" for K in core if K in W and W[K][0] > 1.005]
bad = [f"{t[0]}:{t[1]} K{K:g} {ratio[(t, K)]:.3f}" for t in T for K in core if (t, K) in ratio and ratio[(t, K)] > 1.02]
v = "ADOPT" if adopt else "REJECT" if reject else "NO AUTOMATIC CALL (user decides with the measured speed)"
print(f"\nverdict{'' if complete else ' (PARTIAL DATA)'}: {v}")
if not adopt: print(f"  weighted ratio > 1.005 at: {why or 'none'}; tensors > 1.02 (K1-6): {bad or 'none'}")
