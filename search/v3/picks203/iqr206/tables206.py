#!/usr/bin/env python3
"""tables206.py -- V3 picks vs IQR-V4 on the phase-2 corpus (2.8M rows, untrimmed mean KLD): (1) every model, (2) KLD interpolated at equal
size, (3) bpw needed at equal quality. Interpolation: log KLD linear in bpw between neighbouring models of the same family (no extrapolation)."""
import sys, os, json, math, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, ".."))
import p2lib as L
D = os.path.join(H, "../../../../data/iqr206/docs"); R = L.Reads(2)
F = json.load(open(os.path.join(H, "../../../../data/picks203/picks-frozen.json")))
BPW = {"iq2r": 2.470, "iq2r-l": 2.797, "iq3r-s": 3.157, "iq3r": 3.472, "iq3r-l": 3.804, "iq4r-s": 4.142, "iq4r": 4.473, "iq4r-l": 4.793,
       "iq5r-s": 5.071, "iq5r": 5.433, "iq6r-s": 6.063, "iq6r": 6.439, "iq7r": 7.420}
v3 = []
for s, x in F.items():
    t = x["pick"]; v3.append((R.bpw[t], R.vec(t).sum() / R.pos(t).sum(), f"V3 {float(s):g}"))
iq = []
for n, b in BPW.items():
    J = [json.load(open(f"{D}/{n}-{sl}.json")) for sl in L.P2]
    m = np.concatenate([np.array(j["m"], float) for j in J]); iq.append((b, m[:, 0].sum() / m[:, 1].sum(), f"IQR-V4 {n}"))
v3.sort(); iq.sort()
def kld_at(fam, b):
    for (b0, k0, _), (b1, k1, _) in zip(fam, fam[1:]):
        if b0 <= b <= b1: return math.exp(math.log(k0) + (math.log(k1) - math.log(k0)) * (b - b0) / (b1 - b0))
def bpw_at(fam, k):
    for (b0, k0, _), (b1, k1, _) in zip(fam, fam[1:]):
        if k1 <= k <= k0: return b0 + (b1 - b0) * (math.log(k) - math.log(k0)) / (math.log(k1) - math.log(k0))
print("## 1. Every model (phase-2 corpus, untrimmed mean KLD)\n\n| model | bpw | KLD |\n|---|---|---|")
for b, k, n in sorted(v3 + iq): print(f"| {n} | {b:.3f} | {k:.5f} |")
print("\n## 2. KLD at equal size (interpolated; ratio = IQR-V4 / V3)\n\n| bpw | V3 KLD | IQR-V4 KLD | IQR-V4 / V3 | V3 lower by |\n|---|---|---|---|---|")
for b in (2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0):
    a, c = kld_at(v3, b), kld_at(iq, b)
    if a and c: print(f"| {b:g} | {a:.5f} | {c:.5f} | {c / a:.2f}× | {100 * (1 - a / c):.0f} % |")
print("\n## 3. Size at equal quality (interpolated; compression = IQR-V4 bpw / V3 bpw)\n\n| target KLD | V3 bpw | IQR-V4 bpw | bits saved / weight | compression |\n|---|---|---|---|---|")
for k in (0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002, 0.0015, 0.001, 0.0007):
    a, c = bpw_at(v3, k), bpw_at(iq, k)
    if a and c: print(f"| {k:g} | {a:.2f} | {c:.2f} | {c - a:.2f} | {c / a:.3f}× ({100 * (1 - a / c):.0f} % smaller) |")
