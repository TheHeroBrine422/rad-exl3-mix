#!/usr/bin/env python3
"""cmp206.py DOCSDIR -- IQR-V4 (13 sizes) vs the frozen V3 picks on the 6 phase-2 slices, paired per document (report only).
Each IQR model vs the pick nearest in bpw (board metric: IQR = its model card, V3 = container bytes x 8 / 26,895,998,464), size-adjusted with the pick size's frozen
b: diff = log(KL_iqr / KL_pick) - b * (bpw_pick - bpw_iqr)  (> 0: IQR-V4 worse at equal size). Untrimmed (board-like) and trimmed
(the pick size's phase-2 trim mask). Bootstrap 2,000 by document."""
import sys, os, json, math, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, ".."))
import p2lib as L
D = sys.argv[1]; R = L.Reads(2)
F = json.load(open(os.path.join(H, "../../../../data/picks203/picks-frozen.json")))
C = json.load(open("/workspace/rad-local/workers/lead-srch/picks203/cands-p2.json"))
M = ["iq2r", "iq2r-l", "iq3r-s", "iq3r", "iq3r-l", "iq4r-s", "iq4r", "iq4r-l", "iq5r-s", "iq5r", "iq6r-s", "iq6r", "iq7r"]
picks = {float(s): x for s, x in F.items()}
# board bpw (language model only) from the IQR-V4 model card: the files also carry the vision tower and the DFlash2 drafter
BPW = {"iq2r": 2.470, "iq2r-l": 2.797, "iq3r-s": 3.157, "iq3r": 3.472, "iq3r-l": 3.804, "iq4r-s": 4.142, "iq4r": 4.473, "iq4r-l": 4.793,
       "iq5r-s": 5.071, "iq5r": 5.433, "iq6r-s": 6.063, "iq6r": 6.439, "iq7r": 7.420}
f = lambda e: f"{L.pct(e[0]):+.1f} % [{L.pct(e[1]):+.1f}, {L.pct(e[2]):+.1f}]"
print("| IQR-V4 | bpw | KLD (phase-2, untrimmed) | nearest V3 pick | bpw | KLD | IQR vs V3 at equal size, untrimmed | trimmed |\n|---|---|---|---|---|---|---|---|")
for n in M:
    fs = [f"{D}/{n}-{sl}.json" for sl in L.P2]
    if not all(os.path.exists(p) for p in fs): print(f"| {n} | (incomplete {sum(map(os.path.exists, fs))}/6) |"); continue
    J = [json.load(open(p)) for p in fs]; bi = BPW[n]
    v = np.concatenate([np.array(j["m"], float)[:, 0] for j in J]); pos = np.concatenate([np.array(j["m"], float)[:, 1] for j in J])
    s = min(picks, key=lambda z: abs(z - bi)); x = picks[s]; t = x["pick"]; bp = R.bpw[t]; p = R.vec(t)
    d = -x["b"] * (bp - bi)
    k = f"{s:g}" if f"{s:g}" in C else str(s)
    tags = [c["tag"] for c in C[k] if R.full(c["tag"])]; keep, _, _ = L.trim_mask([R.vec(q) for q in tags], R.pos(tags[0]))
    eu, et = L.boot(v, p, d), L.boot(v[keep], p[keep], d)
    print(f"| {n} | {bi:.3f} | {v.sum() / pos.sum():.5f} | {s:g} {t.split('-', 2)[2]} | {bp:.3f} | {p.sum() / R.pos(t).sum():.5f} | {f(eu)} | {f(et)} |")
