#!/usr/bin/env python3
"""report.py BX.jsonl... [--factors aqfactors.json] -- idea 7 tables from blockx.py's output (markdown on stdout).

Per family x depth band (16 layers) and group (64, 128): the A4 error cut from k A8 blocks = 1 - sum rel_out(k) /
sum rel_out(0) over the band's tensors (sums, because the E1c KLD proxy is k_f * sum rel_out), per selector; the
ceiling (every block A8 at that group) for scale. Rule (w42, fixed before data): A4 is "rescued" if k <= 2 cuts the
error >= 30% on gate_up and on the attn / GDN inputs (attn_in, gdn_in) at >= 2 depth bands each (selector amax, the
deployable one named by the idea; err and inmax reported alongside).
"""
import json, sys
from collections import defaultdict

args = [a for a in sys.argv[1:] if not a.startswith("--")]
fac_path = sys.argv[sys.argv.index("--factors") + 1] if "--factors" in sys.argv else None
if fac_path: args.remove(fac_path)
R = [json.loads(l) for p in args for l in open(p) if l.strip()]
FAMS = ["gate_up", "attn_in", "gdn_in", "down", "attn_out", "ssm_out"]
BANDS = [(0, 15), (16, 31), (32, 47), (48, 63)]
SELS = ["amax", "err", "inmax"]
band = lambda l: next(i for i, (a, b) in enumerate(BANDS) if a <= l <= b)

S = defaultdict(float)     # (fam, band, g, sel, k) -> sum rel_out
A8 = defaultdict(float)    # (fam, band, g) -> sum rel_out with every block A8 (same group)
N = defaultdict(set)
B = {}
pert = defaultdict(list)   # (fam, band, g, sel, k) -> per-tensor cuts
base = {}
for o in R:
    if o["k"] == 0: base[(o["i"], o["g"])] = o["rel_out"]
for o in R:
    f, b = o["fam"], band(o["layer"])
    B[f] = o["B"]; N[(f, b)].add(o["i"])
    if o["k"] == 0:
        for s in SELS: S[(f, b, o["g"], s, 0)] += o["rel_out"]
        A8[(f, b, o["g"])] += o["a8_rel_out"]
    else:
        S[(f, b, o["g"], o["sel"], o["k"])] += o["rel_out"]
        pert[(f, b, o["g"], o["sel"], o["k"])].append(1 - o["rel_out"] / base[(o["i"], o["g"])])

cut = lambda f, b, g, s, k: 1 - S[(f, b, g, s, k)] / S[(f, b, g, s, 0)]
ntens = len({o["i"] for o in R})
print(f"{ntens} tensors, {len(R)} lines\n")
for g in (64, 128):
    print(f"### A4-g{g}: error cut by k A8 blocks (selector amax; err / inmax in brackets at k = 2)\n")
    print("| family | B (blocks/row) | band | tensors | A4 Σrel_out | k=1 | k=2 [err / inmax] | k=4 | all-A8 ceiling | k=2 best-tensor / median |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for f in FAMS:
        for b, (lo, hi) in enumerate(BANDS):
            if not N[(f, b)]: continue
            pt = sorted(pert[(f, b, g, "amax", 2)])
            med = pt[len(pt) // 2]
            print(f"| {f} | {B[f]} | L{lo}–{hi} | {len(N[(f, b)])} | {S[(f, b, g, 'amax', 0)]:.3g} | "
                  f"{cut(f, b, g, 'amax', 1):.1%} | {cut(f, b, g, 'amax', 2):.1%} [{cut(f, b, g, 'err', 2):.1%} / "
                  f"{cut(f, b, g, 'inmax', 2):.1%}] | {cut(f, b, g, 'amax', 4):.1%} | "
                  f"{1 - A8[(f, b, g)] / S[(f, b, g, 'amax', 0)]:.1%} | {pt[-1]:.1%} / {med:.1%} |")
    print()

print("### Rule: k <= 2 cuts >= 30% at >= 2 depth bands, on gate_up and on attn / GDN inputs\n")
print("| family | group | selector | bands passing (k=2) | max cut over bands (k=2) | verdict |")
print("|---|---|---|---|---|---|")
verdict = {}
for g in (64, 128):
    for s in SELS:
        for f in ("gate_up", "attn_in", "gdn_in"):
            c = [cut(f, b, g, s, 2) for b in range(4) if N[(f, b)]]
            ok = sum(x >= 0.30 for x in c) >= 2
            verdict[(g, s, f)] = ok
            print(f"| {f} | g{g} | {s} | {sum(x >= 0.30 for x in c)} of {len(c)} | {max(c):.1%} | {'pass' if ok else 'fail'} |")
print()
for g in (64, 128):
    for s in SELS:
        ok = verdict[(g, s, "gate_up")] and (verdict[(g, s, "attn_in")] or verdict[(g, s, "gdn_in")])
        okall = verdict[(g, s, "gate_up")] and verdict[(g, s, "attn_in")] and verdict[(g, s, "gdn_in")]
        print(f"- g{g} {s}: gate_up {'pass' if verdict[(g, s, 'gate_up')] else 'fail'}, attn_in "
              f"{'pass' if verdict[(g, s, 'attn_in')] else 'fail'}, gdn_in {'pass' if verdict[(g, s, 'gdn_in')] else 'fail'}"
              f" -> {'RESCUES' if ok else 'does not rescue'} A4" + (" (all three)" if okall else ""))
print()

# whole-family view (all 64 layers) + the E1c KLD proxy
fac = json.load(open(fac_path))["factors"] if fac_path else {}
print("### Whole family (all layers): cut and E1c-proxy ΔKLD of A4 on every tensor of the family\n")
print("| family | group | A4 ΔKLD (proxy) | k=1 | k=2 | k=4 | all-A8 ΔKLD (proxy) | k/B at k=2 |")
print("|---|---|---|---|---|---|---|---|")
for f in FAMS:
    for g in (64, 128):
        tot = {k: sum(S[(f, b, g, "amax", k)] for b in range(4)) for k in (0, 1, 2, 4)}
        a8 = sum(A8[(f, b, g)] for b in range(4))
        kf = fac.get(f, float("nan"))
        print(f"| {f} | g{g} | {kf * tot[0]:.2e} | {kf * tot[1]:.2e} ({1 - tot[1] / tot[0]:.0%}) | "
              f"{kf * tot[2]:.2e} ({1 - tot[2] / tot[0]:.0%}) | {kf * tot[4]:.2e} ({1 - tot[4] / tot[0]:.0%}) | "
              f"{kf * a8:.2e} | {2 / B[f]:.1%} |")
