#!/usr/bin/env python3
"""CPU self-test of posgain.py: a synthetic long golden (14 docs x 12k tokens + 2 docs x 40k, sf 0, 70% gen) and score
files with planted seed-paired gains: (A) 9% at every position -> HOLDS; (B) 9% below 2k, 1% beyond -> DECAYS;
(C) 9% below 2k, 5.0% beyond (0.56 of it: below 2/3, above 1/3, precisely measured) -> PARTIAL; (D) 9% / 6.5% but a g08-search1 gain of 11%
(long / search1 0.59, neither above 2/3 nor detectably below) -> INCONCLUSIVE and two extra seeds listed. The
g08-search1 side ("short" files) carries a 9% gain unless stated."""
import json, math, os, subprocess, sys, tempfile
import numpy as np
X = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ok = True
def check(c, msg):
    global ok; ok &= bool(c); print(("PASS " if c else "FAIL ") + msg)
rng = np.random.default_rng(3)

def make(case, seed_sd, g1=0.09):
    T = tempfile.mkdtemp(); G = f"{T}/gl"; L = f"{T}/long"; S = f"{T}/short"
    for d in (G, L, S): os.makedirs(d)
    lens = [12000] * 14 + [40000] * 2; docs, r0 = [], 0
    for n in lens: docs.append({"source": "x", "tokens": n, "score_from": 0, "row0": r0}); r0 += n - 1
    R = r0; json.dump({"rows": R, "docs": docs}, open(f"{G}/kld.json", "w"))
    gen = (rng.random(R) < 0.7).astype(np.uint8); gen.tofile(f"{G}/gen.u8")
    pos = np.concatenate([np.arange(n - 1) for n in lens])
    gshort, glong = {"A": (0.09, 0.09), "B": (0.09, 0.01), "C": (0.09, 0.050), "D": (0.09, 0.065)}[case]
    g = np.where(pos < 2048, gshort, glong)
    docf = np.repeat(rng.lognormal(0, 0.5, len(lens)), [n - 1 for n in lens])
    for K in (3, 2):
        for s in range(3):
            sx = "" if s == 0 else f"_s{s}"
            base = docf * rng.gamma(2.0, 0.05, R) * (1 + pos / 40000)
            arm = base * np.exp(-g + rng.normal(0, seed_sd)) * rng.lognormal(-0.02, 0.2, R)
            hx = base * np.exp(-g / 2 + rng.normal(0, seed_sd) / 2) * rng.lognormal(-0.02, 0.2, R)   # a subset arm: half the gain
            for tag, v in ((f"k{K}base{sx}", base), (f"k{K}edy{sx}", arm), (f"k{K}h-gdn_in{sx}", hx)):
                np.stack([v, np.ones(R), gen, np.zeros(R)], 1).astype(np.float32).tofile(f"{L}/{tag}.glong.f32")
            n2 = 5000; b2 = np.full(n2, 0.1); a2 = b2 * math.exp(-g1 + rng.normal(0, seed_sd))
            for tag, v in ((f"k{K}base{sx}", b2), (f"k{K}edy{sx}", a2)):
                np.stack([v, np.ones(n2), np.ones(n2), np.zeros(n2)], 1).astype(np.float32).tofile(f"{S}/{tag}.f32")
    return G, L, S

for case, want, sd, g1 in (("A", "HOLDS", 0.004, 0.09), ("B", "DECAYS", 0.004, 0.09), ("C", "PARTIAL", 0.004, 0.09),
                           ("D", "INCONCLUSIVE", 0.004, 0.11)):
    G, L, S = make(case, sd, g1)
    r = subprocess.run([sys.executable, f"{X}/posgain.py", G, L, S, "--arms", "edy,h-gdn_in", "--boot", "400", "--minrows", "300"],
                       capture_output=True, text=True)
    if r.returncode: print(r.stderr); check(False, f"case {case} ran"); continue
    out = json.loads(r.stdout.strip().splitlines()[-1]); v = out["verdicts"].get("edy")
    print("\n".join(r.stdout.splitlines()[:7]))
    check(v == want, f"case {case}: verdict {v} (want {want}), by K {out['verdicts'].get('edy_byK')}")
    if case == "A":
        p = out["arms"]["k3edy"]["pooled"]
        check(all(abs(p[b][0] - 0.09) < 0.02 for b in ("0-2k", "2-8k", "8-32k", "32-64k", "long")), "case A: band gains near 0.09")
        fs = out["family_shares"]["k3h-gdn_in"]
        check(abs(fs["0-2k"] - 0.5) < 0.1 and abs(fs["long"] - 0.5) < 0.1, f"case A: subset arm share by band {fs} (planted 0.5)")
    if case == "D":
        nd = subprocess.run([sys.executable, f"{X}/posgain.py", G, L, S, "--arms", "edy", "--boot", "400", "--minrows", "300",
                             "--need-seeds"], capture_output=True, text=True).stdout.split()
        check(nd == ["3", "edy", "3", "3", "edy", "4", "2", "edy", "3", "2", "edy", "4"], f"case D extra seeds: {nd}")
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
