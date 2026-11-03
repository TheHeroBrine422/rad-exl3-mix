#!/usr/bin/env python3
"""judge.py SIMDIR DUMPDIR K CONTROL "FAM FAM ..." OUT.json [--half rep] [--thr 0.90] [--guard 1.3] [--sens-perm P]
[--guard-rows 20000] -- the pass rule of w05-head2s / w05-headgen over seed replicates, from headarms.py's per-position files
(SIMDIR/<fam>-K<k>-s<seed>.kl.f32). Units (documents or 512-row blocks) as headarms; the REPORT half (odd units) by
default. Per family: mean KL over its seeds / mean KL over the control's seeds (ratio, unit-bootstrap 95% CI),
worst seed vs best control seed, guard ratio (positions whose bf16 argmax is outside the 20,000 most sensitive rows).
PASS = ratio <= thr, CI upper < 1, every seed below every control seed, guard ratio <= guard. Winner: the passing
family with the lowest mean KL (or none). Families with fewer than 3 seeds are reported, never passed."""
import glob, json, os, sys
import numpy as np
a = [x for x in sys.argv[1:]]
def opt(k, d):
    if k in a: i = a.index(k); v = a[i + 1]; del a[i:i + 2]; return v
    return d
half, thr, gx, sp = opt("--half", "rep"), float(opt("--thr", 0.90)), float(opt("--guard", 1.3)), opt("--sens-perm", None)
GR = int(opt("--guard-rows", 20000))
SIM, DUMP, K, CTL, FAMS, OUT = a[0], a[1], a[2].lstrip("K"), a[3], a[4].split(), a[5]
N = os.path.getsize(f"{DUMP}/336.x") // (2 * int(open(f"{DUMP}/336.meta").read().split()[0]))
unit = np.fromfile(f"{DUMP}/doc.i32", np.int32) if os.path.exists(f"{DUMP}/doc.i32") else np.arange(N) // 512
_, unit = np.unique(unit, return_inverse=True); nu = unit.max() + 1
us = {"sel": np.arange(0, nu, 2), "rep": np.arange(1, nu, 2), "all": np.arange(nu)}[half]
m = np.isin(unit, us); u = unit[m]; cnt = np.bincount(u, minlength=nu)[us]
guard = np.zeros(N, bool)
top = f"{SIM}/ref.top.i32"
if sp and os.path.exists(top):
    p = np.load(sp); rank = np.empty(len(p), np.int64); rank[p] = np.arange(len(p)); guard = rank[np.fromfile(top, np.int32)] >= GR
def seeds(f):
    fs = sorted(glob.glob(f"{SIM}/{f}-K{K}-s[0-9].kl.f32"))
    return [np.fromfile(x, np.float32).astype(np.float64) for x in fs]
C = seeds(CTL); assert C, f"no control {CTL} at K{K}"
cm = np.mean(C, 0); cs = np.bincount(u, cm[m], minlength=nu)[us]
boot = np.random.default_rng(0).integers(0, len(us), (2000, len(us)))
res = {"K": K, "half": half, "control": CTL, "control_seeds": [float(x[m].mean()) for x in C], "families": {}}
for f in FAMS:
    S = seeds(f)
    if not S: continue
    fm = np.mean(S, 0); fs = np.bincount(u, fm[m], minlength=nu)[us]
    br = fs[boot].sum(1) / cs[boot].sum(1)
    r = {"seeds": [float(x[m].mean()) for x in S], "ratio": float(fm[m].mean() / cm[m].mean()),
         "ci95": [float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))],
         "every_seed_below": bool(max(x[m].mean() for x in S) < min(x[m].mean() for x in C)),
         "guard_ratio": float(fm[m & guard].mean() / cm[m & guard].mean()) if (m & guard).any() else None}
    r["pass"] = bool(len(S) >= 3 and r["ratio"] <= thr and r["ci95"][1] < 1 and r["every_seed_below"]
                     and (r["guard_ratio"] is None or r["guard_ratio"] <= gx))
    res["families"][f] = r
    print(f"K{K} {half} {f:12s} ratio {r['ratio']:.3f} (CI {r['ci95'][0]:.3f}-{r['ci95'][1]:.3f}) seeds {len(S)} "
          f"every-seed-below {r['every_seed_below']} guard x{r['guard_ratio'] or 0:.2f} -> {'PASS' if r['pass'] else 'no'}")
ok = {f: r for f, r in res["families"].items() if r["pass"]}
res["winner"] = min(ok, key=lambda f: ok[f]["ratio"]) if ok else None
print(f"K{K} winner: {res['winner']}")
json.dump(res, open(OUT, "w"), indent=1)
